/*
 * arbiter.c — shared-memory setup and the backend side of the GPU
 * micro-batch arbiter (see arbiter.h for the protocol).
 */
#include "paves.h"
#include "arbiter.h"

#include "miscadmin.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/proc.h"
#include "storage/shmem.h"
#include "utils/wait_event.h"

FvArbShared *fv_arb = NULL;

static shmem_startup_hook_type prev_shmem_startup_hook = NULL;

Size
paves_arbiter_shmem_size(void)
{
    return MAXALIGN(sizeof(FvArbShared));
}

static void
paves_shmem_startup(void)
{
    bool found;

    if (prev_shmem_startup_hook)
        prev_shmem_startup_hook();

    fv_arb = ShmemInitStruct("paves arbiter",
                             paves_arbiter_shmem_size(), &found);
    if (!found)
    {
        int i;

        memset(fv_arb, 0, paves_arbiter_shmem_size());
        pg_atomic_init_u32(&fv_arb->worker_ready, 0);
        {
            int e, s, b;

            for (e = 0; e < FV_FB_INDEXES; e++)
            {
                FvFeedbackEntry *fb = &fv_arb->fb[e];

                pg_atomic_init_u32(&fb->relid_key, 0);
                fb->dbid = InvalidOid;
                for (s = 0; s < 2; s++)
                {
                    pg_atomic_init_u32(&fb->gpu_batch_us[s], 0);
                    pg_atomic_init_u32(&fb->gpu_batch_n16[s], 0);
                    pg_atomic_init_u32(&fb->cpu_ref_active16[s], 0);
                    for (b = 0; b < FV_SEL_BUCKETS; b++)
                    {
                        pg_atomic_init_u32(&fb->gpu_q_us[s][b], 0);
                        pg_atomic_init_u32(&fb->cpu_q_us[s][b], 0);
                    }
                }
            }
        }
        pg_atomic_init_u32(&fv_arb->gpu_inflight, 0);
        pg_atomic_init_u32(&fv_arb->cpu_active, 0);
        pg_atomic_init_u64(&fv_arb->gpu_free_bytes, 0);
        pg_atomic_init_u64(&fv_arb->gpu_total_bytes, 0);
        pg_atomic_init_u32(&fv_arb->gpu_device_plus1, 0);
        for (i = 0; i < FV_ARB_SLOTS; i++)
            pg_atomic_init_u32(&fv_arb->slots[i].state, FV_SLOT_FREE);
    }
}

void
paves_arbiter_shmem_init(void)
{
    prev_shmem_startup_hook = shmem_startup_hook;
    shmem_startup_hook = paves_shmem_startup;
}

void
paves_arbiter_register(void)
{
    BackgroundWorker worker;

    RequestAddinShmemSpace(paves_arbiter_shmem_size());
    paves_arbiter_shmem_init();

    memset(&worker, 0, sizeof(worker));
    worker.bgw_flags = BGWORKER_SHMEM_ACCESS;
    worker.bgw_start_time = BgWorkerStart_PostmasterStart;
    worker.bgw_restart_time = 5;
    snprintf(worker.bgw_name, BGW_MAXLEN, "paves gpu worker");
    snprintf(worker.bgw_type, BGW_MAXLEN, "paves gpu worker");
    snprintf(worker.bgw_library_name, BGW_MAXLEN, "paves");
    snprintf(worker.bgw_function_name, BGW_MAXLEN,
             "paves_arbiter_worker_main");
    RegisterBackgroundWorker(&worker);
}

/* ---------------- backend submit ---------------- */

/* Wait until the worker has released the slot (DONE/ERROR), then free it.
 * Used on both the normal path and the error-cleanup path; bounded because
 * kernel batches are milliseconds. If the worker died mid-request the
 * postmaster restarts it; the state check on worker_ready lets us reclaim
 * a slot the dead worker never picked up. */
static void
arbiter_slot_release(FvArbSlot *slot)
{
    int spins = 0;

    for (;;)
    {
        uint32 st = pg_atomic_read_u32(&slot->state);

        if (st == FV_SLOT_DONE || st == FV_SLOT_ERROR)
            break;
        if (st == FV_SLOT_SUBMITTED)
        {
            uint32 expected = FV_SLOT_SUBMITTED;

            if (pg_atomic_compare_exchange_u32(&slot->state, &expected,
                                               FV_SLOT_FREE))
                return; /* worker never saw it */
        }
        if (st == FV_SLOT_RUNNING &&
            pg_atomic_read_u32(&fv_arb->worker_ready) == 0)
            break; /* worker died holding it; reclaim */
        pg_usleep(1000);
        if (++spins > 10000) /* 10s: give up waiting, reclaim anyway */
            break;
    }
    pg_atomic_write_u32(&slot->state, FV_SLOT_FREE);
}

int
paves_arbiter_submit(Oid relid, uint64 idx_ino, int strategy, int dim,
                       const float *query, FvIndex *idx, const FvPred *pred,
                       uint32 want, uint64 max_visited, int sel_permille,
                       FvHit *out, uint32 *ncnt, uint8 *more)
{
    FvArbSlot *slot = NULL;
    FvGpuBatchPred bp;
    uint64 total_rows;
    FvGpuPredNode nodes[FV_ARB_MAX_NODES];
    int32 children[FV_ARB_MAX_CHILDREN];
    double invals[FV_ARB_MAX_INVALS];
    FvGpuSpan spans[FV_ARB_MAX_SPANS];
    int i;
    int ret = 0;

    if (fv_arb == NULL || pg_atomic_read_u32(&fv_arb->worker_ready) != 1)
        return 0;
    if (dim > FV_ARB_MAX_DIM || want < 1 || want > FV_ARB_MAX_RESULTS)
        return 0;

    /* serialize the predicate first: capacity misses stay on the CPU */
    if (fv_pred_serialize(idx, pred, nodes, FV_ARB_MAX_NODES, children,
                          FV_ARB_MAX_CHILDREN, invals, FV_ARB_MAX_INVALS,
                          spans, FV_ARB_MAX_SPANS, &bp, &total_rows) != 0)
        return 0;

    for (i = 0; i < FV_ARB_SLOTS; i++)
    {
        uint32 expected = FV_SLOT_FREE;

        if (pg_atomic_compare_exchange_u32(&fv_arb->slots[i].state, &expected,
                                           FV_SLOT_FILLING))
        {
            slot = &fv_arb->slots[i];
            break;
        }
    }
    if (slot == NULL)
        return 0; /* queue full: caller runs the query itself */

    slot->pid = MyProcPid;
    slot->waiter = MyLatch;
    slot->dbid = MyDatabaseId;
    slot->relid = relid;
    slot->idx_ino = idx_ino;
    slot->strategy = strategy;
    slot->dim = dim;
    slot->want = want;
    slot->max_visited = max_visited;
    slot->nnodes = bp.nnodes;
    slot->nchildren = bp.nchildren;
    slot->ninvals = bp.ninvals;
    slot->nspans = bp.nspans;
    slot->need_eval = bp.need_eval;
    slot->dedup = bp.dedup;
    slot->sel_permille = sel_permille;
    memcpy(slot->query, query, sizeof(float) * dim);
    if (bp.nnodes > 0)
        memcpy(slot->nodes, nodes, sizeof(FvGpuPredNode) * bp.nnodes);
    if (bp.nchildren > 0)
        memcpy(slot->children, children, sizeof(int32) * bp.nchildren);
    if (bp.ninvals > 0)
        memcpy(slot->invals, invals, sizeof(double) * bp.ninvals);
    if (bp.nspans > 0)
        memcpy(slot->spans, spans, sizeof(FvGpuSpan) * bp.nspans);

    pg_write_barrier();
    pg_atomic_write_u32(&slot->state, FV_SLOT_SUBMITTED);
    pg_atomic_fetch_add_u32(&fv_arb->gpu_inflight, 1);
    if (fv_arb->worker_latch)
        SetLatch(fv_arb->worker_latch);

    /* wait for the answer; on error exits release the slot first */
    PG_TRY();
    {
        for (;;)
        {
            uint32 st = pg_atomic_read_u32(&slot->state);

            if (st == FV_SLOT_DONE || st == FV_SLOT_ERROR)
                break;
            if (pg_atomic_read_u32(&fv_arb->worker_ready) != 1)
            {
                uint32 expected = FV_SLOT_SUBMITTED;

                /* worker gone: reclaim if it never picked the slot up */
                if (st == FV_SLOT_SUBMITTED &&
                    pg_atomic_compare_exchange_u32(&slot->state, &expected,
                                                   FV_SLOT_FREE))
                {
                    slot = NULL;
                    break;
                }
                if (st == FV_SLOT_RUNNING)
                {
                    pg_atomic_write_u32(&slot->state, FV_SLOT_FREE);
                    slot = NULL;
                    break;
                }
            }
            (void) WaitLatch(MyLatch,
                             WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                             10, PG_WAIT_EXTENSION);
            ResetLatch(MyLatch);
            CHECK_FOR_INTERRUPTS();
        }
    }
    PG_CATCH();
    {
        pg_atomic_fetch_sub_u32(&fv_arb->gpu_inflight, 1);
        if (slot != NULL)
            arbiter_slot_release(slot);
        PG_RE_THROW();
    }
    PG_END_TRY();

    pg_atomic_fetch_sub_u32(&fv_arb->gpu_inflight, 1);
    if (slot == NULL)
        return 0; /* worker died before answering: caller falls back */

    pg_read_barrier();
    if (pg_atomic_read_u32(&slot->state) == FV_SLOT_DONE)
    {
        *ncnt = slot->nresults;
        *more = slot->more;
        memcpy(out, slot->results, sizeof(FvHit) * slot->nresults);
        ret = 1;
    }
    else
        ret = -1; /* FV_SLOT_ERROR */
    pg_atomic_write_u32(&slot->state, FV_SLOT_FREE);
    return ret;
}
