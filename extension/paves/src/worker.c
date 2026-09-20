/*
 * worker.c — the paves GPU worker: a background worker that owns the only
 * CUDA context in the cluster and serves shared-memory request slots in
 * micro-batches (see arbiter.h).
 *
 * The worker never connects to a database: index files are addressed by
 * (dbid, relid) under $PGDATA/paves and mmap'ed directly through the
 * engine. A slot whose index inode does not match the worker's view is
 * answered ERROR and the backend falls back to its CPU path.
 *
 * Batch grouping: collected slots are grouped by (dbid, relid, strategy);
 * within a group one kernel runs at the group's max want (top-B or ef) and
 * each slot receives its own prefix. Engine row ids are translated to
 * packed heap TIDs through the worker's index before answering.
 */
#include "paves.h"
#include "arbiter.h"

#include <execinfo.h>
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>

#include "miscadmin.h"
#include "pgstat.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"
#include "storage/latch.h"
#include "storage/proc.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/wait_event.h"

PGDLLEXPORT void paves_arbiter_worker_main(Datum arg);

/*
 * The worker runs kernels over device memory and mmap'ed index files; a
 * fault here takes the whole instance down through postmaster restart, and
 * the platform's core handler may swallow the dump. Print a raw backtrace
 * to the server log (write(2) only — async-signal-safe) before re-raising
 * so the frames are always recoverable via addr2line on paves.so.
 */
static void
worker_fatal_signal(int sig)
{
    void *frames[32];
    int n = backtrace(frames, 32);
    char head[128];
    int len = snprintf(head, sizeof(head),
                       "paves gpu worker: fatal signal %d, %d frames\n",
                       sig, n);

    if (write(STDERR_FILENO, head, (size_t) len) < 0)
        _exit(2);
    backtrace_symbols_fd(frames, n, STDERR_FILENO);
    signal(sig, SIG_DFL);
    raise(sig);
}

/* ---------------- worker-side index cache ---------------- */

typedef struct WIndexKey {
    Oid dbid;
    Oid relid;
} WIndexKey;

typedef struct WIndexEntry {
    WIndexKey key;
    FvIndex *idx; /* NULL: negative entry */
    ino_t ino;
    off_t size;
    time_t mtime;
} WIndexEntry;

static HTAB *windex_cache = NULL;

static FvIndex *
worker_get_index(Oid dbid, Oid relid, uint64 *ino_out)
{
    WIndexKey key;
    WIndexEntry *entry;
    bool found;
    char path[MAXPGPATH];
    struct stat st;

    if (windex_cache == NULL)
    {
        HASHCTL ctl;

        MemSet(&ctl, 0, sizeof(ctl));
        ctl.keysize = sizeof(WIndexKey);
        ctl.entrysize = sizeof(WIndexEntry);
        ctl.hcxt = TopMemoryContext;
        windex_cache = hash_create("paves worker index cache", 16, &ctl,
                                   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
    }

    snprintf(path, sizeof(path), "%s/paves/%u_%u.idx", DataDir, dbid, relid);
    if (stat(path, &st) != 0)
        return NULL;

    key.dbid = dbid;
    key.relid = relid;
    entry = hash_search(windex_cache, &key, HASH_ENTER, &found);
    if (!found)
        entry->idx = NULL;
    if (entry->idx == NULL || entry->ino != st.st_ino ||
        entry->size != st.st_size || entry->mtime != st.st_mtime)
    {
        char errbuf[256];
        FvIndex *idx = fv_index_open(path, errbuf, sizeof(errbuf));

        /* replaced mappings are leaked deliberately (bounded by rebuilds);
         * the GPU copy is freed with the old FvIndex on close, which we
         * skip for the same reason */
        if (idx == NULL)
        {
            elog(WARNING, "paves worker: cannot open %s: %s", path, errbuf);
            entry->idx = NULL;
            return NULL;
        }
        entry->idx = idx;
        entry->ino = st.st_ino;
        entry->size = st.st_size;
        entry->mtime = st.st_mtime;
    }
    *ino_out = (uint64) entry->ino;
    return entry->idx;
}

/* ---------------- batch execution ---------------- */

static void
answer_error(FvArbSlot *slot)
{
    slot->nresults = 0;
    slot->more = 0;
    pg_write_barrier();
    pg_atomic_write_u32(&slot->state, FV_SLOT_ERROR);
    if (slot->waiter)
        SetLatch(slot->waiter);
}

/*
 * Run one (dbid, relid, strategy) group. Slot predicate arrays are used in
 * place (they live in shared memory). Kernel row ids are translated to
 * packed TIDs before answering.
 */
static void
run_group(FvArbSlot **slots, int n)
{
    FvArbSlot *s0 = slots[0];
    uint64 ino = 0;
    FvIndex *idx = worker_get_index(s0->dbid, s0->relid, &ino);
    int dim;
    uint32 want_max = 0;
    uint64 mv_max = 0;
    int i,
        q;
    float *queries;
    FvGpuBatchPred *preds;
    uint32 *entries = NULL;
    FvHit *out;
    uint32 *cnts;
    uint8 *flags;
    int rc;

    if (idx == NULL || fv_index_gpu_ensure(idx, paves_gpu_device) != 0)
    {
        for (i = 0; i < n; i++)
            answer_error(slots[i]);
        return;
    }
    dim = fv_index_dim(idx);

    /* validate per slot: dimension and index-file identity */
    for (i = 0; i < n; i++)
    {
        if (slots[i]->dim != dim || slots[i]->idx_ino != ino)
        {
            answer_error(slots[i]);
            slots[i] = NULL;
        }
    }
    /* compact */
    for (i = 0, q = 0; i < n; i++)
        if (slots[i] != NULL)
            slots[q++] = slots[i];
    n = q;
    if (n == 0)
        return;
    /* the original slots[0] may have been answered ERROR above, after which
     * its backend can free it and another backend refill it with a foreign
     * request — re-anchor the group descriptor on a surviving slot (all
     * survivors are RUNNING and worker-owned until we answer them) */
    s0 = slots[0];

    for (i = 0; i < n; i++)
    {
        want_max = Max(want_max, slots[i]->want);
        mv_max = Max(mv_max, slots[i]->max_visited);
    }

    /* Schedule heavy queries first (LPT): one query = one thread block,
     * and blocks launch in index order, so descending expected work makes
     * waves finish together instead of a long straggler tail. Expected
     * work for a graph query ~ visited ~ want/selectivity (each expansion
     * touches ~degree neighbors and the degree is an index constant, so it
     * ranks identically); brute groups are near-uniform per query. */
    if (s0->strategy == FV_STRAT_GPU_HNSW && n > 2)
    {
        int j,
            k;

        for (j = 1; j < n; j++)
        {
            FvArbSlot *s = slots[j];
            double w = (double) s->want * 1000.0 /
                Max(s->sel_permille, 1);

            for (k = j - 1; k >= 0; k--)
            {
                double wk = (double) slots[k]->want * 1000.0 /
                    Max(slots[k]->sel_permille, 1);

                if (wk >= w)
                    break;
                slots[k + 1] = slots[k];
            }
            slots[k + 1] = s;
        }
    }

    queries = palloc(sizeof(float) * (Size) n * dim);
    preds = palloc(sizeof(FvGpuBatchPred) * n);
    out = palloc(sizeof(FvHit) * (Size) n * want_max);
    cnts = palloc(sizeof(uint32) * n);
    flags = palloc(sizeof(uint8) * n);

    for (i = 0; i < n; i++)
    {
        FvArbSlot *s = slots[i];

        memcpy(queries + (Size) i * dim, s->query, sizeof(float) * dim);
        preds[i].nodes = s->nodes;
        preds[i].nnodes = s->nnodes;
        preds[i].children = s->children;
        preds[i].nchildren = s->nchildren;
        preds[i].invals = s->invals;
        preds[i].ninvals = s->ninvals;
        preds[i].spans = s->spans;
        preds[i].nspans = s->nspans;
        preds[i].need_eval = s->need_eval;
        preds[i].dedup = s->dedup;
    }

    {
        instr_time e0, e1, e2;

        INSTR_TIME_SET_CURRENT(e0);
        if (s0->strategy == FV_STRAT_GPU_HNSW)
        {
            entries = palloc(sizeof(uint32) * n);
            for (i = 0; i < n; i++)
                entries[i] = fv_index_graph_entry(idx,
                                                  queries + (Size) i * dim);
            INSTR_TIME_SET_CURRENT(e1);
            rc = fv_gpu_exec_graph_batch(idx, n, queries, entries, preds,
                                         want_max, mv_max ? mv_max : 0,
                                         out, cnts, flags);
        }
        else
        {
            INSTR_TIME_SET_CURRENT(e1);
            rc = fv_gpu_exec_brute_batch(idx, n, queries, preds, want_max,
                                         out, cnts, flags);
        }
        INSTR_TIME_SET_CURRENT(e2);
        INSTR_TIME_SUBTRACT(e2, e1);
        INSTR_TIME_SUBTRACT(e1, e0);

        /* Publish per-request service time for the planner's device-aware
         * cost model, keyed by this group's index and bucketed by
         * selectivity. Batch wall time is set by its slowest block, so
         * attribute it by each request's analytical weight (graph:
         * w ~ want/selectivity; brute tiled: constant): c = T / max(w),
         * sample for a bucket = c * mean_w(bucket) — a fast bucket riding
         * in a slow batch gets a small sample instead of inheriting the
         * stragglers' cost. Updates weighted by bucket count
         * (alpha = n_b/(n_b+64)): tiny batches must not poison the
         * estimate (pricing at the large-batch marginal rate is what lets
         * traffic grow batches in the first place). Whole-batch wall time
         * and batch size go in as well: the planner prices queue wait in
         * batches (the kernel serves requests batch-parallel, so "queue
         * ahead of me" drains per batch time, not per request time). */
        if (fv_arb != NULL)
        {
            FvFeedbackEntry *fb = fv_feedback_claim(s0->dbid, s0->relid);
            int si = (s0->strategy == FV_STRAT_GPU_HNSW) ? 1 : 0;
            double T = (double) (INSTR_TIME_GET_MICROSEC(e1) +
                                 INSTR_TIME_GET_MICROSEC(e2));
            double w[FV_ARB_SLOTS];
            double wmax = 0;
            double wsum[FV_SEL_BUCKETS] = {0};
            int nb[FV_SEL_BUCKETS] = {0};
            int b;

            for (i = 0; fb != NULL && i < n; i++)
            {
                if (si == 1)
                    w[i] = (double) slots[i]->want * 1000.0 /
                        Max(slots[i]->sel_permille, 1);
                else
                    w[i] = 1.0;
                wmax = Max(wmax, w[i]);
                b = fv_sel_bucket(slots[i]->sel_permille);
                wsum[b] += w[i];
                nb[b]++;
            }
            if (fb != NULL)
            {
                for (b = 0; b < FV_SEL_BUCKETS; b++)
                {
                    if (nb[b] == 0)
                        continue;
                    fv_feedback_ewma(&fb->gpu_q_us[si][b],
                                     (T / Max(wmax, 1e-9)) * (wsum[b] / nb[b]),
                                     (double) nb[b] / (nb[b] + 64.0));
                }
                fv_feedback_ewma(&fb->gpu_batch_us[si], T, 0.125);
                fv_feedback_ewma(&fb->gpu_batch_n16[si], n * 16.0, 0.125);
            }
        }
    }

    for (i = 0; i < n; i++)
    {
        FvArbSlot *s = slots[i];

        if (rc < 0)
        {
            answer_error(s);
            continue;
        }
        {
            uint32 cnt = cnts[i];
            uint32 give = Min(cnt, s->want);
            const FvHit *src = out + (Size) i * want_max;
            uint32 j;

            uint32 kept = 0;

            for (j = 0; j < give; j++)
            {
                uint64 tid = fv_index_tid_checked(idx, src[j].node);

                if (tid == UINT64_MAX)
                {
                    /* A kernel returned a row id outside the index: drop it
                     * rather than index the mmap out of bounds (that would
                     * take down the whole instance). Loud, rate-limited. */
                    static int bad_warned = 0;

                    if (bad_warned < 10)
                    {
                        bad_warned++;
                        elog(WARNING,
                             "paves gpu worker: dropping out-of-range row "
                             "id %llu (strategy %d, batch %d, slot %d, "
                             "result %u/%u, count %llu)",
                             (unsigned long long) src[j].node,
                             s0->strategy, n, i, j, give,
                             (unsigned long long) fv_index_count(idx));
                    }
                    continue;
                }
                s->results[kept].node = tid;
                s->results[kept].dist = src[j].dist;
                kept++;
            }
            give = kept;
            s->nresults = give;
            if (s0->strategy == FV_STRAT_GPU_HNSW)
                /* graph: "more" unless traversal is exhausted; extra
                 * results beyond `want` also mean a wider ask pays off */
                s->more = (flags[i] == 0 || cnt > give) ? 1 : 0;
            else
                /* brute: truncated relative to this slot's want */
                s->more = (flags[i] != 0 || cnt > give) ? 1 : 0;
            pg_write_barrier();
            pg_atomic_write_u32(&s->state, FV_SLOT_DONE);
            if (s->waiter)
                SetLatch(s->waiter);
        }
    }

    pfree(queries);
    pfree(preds);
    if (entries)
        pfree(entries);
    pfree(out);
    pfree(cnts);
    pfree(flags);
}

/* group collected slots by (dbid, relid, strategy) and run each group */
static void
run_batch(FvArbSlot **slots, int n)
{
    int i;

    while (n > 0)
    {
        FvArbSlot *group[FV_ARB_SLOTS];
        int gn = 0;
        int rest = 0;
        FvArbSlot *s0 = slots[0];

        for (i = 0; i < n; i++)
        {
            if (slots[i]->dbid == s0->dbid && slots[i]->relid == s0->relid &&
                slots[i]->strategy == s0->strategy)
                group[gn++] = slots[i];
            else
                slots[rest++] = slots[i];
        }
        run_group(group, gn);
        n = rest;
    }
}

/* ---------------- main loop ---------------- */

static volatile sig_atomic_t got_sigterm = false;

static void
worker_sigterm(SIGNAL_ARGS)
{
    got_sigterm = true;
    SetLatch(MyLatch);
}

static void
worker_on_exit(int code, Datum arg)
{
    if (fv_arb != NULL)
    {
        pg_atomic_write_u32(&fv_arb->worker_ready, 0);
        fv_arb->worker_latch = NULL;
    }
}

static int
collect_submitted(FvArbSlot **batch, int have, int cap)
{
    int i;

    for (i = 0; i < FV_ARB_SLOTS && have < cap; i++)
    {
        uint32 expected = FV_SLOT_SUBMITTED;

        if (pg_atomic_compare_exchange_u32(&fv_arb->slots[i].state, &expected,
                                           FV_SLOT_RUNNING))
            batch[have++] = &fv_arb->slots[i];
    }
    return have;
}

void
paves_arbiter_worker_main(Datum arg)
{
    FvArbSlot *batch[FV_ARB_SLOTS];

    pqsignal(SIGTERM, worker_sigterm);
    pqsignal(SIGSEGV, worker_fatal_signal);
    pqsignal(SIGBUS, worker_fatal_signal);
    BackgroundWorkerUnblockSignals();

    if (fv_arb == NULL)
    {
        elog(WARNING, "paves worker: shared memory not initialized");
        return;
    }
    on_shmem_exit(worker_on_exit, (Datum) 0);
    fv_arb->worker_latch = MyLatch;
    pg_write_barrier();
    pg_atomic_write_u32(&fv_arb->worker_ready, 1);
    elog(LOG, "paves gpu worker started");

    /* publish GPU memory status for paves_capacity() (the worker owns
     * the only CUDA context; backends must never probe) */
    {
        uint64 gfree = 0, gtotal = 0;
        int dev = fv_gpu_mem_probe(&gfree, &gtotal);

        if (dev >= 0)
        {
            pg_atomic_write_u64(&fv_arb->gpu_free_bytes, gfree);
            pg_atomic_write_u64(&fv_arb->gpu_total_bytes, gtotal);
            pg_atomic_write_u32(&fv_arb->gpu_device_plus1, (uint32) dev + 1);
            elog(LOG, "paves gpu worker: device %d, %.1f/%.1f GB free",
                 dev, gfree / 1e9, gtotal / 1e9);
        }
    }

    {
        instr_time stat_start, t0, t1;
        uint64 stat_batches = 0, stat_reqs = 0, stat_exec_us = 0;
        uint64 stat_wait_us = 0;

        INSTR_TIME_SET_CURRENT(stat_start);

        while (!got_sigterm)
        {
            int n = collect_submitted(batch, 0, paves_gpu_batch_max);

            if (n == 0)
            {
                (void) WaitLatch(MyLatch,
                                 WL_LATCH_SET | WL_TIMEOUT | WL_EXIT_ON_PM_DEATH,
                                 100, PG_WAIT_EXTENSION);
                ResetLatch(MyLatch);
                continue;
            }

            /*
             * Micro-batch gathering window. Waiting the full window when the
             * offered load cannot fill the batch just adds dead time between
             * kernels, so the window also closes after gpu_batch_idle_us
             * microseconds without a new arrival: under saturation batches
             * still fill (arrivals keep resetting the idle clock), under
             * moderate load the batch departs as soon as the arrival stream
             * pauses. idle_us = 0 restores the fixed window.
             */
            if (paves_gpu_batch_wait_us > 0 && n < paves_gpu_batch_max)
            {
                int waited = 0;
                int idle = 0;

                while (waited < paves_gpu_batch_wait_us &&
                       n < paves_gpu_batch_max)
                {
                    int got;

                    pg_usleep(20);
                    waited += 20;
                    got = collect_submitted(batch, n, paves_gpu_batch_max);
                    if (paves_gpu_batch_idle_us > 0)
                    {
                        idle = (got > n) ? 0 : idle + 20;
                        if (idle >= paves_gpu_batch_idle_us)
                        {
                            n = got;
                            break;
                        }
                    }
                    n = got;
                }
                stat_wait_us += (uint64) waited;
            }

            INSTR_TIME_SET_CURRENT(t0);
            run_batch(batch, n);
            INSTR_TIME_SET_CURRENT(t1);
            INSTR_TIME_SUBTRACT(t1, t0);
            stat_batches++;
            stat_reqs += n;
            stat_exec_us += (uint64) INSTR_TIME_GET_MICROSEC(t1);

            INSTR_TIME_SET_CURRENT(t1);
            INSTR_TIME_SUBTRACT(t1, stat_start);
            if (INSTR_TIME_GET_MICROSEC(t1) > 2000000)
            {
                elog(LOG,
                     "paves gpu worker: %lu batches, %lu reqs "
                     "(%.1f/batch), exec %.1f us/req, wait %.1f us/batch",
                     (unsigned long) stat_batches, (unsigned long) stat_reqs,
                     (double) stat_reqs / stat_batches,
                     (double) stat_exec_us / Max(stat_reqs, 1),
                     (double) stat_wait_us / stat_batches);
                stat_batches = stat_reqs = stat_exec_us = 0;
                stat_wait_us = 0;
                INSTR_TIME_SET_CURRENT(stat_start);
                {
                    uint64 gfree = 0, gtotal = 0;

                    if (fv_gpu_mem_probe(&gfree, &gtotal) >= 0)
                    {
                        pg_atomic_write_u64(&fv_arb->gpu_free_bytes, gfree);
                        pg_atomic_write_u64(&fv_arb->gpu_total_bytes, gtotal);
                    }
                }
            }
        }
    }

    elog(LOG, "paves gpu worker shutting down");
}
