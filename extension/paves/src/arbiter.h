/*
 * arbiter.h — shared-memory GPU micro-batch arbiter.
 *
 * PostgreSQL is multi-process; a GPU wants large batches from one context.
 * Backends therefore never touch CUDA in worker mode: they serialize the
 * query (vector + flattened predicate + driver spans) into a fixed-size
 * shared-memory slot and wake the paves GPU worker (a background worker
 * owning the only CUDA context). The worker aggregates submitted slots into
 * micro-batches (up to paves.gpu_batch_max requests or
 * paves.gpu_batch_wait_us of gathering), runs one kernel per strategy
 * group, and answers each slot.
 *
 * Responses carry packed heap TIDs (block<<16|offset), not engine row ids:
 * the worker translates through ITS index file, so a concurrent rebuild can
 * only cost recall (like any stale index), never map a row id through the
 * wrong file version.
 *
 * Slot life cycle (pg_atomic state machine, no locks):
 *   FREE -> FILLING (backend CAS) -> SUBMITTED (backend, after fill)
 *        -> RUNNING (worker CAS)  -> DONE | ERROR (worker) -> FREE (backend)
 * A backend that errors out mid-wait waits for the worker to finish the
 * slot before freeing it (bounded; kernel batches are milliseconds).
 */
#ifndef PAVES_ARBITER_H
#define PAVES_ARBITER_H

#include "postgres.h"

#include "port/atomics.h"
#include "storage/latch.h"

#include "engine/engine_api.h"

/* Fixed slot capacities; requests beyond them stay on the CPU/direct path.
 * MAX_DIM matches pgvector's own vector-type limit so users never hit the
 * arbiter cap first (cost: ~8KB per slot). */
#define FV_ARB_MAX_DIM 2000
#define FV_ARB_MAX_NODES 48
#define FV_ARB_MAX_CHILDREN 96
#define FV_ARB_MAX_INVALS 128
#define FV_ARB_MAX_SPANS 128
#define FV_ARB_MAX_RESULTS 1024
/* Slot pool size bounds the micro-batch. The graph kernel's saturation
 * point on wiki-scale data is ~512 concurrent queries (one block each on a
 * ~100-SM part with 3 resident blocks/SM), measured with tools/gpu_micro;
 * ~30 KB/slot puts 512 slots at a comfortable ~16 MB of shared memory. */
#define FV_ARB_SLOTS 512

/* selectivity buckets for the cost-model feedback (log-ish scale) */
#define FV_SEL_BUCKETS 5
static inline int
fv_sel_bucket(int sel_permille)
{
    if (sel_permille >= 200) return 0;  /* >= 20%  */
    if (sel_permille >= 50) return 1;   /* 5..20%  */
    if (sel_permille >= 10) return 2;   /* 1..5%   */
    if (sel_permille >= 2) return 3;    /* 0.2..1% */
    return 4;                           /* < 0.2%  */
}

/* geometric bucket midpoints; anchor selectivities for the planner's
 * shape-based interpolation across buckets */
static inline double
fv_sel_bucket_mid(int b)
{
    static const double mid[FV_SEL_BUCKETS] =
        {0.40, 0.10, 0.022, 0.0045, 0.001};

    return mid[b];
}

/*
 * Per-index runtime feedback for the device-aware cost model. Service time
 * depends heavily on the dataset (N, dim, graph shape): a single global
 * table lets an 8M x 1024d index's latencies poison the routing of a
 * 1M x 128d one, so entries are keyed by (dbid, relid). All fields are
 * EWMAs in microseconds (0 = no data yet); racy read-modify-write updates
 * are acceptable for feedback. Strategy index: 0 = brute, 1 = graph.
 */
#define FV_FB_INDEXES 16
typedef struct FvFeedbackEntry {
    pg_atomic_uint32 relid_key; /* 0 = free; claimed by CAS to relid */
    Oid dbid;                   /* written by the claimer after the CAS */

    /* GPU worker publishes: amortized per-request kernel-batch service
     * time per selectivity bucket, plus whole-batch wall time and average
     * batch size (fixed point x16) — the planner prices queue wait in
     * units of batches, not requests (the kernel is batch-parallel) */
    pg_atomic_uint32 gpu_q_us[2][FV_SEL_BUCKETS];
    pg_atomic_uint32 gpu_batch_us[2];
    pg_atomic_uint32 gpu_batch_n16[2];

    /* backends publish: observed CPU engine time per query AS MEASURED
     * under load, paired with the cpu_active level it was measured at
     * (EWMA x16). The planner extrapolates: cost = obs x max(1,
     * active_now / active_ref) — no assumed cores/bandwidth model, and
     * the live counter gives instant negative feedback against flooding
     * one device (normalizing to an "unloaded" estimate instead proved
     * wrong: dividing by an estimated saturation factor cannot see
     * memory-bandwidth contention, and the mis-normalized samples made
     * the router oscillate) */
    pg_atomic_uint32 cpu_q_us[2][FV_SEL_BUCKETS];
    pg_atomic_uint32 cpu_ref_active16[2];
} FvFeedbackEntry;

typedef enum FvSlotState {
    FV_SLOT_FREE = 0,
    FV_SLOT_FILLING,
    FV_SLOT_SUBMITTED,
    FV_SLOT_RUNNING,
    FV_SLOT_DONE,
    FV_SLOT_ERROR,
} FvSlotState;

typedef struct FvArbSlot {
    pg_atomic_uint32 state;
    int32 pid;      /* owner backend, for diagnostics */
    Latch *waiter;  /* backend's proc latch (shared memory, cross-process) */

    /* request */
    Oid dbid;
    Oid relid;
    uint64 idx_ino; /* backend's index file inode; worker validates */
    int32 strategy; /* FV_STRAT_GPU_BRUTE / FV_STRAT_GPU_HNSW */
    int32 dim;
    uint32 want;        /* top-B (brute) or ef (graph) */
    uint64 max_visited; /* graph */
    int32 nnodes, nchildren, ninvals, nspans;
    int32 need_eval, dedup;
    int32 sel_permille; /* planner's selectivity estimate (feedback bucket) */
    float query[FV_ARB_MAX_DIM];
    FvGpuPredNode nodes[FV_ARB_MAX_NODES];
    int32 children[FV_ARB_MAX_CHILDREN];
    double invals[FV_ARB_MAX_INVALS];
    FvGpuSpan spans[FV_ARB_MAX_SPANS];

    /* response — FvHit.node carries the packed heap TID */
    uint32 nresults;
    uint8 more; /* nonzero: a wider re-request could return more results */
    FvHit results[FV_ARB_MAX_RESULTS];
} FvArbSlot;

typedef struct FvArbShared {
    pg_atomic_uint32 worker_ready; /* 1 while the worker loop is serving */
    Latch *worker_latch;

    /* runtime feedback for the device-aware cost model: per-index EWMA
     * tables (see FvFeedbackEntry) plus global load counters. A single
     * per-strategy average cannot express "the GPU is great at 3% and
     * terrible at 0.1%", hence the selectivity buckets inside. */
    FvFeedbackEntry fb[FV_FB_INDEXES];
    pg_atomic_uint32 gpu_inflight;
    pg_atomic_uint32 cpu_active;

    /* GPU memory status published by the worker (for paves_capacity();
     * 0 = unknown / no GPU picked yet) */
    pg_atomic_uint64 gpu_free_bytes;
    pg_atomic_uint64 gpu_total_bytes;
    pg_atomic_uint32 gpu_device_plus1; /* 0 = none, else device id + 1 */

    FvArbSlot slots[FV_ARB_SLOTS];
} FvArbShared;

extern FvArbShared *fv_arb; /* set by the shmem startup hook; else NULL */

/*
 * Feedback entry lookup / claim. A relid claimed for one database keeps the
 * entry (cross-db relid collisions just lose feedback for the loser, never
 * mix numbers: find checks dbid). Table full => NULL, callers fall back to
 * analytic priors.
 */
static inline FvFeedbackEntry *
fv_feedback_find(Oid dbid, Oid relid)
{
    int i;

    if (fv_arb == NULL)
        return NULL;
    for (i = 0; i < FV_FB_INDEXES; i++)
    {
        FvFeedbackEntry *e = &fv_arb->fb[i];

        if (pg_atomic_read_u32(&e->relid_key) == (uint32) relid)
            return e->dbid == dbid ? e : NULL;
    }
    return NULL;
}

static inline FvFeedbackEntry *
fv_feedback_claim(Oid dbid, Oid relid)
{
    int i;

    if (fv_arb == NULL)
        return NULL;
    for (i = 0; i < FV_FB_INDEXES; i++)
    {
        FvFeedbackEntry *e = &fv_arb->fb[i];
        uint32 key = pg_atomic_read_u32(&e->relid_key);

        if (key == (uint32) relid)
            return e->dbid == dbid ? e : NULL;
        if (key == 0)
        {
            uint32 expected = 0;

            if (pg_atomic_compare_exchange_u32(&e->relid_key, &expected,
                                               (uint32) relid))
            {
                e->dbid = dbid;
                return e;
            }
            /* lost the race: re-check this entry (it may now be ours) */
            i--;
        }
    }
    return NULL;
}

/* racy EWMA update: fine for feedback, atomicity per field is enough */
static inline void
fv_feedback_ewma(pg_atomic_uint32 *field, double sample, double alpha)
{
    uint32 old = pg_atomic_read_u32(field);
    uint32 next = old == 0 ? (uint32) sample
                           : (uint32) (old + alpha * (sample - old));

    pg_atomic_write_u32(field, Max(next, 1));
}

extern Size paves_arbiter_shmem_size(void);
extern void paves_arbiter_shmem_init(void);
extern void paves_arbiter_register(void); /* shmem request + bgworker */

/*
 * Backend API: submit one GPU request and wait for the answer.
 * Returns 1 on success (results/ncnt/more filled), 0 when the arbiter is
 * unavailable or a capacity is exceeded (caller uses the direct/CPU path),
 * -1 when the worker answered ERROR (caller uses the CPU path).
 */
extern int paves_arbiter_submit(Oid relid, uint64 idx_ino, int strategy,
                                  int dim, const float *query, FvIndex *idx,
                                  const FvPred *pred, uint32 want,
                                  uint64 max_visited, int sel_permille,
                                  FvHit *out, uint32 *ncnt, uint8 *more);

#endif /* PAVES_ARBITER_H */
