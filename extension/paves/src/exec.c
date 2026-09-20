/*
 * exec.c — paves CustomScan executor.
 *
 * On the first tuple request: evaluates the query vector, compiles pushdown
 * predicates into an engine bitmap, opens the strategy cursor (brute/HNSW).
 * Then streams candidates in distance order, fetching each heap tuple by TID
 * under the query snapshot (visibility recheck); ExecScan re-evaluates all
 * original quals on every emitted tuple, so pushdown can never produce wrong
 * results, only extra candidates that get filtered.
 *
 * Engine objects live in malloc'ed memory; a MemoryContextCallback on the
 * per-query context guarantees they are released on any exit path, including
 * elog(ERROR) longjmps.
 */
#include "paves.h"
#include "arbiter.h"

#include <sys/stat.h>

#include "access/tableam.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/value.h"
#include "portability/instr_time.h"
#include "storage/itemptr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#define FETCH_BATCH 16

typedef struct FavorScanState {
    CustomScanState css;

    int strategy;
    int k;
    int sel_permille; /* planner's selectivity estimate, in 1/1000 */
    ExprState *qvec_state;
    List *pushdown_clauses;

    FvIndex *idx;
    bool started;
    FvPred *pred;
    uint64 nmatches; /* estimate: 0 authoritative, UINT64_MAX unknown */
    FvCursor *cursor;

    FvHit buf[FETCH_BATCH];
    int buf_n;
    int buf_pos;
    uint64 emitted; /* tuples returned so far */

    /* arbiter (GPU worker) mode: results hold packed TIDs directly */
    bool use_arb;
    FvHit *arb_results;
    uint32 arb_n;
    uint32 arb_pos;
    uint8 arb_more;
    uint32 arb_want;
    uint64 arb_ino;
    float *arb_query;
    uint64 *seen_tids; /* candidates already consumed (for re-requests) */
    int n_seen;
    int seen_cap;
    bool counted_cpu; /* contributes to fv_arb->cpu_active */

    /* CPU-side cost-model feedback: accumulated engine time (search begin +
     * cursor pulls), published on release together with the cpu_active
     * level this query started under — loaded observation + its load,
     * extrapolated by the planner via the live counter */
    uint64 engine_us;
    uint32 active_at_start;
    Oid feedback_relid;

    /* stats for EXPLAIN ANALYZE */
    uint64 fetched;
    uint64 invisible;

    MemoryContextCallback release_cb;
} FavorScanState;

static bool
tid_seen(FavorScanState *fss, uint64 packed)
{
    int i;

    for (i = 0; i < fss->n_seen; i++)
        if (fss->seen_tids[i] == packed)
            return true;
    return false;
}

static void
tid_mark_seen(FavorScanState *fss, uint64 packed)
{
    if (fss->seen_tids == NULL)
    {
        fss->seen_cap = 256;
        fss->seen_tids = palloc(sizeof(uint64) * fss->seen_cap);
    }
    else if (fss->n_seen >= fss->seen_cap)
    {
        fss->seen_cap *= 2;
        fss->seen_tids = repalloc(fss->seen_tids,
                                  sizeof(uint64) * fss->seen_cap);
    }
    fss->seen_tids[fss->n_seen++] = packed;
}

static void
favor_release_engine(void *arg)
{
    FavorScanState *fss = (FavorScanState *) arg;

    if (fss->cursor)
    {
        fv_cursor_end(fss->cursor);
        fss->cursor = NULL;
    }
    if (fss->pred)
    {
        fv_pred_free(fss->pred);
        fss->pred = NULL;
    }
    if (fss->counted_cpu)
    {
        if (fv_arb)
        {
            pg_atomic_fetch_sub_u32(&fv_arb->cpu_active, 1);
            /* publish the observed CPU engine time for the planner's cost
             * model (LIMIT semantics included: this is the work an actual
             * query of this shape costs, not a full cursor drain). Only
             * queries PLANNED as CPU strategies publish: GPU requests that
             * degraded to a CPU cursor run with a widened ef / re-request
             * dedup and are up to 10x dearer than a planned CPU search —
             * they once poisoned these EWMAs badly enough to starve the
             * CPU paths of all traffic. */
            if (fss->engine_us > 0 &&
                (fss->strategy == FV_STRAT_BRUTE ||
                 fss->strategy == FV_STRAT_HNSW))
            {
                FvFeedbackEntry *fb = fv_feedback_claim(MyDatabaseId,
                                                        fss->feedback_relid);

                if (fb != NULL)
                {
                    int si = (fss->strategy == FV_STRAT_HNSW) ? 1 : 0;
                    int b = fv_sel_bucket(fss->sel_permille);
                    uint32 old = pg_atomic_read_u32(&fb->cpu_q_us[si][b]);

                    /* mildly asymmetric EWMA (fast down, slow up): one
                     * startup wave of first-touch page faults must not
                     * park the router off the CPU; genuine load shifts
                     * still get in through sample volume */
                    fv_feedback_ewma(&fb->cpu_q_us[si][b],
                                     (double) fss->engine_us,
                                     (old == 0 ||
                                      (double) fss->engine_us < old)
                                     ? 0.25 : 1.0 / 32.0);
                    fv_feedback_ewma(&fb->cpu_ref_active16[si],
                                     fss->active_at_start * 16.0,
                                     1.0 / 16.0);
                }
            }
        }
        fss->counted_cpu = false;
    }
    fss->engine_us = 0;
}

/* a CPU-side cursor is about to run: feed the planner's load model */
static void
favor_count_cpu(FavorScanState *fss)
{
    if (!fss->counted_cpu && fv_arb)
    {
        uint32 prev = pg_atomic_fetch_add_u32(&fv_arb->cpu_active, 1);

        fss->active_at_start = prev + 1; /* counting ourselves */
        fss->counted_cpu = true;
    }
}

/* accumulate wall time spent inside the engine (for feedback samples) */
#define FV_ENGINE_TIMED(fss, call) \
    do { \
        instr_time _t0, _t1; \
        INSTR_TIME_SET_CURRENT(_t0); \
        call; \
        INSTR_TIME_SET_CURRENT(_t1); \
        INSTR_TIME_SUBTRACT(_t1, _t0); \
        (fss)->engine_us += (uint64) INSTR_TIME_GET_MICROSEC(_t1); \
    } while (0)

/* ---------------- executor callbacks ---------------- */

static Node *
favor_create_scan_state(CustomScan *cscan);

CustomScanMethods paves_scan_methods = {
    .CustomName = "paves",
    .CreateCustomScanState = favor_create_scan_state,
};

static void
favor_begin(CustomScanState *node, EState *estate, int eflags)
{
    FavorScanState *fss = (FavorScanState *) node;
    CustomScan *cscan = (CustomScan *) node->ss.ps.plan;

    fss->strategy = intVal(linitial(cscan->custom_private));
    fss->k = intVal(lsecond(cscan->custom_private));
    fss->sel_permille = intVal(lthird(cscan->custom_private));
    fss->pushdown_clauses = (List *) lsecond(cscan->custom_exprs);
    fss->started = false;
    fss->pred = NULL;
    fss->cursor = NULL;
    fss->buf_n = fss->buf_pos = 0;
    fss->fetched = fss->invisible = fss->emitted = 0;
    fss->use_arb = false;
    fss->arb_results = NULL;
    fss->arb_n = fss->arb_pos = 0;
    fss->arb_more = 0;
    fss->arb_query = NULL;
    fss->seen_tids = NULL;
    fss->n_seen = fss->seen_cap = 0;
    fss->counted_cpu = false;
    fss->engine_us = 0;
    fss->active_at_start = 1;
    fss->feedback_relid = InvalidOid;

    if (eflags & EXEC_FLAG_EXPLAIN_ONLY)
        return;

    /*
     * PG14's ExecInitCustomScan hardcodes a fixed virtual scan slot and has
     * already compiled qual/projection against it — for fixed virtual slots
     * the expression compiler elides the deforming FETCHSOME step, which is
     * wrong for the buffer-heap tuples we return from TID fetches (it would
     * read stale tts_values). Replace the scan slot with a proper table slot
     * and recompile qual and projection against it.
     */
    node->ss.ss_ScanTupleSlot =
        table_slot_create(node->ss.ss_currentRelation,
                          &estate->es_tupleTable);
    node->ss.ps.scanops = table_slot_callbacks(node->ss.ss_currentRelation);
    node->ss.ps.scanopsfixed = true;
    node->ss.ps.scanopsset = true;
    node->ss.ps.qual = ExecInitQual(cscan->scan.plan.qual, &node->ss.ps);
    ExecAssignScanProjectionInfoWithVarno(&node->ss, cscan->scan.scanrelid);

    fss->qvec_state = ExecInitExpr((Expr *) linitial(cscan->custom_exprs),
                                   &node->ss.ps);

    /* captured now: the release callback can fire after relation close */
    fss->feedback_relid = RelationGetRelid(node->ss.ss_currentRelation);

    fss->idx = paves_get_index(RelationGetRelid(node->ss.ss_currentRelation));
    if (fss->idx == NULL)
        ereport(ERROR,
                (errmsg("paves: index for \"%s\" disappeared; run paves_build() "
                        "or re-plan with paves.enable=off",
                        RelationGetRelationName(node->ss.ss_currentRelation))));

    /* engine resources are freed even on error exits */
    fss->release_cb.func = favor_release_engine;
    fss->release_cb.arg = fss;
    MemoryContextRegisterResetCallback(estate->es_query_cxt, &fss->release_cb);
}

/*
 * Submit (or re-submit, wider) this scan's query to the GPU arbiter.
 * first=true sizes the initial request; later calls quadruple (brute) or
 * double (graph) arb_want. Returns false when the arbiter cannot serve the
 * request (caller switches to a direct/CPU cursor).
 */
static bool
favor_arbiter_request(FavorScanState *fss, const float *q, int ef, bool first)
{
    uint32 want;
    uint64 mv = 0;
    int dim = fv_index_dim(fss->idx);
    int rc;

    if (fss->strategy == FV_STRAT_GPU_BRUTE)
        want = first ? (uint32) Max(4 * fss->k, 64) : fss->arb_want * 4;
    else
    {
        want = first ? (uint32) Max(ef, fss->k) : fss->arb_want * 2;
        mv = Max((uint64) 200000, (uint64) want * 400);
    }
    if (dim > FV_ARB_MAX_DIM)
    {
        static bool dim_warned = false;

        if (!dim_warned)
        {
            dim_warned = true;
            ereport(LOG,
                    (errmsg("paves: vector dimension %d exceeds the GPU "
                            "arbiter slot capacity %d; GPU strategies run "
                            "on the CPU (reported once per backend)",
                            dim, FV_ARB_MAX_DIM)));
        }
        return false;
    }
    if (want > FV_ARB_MAX_RESULTS)
        return false;
    if (first &&
        (fv_arb == NULL || pg_atomic_read_u32(&fv_arb->worker_ready) != 1))
    {
        static bool worker_warned = false;

        if (!worker_warned)
        {
            worker_warned = true;
            ereport(LOG,
                    (errmsg("paves: GPU worker not running; GPU "
                            "strategies run on the CPU (reported once per "
                            "backend)")));
        }
        return false;
    }

    if (first)
    {
        char *path = paves_index_path(
            RelationGetRelid(fss->css.ss.ss_currentRelation));
        struct stat st;

        if (stat(path, &st) != 0)
        {
            pfree(path);
            return false;
        }
        pfree(path);
        fss->arb_ino = (uint64) st.st_ino;
        if (fss->arb_results == NULL)
            fss->arb_results = palloc(sizeof(FvHit) * FV_ARB_MAX_RESULTS);
    }

    rc = paves_arbiter_submit(
        RelationGetRelid(fss->css.ss.ss_currentRelation), fss->arb_ino,
        fss->strategy, dim, q, fss->idx, fss->pred, want, mv,
        fss->sel_permille, fss->arb_results, &fss->arb_n, &fss->arb_more);
    if (rc != 1)
        return false;
    fss->arb_want = want;
    fss->arb_pos = 0;
    return true;
}

static void
favor_start_search(FavorScanState *fss)
{
    ExprContext *econtext = fss->css.ss.ps.ps_ExprContext;
    Datum qdatum;
    bool isnull;
    PgvVector *qvec;
    int dim = fv_index_dim(fss->idx);
    float *q;
    ListCell *lc;
    int npush = list_length(fss->pushdown_clauses);
    FvPredNode *root_pred = NULL;

    qdatum = ExecEvalExprSwitchContext(fss->qvec_state, econtext, &isnull);
    if (isnull)
        ereport(ERROR, (errmsg("paves: query vector must not be NULL")));
    qvec = (PgvVector *) PG_DETOAST_DATUM(qdatum);
    if (qvec->dim != dim)
        ereport(ERROR, (errmsg("paves: query vector dimension %d does not "
                               "match index dimension %d", qvec->dim, dim)));
    q = (float *) palloc(sizeof(float) * dim);
    memcpy(q, qvec->x, sizeof(float) * dim);

    if (npush > 0)
    {
        FvPredNode **compiled = palloc(sizeof(FvPredNode *) * npush);
        int n = 0;

        foreach(lc, fss->pushdown_clauses)
        {
            FvPredNode *p = paves_compile_clause((Node *) lfirst(lc),
                                                   fss->idx,
                                                   ((Scan *) fss->css.ss.ps.plan)->scanrelid,
                                                   econtext, &fss->css.ss.ps);

            /* a top-level conjunct that fails to compile is simply skipped:
             * the bitmap becomes a superset and ExecScan's qual recheck
             * removes the extras */
            if (p != NULL)
                compiled[n++] = p;
        }
        if (n == 1)
            root_pred = compiled[0];
        else if (n > 1)
        {
            root_pred = palloc0(sizeof(FvPredNode));
            root_pred->kind = FV_NODE_AND;
            root_pred->nchildren = n;
            root_pred->children = (const FvPredNode *const *) compiled;
        }
    }

    if (root_pred != NULL)
    {
        fss->pred = fv_pred_compile(fss->idx, root_pred);
        fss->nmatches = fv_pred_estimate(fss->pred);
    }
    else
        fss->nmatches = fv_index_count(fss->idx);

    if (fss->nmatches > 0)
    {
        /*
         * Selectivity-adaptive beam width. With inline filtering the
         * search implicitly widens as selectivity drops (the result
         * heap only admits passing nodes), so a low-selectivity filter
         * reaches the same recall with a smaller nominal ef. Scale
         * factors calibrated on SIFT1M across 5%..50% selectivity.
         */
        double sel = fss->pred ? fss->sel_permille / 1000.0 : 1.0;
        double scale = sel >= 0.4 ? 1.0 : sel >= 0.15 ? 0.8 : 0.4;
        int ef = Max(fss->k, (int) (paves_ef_search * scale));

        /* GPU strategies go through the micro-batch arbiter by default;
         * anything the arbiter cannot serve (worker down, queue full,
         * capacity exceeded) falls through to the direct/CPU cursors. */
        if ((fss->strategy == FV_STRAT_GPU_BRUTE ||
             fss->strategy == FV_STRAT_GPU_HNSW) &&
            paves_gpu_mode == FV_GPU_MODE_WORKER &&
            favor_arbiter_request(fss, q, ef, true))
        {
            fss->arb_query = q;
            fss->use_arb = true;
            fss->started = true;
            return;
        }

        /* backend-local CUDA only in direct mode: in worker mode a failed
         * arbiter submit must NOT trigger a per-backend index upload (64
         * backends x 700MB would exhaust the GPU); it degrades to CPU */
        if (fss->strategy == FV_STRAT_GPU_BRUTE)
        {
            if (paves_gpu_mode == FV_GPU_MODE_DIRECT)
                fss->cursor = fv_gpu_brute_begin(fss->idx, q, fss->pred,
                                                 paves_gpu_device,
                                                 paves_threads);
            if (fss->cursor == NULL)
                FV_ENGINE_TIMED(fss,
                                fss->cursor = fv_brute_begin(fss->idx, q,
                                                             fss->pred,
                                                             paves_threads));
        }
        else if (fss->strategy == FV_STRAT_BRUTE)
            FV_ENGINE_TIMED(fss,
                            fss->cursor = fv_brute_begin(fss->idx, q,
                                                         fss->pred,
                                                         paves_threads));
        else
        {
            if (fss->strategy == FV_STRAT_GPU_HNSW &&
                paves_gpu_mode == FV_GPU_MODE_DIRECT)
                fss->cursor = fv_gpu_hnsw_begin(fss->idx, q, fss->pred, ef, 0,
                                                paves_gpu_device,
                                                paves_threads);
            if (fss->cursor == NULL)
                FV_ENGINE_TIMED(fss,
                                fss->cursor = fv_hnsw_begin(fss->idx, q,
                                                            fss->pred, ef, 0));
        }
        if (fss->cursor != NULL)
            favor_count_cpu(fss);
    }
    fss->started = true;
}

static TupleTableSlot *favor_next(CustomScanState *node);

/* consume arbiter results (packed TIDs); re-submit wider when exhausted */
static TupleTableSlot *
favor_next_arbiter(FavorScanState *fss)
{
    TupleTableSlot *slot = fss->css.ss.ss_ScanTupleSlot;
    EState *estate = fss->css.ss.ps.state;

    for (;;)
    {
        FvHit hit;
        ItemPointerData tid;
        uint64 packed;

        CHECK_FOR_INTERRUPTS();

        if (fss->arb_pos >= fss->arb_n)
        {
            if (!fss->arb_more)
                return NULL;
            if (!favor_arbiter_request(fss, fss->arb_query, 0, false))
            {
                /*
                 * Wider than the arbiter can serve (or the worker went
                 * away): continue on a CPU cursor; the main loop skips
                 * candidates this scan already consumed.
                 */
                fss->use_arb = false;
                if (fss->strategy == FV_STRAT_GPU_BRUTE)
                    FV_ENGINE_TIMED(fss,
                                    fss->cursor =
                                    fv_brute_begin(fss->idx, fss->arb_query,
                                                   fss->pred,
                                                   paves_threads));
                else
                    FV_ENGINE_TIMED(fss,
                                    fss->cursor =
                                    fv_hnsw_begin(fss->idx, fss->arb_query,
                                                  fss->pred,
                                                  Max(fss->k,
                                                      (int) fss->arb_want),
                                                  0));
                favor_count_cpu(fss);
                return favor_next((CustomScanState *) fss);
            }
            continue;
        }
        hit = fss->arb_results[fss->arb_pos++];
        packed = hit.node; /* worker already translated to a packed TID */
        if (tid_seen(fss, packed))
            continue; /* served by an earlier, narrower request */
        tid_mark_seen(fss, packed);
        ItemPointerSet(&tid, (BlockNumber) (packed >> 16),
                       (OffsetNumber) (packed & 0xffff));

        fss->fetched++;
        if (table_tuple_fetch_row_version(fss->css.ss.ss_currentRelation,
                                          &tid, estate->es_snapshot, slot))
        {
            fss->emitted++;
            return slot;
        }
        fss->invisible++;
    }
}

static TupleTableSlot *
favor_next(CustomScanState *node)
{
    FavorScanState *fss = (FavorScanState *) node;
    TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
    EState *estate = node->ss.ps.state;

    if (!fss->started)
        favor_start_search(fss);
    if (fss->use_arb)
        return favor_next_arbiter(fss);
    if (fss->cursor == NULL)
        return NULL; /* no matches at all */

    for (;;)
    {
        FvHit hit;
        ItemPointerData tid;
        uint64 packed;

        CHECK_FOR_INTERRUPTS();

        if (fss->buf_pos >= fss->buf_n)
        {
            /*
             * Request no more than the LIMIT still needs: asking for extra
             * candidates can force the HNSW cursor to widen its beam and
             * re-search for results the query will never consume. Only
             * visibility-dropped candidates require a second, small batch.
             */
            int want = (int) Min((int64) FETCH_BATCH,
                                 (int64) fss->k - (int64) fss->emitted);

            if (want < 1)
                want = 4; /* LIMIT satisfied? executor knows best; drip-feed */
            FV_ENGINE_TIMED(fss,
                            fss->buf_n = fv_cursor_next(fss->cursor, fss->buf,
                                                        want));
            fss->buf_pos = 0;
            if (fss->buf_n == 0)
                return NULL; /* exhausted */
        }
        hit = fss->buf[fss->buf_pos++];

        packed = fv_index_tid(fss->idx, hit.node);
        if (fss->n_seen > 0 && tid_seen(fss, packed))
            continue; /* already served through the arbiter */
        ItemPointerSet(&tid, (BlockNumber) (packed >> 16),
                       (OffsetNumber) (packed & 0xffff));

        fss->fetched++;
        if (table_tuple_fetch_row_version(node->ss.ss_currentRelation, &tid,
                                          estate->es_snapshot, slot))
        {
            fss->emitted++;
            return slot;
        }
        fss->invisible++; /* deleted/updated since build: skip candidate */
    }
}

static bool
favor_recheck(CustomScanState *node, TupleTableSlot *slot)
{
    return true; /* quals are evaluated by ExecScan on every tuple */
}

static TupleTableSlot *
favor_exec(CustomScanState *node)
{
    return ExecScan(&node->ss,
                    (ExecScanAccessMtd) favor_next,
                    (ExecScanRecheckMtd) favor_recheck);
}

static void
favor_end(CustomScanState *node)
{
    favor_release_engine(node);
}

static void
favor_rescan(CustomScanState *node)
{
    FavorScanState *fss = (FavorScanState *) node;

    favor_release_engine(fss);
    fss->started = false;
    fss->buf_n = fss->buf_pos = 0;
    fss->emitted = 0;
    fss->use_arb = false;
    fss->arb_n = fss->arb_pos = 0;
    fss->arb_more = 0;
    fss->n_seen = 0;
    ExecScanReScan(&node->ss);
}

static void
favor_explain(CustomScanState *node, List *ancestors, ExplainState *es)
{
    FavorScanState *fss = (FavorScanState *) node;

    ExplainPropertyText("Strategy",
                        fss->strategy == FV_STRAT_BRUTE ? "BruteForce" :
                        fss->strategy == FV_STRAT_HNSW ? "HNSW" :
                        fss->strategy == FV_STRAT_GPU_BRUTE ? "GPU-BruteForce" :
                        "GPU-HNSW",
                        es);
    ExplainPropertyInteger("Pushdown Quals", NULL,
                           list_length(fss->pushdown_clauses), es);
    if (es->analyze && fss->started)
    {
        if (fss->strategy == FV_STRAT_GPU_BRUTE ||
            fss->strategy == FV_STRAT_GPU_HNSW)
            ExplainPropertyText("GPU Mode",
                                fss->use_arb ? "arbiter" : "direct/cpu", es);
        if (fss->nmatches != UINT64_MAX)
            ExplainPropertyInteger("Filter Matches", NULL, (int64) fss->nmatches, es);
        ExplainPropertyInteger("Candidates Fetched", NULL, (int64) fss->fetched, es);
        ExplainPropertyInteger("Invisible Skipped", NULL, (int64) fss->invisible, es);
    }
}

static CustomExecMethods favor_exec_methods = {
    .CustomName = "paves",
    .BeginCustomScan = favor_begin,
    .ExecCustomScan = favor_exec,
    .EndCustomScan = favor_end,
    .ReScanCustomScan = favor_rescan,
    .ExplainCustomScan = favor_explain,
};

static Node *
favor_create_scan_state(CustomScan *cscan)
{
    FavorScanState *fss = (FavorScanState *) newNode(sizeof(FavorScanState),
                                                     T_CustomScanState);

    fss->css.methods = &favor_exec_methods;
    return (Node *) fss;
}
