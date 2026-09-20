/*
 * planner.c — paves planner integration.
 *
 * Recognizes  SELECT ... FROM rel WHERE quals
 *             ORDER BY vec_col <-> <query> LIMIT k
 * on relations that have a paves index file, and injects two CustomPaths
 * (exact brute force / approximate HNSW) whose relative costs are driven by
 * PostgreSQL's own selectivity estimation; the optimizer picks by cost.
 */
#include "paves.h"
#include "arbiter.h"

#include <sys/stat.h>

#include "access/stratnum.h"
#include "catalog/namespace.h"
#include "catalog/pg_class.h"
#include "catalog/pg_type.h"
#include "miscadmin.h"
#include "nodes/makefuncs.h"
#include "nodes/nodeFuncs.h"
#include "nodes/value.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "optimizer/pathnode.h"
#include "optimizer/paths.h"
#include "optimizer/restrictinfo.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"

static set_rel_pathlist_hook_type prev_set_rel_pathlist_hook = NULL;

/* ---------------- pgvector OID lookups (cached per backend) ---------------- */

static Oid cached_vector_typid = InvalidOid;
static Oid cached_l2_opno = InvalidOid;

Oid
paves_vector_typid(void)
{
    if (!OidIsValid(cached_vector_typid))
        cached_vector_typid = TypenameGetTypid("vector");
    return cached_vector_typid;
}

Oid
paves_l2_opno(void)
{
    if (!OidIsValid(cached_l2_opno))
    {
        Oid typid = paves_vector_typid();

        if (OidIsValid(typid))
            cached_l2_opno = OpernameGetOprid(list_make1(makeString("<->")),
                                              typid, typid);
    }
    return cached_l2_opno;
}

/* ---------------- index file cache ---------------- */

typedef struct IndexCacheEntry {
    Oid relid;
    FvIndex *idx;      /* NULL for negative entry */
    ino_t ino;
    off_t size;
    time_t mtime;
} IndexCacheEntry;

static HTAB *index_cache = NULL;

char *
paves_index_path(Oid relid)
{
    return psprintf("%s/paves/%u_%u.idx", DataDir, MyDatabaseId, relid);
}

/*
 * Return the (mmap'ed) index for a relation, or NULL if there is none.
 * Re-opens when the file changed (rebuild). Replaced FvIndex mappings are
 * intentionally not unmapped: open cursors of held portals may still
 * reference them; the leak is bounded by the number of rebuilds per session.
 */
FvIndex *
paves_get_index(Oid relid)
{
    IndexCacheEntry *entry;
    bool found;
    char *path;
    struct stat st;

    if (index_cache == NULL)
    {
        HASHCTL ctl;

        MemSet(&ctl, 0, sizeof(ctl));
        ctl.keysize = sizeof(Oid);
        ctl.entrysize = sizeof(IndexCacheEntry);
        ctl.hcxt = TopMemoryContext;
        index_cache = hash_create("paves index cache", 16, &ctl,
                                  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
    }

    path = paves_index_path(relid);
    if (stat(path, &st) != 0)
    {
        entry = hash_search(index_cache, &relid, HASH_FIND, NULL);
        if (entry)
            entry->idx = NULL; /* file went away; abandon mapping */
        pfree(path);
        return NULL;
    }

    entry = hash_search(index_cache, &relid, HASH_ENTER, &found);
    if (!found)
        entry->idx = NULL;
    if (entry->idx == NULL || entry->ino != st.st_ino ||
        entry->size != st.st_size || entry->mtime != st.st_mtime)
    {
        char errbuf[256];
        FvIndex *idx = fv_index_open(path, errbuf, sizeof(errbuf));

        if (idx == NULL)
        {
            ereport(WARNING,
                    (errmsg("paves: cannot open index for relation %u: %s",
                            relid, errbuf)));
            entry->idx = NULL;
            pfree(path);
            return NULL;
        }
        entry->idx = idx;
        entry->ino = st.st_ino;
        entry->size = st.st_size;
        entry->mtime = st.st_mtime;
    }
    pfree(path);
    return entry->idx;
}

/* ---------------- path generation ---------------- */

static Plan *favor_plan_custom_path(PlannerInfo *root, RelOptInfo *rel,
                                    CustomPath *best_path, List *tlist,
                                    List *clauses, List *custom_plans);

static const CustomPathMethods favor_path_methods = {
    .CustomName = "paves",
    .PlanCustomPath = favor_plan_custom_path,
};

/*
 * Cost model. All four strategies are priced in predicted milliseconds and
 * converted to cost units by one factor, so paths compete on a single
 * scale. Each prediction combines three inputs:
 *
 *   - analytic priors: dataset shape (N rows, dim) through per-strategy
 *     work formulas, constants calibrated on SIFT1M (128d, 1M rows,
 *     EPYC 9654) where measured brute/HNSW crossover sits at ~1.5%
 *     selectivity (model: ~1.9%);
 *   - per-index runtime feedback (fv_arb->fb): the GPU worker publishes
 *     amortized kernel service times, batch wall time and batch size;
 *     backends publish observed CPU engine time normalized to an unloaded
 *     machine. Observations anchor the *level*; the analytic formulas
 *     provide the *shape* across selectivity, interpolating a bucketed
 *     observation to the query's actual selectivity (and to neighbor
 *     buckets when the query's own bucket is still empty);
 *   - live load: gpu_inflight prices GPU queue wait in units of batches
 *     (the kernel serves a whole micro-batch concurrently — pricing the
 *     queue per *request* overestimates wait by the batch size, which is
 *     exactly what pushed auto off the GPU under high concurrency), and
 *     cpu_active scales CPU costs by a saturation factor. The two loops
 *     shed load to whichever device has headroom.
 */
#define FV_DIST_COST_FACTOR   0.02  /* per-dimension SIMD distance work */
#define FV_ENUM_COST_FACTOR   0.2   /* per-candidate posting enumeration+eval */
#define FV_HOP_OVERHEAD       2.0   /* graph traversal overhead per visited */
#define FV_COST_UNITS_PER_MS  7.5   /* cost units per predicted millisecond */
#define FV_GPU_FIXED_MS       0.4   /* submit/wake/answer round trip */
#define FV_GPU_SEED_BRUTE_MS  1.0 /* before the worker reports anything */
#define FV_GPU_SEED_GRAPH_MS  0.5

/* analytic CPU service time in ms, unloaded machine */
static double
cpu_prior_ms(int strategy, double sel, double N, int dim, int k, int threads,
             bool have_pushdown)
{
    double dist_unit = dim * cpu_operator_cost * FV_DIST_COST_FACTOR;
    double units;

    if (strategy == FV_STRAT_BRUTE)
    {
        /* fused enumerate+eval+distance over driver spans, multi-threaded;
         * without a pushdown driver this is a full scan */
        double enumerated = have_pushdown ? Max(sel * N, 1.0) : N;

        units = enumerated *
            (dist_unit + cpu_operator_cost * FV_ENUM_COST_FACTOR) / threads;
    }
    else
    {
        /* single-threaded beam search; with inline filtering the beam must
         * visit ~ef/selectivity nodes before the result heap saturates */
        double sel_eff = Max(sel, 0.0005);
        double ef_eff = Max((double) paves_ef_search, 2.0 * k);
        double visited = Min(N, ef_eff / sel_eff + ef_eff * 4.0);

        units = visited * (dist_unit + cpu_operator_cost * FV_HOP_OVERHEAD);
    }
    return units / FV_COST_UNITS_PER_MS;
}

/* nearest bucket with an observation, preferring the query's own; the
 * caller rescales by its strategy's work shape when src != b */
static uint32
feedback_nearest(pg_atomic_uint32 *row, int b, int *src_b)
{
    int d;

    for (d = 0; d < FV_SEL_BUCKETS; d++)
    {
        uint32 v;

        if (b - d >= 0 && (v = pg_atomic_read_u32(&row[b - d])) != 0)
        {
            *src_b = b - d;
            return v;
        }
        if (b + d < FV_SEL_BUCKETS &&
            (v = pg_atomic_read_u32(&row[b + d])) != 0)
        {
            *src_b = b + d;
            return v;
        }
    }
    return 0;
}

static double
cpu_saturation_factor(int threads_per_query)
{
    double used;

    if (fv_arb == NULL)
        return 1.0;
    used = (double) pg_atomic_read_u32(&fv_arb->cpu_active) *
        Max(threads_per_query, 1);
    return Max(1.0, used / Max(paves_cpu_cores, 1));
}

/*
 * GPU path latency prediction:
 *   fixed round trip
 *   + own amortized service time (observed per-index, per-bucket)
 *   + batch wall time x batches queued ahead (inflight / average batch
 *     size — the worker drains the queue one whole micro-batch at a time).
 * Pricing own service at the amortized (marginal) rate rather than the
 * full batch time is deliberate: it is what lets traffic grow batches.
 */
static double
gpu_path_cost_ms(FvFeedbackEntry *fb, int strategy, double sel)
{
    int si = (strategy == FV_STRAT_GPU_HNSW) ? 1 : 0;
    int b = fv_sel_bucket((int) (sel * 1000));
    double inflight = fv_arb ?
        (double) pg_atomic_read_u32(&fv_arb->gpu_inflight) : 0.0;
    double q_ms = si ? FV_GPU_SEED_GRAPH_MS : FV_GPU_SEED_BRUTE_MS;
    double batch_ms;
    double navg = 1.0;

    if (fb != NULL)
    {
        int src_b = b;
        uint32 q = feedback_nearest(fb->gpu_q_us[si], b, &src_b);
        uint32 bus = pg_atomic_read_u32(&fb->gpu_batch_us[si]);
        uint32 n16 = pg_atomic_read_u32(&fb->gpu_batch_n16[si]);

        if (q != 0)
        {
            q_ms = q / 1000.0;
            /* graph work ~ 1/selectivity; brute is flat across buckets */
            if (si == 1 && src_b != b)
                q_ms *= fv_sel_bucket_mid(src_b) / fv_sel_bucket_mid(b);
        }
        if (n16 != 0)
            navg = Max(n16 / 16.0, 1.0);
        batch_ms = bus != 0 ? bus / 1000.0 : q_ms;
    }
    else
        batch_ms = q_ms; /* cold start: optimistic, lets traffic flow */

    return FV_GPU_FIXED_MS + q_ms + batch_ms * (inflight / navg);
}

/*
 * CPU path latency prediction: observed loaded engine time for this index
 * (level), interpolated across selectivity by the analytic prior (shape),
 * extrapolated to the current load by the ratio of live cpu_active to the
 * level the observation was measured under. The live ratio is the instant
 * negative feedback: flooding the CPU raises every subsequent estimate
 * before the EWMA has absorbed a single new sample. Falls back to the
 * analytic prior x saturation until the first query of this shape lands.
 */
static double
cpu_path_cost_ms(FvFeedbackEntry *fb, int strategy, double sel, double N,
                 int dim, int k, int threads, bool have_pushdown)
{
    int si = (strategy == FV_STRAT_HNSW) ? 1 : 0;
    int b = fv_sel_bucket((int) (sel * 1000));
    double prior_now = cpu_prior_ms(strategy, sel, N, dim, k, threads,
                                    have_pushdown);

    if (fb != NULL)
    {
        int src_b = b;
        uint32 q = feedback_nearest(fb->cpu_q_us[si], b, &src_b);

        if (q != 0)
        {
            double prior_src = cpu_prior_ms(strategy,
                                            fv_sel_bucket_mid(src_b), N, dim,
                                            k, threads, have_pushdown);
            double base = (q / 1000.0) * (prior_now / Max(prior_src, 1e-9));
            double ref = pg_atomic_read_u32(&fb->cpu_ref_active16[si]) / 16.0;
            double now = fv_arb ?
                (double) pg_atomic_read_u32(&fv_arb->cpu_active) + 1.0 : 1.0;

            /* floor at a fraction of the prior: cancelled-query fragments
             * must not flood the CPU with absurdly cheap estimates. No
             * upper clamp — under heavy concurrency the loaded cost
             * legitimately exceeds any multiple of the unloaded prior,
             * and capping it once broke the 448-client equilibrium. */
            base = Max(base, prior_now * 0.125);
            /* symmetric bounded extrapolation of the contention: latency
             * scales ~linearly with the active count in the bandwidth-
             * shared regime, and pricing must react in BOTH directions —
             * up instantly when a flood starts, down when the CPU drains
             * (a one-sided max() left post-flood observations stuck at
             * flood level and starved an idle device) */
            return base * Min(Max(now / Max(ref, 1.0), 0.25), 4.0);
        }
    }
    return prior_now *
        cpu_saturation_factor(strategy == FV_STRAT_BRUTE ? threads : 1);
}

static void
add_favor_path(PlannerInfo *root, RelOptInfo *rel, FvIndex *idx, Oid relid,
               int strategy, int k, Node *qexpr, List *pushdown_clauses,
               double sel, bool have_pushdown)
{
    CustomPath *cp = makeNode(CustomPath);
    double N = (double) fv_index_count(idx);
    int dim = fv_index_dim(idx);
    double matches = Max(sel * N, 1.0);
    int threads = Max(paves_threads, 1);
    FvFeedbackEntry *fb = fv_feedback_find(MyDatabaseId, relid);
    double ms;
    double run_cost;

    if (strategy == FV_STRAT_GPU_BRUTE || strategy == FV_STRAT_GPU_HNSW)
        ms = gpu_path_cost_ms(fb, strategy, sel);
    else
        ms = cpu_path_cost_ms(fb, strategy, sel, N, dim, k, threads,
                              have_pushdown);
    run_cost = ms * FV_COST_UNITS_PER_MS;

    cp->path.pathtype = T_CustomScan;
    cp->path.parent = rel;
    cp->path.pathtarget = rel->reltarget;
    cp->path.param_info = NULL;
    cp->path.parallel_aware = false;
    cp->path.parallel_safe = false;
    cp->path.parallel_workers = 0;
    cp->path.rows = Min((double) k, Max(matches, 1.0));
    cp->path.startup_cost = run_cost * paves_cost_scale;
    cp->path.total_cost = cp->path.startup_cost +
        k * cpu_tuple_cost * paves_cost_scale;
    cp->path.pathkeys = root->query_pathkeys;
    cp->flags = 0;
    cp->custom_paths = NIL;
    cp->custom_private = list_make5(makeInteger(strategy), makeInteger(k),
                                    qexpr, pushdown_clauses,
                                    makeInteger((int) (sel * 1000)));
    cp->methods = &favor_path_methods;

    add_path(rel, (Path *) cp);
}

static void
favor_set_rel_pathlist(PlannerInfo *root, RelOptInfo *rel, Index rti,
                       RangeTblEntry *rte)
{
    PathKey *pk;
    EquivalenceClass *ec;
    ListCell *lc;
    OpExpr *distexpr = NULL;
    Node *qexpr = NULL;
    FvIndex *idx;
    Oid l2op;
    int k;
    List *pushdown_clauses = NIL;
    double sel;
    bool all_pushdown = true;

    if (prev_set_rel_pathlist_hook)
        prev_set_rel_pathlist_hook(root, rel, rti, rte);

#ifdef PAVES_DEBUG_PLAN
#define FV_BAIL(why) do { elog(NOTICE, "paves bail: %s", why); return; } while (0)
#else
#define FV_BAIL(why) return
#endif

    if (!paves_enable)
        FV_BAIL("disabled");
    if (rte->rtekind != RTE_RELATION || rte->relkind != RELKIND_RELATION ||
        rel->reloptkind != RELOPT_BASEREL)
        FV_BAIL("not a plain base relation");
    if (root->query_pathkeys == NIL || list_length(root->query_pathkeys) != 1)
        FV_BAIL("query_pathkeys shape");
    if (root->limit_tuples <= 0 || root->limit_tuples > 1000000)
        FV_BAIL("no usable limit");
    k = (int) root->limit_tuples;

    pk = (PathKey *) linitial(root->query_pathkeys);
    if (pk->pk_strategy != BTLessStrategyNumber)
        FV_BAIL("pathkey not ASC");
    ec = pk->pk_eclass;
    if (ec->ec_has_volatile)
        FV_BAIL("volatile pathkey");

    l2op = paves_l2_opno();
    if (!OidIsValid(l2op))
        FV_BAIL("no <-> operator");

    idx = paves_get_index(rte->relid);
    if (idx == NULL)
        FV_BAIL("no index file");

    /* find EC member of the form: our_vec_col <-> var-free-expr */
    foreach(lc, ec->ec_members)
    {
        EquivalenceMember *em = (EquivalenceMember *) lfirst(lc);
        Node *expr = (Node *) em->em_expr;
        OpExpr *op;
        Node *left, *right, *varside = NULL, *other = NULL;
        Var *var;

        while (expr && IsA(expr, RelabelType))
            expr = (Node *) ((RelabelType *) expr)->arg;
        if (!expr || !IsA(expr, OpExpr))
            continue;
        op = (OpExpr *) expr;
        if (op->opno != l2op || list_length(op->args) != 2)
            continue;
        left = (Node *) linitial(op->args);
        right = (Node *) lsecond(op->args);
        while (left && IsA(left, RelabelType))
            left = (Node *) ((RelabelType *) left)->arg;
        while (right && IsA(right, RelabelType))
            right = (Node *) ((RelabelType *) right)->arg;

        if (left && IsA(left, Var) && ((Var *) left)->varno == rti &&
            ((Var *) left)->varlevelsup == 0)
        {
            varside = left;
            other = (Node *) lsecond(op->args);
        }
        else if (right && IsA(right, Var) && ((Var *) right)->varno == rti &&
                 ((Var *) right)->varlevelsup == 0)
        {
            varside = right;
            other = (Node *) linitial(op->args);
        }
        else
            continue;

        var = (Var *) varside;
        if (var->varattno != fv_index_vec_attnum(idx))
            continue;
        if (contain_var_clause(other) || contain_volatile_functions(other))
            continue;

        distexpr = op;
        qexpr = other;
        break;
    }
    if (distexpr == NULL)
        FV_BAIL("no matching EC member");

    /* classify restriction clauses for engine pushdown */
    foreach(lc, rel->baserestrictinfo)
    {
        RestrictInfo *ri = (RestrictInfo *) lfirst(lc);

        if (ri->pseudoconstant)
        {
            all_pushdown = false;
            continue;
        }
        if (paves_clause_pushdownable((Node *) ri->clause, idx, rti))
            pushdown_clauses = lappend(pushdown_clauses, ri->clause);
        else
            all_pushdown = false;
    }
    (void) all_pushdown;

    sel = rel->tuples > 0 ? Min(1.0, rel->rows / rel->tuples) : 1.0;

    if (paves_force_strategy == FV_FORCE_AUTO ||
        paves_force_strategy == FV_FORCE_BRUTE)
        add_favor_path(root, rel, idx, rte->relid, FV_STRAT_BRUTE, k, qexpr,
                       pushdown_clauses, sel, pushdown_clauses != NIL);
    if (paves_force_strategy == FV_FORCE_AUTO ||
        paves_force_strategy == FV_FORCE_HNSW)
        add_favor_path(root, rel, idx, rte->relid, FV_STRAT_HNSW, k, qexpr,
                       pushdown_clauses, sel, pushdown_clauses != NIL);

    /* GPU paths join the cost race when the arbiter can serve the query */
    if (paves_force_strategy == FV_FORCE_AUTO && paves_gpu_enable &&
        paves_gpu_mode == FV_GPU_MODE_WORKER && fv_arb != NULL &&
        pg_atomic_read_u32(&fv_arb->worker_ready) == 1 &&
        fv_index_dim(idx) <= FV_ARB_MAX_DIM &&
        (uint32) Max(4 * k, 64) <= FV_ARB_MAX_RESULTS)
    {
        add_favor_path(root, rel, idx, rte->relid, FV_STRAT_GPU_BRUTE, k,
                       qexpr, pushdown_clauses, sel, pushdown_clauses != NIL);
        add_favor_path(root, rel, idx, rte->relid, FV_STRAT_GPU_HNSW, k,
                       qexpr, pushdown_clauses, sel, pushdown_clauses != NIL);
    }
    if (paves_force_strategy == FV_FORCE_GPU_BRUTE)
        add_favor_path(root, rel, idx, rte->relid, FV_STRAT_GPU_BRUTE, k,
                       qexpr, pushdown_clauses, sel, pushdown_clauses != NIL);
    if (paves_force_strategy == FV_FORCE_GPU_HNSW)
        add_favor_path(root, rel, idx, rte->relid, FV_STRAT_GPU_HNSW, k,
                       qexpr, pushdown_clauses, sel, pushdown_clauses != NIL);
}

static Plan *
favor_plan_custom_path(PlannerInfo *root, RelOptInfo *rel,
                       CustomPath *best_path, List *tlist,
                       List *clauses, List *custom_plans)
{
    CustomScan *cscan = makeNode(CustomScan);
    int strategy = intVal(linitial(best_path->custom_private));
    int k = intVal(lsecond(best_path->custom_private));
    Node *qexpr = (Node *) lthird(best_path->custom_private);
    List *pushdown = (List *) lfourth(best_path->custom_private);
    int sel_permille = intVal(list_nth(best_path->custom_private, 4));

    cscan->scan.plan.targetlist = tlist;
    cscan->scan.plan.qual = extract_actual_clauses(clauses, false);
    cscan->scan.scanrelid = rel->relid;
    cscan->flags = best_path->flags;
    cscan->custom_plans = NIL;
    /* custom_exprs get setrefs processing (Var fixups); private ints don't */
    cscan->custom_exprs = list_make2(qexpr, pushdown);
    cscan->custom_private = list_make3(makeInteger(strategy), makeInteger(k),
                                       makeInteger(sel_permille));
    cscan->methods = &paves_scan_methods;

    return &cscan->scan.plan;
}

void
paves_register_planner_hooks(void)
{
    prev_set_rel_pathlist_hook = set_rel_pathlist_hook;
    set_rel_pathlist_hook = favor_set_rel_pathlist;
}
