/* paves.h — internal declarations shared by the PG glue files. */
#ifndef PAVES_H
#define PAVES_H

#include "postgres.h"

#include "nodes/pathnodes.h"
#include "nodes/plannodes.h"
#include "nodes/extensible.h"
#include "utils/relcache.h"

#include "engine/engine_api.h"

/* GUCs (defined in paves.c) */
extern bool paves_enable;
extern int paves_ef_search;
extern int paves_threads;
extern int paves_force_strategy; /* FvForceStrategy */
extern int paves_build_m;
extern int paves_build_ef;
extern int paves_build_threads;
extern double paves_cost_scale;
extern int paves_gpu_device;
extern int paves_gpu_mode;
extern int paves_gpu_batch_max;
extern int paves_gpu_batch_wait_us;
extern int paves_gpu_batch_idle_us;
extern int paves_cpu_cores;
extern int paves_gpu_precision; /* 0 = fp32, 1 = sq8 */
extern bool paves_gpu_enable;   /* auto mode considers GPU strategies */

typedef enum FvGpuMode {
    FV_GPU_MODE_WORKER = 0, /* submit to the micro-batch arbiter worker */
    FV_GPU_MODE_DIRECT = 1, /* backend-local CUDA (testing / few backends) */
} FvGpuMode;

typedef enum FvForceStrategy {
    FV_FORCE_AUTO = 0,
    FV_FORCE_BRUTE = 1,
    FV_FORCE_HNSW = 2,
    FV_FORCE_GPU_BRUTE = 3,
    FV_FORCE_GPU_HNSW = 4,
} FvForceStrategy;

typedef enum FvStrategy {
    FV_STRAT_BRUTE = 1,
    FV_STRAT_HNSW = 2,
    FV_STRAT_GPU_BRUTE = 3,
    FV_STRAT_GPU_HNSW = 4,
} FvStrategy;

/* custom_private layout: list_make4(makeInteger(strategy), makeInteger(k),
 * queryExpr, pushdownClauses) built by planner.c; moved to custom_exprs
 * where noted in planner.c. */

/* index cache (planner.c) */
extern FvIndex *paves_get_index(Oid relid);
extern char *paves_index_path(Oid relid);
extern void paves_register_planner_hooks(void);

/* qual compilation (quals.c) */
extern bool paves_clause_pushdownable(Node *clause, FvIndex *idx, Index varno);
extern FvPredNode *paves_compile_clause(Node *clause, FvIndex *idx, Index varno,
                                          ExprContext *econtext, PlanState *ps);

/* pgvector interop */
extern Oid paves_vector_typid(void);
extern Oid paves_l2_opno(void);

/* minimal mirror of pgvector's Vector varlena layout */
typedef struct PgvVector {
    int32 vl_len_;
    int16 dim;
    int16 unused;
    float x[FLEXIBLE_ARRAY_MEMBER];
} PgvVector;

extern CustomScanMethods paves_scan_methods;

#endif /* PAVES_H */
