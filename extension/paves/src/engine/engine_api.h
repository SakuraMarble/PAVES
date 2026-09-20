/*
 * engine_api.h — C interface to the paves C++ vector engine.
 *
 * The engine is pure C++ with no PostgreSQL dependencies. All PG glue code
 * talks to it exclusively through this header. GPU strategies plug in behind
 * the same cursor interface (see README, "GPU extension points").
 *
 * Threading contract: engine worker threads never call back into the caller
 * and never touch PostgreSQL APIs; all entry points below are synchronous and
 * must be called from the backend main thread.
 *
 * Predicate execution model (no O(N) work per query):
 *  - HNSW search evaluates the compiled predicate per visited node.
 *  - Brute-force search enumerates "driver" spans from per-column posting
 *    lists (row ids sorted by column value) and fuses predicate evaluation
 *    with distance computation; full scan is the fallback when no driver
 *    exists (e.g. NOT-only predicates).
 */
#ifndef PAVES_ENGINE_API_H
#define PAVES_ENGINE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FvIndex FvIndex;
typedef struct FvBuilder FvBuilder;
typedef struct FvCursor FvCursor;
typedef struct FvPred FvPred;

#define FV_MAX_COLS 32

/* ---------------- build ---------------- */

FvBuilder *fv_builder_create(int dim, int ncols, const int32_t *col_attnums,
                             uint64_t reserve);
/* scalars[ncols]: numeric value per cached column; isnull -> stored as NaN */
void fv_builder_add_row(FvBuilder *b, const float *vec, uint64_t tid,
                        const double *scalars, const uint8_t *isnull);
/* Builds HNSW graph + posting lists and atomically writes the index file.
 * Returns 0 on success, -1 on failure (message in errbuf). */
int fv_builder_finish(FvBuilder *b, const char *path, int32_t vec_attnum,
                      int M, int ef_construction, int threads,
                      char *errbuf, size_t errlen);
void fv_builder_destroy(FvBuilder *b);

/* ---------------- open / metadata ---------------- */

FvIndex *fv_index_open(const char *path, char *errbuf, size_t errlen);
void fv_index_close(FvIndex *idx);
uint64_t fv_index_count(const FvIndex *idx);
int fv_index_dim(const FvIndex *idx);
int32_t fv_index_vec_attnum(const FvIndex *idx);
int fv_index_ncols(const FvIndex *idx);
/* column slot for a table attnum, or -1 if that column is not cached */
int fv_index_col_slot(const FvIndex *idx, int32_t attnum);
uint64_t fv_index_tid(const FvIndex *idx, uint64_t node);
/* UINT64_MAX when node is out of range (never reads past the mapping) */
uint64_t fv_index_tid_checked(const FvIndex *idx, uint64_t node);
/* tooling: raw row vector; percentile of a cached column's sorted values */
const float *fv_index_vector(const FvIndex *idx, uint64_t node);
double fv_index_col_percentile(const FvIndex *idx, int col_slot, double frac);

/* ---------------- predicates ---------------- */

typedef enum { FV_CMP_LT, FV_CMP_LE, FV_CMP_EQ, FV_CMP_GE, FV_CMP_GT, FV_CMP_NE } FvCmpOp;
typedef enum { FV_NODE_CMP, FV_NODE_IN, FV_NODE_AND, FV_NODE_OR, FV_NODE_NOT } FvNodeKind;

/* Predicate tree handed over by the glue. Semantics: NULL column values are
 * stored as NaN and every comparison with NaN is false, so evaluation can
 * only produce false POSITIVES relative to SQL three-valued logic (via NOT);
 * the executor rechecks all quals on emitted tuples, so this is safe. */
typedef struct FvPredNode {
    int32_t kind;      /* FvNodeKind */
    int32_t col;       /* CMP/IN: column slot */
    int32_t op;        /* CMP: FvCmpOp */
    double value;      /* CMP */
    int32_t nvalues;   /* IN */
    const double *values;
    int32_t nchildren; /* AND/OR/NOT */
    const struct FvPredNode *const *children;
} FvPredNode;

/* Compile (deep-copy) a predicate tree into the engine's internal form,
 * selecting posting-list driver spans. Engine-owned; free with fv_pred_free. */
FvPred *fv_pred_compile(FvIndex *idx, const FvPredNode *tree);
void fv_pred_free(FvPred *pred);
/* Exact match count when a covering driver exists, UINT64_MAX if unknown.
 * A return of 0 is authoritative: no rows can match. */
uint64_t fv_pred_estimate(const FvPred *pred);

/* ---------------- GPU submission format ----------------
 *
 * A compiled predicate serializes into flat arrays that can live in shared
 * memory and be shipped to the GPU worker: nodes reference children/IN
 * values through [child0, child0+nchildren) / [val0, val0+nvalues) ranges in
 * the side arrays; node 0 of a query's slice is its root. Driver spans give
 * the brute-force enumeration set (see engine_api.h predicate model). */
typedef struct FvGpuPredNode {
    int32_t kind;
    int32_t col;
    int32_t op;
    int32_t nchildren;
    int32_t child0;
    int32_t nvalues;
    int32_t val0;
    int32_t pad;
    double value;
} FvGpuPredNode;

typedef struct FvGpuSpan {
    int32_t col;
    uint32_t lo, hi; /* [lo, hi) positions in post_rows(col) */
} FvGpuSpan;

/* One query's serialized predicate, pointing into caller-owned arrays
 * (all pointers may be NULL for "no predicate"). */
typedef struct FvGpuBatchPred {
    const FvGpuPredNode *nodes;
    int nnodes;
    const int32_t *children;
    int nchildren;
    const double *invals;
    int ninvals;
    const FvGpuSpan *spans;
    int nspans;
    int need_eval; /* brute: rows must be re-evaluated */
    int dedup;     /* brute: spans may repeat rows */
} FvGpuBatchPred;

/* Serialize `pred` into caller buffers. Returns 0 and fills bp (pointing at
 * the caller buffers) plus *total_rows (brute enumeration size), or -1 when
 * a capacity is exceeded (caller keeps the query on the CPU). pred may be
 * NULL (no filter). */
int fv_pred_serialize(FvIndex *idx, const FvPred *pred, FvGpuPredNode *nodes,
                      int max_nodes, int32_t *children, int max_children,
                      double *invals, int max_invals, FvGpuSpan *spans,
                      int max_spans, FvGpuBatchPred *bp, uint64_t *total_rows);

/* ---------------- search ---------------- */

typedef struct FvHit {
    uint64_t node;
    float dist; /* squared L2 */
} FvHit;

/* Exact multi-threaded scan: driver spans if available, else full scan.
 * pred may be NULL (no filter). */
FvCursor *fv_brute_begin(FvIndex *idx, const float *query, const FvPred *pred,
                         int threads);
/* GPU exact top-k (same semantics as fv_brute_begin, computed on a CUDA
 * device; the index is uploaded lazily on first use and stays resident).
 * device -1 auto-picks the GPU with the most free memory. Returns NULL when
 * no usable device exists or the build has no GPU support — the caller
 * chooses the CPU fallback. Runtime CUDA errors fall back internally to a
 * CPU brute cursor (fallback_threads workers). */
FvCursor *fv_gpu_brute_begin(FvIndex *idx, const float *query,
                             const FvPred *pred, int device,
                             int fallback_threads);
/* GPU filtered graph search (same semantics as fv_hnsw_begin: traversal is
 * unrestricted, results must satisfy pred). The CPU finds the level-0 entry
 * point via the upper layers; the GPU runs the level-0 beam search. Returns
 * NULL when no usable device exists, ef exceeds the GPU bound, or the build
 * has no GPU support. Runtime CUDA errors fall back internally to a CPU
 * HNSW cursor. */
FvCursor *fv_gpu_hnsw_begin(FvIndex *idx, const float *query,
                            const FvPred *pred, int ef, uint64_t max_visited,
                            int device, int fallback_threads);

/* ---------------- GPU batch execution (arbiter worker) ----------------
 *
 * These run many queries in one kernel launch; the GPU-resident index is
 * uploaded on first use (fv_index_gpu_ensure). All return 0 on success and
 * -1 on CUDA/build errors (callers fall back to CPU per query). Without GPU
 * support in the build they return -1. */

/* Upload the index to a GPU (idempotent). device -1 = auto. */
int fv_index_gpu_ensure(FvIndex *idx, int device);

/* GPU vector precision for future uploads: 0 = fp32 (exact, bit-identical
 * to the CPU cursors), 1 = sq8 (int8 scan + fp16 rerank, ~4x less device
 * memory/bandwidth, approximate results). Set from the paves.gpu_precision
 * GUC; an index already resident keeps the precision it was uploaded with. */
void fv_engine_gpu_precision(int precision);
int fv_engine_gpu_precision_get(void);

/* Capacity probing (for paves_capacity()): the query-tile width of the
 * fast brute path for a dimension (0 = legacy kernel only), and free/total
 * bytes of the best CUDA device (returns device id, -1 = no usable GPU;
 * creates a CUDA context — call from the GPU worker, not from backends). */
int fv_gpu_tile_variant_qt(int dim);
int fv_gpu_mem_probe(uint64_t *free_bytes, uint64_t *total_bytes);

/* Unfiltered greedy descent through the upper HNSW layers: the level-0
 * entry point for a query (used to seed the GPU graph search). */
uint32_t fv_index_graph_entry(FvIndex *idx, const float *query);

/* Exact filtered top-topb for nq queries; queries is nq*dim floats, preds
 * one entry per query. out is nq*topb hits; out_cnt/truncated per query
 * (truncated as in the brute cursor: candidates beyond topb may exist). */
int fv_gpu_exec_brute_batch(FvIndex *idx, int nq, const float *queries,
                            const FvGpuBatchPred *preds, uint32_t topb,
                            FvHit *out, uint32_t *out_cnt, uint8_t *truncated);

/* Filtered graph search for nq queries with per-query entry points; out is
 * nq*ef hits ascending; exhausted per query (no wider ef can add results). */
int fv_gpu_exec_graph_batch(FvIndex *idx, int nq, const float *queries,
                            const uint32_t *entries,
                            const FvGpuBatchPred *preds, uint32_t ef,
                            uint64_t max_visited, FvHit *out,
                            uint32_t *out_cnt, uint8_t *exhausted);
/* Filtered HNSW: traversal is unrestricted, results must satisfy pred.
 * ef is the initial beam width; the cursor widens automatically when the
 * consumer requests more results than the current beam produced.
 * max_visited bounds graph exploration (0 = engine default). */
FvCursor *fv_hnsw_begin(FvIndex *idx, const float *query, const FvPred *pred,
                        int ef, uint64_t max_visited);
/* Next up-to-want hits in ascending distance order (across calls).
 * Returns number produced; 0 = exhausted. */
int fv_cursor_next(FvCursor *cur, FvHit *out, int want);
void fv_cursor_end(FvCursor *cur);

#ifdef __cplusplus
}
#endif
#endif /* PAVES_ENGINE_API_H */
