/*
 * gpu_engine.h — internal interface between engine.cpp and the CUDA engine
 * (gpu_engine.cu). Not part of the public engine API; pure C declarations
 * with no CUDA headers so engine.cpp can include it unconditionally.
 * FvGpuPredNode / FvGpuSpan live in engine_api.h (they are also the shared-
 * memory submission format for the arbiter worker).
 *
 * Threading contract matches engine_api.h: all entry points are synchronous
 * and must be called from the calling process's main thread. CUDA
 * initialization is lazy (first upload), so it always happens after fork().
 */
#ifndef PAVES_GPU_ENGINE_H
#define PAVES_GPU_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#include "engine_api.h" /* FvHit, FvGpuPredNode, FvGpuSpan */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FvGpuIndex FvGpuIndex;

/* GPU-resident vector precision.
 * FP32: exact — bit-identical to the CPU cursors (the original path).
 * SQ8:  scalar-quantized int8 scan (per-dimension centering + one global
 *       scale, distances via dp4a in exact integer arithmetic) with an
 *       fp16 rerank of the top candidates. ~4x less device memory and
 *       bandwidth; results are approximate (recall >= 0.99 in practice,
 *       exactness contract dropped). */
#define FV_GPU_PREC_FP32 0
#define FV_GPU_PREC_SQ8 1

/* Hard cap on per-call top-B; larger requests must fall back to the CPU. */
#define FV_GPU_MAX_TOPB 4096
/* Spans beyond this cap: serialization switches to full scan + need_eval. */
#define FV_GPU_MAX_SPANS 4096
/* Graph search result capacity per query (shared-memory bound); ef beyond
 * this must fall back to the CPU HNSW cursor. */
#define FV_GPU_MAX_EF 512
/* Max queries per batch (one thread block each for graph search). */
#define FV_GPU_MAX_BATCH 1024

/* Per-query descriptor for batched kernels. Predicate arrays are the
 * concatenation of all queries' slices, pre-rebased by the host: proot is
 * the query's root node index (-1 = no predicate) and child0/val0/children
 * entries are absolute into the concatenated arrays. Spans for the query
 * are spans[span0 .. span0+nspans); soff0 indexes its local prefix array
 * (nspans+1 entries) inside the concatenated spanoff array. */
typedef struct FvGpuQueryDesc {
    int32_t proot;
    int32_t need_eval;
    int32_t span0;
    int32_t nspans;
    int32_t soff0;
    uint32_t entry;  /* graph search entry point */
    uint64_t total;  /* brute enumeration size (= count when nspans == 0) */
} FvGpuQueryDesc;

/* 1 if a usable CUDA device exists in this process (initializes CUDA). */
int fv_gpu_available(void);

/* Upload vectors + scalar columns + posting rows + level-0 adjacency to one
 * GPU. l0 may be NULL (no graph search support). device -1 picks the device
 * with the most free memory. precision is FV_GPU_PREC_*. NULL + errbuf on
 * any CUDA failure. */
FvGpuIndex *fv_gpu_index_upload(const float *vectors, const double *cols,
                                const uint32_t *post_rows,
                                const uint32_t *l0, uint32_t l0_stride,
                                uint64_t count, int dim, int ncols, int device,
                                int precision, char *errbuf, size_t errlen);
void fv_gpu_index_free(FvGpuIndex *g);
int fv_gpu_index_device(const FvGpuIndex *g);
int fv_gpu_index_precision(const FvGpuIndex *g);

/*
 * Batched exact filtered top-topb. queries is nq*dim floats; qd one
 * descriptor per query; nodes/children/invals/spans/spanoff are the
 * concatenated predicate arrays described above. Results per query q go to
 * out[q*topb ..] ascending by (dist, node), out_cnt[q] entries;
 * truncated[q] mirrors the CPU brute cursor's flag. dedup[q] nonzero means
 * spans may repeat rows (host merge deduplicates). Returns 0 / -1+errbuf.
 */
int fv_gpu_brute_batch(FvGpuIndex *g, int nq, const float *queries,
                       const FvGpuQueryDesc *qd, const uint8_t *dedup,
                       const FvGpuPredNode *nodes, int nnodes,
                       const int32_t *children, int nchildren,
                       const double *invals, int ninvals,
                       const FvGpuSpan *spans, int nspans_total,
                       const uint64_t *spanoff, int nspanoff, uint32_t topb,
                       FvHit *out, uint32_t *out_cnt, uint8_t *truncated,
                       char *errbuf, size_t errlen);

/*
 * Batched filtered level-0 graph beam search, one thread block per query;
 * entry points in qd[q].entry come from the caller's CPU upper-level
 * descent. Traversal is unrestricted; only predicate-passing nodes enter
 * the per-query result list (capacity ef <= FV_GPU_MAX_EF). exhausted[q]
 * nonzero when re-running with a larger ef cannot produce more results.
 * Returns 0 / -1+errbuf.
 */
int fv_gpu_graph_batch(FvGpuIndex *g, int nq, const float *queries,
                       const FvGpuQueryDesc *qd, const FvGpuPredNode *nodes,
                       int nnodes, const int32_t *children, int nchildren,
                       const double *invals, int ninvals, uint32_t ef,
                       uint64_t max_visited, FvHit *out, uint32_t *out_cnt,
                       uint8_t *exhausted, char *errbuf, size_t errlen);

#ifdef __cplusplus
}
#endif
#endif /* PAVES_GPU_ENGINE_H */
