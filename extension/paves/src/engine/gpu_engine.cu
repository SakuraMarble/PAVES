/*
 * gpu_engine.cu — CUDA backend for paves: batched exact filtered
 * brute-force top-B and batched filtered level-0 graph search.
 *
 * Brute force: FV_GPU_BPQ blocks per query; one warp processes one row at a
 * time (lane 0 interprets the flattened predicate over the columnar scalar
 * zone, all lanes compute the L2 distance with float4 loads and a shuffle
 * reduction, lane 0 maintains a per-warp max-heap in global scratch). The
 * host merges per-warp heaps, mirroring the CPU BruteCursor merge exactly.
 *
 * Graph search: one thread block per query; shared-memory frontier
 * (unexpanded candidates, pop = swap-remove), result list (predicate-passing
 * nodes only, capacity ef) and open-addressing visited hash. Termination
 * mirrors the CPU HNSW cursor. Entry points come from the caller's CPU
 * upper-level descent.
 *
 * No PostgreSQL dependencies. All entry points are synchronous; any CUDA
 * error is reported via errbuf and callers fall back to the CPU. CUDA is
 * initialized lazily so it always happens after fork().
 */
#include "gpu_engine.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#define FV_GPU_THREADS 256
#define FV_GPU_BPQ 8 /* legacy brute: blocks per query */
#define FV_GPU_WPQ (FV_GPU_BPQ * FV_GPU_THREADS / 32) /* warps per query */

/* Tiled brute path: one pass over the vector array serves a whole tile of
 * queries (the legacy kernel reads the array once PER query). */
#define FV_TILE_TOPB 64      /* per-lane heap capacity; topb beyond -> legacy */
#define FV_TILE_WARPS 8      /* = FV_GPU_THREADS / 32 */
#define FV_TILE_MAX_SHMEM (96 * 1024)

#define CUDA_TRY(call)                                                     \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            snprintf(errbuf, errlen, "gpu: %s failed: %s", #call,          \
                     cudaGetErrorString(err__));                           \
            return nullptr;                                                \
        }                                                                  \
    } while (0)

#define CUDA_TRY_I(call)                                                   \
    do {                                                                   \
        cudaError_t err__ = (call);                                        \
        if (err__ != cudaSuccess) {                                        \
            snprintf(errbuf, errlen, "gpu: %s failed: %s", #call,          \
                     cudaGetErrorString(err__));                           \
            return -1;                                                     \
        }                                                                  \
    } while (0)

namespace {

/* grow-only device/pinned buffer */
struct DevBuf {
    void *p = nullptr;
    size_t cap = 0;
    bool pinned = false;
    cudaError_t ensure(size_t need, bool pin = false) {
        /* A buffer must keep its kind: freeing device memory with
         * cudaFreeHost (or the reverse) corrupts the allocator and shows up
         * later as a fault inside an unrelated cudaMemcpy. */
        if (p != nullptr && pinned != pin) {
            if (pinned) cudaFreeHost(p); else cudaFree(p);
            p = nullptr;
            cap = 0;
        }
        if (cap >= need) return cudaSuccess;
        if (p) {
            if (pinned) cudaFreeHost(p); else cudaFree(p);
            p = nullptr;
            cap = 0;
        }
        size_t newcap = need + need / 2;
        cudaError_t err = pin ? cudaMallocHost(&p, newcap) : cudaMalloc(&p, newcap);
        if (err == cudaSuccess) {
            cap = newcap;
            pinned = pin;
        }
        return err;
    }
    void free() {
        if (p) {
            if (pinned) cudaFreeHost(p); else cudaFree(p);
        }
        p = nullptr;
        cap = 0;
    }
    template <typename T> T *as() { return (T *)p; }
};

/* ---------------- phase profiling (FV_GPU_PROF=<seconds>) ----------------
 * Accumulates per-phase wall time for the two batch entry points and dumps
 * one summary line to stderr (-> PG log) every interval. Off by default;
 * when off no extra synchronization is inserted, so production timing is
 * unchanged. All entry points are already synchronous, so steady_clock
 * around synchronizing calls is exact. */
struct ProfAcc {
    uint64_t batches = 0, queries = 0;
    double h2d = 0, kern = 0, kern2 = 0, d2h = 0, host = 0;
    void reset() { *this = ProfAcc(); }
};
static ProfAcc prof_brute, prof_graph, prof_legacy;
static int prof_interval = -2; /* -2 = env unread, -1 = off, >0 = seconds */
static std::chrono::steady_clock::time_point prof_last;

static bool prof_on(void) {
    if (prof_interval == -2) {
        const char *e = getenv("FV_GPU_PROF");
        prof_interval = (e && *e) ? atoi(e) : -1;
        if (prof_interval == 0) prof_interval = 5;
        prof_last = std::chrono::steady_clock::now();
    }
    return prof_interval > 0;
}

typedef std::chrono::steady_clock::time_point ProfT;
static inline ProfT prof_now(void) { return std::chrono::steady_clock::now(); }
static inline double prof_ms(ProfT a, ProfT b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

static void prof_dump_maybe(void) {
    ProfT now = prof_now();
    if (std::chrono::duration<double>(now - prof_last).count() <
        (double)prof_interval)
        return;
    prof_last = now;
    fprintf(stderr,
            "FVPROF {\"brute\":{\"batches\":%llu,\"q\":%llu,\"h2d_ms\":%.2f,"
            "\"kern_ms\":%.2f,\"merge_ms\":%.2f,\"d2h_ms\":%.2f,"
            "\"host_ms\":%.2f},"
            "\"graph\":{\"batches\":%llu,\"q\":%llu,\"h2d_ms\":%.2f,"
            "\"kern_ms\":%.2f,\"d2h_ms\":%.2f,\"host_ms\":%.2f},"
            "\"span\":{\"batches\":%llu,\"q\":%llu,\"kern_ms\":%.2f}}\n",
            (unsigned long long)prof_brute.batches,
            (unsigned long long)prof_brute.queries, prof_brute.h2d,
            prof_brute.kern, prof_brute.kern2, prof_brute.d2h,
            prof_brute.host,
            (unsigned long long)prof_graph.batches,
            (unsigned long long)prof_graph.queries, prof_graph.h2d,
            prof_graph.kern, prof_graph.d2h, prof_graph.host,
            (unsigned long long)prof_legacy.batches,
            (unsigned long long)prof_legacy.queries, prof_legacy.kern);
    fflush(stderr);
    prof_brute.reset();
    prof_graph.reset();
    prof_legacy.reset();
}

/*
 * Guard an about-to-run host->device copy. A fault inside cuMemcpyHtoD is
 * unrecoverable (it kills the worker and, through it, the instance), and its
 * two possible causes need different fixes, so check both here: touch the
 * host source (a freed/unmapped source faults in OUR frame, where the
 * message says so) and verify the destination really is device memory of
 * the current device (a stale or foreign pointer makes the driver treat the
 * copy as host-to-host and write to an arbitrary address).
 * Returns nullptr when the pair is sound, else a description.
 */
static const char *h2d_check(const void *dst, const void *src, size_t bytes,
                             int device) {
    if (bytes == 0) return nullptr;
    if (src == nullptr) return "host source is NULL";
    if (dst == nullptr) return "device destination is NULL";
    /* fault here (not in the driver) if the source went away */
    volatile const unsigned char *s = (const unsigned char *)src;
    unsigned char probe = s[0];
    probe ^= s[bytes - 1];
    (void)probe;
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, dst);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return "destination is not a known CUDA pointer";
    }
    if (attr.type != cudaMemoryTypeDevice) return "destination is not device memory";
    if (attr.device != device) return "destination belongs to another device";
    return nullptr;
}

} // namespace

struct FvGpuIndex {
    int device = -1;
    int precision = FV_GPU_PREC_FP32;
    uint64_t count = 0;
    int dim = 0;
    int ncols = 0;
    uint32_t l0_stride = 0;

    float *d_vecs = nullptr; /* fp32 mode only */
    double *d_cols = nullptr;
    uint32_t *d_prows = nullptr;
    uint32_t *d_l0 = nullptr;

    /* SQ8 mode: packed int8 rows (dp4a scan), fp16 rows (rerank + graph),
     * per-row int norms; per-dim centering + one global scale on the host
     * for query quantization. */
    uint32_t *d_vq = nullptr;   /* count * dq4 packed int8 */
    __half *d_vh = nullptr;     /* count * dpad */
    int32_t *d_vnorm = nullptr; /* count: quantized norms */
    int dpad = 0;               /* dim padded to a multiple of 16 */
    int dq4 = 0;                /* dpad / 4 */
    std::vector<float> qmean;   /* dim */
    float qscale = 1.f, qinv = 1.f;

    /* per-call staging, grow-only */
    DevBuf d_queries, d_qd, d_nodes, d_children, d_invals, d_spans, d_spanoff;
    DevBuf d_scratch, d_counts, d_gout, d_gcnt, d_gexh, d_visited;
    DevBuf h_scratch, h_counts, h_gout, h_gcnt, h_gexh;
    /* tiled brute path */
    DevBuf d_tkeys, d_bout, d_bcnt, d_btrunc;
    DevBuf h_bout, h_bcnt, h_btrunc;
    /* sq8: quantized queries + norms, pre-rerank candidates */
    DevBuf d_qq, d_qnorm, d_btmp, d_btmpcnt;
    DevBuf h_qq, h_qnorm;
    DevBuf d_gdbg, h_gdbg; /* FV_GPU_GDBG per-query graph counters */
    bool tile_shmem_set = false;
    bool tile_warned = false;
};

/* ================= device: predicate ================= */

__device__ static bool fv_eval_dev(const FvGpuPredNode *nodes,
                                   const int32_t *children,
                                   const double *invals, const double *cols,
                                   uint64_t count, int32_t ni, uint32_t row) {
    const FvGpuPredNode n = nodes[ni];
    switch (n.kind) {
        case FV_NODE_CMP: {
            double v = cols[(uint64_t)n.col * count + row];
            switch (n.op) {
                case FV_CMP_LT: return v < n.value;
                case FV_CMP_LE: return v <= n.value;
                case FV_CMP_EQ: return v == n.value;
                case FV_CMP_GE: return v >= n.value;
                case FV_CMP_GT: return v > n.value;
                case FV_CMP_NE: return v == v && v != n.value;
            }
            return false;
        }
        case FV_NODE_IN: {
            double v = cols[(uint64_t)n.col * count + row];
            for (int i = 0; i < n.nvalues; i++)
                if (v == invals[n.val0 + i]) return true;
            return false;
        }
        case FV_NODE_AND:
            for (int i = 0; i < n.nchildren; i++)
                if (!fv_eval_dev(nodes, children, invals, cols, count,
                                 children[n.child0 + i], row))
                    return false;
            return true;
        case FV_NODE_OR:
            for (int i = 0; i < n.nchildren; i++)
                if (fv_eval_dev(nodes, children, invals, cols, count,
                                children[n.child0 + i], row))
                    return true;
            return false;
        case FV_NODE_NOT:
            return !fv_eval_dev(nodes, children, invals, cols, count,
                                children[n.child0], row);
    }
    return false;
}

__device__ static float fv_l2_warp(const float *__restrict__ v,
                                   const float *__restrict__ sq, int dim,
                                   int lane) {
    float acc = 0.f;
    if ((dim & 3) == 0) {
        const float4 *v4 = (const float4 *)v;
        const float4 *q4 = (const float4 *)sq;
        for (int j = lane; j < (dim >> 2); j += 32) {
            float4 a = v4[j], b = q4[j];
            float dx = a.x - b.x, dy = a.y - b.y;
            float dz = a.z - b.z, dw = a.w - b.w;
            acc += dx * dx + dy * dy + dz * dz + dw * dw;
        }
    } else {
        for (int j = lane; j < dim; j += 32) {
            float d = v[j] - sq[j];
            acc += d * d;
        }
    }
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc; /* valid in lane 0 */
}

/* fp16 rows (padded to a multiple of 8), fp32 query, fp32 accumulation */
__device__ static float fv_l2_warp(const __half *__restrict__ v,
                                   const float *__restrict__ sq, int dim,
                                   int lane) {
    float acc = 0.f;
    const __half2 *v2 = (const __half2 *)v;
    for (int j = lane; j < (dim >> 1); j += 32) {
        float2 a = __half22float2(v2[j]);
        float dx = a.x - sq[2 * j], dy = a.y - sq[2 * j + 1];
        acc += dx * dx + dy * dy;
    }
    if ((dim & 1) && lane == 0) {
        float d = __half2float(v[dim - 1]) - sq[dim - 1];
        acc += d * d;
    }
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc; /* valid in lane 0 */
}

/* scalar loads for one-off distances (entry-point seeding) */
__device__ static inline float fv_vec_at(const float *v, int j) { return v[j]; }
__device__ static inline float fv_vec_at(const __half *v, int j) {
    return __half2float(v[j]);
}

/* ================= brute kernel ================= */

/* max-heap by dist on a per-warp global-memory region, lane 0 only */
__device__ static void heap_sift_down(FvHit *h, uint32_t n) {
    uint32_t i = 0;
    for (;;) {
        uint32_t l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l].dist > h[m].dist) m = l;
        if (r < n && h[r].dist > h[m].dist) m = r;
        if (m == i) break;
        FvHit t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

template <typename VT>
__global__ static void fv_brute_kernel(
    const VT *__restrict__ vecs, int vstride, const double *__restrict__ cols,
    const uint32_t *__restrict__ prows, uint64_t count, int dim,
    const float *__restrict__ queries, const FvGpuQueryDesc *__restrict__ qds,
    const FvGpuPredNode *__restrict__ nodes,
    const int32_t *__restrict__ children, const double *__restrict__ invals,
    const FvGpuSpan *__restrict__ spans, const uint64_t *__restrict__ spanoff,
    uint32_t topb, FvHit *scratch, uint32_t *counts) {
    extern __shared__ float sq[];
    const int qidx = blockIdx.x / FV_GPU_BPQ;
    const FvGpuQueryDesc qd = qds[qidx];
    const float *qv = queries + (uint64_t)qidx * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sq[i] = qv[i];
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int wq = (int)((((blockIdx.x % FV_GPU_BPQ) * blockDim.x) +
                          threadIdx.x) >> 5);
    FvHit *heap = scratch + ((uint64_t)qidx * FV_GPU_WPQ + wq) * topb;
    uint32_t hn = 0;

    const uint64_t *soff = spanoff + qd.soff0;
    for (uint64_t e = wq; e < qd.total; e += FV_GPU_WPQ) {
        uint32_t row;
        if (qd.nspans == 0) {
            row = (uint32_t)e;
        } else {
            int lo = 0, hi = qd.nspans - 1;
            while (lo < hi) {
                int mid = (lo + hi + 1) >> 1;
                if (soff[mid] <= e) lo = mid; else hi = mid - 1;
            }
            const FvGpuSpan s = spans[qd.span0 + lo];
            row = prows[(uint64_t)s.col * count +
                        (s.lo + (uint32_t)(e - soff[lo]))];
        }

        int pass = 1;
        if (qd.need_eval) {
            if (lane == 0)
                pass = fv_eval_dev(nodes, children, invals, cols, count,
                                   qd.proot, row);
            pass = __shfl_sync(0xffffffffu, pass, 0);
        }
        if (!pass) continue;

        float acc = fv_l2_warp(vecs + (uint64_t)row * vstride, sq, dim, lane);

        if (lane == 0) {
            if (hn < topb) {
                uint32_t i = hn++;
                heap[i].node = row;
                heap[i].dist = acc;
                while (i) { /* sift up */
                    uint32_t p = (i - 1) >> 1;
                    if (heap[p].dist >= heap[i].dist) break;
                    FvHit t = heap[p];
                    heap[p] = heap[i];
                    heap[i] = t;
                    i = p;
                }
            } else if (acc < heap[0].dist) {
                heap[0].node = row;
                heap[0].dist = acc;
                heap_sift_down(heap, hn);
            }
        }
    }
    if (lane == 0) counts[(uint64_t)qidx * FV_GPU_WPQ + wq] = hn;
}

/* ================= tiled brute kernel ================= */

/*
 * Query-tiled exact scan. Each block owns a contiguous row range and a tile
 * of FV_TILE_QTILE queries (transposed into shared memory); each warp
 * streams its own rows through shared memory and every lane evaluates two
 * queries against the broadcast row. The vector array is therefore read
 * ONCE per query tile instead of once per query — the legacy kernel's
 * 64-query batch reads 64x the data and is bandwidth-bound per query.
 *
 * Spans are not used: the scan is always full and the predicate (when
 * present) is evaluated lazily — only for rows that would enter the
 * candidate heap (below the heap-full threshold every row must be checked,
 * afterwards only rows beating the current worst).
 *
 * Candidates are kept per lane as packed keys (dist_bits<<32 | row): key
 * order == (dist, node) lexicographic order, so heap eviction and the final
 * sort agree and the selected top-B is canonical — identical across
 * recomputes and identical to the CPU cursor's sort-by-(dist,node).
 *
 * Per (query, rowblock, warp) partial heaps go to global scratch (padded
 * with UINT64_MAX); fv_brute_merge_kernel reduces them to the final per-
 * query top-B on the GPU, so only nq*topb hits cross PCIe.
 */
__device__ static inline void tile_key_push(uint64_t *h, int *n,
                                            uint64_t key) {
    int i = (*n)++;
    h[i] = key;
    while (i) {
        int p = (i - 1) >> 1;
        if (h[p] >= h[i]) break;
        uint64_t t = h[p];
        h[p] = h[i];
        h[i] = t;
        i = p;
    }
}

__device__ static inline void tile_key_replace(uint64_t *h, int n,
                                               uint64_t key) {
    h[0] = key;
    int i = 0;
    for (;;) {
        int l = 2 * i + 1, r = l + 1, m = i;
        if (l < n && h[l] > h[m]) m = l;
        if (r < n && h[r] > h[m]) m = r;
        if (m == i) break;
        uint64_t t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

template <int QPL, int LPQ, int RT>
__global__ static void fv_brute_tile_kernel(
    const float *__restrict__ vecs, const double *__restrict__ cols,
    uint64_t count, int dim, int nq, const float *__restrict__ queries,
    const FvGpuQueryDesc *__restrict__ qds,
    const FvGpuPredNode *__restrict__ nodes,
    const int32_t *__restrict__ children, const double *__restrict__ invals,
    uint32_t topb, int rowblocks, uint64_t *__restrict__ tkeys) {
    /* QPL queries per lane, each lane covers 1/LPQ of the dimensions,
     * RT rows per warp iteration. QT = queries per block tile. */
    constexpr int QT = 32 * QPL / LPQ;
    extern __shared__ float smem[];
    float *qT = smem;                        /* [dim][QT] transposed */
    float *rowbuf = smem + (size_t)dim * QT; /* [WARPS][RT][dim] */
    __shared__ int32_t sproot[QT];           /* INT32_MIN = no query */

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int qbase = blockIdx.y * QT;

    for (int idx = tid; idx < dim * QT; idx += blockDim.x) {
        int j = idx / QT;
        int l = idx % QT;
        qT[idx] = (qbase + l < nq) ? queries[(uint64_t)(qbase + l) * dim + j]
                                   : 0.f;
    }
    if (tid < QT)
        sproot[tid] = (qbase + tid < nq) ? qds[qbase + tid].proot : INT32_MIN;
    __syncthreads();

    const uint64_t chunk = (count + rowblocks - 1) / rowblocks;
    const uint64_t r0 = (uint64_t)blockIdx.x * chunk;
    const uint64_t r1 = min(count, r0 + chunk);
    const int cap = (int)topb;

    const int qloc = lane % QT; /* query slot within the tile */
    const int part = lane / QT; /* dimension partition (0..LPQ-1) */
    const int jlo = part * (dim / LPQ);
    const int jhi = jlo + dim / LPQ;

    /* Per-lane heaps live directly in their global scratch slot (L2-cached;
     * ops are rare once warm; local arrays here would blow the 1KB device
     * stack together with the predicate evaluator's recursion frames).
     * Only partition-0 lanes own heaps. */
    uint64_t *hp[QPL];
    int hn[QPL];
    uint64_t worst[QPL];
    int32_t pr[QPL];
#pragma unroll
    for (int s = 0; s < QPL; s++) {
        int gq = qbase + qloc + s * 32;
        hp[s] = tkeys + (((uint64_t)gq * rowblocks + blockIdx.x) *
                             FV_TILE_WARPS + warp) * FV_TILE_TOPB;
        hn[s] = 0;
        worst[s] = UINT64_MAX;
        pr[s] = (part == 0) ? sproot[qloc + s * 32] : INT32_MIN;
    }
    float *rb = rowbuf + warp * RT * dim;

    for (uint64_t rr = r0 + (uint64_t)warp * RT; rr < r1;
         rr += FV_TILE_WARPS * RT) {
        const int nrow = (int)min((uint64_t)RT, r1 - rr);
        for (int t = 0; t < nrow; t++) {
            const float4 *src = (const float4 *)(vecs + (rr + t) * dim);
            float4 *dst = (float4 *)(rb + t * dim);
            for (int j4 = lane; j4 < (dim >> 2); j4 += 32) dst[j4] = src[j4];
        }
        __syncwarp();

        float acc[QPL][RT];
#pragma unroll
        for (int s = 0; s < QPL; s++)
#pragma unroll
            for (int t = 0; t < RT; t++) acc[s][t] = 0.f;
#pragma unroll 4
        for (int j = jlo; j < jhi; j++) {
            float qv[QPL];
#pragma unroll
            for (int s = 0; s < QPL; s++) qv[s] = qT[j * QT + qloc + s * 32];
#pragma unroll
            for (int t = 0; t < RT; t++) {
                float x = rb[t * dim + j];
#pragma unroll
                for (int s = 0; s < QPL; s++) {
                    float d = x - qv[s];
                    acc[s][t] += d * d;
                }
            }
        }
        /* combine dimension partitions across lanes (butterfly) */
        if (LPQ > 1) {
#pragma unroll
            for (int o = 16; o >= QT; o >>= 1)
#pragma unroll
                for (int s = 0; s < QPL; s++)
#pragma unroll
                    for (int t = 0; t < RT; t++)
                        acc[s][t] +=
                            __shfl_xor_sync(0xffffffffu, acc[s][t], o);
        }
        __syncwarp(); /* rb reused next iteration */

        if (part == 0) {
            for (int t = 0; t < nrow; t++) {
                const uint32_t r = (uint32_t)(rr + t);
#pragma unroll
                for (int s = 0; s < QPL; s++) {
                    if (pr[s] == INT32_MIN) continue;
                    uint64_t key =
                        ((uint64_t)__float_as_uint(acc[s][t]) << 32) | r;
                    if (hn[s] < cap) {
                        if (pr[s] < 0 ||
                            fv_eval_dev(nodes, children, invals, cols, count,
                                        pr[s], r)) {
                            tile_key_push(hp[s], &hn[s], key);
                            if (hn[s] == cap) worst[s] = hp[s][0];
                        }
                    } else if (key < worst[s]) {
                        if (pr[s] < 0 ||
                            fv_eval_dev(nodes, children, invals, cols, count,
                                        pr[s], r)) {
                            tile_key_replace(hp[s], hn[s], key);
                            worst[s] = hp[s][0];
                        }
                    }
                }
            }
        }
    }

    /* pad the in-place heaps to FV_TILE_TOPB with UINT64_MAX */
#pragma unroll
    for (int s = 0; s < QPL; s++)
        if (pr[s] != INT32_MIN)
            for (int i = hn[s]; i < FV_TILE_TOPB; i++) hp[s][i] = UINT64_MAX;
}

/* Variant table: pick the widest query tile whose shared-memory footprint
 * (query tile + row buffers) fits. -1 = no fast path for this dimension
 * (the legacy per-query kernel serves it). */
struct TileVariant {
    int qpl, lpq, rt, qt;
    int dim_mult; /* dim must be a multiple of this */
    size_t shm_per_dim;
};
static const TileVariant tile_variants[4] = {
    {2, 1, 4, 64, 4, (64 + 8 * 4) * 4}, /* dim <= 256 */
    {1, 1, 2, 32, 4, (32 + 8 * 2) * 4}, /* dim <= 512 */
    {1, 2, 1, 16, 8, (16 + 8 * 1) * 4}, /* dim <= 1024 */
    {1, 4, 1, 8, 16, (8 + 8 * 1) * 4},  /* dim <= 1536 */
};
static int tile_pick(int dim) {
    for (int v = 0; v < 4; v++)
        if (dim % tile_variants[v].dim_mult == 0 &&
            (size_t)dim * tile_variants[v].shm_per_dim <= FV_TILE_MAX_SHMEM)
            return v;
    return -1;
}

/* Row-dimension grid: keep per-lane streams long enough that heap fill
 * (~topb pushes per lane-query) stays amortized. 1M rows -> 128 blocks
 * (measured optimum), 10M+ rows -> 256. */
static int tile_rowblocks(uint64_t count) {
    int rb = 64;
    while (rb < 256 && (uint64_t)rb * 8192 < count) rb <<= 1;
    return rb;
}

/* ================= SQ8 quantization ================= */

/*
 * Scalar quantization: x_q[d] = clamp(round((x[d] - mean[d]) / s), -127, 127)
 * with per-dimension centering and ONE global scale s. Centering shifts both
 * sides of every distance equally, so it does not distort L2 at all; a
 * single global scale keeps quantized L2 proportional to true L2
 * (dist ~= s^2 * dist_q), which is what lets the scan rank candidates with
 * exact integer arithmetic:
 *     dist_q(x, y) = |x_q|^2 + |y_q|^2 - 2 x_q.y_q      (dp4a dot products)
 * The top candidates are then re-ranked against an fp16 copy of the rows in
 * fp32 arithmetic, which restores recall lost to quantization noise.
 */

__device__ static void atomic_min_f(float *addr, float v) {
    int old = __float_as_int(*addr);
    while (__int_as_float(old) > v) {
        int assumed = old;
        old = atomicCAS((int *)addr, assumed, __float_as_int(v));
        if (old == assumed) break;
    }
}

__device__ static void atomic_max_f(float *addr, float v) {
    int old = __float_as_int(*addr);
    while (__int_as_float(old) < v) {
        int assumed = old;
        old = atomicCAS((int *)addr, assumed, __float_as_int(v));
        if (old == assumed) break;
    }
}

#define FV_STATS_DPT 8 /* dims per thread (dim <= 2048 / 256 threads) */

__global__ static void fv_stats_kernel(const float *__restrict__ rows,
                                       int nrows, int dim,
                                       double *__restrict__ sums,
                                       float *__restrict__ mins,
                                       float *__restrict__ maxs) {
    const int tid = threadIdx.x;
    double s[FV_STATS_DPT];
    float mn[FV_STATS_DPT], mx[FV_STATS_DPT];
#pragma unroll
    for (int k = 0; k < FV_STATS_DPT; k++) {
        s[k] = 0.0;
        mn[k] = 3.4e38f;
        mx[k] = -3.4e38f;
    }
    for (int r = blockIdx.x; r < nrows; r += gridDim.x) {
        const float *row = rows + (uint64_t)r * dim;
        for (int k = 0, d = tid; d < dim; k++, d += blockDim.x) {
            float x = row[d];
            s[k] += x;
            if (x < mn[k]) mn[k] = x;
            if (x > mx[k]) mx[k] = x;
        }
    }
    for (int k = 0, d = tid; d < dim; k++, d += blockDim.x) {
        atomicAdd(&sums[d], s[k]);
        atomic_min_f(&mins[d], mn[k]);
        atomic_max_f(&maxs[d], mx[k]);
    }
}

/* one warp per row: pack int8 (padded with zeros), write fp16 copy + norm */
__global__ static void fv_quant_kernel(const float *__restrict__ rows,
                                       uint64_t row0, int nrows, int dim,
                                       int dpad, int dq4,
                                       const float *__restrict__ mean,
                                       float qinv, uint32_t *__restrict__ vq,
                                       __half *__restrict__ vh,
                                       int32_t *__restrict__ vnorm) {
    const int lane = threadIdx.x & 31;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp >= nrows) return;
    const float *src = rows + (uint64_t)warp * dim;
    const uint64_t out = row0 + warp;
    int norm = 0;
    for (int j4 = lane; j4 < dq4; j4 += 32) {
        uint32_t packed = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
            int d = j4 * 4 + b;
            int xi = 0;
            __half h = __float2half(0.f);
            if (d < dim) {
                float x = src[d];
                float c = (x - mean[d]) * qinv;
                xi = __float2int_rn(c);
                xi = max(-127, min(127, xi));
                h = __float2half(x);
            }
            packed |= (uint32_t)(uint8_t)(int8_t)xi << (8 * b);
            vh[out * dpad + d] = h;
            norm += xi * xi;
        }
        vq[out * dq4 + j4] = packed;
    }
    for (int off = 16; off; off >>= 1)
        norm += __shfl_down_sync(0xffffffffu, norm, off);
    if (lane == 0) vnorm[out] = norm;
}

/* ================= SQ8 tiled brute kernel ================= */

/*
 * Same query-tiled full-scan structure as fv_brute_tile_kernel, but rows and
 * queries are packed int8 and a lane computes dot products with dp4a (4
 * multiply-adds per instruction on 4x less data). The distance is exact
 * integer arithmetic in the quantized domain:
 *     dist_q = |x_q|^2 + |y_q|^2 - 2 x_q.y_q
 * so candidate keys pack the integer distance instead of float bits; heap
 * logic, scratch layout and the merge kernel are shared with the fp32 path.
 * QTL = 32/LPQ lanes cover one dimension partition; a lane owns QPL query
 * slots (qloc + s*QTL); QT = QTL*QPL queries per block tile.
 */
template <int QPL, int LPQ, int RT>
__global__ static void fv_brute_tile8_kernel(
    const uint32_t *__restrict__ vq, const int32_t *__restrict__ vnorm,
    const double *__restrict__ cols, uint64_t count, int dq4, int nq,
    const uint32_t *__restrict__ qq, const int32_t *__restrict__ qnorm,
    const FvGpuQueryDesc *__restrict__ qds,
    const FvGpuPredNode *__restrict__ nodes,
    const int32_t *__restrict__ children, const double *__restrict__ invals,
    uint32_t topb, int rowblocks, uint64_t *__restrict__ tkeys) {
    constexpr int QTL = 32 / LPQ;
    constexpr int QT = QTL * QPL;
    extern __shared__ uint32_t smem8[];
    uint32_t *qT = smem8;                          /* [dq4][QT] transposed */
    uint32_t *rowbuf = smem8 + (size_t)dq4 * QT;   /* [WARPS][RT][dq4] */
    __shared__ int32_t sqn[QT];
    __shared__ int32_t sproot[QT];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int qbase = blockIdx.y * QT;

    for (int idx = tid; idx < dq4 * QT; idx += blockDim.x) {
        int j4 = idx / QT;
        int l = idx % QT;
        qT[idx] = (qbase + l < nq) ? qq[(uint64_t)(qbase + l) * dq4 + j4] : 0;
    }
    if (tid < QT) {
        sproot[tid] = (qbase + tid < nq) ? qds[qbase + tid].proot : INT32_MIN;
        sqn[tid] = (qbase + tid < nq) ? qnorm[qbase + tid] : 0;
    }
    __syncthreads();

    const uint64_t chunk = (count + rowblocks - 1) / rowblocks;
    const uint64_t r0 = (uint64_t)blockIdx.x * chunk;
    const uint64_t r1 = min(count, r0 + chunk);
    const int cap = (int)topb;

    const int qloc = lane % QTL;
    const int part = lane / QTL;
    const int jlo4 = part * (dq4 / LPQ);
    const int jhi4 = jlo4 + dq4 / LPQ;

    uint64_t *hp[QPL];
    int hn[QPL];
    uint64_t worst[QPL];
    int32_t pr[QPL];
#pragma unroll
    for (int s = 0; s < QPL; s++) {
        int gq = qbase + qloc + s * QTL;
        hp[s] = tkeys + (((uint64_t)gq * rowblocks + blockIdx.x) *
                             FV_TILE_WARPS + warp) * FV_TILE_TOPB;
        hn[s] = 0;
        worst[s] = UINT64_MAX;
        pr[s] = (part == 0) ? sproot[qloc + s * QTL] : INT32_MIN;
    }
    uint32_t *rb = rowbuf + (size_t)warp * RT * dq4;

    for (uint64_t rr = r0 + (uint64_t)warp * RT; rr < r1;
         rr += FV_TILE_WARPS * RT) {
        const int nrow = (int)min((uint64_t)RT, r1 - rr);
        for (int t = 0; t < nrow; t++) {
            const uint4 *src = (const uint4 *)(vq + (rr + t) * dq4);
            uint4 *dst = (uint4 *)(rb + (size_t)t * dq4);
            for (int j16 = lane; j16 < (dq4 >> 2); j16 += 32)
                dst[j16] = src[j16];
        }
        __syncwarp();

        int acc[QPL][RT];
#pragma unroll
        for (int s = 0; s < QPL; s++)
#pragma unroll
            for (int t = 0; t < RT; t++) acc[s][t] = 0;
#pragma unroll 4
        for (int j4 = jlo4; j4 < jhi4; j4++) {
            uint32_t qv[QPL];
#pragma unroll
            for (int s = 0; s < QPL; s++)
                qv[s] = qT[(size_t)j4 * QT + qloc + s * QTL];
#pragma unroll
            for (int t = 0; t < RT; t++) {
                uint32_t x = rb[(size_t)t * dq4 + j4];
#pragma unroll
                for (int s = 0; s < QPL; s++)
                    acc[s][t] = __dp4a((int)x, (int)qv[s], acc[s][t]);
            }
        }
        if (LPQ > 1) {
#pragma unroll
            for (int o = 16; o >= QTL; o >>= 1)
#pragma unroll
                for (int s = 0; s < QPL; s++)
#pragma unroll
                    for (int t = 0; t < RT; t++)
                        acc[s][t] +=
                            __shfl_xor_sync(0xffffffffu, acc[s][t], o);
        }
        __syncwarp(); /* rb reused next iteration */

        if (part == 0) {
            for (int t = 0; t < nrow; t++) {
                const uint32_t r = (uint32_t)(rr + t);
                const int vn = vnorm[rr + t];
#pragma unroll
                for (int s = 0; s < QPL; s++) {
                    if (pr[s] == INT32_MIN) continue;
                    uint32_t dq = (uint32_t)(vn + sqn[qloc + s * QTL] -
                                             2 * acc[s][t]);
                    uint64_t key = ((uint64_t)dq << 32) | r;
                    if (hn[s] < cap) {
                        if (pr[s] < 0 ||
                            fv_eval_dev(nodes, children, invals, cols, count,
                                        pr[s], r)) {
                            tile_key_push(hp[s], &hn[s], key);
                            if (hn[s] == cap) worst[s] = hp[s][0];
                        }
                    } else if (key < worst[s]) {
                        if (pr[s] < 0 ||
                            fv_eval_dev(nodes, children, invals, cols, count,
                                        pr[s], r)) {
                            tile_key_replace(hp[s], hn[s], key);
                            worst[s] = hp[s][0];
                        }
                    }
                }
            }
        }
    }

#pragma unroll
    for (int s = 0; s < QPL; s++)
        if (pr[s] != INT32_MIN)
            for (int i = hn[s]; i < FV_TILE_TOPB; i++) hp[s][i] = UINT64_MAX;
}

/* SQ8 variant table: zero-padding lifts the divisibility constraint, so
 * every dim <= 2048 takes the fast path. shm = (QT + 8*RT) bytes/padded
 * dim. */
struct Tile8Variant {
    int qpl, lpq, rt, qt;
    int max_dim;
};
static const Tile8Variant tile8_variants[7] = {
    {2, 1, 8, 64, 256},  /* 32 KB at dim 256 */
    {2, 1, 4, 64, 512},  /* 48 KB at dim 512 */
    {2, 2, 2, 32, 1024}, /* 48 KB at dim 1024: 2 blocks/SM beats the
                          * wider 80 KB tile (measured on wiki1024) */
    {4, 4, 1, 32, 2048}, /* 80 KB at dim 2048 */
    /* experimental (FV_TILE8_V override; smaller tiles, 2 blocks/SM) */
    {4, 2, 2, 64, 1024}, /* 80 KB at dim 1024 */
    {4, 4, 2, 32, 1024}, /* 48 KB at dim 1024 */
    {2, 2, 4, 32, 1024}, /* 64 KB at dim 1024 */
};
static int tile8_pick(int dim) {
    const char *e = getenv("FV_TILE8_V");
    if (e && *e) {
        int v = atoi(e);
        if (v >= 0 && v < 7 && dim <= tile8_variants[v].max_dim) return v;
    }
    for (int v = 0; v < 4; v++)
        if (dim <= tile8_variants[v].max_dim) return v;
    return -1;
}

/* ================= SQ8 rerank kernel ================= */

#define FV_RERANK_PAD 128 /* padded candidate count (>= topb_r), power of 2 */

/*
 * One block per query: recompute exact-fp16 distances for the topb_r
 * approximate candidates (one warp each), bitonic-sort by (dist, node) and
 * emit the final ascending top-topb. Overwrites out_cnt/truncated with the
 * final counts (truncated = topb candidates survived, more rows may match).
 */
__global__ static void fv_rerank_kernel(
    const __half *__restrict__ vh, int dpad, int dim,
    const float *__restrict__ queries, const FvHit *__restrict__ cand,
    const uint32_t *__restrict__ cand_cnt, int topb_r, uint32_t topb,
    FvHit *__restrict__ out, uint32_t *__restrict__ out_cnt,
    uint8_t *__restrict__ truncated) {
    extern __shared__ float sqf[]; /* dim floats */
    __shared__ uint64_t skeys[FV_RERANK_PAD];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const int cnt = (int)cand_cnt[q];

    for (int j = tid; j < dim; j += blockDim.x)
        sqf[j] = queries[(uint64_t)q * dim + j];
    for (int i = tid; i < FV_RERANK_PAD; i += blockDim.x)
        skeys[i] = UINT64_MAX;
    __syncthreads();

    for (int i = warp; i < cnt; i += nwarp) {
        const uint32_t node =
            (uint32_t)(cand[(uint64_t)q * topb_r + i].node & 0xFFFFFFFFu);
        float d = fv_l2_warp(vh + (uint64_t)node * dpad, sqf, dim, lane);
        if (lane == 0)
            skeys[i] = ((uint64_t)__float_as_uint(d) << 32) | node;
    }
    __syncthreads();

    for (int k = 2; k <= FV_RERANK_PAD; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < FV_RERANK_PAD; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = (i & k) == 0;
                    uint64_t a = skeys[i], b = skeys[ixj];
                    if ((a > b) == up) {
                        skeys[i] = b;
                        skeys[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    const int emit = min(cnt, (int)topb);
    if (tid < emit) {
        out[(uint64_t)q * topb + tid].node =
            (uint32_t)(skeys[tid] & 0xFFFFFFFFu);
        out[(uint64_t)q * topb + tid].dist =
            __uint_as_float((uint32_t)(skeys[tid] >> 32));
    }
    if (tid == 0) {
        out_cnt[q] = (uint32_t)emit;
        truncated[q] = (cnt >= (int)topb) ? 1 : 0;
    }
}

/*
 * Reduce one query's partial heaps (ROWBLOCKS * WARPS * TOPB padded keys)
 * to the final ascending top-topb. Phase 1: 64 threads keep a local top-64
 * over strided slices (any global top-64 element is within its slice's
 * top-64). Phase 2: bitonic sort of the 4096 survivors in shared memory.
 * Emits FvHit + count + truncated (count == topb means more rows may match:
 * a dropped passing row implies some lane heap filled its 64 >= topb slots,
 * which forces count == topb).
 */
__global__ static void fv_brute_merge_kernel(
    const uint64_t *__restrict__ tkeys, int nq, uint32_t topb, int rowblocks,
    FvHit *out, uint32_t *out_cnt, uint8_t *truncated) {
    __shared__ uint64_t sm[64 * FV_TILE_TOPB];
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    const uint64_t total =
        (uint64_t)rowblocks * FV_TILE_WARPS * FV_TILE_TOPB;
    const uint64_t *base = tkeys + (uint64_t)q * total;

    if (tid < 64) {
        uint64_t lh[FV_TILE_TOPB];
        int ln = 0;
        /* w0 mirrors lh[0] in a register: the local heap lives in local
         * memory (L2-backed), so gating on the register keeps the common
         * not-competitive case free of local-memory latency */
        uint64_t w0 = UINT64_MAX;
        for (uint64_t i = tid; i < total; i += 64) {
            uint64_t k = base[i];
            if (k >= w0) continue;
            if (ln < FV_TILE_TOPB) {
                tile_key_push(lh, &ln, k);
                if (ln == FV_TILE_TOPB) w0 = lh[0];
            } else {
                tile_key_replace(lh, ln, k);
                w0 = lh[0];
            }
        }
        for (int i = 0; i < ln; i++) sm[tid * FV_TILE_TOPB + i] = lh[i];
        for (int i = ln; i < FV_TILE_TOPB; i++)
            sm[tid * FV_TILE_TOPB + i] = UINT64_MAX;
    }
    __syncthreads();

    for (int k = 2; k <= 64 * FV_TILE_TOPB; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < 64 * FV_TILE_TOPB; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = (i & k) == 0;
                    uint64_t a = sm[i], b = sm[ixj];
                    if ((a > b) == up) {
                        sm[i] = b;
                        sm[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    if (tid < (int)topb && sm[tid] != UINT64_MAX) {
        out[(uint64_t)q * topb + tid].node = (uint32_t)(sm[tid] & 0xFFFFFFFFu);
        out[(uint64_t)q * topb + tid].dist =
            __uint_as_float((uint32_t)(sm[tid] >> 32));
    }
    if (tid == 0) {
        uint32_t cnt = 0;
        while (cnt < topb && sm[cnt] != UINT64_MAX) cnt++;
        out_cnt[q] = cnt;
        truncated[q] = (cnt == topb) ? 1 : 0;
    }
}

/* ================= graph kernel ================= */

#define FV_FRONTIER_CAP 1024
#define FV_GRAPH_THREADS 512
#define FV_GRAPH_W 8    /* frontier nodes expanded per iteration */
#define FV_GRAPH_NB 256 /* neighbor slot buffer (>= W * max degree) */

/* Per-query visited bitmap in global memory (exact, like the CPU cursor's
 * epoch array). A shared-memory hash would saturate: low-selectivity
 * predicates legitimately visit tens of thousands of nodes, and once the
 * hash fills every probe reports "new", degenerating the traversal into
 * revisit loops until max_visited. Returns true when node was unvisited. */
__device__ static bool visit_test_set(uint32_t *vb, uint32_t node) {
    uint32_t mask = 1u << (node & 31);
    uint32_t old = atomicOr(&vb[node >> 5], mask);
    return (old & mask) == 0;
}

/* int8 dot product of one row against the packed query in shared memory */
__device__ static int fv_dot8_warp(const uint32_t *__restrict__ v,
                                   const uint32_t *__restrict__ q8, int dq4,
                                   int lane) {
    int acc = 0;
    for (int j = lane; j < dq4; j += 32)
        acc = __dp4a((int)v[j], (int)q8[j], acc);
    for (int off = 16; off; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    return acc; /* valid in lane 0 */
}

/* distance-domain helpers: DT = float (fp32/fp16 rows, exact-ish) or
 * uint32_t (sq8: exact integer distance in the quantized domain) */
__device__ static inline uint64_t fv_gkey(float d, uint32_t i) {
    return ((uint64_t)__float_as_uint(d) << 32) | i;
}
__device__ static inline uint64_t fv_gkey(uint32_t d, uint32_t i) {
    return ((uint64_t)d << 32) | i;
}
template <typename DT> __device__ static inline DT fv_gmax();
template <> __device__ inline float fv_gmax<float>() { return 3.4e38f; }
template <> __device__ inline uint32_t fv_gmax<uint32_t>() {
    return 0xFFFFFFFFu;
}

/*
 * Filtered level-0 beam search, one block per query. DT=uint32_t runs the
 * traversal on int8 rows (dp4a, 4x less bandwidth per neighbor fetch than
 * fp32) and re-ranks the final result list against the fp16 rows in fp32
 * arithmetic before emitting, so quantization noise does not reach the
 * consumer's top-k. DT=float traverses on VT rows directly (fp32 mode).
 */
template <typename VT, typename DT>
__global__ static void fv_graph_kernel(
    const VT *__restrict__ vecs, int vstride,
    const uint32_t *__restrict__ vq, const int32_t *__restrict__ vnorm,
    int dq4, const double *__restrict__ cols,
    const uint32_t *__restrict__ l0, uint32_t l0_stride, uint64_t count,
    int dim, const float *__restrict__ queries,
    const uint32_t *__restrict__ qq, const int32_t *__restrict__ qnorm,
    const FvGpuQueryDesc *__restrict__ qds,
    const FvGpuPredNode *__restrict__ nodes,
    const int32_t *__restrict__ children, const double *__restrict__ invals,
    uint32_t ef, uint64_t max_visited, int wsel, int deg_clamp,
    uint32_t *visited_bits, uint64_t vb_words, FvHit *out, uint32_t *out_cnt,
    uint8_t *out_exh, uint32_t *__restrict__ dbg) {
    constexpr bool SQ8 = sizeof(DT) == 4 && (DT)0.5f == (DT)0;
    extern __shared__ float sq[];      /* fp32 query [dim] (+ q8 when SQ8) */
    uint32_t *q8 = (uint32_t *)(sq + dim);
    __shared__ DT fr_d[FV_FRONTIER_CAP];
    __shared__ uint32_t fr_n[FV_FRONTIER_CAP];
    __shared__ DT re_d[FV_GPU_MAX_EF];
    __shared__ uint32_t re_n[FV_GPU_MAX_EF];
    __shared__ uint32_t neigh[FV_GRAPH_NB];
    __shared__ DT neigh_d[FV_GRAPH_NB];
    __shared__ int32_t neigh_new[FV_GRAPH_NB];
    __shared__ int32_t neigh_pass[FV_GRAPH_NB];
    __shared__ int fr_cnt, re_cnt, nneigh, stop, exh, nsel;
    __shared__ int sel_idx[FV_GRAPH_W];
    __shared__ uint32_t sel_node[FV_GRAPH_W];
    __shared__ int sel_base[FV_GRAPH_W + 1];
    __shared__ DT worst_res_sh;
    __shared__ DT fr_worst_val;
    __shared__ int fr_worst_idx;
    __shared__ uint64_t visited;
    __shared__ int pass_list[FV_GRAPH_NB]; /* slots that passed the pred */
    __shared__ int npass, nnew, novf;
    __shared__ int ovf_list[FV_GRAPH_NB]; /* frontier-overflow slots */

    const int q = blockIdx.x;
    const FvGpuQueryDesc qd = qds[q];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (int)(blockDim.x >> 5);
    const float *qv = queries + (uint64_t)q * dim;
    uint32_t *vb = visited_bits + (uint64_t)q * vb_words;
    const int32_t qn = SQ8 ? qnorm[q] : 0;

    for (int i = tid; i < dim; i += blockDim.x) sq[i] = qv[i];
    if (SQ8)
        for (int i = tid; i < dq4; i += blockDim.x)
            q8[i] = qq[(uint64_t)q * dq4 + i];
    __shared__ int dbg_evals;
    int dbg_iters = 0;
    if (tid == 0) {
        fr_cnt = 0;
        re_cnt = 0;
        stop = 0;
        exh = 0;
        visited = 0;
        dbg_evals = 0;
        worst_res_sh = fv_gmax<DT>();
    }
    __syncthreads();

    /* seed with the entry point (warp 0, cooperative distance) */
    if (warp == 0) {
        uint32_t ep = qd.entry;
        DT d;
        if (SQ8) {
            int dot = fv_dot8_warp(vq + (uint64_t)ep * dq4, q8, dq4, lane);
            d = (DT)(uint32_t)(vnorm[ep] + qn - 2 * dot);
        } else {
            d = (DT)fv_l2_warp(vecs + (uint64_t)ep * vstride, sq, dim, lane);
        }
        if (lane == 0) {
            visit_test_set(vb, ep);
            visited = 1;
            fr_d[0] = d;
            fr_n[0] = ep;
            fr_cnt = 1;
            bool pass = qd.proot < 0 ||
                        fv_eval_dev(nodes, children, invals, cols, count,
                                    qd.proot, ep);
            if (pass) {
                re_d[0] = d;
                re_n[0] = ep;
                re_cnt = 1;
            }
        }
    }
    __syncthreads();

    while (true) {
        /*
         * ---- warp 0: select the top-W unexpanded frontier entries AND the
         * frontier maximum in one shuffle pass (packed keys keep dist+index
         * in one comparison; no block-wide tree reduction, no extra syncs).
         * Expanding W nodes per iteration divides the serial iteration
         * count — and its fixed selection/sync overhead — by W.
         */
        if (warp == 0) {
            uint64_t best[FV_GRAPH_W];
#pragma unroll
            for (int w = 0; w < FV_GRAPH_W; w++) best[w] = UINT64_MAX;
            DT mxv = (DT)0;
            int mxi = -1;
            for (int i = lane; i < fr_cnt; i += 32) {
                uint64_t key = fv_gkey(fr_d[i], (uint32_t)i);
                if (key < best[FV_GRAPH_W - 1]) {
                    int p = FV_GRAPH_W - 1;
                    while (p > 0 && best[p - 1] > key) {
                        best[p] = best[p - 1];
                        p--;
                    }
                    best[p] = key;
                }
                if (mxi < 0 || fr_d[i] > mxv) {
                    mxv = fr_d[i];
                    mxi = i;
                }
            }
            for (int off = 16; off; off >>= 1) {
                uint64_t other[FV_GRAPH_W];
#pragma unroll
                for (int w = 0; w < FV_GRAPH_W; w++)
                    other[w] = __shfl_down_sync(0xffffffffu, best[w], off);
                DT omv = __shfl_down_sync(0xffffffffu, mxv, off);
                int omi = __shfl_down_sync(0xffffffffu, mxi, off);
                uint64_t m[FV_GRAPH_W];
                int a = 0, b = 0;
#pragma unroll
                for (int w = 0; w < FV_GRAPH_W; w++)
                    m[w] = (b >= FV_GRAPH_W ||
                            (a < FV_GRAPH_W && best[a] <= other[b]))
                               ? best[a++]
                               : other[b++];
#pragma unroll
                for (int w = 0; w < FV_GRAPH_W; w++) best[w] = m[w];
                if (omi >= 0 && (mxi < 0 || omv > mxv)) {
                    mxv = omv;
                    mxi = omi;
                }
            }
            if (lane == 0) {
                /* dynamic expansion width: a large unexpanded frontier
                 * signals an exploration-heavy query (sparse predicate)
                 * where a wide expansion amortizes per-iteration overhead;
                 * a converging query keeps the frontier small and a narrow
                 * expansion avoids wasted speculative distance work */
                int wdyn = fr_cnt >> 6;
                if (wdyn < 4) wdyn = 4;
                if (wdyn > wsel) wdyn = wsel;
                nsel = 0;
                for (int w = 0; w < wdyn; w++)
                    if (best[w] != UINT64_MAX)
                        sel_idx[nsel++] = (int)(best[w] & 0xFFFFFFFFu);
                fr_worst_val = mxv;
                fr_worst_idx = mxi;
                stop = 0;
                if (nsel == 0) {
                    stop = 1;
                    exh = 1; /* frontier empty */
                } else if (visited >= max_visited) {
                    stop = 1;
                    exh = 1;
                } else if (re_cnt >= (int)ef &&
                           best[0] > fv_gkey(worst_res_sh, 0xFFFFFFFFu)) {
                    stop = 1; /* natural stop */
                }
            }
        }
        __syncthreads();
        if (stop) break;
        dbg_iters++;

        /* ---- pop the selected nodes (descending index order keeps the
         * swap-remove indices valid; the frontier-worst index is patched
         * when its entry gets moved) ---- */
        if (tid == 0) {
            for (int a = 1; a < nsel; a++) {
                int v = sel_idx[a];
                int b = a;
                while (b > 0 && sel_idx[b - 1] < v) {
                    sel_idx[b] = sel_idx[b - 1];
                    b--;
                }
                sel_idx[b] = v;
            }
            int nn = 0;
            for (int s = 0; s < nsel; s++) {
                int idx = sel_idx[s];
                uint32_t node = fr_n[idx];
                fr_cnt--;
                fr_d[idx] = fr_d[fr_cnt];
                fr_n[idx] = fr_n[fr_cnt];
                if (fr_worst_idx == fr_cnt) fr_worst_idx = idx;
                sel_node[s] = node;
                int c = (int)l0[(uint64_t)node * l0_stride];
                if (c > deg_clamp) c = deg_clamp;
                sel_base[s] = nn;
                nn += c;
            }
            sel_base[nsel] = nn;
            nneigh = nn;
        }
        __syncthreads();

        /* ---- gather neighbors of all selected nodes cooperatively ---- */
        for (int i = tid; i < nneigh; i += blockDim.x) {
            int s = 0;
            while (s + 1 < nsel && i >= sel_base[s + 1]) s++;
            neigh[i] = l0[(uint64_t)sel_node[s] * l0_stride + 1 +
                          (i - sel_base[s])];
        }
        __syncthreads();

        /* ---- visited filter, compacted into a dense work list
         * (duplicate slots from overlapping neighbor lists dedup through
         * the bitmap: only the first test_set wins). Compaction keeps the
         * distance warps fully loaded instead of skipping visited slots,
         * and NO predicate is evaluated here: the predicate only matters
         * for result-list admission, which is distance-gated below —
         * evaluating it eagerly for every new node paid a random global
         * read per node for nothing (lazy evaluation, like the brute
         * scan's). neigh_pass doubles as the compact index list. ---- */
        if (tid == 0) nnew = 0;
        __syncthreads();
        for (int i = tid; i < nneigh; i += blockDim.x) {
            neigh_new[i] = visit_test_set(vb, neigh[i]) ? 1 : 0;
            if (neigh_new[i]) {
                int c = atomicAdd(&nnew, 1);
                neigh_pass[c] = i;
            }
        }
        __syncthreads();
        const int ncomp = nnew;

        /* ---- distances over the dense list (one warp each) ---- */
        for (int j = warp; j < ncomp; j += nwarp) {
            const int i = neigh_pass[j];
            DT d;
            if (SQ8) {
                int dot = fv_dot8_warp(vq + (uint64_t)neigh[i] * dq4, q8,
                                       dq4, lane);
                d = (DT)(uint32_t)(vnorm[neigh[i]] + qn - 2 * dot);
            } else {
                d = (DT)fv_l2_warp(vecs + (uint64_t)neigh[i] * vstride, sq,
                                   dim, lane);
            }
            if (lane == 0) neigh_d[i] = d;
        }
        __syncthreads();

        /* ---- admit new neighbors. Frontier appends are parallel
         * (atomicAdd slots); the predicate runs lazily here, only for
         * distance-competitive candidates; only rare cases stay serial in
         * thread 0: result-list insertion and overflow replacement. ---- */
        if (tid == 0) {
            npass = 0;
            novf = 0;
        }
        __syncthreads();
        {
            const DT worst_res = worst_res_sh;
            const bool res_full = re_cnt >= (int)ef;

            for (int j = tid; j < ncomp; j += blockDim.x) {
                const int i = neigh_pass[j];
                if (res_full && neigh_d[i] >= worst_res) continue;
                bool pass = qd.proot < 0;
                if (!pass) {
                    if (dbg) atomicAdd(&dbg_evals, 1);
                    pass = fv_eval_dev(nodes, children, invals, cols, count,
                                       qd.proot, neigh[i]);
                }
                if (pass) {
                    int p = atomicAdd(&npass, 1);
                    pass_list[p] = i;
                }
                int pos = atomicAdd(&fr_cnt, 1);
                if (pos < FV_FRONTIER_CAP) {
                    fr_d[pos] = neigh_d[i];
                    fr_n[pos] = neigh[i];
                } else {
                    int o = atomicAdd(&novf, 1);
                    ovf_list[o] = i;
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            DT worst_res = worst_res_sh;

            visited += (uint64_t)nnew;
            if (fr_cnt > FV_FRONTIER_CAP) fr_cnt = FV_FRONTIER_CAP;
            /* overflow: replace the cached worst (stale within the
             * iteration — conservative, costs a little recall at most) */
            for (int o = 0; o < novf; o++) {
                int i = ovf_list[o];
                if (fr_worst_idx >= 0 && neigh_d[i] < fr_worst_val) {
                    fr_d[fr_worst_idx] = neigh_d[i];
                    fr_n[fr_worst_idx] = neigh[i];
                    fr_worst_val = neigh_d[i];
                }
            }
            /* result inserts: only predicate passers, re-checked against
             * the worst that tightens as we insert */
            for (int p = 0; p < npass; p++) {
                int i = pass_list[p];
                DT d = neigh_d[i];
                uint32_t node = neigh[i];

                if (re_cnt >= (int)ef && d >= worst_res) continue;
                if (re_cnt < (int)ef) {
                    re_d[re_cnt] = d;
                    re_n[re_cnt] = node;
                    re_cnt++;
                    if (re_cnt == (int)ef) {
                        worst_res = re_d[0];
                        for (int j2 = 1; j2 < re_cnt; j2++)
                            if (re_d[j2] > worst_res) worst_res = re_d[j2];
                    }
                } else {
                    int wi = 0;
                    for (int j2 = 1; j2 < re_cnt; j2++)
                        if (re_d[j2] > re_d[wi]) wi = j2;
                    re_d[wi] = d;
                    re_n[wi] = node;
                    worst_res = re_d[0];
                    for (int j2 = 1; j2 < re_cnt; j2++)
                        if (re_d[j2] > worst_res) worst_res = re_d[j2];
                }
            }
            worst_res_sh = worst_res;
        }
        __syncthreads();
    }

    /* ---- emit results (unsorted; host sorts <=ef entries). In sq8 mode
     * the traversal ranked candidates by quantized distance; re-rank the
     * emitted list against the fp16 rows in fp32 so the consumer's top-k
     * ordering is unaffected by quantization noise. ---- */
    if (SQ8) {
        for (int i = warp; i < re_cnt; i += nwarp) {
            float d = fv_l2_warp(vecs + (uint64_t)re_n[i] * vstride, sq, dim,
                                 lane);
            if (lane == 0) {
                out[(uint64_t)q * ef + i].node = re_n[i];
                out[(uint64_t)q * ef + i].dist = d;
            }
        }
    } else {
        for (int i = tid; i < re_cnt; i += blockDim.x) {
            out[(uint64_t)q * ef + i].node = re_n[i];
            out[(uint64_t)q * ef + i].dist = (float)re_d[i];
        }
    }
    if (tid == 0) {
        out_cnt[q] = (uint32_t)re_cnt;
        out_exh[q] = (uint8_t)exh;
        if (dbg) {
            dbg[(uint64_t)q * 4 + 0] = (uint32_t)dbg_iters;
            dbg[(uint64_t)q * 4 + 1] = (uint32_t)visited;
            dbg[(uint64_t)q * 4 + 2] = (uint32_t)dbg_evals;
            dbg[(uint64_t)q * 4 + 3] = (uint32_t)re_cnt;
        }
    }
}

/* ================= host code ================= */

extern "C" int fv_gpu_available(void) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
    return n > 0;
}

extern "C" int fv_gpu_index_device(const FvGpuIndex *g) { return g->device; }

/* Pick the device with the most free memory, requiring enough headroom for
 * the index plus batch scratch — on a shared GPU server "most free" can
 * still be a nearly-full card where the upload fits but every later
 * allocation fails. */
/* Devices this process holds live allocations on (one entry per index that
 * is resident). cudaDeviceReset() destroys a device's primary context and
 * with it every allocation made under that context — including page-locked
 * host buffers — so probe-context cleanup must never touch a device that
 * still backs a resident index. */
#define FV_MAX_DEVICES 32
static int dev_refcount[FV_MAX_DEVICES];

static void dev_ref(int device) {
    if (device >= 0 && device < FV_MAX_DEVICES) dev_refcount[device]++;
}

static void dev_unref(int device) {
    if (device >= 0 && device < FV_MAX_DEVICES && dev_refcount[device] > 0)
        dev_refcount[device]--;
}

static int pick_device(size_t need_bytes) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) return -1;
    const size_t margin = 512u << 20;
    int best = -1;
    size_t bestfree = 0;
    for (int i = 0; i < n; i++) {
        if (cudaSetDevice(i) != cudaSuccess) continue;
        size_t f = 0, t = 0;
        if (cudaMemGetInfo(&f, &t) != cudaSuccess) continue;
        if (f > bestfree && f > need_bytes + margin) {
            bestfree = f;
            best = i;
        }
    }
    /* Probing created a context (hundreds of MB) on every device — rude on
     * a shared server and wasteful; tear down the ones we neither won nor
     * already depend on. Resetting a device that backs a resident index
     * would silently invalidate its pointers and crash the next batch. */
    for (int i = 0; i < n; i++) {
        if (i == best) continue;
        if (i < FV_MAX_DEVICES && dev_refcount[i] > 0) continue;
        if (cudaSetDevice(i) == cudaSuccess) cudaDeviceReset();
    }
    if (best >= 0) cudaSetDevice(best);
    (void)cudaGetLastError(); /* consume sticky errors from full devices */
    return best;
}

extern "C" int fv_gpu_index_precision(const FvGpuIndex *g) {
    return g->precision;
}

extern "C" void fv_gpu_index_free(FvGpuIndex *g) {
    if (!g) return;
    if (g->device >= 0) cudaSetDevice(g->device);
    dev_unref(g->device);
    cudaFree(g->d_vecs);
    cudaFree(g->d_vq);
    cudaFree(g->d_vh);
    cudaFree(g->d_vnorm);
    cudaFree(g->d_cols);
    cudaFree(g->d_prows);
    cudaFree(g->d_l0);
    g->d_queries.free();
    g->d_qd.free();
    g->d_nodes.free();
    g->d_children.free();
    g->d_invals.free();
    g->d_spans.free();
    g->d_spanoff.free();
    g->d_scratch.free();
    g->d_counts.free();
    g->d_gout.free();
    g->d_gcnt.free();
    g->d_gexh.free();
    g->d_visited.free();
    g->h_scratch.free();
    g->h_counts.free();
    g->h_gout.free();
    g->h_gcnt.free();
    g->h_gexh.free();
    g->d_tkeys.free();
    g->d_bout.free();
    g->d_bcnt.free();
    g->d_btrunc.free();
    g->h_bout.free();
    g->h_bcnt.free();
    g->h_btrunc.free();
    g->d_qq.free();
    g->d_qnorm.free();
    g->d_gdbg.free();
    g->h_gdbg.free();
    g->d_btmp.free();
    g->d_btmpcnt.free();
    g->h_qq.free();
    g->h_qnorm.free();
    delete g;
}

/* Two chunked PCIe passes over the host vectors: pass 1 reduces per-dim
 * sum/min/max on the GPU (mean + one global scale); pass 2 quantizes each
 * chunk into the resident int8 + fp16 + norm arrays. Peak transient device
 * memory is one staging chunk. Returns cudaSuccess or the first error. */
static cudaError_t sq8_build(FvGpuIndex *g, const float *vectors,
                             double *quant_ms) {
    const uint64_t count = g->count;
    const int dim = g->dim;
    cudaError_t err;
    ProfT q0 = prof_now();

    const uint64_t chunk_rows =
        std::max<uint64_t>(1, (256u << 20) / ((uint64_t)dim * 4));
    DevBuf stage, stats;
    if ((err = stage.ensure(chunk_rows * (uint64_t)dim * 4)) != cudaSuccess)
        return err;
    /* sums (double) + min + max + mean, per dim */
    if ((err = stats.ensure((uint64_t)dim * (8 + 4 + 4 + 4))) != cudaSuccess) {
        stage.free();
        return err;
    }
    double *d_sums = (double *)stats.p;
    float *d_min = (float *)(d_sums + dim);
    float *d_max = d_min + dim;
    float *d_mean = d_max + dim;

    std::vector<double> h_init(dim, 0.0);
    cudaMemcpy(d_sums, h_init.data(), dim * 8, cudaMemcpyHostToDevice);
    std::vector<float> h_minmax(2 * dim);
    for (int d = 0; d < dim; d++) {
        h_minmax[d] = 3.4e38f;
        h_minmax[dim + d] = -3.4e38f;
    }
    cudaMemcpy(d_min, h_minmax.data(), 2 * (uint64_t)dim * 4,
               cudaMemcpyHostToDevice);

    for (uint64_t r0 = 0; r0 < count; r0 += chunk_rows) {
        const int n = (int)std::min(chunk_rows, count - r0);
        if ((err = cudaMemcpy(stage.p, vectors + r0 * dim,
                              (uint64_t)n * dim * 4,
                              cudaMemcpyHostToDevice)) != cudaSuccess)
            goto out;
        fv_stats_kernel<<<256, 256>>>(stage.as<float>(), n, dim, d_sums,
                                      d_min, d_max);
        if ((err = cudaGetLastError()) != cudaSuccess) goto out;
    }

    {
        std::vector<double> sums(dim);
        if ((err = cudaMemcpy(sums.data(), d_sums, dim * 8,
                              cudaMemcpyDeviceToHost)) != cudaSuccess)
            goto out;
        if ((err = cudaMemcpy(h_minmax.data(), d_min, 2 * (uint64_t)dim * 4,
                              cudaMemcpyDeviceToHost)) != cudaSuccess)
            goto out;
        g->qmean.resize(dim);
        float span = 0.f;
        for (int d = 0; d < dim; d++) {
            g->qmean[d] = (float)(sums[d] / (double)count);
            span = std::max(span, std::max(h_minmax[dim + d] - g->qmean[d],
                                           g->qmean[d] - h_minmax[d]));
        }
        if (!(span > 0.f)) span = 1.f;
        g->qscale = span / 127.f;
        g->qinv = 1.f / g->qscale;
        if ((err = cudaMemcpy(d_mean, g->qmean.data(), (uint64_t)dim * 4,
                              cudaMemcpyHostToDevice)) != cudaSuccess)
            goto out;
    }

    for (uint64_t r0 = 0; r0 < count; r0 += chunk_rows) {
        const int n = (int)std::min(chunk_rows, count - r0);
        if ((err = cudaMemcpy(stage.p, vectors + r0 * dim,
                              (uint64_t)n * dim * 4,
                              cudaMemcpyHostToDevice)) != cudaSuccess)
            goto out;
        {
            const int warps_per_block = 8;
            const int blocks = (n + warps_per_block - 1) / warps_per_block;
            fv_quant_kernel<<<blocks, warps_per_block * 32>>>(
                stage.as<float>(), r0, n, dim, g->dpad, g->dq4, d_mean,
                g->qinv, g->d_vq, g->d_vh, g->d_vnorm);
        }
        if ((err = cudaGetLastError()) != cudaSuccess) goto out;
    }
    err = cudaDeviceSynchronize();

out:
    stage.free();
    stats.free();
    *quant_ms = prof_ms(q0, prof_now());
    return err;
}

extern "C" FvGpuIndex *fv_gpu_index_upload(const float *vectors,
                                           const double *cols,
                                           const uint32_t *post_rows,
                                           const uint32_t *l0,
                                           uint32_t l0_stride, uint64_t count,
                                           int dim, int ncols, int device,
                                           int precision, char *errbuf,
                                           size_t errlen) {
    const int dpad = (dim + 15) & ~15;
    const uint64_t vec_bytes =
        precision == FV_GPU_PREC_SQ8
            ? count * ((uint64_t)dpad * 3 + 4) + (256u << 20) /* staging */
            : count * (uint64_t)dim * 4;
    size_t need = vec_bytes + count * (uint64_t)ncols * 12 +
                  (l0 ? count * (uint64_t)l0_stride * 4 : 0);
    ProfT up0 = prof_now();
    if (device < 0) device = pick_device(need);
    if (device < 0) {
        snprintf(errbuf, errlen,
                 "gpu: no CUDA device with %.1f GB free",
                 (need + (512u << 20)) / 1e9);
        return nullptr;
    }
    CUDA_TRY(cudaSetDevice(device));

    FvGpuIndex *g = new FvGpuIndex();
    g->device = device;
    g->precision = precision;
    g->count = count;
    g->dim = dim;
    g->ncols = ncols;
    g->l0_stride = l0_stride;
    g->dpad = dpad;
    g->dq4 = dpad / 4;
    dev_ref(device);

    double quant_ms = 0.0;
    cudaError_t err = cudaSuccess;
    do {
        if (precision == FV_GPU_PREC_SQ8) {
            if ((err = cudaMalloc(&g->d_vq, count * (uint64_t)g->dq4 * 4)) != cudaSuccess) break;
            if ((err = cudaMalloc(&g->d_vh, count * (uint64_t)dpad * 2)) != cudaSuccess) break;
            if ((err = cudaMalloc(&g->d_vnorm, count * 4)) != cudaSuccess) break;
            if ((err = sq8_build(g, vectors, &quant_ms)) != cudaSuccess) break;
        } else {
            if ((err = cudaMalloc(&g->d_vecs, count * (uint64_t)dim * 4)) != cudaSuccess) break;
            if ((err = cudaMemcpy(g->d_vecs, vectors, count * (uint64_t)dim * 4,
                                  cudaMemcpyHostToDevice)) != cudaSuccess) break;
        }
        if (ncols > 0) {
            if ((err = cudaMalloc(&g->d_cols, (uint64_t)ncols * count * 8)) != cudaSuccess) break;
            if ((err = cudaMemcpy(g->d_cols, cols, (uint64_t)ncols * count * 8,
                                  cudaMemcpyHostToDevice)) != cudaSuccess) break;
            if ((err = cudaMalloc(&g->d_prows, (uint64_t)ncols * count * 4)) != cudaSuccess) break;
            if ((err = cudaMemcpy(g->d_prows, post_rows, (uint64_t)ncols * count * 4,
                                  cudaMemcpyHostToDevice)) != cudaSuccess) break;
        }
        if (l0 != nullptr && l0_stride > 0) {
            if ((err = cudaMalloc(&g->d_l0, count * (uint64_t)l0_stride * 4)) != cudaSuccess) break;
            if ((err = cudaMemcpy(g->d_l0, l0, count * (uint64_t)l0_stride * 4,
                                  cudaMemcpyHostToDevice)) != cudaSuccess) break;
        }
    } while (0);
    if (err != cudaSuccess) {
        snprintf(errbuf, errlen, "gpu: index upload failed: %s",
                 cudaGetErrorString(err));
        fv_gpu_index_free(g);
        return nullptr;
    }
    if (prof_on()) {
        cudaDeviceSynchronize();
        fprintf(stderr,
                "FVPROF upload {\"count\":%llu,\"dim\":%d,\"ncols\":%d,"
                "\"precision\":%d,\"bytes\":%llu,\"ms\":%.1f,"
                "\"quant_ms\":%.1f,\"scale\":%.6g}\n",
                (unsigned long long)count, dim, ncols, precision,
                (unsigned long long)need, prof_ms(up0, prof_now()),
                quant_ms, (double)g->qscale);
        fflush(stderr);
    }
    return g;
}

/* tiled path: full scan, lazy predicate, GPU-side merge */
static int fv_gpu_brute_tiled(FvGpuIndex *g, int nq, const float *queries,
                              const FvGpuQueryDesc *qd,
                              const FvGpuPredNode *nodes, int nnodes,
                              const int32_t *children, int nchildren,
                              const double *invals, int ninvals,
                              uint32_t topb, FvHit *out, uint32_t *out_cnt,
                              uint8_t *truncated, int variant,
                              char *errbuf, size_t errlen) {
    const TileVariant &tv = tile_variants[variant];
    const size_t tile_shmem =
        ((size_t)g->dim * tv.qt + (size_t)FV_TILE_WARPS * tv.rt * g->dim) * 4;
    const int rowblocks = tile_rowblocks(g->count);
    const bool prof = prof_on();
    ProfT t0, t1, t2, t3, t4;

    if (prof) t0 = prof_now();
    CUDA_TRY_I(g->d_queries.ensure((uint64_t)nq * g->dim * 4));
    {
        const char *bad = h2d_check(g->d_queries.p, queries,
                                   (uint64_t)nq * g->dim * 4, g->device);
        if (bad != nullptr) {
            snprintf(errbuf, errlen,
                     "gpu: graph queries upload unsafe (%s; nq=%d dim=%d "
                     "dst=%p cap=%zu dev=%d)",
                     bad, nq, g->dim, g->d_queries.p, g->d_queries.cap,
                     g->device);
            return -1;
        }
    }
    CUDA_TRY_I(cudaMemcpy(g->d_queries.p, queries, (uint64_t)nq * g->dim * 4,
                          cudaMemcpyHostToDevice));
    CUDA_TRY_I(g->d_qd.ensure((uint64_t)nq * sizeof(FvGpuQueryDesc)));
    CUDA_TRY_I(cudaMemcpy(g->d_qd.p, qd, (uint64_t)nq * sizeof(FvGpuQueryDesc),
                          cudaMemcpyHostToDevice));
    if (nnodes > 0) {
        CUDA_TRY_I(g->d_nodes.ensure((uint64_t)nnodes * sizeof(FvGpuPredNode)));
        CUDA_TRY_I(cudaMemcpy(g->d_nodes.p, nodes,
                              (uint64_t)nnodes * sizeof(FvGpuPredNode),
                              cudaMemcpyHostToDevice));
    }
    if (nchildren > 0) {
        CUDA_TRY_I(g->d_children.ensure((uint64_t)nchildren * 4));
        CUDA_TRY_I(cudaMemcpy(g->d_children.p, children,
                              (uint64_t)nchildren * 4,
                              cudaMemcpyHostToDevice));
    }
    if (ninvals > 0) {
        CUDA_TRY_I(g->d_invals.ensure((uint64_t)ninvals * 8));
        CUDA_TRY_I(cudaMemcpy(g->d_invals.p, invals, (uint64_t)ninvals * 8,
                              cudaMemcpyHostToDevice));
    }

    const uint64_t keys_per_q =
        (uint64_t)rowblocks * FV_TILE_WARPS * FV_TILE_TOPB;
    CUDA_TRY_I(g->d_tkeys.ensure((uint64_t)nq * keys_per_q * 8));
    CUDA_TRY_I(g->d_bout.ensure((uint64_t)nq * topb * sizeof(FvHit)));
    CUDA_TRY_I(g->d_bcnt.ensure((uint64_t)nq * 4));
    CUDA_TRY_I(g->d_btrunc.ensure((uint64_t)nq));
    CUDA_TRY_I(g->h_bout.ensure((uint64_t)nq * topb * sizeof(FvHit), true));
    CUDA_TRY_I(g->h_bcnt.ensure((uint64_t)nq * 4, true));
    CUDA_TRY_I(g->h_btrunc.ensure((uint64_t)nq, true));

    dim3 grid(rowblocks, (nq + tv.qt - 1) / tv.qt);

    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t1 = prof_now();
    }

#define FV_TILE_LAUNCH(QPL, LPQ, RT)                                        \
    do {                                                                    \
        if (!g->tile_shmem_set && tile_shmem > 40 * 1024) {                 \
            CUDA_TRY_I(cudaFuncSetAttribute(                                \
                fv_brute_tile_kernel<QPL, LPQ, RT>,                         \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                \
                FV_TILE_MAX_SHMEM));                                        \
            g->tile_shmem_set = true;                                       \
        }                                                                   \
        fv_brute_tile_kernel<QPL, LPQ, RT>                                  \
            <<<grid, FV_GPU_THREADS, tile_shmem>>>(                         \
                g->d_vecs, g->d_cols, g->count, g->dim, nq,                 \
                g->d_queries.as<float>(), g->d_qd.as<FvGpuQueryDesc>(),     \
                g->d_nodes.as<FvGpuPredNode>(), g->d_children.as<int32_t>(),\
                g->d_invals.as<double>(), topb, rowblocks,                  \
                g->d_tkeys.as<uint64_t>());                                 \
    } while (0)

    switch (variant) {
        case 0: FV_TILE_LAUNCH(2, 1, 4); break;
        case 1: FV_TILE_LAUNCH(1, 1, 2); break;
        case 2: FV_TILE_LAUNCH(1, 2, 1); break;
        default: FV_TILE_LAUNCH(1, 4, 1); break;
    }
#undef FV_TILE_LAUNCH
    CUDA_TRY_I(cudaGetLastError());
    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t2 = prof_now();
    }

    fv_brute_merge_kernel<<<nq, FV_GPU_THREADS>>>(
        g->d_tkeys.as<uint64_t>(), nq, topb, rowblocks,
        g->d_bout.as<FvHit>(), g->d_bcnt.as<uint32_t>(),
        g->d_btrunc.as<uint8_t>());
    CUDA_TRY_I(cudaGetLastError());
    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t3 = prof_now();
    }

    CUDA_TRY_I(cudaMemcpy(g->h_bout.p, g->d_bout.p,
                          (uint64_t)nq * topb * sizeof(FvHit),
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_bcnt.p, g->d_bcnt.p, (uint64_t)nq * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_btrunc.p, g->d_btrunc.p, (uint64_t)nq,
                          cudaMemcpyDeviceToHost));

    if (prof) t4 = prof_now();
    memcpy(out, g->h_bout.p, (uint64_t)nq * topb * sizeof(FvHit));
    memcpy(out_cnt, g->h_bcnt.p, (uint64_t)nq * 4);
    memcpy(truncated, g->h_btrunc.p, (uint64_t)nq);
    if (prof) {
        prof_brute.batches++;
        prof_brute.queries += (uint64_t)nq;
        prof_brute.h2d += prof_ms(t0, t1);
        prof_brute.kern += prof_ms(t1, t2);
        prof_brute.kern2 += prof_ms(t2, t3);
        prof_brute.d2h += prof_ms(t3, t4);
        prof_brute.host += prof_ms(t4, prof_now());
        prof_dump_maybe();
    }
    return 0;
}

/* SQ8 tiled path: quantize the queries on the host, int8 dp4a full scan,
 * GPU merge to topb_r > topb approximate candidates, fp16 exact rerank down
 * to the final topb. */
static int fv_gpu_brute_tiled8(FvGpuIndex *g, int nq, const float *queries,
                               const FvGpuQueryDesc *qd,
                               const FvGpuPredNode *nodes, int nnodes,
                               const int32_t *children, int nchildren,
                               const double *invals, int ninvals,
                               uint32_t topb, FvHit *out, uint32_t *out_cnt,
                               uint8_t *truncated, int variant,
                               char *errbuf, size_t errlen) {
    const Tile8Variant &tv = tile8_variants[variant];
    const size_t tile_shmem =
        ((size_t)tv.qt + 8 * (size_t)tv.rt) * (size_t)g->dpad;
    const int rowblocks = tile_rowblocks(g->count);
    const int topb_r = (int)std::min<uint32_t>(2 * topb, FV_RERANK_PAD);
    const bool prof = prof_on();
    ProfT t0, t1, t2, t3, t4;

    if (prof) t0 = prof_now();
    /* quantize queries on the host (same centering + scale as the rows) */
    CUDA_TRY_I(g->h_qq.ensure((uint64_t)nq * g->dq4 * 4, true));
    CUDA_TRY_I(g->h_qnorm.ensure((uint64_t)nq * 4, true));
    {
        uint32_t *qq = g->h_qq.as<uint32_t>();
        int32_t *qn = g->h_qnorm.as<int32_t>();
        const float *mean = g->qmean.data();
        for (int q = 0; q < nq; q++) {
            const float *src = queries + (uint64_t)q * g->dim;
            int norm = 0;
            for (int j4 = 0; j4 < g->dq4; j4++) {
                uint32_t packed = 0;
                for (int b = 0; b < 4; b++) {
                    int d = j4 * 4 + b;
                    int xi = 0;
                    if (d < g->dim) {
                        float c = (src[d] - mean[d]) * g->qinv;
                        xi = (int)lrintf(c);
                        xi = std::max(-127, std::min(127, xi));
                    }
                    packed |= (uint32_t)(uint8_t)(int8_t)xi << (8 * b);
                    norm += xi * xi;
                }
                qq[(uint64_t)q * g->dq4 + j4] = packed;
            }
            qn[q] = norm;
        }
    }
    CUDA_TRY_I(g->d_qq.ensure((uint64_t)nq * g->dq4 * 4));
    CUDA_TRY_I(cudaMemcpy(g->d_qq.p, g->h_qq.p, (uint64_t)nq * g->dq4 * 4,
                          cudaMemcpyHostToDevice));
    CUDA_TRY_I(g->d_qnorm.ensure((uint64_t)nq * 4));
    CUDA_TRY_I(cudaMemcpy(g->d_qnorm.p, g->h_qnorm.p, (uint64_t)nq * 4,
                          cudaMemcpyHostToDevice));
    CUDA_TRY_I(g->d_queries.ensure((uint64_t)nq * g->dim * 4));
    CUDA_TRY_I(cudaMemcpy(g->d_queries.p, queries, (uint64_t)nq * g->dim * 4,
                          cudaMemcpyHostToDevice));
    CUDA_TRY_I(g->d_qd.ensure((uint64_t)nq * sizeof(FvGpuQueryDesc)));
    CUDA_TRY_I(cudaMemcpy(g->d_qd.p, qd, (uint64_t)nq * sizeof(FvGpuQueryDesc),
                          cudaMemcpyHostToDevice));
    if (nnodes > 0) {
        CUDA_TRY_I(g->d_nodes.ensure((uint64_t)nnodes * sizeof(FvGpuPredNode)));
        CUDA_TRY_I(cudaMemcpy(g->d_nodes.p, nodes,
                              (uint64_t)nnodes * sizeof(FvGpuPredNode),
                              cudaMemcpyHostToDevice));
    }
    if (nchildren > 0) {
        CUDA_TRY_I(g->d_children.ensure((uint64_t)nchildren * 4));
        CUDA_TRY_I(cudaMemcpy(g->d_children.p, children,
                              (uint64_t)nchildren * 4,
                              cudaMemcpyHostToDevice));
    }
    if (ninvals > 0) {
        CUDA_TRY_I(g->d_invals.ensure((uint64_t)ninvals * 8));
        CUDA_TRY_I(cudaMemcpy(g->d_invals.p, invals, (uint64_t)ninvals * 8,
                              cudaMemcpyHostToDevice));
    }

    const uint64_t keys_per_q =
        (uint64_t)rowblocks * FV_TILE_WARPS * FV_TILE_TOPB;
    CUDA_TRY_I(g->d_tkeys.ensure((uint64_t)nq * keys_per_q * 8));
    CUDA_TRY_I(g->d_btmp.ensure((uint64_t)nq * topb_r * sizeof(FvHit)));
    CUDA_TRY_I(g->d_btmpcnt.ensure((uint64_t)nq * 4));
    CUDA_TRY_I(g->d_bout.ensure((uint64_t)nq * topb * sizeof(FvHit)));
    CUDA_TRY_I(g->d_bcnt.ensure((uint64_t)nq * 4));
    CUDA_TRY_I(g->d_btrunc.ensure((uint64_t)nq));
    CUDA_TRY_I(g->h_bout.ensure((uint64_t)nq * topb * sizeof(FvHit), true));
    CUDA_TRY_I(g->h_bcnt.ensure((uint64_t)nq * 4, true));
    CUDA_TRY_I(g->h_btrunc.ensure((uint64_t)nq, true));

    dim3 grid(rowblocks, (nq + tv.qt - 1) / tv.qt);

    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t1 = prof_now();
    }

#define FV_TILE8_LAUNCH(QPL, LPQ, RT)                                       \
    do {                                                                    \
        if (tile_shmem > 40 * 1024)                                         \
            CUDA_TRY_I(cudaFuncSetAttribute(                                \
                fv_brute_tile8_kernel<QPL, LPQ, RT>,                        \
                cudaFuncAttributeMaxDynamicSharedMemorySize,                \
                FV_TILE_MAX_SHMEM));                                        \
        fv_brute_tile8_kernel<QPL, LPQ, RT>                                 \
            <<<grid, FV_GPU_THREADS, tile_shmem>>>(                         \
                g->d_vq, g->d_vnorm, g->d_cols, g->count, g->dq4, nq,       \
                g->d_qq.as<uint32_t>(), g->d_qnorm.as<int32_t>(),           \
                g->d_qd.as<FvGpuQueryDesc>(), g->d_nodes.as<FvGpuPredNode>(),\
                g->d_children.as<int32_t>(), g->d_invals.as<double>(),      \
                topb, rowblocks, g->d_tkeys.as<uint64_t>());                \
    } while (0)

    switch (variant) {
        case 0: FV_TILE8_LAUNCH(2, 1, 8); break;
        case 1: FV_TILE8_LAUNCH(2, 1, 4); break;
        case 2: FV_TILE8_LAUNCH(2, 2, 2); break;
        case 4: FV_TILE8_LAUNCH(4, 2, 2); break;
        case 5: FV_TILE8_LAUNCH(4, 4, 2); break;
        case 6: FV_TILE8_LAUNCH(2, 2, 4); break;
        default: FV_TILE8_LAUNCH(4, 4, 1); break;
    }
#undef FV_TILE8_LAUNCH
    CUDA_TRY_I(cudaGetLastError());
    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t2 = prof_now();
    }

    fv_brute_merge_kernel<<<nq, FV_GPU_THREADS>>>(
        g->d_tkeys.as<uint64_t>(), nq, (uint32_t)topb_r, rowblocks,
        g->d_btmp.as<FvHit>(), g->d_btmpcnt.as<uint32_t>(),
        g->d_btrunc.as<uint8_t>());
    CUDA_TRY_I(cudaGetLastError());

    fv_rerank_kernel<<<nq, FV_GPU_THREADS, (size_t)g->dim * 4>>>(
        g->d_vh, g->dpad, g->dim, g->d_queries.as<float>(),
        g->d_btmp.as<FvHit>(), g->d_btmpcnt.as<uint32_t>(), topb_r, topb,
        g->d_bout.as<FvHit>(), g->d_bcnt.as<uint32_t>(),
        g->d_btrunc.as<uint8_t>());
    CUDA_TRY_I(cudaGetLastError());
    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t3 = prof_now();
    }

    CUDA_TRY_I(cudaMemcpy(g->h_bout.p, g->d_bout.p,
                          (uint64_t)nq * topb * sizeof(FvHit),
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_bcnt.p, g->d_bcnt.p, (uint64_t)nq * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_btrunc.p, g->d_btrunc.p, (uint64_t)nq,
                          cudaMemcpyDeviceToHost));

    if (prof) t4 = prof_now();
    memcpy(out, g->h_bout.p, (uint64_t)nq * topb * sizeof(FvHit));
    memcpy(out_cnt, g->h_bcnt.p, (uint64_t)nq * 4);
    memcpy(truncated, g->h_btrunc.p, (uint64_t)nq);
    if (prof) {
        prof_brute.batches++;
        prof_brute.queries += (uint64_t)nq;
        prof_brute.h2d += prof_ms(t0, t1);
        prof_brute.kern += prof_ms(t1, t2);
        prof_brute.kern2 += prof_ms(t2, t3);
        prof_brute.d2h += prof_ms(t3, t4);
        prof_brute.host += prof_ms(t4, prof_now());
        prof_dump_maybe();
    }
    return 0;
}

/* Legacy span-driven path: FV_GPU_BPQ blocks per query enumerate the
 * query's driver spans (or the full row range) and keep per-warp heaps;
 * the host merges them. Reads only the enumerated rows, so it wins for
 * highly selective predicates where the tiled full scan wastes the pass. */
static int fv_gpu_brute_legacy(FvGpuIndex *g, int nq, const float *queries,
                               const FvGpuQueryDesc *qd,
                               const uint8_t *dedup,
                               const FvGpuPredNode *nodes, int nnodes,
                               const int32_t *children, int nchildren,
                               const double *invals, int ninvals,
                               const FvGpuSpan *spans, int nspans_total,
                               const uint64_t *spanoff, int nspanoff,
                               uint32_t topb, FvHit *out, uint32_t *out_cnt,
                               uint8_t *truncated, char *errbuf,
                               size_t errlen) {
    const bool prof = prof_on();
    ProfT L0;
    if (prof) L0 = prof_now();
    CUDA_TRY_I(g->d_queries.ensure((uint64_t)nq * g->dim * 4));
    CUDA_TRY_I(cudaMemcpy(g->d_queries.p, queries, (uint64_t)nq * g->dim * 4,
                          cudaMemcpyHostToDevice));
    CUDA_TRY_I(g->d_qd.ensure((uint64_t)nq * sizeof(FvGpuQueryDesc)));
    CUDA_TRY_I(cudaMemcpy(g->d_qd.p, qd, (uint64_t)nq * sizeof(FvGpuQueryDesc),
                          cudaMemcpyHostToDevice));
    if (nnodes > 0) {
        CUDA_TRY_I(g->d_nodes.ensure((uint64_t)nnodes * sizeof(FvGpuPredNode)));
        CUDA_TRY_I(cudaMemcpy(g->d_nodes.p, nodes,
                              (uint64_t)nnodes * sizeof(FvGpuPredNode),
                              cudaMemcpyHostToDevice));
    }
    if (nchildren > 0) {
        CUDA_TRY_I(g->d_children.ensure((uint64_t)nchildren * 4));
        CUDA_TRY_I(cudaMemcpy(g->d_children.p, children,
                              (uint64_t)nchildren * 4,
                              cudaMemcpyHostToDevice));
    }
    if (ninvals > 0) {
        CUDA_TRY_I(g->d_invals.ensure((uint64_t)ninvals * 8));
        CUDA_TRY_I(cudaMemcpy(g->d_invals.p, invals, (uint64_t)ninvals * 8,
                              cudaMemcpyHostToDevice));
    }
    if (nspans_total > 0) {
        CUDA_TRY_I(g->d_spans.ensure((uint64_t)nspans_total * sizeof(FvGpuSpan)));
        CUDA_TRY_I(cudaMemcpy(g->d_spans.p, spans,
                              (uint64_t)nspans_total * sizeof(FvGpuSpan),
                              cudaMemcpyHostToDevice));
    }
    if (nspanoff > 0) {
        CUDA_TRY_I(g->d_spanoff.ensure((uint64_t)nspanoff * 8));
        CUDA_TRY_I(cudaMemcpy(g->d_spanoff.p, spanoff, (uint64_t)nspanoff * 8,
                              cudaMemcpyHostToDevice));
    }

    const uint64_t heaps = (uint64_t)nq * FV_GPU_WPQ;
    CUDA_TRY_I(g->d_scratch.ensure(heaps * topb * sizeof(FvHit)));
    CUDA_TRY_I(g->d_counts.ensure(heaps * 4));
    CUDA_TRY_I(g->h_scratch.ensure(heaps * topb * sizeof(FvHit), true));
    CUDA_TRY_I(g->h_counts.ensure(heaps * 4, true));

    const size_t shmem = (size_t)g->dim * 4;
    if (g->precision == FV_GPU_PREC_SQ8)
        fv_brute_kernel<__half><<<nq * FV_GPU_BPQ, FV_GPU_THREADS, shmem>>>(
            g->d_vh, g->dpad, g->d_cols, g->d_prows, g->count, g->dim,
            g->d_queries.as<float>(), g->d_qd.as<FvGpuQueryDesc>(),
            g->d_nodes.as<FvGpuPredNode>(), g->d_children.as<int32_t>(),
            g->d_invals.as<double>(), g->d_spans.as<FvGpuSpan>(),
            g->d_spanoff.as<uint64_t>(), topb, g->d_scratch.as<FvHit>(),
            g->d_counts.as<uint32_t>());
    else
        fv_brute_kernel<float><<<nq * FV_GPU_BPQ, FV_GPU_THREADS, shmem>>>(
            g->d_vecs, g->dim, g->d_cols, g->d_prows, g->count, g->dim,
            g->d_queries.as<float>(), g->d_qd.as<FvGpuQueryDesc>(),
            g->d_nodes.as<FvGpuPredNode>(), g->d_children.as<int32_t>(),
            g->d_invals.as<double>(), g->d_spans.as<FvGpuSpan>(),
            g->d_spanoff.as<uint64_t>(), topb, g->d_scratch.as<FvHit>(),
            g->d_counts.as<uint32_t>());
    CUDA_TRY_I(cudaGetLastError());

    CUDA_TRY_I(cudaMemcpy(g->h_counts.p, g->d_counts.p, heaps * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_scratch.p, g->d_scratch.p,
                          heaps * topb * sizeof(FvHit),
                          cudaMemcpyDeviceToHost));

    /* host merge per query — mirrors BruteCursor::compute() */
    const FvHit *hs = g->h_scratch.as<FvHit>();
    const uint32_t *hc = g->h_counts.as<uint32_t>();
    std::vector<FvHit> all;
    for (int q = 0; q < nq; q++) {
        all.clear();
        bool trunc = false;
        for (int w = 0; w < FV_GPU_WPQ; w++) {
            uint32_t cnt = hc[(uint64_t)q * FV_GPU_WPQ + w];
            if (cnt >= topb) trunc = true;
            const FvHit *part = hs + ((uint64_t)q * FV_GPU_WPQ + w) * topb;
            all.insert(all.end(), part, part + cnt);
        }
        std::sort(all.begin(), all.end(), [](const FvHit &a, const FvHit &b) {
            return a.dist < b.dist || (a.dist == b.dist && a.node < b.node);
        });
        if (dedup && dedup[q])
            all.erase(std::unique(all.begin(), all.end(),
                                  [](const FvHit &a, const FvHit &b) {
                                      return a.node == b.node;
                                  }),
                      all.end());
        if (all.size() > topb) {
            all.resize(topb);
            trunc = true;
        }
        truncated[q] = (trunc && all.size() < qd[q].total) ? 1 : 0;
        out_cnt[q] = (uint32_t)all.size();
        memcpy(out + (uint64_t)q * topb, all.data(),
               all.size() * sizeof(FvHit));
    }
    if (prof) {
        prof_legacy.batches++;
        prof_legacy.queries += (uint64_t)nq;
        prof_legacy.kern += prof_ms(L0, prof_now());
        prof_dump_maybe();
    }
    return 0;
}

extern "C" int fv_gpu_brute_batch(FvGpuIndex *g, int nq, const float *queries,
                                  const FvGpuQueryDesc *qd,
                                  const uint8_t *dedup,
                                  const FvGpuPredNode *nodes, int nnodes,
                                  const int32_t *children, int nchildren,
                                  const double *invals, int ninvals,
                                  const FvGpuSpan *spans, int nspans_total,
                                  const uint64_t *spanoff, int nspanoff,
                                  uint32_t topb, FvHit *out,
                                  uint32_t *out_cnt, uint8_t *truncated,
                                  char *errbuf, size_t errlen) {
    if (nq < 1 || nq > FV_GPU_MAX_BATCH || topb == 0 ||
        topb > FV_GPU_MAX_TOPB || nspans_total > FV_GPU_MAX_SPANS * nq) {
        snprintf(errbuf, errlen, "gpu: bad brute batch (nq=%d topb=%u)", nq,
                 topb);
        return -1;
    }
    CUDA_TRY_I(cudaSetDevice(g->device));

    const int variant =
        g->precision == FV_GPU_PREC_SQ8 ? tile8_pick(g->dim)
                                        : tile_pick(g->dim);
    if (topb > FV_TILE_TOPB || variant < 0) {
        if (variant < 0 && !g->tile_warned) {
            g->tile_warned = true;
            fprintf(stderr,
                    "paves: no tiled brute variant for dim=%d "
                    "(falling back to the per-query kernel)\n", g->dim);
        }
        return fv_gpu_brute_legacy(g, nq, queries, qd, dedup, nodes, nnodes,
                                   children, nchildren, invals, ninvals,
                                   spans, nspans_total, spanoff, nspanoff,
                                   topb, out, out_cnt, truncated, errbuf,
                                   errlen);
    }

    /* Route each query by its driver-span size: a span enumeration much
     * smaller than a full-scan share (count / query-tile-width) is cheaper
     * on the span kernel; everything else joins the tiled full scan. */
    /* Break-even between the span kernel (reads span_rows * row bytes per
     * query) and the tiled full scan (count * row bytes amortized over QT
     * queries, but measured 4-8x off its bandwidth bound — compute-bound),
     * so the practical cut sits at about count/QT for both modes. */
    const uint64_t span_cut =
        g->precision == FV_GPU_PREC_SQ8
            ? g->count / tile8_variants[variant].qt
            : g->count / tile_variants[variant].qt;
    std::vector<int> tiled_ix, span_ix;
    tiled_ix.reserve(nq);
    for (int q = 0; q < nq; q++) {
        if (qd[q].nspans > 0 && qd[q].total < span_cut)
            span_ix.push_back(q);
        else
            tiled_ix.push_back(q);
    }

    int rc = 0;
    if (span_ix.empty()) {
        rc = g->precision == FV_GPU_PREC_SQ8
                 ? fv_gpu_brute_tiled8(g, nq, queries, qd, nodes, nnodes,
                                       children, nchildren, invals, ninvals,
                                       topb, out, out_cnt, truncated, variant,
                                       errbuf, errlen)
                 : fv_gpu_brute_tiled(g, nq, queries, qd, nodes, nnodes,
                                      children, nchildren, invals, ninvals,
                                      topb, out, out_cnt, truncated, variant,
                                      errbuf, errlen);
        return rc;
    }
    if (tiled_ix.empty())
        return fv_gpu_brute_legacy(g, nq, queries, qd, dedup, nodes, nnodes,
                                   children, nchildren, invals, ninvals,
                                   spans, nspans_total, spanoff, nspanoff,
                                   topb, out, out_cnt, truncated, errbuf,
                                   errlen);

    /* mixed batch: gather each subset (qd references into the concatenated
     * predicate/span arrays stay valid), run both paths, scatter back */
    std::vector<float> sub_q;
    std::vector<FvGpuQueryDesc> sub_qd;
    std::vector<uint8_t> sub_dedup;
    std::vector<FvHit> sub_out;
    std::vector<uint32_t> sub_cnt;
    std::vector<uint8_t> sub_trunc;
    for (int pass = 0; pass < 2; pass++) {
        const std::vector<int> &ix = pass == 0 ? tiled_ix : span_ix;
        const int n = (int)ix.size();
        sub_q.resize((size_t)n * g->dim);
        sub_qd.resize(n);
        sub_dedup.resize(n);
        sub_out.assign((size_t)n * topb, FvHit());
        sub_cnt.resize(n);
        sub_trunc.resize(n);
        for (int i = 0; i < n; i++) {
            memcpy(&sub_q[(size_t)i * g->dim],
                   queries + (uint64_t)ix[i] * g->dim,
                   (size_t)g->dim * 4);
            sub_qd[i] = qd[ix[i]];
            sub_dedup[i] = dedup ? dedup[ix[i]] : 0;
        }
        if (pass == 0)
            rc = g->precision == FV_GPU_PREC_SQ8
                     ? fv_gpu_brute_tiled8(g, n, sub_q.data(), sub_qd.data(),
                                           nodes, nnodes, children, nchildren,
                                           invals, ninvals, topb,
                                           sub_out.data(), sub_cnt.data(),
                                           sub_trunc.data(), variant, errbuf,
                                           errlen)
                     : fv_gpu_brute_tiled(g, n, sub_q.data(), sub_qd.data(),
                                          nodes, nnodes, children, nchildren,
                                          invals, ninvals, topb,
                                          sub_out.data(), sub_cnt.data(),
                                          sub_trunc.data(), variant, errbuf,
                                          errlen);
        else
            rc = fv_gpu_brute_legacy(g, n, sub_q.data(), sub_qd.data(),
                                     sub_dedup.data(), nodes, nnodes,
                                     children, nchildren, invals, ninvals,
                                     spans, nspans_total, spanoff, nspanoff,
                                     topb, sub_out.data(), sub_cnt.data(),
                                     sub_trunc.data(), errbuf, errlen);
        if (rc < 0) return rc;
        for (int i = 0; i < n; i++) {
            memcpy(out + (uint64_t)ix[i] * topb, &sub_out[(size_t)i * topb],
                   (size_t)topb * sizeof(FvHit));
            out_cnt[ix[i]] = sub_cnt[i];
            truncated[ix[i]] = sub_trunc[i];
        }
    }
    return 0;
}

extern "C" int fv_gpu_graph_batch(FvGpuIndex *g, int nq, const float *queries,
                                  const FvGpuQueryDesc *qd,
                                  const FvGpuPredNode *nodes, int nnodes,
                                  const int32_t *children, int nchildren,
                                  const double *invals, int ninvals,
                                  uint32_t ef, uint64_t max_visited,
                                  FvHit *out, uint32_t *out_cnt,
                                  uint8_t *exhausted, char *errbuf,
                                  size_t errlen) {
    if (g->d_l0 == nullptr) {
        snprintf(errbuf, errlen, "gpu: graph not uploaded");
        return -1;
    }
    if (nq < 1 || nq > FV_GPU_MAX_BATCH || ef < 1 || ef > FV_GPU_MAX_EF) {
        snprintf(errbuf, errlen, "gpu: bad graph batch (nq=%d ef=%u)", nq, ef);
        return -1;
    }
    CUDA_TRY_I(cudaSetDevice(g->device));
    const bool prof = prof_on();
    ProfT t0, t1, t2, t3;

    if (prof) t0 = prof_now();
    CUDA_TRY_I(g->d_queries.ensure((uint64_t)nq * g->dim * 4));
    CUDA_TRY_I(cudaMemcpy(g->d_queries.p, queries, (uint64_t)nq * g->dim * 4,
                          cudaMemcpyHostToDevice));
    CUDA_TRY_I(g->d_qd.ensure((uint64_t)nq * sizeof(FvGpuQueryDesc)));
    CUDA_TRY_I(cudaMemcpy(g->d_qd.p, qd, (uint64_t)nq * sizeof(FvGpuQueryDesc),
                          cudaMemcpyHostToDevice));
    if (nnodes > 0) {
        CUDA_TRY_I(g->d_nodes.ensure((uint64_t)nnodes * sizeof(FvGpuPredNode)));
        CUDA_TRY_I(cudaMemcpy(g->d_nodes.p, nodes,
                              (uint64_t)nnodes * sizeof(FvGpuPredNode),
                              cudaMemcpyHostToDevice));
    }
    if (nchildren > 0) {
        CUDA_TRY_I(g->d_children.ensure((uint64_t)nchildren * 4));
        CUDA_TRY_I(cudaMemcpy(g->d_children.p, children,
                              (uint64_t)nchildren * 4,
                              cudaMemcpyHostToDevice));
    }
    if (ninvals > 0) {
        CUDA_TRY_I(g->d_invals.ensure((uint64_t)ninvals * 8));
        CUDA_TRY_I(cudaMemcpy(g->d_invals.p, invals, (uint64_t)ninvals * 8,
                              cudaMemcpyHostToDevice));
    }

    CUDA_TRY_I(g->d_gout.ensure((uint64_t)nq * ef * sizeof(FvHit)));
    CUDA_TRY_I(g->d_gcnt.ensure((uint64_t)nq * 4));
    CUDA_TRY_I(g->d_gexh.ensure((uint64_t)nq));
    CUDA_TRY_I(g->h_gout.ensure((uint64_t)nq * ef * sizeof(FvHit), true));
    CUDA_TRY_I(g->h_gcnt.ensure((uint64_t)nq * 4, true));
    CUDA_TRY_I(g->h_gexh.ensure((uint64_t)nq, true));

    const uint64_t vb_words = (g->count + 31) / 32;
    CUDA_TRY_I(g->d_visited.ensure((uint64_t)nq * vb_words * 4));
    CUDA_TRY_I(cudaMemset(g->d_visited.p, 0, (uint64_t)nq * vb_words * 4));

    /* FV_GPU_GDBG=1: per-query [iterations, visited, predicate evals,
     * results] counters, one distribution line per batch to stderr */
    static int gdbg = -1;
    if (gdbg == -2 || gdbg == -1) {
        const char *e = getenv("FV_GPU_GDBG");
        gdbg = (e && *e && *e != '0') ? 1 : 0;
    }
    uint32_t *dbg_ptr = nullptr;
    if (gdbg) {
        CUDA_TRY_I(g->d_gdbg.ensure((uint64_t)nq * 16));
        CUDA_TRY_I(g->h_gdbg.ensure((uint64_t)nq * 16, true));
        dbg_ptr = g->d_gdbg.as<uint32_t>();
    }

    /* expansion width: bounded by the neighbor buffer, scaled with the
     * beam width — wide beams (large ef) amortize fixed per-iteration cost
     * over more expansions, narrow beams over-expand and waste distance
     * work on candidates the beam would never reach */
    const int maxdeg = (int)g->l0_stride - 1;
    int wsel = FV_GRAPH_NB / (maxdeg > 0 ? maxdeg : 1);
    if (wsel > FV_GRAPH_W) wsel = FV_GRAPH_W;
    if (wsel < 1) wsel = 1;
    const int deg_clamp = FV_GRAPH_NB / wsel;

    /* sq8: quantize the queries for the int8 traversal (the fp32 copies
     * already uploaded above serve the final in-kernel rerank) */
    if (g->precision == FV_GPU_PREC_SQ8) {
        CUDA_TRY_I(g->h_qq.ensure((uint64_t)nq * g->dq4 * 4, true));
        CUDA_TRY_I(g->h_qnorm.ensure((uint64_t)nq * 4, true));
        uint32_t *qq = g->h_qq.as<uint32_t>();
        int32_t *qn = g->h_qnorm.as<int32_t>();
        const float *mean = g->qmean.data();
        for (int q = 0; q < nq; q++) {
            const float *src = queries + (uint64_t)q * g->dim;
            int norm = 0;
            for (int j4 = 0; j4 < g->dq4; j4++) {
                uint32_t packed = 0;
                for (int b = 0; b < 4; b++) {
                    int d = j4 * 4 + b;
                    int xi = 0;
                    if (d < g->dim) {
                        float c = (src[d] - mean[d]) * g->qinv;
                        xi = (int)lrintf(c);
                        xi = std::max(-127, std::min(127, xi));
                    }
                    packed |= (uint32_t)(uint8_t)(int8_t)xi << (8 * b);
                    norm += xi * xi;
                }
                qq[(uint64_t)q * g->dq4 + j4] = packed;
            }
            qn[q] = norm;
        }
        CUDA_TRY_I(g->d_qq.ensure((uint64_t)nq * g->dq4 * 4));
        CUDA_TRY_I(cudaMemcpy(g->d_qq.p, g->h_qq.p,
                              (uint64_t)nq * g->dq4 * 4,
                              cudaMemcpyHostToDevice));
        CUDA_TRY_I(g->d_qnorm.ensure((uint64_t)nq * 4));
        CUDA_TRY_I(cudaMemcpy(g->d_qnorm.p, g->h_qnorm.p, (uint64_t)nq * 4,
                              cudaMemcpyHostToDevice));
    }

    const size_t shmem = (size_t)g->dim * 4 +
        (g->precision == FV_GPU_PREC_SQ8 ? (size_t)g->dq4 * 4 : 0);
    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t1 = prof_now();
    }
    if (g->precision == FV_GPU_PREC_SQ8)
        fv_graph_kernel<__half, uint32_t><<<nq, FV_GRAPH_THREADS, shmem>>>(
            g->d_vh, g->dpad, g->d_vq, g->d_vnorm, g->dq4, g->d_cols,
            g->d_l0, g->l0_stride, g->count, g->dim,
            g->d_queries.as<float>(), g->d_qq.as<uint32_t>(),
            g->d_qnorm.as<int32_t>(), g->d_qd.as<FvGpuQueryDesc>(),
            g->d_nodes.as<FvGpuPredNode>(), g->d_children.as<int32_t>(),
            g->d_invals.as<double>(), ef, max_visited, wsel, deg_clamp,
            g->d_visited.as<uint32_t>(), vb_words, g->d_gout.as<FvHit>(),
            g->d_gcnt.as<uint32_t>(), g->d_gexh.as<uint8_t>(), dbg_ptr);
    else
        fv_graph_kernel<float, float><<<nq, FV_GRAPH_THREADS, shmem>>>(
            g->d_vecs, g->dim, nullptr, nullptr, 0, g->d_cols,
            g->d_l0, g->l0_stride, g->count, g->dim,
            g->d_queries.as<float>(), nullptr, nullptr,
            g->d_qd.as<FvGpuQueryDesc>(),
            g->d_nodes.as<FvGpuPredNode>(), g->d_children.as<int32_t>(),
            g->d_invals.as<double>(), ef, max_visited, wsel, deg_clamp,
            g->d_visited.as<uint32_t>(), vb_words, g->d_gout.as<FvHit>(),
            g->d_gcnt.as<uint32_t>(), g->d_gexh.as<uint8_t>(), dbg_ptr);
    CUDA_TRY_I(cudaGetLastError());
    if (prof) {
        CUDA_TRY_I(cudaDeviceSynchronize());
        t2 = prof_now();
    }

    CUDA_TRY_I(cudaMemcpy(g->h_gcnt.p, g->d_gcnt.p, (uint64_t)nq * 4,
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_gexh.p, g->d_gexh.p, (uint64_t)nq,
                          cudaMemcpyDeviceToHost));
    CUDA_TRY_I(cudaMemcpy(g->h_gout.p, g->d_gout.p,
                          (uint64_t)nq * ef * sizeof(FvHit),
                          cudaMemcpyDeviceToHost));

    if (gdbg) {
        CUDA_TRY_I(cudaMemcpy(g->h_gdbg.p, g->d_gdbg.p, (uint64_t)nq * 16,
                              cudaMemcpyDeviceToHost));
        const uint32_t *d = g->h_gdbg.as<uint32_t>();
        std::vector<uint32_t> it(nq);
        uint64_t vis = 0, ev = 0, res = 0;
        for (int q = 0; q < nq; q++) {
            it[q] = d[(uint64_t)q * 4];
            vis += d[(uint64_t)q * 4 + 1];
            ev += d[(uint64_t)q * 4 + 2];
            res += d[(uint64_t)q * 4 + 3];
        }
        std::sort(it.begin(), it.end());
        fprintf(stderr,
                "FVGDBG {\"nq\":%d,\"ef\":%u,\"iters\":[%u,%u,%u],"
                "\"iters_sum\":%llu,\"visited_avg\":%.0f,"
                "\"evals_avg\":%.0f,\"res_avg\":%.1f}\n",
                nq, ef, it[0], it[nq / 2], it[nq - 1],
                (unsigned long long)std::accumulate(it.begin(), it.end(),
                                                    0ull),
                (double)vis / nq, (double)ev / nq, (double)res / nq);
        fflush(stderr);
    }

    if (prof) t3 = prof_now();
    const FvHit *hout = g->h_gout.as<FvHit>();
    for (int q = 0; q < nq; q++) {
        uint32_t cnt = g->h_gcnt.as<uint32_t>()[q];
        FvHit *dst = out + (uint64_t)q * ef;
        memcpy(dst, hout + (uint64_t)q * ef, cnt * sizeof(FvHit));
        std::sort(dst, dst + cnt, [](const FvHit &a, const FvHit &b) {
            return a.dist < b.dist || (a.dist == b.dist && a.node < b.node);
        });
        out_cnt[q] = cnt;
        exhausted[q] = g->h_gexh.as<uint8_t>()[q];
    }
    if (prof) {
        prof_graph.batches++;
        prof_graph.queries += (uint64_t)nq;
        prof_graph.h2d += prof_ms(t0, t1);
        prof_graph.kern += prof_ms(t1, t2);
        prof_graph.d2h += prof_ms(t2, t3);
        prof_graph.host += prof_ms(t3, prof_now());
        prof_dump_maybe();
    }
    return 0;
}

/* ================= capacity probing ================= */

extern "C" int fv_gpu_tile_variant_qt(int dim) {
    int v = tile_pick(dim);
    return v < 0 ? 0 : tile_variants[v].qt;
}

/* Free/total memory of the best-candidate (or already selected) CUDA
 * device. The first call probes all devices (tearing down extra contexts);
 * later calls just query the chosen one. Returns the device id, or -1.
 * Creates a CUDA context: call from the GPU worker, not from backends. */
extern "C" int fv_gpu_mem_probe(uint64_t *free_b, uint64_t *total_b) {
    static int probed_dev = -2;
    size_t f = 0, t = 0;
    if (probed_dev == -2) probed_dev = pick_device(0);
    if (probed_dev < 0) return -1;
    if (cudaSetDevice(probed_dev) != cudaSuccess ||
        cudaMemGetInfo(&f, &t) != cudaSuccess)
        return -1;
    *free_b = (uint64_t)f;
    *total_b = (uint64_t)t;
    return probed_dev;
}
