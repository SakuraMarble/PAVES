/*
 * engine.cpp — paves C++ vector engine.
 *
 * Contains: index file format + builder, parallel HNSW construction,
 * filtered HNSW search with iterative widening, multi-threaded SIMD
 * brute-force top-k over posting-list driver spans, thread pool.
 *
 * No PostgreSQL headers here; interface is engine_api.h. No exceptions
 * escape: all entry points return error codes / NULL.
 */
#include "engine_api.h"
#include "gpu_engine.h"

#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

/* ================= distance kernels ================= */

static inline float l2sq(const float *a, const float *b, int dim) {
#if defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        __m512 d0 = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        __m512 d1 = _mm512_sub_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16));
        acc0 = _mm512_fmadd_ps(d0, d0, acc0);
        acc1 = _mm512_fmadd_ps(d1, d1, acc1);
    }
    for (; i + 16 <= dim; i += 16) {
        __m512 d = _mm512_sub_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i));
        acc0 = _mm512_fmadd_ps(d, d, acc0);
    }
    float sum = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#else
    float sum = 0;
    for (int i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
#endif
}

/* ================= thread pool ================= */

class ThreadPool {
    /* Each parallel_for creates a refcounted Job. Workers keep a shared_ptr
     * to the job they picked up, so a late worker can never claim task
     * indices from (or run the closure of) a newer job — it only sees its
     * own, already-exhausted job. */
    struct Job {
        std::function<void(int)> fn;
        int total = 0;
        std::atomic<int> next{0};
        std::atomic<int> pending{0};
    };

  public:
    static ThreadPool &instance() {
        /* Intentionally leaked: a static instance would run ~thread() on
         * joinable workers during backend exit() and abort the process. */
        static ThreadPool *pool = new ThreadPool();
        return *pool;
    }

    /* Run fn(task_index) for 0..ntasks-1 using up to `threads` workers.
     * The calling thread participates. Synchronous. */
    void parallel_for(int threads, int ntasks, const std::function<void(int)> &fn) {
        if (threads <= 1 || ntasks <= 1) {
            for (int i = 0; i < ntasks; i++) fn(i);
            return;
        }
        ensure_workers(threads - 1);
        auto job = std::make_shared<Job>();
        job->fn = fn;
        job->total = ntasks;
        job->pending.store(ntasks, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            current_ = job;
            generation_++;
            cv_.notify_all();
        }
        run_job(*job);
        {
            std::unique_lock<std::mutex> lk(mu_);
            done_cv_.wait(lk, [&] {
                return job->pending.load(std::memory_order_acquire) == 0;
            });
            if (current_ == job)
                current_.reset();
        }
    }

  private:
    ThreadPool() = default;

    void ensure_workers(int n) {
        std::lock_guard<std::mutex> lk(mu_);
        while ((int)workers_.size() < n)
            workers_.emplace_back([this] { worker_loop(); });
    }

    void run_job(Job &job) {
        for (;;) {
            int i = job.next.fetch_add(1, std::memory_order_relaxed);
            if (i >= job.total) break;
            job.fn(i);
            if (job.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lk(mu_); /* serialize with waiter */
                done_cv_.notify_all();
            }
        }
    }

    void worker_loop() {
        sigset_t set;
        sigfillset(&set);
        pthread_sigmask(SIG_BLOCK, &set, nullptr); /* never handle PG signals */
        uint64_t seen = 0;
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return generation_ != seen && current_ != nullptr; });
                seen = generation_;
                job = current_;
            }
            run_job(*job);
        }
    }

    std::mutex mu_;
    std::condition_variable cv_, done_cv_;
    std::vector<std::thread> workers_;
    std::shared_ptr<Job> current_;
    uint64_t generation_ = 0;
};

/* ================= file format ================= */

constexpr char FV_MAGIC[8] = {'P', 'G', 'F', 'A', 'V', 'R', '0', '1'};
constexpr uint32_t FV_VERSION = 1;

struct FileHeader {
    char magic[8];
    uint32_t version;
    uint32_t dim;
    uint64_t count;
    int32_t vec_attnum;
    uint32_t ncols;
    int32_t col_attnums[FV_MAX_COLS];
    uint32_t M;   /* upper-level degree cap; level-0 cap is 2M */
    uint32_t ef_construction;
    uint32_t max_level;
    uint64_t entry_node;
    uint64_t off_vectors;    /* float[count*dim] */
    uint64_t off_tids;       /* uint64[count] */
    uint64_t off_cols;       /* double[ncols*count], column-major */
    uint64_t off_post_vals;  /* double[ncols*count]: per-col sorted values */
    uint64_t off_post_rows;  /* uint32[ncols*count]: per-col rows by value */
    uint64_t off_levels;     /* uint8[count] */
    uint64_t off_l0;         /* uint32[count*(2M+1)]: [cnt, neigh...] */
    uint64_t off_upper_offs; /* uint64[count+1] slot offsets into upper */
    uint64_t off_upper;      /* uint32[]: per node, level slabs of (M+1) */
    uint64_t file_size;
};

constexpr uint64_t align64(uint64_t x) { return (x + 63) & ~uint64_t(63); }

/* splitmix64 for deterministic level assignment */
static inline uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97f4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

} // namespace

/* ================= index (query side) ================= */

struct FvIndex {
    void *map = nullptr;
    size_t map_size = 0;
    const FileHeader *h = nullptr;
    const float *vectors = nullptr;
    const uint64_t *tids = nullptr;
    const double *cols = nullptr;
    const double *post_vals = nullptr;
    const uint32_t *post_rows = nullptr;
    const uint8_t *levels = nullptr;
    const uint32_t *l0 = nullptr;
    const uint64_t *upper_offs = nullptr;
    const uint32_t *upper = nullptr;
    uint32_t l0_stride = 0;
    std::vector<uint64_t> nan_start; /* per col: first NaN slot in postings */

    /* epoch-tagged visited scratch, reused across queries in this backend */
    std::vector<uint32_t> visit_tags;
    uint32_t visit_epoch = 0;
    bool scratch_busy = false;

    /* GPU residency: uploaded lazily on the first GPU cursor; a failed
     * upload is remembered so we don't retry on every query */
    FvGpuIndex *gpu = nullptr;
    bool gpu_failed = false;

    const float *vec(uint64_t node) const { return vectors + node * h->dim; }
    const double *col(int c) const { return cols + (uint64_t)c * h->count; }
    const double *pvals(int c) const { return post_vals + (uint64_t)c * h->count; }
    const uint32_t *prows(int c) const { return post_rows + (uint64_t)c * h->count; }
    const uint32_t *neigh0(uint64_t node) const { return l0 + node * l0_stride; }
    const uint32_t *neigh_upper(uint64_t node, int level) const {
        return upper + upper_offs[node] + (uint64_t)(level - 1) * (h->M + 1);
    }
};

extern "C" FvIndex *fv_index_open(const char *path, char *errbuf, size_t errlen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        snprintf(errbuf, errlen, "cannot open %s: %s", path, strerror(errno));
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || (size_t)st.st_size < sizeof(FileHeader)) {
        snprintf(errbuf, errlen, "bad index file %s", path);
        close(fd);
        return nullptr;
    }
    void *map = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (map == MAP_FAILED) {
        snprintf(errbuf, errlen, "mmap failed: %s", strerror(errno));
        return nullptr;
    }
    const FileHeader *h = (const FileHeader *)map;
    if (memcmp(h->magic, FV_MAGIC, 8) != 0 || h->version != FV_VERSION ||
        h->file_size != (uint64_t)st.st_size) {
        snprintf(errbuf, errlen, "index file %s: bad magic/version/size", path);
        munmap(map, st.st_size);
        return nullptr;
    }
    FvIndex *idx = new FvIndex();
    idx->map = map;
    idx->map_size = st.st_size;
    idx->h = h;
    const char *base = (const char *)map;
    idx->vectors = (const float *)(base + h->off_vectors);
    idx->tids = (const uint64_t *)(base + h->off_tids);
    idx->cols = (const double *)(base + h->off_cols);
    idx->post_vals = (const double *)(base + h->off_post_vals);
    idx->post_rows = (const uint32_t *)(base + h->off_post_rows);
    idx->levels = (const uint8_t *)(base + h->off_levels);
    idx->l0 = (const uint32_t *)(base + h->off_l0);
    idx->upper_offs = (const uint64_t *)(base + h->off_upper_offs);
    idx->upper = (const uint32_t *)(base + h->off_upper);
    idx->l0_stride = 2 * h->M + 1;
    idx->nan_start.resize(h->ncols);
    for (uint32_t c = 0; c < h->ncols; c++) {
        const double *v = idx->pvals(c);
        uint64_t lo = 0, hi = h->count;
        while (lo < hi) { /* NaNs sort last */
            uint64_t mid = (lo + hi) / 2;
            if (std::isnan(v[mid])) hi = mid; else lo = mid + 1;
        }
        idx->nan_start[c] = lo;
    }
    return idx;
}

extern "C" void fv_index_close(FvIndex *idx) {
    if (!idx) return;
#ifdef FV_GPU
    if (idx->gpu) fv_gpu_index_free(idx->gpu);
#endif
    if (idx->map) munmap(idx->map, idx->map_size);
    delete idx;
}

extern "C" uint64_t fv_index_count(const FvIndex *idx) { return idx->h->count; }
extern "C" int fv_index_dim(const FvIndex *idx) { return idx->h->dim; }
extern "C" int32_t fv_index_vec_attnum(const FvIndex *idx) { return idx->h->vec_attnum; }
extern "C" int fv_index_ncols(const FvIndex *idx) { return idx->h->ncols; }
extern "C" int fv_index_col_slot(const FvIndex *idx, int32_t attnum) {
    for (uint32_t i = 0; i < idx->h->ncols; i++)
        if (idx->h->col_attnums[i] == attnum) return (int)i;
    return -1;
}
extern "C" uint64_t fv_index_tid(const FvIndex *idx, uint64_t node) {
    return idx->tids[node];
}

/* raw vector access + posting-value percentile: benchmark tooling reuses
 * index rows as query vectors and derives predicate constants of a target
 * selectivity from the per-column sorted values */
extern "C" const float *fv_index_vector(const FvIndex *idx, uint64_t node) {
    return idx->vec(node);
}

extern "C" double fv_index_col_percentile(const FvIndex *idx, int col_slot,
                                          double frac) {
    uint64_t n = idx->nan_start[col_slot]; /* non-NaN prefix */
    if (n == 0) return 0.0;
    uint64_t pos = (uint64_t)(frac * (double)(n - 1));
    if (pos >= n) pos = n - 1;
    return idx->pvals(col_slot)[pos];
}

/* Same, but rejects out-of-range node ids instead of reading past the
 * mapping. Results coming back from a GPU kernel are indexed into the
 * mmap'd TID array, so a single bad id would segfault the whole instance;
 * the arbiter worker uses this variant and drops offending candidates. */
extern "C" uint64_t fv_index_tid_checked(const FvIndex *idx, uint64_t node) {
    if (node >= idx->h->count) return UINT64_MAX;
    return idx->tids[node];
}

/* ================= predicates ================= */

namespace {

struct PredN {
    int32_t kind;
    int32_t col;
    int32_t op;
    double value;
    std::vector<double> values;  /* IN */
    std::vector<int32_t> children;
};

struct Span {
    int32_t col;
    uint64_t lo, hi; /* [lo, hi) positions in prows(col) */
};

} // namespace

struct FvPred {
    const FvIndex *idx;
    std::vector<PredN> nodes; /* node 0 = root */

    bool has_driver = false;
    bool covers = false;   /* driver spans == exact match set */
    bool may_dup = false;  /* same row may appear in multiple spans */
    uint64_t driver_size = 0;
    std::vector<Span> spans;

    bool eval_node(int32_t ni, uint32_t row) const {
        const PredN &n = nodes[ni];
        switch (n.kind) {
            case FV_NODE_CMP: {
                double v = idx->col(n.col)[row];
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
                double v = idx->col(n.col)[row];
                for (double x : n.values)
                    if (v == x) return true;
                return false;
            }
            case FV_NODE_AND:
                for (int32_t c : n.children)
                    if (!eval_node(c, row)) return false;
                return true;
            case FV_NODE_OR:
                for (int32_t c : n.children)
                    if (eval_node(c, row)) return true;
                return false;
            case FV_NODE_NOT:
                return !eval_node(n.children[0], row);
        }
        return false;
    }
    bool eval(uint32_t row) const { return eval_node(0, row); }
};

namespace {

/* binary search boundaries in the finite prefix of a posting values array */
static uint64_t post_lower(const FvIndex *idx, int col, double v) {
    const double *a = idx->pvals(col);
    uint64_t lo = 0, hi = idx->nan_start[col];
    while (lo < hi) {
        uint64_t mid = (lo + hi) / 2;
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}
static uint64_t post_upper(const FvIndex *idx, int col, double v) {
    const double *a = idx->pvals(col);
    uint64_t lo = 0, hi = idx->nan_start[col];
    while (lo < hi) {
        uint64_t mid = (lo + hi) / 2;
        if (a[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

struct DriverInfo {
    bool ok = false;
    bool covers = false;
    bool may_dup = false;
    uint64_t size = 0;
    std::vector<Span> spans;
};

static DriverInfo make_driver(const FvIndex *idx, const FvPred *p, int32_t ni) {
    const PredN &n = p->nodes[ni];
    DriverInfo d;
    switch (n.kind) {
        case FV_NODE_CMP: {
            if (std::isnan(n.value)) { /* always-false leaf */
                if (n.op == FV_CMP_EQ) {
                    d.ok = true;
                    d.covers = true;
                    d.size = 0;
                }
                return d;
            }
            uint64_t lo = 0, hi = 0, fin = idx->nan_start[n.col];
            switch (n.op) {
                case FV_CMP_LT: lo = 0; hi = post_lower(idx, n.col, n.value); break;
                case FV_CMP_LE: lo = 0; hi = post_upper(idx, n.col, n.value); break;
                case FV_CMP_EQ:
                    lo = post_lower(idx, n.col, n.value);
                    hi = post_upper(idx, n.col, n.value);
                    break;
                case FV_CMP_GE: lo = post_lower(idx, n.col, n.value); hi = fin; break;
                case FV_CMP_GT: lo = post_upper(idx, n.col, n.value); hi = fin; break;
                default: return d; /* NE: no driver */
            }
            d.ok = true;
            d.covers = true;
            if (hi > lo) {
                d.spans.push_back({n.col, lo, hi});
                d.size = hi - lo;
            }
            return d;
        }
        case FV_NODE_IN: {
            d.ok = true;
            d.covers = true;
            for (double v : n.values) {
                if (std::isnan(v)) continue;
                uint64_t lo = post_lower(idx, n.col, v);
                uint64_t hi = post_upper(idx, n.col, v);
                if (hi > lo) {
                    d.spans.push_back({n.col, lo, hi});
                    d.size += hi - lo;
                }
            }
            return d;
        }
        case FV_NODE_AND: {
            /* pick the smallest child driver; additionally, CMP children on
             * the same column intersect into one posting range (e.g.
             * c >= a AND c < b), which is usually far smaller than any
             * single conjunct's span */
            DriverInfo best;
            for (int32_t c : n.children) {
                DriverInfo cd = make_driver(idx, p, c);
                if (cd.ok && (!best.ok || cd.size < best.size)) best = cd;
            }
            if (best.ok && n.children.size() > 1)
                best.covers = false; /* other conjuncts still need eval */
            {
                struct Range { uint64_t lo, hi; bool valid; int nconj; };
                std::vector<std::pair<int32_t, Range>> percol;
                for (int32_t c : n.children) {
                    const PredN &ch = p->nodes[c];
                    if (ch.kind != FV_NODE_CMP || std::isnan(ch.value))
                        continue;
                    uint64_t lo = 0, hi = 0, fin = idx->nan_start[ch.col];
                    switch (ch.op) {
                        case FV_CMP_LT:
                            lo = 0; hi = post_lower(idx, ch.col, ch.value);
                            break;
                        case FV_CMP_LE:
                            lo = 0; hi = post_upper(idx, ch.col, ch.value);
                            break;
                        case FV_CMP_EQ:
                            lo = post_lower(idx, ch.col, ch.value);
                            hi = post_upper(idx, ch.col, ch.value);
                            break;
                        case FV_CMP_GE:
                            lo = post_lower(idx, ch.col, ch.value); hi = fin;
                            break;
                        case FV_CMP_GT:
                            lo = post_upper(idx, ch.col, ch.value); hi = fin;
                            break;
                        default: continue; /* NE folds nothing */
                    }
                    Range *r = nullptr;
                    for (auto &pc : percol)
                        if (pc.first == ch.col) r = &pc.second;
                    if (!r) {
                        percol.push_back({ch.col, {0, UINT64_MAX, true, 0}});
                        r = &percol.back().second;
                    }
                    r->lo = std::max(r->lo, lo);
                    r->hi = std::min(r->hi, hi);
                    r->nconj++;
                }
                for (auto &pc : percol) {
                    if (pc.second.nconj < 2) continue; /* single: covered above */
                    DriverInfo cd;
                    cd.ok = true;
                    cd.covers = true;
                    if (pc.second.hi > pc.second.lo) {
                        cd.spans.push_back(
                            {pc.first, pc.second.lo, pc.second.hi});
                        cd.size = pc.second.hi - pc.second.lo;
                    }
                    if (!best.ok || cd.size < best.size) {
                        best = cd;
                        /* covers survives only if the intersection
                         * absorbed every conjunct */
                        best.covers =
                            (size_t)pc.second.nconj == n.children.size();
                    }
                }
            }
            return best;
        }
        case FV_NODE_OR: {
            d.ok = true;
            d.covers = true;
            for (int32_t c : n.children) {
                DriverInfo cd = make_driver(idx, p, c);
                if (!cd.ok) return DriverInfo(); /* one child unbounded: no driver */
                d.spans.insert(d.spans.end(), cd.spans.begin(), cd.spans.end());
                d.size += cd.size;
                d.covers = d.covers && cd.covers;
                d.may_dup = d.may_dup || cd.may_dup;
            }
            if (n.children.size() > 1) d.may_dup = true;
            return d;
        }
        default:
            return d; /* NOT: no driver */
    }
}

static int32_t copy_tree(FvPred *p, const FvPredNode *src) {
    int32_t ni = (int32_t)p->nodes.size();
    p->nodes.emplace_back();
    {
        PredN &n = p->nodes[ni];
        n.kind = src->kind;
        n.col = src->col;
        n.op = src->op;
        n.value = src->value;
        if (src->kind == FV_NODE_IN)
            n.values.assign(src->values, src->values + src->nvalues);
    }
    if (src->kind == FV_NODE_AND || src->kind == FV_NODE_OR ||
        src->kind == FV_NODE_NOT) {
        std::vector<int32_t> children;
        for (int i = 0; i < src->nchildren; i++)
            children.push_back(copy_tree(p, src->children[i]));
        p->nodes[ni].children = std::move(children);
    }
    return ni;
}

} // namespace

extern "C" FvPred *fv_pred_compile(FvIndex *idx, const FvPredNode *tree) {
    FvPred *p = new FvPred();
    p->idx = idx;
    copy_tree(p, tree);
    DriverInfo d = make_driver(idx, p, 0);
    p->has_driver = d.ok;
    p->covers = d.covers;
    p->may_dup = d.may_dup;
    p->driver_size = d.size;
    p->spans = std::move(d.spans);
    return p;
}

extern "C" void fv_pred_free(FvPred *pred) { delete pred; }

extern "C" uint64_t fv_pred_estimate(const FvPred *pred) {
    if (!pred->has_driver) return UINT64_MAX;
    return pred->driver_size; /* 0 is authoritative, else an upper bound */
}

/* ================= builder ================= */

struct FvBuilder {
    int dim = 0;
    int ncols = 0;
    int32_t col_attnums[FV_MAX_COLS];
    std::vector<float> vectors;
    std::vector<uint64_t> tids;
    std::vector<std::vector<double>> cols;
    uint64_t count = 0;
};

extern "C" FvBuilder *fv_builder_create(int dim, int ncols, const int32_t *col_attnums,
                                        uint64_t reserve) {
    if (ncols > FV_MAX_COLS) return nullptr;
    FvBuilder *b = new FvBuilder();
    b->dim = dim;
    b->ncols = ncols;
    for (int i = 0; i < ncols; i++) b->col_attnums[i] = col_attnums[i];
    b->cols.resize(ncols);
    if (dim > 0) b->vectors.reserve(reserve * dim);
    b->tids.reserve(reserve);
    for (auto &c : b->cols) c.reserve(reserve);
    return b;
}

extern "C" void fv_builder_add_row(FvBuilder *b, const float *vec, uint64_t tid,
                                   const double *scalars, const uint8_t *isnull) {
    b->vectors.insert(b->vectors.end(), vec, vec + b->dim);
    b->tids.push_back(tid);
    for (int i = 0; i < b->ncols; i++)
        b->cols[i].push_back(isnull && isnull[i] ? std::nan("") : scalars[i]);
    b->count++;
}

extern "C" void fv_builder_destroy(FvBuilder *b) { delete b; }

namespace {

/* -------- HNSW construction state -------- */

struct Spinlock {
    std::atomic_flag f = ATOMIC_FLAG_INIT;
    void lock() {
        while (f.test_and_set(std::memory_order_acquire)) {
#if defined(__x86_64__)
            _mm_pause();
#endif
        }
    }
    void unlock() { f.clear(std::memory_order_release); }
};

struct HnswBuild {
    const float *vectors;
    int dim;
    uint64_t count;
    uint32_t M, M0, efc;
    double mL;

    std::vector<uint8_t> levels;
    std::vector<uint32_t> l0;      /* stride M0+1: [cnt, ...] */
    std::vector<uint32_t *> upper; /* per node, level slabs of (M+1) */
    std::vector<Spinlock> locks;
    std::atomic<uint64_t> entry{0};
    std::atomic<int> entry_level{-1};
    std::mutex entry_mu;

    uint32_t l0_stride;

    HnswBuild(const float *v, int d, uint64_t n, uint32_t m, uint32_t ef)
        : vectors(v), dim(d), count(n), M(m), M0(2 * m), efc(ef),
          mL(1.0 / std::log((double)m)), levels(n), upper(n, nullptr), locks(n) {
        l0_stride = M0 + 1;
        l0.assign(n * (uint64_t)l0_stride, 0);
        for (uint64_t i = 0; i < n; i++) {
            double u = (splitmix64(i * 0x5851F42D4C957F2DULL + 12345) >> 11) *
                       (1.0 / 9007199254740992.0);
            if (u < 1e-18) u = 1e-18;
            int lvl = (int)(-std::log(u) * mL);
            if (lvl > 30) lvl = 30;
            levels[i] = (uint8_t)lvl;
            if (lvl > 0)
                upper[i] = (uint32_t *)calloc((size_t)lvl * (M + 1), 4);
        }
    }
    ~HnswBuild() {
        for (auto pp : upper) free(pp);
    }

    const float *vec(uint64_t i) const { return vectors + i * dim; }
    float dist(uint64_t a, uint64_t b) const { return l2sq(vec(a), vec(b), dim); }
    float distq(const float *q, uint64_t b) const { return l2sq(q, vec(b), dim); }

    uint32_t *list(uint64_t node, int level) {
        if (level == 0) return &l0[node * (uint64_t)l0_stride];
        return upper[node] + (size_t)(level - 1) * (M + 1);
    }

    uint64_t greedy(const float *q, uint64_t ep, int level) {
        float d = distq(q, ep);
        bool changed = true;
        while (changed) {
            changed = false;
            locks[ep].lock();
            uint32_t *lst = list(ep, level);
            uint32_t cnt = lst[0];
            uint32_t neigh[64];
            for (uint32_t i = 0; i < cnt; i++) neigh[i] = lst[1 + i];
            locks[ep].unlock();
            for (uint32_t i = 0; i < cnt; i++) {
                float nd = distq(q, neigh[i]);
                if (nd < d) {
                    d = nd;
                    ep = neigh[i];
                    changed = true;
                }
            }
        }
        return ep;
    }

    void search_layer(const float *q, uint64_t ep, int level, uint32_t ef,
                      std::vector<uint32_t> &visit_tags, uint32_t epoch,
                      std::vector<std::pair<float, uint32_t>> &out) {
        using P = std::pair<float, uint32_t>;
        std::priority_queue<P, std::vector<P>, std::greater<P>> cand; /* min */
        std::priority_queue<P> res;                                    /* max */
        float epd = distq(q, ep);
        cand.push({epd, (uint32_t)ep});
        res.push({epd, (uint32_t)ep});
        visit_tags[ep] = epoch;
        while (!cand.empty()) {
            auto c = cand.top();
            if (res.size() >= ef && c.first > res.top().first) break;
            cand.pop();
            locks[c.second].lock();
            uint32_t *lst = list(c.second, level);
            uint32_t cnt = lst[0];
            uint32_t neigh[64];
            for (uint32_t i = 0; i < cnt; i++) neigh[i] = lst[1 + i];
            locks[c.second].unlock();
            for (uint32_t i = 0; i < cnt; i++) {
                uint32_t n2 = neigh[i];
                if (visit_tags[n2] == epoch) continue;
                visit_tags[n2] = epoch;
                float d = distq(q, n2);
                if (res.size() < ef || d < res.top().first) {
                    cand.push({d, n2});
                    res.push({d, n2});
                    if (res.size() > ef) res.pop();
                }
            }
        }
        out.clear();
        out.resize(res.size());
        for (size_t i = res.size(); i-- > 0;) {
            out[i] = res.top();
            res.pop();
        }
    }

    void select_neighbors(std::vector<std::pair<float, uint32_t>> &cand, uint32_t m) {
        if (cand.size() <= m) return;
        std::vector<std::pair<float, uint32_t>> kept;
        kept.reserve(m);
        for (auto &c : cand) {
            if (kept.size() >= m) break;
            bool good = true;
            for (auto &kp : kept) {
                if (dist(c.second, kp.second) < c.first) {
                    good = false;
                    break;
                }
            }
            if (good) kept.push_back(c);
        }
        cand.swap(kept);
    }

    void link(uint64_t node, uint32_t target, int level) {
        uint32_t cap = level == 0 ? M0 : M;
        locks[node].lock();
        uint32_t *lst = list(node, level);
        if (lst[0] < cap) {
            lst[1 + lst[0]] = target;
            lst[0]++;
            locks[node].unlock();
            return;
        }
        std::vector<std::pair<float, uint32_t>> cand;
        cand.reserve(cap + 1);
        cand.push_back({dist(node, target), target});
        for (uint32_t i = 0; i < lst[0]; i++)
            cand.push_back({dist(node, lst[1 + i]), lst[1 + i]});
        std::sort(cand.begin(), cand.end());
        select_neighbors(cand, cap);
        lst[0] = (uint32_t)cand.size();
        for (size_t i = 0; i < cand.size(); i++) lst[1 + i] = cand[i].second;
        locks[node].unlock();
    }

    void insert(uint64_t node, std::vector<uint32_t> &visit_tags, uint32_t &epoch) {
        const float *q = vec(node);
        int l = levels[node];
        int el = entry_level.load(std::memory_order_acquire);
        uint64_t ep = entry.load(std::memory_order_acquire);
        if (el < 0) {
            std::lock_guard<std::mutex> lk(entry_mu);
            if (entry_level.load() < 0) {
                entry.store(node);
                entry_level.store(l);
                return;
            }
            el = entry_level.load();
            ep = entry.load();
        }
        for (int lc = el; lc > l; lc--) ep = greedy(q, ep, lc);
        std::vector<std::pair<float, uint32_t>> cand;
        for (int lc = std::min(el, l); lc >= 0; lc--) {
            epoch++;
            search_layer(q, ep, lc, efc, visit_tags, epoch, cand);
            ep = cand.empty() ? ep : cand[0].second;
            std::vector<std::pair<float, uint32_t>> sel = cand;
            select_neighbors(sel, M);
            locks[node].lock();
            uint32_t *lst = list(node, lc);
            lst[0] = (uint32_t)sel.size();
            for (size_t i = 0; i < sel.size(); i++) lst[1 + i] = sel[i].second;
            locks[node].unlock();
            for (auto &s : sel) link(s.second, (uint32_t)node, lc);
        }
        if (l > entry_level.load(std::memory_order_acquire)) {
            std::lock_guard<std::mutex> lk(entry_mu);
            if (l > entry_level.load()) {
                entry.store(node);
                entry_level.store(l);
            }
        }
    }
};

} // namespace

extern "C" int fv_builder_finish(FvBuilder *b, const char *path, int32_t vec_attnum,
                                 int M, int ef_construction, int threads,
                                 char *errbuf, size_t errlen) {
    const uint64_t n = b->count;
    if (n == 0) {
        snprintf(errbuf, errlen, "no rows to index");
        return -1;
    }
    if (n >= UINT32_MAX) {
        snprintf(errbuf, errlen, "too many rows (max %u)", UINT32_MAX - 1);
        return -1;
    }
    HnswBuild g(b->vectors.data(), b->dim, n, (uint32_t)M, (uint32_t)ef_construction);

    /* insert node 0 first to establish the entry point, then parallel */
    {
        std::vector<uint32_t> tags(n, 0);
        uint32_t epoch = 0;
        g.insert(0, tags, epoch);
    }
    int T = std::max(1, threads);
    std::atomic<uint64_t> next{1};
    ThreadPool::instance().parallel_for(T, T, [&](int) {
        std::vector<uint32_t> tags(n, 0);
        uint32_t epoch = 0;
        for (;;) {
            uint64_t i = next.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            g.insert(i, tags, epoch);
        }
    });

    /* -------- posting lists: per column, rows sorted by value (NaN last) -- */
    std::vector<std::vector<uint32_t>> prow(b->ncols);
    std::vector<std::vector<double>> pval(b->ncols);
    ThreadPool::instance().parallel_for(std::min(T, std::max(b->ncols, 1)),
                                        b->ncols, [&](int c) {
        const std::vector<double> &col = b->cols[c];
        auto &rows = prow[c];
        rows.resize(n);
        for (uint64_t i = 0; i < n; i++) rows[i] = (uint32_t)i;
        std::sort(rows.begin(), rows.end(), [&](uint32_t a, uint32_t d2) {
            double va = col[a], vb = col[d2];
            bool na = std::isnan(va), nb = std::isnan(vb);
            if (na != nb) return nb; /* NaN sorts last */
            if (na) return false;
            if (va != vb) return va < vb;
            return a < d2;
        });
        auto &vals = pval[c];
        vals.resize(n);
        for (uint64_t i = 0; i < n; i++) vals[i] = col[rows[i]];
    });

    /* ---- serialize ---- */
    FileHeader h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, FV_MAGIC, 8);
    h.version = FV_VERSION;
    h.dim = (uint32_t)b->dim;
    h.count = n;
    h.vec_attnum = vec_attnum;
    h.ncols = (uint32_t)b->ncols;
    for (int i = 0; i < b->ncols; i++) h.col_attnums[i] = b->col_attnums[i];
    h.M = (uint32_t)M;
    h.ef_construction = (uint32_t)ef_construction;
    h.max_level = (uint32_t)g.entry_level.load();
    h.entry_node = g.entry.load();

    uint64_t off = align64(sizeof(FileHeader));
    h.off_vectors = off;
    off = align64(off + n * (uint64_t)b->dim * 4);
    h.off_tids = off;
    off = align64(off + n * 8);
    h.off_cols = off;
    off = align64(off + (uint64_t)b->ncols * n * 8);
    h.off_post_vals = off;
    off = align64(off + (uint64_t)b->ncols * n * 8);
    h.off_post_rows = off;
    off = align64(off + (uint64_t)b->ncols * n * 4);
    h.off_levels = off;
    off = align64(off + n);
    h.off_l0 = off;
    off = align64(off + n * (uint64_t)g.l0_stride * 4);
    h.off_upper_offs = off;
    off = align64(off + (n + 1) * 8);

    std::vector<uint64_t> upper_offs(n + 1);
    uint64_t slots = 0;
    for (uint64_t i = 0; i < n; i++) {
        upper_offs[i] = slots;
        slots += (uint64_t)g.levels[i] * (M + 1);
    }
    upper_offs[n] = slots;
    h.off_upper = off;
    off = align64(off + slots * 4);
    h.file_size = off;

    std::string tmp = std::string(path) + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) {
        snprintf(errbuf, errlen, "cannot create %s: %s", tmp.c_str(), strerror(errno));
        return -1;
    }
    auto write_at = [&](uint64_t o, const void *p, uint64_t len) {
        return fseeko(f, (off_t)o, SEEK_SET) == 0 &&
               (len == 0 || fwrite(p, 1, len, f) == len);
    };
    bool ok = write_at(0, &h, sizeof(h)) &&
              write_at(h.off_vectors, b->vectors.data(), n * (uint64_t)b->dim * 4) &&
              write_at(h.off_tids, b->tids.data(), n * 8) &&
              write_at(h.off_levels, g.levels.data(), n) &&
              write_at(h.off_l0, g.l0.data(), n * (uint64_t)g.l0_stride * 4) &&
              write_at(h.off_upper_offs, upper_offs.data(), (n + 1) * 8);
    for (int c = 0; c < b->ncols && ok; c++)
        ok = write_at(h.off_cols + (uint64_t)c * n * 8, b->cols[c].data(), n * 8) &&
             write_at(h.off_post_vals + (uint64_t)c * n * 8, pval[c].data(), n * 8) &&
             write_at(h.off_post_rows + (uint64_t)c * n * 4, prow[c].data(), n * 4);
    if (ok) {
        ok = fseeko(f, (off_t)h.off_upper, SEEK_SET) == 0;
        for (uint64_t i = 0; i < n && ok; i++) {
            uint64_t len = (uint64_t)g.levels[i] * (M + 1) * 4;
            if (len)
                ok = fwrite(g.upper[i], 1, len, f) == len;
        }
        if (ok && fseeko(f, (off_t)h.file_size - 1, SEEK_SET) == 0) {
            char z = 0;
            ok = fwrite(&z, 1, 1, f) == 1;
        }
    }
    if (!ok || fflush(f) != 0 || fsync(fileno(f)) != 0) {
        snprintf(errbuf, errlen, "write failed: %s", strerror(errno));
        fclose(f);
        unlink(tmp.c_str());
        return -1;
    }
    fclose(f);
    if (rename(tmp.c_str(), path) != 0) {
        snprintf(errbuf, errlen, "rename failed: %s", strerror(errno));
        unlink(tmp.c_str());
        return -1;
    }
    return 0;
}

/* ================= cursors ================= */

struct FvCursor {
    virtual int next(FvHit *out, int want) = 0;
    virtual ~FvCursor() = default;
};

extern "C" int fv_cursor_next(FvCursor *cur, FvHit *out, int want) {
    return cur->next(out, want);
}
extern "C" void fv_cursor_end(FvCursor *cur) { delete cur; }

/* -------- brute force -------- */

namespace {

struct BruteCursor final : FvCursor {
    FvIndex *idx;
    std::vector<float> q;
    const FvPred *pred; /* may be null */
    int threads;
    std::vector<FvHit> results; /* ascending */
    size_t pos = 0;
    uint64_t topb = 0;
    bool complete = false;

    struct Task {
        int32_t col; /* -1: identity rows [lo,hi) */
        uint64_t lo, hi;
    };
    std::vector<Task> tasks;
    uint64_t total_rows = 0; /* rows enumerated by tasks (upper bound) */

    void build_tasks() {
        const uint64_t chunk = 65536;
        auto add_range = [&](int32_t col, uint64_t lo, uint64_t hi) {
            for (uint64_t s = lo; s < hi; s += chunk)
                tasks.push_back({col, s, std::min(hi, s + chunk)});
            total_rows += hi - lo;
        };
        if (pred && pred->has_driver) {
            for (const Span &s : pred->spans)
                add_range(s.col, s.lo, s.hi);
        } else {
            add_range(-1, 0, idx->h->count);
        }
    }

    void compute(uint64_t B) {
        topb = B;
        const int dim = idx->h->dim;
        const float *qp = q.data();
        const bool need_eval = pred && (!pred->has_driver || !pred->covers ||
                                        pred->may_dup /* dup rows re-eval is fine */);
        std::vector<std::vector<FvHit>> parts(tasks.size());
        ThreadPool::instance().parallel_for(threads, (int)tasks.size(), [&](int t) {
            const Task &task = tasks[t];
            const uint32_t *rows =
                task.col >= 0 ? idx->prows(task.col) : nullptr;
            std::priority_queue<std::pair<float, uint32_t>> heap;
            for (uint64_t i = task.lo; i < task.hi; i++) {
                uint32_t r = rows ? rows[i] : (uint32_t)i;
                if (need_eval && !pred->eval(r)) continue;
                float d = l2sq(qp, idx->vec(r), dim);
                if (heap.size() < B)
                    heap.push({d, r});
                else if (d < heap.top().first) {
                    heap.pop();
                    heap.push({d, r});
                }
            }
            auto &part = parts[t];
            part.resize(heap.size());
            for (size_t i = heap.size(); i-- > 0;) {
                part[i] = {heap.top().second, heap.top().first};
                heap.pop();
            }
        });
        results.clear();
        for (auto &p2 : parts) results.insert(results.end(), p2.begin(), p2.end());
        std::sort(results.begin(), results.end(), [](const FvHit &a, const FvHit &b) {
            return a.dist < b.dist || (a.dist == b.dist && a.node < b.node);
        });
        if (pred && pred->may_dup)
            results.erase(std::unique(results.begin(), results.end(),
                                      [](const FvHit &a, const FvHit &b) {
                                          return a.node == b.node;
                                      }),
                          results.end());
        bool truncated = false;
        for (auto &p2 : parts)
            if (p2.size() >= B) truncated = true;
        if (results.size() > B) {
            results.resize(B);
            truncated = true;
        }
        complete = !truncated || results.size() >= total_rows;
        pos = 0;
    }

    int next(FvHit *out, int want) override {
        if (topb == 0) {
            build_tasks();
            compute(std::max<uint64_t>(4 * (uint64_t)want, 64));
        }
        int got = 0;
        while (got < want) {
            if (pos >= results.size()) {
                if (complete) break;
                size_t old = pos;
                compute(topb * 4);
                pos = old; /* deterministic recompute: same prefix order */
                continue;
            }
            out[got++] = results[pos++];
        }
        return got;
    }
};

} // namespace

extern "C" FvCursor *fv_brute_begin(FvIndex *idx, const float *query,
                                    const FvPred *pred, int threads) {
    BruteCursor *c = new BruteCursor();
    c->idx = idx;
    c->q.assign(query, query + idx->h->dim);
    c->pred = pred;
    c->threads = std::max(1, threads);
    return c;
}

/* -------- GPU predicate serialization -------- */

extern "C" int fv_pred_serialize(FvIndex *idx, const FvPred *pred,
                                 FvGpuPredNode *nodes, int max_nodes,
                                 int32_t *children, int max_children,
                                 double *invals, int max_invals,
                                 FvGpuSpan *spans, int max_spans,
                                 FvGpuBatchPred *bp, uint64_t *total_rows) {
    memset(bp, 0, sizeof(*bp));
    *total_rows = idx->h->count;
    if (!pred) return 0;

    int nn = 0, nc = 0, nv = 0;
    for (size_t i = 0; i < pred->nodes.size(); i++) {
        const PredN &n = pred->nodes[i];
        if (nn >= max_nodes || nc + (int)n.children.size() > max_children ||
            nv + (int)n.values.size() > max_invals)
            return -1;
        FvGpuPredNode gn;
        gn.kind = n.kind;
        gn.col = n.col;
        gn.op = n.op;
        gn.value = n.value;
        gn.nchildren = (int32_t)n.children.size();
        gn.child0 = nc;
        gn.nvalues = (int32_t)n.values.size();
        gn.val0 = nv;
        gn.pad = 0;
        for (int32_t c : n.children) children[nc++] = c;
        for (double v : n.values) invals[nv++] = v;
        nodes[nn++] = gn;
    }
    bp->nodes = nodes;
    bp->nnodes = nn;
    bp->children = children;
    bp->nchildren = nc;
    bp->invals = invals;
    bp->ninvals = nv;
    bp->need_eval = (!pred->has_driver || !pred->covers || pred->may_dup);
    bp->dedup = pred->may_dup ? 1 : 0;
    if (pred->has_driver && (int)pred->spans.size() <= max_spans) {
        uint64_t tot = 0;
        int ns = 0;
        for (const Span &s : pred->spans) {
            spans[ns].col = s.col;
            spans[ns].lo = (uint32_t)s.lo;
            spans[ns].hi = (uint32_t)s.hi;
            tot += s.hi - s.lo;
            ns++;
        }
        bp->spans = spans;
        bp->nspans = ns;
        *total_rows = tot;
    } else if (pred->has_driver) {
        /* too many spans: full scan, predicate re-evaluated per row */
        bp->need_eval = 1;
    }
    return 0;
}

/* -------- GPU batch execution wrappers -------- */

#ifdef FV_GPU

namespace {

struct GpuConcat {
    std::vector<FvGpuPredNode> nodes;
    std::vector<int32_t> children;
    std::vector<double> invals;
    std::vector<FvGpuSpan> spans;
    std::vector<uint64_t> soff;
    std::vector<FvGpuQueryDesc> qd;
    std::vector<uint8_t> dedup;

    void build(FvIndex *idx, int nq, const FvGpuBatchPred *preds,
               const uint32_t *entries) {
        qd.resize(nq);
        dedup.resize(nq);
        for (int q = 0; q < nq; q++) {
            const FvGpuBatchPred &p = preds[q];
            FvGpuQueryDesc &d = qd[q];
            int32_t base_n = (int32_t)nodes.size();
            int32_t base_c = (int32_t)children.size();
            int32_t base_v = (int32_t)invals.size();
            d.proot = p.nnodes > 0 ? base_n : -1;
            d.need_eval = p.need_eval;
            d.entry = entries ? entries[q] : 0;
            dedup[q] = (uint8_t)p.dedup;
            for (int i = 0; i < p.nnodes; i++) {
                FvGpuPredNode gn = p.nodes[i];
                gn.child0 += base_c;
                gn.val0 += base_v;
                nodes.push_back(gn);
            }
            for (int i = 0; i < p.nchildren; i++)
                children.push_back(p.children[i] + base_n);
            invals.insert(invals.end(), p.invals, p.invals + p.ninvals);
            d.span0 = (int32_t)spans.size();
            d.nspans = p.nspans;
            d.soff0 = (int32_t)soff.size();
            if (p.nspans > 0) {
                uint64_t tot = 0;
                for (int i = 0; i < p.nspans; i++) {
                    soff.push_back(tot);
                    tot += p.spans[i].hi - p.spans[i].lo;
                    spans.push_back(p.spans[i]);
                }
                soff.push_back(tot);
                d.total = tot;
            } else {
                d.soff0 = 0;
                d.total = idx->h->count;
            }
        }
    }
};

} // namespace

#endif /* FV_GPU */

extern "C" int fv_index_gpu_ensure(FvIndex *idx, int device);

extern "C" int fv_gpu_exec_brute_batch(FvIndex *idx, int nq,
                                       const float *queries,
                                       const FvGpuBatchPred *preds,
                                       uint32_t topb, FvHit *out,
                                       uint32_t *out_cnt, uint8_t *truncated) {
#ifdef FV_GPU
    if (fv_index_gpu_ensure(idx, -1) != 0) return -1;
    GpuConcat c;
    c.build(idx, nq, preds, nullptr);
    char errbuf[256];
    int rc = fv_gpu_brute_batch(
        idx->gpu, nq, queries, c.qd.data(), c.dedup.data(), c.nodes.data(),
        (int)c.nodes.size(), c.children.data(), (int)c.children.size(),
        c.invals.data(), (int)c.invals.size(), c.spans.data(),
        (int)c.spans.size(), c.soff.data(), (int)c.soff.size(), topb, out,
        out_cnt, truncated, errbuf, sizeof(errbuf));
    if (rc < 0) fprintf(stderr, "paves: %s\n", errbuf);
    return rc;
#else
    (void)idx; (void)nq; (void)queries; (void)preds; (void)topb; (void)out;
    (void)out_cnt; (void)truncated;
    return -1;
#endif
}

extern "C" int fv_gpu_exec_graph_batch(FvIndex *idx, int nq,
                                       const float *queries,
                                       const uint32_t *entries,
                                       const FvGpuBatchPred *preds,
                                       uint32_t ef, uint64_t max_visited,
                                       FvHit *out, uint32_t *out_cnt,
                                       uint8_t *exhausted) {
#ifdef FV_GPU
    if (fv_index_gpu_ensure(idx, -1) != 0) return -1;
    GpuConcat c;
    c.build(idx, nq, preds, entries);
    char errbuf[256];
    int rc = fv_gpu_graph_batch(
        idx->gpu, nq, queries, c.qd.data(), c.nodes.data(),
        (int)c.nodes.size(), c.children.data(), (int)c.children.size(),
        c.invals.data(), (int)c.invals.size(), ef, max_visited, out, out_cnt,
        exhausted, errbuf, sizeof(errbuf));
    if (rc < 0) fprintf(stderr, "paves: %s\n", errbuf);
    return rc;
#else
    (void)idx; (void)nq; (void)queries; (void)entries; (void)preds; (void)ef;
    (void)max_visited; (void)out; (void)out_cnt; (void)exhausted;
    return -1;
#endif
}

/* -------- GPU brute force cursor -------- */

#ifdef FV_GPU

namespace {

/*
 * GPU exact top-B cursor. Mirrors BruteCursor's widening protocol: compute
 * top-B, emit; if the consumer needs more and the result set was truncated,
 * recompute with B*4 (per-row distances are bitwise deterministic across
 * recomputes, so the emitted prefix order is stable). Any CUDA failure or a
 * B beyond the GPU cap falls back to draining a CPU BruteCursor; because
 * CPU and GPU accumulate float distances in different orders, the fallback
 * filters out already-emitted nodes instead of assuming a stable prefix.
 */
struct GpuBruteCursor final : FvCursor {
    FvIndex *idx;
    std::vector<float> q;
    const FvPred *pred; /* may be null */
    int fallback_threads;

    std::vector<FvGpuPredNode> gnodes;
    std::vector<int32_t> gchildren;
    std::vector<double> ginvals;
    std::vector<FvGpuSpan> gspans;
    FvGpuBatchPred bp;
    uint64_t total_rows = 0;

    std::vector<FvHit> results; /* ascending */
    size_t pos = 0;
    uint64_t topb = 0;
    bool complete = false;
    bool on_cpu = false;
    std::vector<uint32_t> emitted;

    void prepare() {
        size_t nn = 0, nc = 0, nv = 0, ns = 0;
        if (pred) {
            nn = pred->nodes.size();
            for (const PredN &n : pred->nodes) {
                nc += n.children.size();
                nv += n.values.size();
            }
            ns = std::min(pred->spans.size(), (size_t)FV_GPU_MAX_SPANS);
        }
        gnodes.resize(nn);
        gchildren.resize(nc);
        ginvals.resize(nv);
        gspans.resize(ns);
        fv_pred_serialize(idx, pred, gnodes.data(), (int)nn, gchildren.data(),
                          (int)nc, ginvals.data(), (int)nv, gspans.data(),
                          (int)ns, &bp, &total_rows);
    }

    void fallback_cpu() {
        on_cpu = true;
        FvCursor *c = fv_brute_begin(idx, q.data(), pred, fallback_threads);
        results.clear();
        FvHit buf[256];
        for (;;) {
            int n = fv_cursor_next(c, buf, 256);
            if (n == 0) break;
            results.insert(results.end(), buf, buf + n);
        }
        fv_cursor_end(c);
        if (!emitted.empty()) {
            /* skip nodes already returned before the fallback */
            std::vector<FvHit> kept;
            kept.reserve(results.size());
            for (const FvHit &h : results) {
                bool was = false;
                for (uint32_t e : emitted)
                    if (e == (uint32_t)h.node) {
                        was = true;
                        break;
                    }
                if (!was) kept.push_back(h);
            }
            results.swap(kept);
        }
        pos = 0;
        complete = true;
    }

    void compute(uint64_t B) {
        if (B > FV_GPU_MAX_TOPB) {
            fallback_cpu();
            return;
        }
        uint32_t cnt = 0;
        uint8_t trunc = 0;
        results.resize(B);
        int rc = fv_gpu_exec_brute_batch(idx, 1, q.data(), &bp, (uint32_t)B,
                                         results.data(), &cnt, &trunc);
        if (rc < 0) {
            fallback_cpu();
            return;
        }
        results.resize(cnt);
        topb = B;
        complete = !trunc;
    }

    int next(FvHit *out, int want) override {
        if (topb == 0 && !on_cpu) {
            prepare();
            compute(std::max<uint64_t>(4 * (uint64_t)want, 64));
            pos = 0;
        }
        int got = 0;
        while (got < want) {
            if (pos >= results.size()) {
                if (complete) break;
                size_t old = pos;
                compute(topb * 4);
                if (!on_cpu) {
                    if (idx->gpu != nullptr &&
                        fv_gpu_index_precision(idx->gpu) != FV_GPU_PREC_FP32) {
                        /* approximate distances: the wider run's prefix may
                         * differ — filter already-emitted nodes instead */
                        std::vector<FvHit> kept;
                        kept.reserve(results.size());
                        for (const FvHit &h : results) {
                            bool was = false;
                            for (uint32_t e : emitted)
                                if (e == (uint32_t)h.node) {
                                    was = true;
                                    break;
                                }
                            if (!was) kept.push_back(h);
                        }
                        results.swap(kept);
                        pos = 0;
                    } else {
                        pos = old; /* deterministic recompute: same prefix */
                    }
                }
                continue;
            }
            emitted.push_back((uint32_t)results[pos].node);
            out[got++] = results[pos++];
        }
        return got;
    }
};

} // namespace

#endif /* FV_GPU */

#ifndef FV_GPU
extern "C" int fv_gpu_tile_variant_qt(int dim) {
    (void)dim;
    return 0;
}
extern "C" int fv_gpu_mem_probe(uint64_t *free_bytes, uint64_t *total_bytes) {
    (void)free_bytes; (void)total_bytes;
    return -1;
}
#endif

/* Requested GPU vector precision for uploads (FV_GPU_PREC_*); set from the
 * paves.gpu_precision GUC before the first upload. Applies to uploads
 * only — an index already resident keeps its precision. */
static int fv_gpu_precision_req = 1; /* FV_GPU_PREC_SQ8 */

extern "C" void fv_engine_gpu_precision(int precision) {
    fv_gpu_precision_req = precision;
}

extern "C" int fv_engine_gpu_precision_get(void) {
    return fv_gpu_precision_req;
}

extern "C" int fv_index_gpu_ensure(FvIndex *idx, int device) {
#ifdef FV_GPU
    if (idx->gpu != nullptr) return 0;
    if (idx->gpu_failed) return -1;
    char errbuf[256];
    idx->gpu = fv_gpu_index_upload(idx->vectors, idx->cols, idx->post_rows,
                                   idx->l0, idx->l0_stride, idx->h->count,
                                   (int)idx->h->dim, (int)idx->h->ncols,
                                   device, fv_gpu_precision_req, errbuf,
                                   sizeof(errbuf));
    if (idx->gpu == nullptr) {
        fprintf(stderr, "paves: %s\n", errbuf);
        idx->gpu_failed = true;
        return -1;
    }
    return 0;
#else
    (void)idx; (void)device;
    return -1;
#endif
}

/* Unfiltered greedy descent through the upper HNSW layers: the level-0
 * entry point for a query (used to seed the GPU graph search). */
extern "C" uint32_t fv_index_graph_entry(FvIndex *idx, const float *query) {
    uint64_t ep = idx->h->entry_node;
    for (int lc = (int)idx->h->max_level; lc >= 1; lc--) {
        float d = l2sq(query, idx->vec(ep), idx->h->dim);
        bool changed = true;
        while (changed) {
            changed = false;
            const uint32_t *lst = idx->neigh_upper(ep, lc);
            uint32_t cnt = lst[0];
            for (uint32_t i = 0; i < cnt; i++) {
                float nd = l2sq(query, idx->vec(lst[1 + i]), idx->h->dim);
                if (nd < d) {
                    d = nd;
                    ep = lst[1 + i];
                    changed = true;
                }
            }
        }
    }
    return (uint32_t)ep;
}

extern "C" FvCursor *fv_gpu_brute_begin(FvIndex *idx, const float *query,
                                        const FvPred *pred, int device,
                                        int fallback_threads) {
#ifdef FV_GPU
    if (fv_index_gpu_ensure(idx, device) != 0) return nullptr;
    GpuBruteCursor *c = new GpuBruteCursor();
    c->idx = idx;
    c->q.assign(query, query + idx->h->dim);
    c->pred = pred;
    c->fallback_threads = std::max(1, fallback_threads);
    return c;
#else
    (void)idx; (void)query; (void)pred; (void)device; (void)fallback_threads;
    return nullptr;
#endif
}

/* -------- filtered HNSW -------- */

namespace {

struct HnswCursor final : FvCursor {
    FvIndex *idx;
    std::vector<float> q;
    const FvPred *pred;
    uint32_t ef;
    uint64_t max_visited;
    uint64_t visited_cnt = 0;

    using P = std::pair<float, uint32_t>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> cand; /* frontier */
    std::priority_queue<P> res; /* passing results, max-heap, cap ef */
    std::vector<FvHit> ready;   /* sorted batch to emit */
    size_t pos = 0;
    std::vector<uint32_t> emitted;
    bool exhausted = false;
    bool started = false;

    std::vector<uint32_t> *tags = nullptr;
    std::vector<uint32_t> private_tags;
    uint32_t epoch = 0;
    bool own_scratch = false;

    ~HnswCursor() override {
        if (own_scratch) idx->scratch_busy = false;
    }

    bool pass(uint32_t node) const { return !pred || pred->eval(node); }
    bool seen(uint32_t node) const { return (*tags)[node] == epoch; }
    void mark(uint32_t node) { (*tags)[node] = epoch; }

    void init_scratch() {
        if (!idx->scratch_busy) {
            if (idx->visit_tags.size() < idx->h->count)
                idx->visit_tags.assign(idx->h->count, 0);
            idx->scratch_busy = true;
            own_scratch = true;
            tags = &idx->visit_tags;
            epoch = ++idx->visit_epoch;
            if (epoch == 0) { /* wrapped */
                std::fill(tags->begin(), tags->end(), 0);
                epoch = ++idx->visit_epoch;
            }
        } else {
            private_tags.assign(idx->h->count, 0);
            tags = &private_tags;
            epoch = 1;
        }
    }

    void start() {
        started = true;
        init_scratch();
        const float *qp = q.data();
        uint64_t ep = idx->h->entry_node;
        for (int lc = (int)idx->h->max_level; lc >= 1; lc--) {
            float d = l2sq(qp, idx->vec(ep), idx->h->dim);
            bool changed = true;
            while (changed) {
                changed = false;
                const uint32_t *lst = idx->neigh_upper(ep, lc);
                uint32_t cnt = lst[0];
                for (uint32_t i = 0; i < cnt; i++) {
                    float nd = l2sq(qp, idx->vec(lst[1 + i]), idx->h->dim);
                    if (nd < d) {
                        d = nd;
                        ep = lst[1 + i];
                        changed = true;
                    }
                }
            }
        }
        float epd = l2sq(qp, idx->vec(ep), idx->h->dim);
        mark((uint32_t)ep);
        visited_cnt = 1;
        cand.push({epd, (uint32_t)ep});
        if (pass((uint32_t)ep)) res.push({epd, (uint32_t)ep});
        run_phase();
    }

    void run_phase() {
        const float *qp = q.data();
        const int dim = idx->h->dim;
        while (!cand.empty()) {
            auto c = cand.top();
            if (res.size() >= ef && c.first > res.top().first) break;
            if (visited_cnt >= max_visited) {
                exhausted = true;
                break;
            }
            cand.pop();
            const uint32_t *lst = idx->neigh0(c.second);
            uint32_t cnt = lst[0];
            for (uint32_t i = 0; i < cnt; i++)
                _mm_prefetch((const char *)idx->vec(lst[1 + i]), _MM_HINT_T0);
            for (uint32_t i = 0; i < cnt; i++) {
                uint32_t n2 = lst[1 + i];
                if (seen(n2)) continue;
                mark(n2);
                visited_cnt++;
                float d = l2sq(qp, idx->vec(n2), dim);
                if (res.size() < ef || d < res.top().first) {
                    cand.push({d, n2});
                    if (pass(n2)) {
                        res.push({d, n2});
                        if (res.size() > ef) res.pop();
                    }
                }
            }
        }
        if (cand.empty()) exhausted = true;
        std::vector<P> all;
        {
            auto tmpres = res; /* copy: res keeps state for widening */
            all.reserve(tmpres.size());
            while (!tmpres.empty()) {
                all.push_back(tmpres.top());
                tmpres.pop();
            }
        }
        std::sort(all.begin(), all.end());
        ready.clear();
        pos = 0;
        for (auto &p2 : all) {
            bool was = false;
            for (uint32_t e : emitted)
                if (e == p2.second) {
                    was = true;
                    break;
                }
            if (!was) ready.push_back({p2.second, p2.first});
        }
    }

    int next(FvHit *out, int want) override {
        if (!started) start();
        int got = 0;
        while (got < want) {
            if (pos >= ready.size()) {
                if (exhausted) break;
                ef *= 2;
                run_phase();
                if (ready.empty() && exhausted) break;
                continue;
            }
            emitted.push_back((uint32_t)ready[pos].node);
            out[got++] = ready[pos++];
        }
        return got;
    }
};

} // namespace

/* -------- GPU filtered graph search -------- */

#ifdef FV_GPU

namespace {

/*
 * GPU graph cursor: the CPU runs the (unfiltered) upper-level greedy descent
 * to find the level-0 entry point; the GPU runs the filtered level-0 beam
 * search (fv_graph_kernel). Widening re-runs the whole search with ef*2 and
 * filters out already-emitted nodes (the GPU search is not resumable). CUDA
 * errors or ef beyond the shared-memory bound fall back to draining a CPU
 * HNSW cursor.
 */
struct GpuHnswCursor final : FvCursor {
    FvIndex *idx;
    std::vector<float> q;
    const FvPred *pred; /* may be null */
    uint32_t ef;
    uint64_t max_visited;
    int fallback_threads;

    std::vector<FvGpuPredNode> gnodes;
    std::vector<int32_t> gchildren;
    std::vector<double> ginvals;
    std::vector<FvGpuSpan> gspans;
    FvGpuBatchPred bp;
    uint64_t total_rows = 0;

    uint32_t entry = 0;
    std::vector<FvHit> ready;
    size_t pos = 0;
    std::vector<uint32_t> emitted;
    bool exhausted = false;
    bool started = false;
    bool on_cpu = false;

    void prepare() {
        size_t nn = 0, nc = 0, nv = 0, ns = 0;
        if (pred) {
            nn = pred->nodes.size();
            for (const PredN &n : pred->nodes) {
                nc += n.children.size();
                nv += n.values.size();
            }
            ns = std::min(pred->spans.size(), (size_t)FV_GPU_MAX_SPANS);
        }
        gnodes.resize(nn);
        gchildren.resize(nc);
        ginvals.resize(nv);
        gspans.resize(ns);
        fv_pred_serialize(idx, pred, gnodes.data(), (int)nn, gchildren.data(),
                          (int)nc, ginvals.data(), (int)nv, gspans.data(),
                          (int)ns, &bp, &total_rows);
    }

    void refill_ready(const FvHit *hits, uint32_t cnt) {
        ready.clear();
        pos = 0;
        for (uint32_t i = 0; i < cnt; i++) {
            bool was = false;
            for (uint32_t e : emitted)
                if (e == (uint32_t)hits[i].node) {
                    was = true;
                    break;
                }
            if (!was) ready.push_back(hits[i]);
        }
    }

    void fallback_cpu() {
        on_cpu = true;
        FvCursor *c = fv_hnsw_begin(idx, q.data(), pred, (int)ef, max_visited);
        std::vector<FvHit> all;
        FvHit buf[256];
        for (;;) {
            int n = fv_cursor_next(c, buf, 256);
            if (n == 0) break;
            all.insert(all.end(), buf, buf + n);
        }
        fv_cursor_end(c);
        refill_ready(all.data(), (uint32_t)all.size());
        exhausted = true;
    }

    void run(uint32_t ef_now) {
        std::vector<FvHit> res(ef_now);
        uint32_t cnt = 0;
        uint8_t exh = 0;
        int rc = fv_gpu_exec_graph_batch(idx, 1, q.data(), &entry, &bp,
                                         ef_now, max_visited, res.data(),
                                         &cnt, &exh);
        if (rc < 0) {
            fallback_cpu();
            return;
        }
        ef = ef_now;
        exhausted = exh != 0;
        refill_ready(res.data(), cnt);
    }

    int next(FvHit *out, int want) override {
        if (!started) {
            started = true;
            prepare();
            entry = fv_index_graph_entry(idx, q.data());
            run(ef);
        }
        int got = 0;
        while (got < want) {
            if (pos >= ready.size()) {
                if (exhausted) break;
                uint64_t nef = (uint64_t)ef * 2;
                if (nef > FV_GPU_MAX_EF) {
                    fallback_cpu();
                    continue;
                }
                run((uint32_t)nef);
                continue;
            }
            emitted.push_back((uint32_t)ready[pos].node);
            out[got++] = ready[pos++];
        }
        return got;
    }
};

} // namespace

#endif /* FV_GPU */

extern "C" FvCursor *fv_gpu_hnsw_begin(FvIndex *idx, const float *query,
                                       const FvPred *pred, int ef,
                                       uint64_t max_visited, int device,
                                       int fallback_threads) {
#ifdef FV_GPU
    if (ef < 1 || ef > FV_GPU_MAX_EF) return nullptr;
    if (fv_index_gpu_ensure(idx, device) != 0) return nullptr;
    GpuHnswCursor *c = new GpuHnswCursor();
    c->idx = idx;
    c->q.assign(query, query + idx->h->dim);
    c->pred = pred;
    c->ef = (uint32_t)ef;
    c->max_visited = max_visited ? max_visited
                                 : std::max<uint64_t>(200000, (uint64_t)ef * 400);
    c->fallback_threads = std::max(1, fallback_threads);
    return c;
#else
    (void)idx; (void)query; (void)pred; (void)ef; (void)max_visited;
    (void)device; (void)fallback_threads;
    return nullptr;
#endif
}

extern "C" FvCursor *fv_hnsw_begin(FvIndex *idx, const float *query,
                                   const FvPred *pred, int ef,
                                   uint64_t max_visited) {
    HnswCursor *c = new HnswCursor();
    c->idx = idx;
    c->q.assign(query, query + idx->h->dim);
    c->pred = pred;
    c->ef = (uint32_t)std::max(ef, 1);
    c->max_visited = max_visited ? max_visited
                                 : std::max<uint64_t>(200000, (uint64_t)ef * 400);
    return c;
}
