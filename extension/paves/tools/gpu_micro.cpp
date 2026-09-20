/*
 * gpu_micro — operator-level saturation microbenchmark for the paves GPU
 * engine. Bypasses PostgreSQL entirely: mmaps an index file, synthesizes an
 * unbounded query stream (index rows reused as query vectors, predicate
 * constants derived from posting-value percentiles for a target
 * selectivity) and drives fv_gpu_exec_{brute,graph}_batch back-to-back.
 *
 * This measures the kernel-side throughput ceiling as a function of batch
 * size — the "queries arrive infinitely fast" limit — with zero arbiter,
 * gathering-window or backend overhead. FV_GPU_PROF / FV_GPU_GDBG work as
 * in the server.
 *
 *   gpu_micro <index.idx> [--strategy graph|brute] [--sel 0.10]
 *             [--col 0] [--ef 200] [--topb 64] [--batch 32,64,...]
 *             [--secs 3] [--precision sq8|fp32] [--sort-work 0|1]
 *
 * Output: one JSON line per batch size with QPS and per-batch latency.
 */
#include "../src/engine/engine_api.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <index.idx> [options]\n", argv[0]);
        return 1;
    }
    const char *path = argv[1];
    std::string strategy = "graph";
    std::string qfile; /* raw float32 query vectors; index rows otherwise */
    double sel = 0.10;
    int col = 0, ef = 200, topb = 64, secs = 3, precision = 1;
    std::vector<int> batches = {16, 32, 64, 128, 256, 512, 1024};
    for (int a = 2; a < argc; a++) {
        std::string k = argv[a];
        auto next = [&]() { return std::string(argv[++a]); };
        if (k == "--strategy") strategy = next();
        else if (k == "--queries") qfile = next();
        else if (k == "--sel") sel = atof(next().c_str());
        else if (k == "--col") col = atoi(next().c_str());
        else if (k == "--ef") ef = atoi(next().c_str());
        else if (k == "--topb") topb = atoi(next().c_str());
        else if (k == "--secs") secs = atoi(next().c_str());
        else if (k == "--precision") precision = next() == "fp32" ? 0 : 1;
        else if (k == "--batch") {
            batches.clear();
            std::string v = next();
            for (size_t p = 0; p < v.size();) {
                size_t c = v.find(',', p);
                if (c == std::string::npos) c = v.size();
                batches.push_back(atoi(v.substr(p, c - p).c_str()));
                p = c + 1;
            }
        }
    }

    char errbuf[256];
    FvIndex *idx = fv_index_open(path, errbuf, sizeof(errbuf));
    if (!idx) {
        fprintf(stderr, "open failed: %s\n", errbuf);
        return 1;
    }
    const uint64_t count = fv_index_count(idx);
    const int dim = fv_index_dim(idx);
    fv_engine_gpu_precision(precision);
    if (fv_index_gpu_ensure(idx, -1) != 0) {
        fprintf(stderr, "gpu ensure failed\n");
        return 1;
    }

    /* predicate: col < percentile(sel) — one compiled pred shared by all
     * queries (the serialized slices are concatenated per query anyway) */
    FvPred *pred = nullptr;
    FvGpuBatchPred bp;
    memset(&bp, 0, sizeof(bp));
    std::vector<FvGpuPredNode> gnodes(8);
    std::vector<int32_t> gchildren(8);
    std::vector<double> ginvals(8);
    std::vector<FvGpuSpan> gspans(4096);
    uint64_t total_rows = count;
    if (sel < 1.0) {
        double thresh = fv_index_col_percentile(idx, col, sel);
        FvPredNode leaf;
        memset(&leaf, 0, sizeof(leaf));
        leaf.kind = FV_NODE_CMP;
        leaf.col = col; /* NB: engine slot, caller must pass a valid slot */
        leaf.op = FV_CMP_LT;
        leaf.value = thresh;
        pred = fv_pred_compile(idx, &leaf); /* col is the engine slot */
        if (!pred) {
            fprintf(stderr, "pred compile failed\n");
            return 1;
        }
        if (fv_pred_serialize(idx, pred, gnodes.data(), (int)gnodes.size(),
                              gchildren.data(), (int)gchildren.size(),
                              ginvals.data(), (int)ginvals.size(),
                              gspans.data(), (int)gspans.size(), &bp,
                              &total_rows) != 0) {
            fprintf(stderr, "pred serialize failed\n");
            return 1;
        }
    }

    /* traversal budget, mirroring the executor (0 would stop the kernel
     * after its first iteration — the GPU batch API has no "default") */
    const uint64_t max_visited =
        std::max<uint64_t>(200000, (uint64_t)ef * 400);

    const int maxb = *std::max_element(batches.begin(), batches.end());
    std::vector<float> queries((size_t)maxb * dim);
    std::vector<uint32_t> entries(maxb);
    std::vector<FvGpuBatchPred> preds(maxb, bp);
    std::vector<FvHit> out((size_t)maxb * (strategy == "graph" ? ef : topb));
    std::vector<uint32_t> cnts(maxb);
    std::vector<uint8_t> flags(maxb);

    /* Query pool. Real held-out queries whenever possible: reusing index
     * rows makes graph search pathologically easy (the beam converges on
     * the exact-match row almost immediately), inflating QPS 20-30x. */
    if (!qfile.empty()) {
        FILE *f = fopen(qfile.c_str(), "rb");
        if (!f) {
            fprintf(stderr, "cannot open %s\n", qfile.c_str());
            return 1;
        }
        fseek(f, 0, SEEK_END);
        long nq_avail = ftell(f) / ((long)dim * 4);
        for (int i = 0; i < maxb; i++) {
            long q = i % nq_avail;
            fseek(f, q * (long)dim * 4, SEEK_SET);
            if (fread(&queries[(size_t)i * dim], 4, dim, f) != (size_t)dim) {
                fprintf(stderr, "short read in %s\n", qfile.c_str());
                return 1;
            }
        }
        fclose(f);
    } else {
        for (int i = 0; i < maxb; i++) {
            uint64_t row = ((uint64_t)i * 2654435761u + 12345) % count;
            memcpy(&queries[(size_t)i * dim], fv_index_vector(idx, row),
                   sizeof(float) * dim);
        }
    }
    if (strategy == "graph")
        for (int i = 0; i < maxb; i++)
            entries[i] = fv_index_graph_entry(idx, &queries[(size_t)i * dim]);

    printf("{\"index\":\"%s\",\"count\":%llu,\"dim\":%d,\"strategy\":\"%s\","
           "\"sel\":%.4f,\"driver_rows\":%llu,\"ef\":%d,\"topb\":%d,"
           "\"precision\":%d}\n",
           path, (unsigned long long)count, dim, strategy.c_str(), sel,
           (unsigned long long)total_rows, ef, topb, precision);

    for (int nb : batches) {
        /* warmup */
        for (int w = 0; w < 2; w++) {
            if (strategy == "graph")
                fv_gpu_exec_graph_batch(idx, nb, queries.data(),
                                        entries.data(), preds.data(),
                                        (uint32_t)ef, max_visited,
                                        out.data(), cnts.data(),
                                        flags.data());
            else
                fv_gpu_exec_brute_batch(idx, nb, queries.data(), preds.data(),
                                        (uint32_t)topb, out.data(),
                                        cnts.data(), flags.data());
        }
        double t0 = now_s(), t1 = t0;
        long iters = 0;
        while ((t1 = now_s()) - t0 < (double)secs) {
            int rc;
            if (strategy == "graph")
                rc = fv_gpu_exec_graph_batch(idx, nb, queries.data(),
                                             entries.data(), preds.data(),
                                             (uint32_t)ef, max_visited,
                                             out.data(), cnts.data(),
                                             flags.data());
            else
                rc = fv_gpu_exec_brute_batch(idx, nb, queries.data(),
                                             preds.data(), (uint32_t)topb,
                                             out.data(), cnts.data(),
                                             flags.data());
            if (rc != 0) {
                fprintf(stderr, "batch failed rc=%d\n", rc);
                return 1;
            }
            iters++;
        }
        double dt = t1 - t0;
        printf("{\"batch\":%d,\"iters\":%ld,\"qps\":%.1f,"
               "\"batch_ms\":%.3f}\n",
               nb, iters, (double)nb * iters / dt, dt * 1e3 / iters);
        fflush(stdout);
    }
    return 0;
}
