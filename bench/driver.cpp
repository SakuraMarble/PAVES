// Concurrent QPS driver for vector search benchmarks (libpq).
//
// Each client thread owns one connection, applies session SETs, then loops
// over a query file (one SQL statement per line), starting at a per-thread
// offset. Reports QPS and latency percentiles as JSON on stdout.
//
// Build: g++ -O2 -std=c++17 driver.cpp -o driver \
//        -I$(pg_config --includedir) -L$(pg_config --libdir) -lpq -lpthread
//
// Usage: ./driver --conninfo "host=... port=... dbname=..." --queries file.sql \
//                 --clients 64 --duration 10 --warmup 2 [--set "a=b" --set "c=d"]

#include <libpq-fe.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

struct Args {
    std::string conninfo;
    std::string queries_file;
    std::vector<std::string> sets;
    int clients = 16;
    double duration = 10.0;
    double warmup = 2.0;
};

static void die(const std::string& msg) {
    fprintf(stderr, "driver: %s\n", msg.c_str());
    exit(1);
}

struct ThreadResult {
    long completed = 0;
    long errors = 0;
    std::vector<float> lat_ms;  // measured-phase latencies
};

int main(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (++i >= argc) die("missing value for " + a);
            return argv[i];
        };
        if (a == "--conninfo") args.conninfo = next();
        else if (a == "--queries") args.queries_file = next();
        else if (a == "--clients") args.clients = atoi(next().c_str());
        else if (a == "--duration") args.duration = atof(next().c_str());
        else if (a == "--warmup") args.warmup = atof(next().c_str());
        else if (a == "--set") args.sets.push_back(next());
        else die("unknown arg: " + a);
    }
    if (args.clients < 1 || args.clients > 512 || args.duration <= 0 || args.warmup < 0) die("invalid clients/duration/warmup");
    if (args.conninfo.empty() || args.queries_file.empty()) die("--conninfo and --queries required");

    std::vector<std::string> queries;
    {
        std::ifstream in(args.queries_file);
        if (!in) die("cannot open " + args.queries_file);
        std::string line;
        while (std::getline(in, line))
            if (!line.empty() && line[0] != '#') queries.push_back(line);
    }
    if (queries.empty()) die("no queries");

    std::atomic<bool> measuring{false}, stop{false};
    std::vector<ThreadResult> results(args.clients);
    std::atomic<int> ready{0};

    auto worker = [&](int tid) {
        PGconn* conn = PQconnectdb(args.conninfo.c_str());
        if (PQstatus(conn) != CONNECTION_OK) {
            fprintf(stderr, "conn failed: %s\n", PQerrorMessage(conn));
            results[tid].errors = -1;
            PQfinish(conn);
            ready++;
            return;
        }
        for (auto& s : args.sets) {
            std::string sql = "SET " + s;
            PGresult* r = PQexec(conn, sql.c_str());
            if (PQresultStatus(r) != PGRES_COMMAND_OK) {
                fprintf(stderr, "SET failed: %s\n", PQerrorMessage(conn));
                PQclear(r);
                PQfinish(conn);
                results[tid].errors = -1;
                ready++;
                return;
            }
            PQclear(r);
        }
        ready++;
        size_t idx = (queries.size() * tid) / args.clients;  // stagger start offsets
        ThreadResult& res = results[tid];
        while (!stop.load(std::memory_order_relaxed)) {
            const std::string& q = queries[idx];
            idx = (idx + 1) % queries.size();
            bool began_measured = measuring.load(std::memory_order_relaxed);
            auto t0 = Clock::now();
            PGresult* r = PQexec(conn, q.c_str());
            auto t1 = Clock::now();
            bool ok = PQresultStatus(r) == PGRES_TUPLES_OK;
            if (!ok && res.errors == 0)
                fprintf(stderr, "query error: %s\n", PQerrorMessage(conn));
            PQclear(r);
            if (!ok) res.errors++;
            if (began_measured && measuring.load(std::memory_order_relaxed)) {
                if (ok) {
                    res.completed++;
                    res.lat_ms.push_back(
                        std::chrono::duration<float, std::milli>(t1 - t0).count());
                }
            }
        }
        PQfinish(conn);
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < args.clients; i++) threads.emplace_back(worker, i);
    while (ready.load() < args.clients) std::this_thread::sleep_for(std::chrono::milliseconds(10));

    std::this_thread::sleep_for(std::chrono::duration<double>(args.warmup));
    auto m0 = Clock::now();
    measuring = true;
    std::this_thread::sleep_for(std::chrono::duration<double>(args.duration));
    measuring = false;
    auto m1 = Clock::now();
    stop = true;
    for (auto& t : threads) t.join();

    double elapsed = std::chrono::duration<double>(m1 - m0).count();
    long completed = 0, errors = 0;
    std::vector<float> lats;
    for (auto& r : results) {
        if (r.errors < 0) die("a client failed to connect");
        completed += r.completed;
        errors += r.errors;
        lats.insert(lats.end(), r.lat_ms.begin(), r.lat_ms.end());
    }
    std::sort(lats.begin(), lats.end());
    auto pct = [&](double p) -> double {
        if (lats.empty()) return 0;
        size_t i = std::min(lats.size() - 1, (size_t)(p * lats.size()));
        return lats[i];
    };
    double mean = 0;
    for (float v : lats) mean += v;
    if (!lats.empty()) mean /= lats.size();

    printf("{\"clients\": %d, \"elapsed_s\": %.3f, \"completed\": %ld, \"errors\": %ld, "
           "\"qps\": %.1f, \"lat_ms\": {\"mean\": %.3f, \"p50\": %.3f, \"p95\": %.3f, \"p99\": %.3f}}\n",
           args.clients, elapsed, completed, errors, completed / elapsed,
           mean, pct(0.50), pct(0.95), pct(0.99));
    return errors != 0 || completed == 0 ? 2 : 0;  // Any error invalidates the run.
}
