/* paves.c — extension entry point: GUCs, hook and method registration. */
#include "paves.h"
#include "arbiter.h"

#include "fmgr.h"
#include "miscadmin.h"
#include "utils/guc.h"

PG_MODULE_MAGIC;

bool paves_enable = true;
int paves_ef_search = 100;
int paves_threads = 8;
int paves_force_strategy = FV_FORCE_AUTO;
int paves_build_m = 16;
int paves_build_ef = 200;
int paves_build_threads = 64;
double paves_cost_scale = 0.01;
int paves_gpu_device = -1;
int paves_gpu_mode = FV_GPU_MODE_WORKER;
int paves_gpu_batch_max = 64;
int paves_gpu_batch_wait_us = 200;
int paves_gpu_batch_idle_us = 40;
int paves_cpu_cores = 384;
int paves_gpu_precision = 1; /* sq8 */
bool paves_gpu_enable = true;

static const struct config_enum_entry gpu_mode_options[] = {
    {"worker", FV_GPU_MODE_WORKER, false},
    {"direct", FV_GPU_MODE_DIRECT, false},
    {NULL, 0, false},
};

static const struct config_enum_entry gpu_precision_options[] = {
    {"fp32", 0, false},
    {"sq8", 1, false},
    {NULL, 0, false},
};

static void
gpu_precision_assign(int newval, void *extra)
{
    fv_engine_gpu_precision(newval);
}

static const struct config_enum_entry force_strategy_options[] = {
    {"auto", FV_FORCE_AUTO, false},
    {"brute", FV_FORCE_BRUTE, false},
    {"hnsw", FV_FORCE_HNSW, false},
    {"gpu_brute", FV_FORCE_GPU_BRUTE, false},
    {"gpu_hnsw", FV_FORCE_GPU_HNSW, false},
    {NULL, 0, false},
};

void _PG_init(void);

void
_PG_init(void)
{
    DefineCustomBoolVariable("paves.enable",
                             "Enable paves custom scan paths.",
                             NULL, &paves_enable, true,
                             PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.ef_search",
                            "HNSW search beam width (adapted upward by selectivity).",
                            NULL, &paves_ef_search, 100, 10, 10000,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.threads",
                            "Worker threads per query for engine execution.",
                            NULL, &paves_threads, 8, 1, 256,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomEnumVariable("paves.force_strategy",
                             "Force a specific search strategy "
                             "(auto/brute/hnsw/gpu_brute/gpu_hnsw).",
                             NULL, &paves_force_strategy, FV_FORCE_AUTO,
                             force_strategy_options,
                             PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.gpu_device",
                            "CUDA device for GPU strategies "
                            "(-1 = auto-pick by free memory).",
                            NULL, &paves_gpu_device, -1, -1, 15,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomEnumVariable("paves.gpu_mode",
                             "GPU execution mode: worker (micro-batch "
                             "arbiter) or direct (backend-local CUDA).",
                             NULL, &paves_gpu_mode, FV_GPU_MODE_WORKER,
                             gpu_mode_options,
                             PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.gpu_batch_max",
                            "Max requests per GPU micro-batch.",
                            NULL, &paves_gpu_batch_max, 64, 1,
                            FV_ARB_SLOTS,
                            PGC_SIGHUP, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.gpu_batch_wait_us",
                            "Micro-batch gathering window in microseconds.",
                            NULL, &paves_gpu_batch_wait_us, 200, 0, 20000,
                            PGC_SIGHUP, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.gpu_batch_idle_us",
                            "Close the gathering window early after this "
                            "many microseconds without a new arrival "
                            "(0 = always wait the full window).",
                            NULL, &paves_gpu_batch_idle_us, 40, 0, 20000,
                            PGC_SIGHUP, 0, NULL, NULL, NULL);
    DefineCustomBoolVariable("paves.gpu_enable",
                             "Let auto routing consider GPU strategies "
                             "(off = CPU-only auto; forced gpu_* strategies "
                             "are unaffected).",
                             NULL, &paves_gpu_enable, true,
                             PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomEnumVariable("paves.gpu_precision",
                             "GPU-resident vector precision: fp32 (exact) "
                             "or sq8 (int8 scan + fp16 rerank, ~4x less "
                             "memory/bandwidth, approximate). Takes effect "
                             "at index upload (worker restart).",
                             NULL, &paves_gpu_precision, 1,
                             gpu_precision_options,
                             PGC_SIGHUP, 0, NULL, gpu_precision_assign, NULL);
    DefineCustomIntVariable("paves.cpu_cores",
                            "CPU cores assumed by the saturation term of "
                            "the device-aware cost model.",
                            NULL, &paves_cpu_cores, 384, 1, 4096,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.build_m",
                            "HNSW graph degree (M) used by paves_build().",
                            NULL, &paves_build_m, 16, 4, 64,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.build_ef_construction",
                            "HNSW ef_construction used by paves_build().",
                            NULL, &paves_build_ef, 200, 8, 2000,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomIntVariable("paves.build_threads",
                            "Threads used by paves_build().",
                            NULL, &paves_build_threads, 64, 1, 384,
                            PGC_USERSET, 0, NULL, NULL, NULL);
    DefineCustomRealVariable("paves.cost_scale",
                             "Multiplier applied to paves path costs "
                             "(keeps brute/hnsw ratio, undercuts other plans).",
                             NULL, &paves_cost_scale, 0.01, 1e-6, 1e6,
                             PGC_USERSET, 0, NULL, NULL, NULL);

    paves_register_planner_hooks();
    RegisterCustomScanMethods(&paves_scan_methods);

    if (process_shared_preload_libraries_in_progress)
        paves_arbiter_register();
}
