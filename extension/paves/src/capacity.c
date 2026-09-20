/*
 * capacity.c — paves_capacity(regclass): read-only pre-flight check.
 *
 * Building an index and uploading it to a GPU are minutes-long, tens-of-GB
 * operations, and several fast paths degrade SILENTLY when a limit is hit
 * (arbiter dim cap -> CPU, unsupported dim -> legacy kernel). This function
 * turns "try it and see" into "look it up": it estimates every relevant
 * size from the relation's statistics and reports, with numbers, what fits
 * where and which fast paths the table will get.
 *
 * GPU free memory comes from shared memory, published by the GPU worker
 * (the only process allowed to own a CUDA context); backends never probe.
 */
#include "paves.h"
#include "arbiter.h"

#include <unistd.h>

#include "access/relation.h"
#include "access/table.h"
#include "catalog/pg_type.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"

PG_FUNCTION_INFO_V1(paves_capacity);

static bool
cacheable_scalar_oid(Oid t)
{
    return t == BOOLOID || t == INT2OID || t == INT4OID || t == INT8OID ||
        t == FLOAT4OID || t == FLOAT8OID;
}

static char *
fmt_bytes(double b)
{
    if (b >= 1e12)
        return psprintf("%.2f TB", b / 1e12);
    if (b >= 1e9)
        return psprintf("%.2f GB", b / 1e9);
    if (b >= 1e6)
        return psprintf("%.1f MB", b / 1e6);
    return psprintf("%.0f KB", b / 1e3);
}

static void
emit(Tuplestorestate *ts, TupleDesc td, const char *metric,
     const char *value, const char *ok, const char *note)
{
    Datum values[4];
    bool nulls[4] = {false, false, false, false};

    values[0] = CStringGetTextDatum(metric);
    values[1] = CStringGetTextDatum(value);
    values[2] = CStringGetTextDatum(ok);
    values[3] = CStringGetTextDatum(note ? note : "");
    tuplestore_putvalues(ts, td, values, nulls);
}

Datum
paves_capacity(PG_FUNCTION_ARGS)
{
    Oid relid = PG_GETARG_OID(0);
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate *ts;
    MemoryContext oldcxt;
    Relation rel;
    double n = 0;
    int dim = 0;
    int ncols = 0;
    int nskipped = 0;
    int i;
    int M = paves_build_m;
    double sys_ram;

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR, (errmsg("set-valued function called in context that "
                               "cannot accept a set")));
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("return type must be a row type")));

    oldcxt = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    ts = tuplestore_begin_heap(true, false, work_mem);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = ts;
    rsinfo->setDesc = CreateTupleDescCopy(tupdesc);
    MemoryContextSwitchTo(oldcxt);

    rel = table_open(relid, AccessShareLock);
    n = Max(rel->rd_rel->reltuples, 0);
    for (i = 0; i < RelationGetDescr(rel)->natts; i++)
    {
        Form_pg_attribute att = TupleDescAttr(RelationGetDescr(rel), i);

        if (att->attisdropped)
            continue;
        if (att->atttypid == paves_vector_typid() && dim == 0)
            dim = att->atttypmod; /* vector typmod is the dimension */
        else if (cacheable_scalar_oid(att->atttypid))
        {
            if (ncols < FV_MAX_COLS)
                ncols++;
            else
                nskipped++;
        }
    }
    table_close(rel, AccessShareLock);

    sys_ram = (double) sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);

    emit(ts, tupdesc, "rows",
         n > 0 ? psprintf("%.0f", n) : "unknown",
         n > 0 ? "ok" : "?",
         n > 0 ? "pg_class.reltuples estimate" : "run ANALYZE first");
    emit(ts, tupdesc, "row_limit", "4294967295",
         n < 4294967295.0 ? "ok" : "EXCEEDED",
         "row ids are uint32 throughout the engine");
    emit(ts, tupdesc, "dim", psprintf("%d", dim),
         dim > 0 ? "ok" : "MISSING",
         dim > 0 ? "" : "no pgvector column found");
    emit(ts, tupdesc, "cached_scalar_columns",
         psprintf("%d%s", ncols,
                  nskipped ? psprintf(" (+%d beyond cap)", nskipped) : ""),
         nskipped == 0 ? "ok" : "PARTIAL",
         nskipped ? "columns beyond FV_MAX_COLS use executor recheck only"
                  : "");

    if (n > 0 && dim > 0)
    {
        double idx_bytes = n * (4.0 * dim + 20.0 * ncols + 8 + 1 +
                                (2.0 * M + 1) * 4 + 8 + 3);
        double dpad = (double) ((dim + 15) & ~15);
        double gpu_bytes = paves_gpu_precision == 1
            ? n * (3.0 * dpad + 4 + 12.0 * ncols + (2.0 * M + 1) * 4)
            : n * (4.0 * dim + 12.0 * ncols + (2.0 * M + 1) * 4);
        double build_peak = idx_bytes * 2.5;
        double build_secs = 9.0 * (n / 1e6) * (dim / 128.0) *
            (64.0 / Max(paves_build_threads, 1));
        uint64 gpu_free = fv_arb ? pg_atomic_read_u64(&fv_arb->gpu_free_bytes)
                                 : 0;
        uint32 gpu_devp1 = fv_arb
            ? pg_atomic_read_u32(&fv_arb->gpu_device_plus1) : 0;
        int tile_qt = fv_gpu_tile_variant_qt(dim);

        emit(ts, tupdesc, "index_size_est", fmt_bytes(idx_bytes),
             idx_bytes < sys_ram * 0.5 ? "ok" : "LARGE",
             "must stay well under RAM (mmap page cache)");
        emit(ts, tupdesc, "build_peak_mem", fmt_bytes(build_peak),
             build_peak < sys_ram * 0.8 ? "ok" : "WILL NOT FIT",
             psprintf("system RAM %s", fmt_bytes(sys_ram)));
        emit(ts, tupdesc, "build_disk_need", fmt_bytes(idx_bytes * 2.0),
             "?", "two index copies exist transiently during rebuild");
        emit(ts, tupdesc, "build_time_est",
             psprintf("%.0f s", build_secs), "?",
             psprintf("calibrated on SIFT1M, %d threads",
                      paves_build_threads));
        emit(ts, tupdesc, "per_backend_visited",
             fmt_bytes(n * 4.0), "?",
             "CPU HNSW scratch, multiplied by connection count");

        if (gpu_devp1 > 0 && gpu_free > 0)
            emit(ts, tupdesc, "gpu_resident", fmt_bytes(gpu_bytes),
                 gpu_bytes + (512.0 * 1024 * 1024) < (double) gpu_free
                     ? "ok" : "WILL NOT FIT",
                 psprintf("device %u has %s free (worker-reported)",
                          gpu_devp1 - 1, fmt_bytes((double) gpu_free)));
        else
            emit(ts, tupdesc, "gpu_resident", fmt_bytes(gpu_bytes), "?",
                 "GPU worker not running or no usable device");

        emit(ts, tupdesc, "gpu_fast_brute",
             tile_qt > 0 ? psprintf("tiled (QT=%d)", tile_qt) : "legacy",
             tile_qt > 0 ? "ok" : "SLOW",
             tile_qt > 0 ? "" :
             "no tile variant for this dim; per-query kernel");
        emit(ts, tupdesc, "gpu_arbiter_dim",
             psprintf("%d <= %d", dim, FV_ARB_MAX_DIM),
             dim <= FV_ARB_MAX_DIM ? "ok" : "EXCEEDED",
             dim <= FV_ARB_MAX_DIM ? "" :
             "GPU strategies silently run on the CPU");
    }

    return (Datum) 0;
}

/*
 * paves_feedback() — dump the cost model's runtime feedback tables:
 * one row per (index, strategy-side, selectivity bucket) with the GPU and
 * CPU EWMAs the planner reads, plus the live load counters. Debugging /
 * calibration aid for the auto router.
 */
PG_FUNCTION_INFO_V1(paves_feedback);

Datum
paves_feedback(PG_FUNCTION_ARGS)
{
    ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
    TupleDesc tupdesc;
    Tuplestorestate *ts;
    MemoryContext oldcxt;
    int e, s, b;

    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo))
        ereport(ERROR, (errmsg("set-valued function called in context that "
                               "cannot accept a set")));
    if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
        ereport(ERROR, (errmsg("return type must be a row type")));

    oldcxt = MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    ts = tuplestore_begin_heap(true, false, work_mem);
    rsinfo->returnMode = SFRM_Materialize;
    rsinfo->setResult = ts;
    rsinfo->setDesc = CreateTupleDescCopy(tupdesc);
    MemoryContextSwitchTo(oldcxt);

    if (fv_arb == NULL)
        return (Datum) 0;

    for (e = 0; e < FV_FB_INDEXES; e++)
    {
        FvFeedbackEntry *fb = &fv_arb->fb[e];
        uint32 key = pg_atomic_read_u32(&fb->relid_key);

        if (key == 0)
            continue;
        for (s = 0; s < 2; s++)
        {
            for (b = 0; b < FV_SEL_BUCKETS; b++)
            {
                Datum values[12];
                bool nulls[12] = {false};

                values[0] = ObjectIdGetDatum(fb->dbid);
                values[1] = ObjectIdGetDatum((Oid) key);
                values[2] = CStringGetTextDatum(s ? "graph" : "brute");
                values[3] = Int32GetDatum(b);
                values[4] = Float8GetDatum(fv_sel_bucket_mid(b));
                values[5] = Int64GetDatum(
                    (int64) pg_atomic_read_u32(&fb->gpu_q_us[s][b]));
                values[6] = Int64GetDatum(
                    (int64) pg_atomic_read_u32(&fb->gpu_batch_us[s]));
                values[7] = Float8GetDatum(
                    pg_atomic_read_u32(&fb->gpu_batch_n16[s]) / 16.0);
                values[8] = Int64GetDatum(
                    (int64) pg_atomic_read_u32(&fb->cpu_q_us[s][b]));
                values[9] = Float8GetDatum(
                    pg_atomic_read_u32(&fb->cpu_ref_active16[s]) / 16.0);
                values[10] = Int32GetDatum(
                    (int32) pg_atomic_read_u32(&fv_arb->gpu_inflight));
                values[11] = Int32GetDatum(
                    (int32) pg_atomic_read_u32(&fv_arb->cpu_active));
                tuplestore_putvalues(ts, tupdesc, values, nulls);
            }
        }
    }
    return (Datum) 0;
}
