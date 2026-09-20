/*
 * build.c — paves_build(regclass): scan the heap, build the engine index
 * file (vectors + HNSW graph + columnar scalar cache + TIDs), atomically
 * replacing any previous file.
 */
#include "paves.h"

#include <sys/stat.h>

#include "access/table.h"
#include "access/tableam.h"
#include "catalog/pg_type.h"
#include "fmgr.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "storage/itemptr.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"

PG_FUNCTION_INFO_V1(paves_build);

static bool
cacheable_scalar_type(Oid typid)
{
    return typid == BOOLOID || typid == INT2OID || typid == INT4OID ||
           typid == INT8OID || typid == FLOAT4OID || typid == FLOAT8OID;
}

static double
scalar_datum_to_double(Oid typid, Datum d)
{
    switch (typid)
    {
        case BOOLOID: return DatumGetBool(d) ? 1.0 : 0.0;
        case INT2OID: return (double) DatumGetInt16(d);
        case INT4OID: return (double) DatumGetInt32(d);
        case INT8OID: return (double) DatumGetInt64(d);
        case FLOAT4OID: return (double) DatumGetFloat4(d);
        case FLOAT8OID: return DatumGetFloat8(d);
        default: return 0.0;
    }
}

Datum
paves_build(PG_FUNCTION_ARGS)
{
    Oid relid = PG_GETARG_OID(0);
    Relation rel;
    TupleDesc tupdesc;
    Oid vectypid = paves_vector_typid();
    int vec_attnum = 0;
    int dim = -1;
    int ncols = 0;
    int32 col_attnums[FV_MAX_COLS];
    Oid col_types[FV_MAX_COLS];
    int i;
    TableScanDesc scan;
    TupleTableSlot *slot;
    FvBuilder *builder;
    uint64 count = 0;
    char *path;
    char *dir;
    char errbuf[512];
    float *vecbuf = NULL;
    int vecbuf_dim = 0;

    if (!OidIsValid(vectypid))
        ereport(ERROR, (errmsg("paves: pgvector extension not installed "
                               "(type \"vector\" not found)")));

    rel = table_open(relid, AccessShareLock);
    tupdesc = RelationGetDescr(rel);

    for (i = 0; i < tupdesc->natts; i++)
    {
        Form_pg_attribute att = TupleDescAttr(tupdesc, i);

        if (att->attisdropped)
            continue;
        if (att->atttypid == vectypid && vec_attnum == 0)
        {
            vec_attnum = att->attnum;
            if (att->atttypmod > 0)
                dim = att->atttypmod;
        }
        else if (cacheable_scalar_type(att->atttypid) && ncols < FV_MAX_COLS)
        {
            col_attnums[ncols] = att->attnum;
            col_types[ncols] = att->atttypid;
            ncols++;
        }
    }
    if (vec_attnum == 0)
        ereport(ERROR, (errmsg("paves: relation \"%s\" has no vector column",
                               RelationGetRelationName(rel))));

    builder = fv_builder_create(dim > 0 ? dim : 0, ncols, col_attnums,
                                (uint64) Max(rel->rd_rel->reltuples, 1024));
    if (builder == NULL)
        ereport(ERROR, (errmsg("paves: builder creation failed")));

    slot = table_slot_create(rel, NULL);
    scan = table_beginscan(rel, GetActiveSnapshot(), 0, NULL);
    while (table_scan_getnextslot(scan, ForwardScanDirection, slot))
    {
        Datum vdatum;
        bool visnull;
        PgvVector *vec;
        double scalars[FV_MAX_COLS];
        uint8 isnull[FV_MAX_COLS];
        uint64 packed_tid;
        int c;

        if ((count & 0x1fff) == 0)
            CHECK_FOR_INTERRUPTS();

        vdatum = slot_getattr(slot, vec_attnum, &visnull);
        if (visnull)
            continue; /* NULL vectors are not indexed (they sort last anyway) */
        vec = (PgvVector *) PG_DETOAST_DATUM(vdatum);
        if (dim < 0)
        {
            /* unconstrained vector column: adopt the first row's dimension.
             * fv_builder_create got dim=0, so rebuild the builder now. */
            dim = vec->dim;
            fv_builder_destroy(builder);
            builder = fv_builder_create(dim, ncols, col_attnums,
                                        (uint64) Max(rel->rd_rel->reltuples, 1024));
        }
        if (vec->dim != dim)
            ereport(ERROR, (errmsg("paves: mixed vector dimensions "
                                   "(%d vs %d) are not supported",
                                   vec->dim, dim)));
        /* copy out: the engine keeps its own buffer; detoasted datum may be
         * palloc'd per row, keep memory flat */
        if (vecbuf_dim != dim)
        {
            vecbuf = (float *) palloc(sizeof(float) * dim);
            vecbuf_dim = dim;
        }
        memcpy(vecbuf, vec->x, sizeof(float) * dim);
        if ((Pointer) vec != DatumGetPointer(vdatum))
            pfree(vec);

        for (c = 0; c < ncols; c++)
        {
            Datum d;
            bool n;

            d = slot_getattr(slot, col_attnums[c], &n);
            isnull[c] = n ? 1 : 0;
            scalars[c] = n ? 0.0 : scalar_datum_to_double(col_types[c], d);
        }

        packed_tid = ((uint64) ItemPointerGetBlockNumber(&slot->tts_tid) << 16) |
                     (uint64) ItemPointerGetOffsetNumber(&slot->tts_tid);
        fv_builder_add_row(builder, vecbuf, packed_tid, scalars, isnull);
        count++;
    }
    table_endscan(scan);
    ExecDropSingleTupleTableSlot(slot);

    if (count == 0)
    {
        fv_builder_destroy(builder);
        table_close(rel, AccessShareLock);
        ereport(ERROR, (errmsg("paves: relation contains no indexable rows")));
    }

    dir = psprintf("%s/paves", DataDir);
    if (mkdir(dir, S_IRWXU) != 0 && errno != EEXIST)
        ereport(ERROR, (errmsg("paves: cannot create directory %s: %m", dir)));
    path = paves_index_path(relid);

    ereport(NOTICE, (errmsg("paves: building HNSW graph for " UINT64_FORMAT
                            " rows (dim=%d, M=%d, ef_construction=%d, threads=%d)",
                            count, dim, paves_build_m, paves_build_ef,
                            paves_build_threads)));

    if (fv_builder_finish(builder, path, vec_attnum, paves_build_m,
                          paves_build_ef, paves_build_threads,
                          errbuf, sizeof(errbuf)) != 0)
    {
        fv_builder_destroy(builder);
        table_close(rel, AccessShareLock);
        ereport(ERROR, (errmsg("paves: index build failed: %s", errbuf)));
    }
    fv_builder_destroy(builder);
    table_close(rel, AccessShareLock);

    PG_RETURN_INT64((int64) count);
}
