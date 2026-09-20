-- paves extension objects
\echo Use "CREATE EXTENSION paves" to load this file. \quit

CREATE FUNCTION paves_build(rel regclass)
RETURNS bigint
AS 'MODULE_PATHNAME', 'paves_build'
LANGUAGE C STRICT;

COMMENT ON FUNCTION paves_build(regclass) IS
'Scan the table and (re)build the paves in-memory vector index file '
'(vectors + HNSW graph + scalar column cache). Atomic replace.';

-- read-only pre-flight check: sizes, fits, fast-path availability
CREATE FUNCTION paves_capacity(rel regclass)
RETURNS TABLE(metric text, value text, ok text, note text)
AS 'MODULE_PATHNAME', 'paves_capacity'
LANGUAGE C STRICT;

-- cost-model runtime feedback dump: per (index, strategy, selectivity
-- bucket) EWMAs the auto router reads, plus live load counters
CREATE FUNCTION paves_feedback(
    OUT dbid oid, OUT relid oid, OUT strat text, OUT bucket int,
    OUT bucket_mid float8, OUT gpu_q_us bigint, OUT gpu_batch_us bigint,
    OUT gpu_batch_n float8, OUT cpu_q_us bigint, OUT cpu_ref_active float8,
    OUT gpu_inflight int, OUT cpu_active int)
RETURNS SETOF record
AS 'MODULE_PATHNAME', 'paves_feedback'
LANGUAGE C;
