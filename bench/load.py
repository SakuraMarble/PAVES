#!/usr/bin/env python3
"""Load a NEW database's tables and build both indexes from identical vectors."""
import argparse
import json
import time
from pathlib import Path

import psycopg2

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--dsn', required=True)
ap.add_argument('--data', required=True, type=Path)
ap.add_argument('--build-threads', type=int, default=32)
args = ap.parse_args()
conn = psycopg2.connect(args.dsn)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute('CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS paves')
    # Deliberately fail if the tables exist. Never delete an existing dataset.
    cur.execute('''CREATE TABLE sift_base (id int PRIMARY KEY, bool_attr int NOT NULL,
        int_attr int NOT NULL, float_attr float8 NOT NULL, embedding vector(128) NOT NULL);
        CREATE TABLE sift_query (id int PRIMARY KEY, embedding vector(128) NOT NULL)''')
    for table, filename in [('sift_base', 'base.csv'), ('sift_query', 'query.csv')]:
        with (args.data / filename).open() as f:
            cur.copy_expert(f'COPY {table} FROM STDIN WITH (FORMAT csv)', f)
    cur.execute('''CREATE INDEX ON sift_base (bool_attr);
        CREATE INDEX ON sift_base (int_attr); CREATE INDEX ON sift_base (float_attr);
        ALTER TABLE sift_base SET (autovacuum_enabled=false); ANALYZE''')
    timings = {}
    start = time.monotonic()
    cur.execute('''CREATE INDEX sift_base_hnsw ON sift_base USING hnsw
        (embedding vector_l2_ops) WITH (m=16, ef_construction=200)''')
    timings['pgvector_build_s'] = time.monotonic() - start
    print(json.dumps(timings), flush=True)
    cur.execute('SET paves.build_threads = %s', (args.build_threads,))
    cur.execute('SET paves.build_m = 16; SET paves.build_ef_construction = 200')
    start = time.monotonic()
    cur.execute("SELECT paves_build('sift_base')")
    print(cur.fetchall(), flush=True)
    timings['paves_build_s'] = time.monotonic() - start
    cur.execute('VACUUM ANALYZE sift_base')
    (args.data / 'build.json').write_text(json.dumps(timings, indent=2) + '\n')
conn.close()
