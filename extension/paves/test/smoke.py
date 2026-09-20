#!/usr/bin/env python3
"""Read-only SIFT correctness smoke test against independent PostgreSQL exact scans."""
import argparse
import json
import math

import psycopg2

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--dsn', required=True)
ap.add_argument('--gpu', action='store_true', help='also exercise GPU paths; use ef=100')
args = ap.parse_args()
conn = psycopg2.connect(args.dsn)
conn.autocommit = True
predicates = ['bool_attr=1', 'int_attr=0', 'int_attr IN (0,1,2)',
              'float_attr>=0 AND float_attr<10', 'float_attr>=0 AND float_attr<50',
              'int_attr=0 AND float_attr>=0 AND float_attr<50',
              '(int_attr % 3)=0', 'int_attr=99', 'true']
results = []
with conn.cursor() as cur:
    for qid in (3, 17, 101):
        cur.execute('SELECT embedding::text FROM sift_query WHERE id=%s', (qid,))
        vector = cur.fetchone()[0]
        for pred in predicates:
            sql = f"SELECT id, embedding <-> '{vector}' AS d FROM sift_base WHERE {pred} ORDER BY embedding <-> '{vector}' LIMIT 10"
            cur.execute('SET paves.enable=off; SET enable_indexscan=off; SET enable_indexonlyscan=off; SET enable_bitmapscan=off')
            cur.execute(sql.replace(' LIMIT 10', ', id LIMIT 10'))
            exact = cur.fetchall()
            strategies = ['brute', 'hnsw'] + (['gpu_brute', 'gpu_hnsw'] if args.gpu else [])
            for strategy in strategies:
                cur.execute('RESET enable_indexscan; RESET enable_indexonlyscan; RESET enable_bitmapscan; SET paves.enable=on; SET paves.ef_search=100')
                cur.execute('SET paves.force_strategy=%s', (strategy,))
                cur.execute(sql)
                found = cur.fetchall()
                assert len(found) == len(exact), (qid, pred, strategy, 'underfull')
                assert len({r[0] for r in found}) == len(found), 'duplicate IDs'
                if found:
                    # psycopg2's parameter parser treats literal modulo '%' specially.
                    parameterized_predicate = pred.replace('%', '%%')
                    cur.execute(f'SELECT count(*) FROM sift_base WHERE id=ANY(%s) AND NOT ({parameterized_predicate})', ([r[0] for r in found],))
                    assert cur.fetchone()[0] == 0, 'predicate violation'
                if strategy == 'brute':
                    assert all(math.isclose(a[1], b[1], rel_tol=1e-6, abs_tol=1e-6)
                               for a, b in zip(found, exact)), (qid, pred, strategy, found, exact)
                # Smoke ANN validity and retain recall; aggregate recall gates live in benchmark.py.
                hits = len({x[0] for x in found} & {x[0] for x in exact})
                results.append({'qid': qid, 'predicate': pred, 'strategy': strategy,
                                'recall': hits / len(exact) if exact else 1.0})
print(json.dumps({'passed': True, 'checks': len(results), 'results': results}, indent=2))
conn.close()
