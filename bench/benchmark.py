#!/usr/bin/env python3
"""Recall-gated, calibrated comparison. Exit nonzero unless all acceptance gates pass."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time

import psycopg2
from prepare import FILTERS


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def connect(dsn, settings):
    c = psycopg2.connect(dsn)
    c.autocommit = True
    with c.cursor() as cur:
        cur.execute('SET statement_timeout = 120000')
        for setting in settings:
            cur.execute('SET ' + setting)
    return c


def evaluate(dsn, rows, settings, threads, truth=False):
    def shard_work(shard):
        conn = connect(dsn, settings)
        result = []
        try:
            with conn.cursor() as cur:
                for row in shard:
                    sql = row['sql']
                    if truth:
                        sql = sql.replace(' LIMIT 10', ', id LIMIT 10')
                    cur.execute(sql)
                    ids = [x[0] for x in cur.fetchall()]
                    result.append({'qid': row['qid'], 'type': row['type'], 'ids': ids})
        finally:
            conn.close()
        return result
    with ThreadPoolExecutor(threads) as pool:
        parts = pool.map(shard_work, [rows[i::threads] for i in range(threads)])
        return sorted([r for part in parts for r in part], key=lambda x: x['qid'])


def recall(rows, returned):
    expected = {r['qid']: r for r in rows}
    scores = {}
    details = []
    for item in returned:
        row = expected[item['qid']]
        gt = set(row['gt'])
        score = len(gt & set(item['ids'])) / len(gt) if gt else float(not item['ids'])
        scores.setdefault(item['type'], []).append(score)
        details.append(dict(item, recall=score, expected_count=len(gt)))
    assert len(returned) == len(rows)
    return {k: statistics.mean(v) for k, v in scores.items()}, details


def driver(args, query_file, settings, duration=None):
    cmd = [str(args.driver), '--conninfo', args.dsn, '--queries', str(query_file),
           '--clients', str(args.clients), '--duration', str(duration or args.duration),
           '--warmup', str(args.warmup)]
    for setting in settings:
        cmd += ['--set', setting]
    process = subprocess.run(cmd, text=True, capture_output=True, check=True, timeout=600)
    result = json.loads(process.stdout)
    assert result['errors'] == 0 and result['completed'] > 0, result
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dsn', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--driver', type=Path, default=Path(__file__).resolve().with_name('driver'))
    ap.add_argument('--clients', type=int, default=64)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--duration', type=float, default=10)
    ap.add_argument('--warmup', type=float, default=2)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--target', type=float, default=0.9)
    ap.add_argument('--min-speedup', type=float, default=5)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    events = args.out / 'events.jsonl'
    def emit(record):
        record['utc'] = datetime.now(timezone.utc).isoformat()
        with events.open('a') as f:
            f.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    write_json(args.out / 'parameters.json', {k: str(v) if isinstance(v, Path) else v
                                              for k, v in vars(args).items() if k != 'dsn'})
    conn = connect(args.dsn, ['paves.enable=off'])
    with conn.cursor() as cur:
        cur.execute('SELECT id, embedding::text FROM sift_query WHERE id < 2004 ORDER BY id')
        queries = cur.fetchall()
        assert [q[0] for q in queries] == list(range(2004))
        cur.execute('SELECT count(*) FROM sift_base')
        assert cur.fetchone()[0] == 1000000
        cur.execute('SELECT version()')
        version = cur.fetchone()[0]
        cur.execute('SELECT extname, extversion FROM pg_extension')
        extensions = dict(cur.fetchall())
        cur.execute('SELECT name, setting, unit FROM pg_settings WHERE source != \'default\'')
        settings = cur.fetchall()
        cur.execute("SELECT indexname, indexdef FROM pg_indexes WHERE tablename='sift_base'")
        indexes = cur.fetchall()
        write_json(args.out / 'database.json', dict(version=version, extensions=extensions,
                                                   settings=settings, indexes=indexes))
    conn.close()
    rows = []
    for qid, embedding in queries:
        kind, pred = FILTERS[qid % len(FILTERS)]
        rows.append({'qid': qid, 'type': kind,
                     'sql': f"SELECT id FROM sift_base WHERE {pred} ORDER BY embedding <-> '{embedding}' LIMIT 10"})
    truth_settings = ['paves.enable=off', 'enable_indexscan=off',
                      'enable_indexonlyscan=off', 'enable_bitmapscan=off',
                      'max_parallel_workers_per_gather=0']
    emit({'phase': 'truth_start', 'queries': len(rows)})
    gt = evaluate(args.dsn, rows, truth_settings, args.threads, truth=True)
    for row, answer in zip(rows, gt):
        assert row['qid'] == answer['qid'] and len(answer['ids']) == 10
        row['gt'] = answer['ids']
    with (args.out / 'workload.jsonl').open('w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    emit({'phase': 'truth_done', 'queries': len(rows)})
    cal, confirmation = rows[:1002], rows[1002:]
    def query_file(name, subset):
        path = args.out / (name + '.sql')
        path.write_text('\n'.join(r['sql'] for r in subset) + '\n')
        return path
    cal_file = query_file('calibration', cal)
    candidates = {'pgvector': [], 'paves': []}
    for mode in ('off', 'strict_order', 'relaxed_order'):
        for ef in (20, 30, 40, 60, 80, 120, 200):
            candidates['pgvector'].append(['paves.enable=off', f'hnsw.ef_search={ef}',
                f'hnsw.iterative_scan={mode}', 'hnsw.max_scan_tuples=200000'])
    for ef in (10, 15, 20, 25, 30, 40, 60):
        candidates['paves'].append(['paves.enable=on', 'paves.force_strategy=auto',
                                   f'paves.ef_search={ef}'])
    selected = {}
    for system, configs in candidates.items():
        valid = []
        for sets in configs:
            rec, details = recall(cal, evaluate(args.dsn, cal, sets, args.threads))
            item = {'phase': 'calibrate', 'system': system, 'settings': sets, 'recall': rec}
            if min(rec.values()) >= args.target:
                measurements = [driver(args, cal_file, sets, duration=3) for _ in range(2)]
                item['qps_runs'] = measurements
                item['qps'] = statistics.median(m['qps'] for m in measurements)
                valid.append(item)
            emit(item)
        if not valid:
            raise RuntimeError(f'{system}: no configuration meets recall on calibration set')
        selected[system] = max(valid, key=lambda c: c['qps'])['settings']
    write_json(args.out / 'selected.json', selected)
    emit({'phase': 'selected', 'settings': selected})
    # Freeze settings here. Confirmation must never tune against these queries.
    recalls = {}
    for system, sets in selected.items():
        returned = evaluate(args.dsn, confirmation, sets, args.threads)
        recalls[system], details = recall(confirmation, returned)
        write_json(args.out / f'{system}-recall.json', details)
        emit({'phase': 'confirmation', 'system': system, 'recall': recalls[system]})
    # Actual plans for both systems and exact reference, one query per class.
    plans = {}
    for system, sets in dict(selected, exact=truth_settings).items():
        c = connect(args.dsn, sets)
        plans[system] = {}
        with c.cursor() as cur:
            for row in confirmation[:6]:
                sql = row['sql'] if system != 'exact' else row['sql'].replace(' LIMIT 10', ', id LIMIT 10')
                cur.execute('EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) ' + sql)
                plans[system][row['type']] = cur.fetchone()[0]
        c.close()
    write_json(args.out / 'plans.json', plans)
    files = {kind: query_file(kind, [r for r in confirmation if r['type'] == kind])
             for kind, _ in FILTERS}
    files['mixed'] = query_file('mixed', confirmation)
    measured = {kind: {s: [] for s in selected} for kind in files}
    for repeat in range(args.repeats):
        for kind, path in files.items():
            systems = list(selected)
            random.Random(20260920 + repeat).shuffle(systems)
            for system in systems:
                result = driver(args, path, selected[system])
                measured[kind][system].append(result)
                emit(dict(phase='measure', repeat=repeat, workload=kind, system=system, **result))
    summary = []
    for kind, runs in measured.items():
        medians = {s: statistics.median(r['qps'] for r in runs[s]) for s in selected}
        ratio = medians['paves'] / medians['pgvector']
        # Conservative spread check; does not replace the documented median statistic.
        worst = min(r['qps'] for r in runs['paves']) / max(r['qps'] for r in runs['pgvector'])
        row = {'workload': kind, 'median_qps': medians, 'speedup': ratio,
               'min_over_max_speedup': worst,
               'recall': {s: recalls[s].get(kind, statistics.mean(recalls[s].values())) for s in selected}}
        summary.append(row)
    passed = (all(min(v.values()) >= args.target for v in recalls.values())
              and all(r['speedup'] >= args.min_speedup for r in summary)
              and all(len(r['ids']) == r['expected_count'] for s in selected
                      for r in json.loads((args.out / f'{s}-recall.json').read_text())))
    report = {'passed': passed, 'scope': 'SIFT1M six synthetic predicate classes, SQL throughput',
              'target_recall': args.target, 'min_speedup': args.min_speedup,
              'selected': selected, 'results': summary}
    write_json(args.out / 'summary.json', report)
    emit({'phase': 'done', **report})
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
