#!/usr/bin/env python3
"""Offline audit: recompute recall and speedups from saved IDs and raw timings."""
import argparse
import json
import math
from pathlib import Path
import statistics

CLASSES = ['Equality_bool', 'Equality_int', 'Inclusion', 'Range_10', 'Range_50', 'Logic']
SYSTEMS = ['pgvector', 'paves']


def audit(root):
    summary = json.loads((root / 'summary.json').read_text())
    params = json.loads((root / 'parameters.json').read_text())
    selected = json.loads((root / 'selected.json').read_text())
    assert params['repeats'] >= 3
    assert params['target'] >= 0.9 and params['min_speedup'] >= 5
    workload = [json.loads(s) for s in (root / 'workload.jsonl').read_text().splitlines()]
    expected = {r['qid']: r for r in workload if r['qid'] >= 1002}
    assert set(expected) == set(range(1002, 2004))
    recalls = {}
    for system in SYSTEMS:
        records = json.loads((root / f'{system}-recall.json').read_text())
        assert len(records) == len(expected)
        assert {r['qid'] for r in records} == set(expected)
        scores = {k: [] for k in CLASSES}
        for r in records:
            row = expected[r['qid']]
            assert r['type'] == row['type']
            assert len(set(r['ids'])) == len(r['ids']) == len(row['gt']) == 10
            score = len(set(r['ids']) & set(row['gt'])) / 10
            assert math.isclose(score, r['recall'])
            scores[r['type']].append(score)
        assert all(len(v) == 167 for v in scores.values())
        recalls[system] = {k: statistics.mean(v) for k, v in scores.items()}
    events = [json.loads(s) for s in (root / 'events.jsonl').read_text().splitlines()]
    calibration = [r for r in events if r['phase'] == 'calibrate' and 'qps' in r]
    assert all(set(r['recall']) == set(CLASSES) and min(r['recall'].values()) >= params['target']
               for r in calibration)
    for system in SYSTEMS:
        best = max((r for r in calibration if r['system'] == system), key=lambda r: r['qps'])
        assert best['settings'] == selected[system]
    concurrent = json.loads((root / 'concurrent-recall.json').read_text())
    assert concurrent['threads'] == params['clients']
    for system in SYSTEMS:
        records = json.loads((root / f'{system}-recall-concurrent.json').read_text())
        assert len(records) == len(expected)
        assert {r['qid'] for r in records} == set(expected)
        scores = {k: [] for k in CLASSES}
        for r in records:
            row = expected[r['qid']]
            assert r['type'] == row['type']
            assert len(set(r['ids'])) == len(r['ids']) == len(row['gt']) == 10
            score = len(set(r['ids']) & set(row['gt'])) / 10
            assert math.isclose(score, r['recall'])
            scores[r['type']].append(score)
        saved = concurrent['systems'][system]
        assert saved['settings'] == selected[system] and saved['full_count']
        assert all(math.isclose(statistics.mean(v), saved['recall'][k])
                   for k, v in scores.items())
    concurrent_passed = all(min(r['recall'].values()) >= params['target']
                            for r in concurrent['systems'].values())
    assert concurrent['passed'] == concurrent_passed
    reports = {r['workload']: r for r in summary['results']}
    assert set(reports) == set(CLASSES + ['mixed'])
    output = []
    for kind in CLASSES + ['mixed']:
        qps = {}
        for system in SYSTEMS:
            records = [r for r in events if r['phase'] == 'measure'
                       and r['system'] == system and r['workload'] == kind]
            assert len(records) == params['repeats']
            assert {r['repeat'] for r in records} == set(range(params['repeats']))
            assert all(r['errors'] == 0 and r['clients'] == params['clients']
                       and r['completed'] > 0 and r['elapsed_s'] >= params['duration'] * .99
                       and math.isfinite(r['qps']) and math.isclose(
                           r['completed'] / r['elapsed_s'], r['qps'], rel_tol=.001, abs_tol=.1)
                       for r in records)
            qps[system] = statistics.median(r['qps'] for r in records)
            assert math.isclose(qps[system], reports[kind]['median_qps'][system])
        speedup = qps['paves'] / qps['pgvector']
        assert math.isclose(speedup, reports[kind]['speedup'])
        output.append({'workload': kind, 'speedup': speedup})
    passed = (all(min(v.values()) >= params['target'] for v in recalls.values())
              and all(r['speedup'] >= params['min_speedup'] for r in output))
    assert summary['passed'] == passed
    return {'passed': passed and concurrent_passed, 'recall': recalls,
            'concurrent_recall': concurrent['systems'], 'results': output}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', type=Path)
    args = ap.parse_args()
    result = audit(args.results)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
