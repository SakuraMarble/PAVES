#!/usr/bin/env python3
"""Recheck frozen settings at the throughput client's concurrency; never retune."""
import argparse
import json
from pathlib import Path

from benchmark import evaluate, recall, write_json

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--dsn', required=True)
ap.add_argument('--results', required=True, type=Path)
args = ap.parse_args()
root = args.results
params = json.loads((root / 'parameters.json').read_text())
selected = json.loads((root / 'selected.json').read_text())
rows = [json.loads(s) for s in (root / 'workload.jsonl').read_text().splitlines()]
rows = [r for r in rows if r['qid'] >= 1002]
report = {'threads': params['clients'], 'systems': {}}
for system, settings in selected.items():
    returned = evaluate(args.dsn, rows, settings, params['clients'])
    scores, details = recall(rows, returned)
    report['systems'][system] = {'recall': scores, 'settings': settings,
                                'full_count': all(len(d['ids']) == d['expected_count'] for d in details)}
    write_json(root / f'{system}-recall-concurrent.json', details)
report['passed'] = all(min(s['recall'].values()) >= params['target'] and s['full_count']
                       for s in report['systems'].values())
write_json(root / 'concurrent-recall.json', report)
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['passed'] else 1)
