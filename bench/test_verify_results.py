"""Ensure invalid measurements cannot pass the offline publication gate."""
import json
from pathlib import Path
import tempfile
import unittest

from verify_results import audit, CLASSES, SYSTEMS


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.selected = {s: [f'system={s}'] for s in SYSTEMS}
        self.write('parameters.json', {'repeats': 3, 'target': .9, 'min_speedup': 5,
                                       'clients': 64, 'duration': 10})
        self.write('selected.json', self.selected)
        workload = [{'qid': i, 'type': CLASSES[i % 6], 'gt': list(range(10))}
                    for i in range(2004)]
        self.write_lines('workload.jsonl', workload)
        records = [{'qid': r['qid'], 'type': r['type'], 'ids': r['gt'], 'recall': 1.0}
                   for r in workload if r['qid'] >= 1002]
        for s in SYSTEMS:
            self.write(f'{s}-recall.json', records)
            self.write(f'{s}-recall-concurrent.json', records)
        self.write('concurrent-recall.json', {'threads': 64, 'passed': True, 'systems': {
            s: {'settings': self.selected[s], 'full_count': True,
                'recall': {k: 1.0 for k in CLASSES}} for s in SYSTEMS}})
        self.events = [{'phase': 'calibrate', 'system': s, 'settings': self.selected[s],
                        'qps': 10, 'recall': {k: 1.0 for k in CLASSES}} for s in SYSTEMS]
        self.events += [{'phase': 'measure', 'system': s, 'workload': k, 'repeat': r,
                         'clients': 64, 'errors': 0,
                         'completed': 6000 if s == 'paves' else 1000, 'elapsed_s': 10,
                         'qps': 600 if s == 'paves' else 100}
                        for s in SYSTEMS for k in CLASSES + ['mixed'] for r in range(3)]
        self.write_lines('events.jsonl', self.events)
        self.summary = {'passed': True, 'results': [
            {'workload': k, 'median_qps': {'paves': 600, 'pgvector': 100}, 'speedup': 6}
            for k in CLASSES + ['mixed']]}
        self.write('summary.json', self.summary)

    def write(self, name, obj):
        (self.root / name).write_text(json.dumps(obj))

    def write_lines(self, name, records):
        (self.root / name).write_text('\n'.join(json.dumps(r) for r in records))

    def test_valid(self):
        self.assertTrue(audit(self.root)['passed'])

    def test_fabricated_speedup(self):
        self.summary['results'][0]['speedup'] = 9
        self.write('summary.json', self.summary)
        with self.assertRaises(AssertionError):
            audit(self.root)

    def test_any_query_error(self):
        self.events[-1]['errors'] = 1
        self.write_lines('events.jsonl', self.events)
        with self.assertRaises(AssertionError):
            audit(self.root)

    def test_missing_repeat(self):
        self.write_lines('events.jsonl', self.events[:-1])
        with self.assertRaises(AssertionError):
            audit(self.root)

    def test_fake_recall(self):
        records = json.loads((self.root / 'paves-recall.json').read_text())
        records[0]['ids'] = list(range(20, 30))
        self.write('paves-recall.json', records)
        with self.assertRaises(AssertionError):
            audit(self.root)

    def test_inconsistent_qps(self):
        self.events[-1]['completed'] = 1
        self.write_lines('events.jsonl', self.events)
        with self.assertRaises(AssertionError):
            audit(self.root)

    def test_duplicate_result(self):
        records = json.loads((self.root / 'paves-recall-concurrent.json').read_text())
        records[0]['ids'] = [0] * 10
        self.write('paves-recall-concurrent.json', records)
        with self.assertRaises(AssertionError):
            audit(self.root)


if __name__ == '__main__':
    unittest.main()
