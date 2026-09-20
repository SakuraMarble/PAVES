#!/usr/bin/env python3
"""Prepare SIFT1M with deterministic synthetic attributes; no ground truth from PAVES."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

FILTERS = [
    ('Equality_bool', 'bool_attr = 1'),
    ('Equality_int', 'int_attr = 0'),
    ('Inclusion', 'int_attr IN (0,1,2)'),
    ('Range_10', 'float_attr >= 0 AND float_attr < 10'),
    ('Range_50', 'float_attr >= 0 AND float_attr < 50'),
    ('Logic', 'int_attr = 0 AND float_attr >= 0 AND float_attr < 50'),
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', default=20240617, type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with h5py.File(args.source) as h5:
        n, d = h5['train'].shape
        assert (n, d) == (1000000, 128), (n, d)
        attrs = [rng.integers(0, 2, n, dtype=np.int32),
                 rng.integers(0, 10, n, dtype=np.int32),
                 rng.uniform(0, 100, n).astype(np.float32)]
        def vec(row):
            return '[' + ','.join(f'{float(v):.9g}' for v in row) + ']'
        with (args.out / 'base.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            for start in range(0, n, 10000):
                for offset, row in enumerate(h5['train'][start:start + 10000]):
                    i = start + offset
                    writer.writerow([i, int(attrs[0][i]), int(attrs[1][i]),
                                     f'{float(attrs[2][i]):.9g}', vec(row)])
        with (args.out / 'query.csv').open('w', newline='') as f:
            writer = csv.writer(f)
            # Separate calibration and confirmation queries, 1,002 each (167 per class).
            for i, row in enumerate(h5['test'][:2004]):
                writer.writerow([i, vec(row)])
    manifest = {'source_sha256': sha256(args.source), 'seed': args.seed,
                'n': n, 'd': d, 'query_count': 2004,
                'calibration': [0, 1002], 'confirmation': [1002, 2004],
                'filters': FILTERS,
                'files': {name: sha256(args.out / name) for name in ('base.csv', 'query.csv')}}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
