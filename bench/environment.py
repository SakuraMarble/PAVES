#!/usr/bin/env python3
"""Record reproducibility facts; never capture the whole environment or credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def capture(command):
    try:
        r = subprocess.run(command, text=True, capture_output=True, timeout=30)
        return {'command': command, 'returncode': r.returncode, 'stdout': r.stdout, 'stderr': r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'command': command, 'error': str(e)}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--out', required=True, type=Path)
ap.add_argument('--pg-prefix', required=True, type=Path)
args = ap.parse_args()
repo = Path(__file__).resolve().parents[1]
pg = args.pg_prefix
report = {'utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
          'platform': platform.platform(), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
          'source_sha256': {str(p.relative_to(repo)): digest(p)
                            for directory in ('extension', 'bench')
                            for p in (repo / directory).rglob('*')
                            if p.is_file() and p.suffix in ('.c', '.cpp', '.h', '.cu', '.py', '.sh', '.sql', '.control')},
          'commands': [capture(c) for c in [
              ['lscpu'], ['free', '-b'], ['uname', '-a'], ['g++', '--version'],
              ['/usr/local/cuda/bin/nvcc', '--version'],
              ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv'],
              [str(pg / 'bin/pg_config'), '--configure'],
              [str(pg / 'bin/pg_config'), '--version'],
              ['git', '-C', str(repo), 'rev-parse', 'HEAD']]]}
report['source_sha256']['extension/paves/Makefile'] = digest(repo / 'extension/paves/Makefile')
report['binaries_sha256'] = {name: digest(pg / name) for name in
                            ('bin/postgres', 'lib/postgresql/paves.so', 'lib/postgresql/vector.so')}
args.out.write_text(json.dumps(report, indent=2) + '\n')
