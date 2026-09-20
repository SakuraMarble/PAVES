#!/usr/bin/env bash
# Complete run from source and public SIFT data. A new runtime directory is required.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
export PAVES_RUN=${PAVES_RUN:-$REPO/runtime}
export PG_PREFIX=${PG_PREFIX:-$PAVES_RUN/pg14}
export PAVES_PORT=${PAVES_PORT:-55420}
PYTHON=${PYTHON:-python3.12}
mkdir -p "$PAVES_RUN"
if [[ -e "$PAVES_RUN/pgdata" ]]; then
    echo 'Existing cluster found. Use a fresh PAVES_RUN directory; no data is overwritten.' >&2
    exit 2
fi
"$PYTHON" -m venv "$PAVES_RUN/venv"
"$PAVES_RUN/venv/bin/pip" install -r "$REPO/bench/requirements.txt"
bash "$REPO/bench/bootstrap.sh" > "$PAVES_RUN/build.log" 2>&1
SIFT_SOURCE=${SIFT_SOURCE:-$PAVES_RUN/sift-128-euclidean.hdf5}
if [[ ! -f "$SIFT_SOURCE" ]]; then
    curl -fL --retry 3 -o "$SIFT_SOURCE" https://ann-benchmarks.com/sift-128-euclidean.hdf5
fi
EXPECTED=dd6f0a6ed6b7ebb8934680f861a33ed01ff33991eaee4fd60914d854a0ca5984
printf '%s  %s\n' "$EXPECTED" "$SIFT_SOURCE" | sha256sum -c -
"$PAVES_RUN/venv/bin/python" "$REPO/bench/prepare.py" --source "$SIFT_SOURCE" --out "$PAVES_RUN/data" > "$PAVES_RUN/prepare.log"
bash "$REPO/bench/cluster.sh" start
trap 'bash "$REPO/bench/cluster.sh" stop' EXIT
"$PG_PREFIX/bin/createdb" -h "$PAVES_RUN/socket" -p "$PAVES_PORT" paves_bench
DSN="host=$PAVES_RUN/socket port=$PAVES_PORT dbname=paves_bench"
"$PAVES_RUN/venv/bin/python" "$REPO/bench/load.py" --dsn "$DSN" --data "$PAVES_RUN/data" > "$PAVES_RUN/load.log" 2>&1
"$PAVES_RUN/venv/bin/python" "$REPO/bench/environment.py" --out "$PAVES_RUN/environment.json" --pg-prefix "$PG_PREFIX"
"$PAVES_RUN/venv/bin/python" "$REPO/bench/benchmark.py" --dsn "$DSN" --out "$PAVES_RUN/results" | tee "$PAVES_RUN/benchmark.log"
"$PAVES_RUN/venv/bin/python" "$REPO/bench/recheck.py" --dsn "$DSN" --results "$PAVES_RUN/results"
"$PAVES_RUN/venv/bin/python" "$REPO/bench/verify_results.py" "$PAVES_RUN/results"
"$PAVES_RUN/venv/bin/python" "$REPO/extension/paves/test/smoke.py" --dsn "$DSN" --gpu > "$PAVES_RUN/smoke.json"
