#!/usr/bin/env bash
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${PAVES_RUN:-$REPO/runtime}
PG_PREFIX=${PG_PREFIX:-$RUN/pg14}
export PATH="$PG_PREFIX/bin:$PATH"
mkdir -p "$RUN/socket"
case ${1:-start} in
start)
    if [[ ! -f "$RUN/pgdata/PG_VERSION" ]]; then
        initdb -D "$RUN/pgdata" --no-locale -E UTF8
        cat >> "$RUN/pgdata/postgresql.conf" <<CONF
port = ${PAVES_PORT:-55420}
listen_addresses = ''
unix_socket_directories = '$RUN/socket'
shared_buffers = '${SHARED_BUFFERS:-8GB}'
work_mem = '64MB'
maintenance_work_mem = '${MAINTENANCE_WORK_MEM:-4GB}'
effective_cache_size = '${EFFECTIVE_CACHE_SIZE:-64GB}'
max_connections = 100
max_worker_processes = 32
max_parallel_workers = 32
max_parallel_maintenance_workers = 16
max_parallel_workers_per_gather = 0
jit = off
shared_preload_libraries = 'paves'
paves.gpu_device = 0
paves.gpu_batch_max = 512
paves.gpu_precision = 'sq8'
CONF
    fi
    pg_ctl -D "$RUN/pgdata" -l "$RUN/postgres.log" -w start
    ;;
stop) pg_ctl -D "$RUN/pgdata" -m fast -w stop ;;
*) echo 'usage: cluster.sh start|stop' >&2; exit 2 ;;
esac
