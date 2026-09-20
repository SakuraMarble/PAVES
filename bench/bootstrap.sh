#!/usr/bin/env bash
# User-space toolchain; never installs into the system PostgreSQL.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
RUN=${PAVES_RUN:-$REPO/runtime}
PG_PREFIX=${PG_PREFIX:-$RUN/pg14}
JOBS=${JOBS:-16}
mkdir -p "$RUN/src"
if [[ ! -x "$PG_PREFIX/bin/pg_config" ]]; then
    cd "$RUN/src"
    curl -fL --retry 3 -o postgresql-14.23.tar.bz2 https://ftp.postgresql.org/pub/source/v14.23/postgresql-14.23.tar.bz2
    tar -xjf postgresql-14.23.tar.bz2
    cd postgresql-14.23
    ./configure --prefix="$PG_PREFIX" CFLAGS=-O2
    make -j"$JOBS"
    make install
fi
export PATH="$PG_PREFIX/bin:$PATH"
if [[ ! -f "$($PG_PREFIX/bin/pg_config --pkglibdir)/vector.so" ]]; then
    git clone --depth 1 --branch v0.8.5 https://github.com/pgvector/pgvector.git "$RUN/src/pgvector"
    make -C "$RUN/src/pgvector" -j"$JOBS" PG_CONFIG="$PG_PREFIX/bin/pg_config" OPTFLAGS=-march=native
    make -C "$RUN/src/pgvector" install PG_CONFIG="$PG_PREFIX/bin/pg_config"
fi
make -C "$REPO/extension/paves" clean PG_CONFIG="$PG_PREFIX/bin/pg_config"
make -C "$REPO/extension/paves" -j"$JOBS" PG_CONFIG="$PG_PREFIX/bin/pg_config" CUDA_ARCH="${CUDA_ARCH:-sm_89}"
make -C "$REPO/extension/paves" install PG_CONFIG="$PG_PREFIX/bin/pg_config"
g++ -O2 -std=c++17 "$REPO/bench/driver.cpp" -o "$REPO/bench/driver" \
    -I"$(pg_config --includedir)" -L"$(pg_config --libdir)" \
    -Wl,-rpath,"$(pg_config --libdir)" -lpq -lpthread
