# paves — PGXS build with mixed C (PG glue), C++17 (engine) and optional
# CUDA (gpu_engine). GPU support is compiled in automatically when nvcc is
# found; without it the extension builds CPU-only and fv_gpu_brute_begin
# returns NULL.
.DEFAULT_GOAL := all
MODULE_big = paves
OBJS = src/paves.o src/planner.o src/quals.o src/exec.o src/build.o \
       src/arbiter.o src/worker.o src/capacity.o src/engine/engine.o

EXTENSION = paves
DATA = sql/paves--0.1.sql

PG_CPPFLAGS = -I$(srcdir)/src
PG_CFLAGS = -O3 -march=native
# -rdynamic: keep symbol names resolvable in the worker's crash backtrace
SHLIB_LINK = -lstdc++ -lpthread -lm -rdynamic

ENGINE_CXXFLAGS = -std=c++17 -O3 -march=native -fPIC -Wall -Wextra \
                  -Wno-unused-parameter -fno-strict-aliasing

NVCC := $(shell command -v nvcc 2>/dev/null)
ifeq ($(NVCC),)
NVCC := $(firstword $(wildcard /usr/local/cuda*/bin/nvcc))
endif
ifneq ($(NVCC),)
OBJS += src/engine/gpu_engine.o
ENGINE_CXXFLAGS += -DFV_GPU
CUDA_LIBDIR := $(abspath $(dir $(NVCC))../lib64)
SHLIB_LINK += -L$(CUDA_LIBDIR) -lcudart_static -lrt -ldl
CUDA_ARCH ?= sm_89
NVCCFLAGS = -arch=$(CUDA_ARCH) -O3 -std=c++17 -Xcompiler -fPIC
endif

# PGXS has no header dependency tracking for plain .c objects; a stale
# arbiter.o after an arbiter.h change once shipped a shared-memory segment
# sized for the old slot count (instant worker segfault). Declare them.
$(OBJS): src/paves.h src/arbiter.h src/engine/engine_api.h

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# operator-level saturation microbenchmark (no PostgreSQL dependency)
gpu_micro: tools/gpu_micro.cpp src/engine/engine.o src/engine/gpu_engine.o
	$(CXX) $(ENGINE_CXXFLAGS) -Isrc -o $@ tools/gpu_micro.cpp \
	    src/engine/engine.o src/engine/gpu_engine.o \
	    -L$(CUDA_LIBDIR) -lcudart_static -lrt -ldl -lpthread

src/engine/engine.o: src/engine/engine.cpp src/engine/engine_api.h \
                     src/engine/gpu_engine.h
	$(CXX) $(ENGINE_CXXFLAGS) -I$(srcdir)/src -c $< -o $@

ifneq ($(NVCC),)
src/engine/gpu_engine.o: src/engine/gpu_engine.cu src/engine/gpu_engine.h \
                         src/engine/engine_api.h
	$(NVCC) $(NVCCFLAGS) -I$(srcdir)/src -c $< -o $@
endif
