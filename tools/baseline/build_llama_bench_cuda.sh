#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build llama_bench against the llama.cpp CUDA build (build-cuda13). Dev-time
# only. Requires tools/baseline/build_llama_bench.sh's sibling build-cuda13 to
# exist (see the llama.cpp CUDA build command at the bottom of this file). CUDA
# 13.0 is the toolkit every comparison since round 18 uses; the older
# build-cuda (12.0) tree is kept only for history.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:?set LLAMA_ROOT to the llama.cpp checkout}"
LLAMA_BIN="$LLAMA_ROOT/build-cuda13/bin"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -I$LLAMA_ROOT/include -I$LLAMA_ROOT/ggml/include"

# CUDA build adds libggml-cuda.so; rpath so the runtime finds it without
# LD_LIBRARY_PATH.
LDFLAGS="-L$LLAMA_BIN -Wl,-rpath,$LLAMA_BIN \
  -lllama -lggml-cpu -lggml-cuda -lggml -lggml-base"

"$CXX" $CXXFLAGS "$(dirname "$0")/llama_bench.cpp" -o "$(dirname "$0")/llama_bench_cuda" $LDFLAGS

echo "built $(dirname "$0")/llama_bench_cuda"

# llama.cpp CUDA build (run once, in the llama.cpp tree), CUDA 13.0:
#   cmake -B build-cuda13 -S . -DGGML_CUDA=ON -DGGML_CPU=ON \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
#         -DLLAMA_CURL=OFF \
#         -DCMAKE_CUDA_COMPILER=/opt/cuda-13.0/bin/nvcc
#   cmake --build build-cuda13 --config Release -j
