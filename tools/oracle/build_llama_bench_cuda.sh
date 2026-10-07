#!/usr/bin/env bash
# Build llama_bench against the llama.cpp CUDA build (build-cuda). Dev-time only.
# Requires tools/oracle/build_llama_bench.sh's sibling build-cuda to exist (see
# the llama.cpp CUDA build command at the bottom of this file).
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:-/path/to/llama.cpp}"
LLAMA_BIN="$LLAMA_ROOT/build-cuda/bin"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -I$LLAMA_ROOT/include -I$LLAMA_ROOT/ggml/include"

# CUDA build adds libggml-cuda.so; rpath so the runtime finds it without
# LD_LIBRARY_PATH.
LDFLAGS="-L$LLAMA_BIN -Wl,-rpath,$LLAMA_BIN \
  -lllama -lggml-cpu -lggml-cuda -lggml -lggml-base"

"$CXX" $CXXFLAGS "$(dirname "$0")/llama_bench.cpp" -o "$(dirname "$0")/llama_bench_cuda" $LDFLAGS

echo "built $(dirname "$0")/llama_bench_cuda"

# llama.cpp CUDA build (run once, in the llama.cpp tree):
#   cmake -B build-cuda -S . -DGGML_CUDA=ON -DGGML_CPU=ON \
#         -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -DLLAMA_CURL=OFF
#   cmake --build build-cuda --config Release -j
