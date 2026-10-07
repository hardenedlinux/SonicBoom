#!/usr/bin/env bash
# Build the llama.cpp CPU decode-throughput benchmark. Dev-time only: links
# against the llama.cpp CPU-only build, not SonicBoom. See llama_bench.cpp.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:?set LLAMA_ROOT to the llama.cpp checkout}"
LLAMA_BIN="$LLAMA_ROOT/build/bin"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -I$LLAMA_ROOT/include -I$LLAMA_ROOT/ggml/include"

LDFLAGS="-L$LLAMA_BIN -Wl,-rpath,$LLAMA_BIN \
  -lllama -lggml-cpu -lggml -lggml-base"

"$CXX" $CXXFLAGS "$(dirname "$0")/llama_bench.cpp" -o "$(dirname "$0")/llama_bench" $LDFLAGS

echo "built $(dirname "$0")/llama_bench"
