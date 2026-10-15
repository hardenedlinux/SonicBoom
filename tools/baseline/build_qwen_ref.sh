#!/usr/bin/env bash
# Build the llama.cpp greedy reference-token tool for the Qwen2 tests. Dev-time
# only: links against the llama.cpp CPU-only build, not SonicBoom. See
# qwen_ref.cpp.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:?set LLAMA_ROOT to the llama.cpp checkout}"
LLAMA_BIN="$LLAMA_ROOT/build/bin"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -I$LLAMA_ROOT/include -I$LLAMA_ROOT/ggml/include"

LDFLAGS="-L$LLAMA_BIN -Wl,-rpath,$LLAMA_BIN \
  -lllama -lggml-cpu -lggml -lggml-base"

"$CXX" $CXXFLAGS "$(dirname "$0")/qwen_ref.cpp" -o "$(dirname "$0")/qwen_ref" $LDFLAGS

echo "built $(dirname "$0")/qwen_ref"
