#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build the real-text tokenizer against llama.cpp (baseline reference). Dev-time
# only: links llama.cpp, not SonicBoom.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:?set LLAMA_ROOT to the llama.cpp checkout}"
LLAMA_BIN="$LLAMA_ROOT/build/bin"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -I$LLAMA_ROOT/include -I$LLAMA_ROOT/ggml/include"

LDFLAGS="-L$LLAMA_BIN -Wl,-rpath,$LLAMA_BIN \
  -lllama -lggml-cpu -lggml -lggml-base"

"$CXX" $CXXFLAGS "$(dirname "$0")/tokenize_text.cpp" -o "$(dirname "$0")/tokenize_text" $LDFLAGS

echo "built $(dirname "$0")/tokenize_text"
