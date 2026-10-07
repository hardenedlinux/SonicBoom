#!/usr/bin/env bash
# Build the Gemma 4 oracle dump tool against the llama.cpp CPU-only build.
# This tool is dev-time only; it is NOT part of the SonicBoom CMake build and
# adds no llama.cpp dependency to libsonicboom.so.
set -euo pipefail

LLAMA_ROOT="${LLAMA_ROOT:-/path/to/llama.cpp}"
LLAMA_BIN="$LLAMA_ROOT/build/bin"

CXX="${CXX:-g++}"
# C++17 is enough for the llama.h public API; no need for C++23 here.
CXXFLAGS="-std=c++17 -O2 -I$LLAMA_ROOT/include -I$LLAMA_ROOT/ggml/include"

LDFLAGS="-L$LLAMA_BIN -Wl,-rpath,$LLAMA_BIN \
  -lllama -lggml-cpu -lggml -lggml-base"

"$CXX" $CXXFLAGS "$(dirname "$0")/gemma4_dump.cpp" -o "$(dirname "$0")/gemma4_dump" $LDFLAGS

echo "built $(dirname "$0")/gemma4_dump"
