#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build tools/sb-moss against libsonicboom.so. The encoder and the tokenizer
# loader are Layer 2 (core/include/sonicboom/{model,vocab}); nothing here
# reaches into core/src.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LIB="$ROOT/build/core"
INC="-I$ROOT/core/include -I$ROOT/core/compiler/include -I$ROOT/build/core/compiler/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 $INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom"
# The CUDA DiT (moss_dit_cuda.h) is only linked into a SONICBOOM_USE_CUDA
# build; detect it from the library so a plain build still compiles the tool.
# (A `nm | grep -q` pipeline is wrong here: under `set -o pipefail` grep's
# early exit SIGPIPEs nm and the pipeline reports failure on a match.)
symbols="$(nm -DC --defined-only "$LIB/libsonicboom.so" 2>/dev/null || true)"
case "$symbols" in
  *"sonicboom::moss::dit_cuda_available()"*)
    CXXFLAGS="$CXXFLAGS -DSONICBOOM_TOOL_CUDA=1" ;;
esac

"$CXX" $CXXFLAGS "$(dirname "$0")/sb-moss.cpp" -o "$(dirname "$0")/sb-moss" $LDFLAGS

echo "built $(dirname "$0")/sb-moss"
