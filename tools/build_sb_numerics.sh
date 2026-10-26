#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build tools/sb-numerics against libsonicboom.so. The per-kernel probes go
# through the same internal entry points the out-of-tree test_glue_codegen uses
# (core/src/... headers); everything else is the public Layer 2 surface.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LIB="$ROOT/build/core"
INC="-I$ROOT/core/include -I$ROOT/core/compiler/include -I$ROOT/build/core/compiler/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 $INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom -lcudart -lcuda"

"$CXX" $CXXFLAGS "$(dirname "$0")/sb-numerics.cpp" -o "$(dirname "$0")/sb-numerics" $LDFLAGS

echo "built $(dirname "$0")/sb-numerics"
