#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build tools/sb-tune against libsonicboom.so. It reads and writes the same
# per-device cache files the runtime uses (core/src/planner headers), so it
# links only the published Layer 2 library.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LIB="$ROOT/build/core"
INC="-I$ROOT/core/include -I$ROOT/core/compiler/include -I$ROOT/core/src"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 $INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom -lcudart -lcuda"

"$CXX" $CXXFLAGS "$(dirname "$0")/sb-tune.cpp" -o "$(dirname "$0")/sb-tune" $LDFLAGS

echo "built $(dirname "$0")/sb-tune"
