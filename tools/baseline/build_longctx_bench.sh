#!/usr/bin/env bash
# Build the long-context decode-throughput tool against libsonicboom.so.
# Dev-time only (mirrors build_sonicboom.sh: links the SHARED core + public
# headers).
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/core/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 -I$INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom"

"$CXX" $CXXFLAGS "$(dirname "$0")/longctx_bench.cpp" -o "$(dirname "$0")/longctx_bench" $LDFLAGS

echo "built $(dirname "$0")/longctx_bench"
