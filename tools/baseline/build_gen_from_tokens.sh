#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build the token-id greedy generator against libsonicboom.so. Dev-time only
# (mirrors build_longctx_bench.sh).
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/core/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 -I$INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom"

"$CXX" $CXXFLAGS "$(dirname "$0")/gen_from_tokens.cpp" -o "$(dirname "$0")/gen_from_tokens" $LDFLAGS

echo "built $(dirname "$0")/gen_from_tokens"
