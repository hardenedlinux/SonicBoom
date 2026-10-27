#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build tools/bench_train against libsonicboom.so (the SHARED core), through
# the C ABI only.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/capi/include"

CXX="${CXX:-g++-13}"
# The bench input (train/tiny-train.sx) is test data and lives with the private
# suites. Derive the data directory from SONICBOOM_PRIVATE_DATA_DIR, or from
# SONICBOOM_PRIVATE_ROOT (whose tests/data is the same directory). Without it
# bench_train builds but needs an explicit --fixture at run time.
PRIVATE_DATA="${SONICBOOM_PRIVATE_DATA_DIR:-}"
if [ -z "$PRIVATE_DATA" ] && [ -n "${SONICBOOM_PRIVATE_ROOT:-}" ]; then
    PRIVATE_DATA="$SONICBOOM_PRIVATE_ROOT/tests/data"
fi
DATA_DEF=""
if [ -n "$PRIVATE_DATA" ]; then
    DATA_DEF="-DSONICBOOM_PRIVATE_DATA_DIR=\"$PRIVATE_DATA\""
fi

CXXFLAGS="-std=c++23 -O2 -I$INC $DATA_DEF"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom"

"$CXX" $CXXFLAGS "$(dirname "$0")/bench_train.cpp" -o "$(dirname "$0")/bench_train" $LDFLAGS

echo "built $(dirname "$0")/bench_train"
