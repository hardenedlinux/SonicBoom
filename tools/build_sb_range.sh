#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build tools/sb-range against libsonicboom.so. It loads a model, runs the CPU
# spine with the calibration hook on, and invokes the Boom analysis that the
# plan lowering exposes under SONICBOOM_BOOM_ANALYSIS_FACTS.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
LIB="$ROOT/build/core"
INC="-I$ROOT/core/include -I$ROOT/core/compiler/include -I$ROOT/core/src"
INC="$INC -I${SONICBOOM_LLVM_INSTALL:-/home/alexon/Project/SonicBoom/build/llvm-mlir-install/include}"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 $INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom -lcudart -lcuda"

"$CXX" $CXXFLAGS "$(dirname "$0")/sb-range.cpp" -o "$(dirname "$0")/sb-range" $LDFLAGS

echo "built $(dirname "$0")/sb-range"
