#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
# Build the mmq-vs-dp4a crossover microbenchmark against libsonicboom.so (the
# SHARED core) + libcudart. Dev-time only: it reaches the public
# quant::cuda::matmul_f32 entry point via <sonicboom/quant/quantized_matmul_cuda.h>,
# so this is NOT part of the CMake build (mirrors build_gemv_bench.sh).
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/core/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 -I$INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom -lcudart"

"$CXX" $CXXFLAGS "$(dirname "$0")/mmq_minn_bench.cpp" -o "$(dirname "$0")/mmq_minn_bench" $LDFLAGS

echo "built $(dirname "$0")/mmq_minn_bench"
