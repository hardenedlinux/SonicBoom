#!/usr/bin/env bash
# Build the standalone K-quant gemv microbenchmark against libsonicboom.so (the
# SHARED core) + libcudart. Dev-time only: it reaches the internal
# quant::cuda::matvec_f32_dev symbol via core/src/quant/cuda_resident.h, which is
# not part of the public Layer 2 surface, so this is NOT part of the CMake build
# (mirrors build_attention_bench.sh).
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/core/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 -I$INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom -lcudart"

"$CXX" $CXXFLAGS "$(dirname "$0")/gemv_bench.cpp" -o "$(dirname "$0")/gemv_bench" $LDFLAGS

echo "built $(dirname "$0")/gemv_bench"
