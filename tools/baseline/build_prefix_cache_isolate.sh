#!/usr/bin/env bash
# Build the prefix-cache isolation tool against libsonicboom.so + the CUDA
# runtime (it reads device KV buffers through hal::Buffer::id with cudaMemcpy).
# Dev-time only.
set -euo pipefail

ROOT="${SONICBOOM_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
LIB="$ROOT/build/core"
INC="$ROOT/core/include"

CXX="${CXX:-g++-13}"
CXXFLAGS="-std=c++23 -O2 -I$INC"
LDFLAGS="-L$LIB -Wl,-rpath,$LIB -lsonicboom -lcudart"

"$CXX" $CXXFLAGS "$(dirname "$0")/prefix_cache_isolate.cpp" -o "$(dirname "$0")/prefix_cache_isolate" $LDFLAGS

echo "built $(dirname "$0")/prefix_cache_isolate"
