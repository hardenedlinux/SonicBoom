#!/usr/bin/env bash

# Locale-independent output: numbers must parse/print identically everywhere.
export LC_ALL=C LANG=C
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

# The measured decode step is bracketed by cudaEvents, so the tool needs a
# runtime API. Link the one the core was built with (the build cache's nvcc
# names its toolkit), or the timing calls and the kernels would come from two
# different runtimes in one process; fall back to $CUDA_HOME and the system.
CUDART=""
CUDAINC=""
pick_toolkit() { # <toolkit root>
  for lib in "$1/lib64/libcudart.so" \
             "$1/targets/x86_64-linux/lib/libcudart.so" \
             "$1/lib/x86_64-linux-gnu/libcudart.so"; do
    if [ -f "$lib" ]; then
      CUDART="$lib"
      if [ -d "$1/include" ]; then CUDAINC="-I$1/include"; fi
      return 0
    fi
  done
  return 1
}
NVCC=""
if [ -f "$ROOT/build/CMakeCache.txt" ]; then
  NVCC=$(sed -n 's/^CMAKE_CUDA_COMPILER:STRING=//p' "$ROOT/build/CMakeCache.txt")
fi
found=0
if [ -n "$NVCC" ]; then
  toolkit=$(cd "$(dirname "$NVCC")/.." 2>/dev/null && pwd) || toolkit=""
  if [ -n "$toolkit" ] && pick_toolkit "$toolkit"; then found=1; fi
fi
if [ "$found" -eq 0 ] && [ -n "${CUDA_HOME:-}" ]; then
  if pick_toolkit "$CUDA_HOME"; then found=1; fi
fi
if [ "$found" -eq 0 ] && [ -d /usr/local/cuda ]; then
  if pick_toolkit /usr/local/cuda; then found=1; fi
fi
if [ "$found" -eq 0 ]; then
  if pick_toolkit /usr; then found=1; fi
fi
if [ "$found" -eq 0 ]; then
  echo "libcudart not found" >&2
  exit 1
fi
CXXFLAGS="$CXXFLAGS $CUDAINC"
LDFLAGS="$LDFLAGS $CUDART -Wl,-rpath,${CUDART%/*}"

"$CXX" $CXXFLAGS "$(dirname "$0")/longctx_bench.cpp" -o "$(dirname "$0")/longctx_bench" $LDFLAGS

echo "built $(dirname "$0")/longctx_bench"
