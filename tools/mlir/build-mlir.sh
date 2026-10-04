#!/usr/bin/env bash
# One-shot build + install of LLVM/MLIR (llvmorg-23.1.2) from the SonicBoom
# submodule, for the MLIR infrastructure integration.
#
# Produces a gitignored prefix at build/llvm-mlir-install that SonicBoom's CMake
# consumes via find_package(MLIR). MLIR/LLVM are built as static libraries and
# later linked into libsonicboom.so — they are NOT a runtime dependency, NOT
# forked, and no system LLVM/MLIR is used.
#
# Usage: tools/mlir/build-mlir.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="$ROOT/third_party/llvm-project/llvm"
BUILD="$ROOT/build/llvm-mlir"
PREFIX="$ROOT/build/llvm-mlir-install"

# Map the host architecture to the LLVM target name. `LLVM_TARGETS_TO_BUILD`
# accepts concrete target names (not "Native"/"host") in this LLVM version.
# NVPTX is always added so the install can emit PTX for the GPU backend; it is a
# pure codegen target (no CUDA toolchain is needed at build time).
case "$(uname -m)" in
  x86_64|amd64)   HOST_TARGET=X86 ;;
  aarch64|arm64)  HOST_TARGET=AArch64 ;;
  *) echo "error: unsupported host arch: $(uname -m)" >&2; exit 1 ;;
esac

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-13 \
  -DCMAKE_CXX_COMPILER=g++-13 \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="$HOST_TARGET;NVPTX" \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DMLIR_INCLUDE_EXAMPLES=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_INSTALL_UTILS=OFF \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_ENABLE_ZLIB=OFF \
  -DLLVM_ENABLE_ZSTD=OFF \
  -DCMAKE_INSTALL_PREFIX="$PREFIX"

cmake --build "$BUILD"
cmake --install "$BUILD"

echo
echo "MLIR installed to: $PREFIX"
echo "Now configure SonicBoom with:"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-13"
