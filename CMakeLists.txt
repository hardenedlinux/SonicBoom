# SonicBoom — native tensor runtime
#
# v0 is a generic interpreter: Layer 2 (SonicBoom) → Layer 1 (native-torch) →
# Layer 0 (CPU). CPU-only for v0; CUDA is a deferred manifest amendment.

cmake_minimum_required(VERSION 3.20)

project(SonicBoom
  VERSION 0.0.0
  DESCRIPTION "Native tensor runtime"
  LANGUAGES C CXX)

# SonicBoom-owned code targets C++23 with g++-13.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable CTest up front so every subdirectory (tests/core, bindings/guile)
# registers its suites under a single `ctest` invocation.
enable_testing()

# libsonicboom.so is the final published core library; native-torch and MLIR are
# statically linked into it, so all static libraries must be position-independent.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Backend selection. CPU is always on; CUDA is opt-in (SONICBOOM_USE_CUDA=ON),
# wiring the native-torch CUDA sources into libsonicboom.so and linking the
# CUDA runtime. cuDNN / MAGMA / cutlass remain disabled (see the CUDA plan).
option(SONICBOOM_USE_CUDA "Enable CUDA support (native-torch CUDA kernels)" OFF)

if(SONICBOOM_USE_CUDA)
  # RTX 3050 = compute capability 8.6 (sm_86).
  set(CMAKE_CUDA_ARCHITECTURES 86)
  # native-torch headers (c10/util/intrusive_ptr.h) use C++20's operator<=>,
  # so .cu host compilation needs C++20 (nvcc 12.0's highest supported host
  # standard). This mirrors the C++23 the CPU native-torch TUs already compile
  # under; C++20 is the closest nvcc 12.0 can reach.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

# SonicCross build-time code generator for the native-torch sources (matches
# upstream PyTorch's build-time torchgen). Defines the generation hook used by
# third_party/native-torch; `guild` is a required build-time dependency.
include(tools/generation/GenerateNativeTorch.cmake)

# The native-torch implementation dependency (c10 + aten/core). It is built
# with its own language constraints (C++17) and CPU-only; its headers must not
# leak into SonicBoom's public Layer 2 interface.
add_subdirectory(third_party/native-torch)

# MLIR integration: link the official MLIR CMake targets into libsonicboom.so.
# MLIR/LLVM are built once from third_party/llvm-project via
# tools/mlir/build-mlir.sh and installed to build/llvm-mlir-install; they are not
# a runtime dependency and no system MLIR/LLVM is used.
option(SONICBOOM_ENABLE_MLIR "Link MLIR into libsonicboom.so" ON)
if(SONICBOOM_ENABLE_MLIR)
  set(MLIR_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build/llvm-mlir-install"
      CACHE PATH "Prefix where tools/mlir/build-mlir.sh installed LLVM/MLIR")
  set(MLIR_DIR "${MLIR_INSTALL_PREFIX}/lib/cmake/mlir"
      CACHE PATH "MLIR CMake package directory")
  set(LLVM_DIR "${MLIR_INSTALL_PREFIX}/lib/cmake/llvm"
      CACHE PATH "LLVM CMake package directory")
  find_package(MLIR CONFIG REQUIRED)
endif()

# SonicBoom stable C ABI (Layer 3). Built as an object library and linked into
# libsonicboom.so; added before core so its target is available there.
add_subdirectory(capi)

# SonicBoom core: Layer 2 public API (core/include/sonicboom/) + Layer 1 adapter
# (core/layer1/). The adapter is the only SonicBoom-owned code that uses c10/ATen.
add_subdirectory(core)

# M2 Layer 2 boundary tests.
add_subdirectory(tests/core)

# Guile binding (Layer 3): Scheme sources + tests. The binding targets Guile
# 3.0.9+.
add_subdirectory(bindings/guile)
