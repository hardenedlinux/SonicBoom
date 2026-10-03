# SonicBoom — native tensor runtime
#
# v0 is a generic interpreter: Layer 2 (SonicBoom) → Layer 1 (native-torch) →
# Layer 0 (CPU). CPU-only for v0; CUDA is a deferred manifest amendment.

cmake_minimum_required(VERSION 3.20)

project(SonicBoom
  VERSION 0.0.0
  DESCRIPTION "Native tensor runtime"
  LANGUAGES CXX)

# SonicBoom-owned code targets C++23 with g++-13.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Backend selection. Only CPU is in v0 scope.
option(SONICBOOM_USE_CUDA "Enable CUDA support (deferred — not in v0)" OFF)

# SonicCross regeneration hook: optional build-time code generator for the
# native-torch sources. Exposes the `generate-native-torch` target only when
# guild/SonicCross is available; normal builds use the checked-in generated/
# tree and never require SonicCross.
include(tools/generation/GenerateNativeTorch.cmake)

# The native-torch implementation dependency (c10 + aten/core). It is built
# with its own language constraints (C++17) and CPU-only; its headers must not
# leak into SonicBoom's public Layer 2 interface.
add_subdirectory(third_party/native-torch)

# SonicBoom core: Layer 2 public API (core/include/sonicboom/) + Layer 1 adapter
# (core/layer1/). The adapter is the only SonicBoom-owned code that uses c10/ATen.
add_subdirectory(core)

# M2 Layer 2 boundary tests.
add_subdirectory(tests/core)
