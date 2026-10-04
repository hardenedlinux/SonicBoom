# SonicBoom stable C ABI (Layer 3).
#
# The C ABI is the long-term binary-compatibility boundary consumed by language
# bindings (first Guile) and other C callers. It is a thin facade over the
# SonicBoom core, compiled as an OBJECT library so its `extern "C"` symbols are
# linked into and exported from libsonicboom.so — the single published core
# library that FFI consumers load. This header/source pair exposes no C++,
# MLIR, LLVM, ATen, or c10 types across the boundary.

add_library(sonicboom_capi OBJECT src/capi.cpp)

# capi.h is the public header for C consumers; the sx:: core headers are a
# private implementation dependency (never re-exposed).
target_include_directories(sonicboom_capi
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_SOURCE_DIR}/core/include)
