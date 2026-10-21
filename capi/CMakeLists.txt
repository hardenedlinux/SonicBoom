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
  # core/src is private internal state (the decision registry, printed by
  # sb_explain_*), never re-exposed through capi.h.
  PRIVATE ${CMAKE_SOURCE_DIR}/core/include ${CMAKE_SOURCE_DIR}/core/src)

# Build info for sb_build_info(). The commit and tag come from the generated
# header written at build time by cmake/WriteBuildTag.cmake (rewritten only when
# the revision changes, so an unchanged build does not recompile this object);
# the CUDA architectures nvcc really compiles and the LLVM/MLIR that was linked
# are resolved in the top-level CMakeLists.txt and compiled in as string
# literals; see capi.h. The platform comes from the compiler's own target, so it
# follows a cross build rather than the builder's uname.
add_dependencies(sonicboom_capi sonicboom_build_tag)
target_include_directories(sonicboom_capi PRIVATE
  ${CMAKE_BINARY_DIR}/generated)
if(CMAKE_SYSTEM_PROCESSOR)
  set(SONICBOOM_BUILD_PLATFORM "${CMAKE_SYSTEM_PROCESSOR}")
else()
  set(SONICBOOM_BUILD_PLATFORM "unknown")
endif()
target_compile_definitions(sonicboom_capi PRIVATE
  SB_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  SB_BUILD_CUDA_ENABLED=$<IF:$<BOOL:${SONICBOOM_USE_CUDA}>,1,0>
  SB_BUILD_CUDA_ARCHS="${SONICBOOM_BUILD_CUDA_ARCHS}"
  SB_BUILD_CUDA_TOOLKIT="${SONICBOOM_BUILD_CUDA_TOOLKIT}"
  SB_BUILD_LLVM_VERSION="${SONICBOOM_BUILD_LLVM_VERSION}"
  SB_BUILD_LLVM_MAJOR=${SONICBOOM_BUILD_LLVM_MAJOR}   # a number, not a literal
  SB_BUILD_LLVM_ORIGIN="${SONICBOOM_BUILD_LLVM_ORIGIN}"
  SB_BUILD_PLATFORM="${SONICBOOM_BUILD_PLATFORM}")
