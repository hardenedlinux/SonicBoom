# SonicBoom — native tensor runtime
#
# v0 is a generic interpreter: Layer 2 (SonicBoom) → Layer 1 (native-torch) →
# Layer 0 (CPU). CPU-only for v0; CUDA is a deferred manifest amendment.

cmake_minimum_required(VERSION 3.24)

project(SonicBoom
  VERSION 0.0.3
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

# Vulkan is a community target. This option only builds the Vulkan *stub*
# (VulkanBackend: a Backend subclass with zero kernels — no real Vulkan compute,
# no Vulkan loader linked). It is OFF by default and exercises only the backend
# authoring scaffold; the real Vulkan plugin is future community work.
option(SONICBOOM_USE_VULKAN "Enable the Vulkan backend stub (community scaffold)" OFF)

if(SONICBOOM_USE_CUDA)
  # Default to the GPU detected at build time (native); override with
  # -DCMAKE_CUDA_ARCHITECTURES=<arch>, required on GPU-less build machines.
  # Must be set before enable_language(CUDA), which otherwise adopts the
  # compiler default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA SM archs (native = detected GPU)")
  endif()

  # native-torch headers (c10/util/intrusive_ptr.h) use C++20's operator<=>,
  # so .cu host compilation needs C++20 (nvcc 12.0's highest supported host
  # standard). This mirrors the C++23 the CPU native-torch TUs already compile
  # under; C++20 is the closest nvcc 12.0 can reach.
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # Compress embedded SASS in every fatbin. The CUDA driver decompresses a
  # module when it is loaded (already a one-time cost before the first launch),
  # so kernel execution is unaffected while libsonicboom.so loses most of the
  # native-torch ATen kernel image size (~70% of .nv_fatbin in a sample TU).
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin -compress-all")
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
# MLIR/LLVM are a build-time input, not a runtime dependency.
#
# MLIR is REQUIRED. The Boom IR dialect and the rewrite pipeline that folds the
# emitter's graph into the plan are the only rewrite path there is: a build
# without the compiler library would run unfused graphs, so "no MLIR" is a
# configure-time error rather than a supported configuration.
#
# Two ways to point at an LLVM/MLIR installation:
#   1. -DMLIR_INSTALL_PREFIX=<prefix>  the prefix tools/mlir/build-mlir.sh
#      installs to (default: <source>/build/llvm-mlir-install);
#   2. -DMLIR_DIR=<dir> [-DLLVM_DIR=<dir>]  the CMake package directories of a
#      distribution or otherwise external LLVM/MLIR.
# Either way the third_party/llvm-project submodule is NOT needed: the
# submodule is only an input to tools/mlir/build-mlir.sh, which produces the
# install. A build with the submodule empty is the supported external case.
set(MLIR_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/build/llvm-mlir-install"
    CACHE PATH "Prefix where tools/mlir/build-mlir.sh installed LLVM/MLIR")

# A directory given with -DMLIR_DIR/-DLLVM_DIR is used as is, working or not:
# a path the user chose that does not contain the package is their error to
# see, named in the message below, not something to quietly replace. Only a
# path this file derived itself (recorded in the *_DERIVED markers) is
# re-derived, so changing MLIR_INSTALL_PREFIX takes effect on the next
# configure without a fresh build directory.
if(NOT DEFINED SONICBOOM_MLIR_DIR_DERIVED AND MLIR_DIR)
  set(_sb_mlir_dir_given TRUE)
endif()
if(NOT DEFINED SONICBOOM_LLVM_DIR_DERIVED AND LLVM_DIR)
  set(_sb_llvm_dir_given TRUE)
endif()
if(NOT MLIR_DIR OR (NOT _sb_mlir_dir_given AND NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake"))
  set(MLIR_DIR "${MLIR_INSTALL_PREFIX}/lib/cmake/mlir"
      CACHE PATH "MLIR CMake package directory" FORCE)
  set(SONICBOOM_MLIR_DIR_DERIVED TRUE CACHE INTERNAL "MLIR_DIR was derived from MLIR_INSTALL_PREFIX")
endif()
if(NOT LLVM_DIR OR (NOT _sb_llvm_dir_given AND NOT EXISTS "${LLVM_DIR}/LLVMConfig.cmake"))
  get_filename_component(_sb_mlir_cmake_dir "${MLIR_DIR}" DIRECTORY)
  if(EXISTS "${_sb_mlir_cmake_dir}/llvm/LLVMConfig.cmake")
    set(LLVM_DIR "${_sb_mlir_cmake_dir}/llvm"
        CACHE PATH "LLVM CMake package directory" FORCE)
  else()
    set(LLVM_DIR "${MLIR_INSTALL_PREFIX}/lib/cmake/llvm"
        CACHE PATH "LLVM CMake package directory" FORCE)
  endif()
  set(SONICBOOM_LLVM_DIR_DERIVED TRUE CACHE INTERNAL "LLVM_DIR was derived from MLIR_DIR")
endif()

if(NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
  message(FATAL_ERROR
    "MLIR is required but was not found at MLIR_DIR=${MLIR_DIR}.\n"
    "Build and install it from the submodule:\n"
    "  git submodule update --init third_party/llvm-project\n"
    "  tools/mlir/build-mlir.sh            # installs to ${MLIR_INSTALL_PREFIX}\n"
    "or point this build at an existing LLVM/MLIR with\n"
    "  -DMLIR_DIR=<prefix>/lib/cmake/mlir -DLLVM_DIR=<prefix>/lib/cmake/llvm\n"
    "(SonicBoom's MLIR path is tested with LLVM/MLIR 23.)")
endif()

find_package(MLIR CONFIG REQUIRED)

# The MLIR path is developed and tested against one LLVM major. A different
# one is a warning, not an error: the failure mode is a lowering that needs
# adjusting, not a binary that cannot be built.
if(LLVM_VERSION_MAJOR AND NOT LLVM_VERSION_MAJOR EQUAL 23)
  message(WARNING
    "SonicBoom's MLIR path is tested with LLVM/MLIR 23; this build uses "
    "${LLVM_PACKAGE_VERSION}. The lowering may need changes for this version; "
    "treat resulting compile/runtime failures as untriaged until it runs on 23.")
endif()

# Build info. Everything sb_build_info() reports is resolved here, once, and
# compiled into the library, so a benchmark result or a bug report can name the
# exact binary. See capi/include/sonicboom/capi.h.
set(SONICBOOM_BUILD_GIT "unknown")
set(SONICBOOM_BUILD_GIT_TAG "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  OUTPUT_VARIABLE _sb_git OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _sb_git_rc)
  if(_sb_git_rc EQUAL 0 AND _sb_git)
    set(SONICBOOM_BUILD_GIT "${_sb_git}")
    execute_process(COMMAND ${GIT_EXECUTABLE} status --porcelain
                            --untracked-files=no
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE _sb_git_dirty
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_sb_git_dirty)
      set(SONICBOOM_BUILD_GIT "${SONICBOOM_BUILD_GIT}-dirty")
    endif()
    # The tag is the human name of the same revision (`git describe` adds the
    # commit distance when HEAD is past the last tag); a bare commit cannot say
    # which release a build belongs to.
    execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always --dirty
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE _sb_git_tag OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE _sb_tag_rc)
    if(_sb_tag_rc EQUAL 0 AND _sb_git_tag)
      set(SONICBOOM_BUILD_GIT_TAG "${_sb_git_tag}")
    endif()
  endif()
endif()

# CPU architectures. CMAKE_CUDA_ARCHITECTURES may be a token ('native', 'all',
# 'all-major') that only nvcc resolves, and it resolves it per compilation — so
# resolve it here the same way, with a dry run, rather than report the token.
set(SONICBOOM_BUILD_CUDA_ARCHS "")
if(SONICBOOM_USE_CUDA AND CMAKE_CUDA_COMPILER)
  set(_sb_archs "${CMAKE_CUDA_ARCHITECTURES}")
  if(_sb_archs MATCHES "^(native|all|all-major)$")
    set(_sb_arch_probe "${CMAKE_BINARY_DIR}/sb-arch-probe.cu")
    file(WRITE "${_sb_arch_probe}" "// SonicBoom build-info arch probe\n")
    execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} -arch=${_sb_archs} --dryrun -cubin
              -o ${CMAKE_BINARY_DIR}/sb-arch-probe.cubin "${_sb_arch_probe}"
      OUTPUT_VARIABLE _sb_arch_out ERROR_VARIABLE _sb_arch_err
      RESULT_VARIABLE _sb_arch_rc)
    file(REMOVE "${_sb_arch_probe}" "${CMAKE_BINARY_DIR}/sb-arch-probe.cubin")
    if(_sb_arch_rc EQUAL 0)
      # The dry run prints the resolved toolchain commands; the SM number is the
      # one cicc is invoked with (`-arch compute_86`), which is exactly what
      # ptxas then compiles for (`-arch=sm_86`).
      string(REGEX MATCHALL "compute_[0-9]+" _sb_arch_hits "${_sb_arch_out}${_sb_arch_err}")
      foreach(_hit IN LISTS _sb_arch_hits)
        string(REGEX REPLACE "compute_" "" _num "${_hit}")
        list(APPEND _sb_arch_nums "${_num}")
      endforeach()
      list(REMOVE_DUPLICATES _sb_arch_nums)
      list(JOIN _sb_arch_nums "," SONICBOOM_BUILD_CUDA_ARCHS)
    endif()
    if(NOT SONICBOOM_BUILD_CUDA_ARCHS)
      set(SONICBOOM_BUILD_CUDA_ARCHS "${_sb_archs}")  # unresolvable: report the token
    endif()
  else()
    # A plain list, e.g. "86" or "80;86"; CMake also allows '86-real'/'86-virtual'.
    foreach(_a IN LISTS _sb_archs)
      string(REGEX REPLACE "-(real|virtual)$" "" _num "${_a}")
      list(APPEND _sb_arch_nums "${_num}")
    endforeach()
    list(REMOVE_DUPLICATES _sb_arch_nums)
    list(JOIN _sb_arch_nums "," SONICBOOM_BUILD_CUDA_ARCHS)
  endif()
endif()

set(SONICBOOM_BUILD_CUDA_TOOLKIT "")
if(SONICBOOM_USE_CUDA AND CUDAToolkit_VERSION)
  set(SONICBOOM_BUILD_CUDA_TOOLKIT "${CUDAToolkit_VERSION}")
endif()

# MLIR is required, so these are always a real version/origin; the "none"
# defaults are kept only because sb_build_info() declares the fields and a
# CPU/toolchain-only reader should not have to know that.
set(SONICBOOM_BUILD_LLVM_VERSION "none")
set(SONICBOOM_BUILD_LLVM_MAJOR "0")
set(SONICBOOM_BUILD_LLVM_ORIGIN "none")
if(LLVM_PACKAGE_VERSION)
  set(SONICBOOM_BUILD_LLVM_VERSION "${LLVM_PACKAGE_VERSION}")
else()
  set(SONICBOOM_BUILD_LLVM_VERSION "unknown")
endif()
if(LLVM_VERSION_MAJOR)
  set(SONICBOOM_BUILD_LLVM_MAJOR "${LLVM_VERSION_MAJOR}")
endif()
# 'bundled' = the install tools/mlir/build-mlir.sh produces inside this source
# tree; anything else is an external LLVM/MLIR.
string(FIND "${MLIR_INSTALL_PREFIX}" "${CMAKE_SOURCE_DIR}" _sb_prefix_in_src)
if(_sb_prefix_in_src EQUAL 0)
  set(SONICBOOM_BUILD_LLVM_ORIGIN "bundled")
else()
  set(SONICBOOM_BUILD_LLVM_ORIGIN "external")
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
