# SonicBoom — native tensor runtime
#
# v0 is a generic interpreter: Layer 2 (SonicBoom) → Layer 1 (native-torch) →
# Layer 0 (CPU). CPU-only for v0; CUDA is a deferred manifest amendment.

cmake_minimum_required(VERSION 3.24)

project(SonicBoom
  VERSION 0.0.3
  DESCRIPTION "Native tensor runtime"
  LANGUAGES C CXX)

# Performance is measured against llama.cpp on the same hardware, so a default
# configure must produce an optimized binary: an unset build type becomes
# Release (-O3 -DNDEBUG for the host; CMAKE_CUDA_FLAGS_RELEASE likewise for
# nvcc). Multiconfig generators choose their own configuration.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (Release = -O3)" FORCE)
endif()

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

# Optional out-of-tree test suites. Empty by default: the public build and its
# tests reference no path outside this repository. When a developer points this
# at an out-of-tree tests/ directory, that directory's suites are built and
# registered into the same build and ctest run (see tests/core/CMakeLists.txt).
set(SONICBOOM_PRIVATE_TESTS_DIR "" CACHE PATH
    "Out-of-tree private test suites directory (optional)")

# Sanitizer instrumentation for SonicBoom-owned code. Empty = a normal build.
# Accepted values are "address;undefined" (ASan + UBSan) and "thread" (TSan);
# both want a separate build directory because the sanitizer runtime must be
# linked into every executable that loads libsonicboom.so.
#
# Scope: only SonicBoom's own targets are instrumented (core, compiler, C ABI,
# tests). third_party/native-torch and the installed MLIR/LLVM are NOT: they are
# implementation dependencies with their own upstream build constraints, and a
# report inside them is not SonicBoom's to triage. Instrumented code still
# calls into them, and ASan/TSan interceptors see the process-wide heap, so the
# runtime boundary is still observed.
set(SONICBOOM_SANITIZE "" CACHE STRING
    "Sanitizers for SonicBoom-owned targets (empty, 'address;undefined', or 'thread')")
set_property(CACHE SONICBOOM_SANITIZE PROPERTY STRINGS
             "" "address;undefined" "thread")
if(SONICBOOM_SANITIZE)
  if(NOT SONICBOOM_SANITIZE STREQUAL "address;undefined"
     AND NOT SONICBOOM_SANITIZE STREQUAL "thread")
    message(FATAL_ERROR
      "SONICBOOM_SANITIZE='${SONICBOOM_SANITIZE}' is not supported; use "
      "'address;undefined' or 'thread' (empty for a normal build).")
  endif()
  string(REPLACE ";" "," _sb_sanitize_list "${SONICBOOM_SANITIZE}")
  set(SONICBOOM_SANITIZE_FLAGS "-fsanitize=${_sb_sanitize_list}")

  # UBSan's vptr check has to be off. `-fsanitize=undefined` instruments every
  # virtual call and refers to `typeinfo` for the static type of the object,
  # but the installed MLIR/LLVM is built without RTTI, so there is no
  # `typeinfo for mlir::BuiltinDialect`, `llvm::raw_ostream`, ... The Boom
  # dialect and the plan emitter call into MLIR virtually throughout, and with
  # the check on libsonicboom.so leaves those references undefined and every
  # test that links it fails. None of the other UBSan checks depend on RTTI.
  set(SONICBOOM_SANITIZE_COMPILE_EXTRA "")
  if("undefined" IN_LIST SONICBOOM_SANITIZE)
    set(SONICBOOM_SANITIZE_COMPILE_EXTRA "-fno-sanitize=vptr")
  endif()
endif()

# Apply the sanitizer flags to one SonicBoom-owned target. Called from the
# subdirectory CMakeLists for every target whose code is ours.
#
# Host languages only: nvcc accepts neither -fsanitize=host-sanitizer form, and
# device code is not instrumented by the host toolchain anyway. The host TUs of
# a CUDA-enabled target are still instrumented, including the MOSS Q8_0 loader's
# std::thread workers.
function(sonicboom_sanitize target)
  if(NOT SONICBOOM_SANITIZE)
    return()
  endif()
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:C,CXX>:${SONICBOOM_SANITIZE_FLAGS}>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:${SONICBOOM_SANITIZE_COMPILE_EXTRA}>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>")
  target_link_options(${target} PRIVATE ${SONICBOOM_SANITIZE_FLAGS})
endfunction()

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

  # Locate nvcc before the language probe and refuse to configure without it.
  # Distributions do not agree on its location (Ubuntu's packaged CUDA puts it
  # in /usr/bin, NVIDIA's runfile in /usr/local/cuda/bin), so PATH is searched
  # rather than any one prefix. A missing compiler must stop the configure here,
  # named: quietly continuing leaves ninja on a stale build graph that fails
  # much later in an unrelated translation unit.
  if(NOT CMAKE_CUDA_COMPILER)
    find_program(CMAKE_CUDA_COMPILER NAMES nvcc nvcc.exe
                 DOC "CUDA compiler (nvcc)")
  endif()
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR
      "SONICBOOM_USE_CUDA=ON but nvcc was not found on PATH.\n"
      "Install the CUDA toolkit, add its bin directory to PATH, or configure "
      "with -DCMAKE_CUDA_COMPILER=/path/to/nvcc.")
  endif()
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

# Build info. Everything sb_build_info() reports is compiled into the library,
# so a benchmark result or a bug report can name the exact binary. See
# capi/include/sonicboom/capi.h.
#
# The revision (SB_BUILD_GIT / SB_BUILD_GIT_TAG) is resolved at BUILD time, not
# at configure time: a configure-time constant goes stale as soon as the next
# commit lands in a tree that is not re-configured. cmake/WriteBuildTag.cmake
# rewrites the generated header only when its content changes, so rebuilding
# after a no-op (or a build of an unchanged revision) neither recompiles capi.cpp
# nor relinks libsonicboom.so.
set(SONICBOOM_BUILD_TAG_HEADER "${CMAKE_BINARY_DIR}/generated/sb_build_tag.h")
set(SONICBOOM_BUILD_TAG_SCRIPT "${CMAKE_SOURCE_DIR}/cmake/WriteBuildTag.cmake")
find_package(Git QUIET)

# What a new revision looks like to the build system: the ref HEAD points at
# (rewritten by a commit), HEAD itself (a detached checkout), the index (a
# commit or a staged change), and packed-refs (a branch packed by gc). The
# header is a custom-command output on these, so the build that follows a
# commit resolves the new revision in the same run — the objects that include
# the header recompile then, not one build later. Uncommitted worktree edits
# are not in this list; they are picked up by the next build or reconfigure.
set(SONICBOOM_BUILD_TAG_DEPS "${SONICBOOM_BUILD_TAG_SCRIPT}")
if(GIT_EXECUTABLE)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --absolute-git-dir
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                  OUTPUT_VARIABLE _sb_git_dir
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _sb_git_dir_rc)
  if(_sb_git_dir_rc EQUAL 0 AND _sb_git_dir)
    list(APPEND SONICBOOM_BUILD_TAG_DEPS "${_sb_git_dir}/HEAD"
                "${_sb_git_dir}/index")
    # These two may legitimately not exist (a fully packed repository, a
    # packed-away branch); a dependency on a missing file would make the build
    # fail, so only real ones are listed.
    if(EXISTS "${_sb_git_dir}/packed-refs")
      list(APPEND SONICBOOM_BUILD_TAG_DEPS "${_sb_git_dir}/packed-refs")
    endif()
    execute_process(COMMAND ${GIT_EXECUTABLE} symbolic-ref -q HEAD
                    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                    OUTPUT_VARIABLE _sb_git_ref
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE _sb_git_ref_rc)
    if(_sb_git_ref_rc EQUAL 0 AND _sb_git_ref AND
       EXISTS "${_sb_git_dir}/${_sb_git_ref}")
      list(APPEND SONICBOOM_BUILD_TAG_DEPS "${_sb_git_dir}/${_sb_git_ref}")
    endif()
  endif()
endif()

# Populate the header at configure time too, so the file exists before the
# first compile and is correct straight after configuring.
execute_process(COMMAND ${CMAKE_COMMAND}
                        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
                        -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
                        -DOUTPUT=${SONICBOOM_BUILD_TAG_HEADER}
                        -P ${SONICBOOM_BUILD_TAG_SCRIPT}
                RESULT_VARIABLE _sb_tag_rc)
if(NOT _sb_tag_rc EQUAL 0)
  message(FATAL_ERROR "failed to write ${SONICBOOM_BUILD_TAG_HEADER}")
endif()
add_custom_command(OUTPUT ${SONICBOOM_BUILD_TAG_HEADER}
  COMMAND ${CMAKE_COMMAND}
          -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
          -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DOUTPUT=${SONICBOOM_BUILD_TAG_HEADER}
          -P ${SONICBOOM_BUILD_TAG_SCRIPT}
  DEPENDS ${SONICBOOM_BUILD_TAG_DEPS}
  COMMENT "sonicboom: resolving build tag"
  VERBATIM)
add_custom_target(sonicboom_build_tag DEPENDS ${SONICBOOM_BUILD_TAG_HEADER})

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
