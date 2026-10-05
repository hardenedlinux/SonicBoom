# native-torch — implementation dependency of SonicBoom (Layer 1).
#
# This tree is the migrated/adapted native implementation derived from PyTorch.
# It is NOT exposed through SonicBoom public headers: core/include/sonicboom/
# must never include <ATen/...>, <c10/...>, or <torch/...>.
#
# Dependency-closure amendment (see design/native-torch-migration.md §15):
#   - torch/headeronly/**   — minimal header-only subset now staged under this
#                             tree; resolves via NATIVE_TORCH_ROOT.
#   - cpuinfo               — EXCLUDED (thread_pool.cpp not on the v0 path).
#   - fmt                   — EXCLUDED (signal_handler/tempfile excluded; env.cpp
#                             fmt include guarded to Windows-only, manifest M4).
# The aten/core closure is not built yet: it needs the M1 generated headers.

set(NATIVE_TORCH_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ---------------------------------------------------------------------------
# Generated/config headers (manifest M2): SonicBoom supplies deterministic
# CPU-only configuration; no upstream generated build output is used.
# ---------------------------------------------------------------------------
set(ATEN_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

# ATen/Config.h — the backend switches are literal (all backends except CUDA stay
# disabled); USE_CUDA/USE_ROCM/USE_DISTRIBUTED are object-like and are NOT defined
# here (they come from compile definitions — see the comment in Config.h.in).
set(ATEN_CONFIG_IN  ${NATIVE_TORCH_ROOT}/aten/src/ATen/Config.h.in)
set(ATEN_CONFIG_OUT ${ATEN_GENERATED_DIR}/ATen/Config.h)
configure_file(${ATEN_CONFIG_IN} ${ATEN_CONFIG_OUT} @ONLY)

# ATen/cuda/CUDAConfig.h — CUDA-backend switches. cuDNN / cusparseLt / hipsparseLt /
# ROCm / MAGMA are all OFF (their sources/optional libs are not in v0); the CUDA
# linalg kernels fall back to cusolver via USE_MAGMA=OFF. Generated only when CUDA
# is enabled (ATen/cuda/*.cu include it).
if(SONICBOOM_USE_CUDA)
  set(AT_CUDNN_ENABLED 0)
  set(AT_CUSPARSELT_ENABLED 0)
  set(AT_HIPSPARSELT_ENABLED 0)
  set(AT_ROCM_ENABLED 0)
  set(AT_MAGMA_ENABLED 0)
  set(NVCC_FLAGS_EXTRA "")
  set(ATEN_CUDA_CONFIG_IN  ${NATIVE_TORCH_ROOT}/aten/src/ATen/cuda/CUDAConfig.h.in)
  set(ATEN_CUDA_CONFIG_OUT ${ATEN_GENERATED_DIR}/ATen/cuda/CUDAConfig.h)
  configure_file(${ATEN_CUDA_CONFIG_IN} ${ATEN_CUDA_CONFIG_OUT} @ONLY)
endif()

# torch/headeronly/macros/cmake_macros.h — all C10_USE_* flags OFF for the
# minimal static CPU build (they are left unset, so #cmakedefine → undef).
set(C10_MACROS_IN  ${NATIVE_TORCH_ROOT}/torch/headeronly/macros/cmake_macros.h.in)
set(C10_MACROS_OUT ${ATEN_GENERATED_DIR}/torch/headeronly/macros/cmake_macros.h)
configure_file(${C10_MACROS_IN} ${C10_MACROS_OUT})

# c10/cuda/impl/cuda_cmake_macros.h — CUDA export macros. C10_CUDA_BUILD_SHARED_LIBS
# is left unset (#cmakedefine → undef) for the static build; the generated header
# resolves via <c10/cuda/impl/...> under ATEN_GENERATED_DIR.
if(SONICBOOM_USE_CUDA)
  set(C10_CUDA_MACROS_IN  ${NATIVE_TORCH_ROOT}/c10/cuda/impl/cuda_cmake_macros.h.in)
  set(C10_CUDA_MACROS_OUT ${ATEN_GENERATED_DIR}/c10/cuda/impl/cuda_cmake_macros.h)
  configure_file(${C10_CUDA_MACROS_IN} ${C10_CUDA_MACROS_OUT})
endif()

# Build-time generation (matching upstream PyTorch's build-time torchgen):
# SonicCross emits the per-op split headers + aggregate headers + Register{key}.cpp
# + Operators*.cpp into ATEN_GENERATED_DIR/ATen (gitignored, never committed).
# `guild` is a required build-time dependency (see tools/generation/).
sonicboom_generate_native_torch("${ATEN_GENERATED_DIR}/ATen")

# Include roots:
#   <c10/...>              resolves under NATIVE_TORCH_ROOT (the parent of c10/)
#   <torch/headeronly/...> resolves under NATIVE_TORCH_ROOT (staged subset)
#   <ATen/...>             resolves under NATIVE_TORCH_ROOT/aten/src
#   generated headers       resolve under ATEN_GENERATED_DIR
set(NATIVE_TORCH_INCLUDE_DIRS
  ${NATIVE_TORCH_ROOT}
  ${NATIVE_TORCH_ROOT}/aten/src
  ${ATEN_GENERATED_DIR})

# ---------------------------------------------------------------------------
# cpuinfo — vendored submodule (manifest §15.D, supersedes the §15.B EXCLUDE).
# Backs c10::ThreadPool (thread_pool.cpp) for CPU intra-op parallelism.
# ---------------------------------------------------------------------------
set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL "" FORCE)
set(CPUINFO_LIBRARY_TYPE static CACHE STRING "" FORCE)
add_subdirectory(${NATIVE_TORCH_ROOT}/third_party/cpuinfo
                 ${CMAKE_CURRENT_BINARY_DIR}/cpuinfo
                 EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# c10 — core + util + macros. The low-level foundation (types, storage,
# allocator, dispatch keys, refcounting). C++17 (native-torch's own standard,
# not SonicBoom's C++23).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE C10_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/c10/core/*.cpp
  ${NATIVE_TORCH_ROOT}/c10/util/*.cpp)

# Sources excluded from v0 (manifest §15.C):
#   signal_handler.cpp — crash/signal handling; not on the v0 path. (fmt)
#   tempfile.cpp       — temp-file helper; not on the v0 path. (fmt)
# thread_pool.cpp is KEPT: §15.D supersedes the §15.B exclusion — cpuinfo is
# required for the full CPU path (thread pool + parallel dispatch).
list(FILTER C10_SOURCES EXCLUDE REGEX "/(signal_handler|tempfile)\\.cpp$")

# Layer A — c10/cuda: CUDA device/stream/event primitives backed by the CUDA
# runtime API only (cuda_runtime.h / cuda_runtime_api.h). No cublas/nvrtc here.
if(SONICBOOM_USE_CUDA)
  file(GLOB C10_CUDA_SOURCES CONFIGURE_DEPENDS
    ${NATIVE_TORCH_ROOT}/c10/cuda/*.cpp
    # impl/CUDAGuardImpl.cpp registers the CUDA DeviceGuardImpl
    # (C10_REGISTER_GUARD_IMPL(CUDA, ...)); without it, the first CPU->CUDA
    # tensor move fails with "not linked with support for cuda devices".
    # impl/CUDATest.cpp is a self-test helper (c10_cuda_test()) and is not
    # part of the runtime, so it is intentionally excluded.
    ${NATIVE_TORCH_ROOT}/c10/cuda/impl/CUDAGuardImpl.cpp)
  list(APPEND C10_SOURCES ${C10_CUDA_SOURCES})
endif()

add_library(c10 STATIC ${C10_SOURCES})
target_include_directories(c10 PRIVATE ${NATIVE_TORCH_INCLUDE_DIRS})
target_compile_features(c10 PRIVATE cxx_std_17)
target_compile_definitions(c10 PRIVATE C10_BUILD_MAIN_LIB)
# cpuinfo backs c10::ThreadPool (thread_pool.cpp) — §15.D.
target_link_libraries(c10 PUBLIC cpuinfo)
# c10/cuda sources include <cuda_runtime.h>; CUDA::cudart provides the include
# path and is linked transitively into libsonicboom.so (PUBLIC here).
if(SONICBOOM_USE_CUDA)
  target_link_libraries(c10 PUBLIC CUDA::cudart)
endif()

# ---------------------------------------------------------------------------
# aten_core — full ATen CPU implementation: dispatcher core + tensor facade +
# native CPU kernels + SonicCross-generated registration/dispatch sources.
# C++17 (native-torch's own standard). CPU-first.
#
# Source layout mirrors upstream aten/src/ATen/CMakeLists.txt for a CPU build:
#   core/ (recursive)         — dispatcher, schema, IValue, boxing
#   *.cpp + detail/ + cpu/    — top-level tensor facade + parallel/thread pool
#   native/*.cpp              — top-level kernel/dispatch files (mixed CPU)
#   native/cpu/               — CPU kernels
#   native/{utils,sparse,quantized(cpu),nested,ao_sparse(cpu)}
#   generated (build dir)     — Register{key}.cpp, Operators*.cpp, Functions.cpp
#
# Excluded (deferred / non-CPU / not on the v0 path):
#   Python{...}*.cpp, Variable{...}*.cpp   — Python bindings / autograd
#   Generator*.cpp, NestedIntSymNodeImpl   — generator + SymInt (§4.1.5)
#   Formatting.cpp                         — torch._C frontend + fmt (deferred)
#   op_registration/{op_registration,infer_schema}.cpp — JIT schema parser + fmt
#   custom_class.cpp, TorchDispatchUtils.cpp — torch::class_ binding + util
#   Register{CUDA,HPU,Mkldnn,...}*.cpp      — non-CPU backend registration
# ---------------------------------------------------------------------------
file(GLOB ATEN_CORE_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/core/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/core/**/*.cpp)
file(GLOB ATEN_FACADE_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/detail/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/cpu/*.cpp)
file(GLOB ATEN_NATIVE_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/cpu/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/utils/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/mkldnn/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/transformers/*.cpp)
file(GLOB ATEN_NATIVE_EXTRA_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/sparse/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/nested/*.cpp)
# Layer B — ATen CUDA infrastructure (ATen/cuda/*.cpp + *.cu): cub, CUDABlas,
# CUDAContext, jiterator (nvrtc), tunable GEMM, allocator hooks. Compiled only
# when CUDA is enabled; the .cu files are compiled by nvcc.
if(SONICBOOM_USE_CUDA)
  file(GLOB ATEN_CUDA_SOURCES CONFIGURE_DEPENDS
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/cuda/*.cpp
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/cuda/*.cu
    # detail/ + tunable/ are ATen/cuda infrastructure too (upstream globs them
    # under cuda/*.cpp / cuda/*.cu with the subdirs). detail/CUDAHooks.cpp
    # registers the CUDA backend hooks (without it at::GetCUDAHooks() is null);
    # detail/BLASConstants.cu + IndexUtils.cu define get_cublas_device_zero /
    # maybeOverlappingIndices; tunable/{Tunable,StreamTimer}.cpp define the
    # tunable-GEMM context (TuningContext/TuningResultsManager/getTuningContext).
    # nvrtc_stub/ is intentionally NOT globbed: it is the dlopen-based nvrtc
    # stub used only when USE_NVRTC=0, and jiterator.cu (top-level) already
    # provides nvrtc when it is present.
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/cuda/detail/*.cpp
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/cuda/detail/*.cu
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/cuda/tunable/*.cpp)
  # Layer C — dense CUDA kernels (native/cuda/*.cu + *.cpp + linalg/*.cpp).
  # linalg compiles against cusolver (AT_MAGMA_ENABLED()=0 via CUDAConfig.h).
  # Excluded: the four CUTLASS-backed kernels (cutlass is not vendored — see
  # the CUDA plan). cutlass_extensions/ + GroupMMCommon.cuh + cutlass_common.cuh
  # are header-only and only pulled in by those excluded .cu files. int4mm.cu
  # uses the toolkit <mma.h> (not cutlass) and is kept.
  file(GLOB ATEN_CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/cuda/*.cu
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/cuda/*.cpp
    ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/cuda/linalg/*.cpp
    # Generated structured-ufunc CUDA kernels (UfuncCUDA_*.cu, e.g. add) — the
    # CUDA analogue of the UfuncCPUKernel_*.cpp wired for CPU below. Without
    # this, structured_ufunc_add_CUDA::impl is undefined and aten::add on CUDA
    # has no kernel.
    ${ATEN_GENERATED_DIR}/ATen/UfuncCUDA_*.cu)
  list(FILTER ATEN_CUDA_KERNEL_SOURCES EXCLUDE REGEX
    "/(GroupMM|MixedDtypesLinear|RowwiseScaledMM|ScaledGroupMM)\\.cu$")
endif()
# Quantized facade (aten/src/ATen/quantized/) — the Tensor-level quantizer API
# (TensorBase::quantizer(), QTensorImpl, make_per_*_affine_quantizer). This is
# the tensor facade, not the deferred quantized INT8 compute kernels.
file(GLOB ATEN_QUANTIZED_FACADE_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/quantized/*.cpp)
# Quantized boundary kernels: float<->quantized boundary ops (fake_quantize,
# dequantize, quantize_per_*, quantized_copy_from_float_, qparams). These are
# fbgemm-FREE — fbgemm code is guarded by #ifdef USE_FBGEMM and compiled out.
# The fbgemm-packed linear prepack/unpack files (library/qlinear_unpack/
# qconv_unpack) are deferred and excluded (QuantizedStubs.cpp supplies the
# register_linear_params + wrapped-linear symbols as throwing stubs).
file(GLOB ATEN_QUANTIZED_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/quantized/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/quantized/cpu/MakePerTensorQuantizedTensor.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/quantized/cpu/fused_obs_fake_quant.cpp)
list(FILTER ATEN_QUANTIZED_SOURCES EXCLUDE REGEX
  "/(library|qlinear_unpack|qconv_unpack)\\.cpp$")
file(GLOB ATEN_GENERATED_SOURCES CONFIGURE_DEPENDS
  ${ATEN_GENERATED_DIR}/ATen/*.cpp
  ${ATEN_GENERATED_DIR}/ATen/core/*.cpp)

set(ATEN_SOURCES
  ${ATEN_CORE_SOURCES}
  ${ATEN_FACADE_SOURCES}
  ${ATEN_QUANTIZED_FACADE_SOURCES}
  ${ATEN_NATIVE_SOURCES}
  ${ATEN_NATIVE_EXTRA_SOURCES}
  ${ATEN_CUDA_SOURCES}
  ${ATEN_CUDA_KERNEL_SOURCES}
  ${ATEN_QUANTIZED_SOURCES}
  ${ATEN_GENERATED_SOURCES})

# torch::Library registration frontend (migrated subset — see §22 report).
# NOTE: torch/library.h is NOT header-only — it declares torch::Library::_def/
# _impl/_fallback and torch::CppFunction out-of-line, implemented in
# aten/src/ATen/core/library.cpp (kept in ATEN_CORE_SOURCES above; it backs
# every TORCH_LIBRARY / TORCH_LIBRARY_IMPL static init). The JIT schema-parser
# translation units below back m.def("...")'s parseSchema -> lexer ->
# schema_type_parser path. parse_string_literal.h and source_range_serialization.h
# are header-only; strtod.cpp backs strtod_c (locale-independent number parsing).
list(APPEND ATEN_SOURCES
  ${NATIVE_TORCH_ROOT}/torch/csrc/jit/frontend/error_report.cpp
  ${NATIVE_TORCH_ROOT}/torch/csrc/jit/frontend/function_schema_parser.cpp
  ${NATIVE_TORCH_ROOT}/torch/csrc/jit/frontend/lexer.cpp
  ${NATIVE_TORCH_ROOT}/torch/csrc/jit/frontend/schema_type_parser.cpp
  ${NATIVE_TORCH_ROOT}/torch/csrc/jit/frontend/source_range.cpp
  ${NATIVE_TORCH_ROOT}/torch/csrc/jit/frontend/strtod.cpp)

# Backend Context stubs (self-contained; provide metal_copy_/vulkan_copy_/
# is_vulkan_available when those backends are not linked), the XNNPACK shim
# (no-op xnnpack::* when USE_XNNPACK is off), and the caffe2 embedding
# perfkernel (generic BASE impl; AVX2/SVE variants are not built in v0).
# mkl/SpectralOps.cpp: with AT_MKL_ENABLED()=0 and AT_POCKETFFT_ENABLED()=0 it
# compiles only the #else branch — REGISTER_NO_CPU_DISPATCH for
# fft_fill_with_conjugate_symmetry_stub + throwing _fft_*_mkl stubs (fft is
# deferred; see CLAUDE.md §4.1.5).
list(APPEND ATEN_SOURCES
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/vulkan/Context.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/metal/Context.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/xnnpack/Shim.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/mkl/SpectralOps.cpp
  ${NATIVE_TORCH_ROOT}/caffe2/perfkernels/embedding_lookup_idx.cc)

# Deferred / non-v0 exclusions (see header comment):
#   (library.cpp is NOT excluded: it is the torch::Library registration impl
#    required by every generated TORCH_LIBRARY / TORCH_LIBRARY_IMPL static init.)
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/(Formatting|TorchDispatchUtils)\\.cpp$")
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/(Generator|GeneratorForPrivateuseone|NestedIntSymNodeImpl)\\.cpp$")
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/PythonFallbackKernel\\.cpp$|/VariableFallbackKernel\\.cpp$")
# Dispatch fallback kernels registered via the torch::Library frontend
# (torch/library.h — torch/csrc/api C++ frontend + fmt; deferred, see §22 report):
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/(BackendSelectFallbackKernel|MetaFallbackKernel)\\.cpp$")
# gtest unit tests that live inside the core/ source tree (not part of libATen):
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/.*_test\\.cpp$")
# Core CPU compute now INCLUDED (were previously excluded as "LAPACK/caffe2"):
#   Version.cpp         — adapted in-tree to drop caffe2::GetBuildOptions()
#   EmbeddingBag.cpp    — caffe2 perfkernels migrated (caffe2/perfkernels/)
#   {Batch,}LinearAlgebra* — LAPACK-guarded (#if AT_BUILD_WITH_LAPACK()=0)
# Non-CPU backend registration (refined as the CUDA build completes):
#   RegisterQuantized*  — quantized dtypes deferred (needs fbgemm/qnnpack)
#   Register{HPU,Mkldnn}* — other backends not in v0
#   Register{SparseCUDA,SparseCsrCUDA,NestedTensorCUDA}* — sparse/nested CUDA
#     kernels live in native/{sparse,nested}/cuda/ and are not yet compiled
#     (separate source glob, deferred). RegisterCUDA (dense) is compiled only
#     when SONICBOOM_USE_CUDA — see the second filter below.
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/Register.*(HPU|Mkldnn|Quantized|SparseCUDA|SparseCsrCUDA|NestedTensorCUDA).*\\.cpp$")
if(NOT SONICBOOM_USE_CUDA)
  list(FILTER ATEN_SOURCES EXCLUDE REGEX
    "/Register.*CUDA.*\\.cpp$")
endif()
# SonicCross emits BOTH the per-op split files (RegisterCPU_0..N.cpp,
# Operators_0..N.cpp) and the monolithic "Everything" variants. With
# --per-operator-headers we compile the split files only; the Everything
# variants duplicate the same registrations and cause multiple-definition
# link errors, so they are excluded here.
list(FILTER ATEN_SOURCES EXCLUDE REGEX
  "/.*Everything\\.cpp$")

# CPU kernels are compiled for the DEFAULT capability only (no AVX2/AVX512
# multi-dispatch in v0 — deferred). CPU_CAPABILITY must be defined so that
# REGISTER_DISPATCH in native/cpu + the generated UfuncCPUKernel sources
# resolves, and so `inline namespace CPU_CAPABILITY` becomes a named namespace
# (matching upstream's `-DCPU_CAPABILITY=DEFAULT -DCPU_CAPABILITY_DEFAULT`).
file(GLOB ATEN_CPU_CAPABILITY_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/native/cpu/*.cpp
  ${ATEN_GENERATED_DIR}/ATen/UfuncCPUKernel_*.cpp)
set_source_files_properties(${ATEN_CPU_CAPABILITY_SOURCES} PROPERTIES
  COMPILE_DEFINITIONS "CPU_CAPABILITY=DEFAULT;CPU_CAPABILITY_DEFAULT")

add_library(aten_core STATIC ${ATEN_SOURCES})
target_include_directories(aten_core PRIVATE
  ${NATIVE_TORCH_INCLUDE_DIRS}
  ${NATIVE_TORCH_ROOT}/third_party/cpuinfo/include
  ${NATIVE_TORCH_ROOT}/third_party/sleef/include
  ${NATIVE_TORCH_ROOT}/third_party/FP16/include
  ${NATIVE_TORCH_ROOT}/third_party/fmt/include)
target_compile_features(aten_core PRIVATE cxx_std_17)
target_compile_definitions(aten_core PRIVATE C10_BUILD_MAIN_LIB FMT_HEADER_ONLY)
target_link_libraries(aten_core PUBLIC c10)
# ATen/cuda sources need cublas/cusolver/cusparse/cufft/nvrtc (include dirs + the
# final link into libsonicboom.so). cudart comes transitively via c10. Linked
# PRIVATE so native-torch CUDA libs never leak into Layer 2 consumers.
if(SONICBOOM_USE_CUDA)
  target_link_libraries(aten_core PRIVATE
    CUDA::cublas
    CUDA::cusolver
    CUDA::cusparse
    CUDA::cufft
    CUDA::nvrtc)
  # USE_CUDA is the object-like master switch the ATen/c10 CUDA code checks via
  # `#if defined(USE_CUDA)` (it is NOT in Config.h — see Config.h.in). USE_ROCM is
  # deliberately left undefined so the ROCm-only paths (e.g. hipsolver in
  # CUDAContextLight.h) stay compiled out.
  target_compile_definitions(aten_core PRIVATE USE_CUDA=1)
  # Cub/CCCL wrapping + CUDA feature flags, mirroring upstream
  # cmake/Dependencies.cmake: wrap CUB's namespace as at_cuda_detail (so
  # at::native::cub aliases it — see aten/src/ATen/cuda/cub.cuh), suppress
  # cusparse deprecation spam, enable FP16 intrinsics, and — critically —
  # disable __half's built-in arithmetic/conversion operators (the
  # __CUDA_NO_HALF_* macros) so c10::Half comparisons disambiguate to the
  # operator float() path instead of colliding with cuda_fp16.hpp's
  # operator>=/(__half,__half) (see ActivationHardshrinkKernel.cu).
  target_compile_definitions(aten_core PRIVATE
    CUB_WRAPPED_NAMESPACE=at_cuda_detail
    DISABLE_CUSPARSE_DEPRECATED
    CUDA_HAS_FP16=1
    __CUDA_NO_HALF_OPERATORS__
    __CUDA_NO_HALF_CONVERSIONS__
    __CUDA_NO_HALF2_OPERATORS__
    __CUDA_NO_BFLOAT16_CONVERSIONS__)
  target_compile_options(aten_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr -Wno-deprecated-gpu-targets>)
endif()

# ---------------------------------------------------------------------------
# dispatch_proof — M1 smoke test: schema -> registerDef -> registerImpl ->
# findSchema -> callBoxed. Links the dispatcher core and proves registration
# + lookup + boxed invocation without the Tensor op facade (manifest M1).
# ---------------------------------------------------------------------------
add_executable(dispatch_proof tests/dispatch_proof.cpp)
target_include_directories(dispatch_proof PRIVATE ${NATIVE_TORCH_INCLUDE_DIRS})
target_compile_features(dispatch_proof PRIVATE cxx_std_17)
target_link_libraries(dispatch_proof PRIVATE aten_core)

# ---------------------------------------------------------------------------
# numeric_proof — Stage D numeric-correctness smoke test. Goes through the real
# Tensor facade + native CPU kernels (factory -> binary -> unary -> reduction)
# and asserts concrete results.
#
# aten_core is a STATIC library whose generated Register{Schema,CPU,...}*.cpp
# register kernels via static initializers (TORCH_LIBRARY / TORCH_LIBRARY_IMPL).
# The linker would drop those objects (no symbol is referenced from the test),
# so we force them in with --whole-archive. This is the static-linking analogue
# of upstream's shared libtorch, whose static initializers always run.
# ---------------------------------------------------------------------------
add_executable(numeric_proof tests/numeric_proof.cpp)
target_include_directories(numeric_proof PRIVATE ${NATIVE_TORCH_INCLUDE_DIRS})
target_compile_features(numeric_proof PRIVATE cxx_std_17)
target_link_libraries(numeric_proof PRIVATE
  "-Wl,--whole-archive" aten_core "-Wl,--no-whole-archive")
