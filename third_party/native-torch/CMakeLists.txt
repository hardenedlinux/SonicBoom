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

# ATen/Config.h
set(ATEN_CONFIG_IN  ${NATIVE_TORCH_ROOT}/aten/src/ATen/Config.h.in)
set(ATEN_CONFIG_OUT ${ATEN_GENERATED_DIR}/ATen/Config.h)
configure_file(${ATEN_CONFIG_IN} ${ATEN_CONFIG_OUT} COPYONLY)

# torch/headeronly/macros/cmake_macros.h — all C10_USE_* flags OFF for the
# minimal static CPU build (they are left unset, so #cmakedefine → undef).
set(C10_MACROS_IN  ${NATIVE_TORCH_ROOT}/torch/headeronly/macros/cmake_macros.h.in)
set(C10_MACROS_OUT ${ATEN_GENERATED_DIR}/torch/headeronly/macros/cmake_macros.h)
configure_file(${C10_MACROS_IN} ${C10_MACROS_OUT})

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
# c10 — core + util + macros. The low-level foundation (types, storage,
# allocator, dispatch keys, refcounting). C++17 (native-torch's own standard,
# not SonicBoom's C++23).
# ---------------------------------------------------------------------------
file(GLOB_RECURSE C10_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/c10/core/*.cpp
  ${NATIVE_TORCH_ROOT}/c10/util/*.cpp)

# Sources excluded from v0 (manifest amendment §15.B/§15.C):
#   thread_pool.cpp    — parallelism primitive used by native kernels (BLACK
#                        BOX); not referenced by the c10 core compile closure.
#                        Excluding it removes the cpuinfo dependency.
#   signal_handler.cpp — crash/signal handling; not on the v0 path. (fmt)
#   tempfile.cpp       — temp-file helper; not on the v0 path. (fmt)
list(FILTER C10_SOURCES EXCLUDE REGEX "/(thread_pool|signal_handler|tempfile)\\.cpp$")

add_library(c10 STATIC ${C10_SOURCES})
target_include_directories(c10 PRIVATE ${NATIVE_TORCH_INCLUDE_DIRS})
target_compile_features(c10 PRIVATE cxx_std_17)
target_compile_definitions(c10 PRIVATE C10_BUILD_MAIN_LIB)

# ---------------------------------------------------------------------------
# aten/core — dispatcher core (schema, IValue, dispatcher, boxing). C++17.
#
# This is the minimal v0 compile set (manifest M1): the schema / type / IValue
# / dispatcher / boxing closure that supports operator registration and
# dispatch lookup. It does NOT yet contain the generated operator headers
# (ATen/ops/*.h) or the tensor op facade — those are added with the M1 op
# subset in a later step.
#
# Excluded (deferred, outside the v0 M1 boundary):
#   library.cpp                  — torch/csrc library frontend + fmt
#   Formatting.cpp               — tensor pretty-print (fmt + op methods)
#   op_registration/*.cpp        — function_schema_parser.h (JIT frontend) + fmt
#   adaption.cpp                 — ATen/Tensor.h top-level op facade
#   Tensor.cpp                   — needs generated ATen/ops/*.h (added with ops)
#   Python{...}*.cpp             — Python bindings (not a v0 runtime dep)
#   Variable{...}*.cpp           — autograd / variable hooks (deferred)
#   Generator*.cpp               — generator semantics (deferred, §4.1.5)
#   NestedIntSymNodeImpl.cpp     — SymInt (deferred, §4.1.5)
# ---------------------------------------------------------------------------
file(GLOB_RECURSE ATEN_CORE_SOURCES CONFIGURE_DEPENDS
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/core/*.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/core/**/*.cpp)

list(FILTER ATEN_CORE_SOURCES EXCLUDE REGEX
  "/(library|Formatting|adaption|Tensor|Generator|GeneratorForPrivateuseone|NestedIntSymNodeImpl|BackendSelectFallbackKernel|MetaFallbackKernel|custom_class|TorchDispatchUtils)\\.cpp$")
list(FILTER ATEN_CORE_SOURCES EXCLUDE REGEX
  "/op_registration/(op_registration|infer_schema)\\.cpp$")
list(FILTER ATEN_CORE_SOURCES EXCLUDE REGEX
  "/PythonFallbackKernel\\.cpp$")
list(FILTER ATEN_CORE_SOURCES EXCLUDE REGEX
  "/Variable(FallbackKernel|HooksInterface)\\.cpp$")

# M1 hand-authored sources outside the aten/core glob (manifest M1):
#   SequenceNumber.cpp   — sequence counter (COPY, tiny)
#   record_function.cpp  — no-op profiler hooks (ADAPT, profiler deferred)
#   ops/minimal_ops.cpp  — minimal generated operator-method definitions
# find_schema_differences.cpp lives under core/op_registration/ and is covered
# by the core/** glob above (it is not in the exclusion list).
list(APPEND ATEN_CORE_SOURCES
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/SequenceNumber.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/record_function.cpp
  ${NATIVE_TORCH_ROOT}/aten/src/ATen/ops/minimal_ops.cpp)

add_library(aten_core STATIC ${ATEN_CORE_SOURCES})
target_include_directories(aten_core PRIVATE ${NATIVE_TORCH_INCLUDE_DIRS})
target_compile_features(aten_core PRIVATE cxx_std_17)
target_compile_definitions(aten_core PRIVATE C10_BUILD_MAIN_LIB)
target_link_libraries(aten_core PUBLIC c10)

# ---------------------------------------------------------------------------
# dispatch_proof — M1 smoke test: schema -> registerDef -> registerImpl ->
# findSchema -> callBoxed. Links the dispatcher core and proves registration
# + lookup + boxed invocation without the Tensor op facade (manifest M1).
# ---------------------------------------------------------------------------
add_executable(dispatch_proof tests/dispatch_proof.cpp)
target_include_directories(dispatch_proof PRIVATE ${NATIVE_TORCH_INCLUDE_DIRS})
target_compile_features(dispatch_proof PRIVATE cxx_std_17)
target_link_libraries(dispatch_proof PRIVATE aten_core)
