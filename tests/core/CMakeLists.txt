# M2 Layer 2 boundary and published-contract tests (manifest M2 §tests). They
# include only <sonicboom/...> headers and link libsonicboom.so (the SHARED
# core library). The engine-internal regression suites (kernels, codegen,
# tuning/dispatch, planner internals, speculative decoding) live outside this
# repository; when a developer sets SONICBOOM_PRIVATE_TESTS_DIR they build and
# register into the same run (bottom of this file).
#
# Every suite is both built (add_executable) and registered with CTest so a
# single `ctest` run exercises the whole C++ suite next to the Guile suites.

# SONICBOOM_SANITIZE: every target in this directory (including the optional
# private suites added at the bottom) is SonicBoom-owned test code, so the whole
# directory takes the flags. Each executable must carry the sanitizer runtime:
# it loads the instrumented libsonicboom.so.
if(SONICBOOM_SANITIZE)
  # SONICBOOM_SANITIZE_COMPILE_EXTRA carries -fno-sanitize=vptr for the
  # address;undefined build (no RTTI in MLIR; root CMakeLists.txt explains).
  add_compile_options(${SONICBOOM_SANITIZE_FLAGS}
                      ${SONICBOOM_SANITIZE_COMPILE_EXTRA}
                      -fno-omit-frame-pointer)
  add_link_options(${SONICBOOM_SANITIZE_FLAGS})
endif()

# Test data (fixtures, goldens, bench inputs) lives with the private suites:
# SONICBOOM_PRIVATE_TESTS_DIR/data. The public tree carries none. Tests that
# read it take this definition and skip (exit 77) when it is unset or the file
# is absent, so a public-only checkout still configures and passes.
set(SB_PRIVATE_DATA_DIR "")
if(SONICBOOM_PRIVATE_TESTS_DIR)
  set(SB_PRIVATE_DATA_DIR "${SONICBOOM_PRIVATE_TESTS_DIR}/data")
endif()

# Register a C++ suite with CTest. The suites load libsonicboom.so at runtime,
# so the loader path must include its directory; the executable path is resolved
# through a generator expression so it stays valid across build configurations.
function(sb_add_cpp_test name)
  add_test(NAME ${name}
           COMMAND ${CMAKE_COMMAND} -E env
                   LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                   $<TARGET_FILE:${name}>)
endfunction()

foreach(test_name IN ITEMS test_schema test_value test_operator test_backend test_tensor test_s_expr test_s_expr_exec test_native_ops test_autograd test_gguf test_dequant test_nn test_gemma4 test_qwen2_config test_vocab test_clip_grad_norm test_safetensors test_grammar test_sampling test_png)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  sb_add_cpp_test(${test_name})
endforeach()

# test_clip_grad_norm exercises the fused device kernels directly when the
# library was built with CUDA (the internal kernel header is CUDA-only).
if(SONICBOOM_USE_CUDA)
  target_compile_definitions(test_clip_grad_norm PRIVATE SB_TEST_CUDA=1)
  set_tests_properties(test_clip_grad_norm PROPERTIES RESOURCE_LOCK gpu_vram)
endif()

# The host-only tests that read a real GGUF (dequantization diff, model load,
# tokenizer) are correctness goldens over a multi-GB file, several seconds each
# and skipped only when the GGUF is absent. They go with the rest of the
# model-driven tests into the nightly/weekly suite. (test_qwen2_config stays in
# the per-commit set: it parses the config header only, ~1 s.) Each needs the
# repository root to locate models/.
foreach(test_name IN ITEMS test_dequant test_gemma4 test_qwen2_config test_vocab)
  target_compile_definitions(${test_name} PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
endforeach()
set_tests_properties(test_dequant test_gemma4 test_vocab
  PROPERTIES LABELS long)

# Real GPU closed-loop test of the backward ops (matmul, softmax,
# cross-entropy): device kernels vs an independent fp64 reference, plus
# finite-difference gradient checks. Only built when the CUDA build is enabled;
# at runtime it skips cleanly (exit 0) when no CUDA device is present.
if(SONICBOOM_USE_CUDA)
  add_executable(test_backward_ops test_backward_ops.cpp)
  target_link_libraries(test_backward_ops PRIVATE sonicboom)
  sb_add_cpp_test(test_backward_ops)
  set_tests_properties(test_backward_ops PROPERTIES RESOURCE_LOCK gpu_vram)
endif()

# Static execution planner persistence test. It includes only
# <sonicboom/planner/...> + <sonicboom/sx/...> public headers.
add_executable(test_persistence test_persistence.cpp)
target_link_libraries(test_persistence PRIVATE sonicboom)
sb_add_cpp_test(test_persistence)

# test_e2e_load is the independent-inference half of the Guile → C++ loop. It
# takes a required <artifact-dir> argument and is driven by the `guile-cpp-e2e`
# orchestration (bindings/guile), so it is built but not registered as a
# standalone CTest entry.
add_executable(test_e2e_load test_e2e_load.cpp)
target_link_libraries(test_e2e_load PRIVATE sonicboom)

# Integration tests: parse the frozen ResNet-18 S-Expr v0.1 example document.
# They need the repository root to locate the example file at build/run time.
foreach(test_name IN ITEMS test_s_expr_resnet18 test_s_expr_resnet18_exec test_onnx_import_resnet18_exec)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  target_compile_definitions(${test_name} PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}"
    SONICBOOM_PRIVATE_DATA_DIR="${SB_PRIVATE_DATA_DIR}"
    SONICBOOM_IMPORT_DIR="${CMAKE_BINARY_DIR}/import")
  sb_add_cpp_test(${test_name})
endforeach()

# The two end-to-end ResNet-18 tests run over generated artifacts (the weight /
# input / reference sidecars and the importer's output, all under
# <build>/import/), none of which is committed or written into the source tree.
# A setup fixture generates them from models/resnet18.onnx before either test
# starts, so a fresh checkout passes under a plain `ctest`; the tests keep their
# exit-77 path for the case where the (external) model store is unavailable.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_EXECUTABLE)
  set(Python3_EXECUTABLE python3)
endif()
set(SB_RESNET18_PYTHON "${CMAKE_SOURCE_DIR}/models/.venv/bin/python")
if(NOT EXISTS "${SB_RESNET18_PYTHON}")
  set(SB_RESNET18_PYTHON "${Python3_EXECUTABLE}")
endif()
add_test(NAME resnet18_artifacts
         COMMAND ${CMAKE_COMMAND} -E env
                 "SONICBOOM_PRIVATE_DATA_DIR=${SB_PRIVATE_DATA_DIR}"
                 ${CMAKE_CURRENT_SOURCE_DIR}/gen_resnet18_artifacts.sh
                 "${CMAKE_SOURCE_DIR}" "${SB_RESNET18_PYTHON}"
                 "${CMAKE_BINARY_DIR}/import")
set_tests_properties(resnet18_artifacts PROPERTIES
  SKIP_RETURN_CODE 77
  FIXTURES_SETUP resnet18_artifacts)
set_tests_properties(
  test_s_expr_resnet18_exec test_onnx_import_resnet18_exec
  PROPERTIES
    SKIP_RETURN_CODE 77
    FIXTURES_REQUIRED resnet18_artifacts)
# The parse-only test needs the private fixture but no generated sidecars.
set_tests_properties(test_s_expr_resnet18 PROPERTIES SKIP_RETURN_CODE 77)

# C ABI test, compiled as C to prove <sonicboom/capi.h> is a genuine C header
# and libsonicboom.so exports its entry points with C linkage. Links libm for
# fabsf.
add_executable(test_capi test_capi.c)
target_link_libraries(test_capi PRIVATE sonicboom m)
sb_add_cpp_test(test_capi)

# Multi-input / multi-output C ABI: one graph returning both a+b and a*b,
# driven through sb_run_multi and the per-index introspection.
add_executable(test_capi_multi test_capi_multi.c)
target_link_libraries(test_capi_multi PRIVATE sonicboom m)
sb_add_cpp_test(test_capi_multi)

# Seeded sampling through the C ABI (qwen2.5-0.5b, CUDA): determinism, seed
# sensitivity, prefix-reuse identity, the greedy golden and grammar+sampling.
# Skips cleanly (exit 77) without the model, the golden or a CUDA device.
if(SONICBOOM_USE_CUDA)
  add_executable(test_capi_sampling test_capi_sampling.c)
  target_link_libraries(test_capi_sampling PRIVATE sonicboom m CUDA::cudart)
  target_compile_definitions(test_capi_sampling PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}"
    SONICBOOM_PRIVATE_DATA_DIR="${SB_PRIVATE_DATA_DIR}")
  sb_add_cpp_test(test_capi_sampling)
  set_tests_properties(test_capi_sampling PROPERTIES
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)
endif()

# Image generation through the C ABI (MOSS-Image, CUDA, 512x512, 4 steps): the
# pixels match `sb-moss sample` under the same settings, the same request is
# deterministic, image 0 of a batch equals the single image, and a cancelling
# progress callback returns SB_ERR_CANCELLED. The sb-moss reference is the
# out-of-tree tool built by tools/build_sb_moss.sh; the test skips (77) when it
# or the model is absent.
if(SONICBOOM_USE_CUDA)
  add_executable(test_capi_image test_capi_image.c)
  target_link_libraries(test_capi_image PRIVATE sonicboom m CUDA::cudart)
  target_compile_definitions(test_capi_image PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}"
    SB_MOSS_BIN="${CMAKE_SOURCE_DIR}/tools/sb-moss")
  sb_add_cpp_test(test_capi_image)
  set_tests_properties(test_capi_image PROPERTIES
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)
endif()

# The published inference guarantees (determinism, prefix reuse, KV-capacity
# independence, speculative-decoding identity) on qwen2.5-0.5b and Gemma 4 E4B,
# greedy, with the decode chunk pinned. Skips (77) without a CUDA device or the
# model GGUFs; one process per (model, check) keeps the device pools bounded.
if(SONICBOOM_USE_CUDA)
  add_executable(test_guarantees test_guarantees.cpp)
  target_link_libraries(test_guarantees PRIVATE sonicboom m CUDA::cudart)
  target_compile_definitions(test_guarantees PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
  sb_add_cpp_test(test_guarantees)
  set_tests_properties(test_guarantees PROPERTIES
    SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)
endif()

# R2: SonicBoomParam + fused AdamW through the C ABI, 10 steps against an
# independent double-precision reference (max-abs <= 1e-6).
add_executable(test_adamw test_adamw.cpp)
target_link_libraries(test_adamw PRIVATE sonicboom)
sb_add_cpp_test(test_adamw)

# R7: named parameter checkpoints through the C ABI. save → load is bit-exact
# and load order independent; malformed files and name/shape mismatches are
# rejected without touching any target.
add_executable(test_checkpoint test_checkpoint.cpp)
target_link_libraries(test_checkpoint PRIVATE sonicboom)
sb_add_cpp_test(test_checkpoint)

# Boom IR dialect test: op registration against planner::OpKind, the generated
# assembly formats, the type verifiers. It links the compiler library
# (sonicboom_compiler), which is always built — MLIR is a required dependency
# (root CMakeLists.txt) — and it is a compiler-side test: nothing here runs Boom
# IR through the runtime.
add_executable(test_boom_dialect test_boom_dialect.cpp)
target_link_libraries(test_boom_dialect PRIVATE sonicboom sonicboom_compiler
                                             MLIRAsmParser)
sb_add_cpp_test(test_boom_dialect)

# Blade -> Boom IR lowering test (docs/boom-ir.md): the
# direct document lowering must build the same IR as
# graphToBoom(adapt_graph(doc)) for hand-written v0.1 documents, including
# parameters, external sidecars, every attribute kind and the q8_1 layout.
# Compiler-side and CPU-only: nothing here runs a plan.
add_executable(test_blade_to_boom test_blade_to_boom.cpp)
target_link_libraries(test_blade_to_boom PRIVATE sonicboom sonicboom_compiler
                                                    MLIRAsmParser)
sb_add_cpp_test(test_blade_to_boom)

# The GPU tests run one at a time (RESOURCE_LOCK gpu_vram).
#
# The lock covers every test that allocates device memory. Two model tests at
# once do not fit on this device (8192 MiB): a test's peak allocation is sized
# against the memory that is free when it starts, and the second one's
# allocation then comes out of that, so whichever loses fails its own
# allocation. That is a nondeterministic failure no gate can live with, so the
# device tests are serialized. CPU-only tests stay parallel.
#
# Deliberately not locked: the tests that read a GGUF on the host without
# touching the device (test_gguf, test_dequant, test_gemma4, test_qwen2_config,
# test_vocab), and everything else in the suite.

# Out-of-tree suites (optional): engine-internal regression suites. The public
# build references no path outside this repository; when a developer sets
# SONICBOOM_PRIVATE_TESTS_DIR, that directory's CMakeLists.txt is added here,
# so its suites build and register into the same build and ctest run. It links
# the same targets and reuses sb_add_cpp_test above.
if(SONICBOOM_PRIVATE_TESTS_DIR)
  add_subdirectory("${SONICBOOM_PRIVATE_TESTS_DIR}" private_tests)
endif()
