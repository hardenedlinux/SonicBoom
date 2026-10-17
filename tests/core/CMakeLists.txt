# M2 Layer 2 boundary tests (manifest M2 §tests). They include only
# <sonicboom/...> headers and link libsonicboom.so (the SHARED core library).
# test_mlir additionally proves the statically-linked MLIR infrastructure.
#
# Every suite is both built (add_executable) and registered with CTest so a
# single `ctest` run exercises the whole C++ suite next to the Guile suites.

# Register a C++ suite with CTest. The suites load libsonicboom.so at runtime,
# so the loader path must include its directory; the executable path is resolved
# through a generator expression so it stays valid across build configurations.
function(sb_add_cpp_test name)
  add_test(NAME ${name}
           COMMAND ${CMAKE_COMMAND} -E env
                   LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                   $<TARGET_FILE:${name}>)
endfunction()

foreach(test_name IN ITEMS test_schema test_value test_operator test_backend test_tensor test_mlir test_s_expr test_s_expr_lowering test_s_expr_exec test_native_dispatch test_native_ops test_autograd test_gguf test_dequant test_quantized_matmul test_nn test_gemma4 test_qwen2_config test_qwen2_generate test_generate test_prefix_cache test_vocab test_plan_record test_plan_cache test_runtime_features)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  sb_add_cpp_test(${test_name})
endforeach()

# The runtime-feature disable list is parsed once per process, so the mode that
# needs a different value of it is a separate invocation. (The unknown-name
# error is exercised in-process by the default test, which forks a child.)
add_test(NAME test_runtime_features_disabled
         COMMAND ${CMAKE_COMMAND} -E env
                 LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                 SONICBOOM_FEATURE_DISABLE=split_kv,plan_cache
                 $<TARGET_FILE:test_runtime_features> --disabled)

# test_dequant optionally diffs against the real Gemma 4 GGUF (skips cleanly
# when the file is absent). It needs the repository root to locate models/.
target_compile_definitions(test_dequant PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_gemma4 loads the real Gemma 4 GGUF (skips cleanly when absent); same
# repository-root requirement for locating models/.
target_compile_definitions(test_gemma4 PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_qwen2_config loads the Qwen2.5 GGUF (skips cleanly when absent); same
# repository-root requirement for locating models/.
target_compile_definitions(test_qwen2_config PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_qwen2_generate (Qwen2 prefill + greedy decode vs the llama.cpp golden
# streams) needs the repository root to locate models/ and tests/data/qwen2/.
target_compile_definitions(test_qwen2_generate PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_generate (autoregressive decode + CUDA-vs-oracle agreement + timing) also
# needs the repository root to locate models/.
target_compile_definitions(test_generate PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_prefix_cache (C.4 prefix-cache reuse acceptance) needs the repository
# root to locate models/ and the real_prompt.tokens fixture.
target_compile_definitions(test_prefix_cache PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_vocab loads the real Gemma 4 GGUF tokenizer (skips cleanly when absent);
# same repository-root requirement for locating models/.
target_compile_definitions(test_vocab PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_plan_record (execution-plan record determinism + committed goldens) needs
# the repository root for models/ and tests/data/plans/. It exits 77 (CTest:
# Skipped) when no CUDA device or no model GGUF is available — the record needs
# a device, and the goldens are per model.
target_compile_definitions(test_plan_record PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
set_tests_properties(test_plan_record
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram)

# test_plan_cache (prefill segment cache: a hit must be indistinguishable from a
# fresh build) needs the repository root for models/, and the same CUDA + GGUF
# skip conditions as test_plan_record.
target_compile_definitions(test_plan_cache PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
set_tests_properties(test_plan_cache
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram)

# The GPU tests run one at a time (RESOURCE_LOCK gpu_vram).
#
# The lock covers every test that allocates device memory. Two of them at once
# do not fit on this device (8192 MiB): the measured peak of a single run is
# 7726 MiB for test_prefix_cache, 4780 for test_plan_cache, 3784 for
# test_plan_record and test_boom_conversion, 3370 for test_generate and 924 for
# test_qwen2_generate (idle 242). `ctest -jN` runs tests in parallel by default
# (the default is -j1 for `ctest` but -jN for a CI driver), and the two peaks are
# not independent budgets: the run that gets there first sizes its device pools
# against the memory that is free, so the second one's allocation comes out of
# that, and whichever loses fails its own allocation — measured, `ctest -j4`
# fails with "device buffer allocation failed" (test_prefix_cache,
# guile-inference) and with kernel entry points returning failure on their
# internal allocation (test_cuda_quant). It is not a wrong-answer failure, but a
# nondeterministic one, so it is not something a gate can live with.
#
# That is also why the CUDA unit tests are in: they peak at only 334-380 MiB
# (140 above idle) and pass on their own, but they fail beside a model test.
# Being small does not make a test safe to co-schedule — only serializing the
# GPU tests does.
#
# Deliberately not locked: the tests that read a GGUF on the host without
# touching the device (test_gguf, test_dequant, test_gemma4, test_qwen2_config,
# test_vocab), and everything else in the suite — CPU-only tests stay parallel.
set_tests_properties(test_generate test_qwen2_generate test_prefix_cache
  PROPERTIES RESOURCE_LOCK gpu_vram)

# Real GPU closed-loop test. Only built when the CUDA build is enabled; at
# runtime it skips cleanly (exit 0) when no CUDA device is present.
if(SONICBOOM_USE_CUDA)
  add_executable(test_cuda test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE sonicboom)
  sb_add_cpp_test(test_cuda)

  # CUDA K-quant matvec differential test (CUDA gemv vs scalar CPU reference).
  add_executable(test_cuda_quant test_cuda_quant.cpp)
  target_link_libraries(test_cuda_quant PRIVATE sonicboom)
  sb_add_cpp_test(test_cuda_quant)

  # CUDA single-row gather differential test (device gather vs dequantize_row_f32).
  add_executable(test_cuda_gather test_cuda_gather.cpp)
  target_link_libraries(test_cuda_gather PRIVATE sonicboom)
  sb_add_cpp_test(test_cuda_gather)

  # CUDA elementwise/norm/RoPE/dense-matvec differential test (each nn::cuda::*
  # kernel vs its CPU reference).
  add_executable(test_cuda_elementwise test_cuda_elementwise.cpp)
  target_link_libraries(test_cuda_elementwise PRIVATE sonicboom)
  sb_add_cpp_test(test_cuda_elementwise)

  # Compiler-generated glue kernel (Stage 5): the MLIR -> NVVM -> PTX pipeline
  # and the driver JIT, byte-compared with the hand-written kernel the region
  # replaces. Links the compiler library for generate_glue_ptx and the CUDA
  # driver for the module load.
  add_executable(test_glue_codegen test_glue_codegen.cpp)
  target_link_libraries(test_glue_codegen PRIVATE sonicboom sonicboom_compiler
                                                CUDA::cudart CUDA::cuda_driver)
  sb_add_cpp_test(test_glue_codegen)

  # Small, but still GPU tests: they fail beside a model test (see "The GPU
  # tests run one at a time" above).
  set_tests_properties(test_cuda test_cuda_quant test_cuda_gather
                       test_cuda_elementwise test_glue_codegen
    PROPERTIES RESOURCE_LOCK gpu_vram)
endif()

# Static execution planner tests (unit + integration). They include only
# <sonicboom/planner/...> + <sonicboom/sx/...> public headers.
foreach(test_name IN ITEMS test_planner_resource test_planner_cost_model test_planner_graph test_execution_plan test_planner test_memory_planner test_plan_validator test_transfer_scheduler test_runtime_executor test_coexecution test_coexecution_merge test_coexecution_region test_coexecution_benchmark test_planner_telemetry test_planner_benchmark test_runtime test_persistence)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  sb_add_cpp_test(${test_name})
endforeach()

# test_e2e_load is the independent-inference half of the Guile → C++ loop. It
# takes a required <artifact-dir> argument and is driven by the `guile-cpp-e2e`
# orchestration (bindings/guile), so it is built but not registered as a
# standalone CTest entry.
add_executable(test_e2e_load test_e2e_load.cpp)
target_link_libraries(test_e2e_load PRIVATE sonicboom)

# Integration tests: parse the frozen ResNet-18 S-Expr v0.1 example document.
# They need the repository root to locate the example file at build/run time.
foreach(test_name IN ITEMS test_s_expr_resnet18 test_s_expr_resnet18_mlir test_s_expr_resnet18_exec test_onnx_import_resnet18_exec test_planner_acceptance)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  target_compile_definitions(${test_name} PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
  sb_add_cpp_test(${test_name})
endforeach()

# The two end-to-end ResNet-18 tests run over generated artifacts (the weight /
# input / reference sidecars under design/, and the importer's output under
# build/import/), none of which is committed — see .gitignore. On a fresh
# checkout they exit 77 with a message naming the file and the generator, so
# CTest reports them as Skipped rather than Failed.
set_tests_properties(
  test_s_expr_resnet18_exec test_onnx_import_resnet18_exec
  PROPERTIES SKIP_RETURN_CODE 77)

# C ABI test, compiled as C to prove <sonicboom/capi.h> is a genuine C header
# and libsonicboom.so exports its entry points with C linkage. Links libm for
# fabsf.
add_executable(test_capi test_capi.c)
target_link_libraries(test_capi PRIVATE sonicboom m)
sb_add_cpp_test(test_capi)

# Boom IR dialect test: op registration against planner::OpKind, the generated
# assembly formats, the type verifiers. It links the compiler library
# (sonicboom_compiler), which is always built — MLIR is a required dependency
# (root CMakeLists.txt) — and it is a compiler-side test: nothing here runs Boom
# IR through the runtime.
add_executable(test_boom_dialect test_boom_dialect.cpp)
target_link_libraries(test_boom_dialect PRIVATE sonicboom sonicboom_compiler
                                             MLIRAsmParser)
sb_add_cpp_test(test_boom_dialect)

# Boom IR conversion test (design/mlir-migration.md §7a): Graph -> Boom IR ->
# Graph over the real models, and the plan gate — the round-tripped graph, fed
# back through record_{decode,prefill}_plan's GraphTransform seam, must produce
# the golden plan (tests/data/plans/) byte for byte. Needs the repository root
# for those goldens and for models/, and exits 77 (Skipped) without a CUDA
# device or the model GGUFs, exactly as test_plan_record does.
add_executable(test_boom_conversion test_boom_conversion.cpp)
target_link_libraries(test_boom_conversion PRIVATE sonicboom sonicboom_compiler
                                                  MLIRAsmParser)
target_compile_definitions(test_boom_conversion PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
sb_add_cpp_test(test_boom_conversion)
# It loads both models onto the GPU for the plan gate, so it takes the same
# lock as the other model tests (see "The model tests run one at a time").
set_tests_properties(test_boom_conversion
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram)
