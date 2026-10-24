# M2 Layer 2 boundary tests (manifest M2 §tests). They include only
# <sonicboom/...> headers and link libsonicboom.so (the SHARED core library).
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

foreach(test_name IN ITEMS test_schema test_value test_operator test_backend test_tensor test_s_expr test_s_expr_exec test_native_dispatch test_native_ops test_autograd test_gguf test_dequant test_quantized_matmul test_nn test_gemma4 test_qwen2_config test_prefix_cache test_vocab test_plan_cache test_spec_invariance test_spec_rollback test_spec_program test_spec_bootstrap test_spec_autoprobe test_ngram_drafter test_clip_grad_norm test_runtime_features test_autotune_cache test_autotune_runner test_decision_registry test_safetensors test_grammar)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  sb_add_cpp_test(${test_name})
endforeach()

# test_generate and test_qwen2_generate are the executables behind the
# variant-registered entries below (decode-chunk, spec, numerics profiles,
# deepseek). They are built but not registered under their bare names: the
# variants already cover the default path.
foreach(test_name IN ITEMS test_generate test_qwen2_generate)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
endforeach()

# test_clip_grad_norm exercises the fused device kernels directly when the
# library was built with CUDA (the internal kernel header is CUDA-only).
if(SONICBOOM_USE_CUDA)
  target_compile_definitions(test_clip_grad_norm PRIVATE SB_TEST_CUDA=1)
  set_tests_properties(test_clip_grad_norm PROPERTIES RESOURCE_LOCK gpu_vram)
endif()

# test_autotune_runner's Q2 section runs the real q8_1 gemv on a synthetic
# weight, so it allocates device memory (small, but the lock is per-test).
set_tests_properties(test_autotune_runner PROPERTIES RESOURCE_LOCK gpu_vram)

# The S1 drafter-coverage gate, registered under the spec-* family name so it
# is selected with the other speculative-decode suites (`ctest -R spec`).
add_test(NAME test_spec_drafter
         COMMAND ${CMAKE_COMMAND} -E env
                 LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                 $<TARGET_FILE:test_ngram_drafter>)

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

# The host-only tests that read a real GGUF (dequantization diff, model load,
# tokenizer) are correctness goldens over a multi-GB file, several seconds each
# and skipped only when the GGUF is absent. They go with the rest of the
# model-driven tests into the nightly/weekly suite. (test_qwen2_config stays in
# the per-commit set: it parses the config header only, ~1 s.)
set_tests_properties(test_dequant test_gemma4 test_vocab
  PROPERTIES LABELS long)

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

# The 8K-context prefix cases (8191 / 8192 / 8193) run in their own process.
# On an 8 GiB device the allocations retained by the preceding cases push the
# 8K peak over the limit (the 8191 CUDA graph fails to instantiate with "out of
# memory" in the combined run) while the same cases fit when the process starts
# fresh. Same executable, two enumerations; the device lock serializes them.
add_test(NAME test_prefix_cache_long
         COMMAND ${CMAKE_COMMAND} -E env
                 LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                 $<TARGET_FILE:test_prefix_cache> --long)
set_tests_properties(test_prefix_cache_long
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

# test_vocab loads the real Gemma 4 GGUF tokenizer (skips cleanly when absent);
# same repository-root requirement for locating models/.
target_compile_definitions(test_vocab PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_plan_cache (prefill segment cache: a hit must be indistinguishable from a
# fresh build) needs the repository root for models/, and the same CUDA + GGUF
# skip conditions as the out-of-tree plan-record suite.
target_compile_definitions(test_plan_cache PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
set_tests_properties(test_plan_cache
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

# test_spec_invariance (B3 verify-step invariance) needs both model GGUFs,
# looked up relative to the repository root. It forks one child per
# (model, depth, k) case, so it is a device-memory test like the others.
target_compile_definitions(test_spec_invariance PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_spec_rollback (B4 KV rollback) needs both model GGUFs, looked up
# relative to the repository root, one forked child per (model, depth, case).
target_compile_definitions(test_spec_rollback PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_spec_program (B6 speculative loop) needs both model GGUFs, looked up
# relative to the repository root; it forks one child per (model, depth) case.
target_compile_definitions(test_spec_program PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_spec_bootstrap (S2/S3 cold-start bootstrap) loads a model GGUF the same
# way, so it needs the repository root too.
target_compile_definitions(test_spec_bootstrap PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_spec_autoprobe (V2 probe decision, cache round-trip, env override) loads
# the Gemma GGUF and runs one forked generation per case.
target_compile_definitions(test_spec_autoprobe PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# test_ngram_drafter's coverage case feeds the code-edit trace's source files
# through the drafter; it needs the repository root to read them.
target_compile_definitions(test_ngram_drafter PRIVATE
  SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")

# The GPU tests run one at a time (RESOURCE_LOCK gpu_vram).
#
# The lock covers every test that allocates device memory. Two of them at once
# do not fit on this device (8192 MiB): the measured peak of a single run is
# 4210 MiB for test_prefix_cache, 4780 for test_plan_cache, 3784 for the
# out-of-tree plan-golden suites, 3370 for test_generate and 924 for
# test_qwen2_generate (idle 242). The prefix-cache figure is measured after the
# test was split into one forked child per case (2026-10-18): nvidia-smi
# memory.used sampled every 40 ms, maximum over both the short and --long
# enumerations, is 4210 MiB (4142 MiB short, 4210 MiB long). The earlier
# 7726/7628/7830 MiB figures were the single-process test, which accumulated
# every case's device pools and captured graphs; the pre-fork source run
# against the same library peaks at 7806 MiB, so that 7628-7830 spread is the
# sampling point and the planner's free-memory-dependent pool sizing, not a
# code change. `ctest -jN` runs tests in parallel by default
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
# These are labelled long: each loads a real GGUF and runs a whole
# prefill/decode or speculative session to compare against token goldens
# (~15 s to ~2 min apiece, serialized by the device lock), so they belong to
# the nightly/weekly suite rather than the per-commit one.
set_tests_properties(test_prefix_cache
                              test_spec_invariance test_spec_rollback
                              test_spec_program test_spec_bootstrap
                              test_spec_autoprobe
  PROPERTIES RESOURCE_LOCK gpu_vram LABELS long)

# The decode-chunk path (k greedy steps captured per graph, §10) is off by
# default — autotuned to k=1, since the decode step is GPU-bound and chunking
# measured 0.2-0.5% slower. Run the same greedy-stream checks with it on so the
# path stays covered; k=4 is the ablation value the internal harness uses.
add_test(NAME test_generate_decode_chunk
         COMMAND ${CMAKE_COMMAND} -E env
                 LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                 SONICBOOM_DECODE_CHUNK=4
                 $<TARGET_FILE:test_generate>)
set_tests_properties(test_generate_decode_chunk
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

# The speculative loop (SONICBOOM_SPEC, B6) is also parsed once per process.
# Run the golden-stream checks with it on: the drafter's proposals must not
# change a single token, and the prefill path's speculative iterations must
# still match the per-token prompt's stream.
add_test(NAME test_generate_spec
         COMMAND ${CMAKE_COMMAND} -E env
                 LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                 SONICBOOM_SPEC=on
                 $<TARGET_FILE:test_generate>)
set_tests_properties(test_generate_spec
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

# The same coverage for the other consumer of the chunked loop: the
# prefill-then-decode path (prefill_decode_cuda_impl) must produce the same
# golden streams with the chunked replay.
add_test(NAME test_qwen2_generate_decode_chunk
         COMMAND ${CMAKE_COMMAND} -E env
                 LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                 SONICBOOM_DECODE_CHUNK=4
                 $<TARGET_FILE:test_qwen2_generate>)
set_tests_properties(test_qwen2_generate_decode_chunk
  PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

# The third model (T8): DeepSeek-R1-Distill-Qwen-1.5B is Qwen2 architecture, so
# it runs the same executable and the same checks against its own fixture
# directory — its prompts, its llama.cpp references and its token goldens. The
# directory's `model` file names the GGUF, so nothing here is model-specific
# beyond the data path. It is registered through the numerics-profile entries
# below (strict and standard); skips cleanly (exit 0) when the GGUF is absent.

# Both numerics profiles are tested contracts (core/include/sonicboom/numerics.h).
# Strict is the historical semantics and compares the historical goldens
# (pN.golden, kGolden); standard is the default and has its own goldens
# (pN.standard.golden). Each binary also checks determinism, the goldens and
# prefix reuse internally, so those hold under both profiles. The base entries above
# run under the default profile; these name it explicitly so an environment
# change cannot silently move what is checked.
foreach(profile IN ITEMS strict standard)
  add_test(NAME test_generate_numerics_${profile}
           COMMAND ${CMAKE_COMMAND} -E env
                   LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                   SONICBOOM_NUMERICS=${profile}
                   $<TARGET_FILE:test_generate>)
  set_tests_properties(test_generate_numerics_${profile}
    PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

  add_test(NAME test_qwen2_generate_numerics_${profile}
           COMMAND ${CMAKE_COMMAND} -E env
                   LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                   SONICBOOM_NUMERICS=${profile}
                   $<TARGET_FILE:test_qwen2_generate>)
  set_tests_properties(test_qwen2_generate_numerics_${profile}
    PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)

  add_test(NAME test_deepseek_generate_numerics_${profile}
           COMMAND ${CMAKE_COMMAND} -E env
                   LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
                   SONICBOOM_QWEN_DATA=${CMAKE_SOURCE_DIR}/tests/data/deepseek-r1-qwen15b
                   SONICBOOM_NUMERICS=${profile}
                   $<TARGET_FILE:test_qwen2_generate>)
  set_tests_properties(test_deepseek_generate_numerics_${profile}
    PROPERTIES SKIP_RETURN_CODE 77 RESOURCE_LOCK gpu_vram LABELS long)
endforeach()

# Real GPU closed-loop test. Only built when the CUDA build is enabled; at
# runtime it skips cleanly (exit 0) when no CUDA device is present.
if(SONICBOOM_USE_CUDA)
  add_executable(test_cuda test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE sonicboom)
  sb_add_cpp_test(test_cuda)

  # CUDA K-quant matvec differential test (CUDA gemv vs scalar CPU reference).
  # It also drives the internal q8_1 entry points directly (the G1 epilogue and
  # G2 mixed-quant sibling launch), which needs device allocation of its own.
  add_executable(test_cuda_quant test_cuda_quant.cpp)
  target_link_libraries(test_cuda_quant PRIVATE sonicboom CUDA::cudart)
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

  # R3: backward ops (matmul, softmax, cross-entropy) — device kernels vs an
  # independent fp64 reference, plus finite-difference gradient checks.
  add_executable(test_backward_ops test_backward_ops.cpp)
  target_link_libraries(test_backward_ops PRIVATE sonicboom)
  sb_add_cpp_test(test_backward_ops)

  # Compiler-generated glue kernel (Stage 5): the MLIR -> NVVM -> PTX pipeline
  # and the driver JIT, byte-compared with the hand-written kernel the region
  # replaces. Links the compiler library for generate_glue_ptx and the CUDA
  # driver for the module load.
  add_executable(test_glue_codegen test_glue_codegen.cpp)
  target_link_libraries(test_glue_codegen PRIVATE sonicboom sonicboom_compiler
                                                CUDA::cudart CUDA::cuda_driver)
  sb_add_cpp_test(test_glue_codegen)

  # Compiler-generated FlashAttention-2 kernel (Stage 3a): the codegen path of
  # flash_attention_chunk_dev_pos at head_dim 64, differential against an fp64
  # reference with the kernel's f16 staging simulated, plus bitwise batch
  # invariance. SONICBOOM_ATTN_IMPL=codegen is set by the test itself.
  add_executable(test_cuda_attention_codegen test_cuda_attention_codegen.cpp)
  target_link_libraries(test_cuda_attention_codegen PRIVATE sonicboom
                                                        sonicboom_compiler
                                                        CUDA::cudart
                                                        CUDA::cuda_driver)
  sb_add_cpp_test(test_cuda_attention_codegen)

  # Compiler-generated FlashAttention-2 kernel for wide heads (Stage 3b):
  # head_dim 256 and 512 (Gemma sliding-window / global heads), where the
  # generator splits head_dim across 2 and 4 warps and exchanges the partial S
  # through shared memory. Same differential + batch-invariance contract as
  # the 3a test, with the window length not a multiple of Bc.
  add_executable(test_cuda_attention_codegen_wide
                 test_cuda_attention_codegen_wide.cpp)
  target_link_libraries(test_cuda_attention_codegen_wide PRIVATE sonicboom
                                                             sonicboom_compiler
                                                             CUDA::cudart
                                                             CUDA::cuda_driver)
  sb_add_cpp_test(test_cuda_attention_codegen_wide)

  # Q3 prefill attention autotune: shape classes
  # (head_dim, seq_len bucket, query-head bucket), the six
  # tile_seq x tile_head candidates, the gate/cache, and (on a device) a tuned
  # d64/gqa7 prefill that must be bit-identical to the (16,16) reference and
  # stable across three cold runs.
  add_executable(test_attn_prefill_autotune test_attn_prefill_autotune.cpp)
  target_link_libraries(test_attn_prefill_autotune PRIVATE sonicboom
                                                          CUDA::cudart)
  sb_add_cpp_test(test_attn_prefill_autotune)

  # Compiler-generated flash-decoding (Stage 5): the codegen path of
  # decode_attention_dev — `fa_decode` partials + `fa_combine`, at head_dim
  # 256/512 and the forced d64 GQA shape. Differential against an fp64
  # softmax over the KV cache, determinism, "not worse than the hand-written
  # split-KV kernel it replaces", and bit-identity with that kernel at the
  # default chunk. SONICBOOM_ATTN_DECODE is set by the test itself.
  add_executable(test_cuda_attention_decode_codegen
                 test_cuda_attention_decode_codegen.cpp)
  target_link_libraries(test_cuda_attention_decode_codegen PRIVATE sonicboom
                                                              sonicboom_compiler
                                                              CUDA::cudart
                                                              CUDA::cuda_driver)
  sb_add_cpp_test(test_cuda_attention_decode_codegen)

  # Compiler-generated dense bf16 GEMM (step 10a): the codegen path of the
  # strided dense projection (Y[N][M] = X[N][K] W[M][K]^T). Differential
  # against an fp64 reference over bf16-rounded operands, a tail shape where
  # every dimension misses the tile multiple, and the bitwise batch-invariance
  # contract (N = 1 / 7 / 64 / 512, and a shifted-base re-run that moves the
  # same token to a different block coordinate).
  add_executable(test_cuda_dense_gemm_codegen
                 test_cuda_dense_gemm_codegen.cpp)
  target_link_libraries(test_cuda_dense_gemm_codegen PRIVATE sonicboom
                                                            sonicboom_compiler
                                                            CUDA::cudart
                                                            CUDA::cuda_driver
                                                            CUDA::cublas)
  sb_add_cpp_test(test_cuda_dense_gemm_codegen)

  # Compiler-generated q8_1 tensor-core matmul (MMQ1): the int8 m16n8k32 path
  # of the prefill projection. Differential against the kernel's arithmetic
  # (exact per-32 integer dot, scaled by both operands' scales) in fp64, both
  # activation scale layouts, a tail shape, and bitwise batch invariance.
  add_executable(test_cuda_mmq_tc_codegen test_cuda_mmq_tc_codegen.cpp)
  target_link_libraries(test_cuda_mmq_tc_codegen PRIVATE sonicboom
                                                         sonicboom_compiler
                                                         CUDA::cudart
                                                         CUDA::cuda_driver)
  sb_add_cpp_test(test_cuda_mmq_tc_codegen)

  # Small, but still GPU tests: they fail beside a model test (see "The GPU
  # tests run one at a time" above).
  set_tests_properties(test_cuda test_cuda_quant test_cuda_gather
                       test_cuda_elementwise test_glue_codegen
                       test_cuda_attention_codegen
                       test_cuda_attention_codegen_wide
                       test_attn_prefill_autotune
                       test_cuda_attention_decode_codegen
                       test_cuda_dense_gemm_codegen
                       test_cuda_mmq_tc_codegen
    PROPERTIES RESOURCE_LOCK gpu_vram)
endif()

# Static execution planner tests (unit + integration). They include only
# <sonicboom/planner/...> + <sonicboom/sx/...> public headers.
foreach(test_name IN ITEMS test_planner_resource test_planner_cost_model test_planner_graph test_execution_plan test_planner test_memory_planner test_plan_validator test_transfer_scheduler test_runtime_executor test_planner_telemetry test_persistence)
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
foreach(test_name IN ITEMS test_s_expr_resnet18 test_s_expr_resnet18_exec test_onnx_import_resnet18_exec test_planner_acceptance)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  target_compile_definitions(${test_name} PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}"
    SONICBOOM_IMPORT_DIR="${CMAKE_BINARY_DIR}/import")
  sb_add_cpp_test(${test_name})
endforeach()

# The two end-to-end ResNet-18 tests run over generated artifacts (the weight /
# input / reference sidecars under tests/data/, and the importer's output under
# <build>/import/), none of which is committed — see .gitignore. A setup fixture
# generates them from models/resnet18.onnx before either test starts, so a fresh
# checkout passes under a plain `ctest`; the tests keep their exit-77 path for
# the case where the (external) model store is unavailable.
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_EXECUTABLE)
  set(Python3_EXECUTABLE python3)
endif()
set(SB_RESNET18_PYTHON "${CMAKE_SOURCE_DIR}/models/.venv/bin/python")
if(NOT EXISTS "${SB_RESNET18_PYTHON}")
  set(SB_RESNET18_PYTHON "${Python3_EXECUTABLE}")
endif()
add_test(NAME resnet18_artifacts
         COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/gen_resnet18_artifacts.sh
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

# PrefillDispatchPass (MMQ1): a model-free graph pins the dispatch decision under
# the two switches (SONICBOOM_MMQ_TC auto/on/off, SONICBOOM_MMQ_TC_MAX_MB), the
# pass-disable ablation, the shape eligibility, and the attribute carry-over.
add_executable(test_prefill_dispatch test_prefill_dispatch.cpp)
target_link_libraries(test_prefill_dispatch PRIVATE sonicboom sonicboom_compiler
                                                   MLIRAsmParser)
sb_add_cpp_test(test_prefill_dispatch)

# VS1 async-prefetch skeleton: AsyncPrefetchPass under SONICBOOM_ASYNC_PREFETCH
# and SONICBOOM_VRAM_THRESHOLD_MB (marker placement, operands and bytes, the
# pass-disable ablation, the boomToGraph round trip), the runtime's no-op
# default and prefetch stream, and the LRU eviction stub.
add_executable(test_async_prefetch_noop test_async_prefetch_noop.cpp)
target_link_libraries(test_async_prefetch_noop PRIVATE sonicboom
                                                        sonicboom_compiler
                                                        MLIRAsmParser)
if(SONICBOOM_USE_CUDA)
  target_compile_definitions(test_async_prefetch_noop PRIVATE SB_TEST_CUDA=1)
endif()
sb_add_cpp_test(test_async_prefetch_noop)
if(SONICBOOM_USE_CUDA)
  # begin() creates the prefetch stream, so the suite is a GPU consumer.
  set_tests_properties(test_async_prefetch_noop PROPERTIES RESOURCE_LOCK gpu_vram)
endif()

# SD1 DraftModel interface: NgramDraftModel (direct and through DraftModel*)
# must propose token-identically to NGramDrafter, max_drafts must cross the
# interface, the factory must select the n-gram drafter unless
# SONICBOOM_DRAFT_MODEL is set, and LlmDraftModel must throw the
# not-implemented message.
add_executable(test_draft_model_interface test_draft_model_interface.cpp)
target_link_libraries(test_draft_model_interface PRIVATE sonicboom)
sb_add_cpp_test(test_draft_model_interface)

# Blade -> Boom IR lowering test (docs/boom-ir.md): the
# direct document lowering must build the same IR as
# graphToBoom(adapt_graph(doc)) for hand-written v0.1 documents, including
# parameters, external sidecars, every attribute kind and the q8_1 layout.
# Compiler-side and CPU-only: nothing here runs a plan.
add_executable(test_blade_to_boom test_blade_to_boom.cpp)
target_link_libraries(test_blade_to_boom PRIVATE sonicboom sonicboom_compiler
                                                    MLIRAsmParser)
sb_add_cpp_test(test_blade_to_boom)

# Out-of-tree suites (optional): plan goldens and the internal measurement
# harness. The public build references no path outside this repository; when a
# developer sets SONICBOOM_PRIVATE_TESTS_DIR, that directory's CMakeLists.txt is
# added here, so its suites build and register into the same build and ctest
# run. It links the same targets and reuses sb_add_cpp_test above.
if(SONICBOOM_PRIVATE_TESTS_DIR)
  add_subdirectory("${SONICBOOM_PRIVATE_TESTS_DIR}" private_tests)
endif()
