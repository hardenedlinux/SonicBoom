# SonicBoom Guile binding — Scheme sources + tests.
#
# Two suites:
#   guile-tensor  — pure Scheme (tensor/parameter records); no native library.
#   guile-nn      — layer/model composition; runs forward passes through
#                   libsonicboom.so via the C ABI, so it needs the shared
#                   library on the dynamic-link search path.
# Run via CTest:
#
#   ctest -R guile
#
# or directly:
#
#   guile -L bindings/guile/scheme bindings/guile/tests/test-tensor.scm
#   LD_LIBRARY_PATH=build/core guile -L bindings/guile/scheme \
#       bindings/guile/tests/test-nn.scm

find_program(GUILE_EXECUTABLE guile)

if(GUILE_EXECUTABLE)
  # GUILE_AUTO_COMPILE=0 keeps test output clean (no .go cache noise).
  add_test(
    NAME guile-tensor
    COMMAND ${CMAKE_COMMAND} -E env GUILE_AUTO_COMPILE=0
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-tensor.scm)

  # guile-bpe trains and round-trips a byte-level BPE tokenizer; pure Scheme,
  # no native library dependency.
  add_test(
    NAME guile-bpe
    COMMAND ${CMAKE_COMMAND} -E env GUILE_AUTO_COMPILE=0
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/tokenizer/test-bpe.scm)

  # guile-loader streams text/JSONL files into packed int32 batches; pure
  # Scheme, no native library dependency.
  add_test(
    NAME guile-loader
    COMMAND ${CMAKE_COMMAND} -E env GUILE_AUTO_COMPILE=0
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/test-loader.scm)

  # guile-lr-schedule checks the warmup / cosine / constant learning-rate
  # schedules; pure Scheme, no native library dependency.
  add_test(
    NAME guile-lr-schedule
    COMMAND ${CMAKE_COMMAND} -E env GUILE_AUTO_COMPILE=0
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/train/test-lr-schedule.scm)

  # The numerics profile binding, once per profile: the environment names the
  # profile the process must run under, and the test checks the latched value
  # rather than assuming a default.
  foreach(profile IN ITEMS strict standard)
    add_test(
      NAME guile-numerics-${profile}
      COMMAND ${CMAKE_COMMAND} -E env
              GUILE_AUTO_COMPILE=0
              LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
              SONICBOOM_NUMERICS=${profile}
              ${GUILE_EXECUTABLE} --no-auto-compile
              -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-numerics.scm)
  endforeach()

  # guile-nn loads libsonicboom.so (built by the `sonicboom` target), so put its
  # directory on LD_LIBRARY_PATH for the dynamic-link call.
  add_test(
    NAME guile-nn
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-nn.scm)

  # guile-train drives the native autograd tape; same shared-library dependency
  # as guile-nn.
  add_test(
    NAME guile-train
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-train.scm)

  # guile-checkpoint round-trips named float32 tensors (parameters and AdamW
  # moments) through (sonicboom checkpoint) → sb_save_params/sb_load_params,
  # including the error paths; same shared-library dependency as guile-nn.
  add_test(
    NAME guile-checkpoint
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-checkpoint.scm)

  # guile-explain reads the planner's decision registry through
  # sb_explain_table/sb_explain_json and checks both views; same shared-library
  # dependency as guile-nn. It tunes nothing, so it needs no device.
  add_test(
    NAME guile-explain
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-explain.scm)

  # guile-param replays tests/core/test_adamw.cpp's 10-step AdamW sequence
  # through the sb_param_* surface. When a CUDA device is present the parameter
  # buffers are device-resident, so it holds the device lock.
  add_test(
    NAME guile-param
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-param.scm)
  set_tests_properties(guile-param PROPERTIES RESOURCE_LOCK gpu_vram)

  # guile-executable compiles a graph once and runs it repeatedly through
  # libsonicboom.so, checking numerics, reuse and handle lifetime; same
  # shared-library dependency as guile-nn.
  add_test(
    NAME guile-executable
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-executable.scm)

  # guile-trace-lower lowers a traced forward graph onto the S-Expr v0.1
  # vocabulary and executes it through libsonicboom.so (`run-s-expr`); same
  # shared-library dependency as guile-nn.
  add_test(
    NAME guile-trace-lower
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-trace-lower.scm)

  # guile-train-tiny traces a whole tiny Transformer training step (forward,
  # `backward!`, fused AdamW), compiles it with sb_compile_device and runs it
  # on the GPU, checking every parameter gradient against the pure-Scheme
  # reference (1e-5) and that the loss falls. Same shared-library dependency
  # as guile-nn, plus the GPU.
  add_test(
    NAME guile-train-tiny
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/trace/test-train-tiny.scm)
  # It is a correctness test whose time is dominated by the pure-Scheme
  # reference interpreter (~50 s of ~57 s), so it is labelled "long": the
  # default suite runs `ctest -LE long`, and the nightly/weekly CI job runs
  # `ctest -L long`. It still takes the GPU lock when it does run.
  set_tests_properties(guile-train-tiny PROPERTIES
    LABELS long
    RESOURCE_LOCK gpu_vram)

  # test_adamw_lr_input (D2) runs ten AdamW steps with a schedule delivered as
  # a learning-rate tensor, checks that all ten share one compiled segment (the
  # attribute form needs one per rate), compares the loss curve against the
  # attribute form and the interpreter, and exercises the
  # SONICBOOM_ADAMW_LR_ATTR=1 ablation switch. GPU-bound like guile-train-tiny.
  add_test(
    NAME test_adamw_lr_input
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/trace/test-adamw-lr-input.scm)
  set_tests_properties(test_adamw_lr_input PROPERTIES RESOURCE_LOCK gpu_vram)

  # guile-nn-trace runs a `sequential` model through both
  # routes — the emitted-text forward pass and the traced/lowered one — and
  # checks that they agree; same shared-library dependency as guile-nn.
  add_test(
    NAME guile-nn-trace
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-nn-trace.scm)

  # guile-inference drives the minimal LLM C ABI (sb_llm_*) through
  # (sonicboom inference): load-model / make-session / generate. It loads the
  # real Gemma 4 GGUF (skipping cleanly when absent) and checks the generated
  # token stream against the golden decode tokens.
  add_test(
    NAME guile-inference
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            SONICBOOM_SRC_DIR=${CMAKE_SOURCE_DIR}
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-inference.scm)
    # It runs the model on the GPU, so it takes the same device lock as the C++
    # model tests (tests/core/CMakeLists.txt): two of those at once do not fit
    # on this device, and this one fails beside one of them with "device buffer
    # allocation failed".
    # It loads the real GGUF and generates tokens, so it is a model-driven
    # golden like the C++ end-to-end suites: nightly/weekly, not per commit.
    set_tests_properties(guile-inference PROPERTIES
      RESOURCE_LOCK gpu_vram LABELS long)

  # guile-image drives the MOSS-Image C ABI (sb_image_*) through
  # (sonicboom image): load-image-model / generate-images / write-png. It
  # loads the real MOSS-Image model (skipping cleanly when absent or without a
  # CUDA device), so it is a model-driven golden like guile-inference and takes
  # the same device lock.
  add_test(
    NAME guile-image
    COMMAND ${CMAKE_COMMAND} -E env
            GUILE_AUTO_COMPILE=0
            LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
            SONICBOOM_SRC_DIR=${CMAKE_SOURCE_DIR}
            ${GUILE_EXECUTABLE} --no-auto-compile
            -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-image.scm)
  set_tests_properties(guile-image PROPERTIES
    RESOURCE_LOCK gpu_vram LABELS long)

  # guile-serve-http drives the OpenAI-compatible server (serve/app.scm) end to
  # end — plain JSON, SSE over a raw socket, a WebSocket — against the mock
  # backend (--backend mock), which needs no GGUF. This is the protocol-level
  # suite; tools/test-api-server.sh is the real-backend smoke test. It needs
  # GNU Artanis, so register it only when Artanis is importable.
  execute_process(COMMAND ${GUILE_EXECUTABLE} --no-auto-compile -c
                          "(use-modules (artanis env))"
                  RESULT_VARIABLE GUILE_ARTANIS_RESULT
                  OUTPUT_QUIET ERROR_QUIET)
  if(GUILE_ARTANIS_RESULT EQUAL 0)
    # The test spawns the server with a repository-relative path, so it must run
    # from the repository root.
    add_test(
      NAME guile-serve-http
      COMMAND ${CMAKE_COMMAND} -E env
              GUILE_AUTO_COMPILE=0
              LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
              ${GUILE_EXECUTABLE} --no-auto-compile
              -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/serve/test-serve-http.scm)
    set_tests_properties(guile-serve-http PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # guile-serve-images drives POST /v1/images/generations against the real
    # MOSS-Image model (the mock chat backend, so no GGUF is loaded beside it).
    # The script skips when the model or a CUDA device is absent; it generates
    # images, so it holds the device lock.
    add_test(
      NAME guile-serve-images
      COMMAND ${CMAKE_COMMAND} -E env
              GUILE_AUTO_COMPILE=0
              LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
              SONICBOOM_SRC_DIR=${CMAKE_SOURCE_DIR}
              ${GUILE_EXECUTABLE} --no-auto-compile
              -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/serve/test-serve-images.scm)
    set_tests_properties(guile-serve-images PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      RESOURCE_LOCK gpu_vram LABELS long)
  else()
    message(STATUS "GNU Artanis not found; skipping the guile-serve-http test")
  endif()

  # guile-serve-units drives the serving layer below HTTP — chat templates, the
  # mock backend and its sampler, the session pool, the engine — with no GGUF
  # and no HTTP server. It needs guile-json but not Artanis, so it is registered
  # on its own (before this block it was only runnable by hand).
  execute_process(COMMAND ${GUILE_EXECUTABLE} --no-auto-compile -c
                          "(use-modules (json))"
                  RESULT_VARIABLE GUILE_JSON_RESULT
                  OUTPUT_QUIET ERROR_QUIET)
  if(GUILE_JSON_RESULT EQUAL 0)
    add_test(
      NAME guile-serve-units
      COMMAND ${CMAKE_COMMAND} -E env
              GUILE_AUTO_COMPILE=0
              LD_LIBRARY_PATH=$<TARGET_FILE_DIR:sonicboom>
              ${GUILE_EXECUTABLE} --no-auto-compile
              -L ${CMAKE_CURRENT_SOURCE_DIR}/scheme
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/serve/test-serve-units.scm)
    set_tests_properties(guile-serve-units PROPERTIES
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  else()
    message(STATUS "guile-json not found; skipping the guile-serve-units test")
  endif()

  # guile-cpp-e2e: the full training → export → independent C++ load loop. The
  # Guile exporter (train + sb_export_model) writes an artifact, then the C++
  # loader (test_e2e_load, built in tests/core) reloads it and checks its
  # forward outputs match the trained model's predictions.
  add_test(
    NAME guile-cpp-e2e
    COMMAND sh ${CMAKE_CURRENT_SOURCE_DIR}/tests/run-e2e.sh
            ${GUILE_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/scheme
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test-e2e-export.scm
            $<TARGET_FILE:test_e2e_load>
            $<TARGET_FILE_DIR:sonicboom>)
else()
  message(STATUS "Guile interpreter not found; skipping Guile binding tests")
endif()
