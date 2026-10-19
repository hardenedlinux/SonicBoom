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
  else()
    message(STATUS "GNU Artanis not found; skipping the guile-serve-http test")
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
