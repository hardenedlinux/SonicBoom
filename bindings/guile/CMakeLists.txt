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
