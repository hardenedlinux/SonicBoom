# SonicCross build-time generation (matches upstream PyTorch's build-time
# torchgen). Defines `sonicboom_generate_native_torch(<ATEN_DEST_DIR>)`, which
# runs `guild torchgen --per-operator-headers` and writes the generated C++
# (per-op split headers + aggregate headers + Register{key}.cpp + Operators*.cpp)
# under <ATEN_DEST_DIR>. The destination is the `ATen/` parent inside the build
# directory; generated sources are gitignored and never committed.
#
# SonicCross is a REQUIRED build-time dependency: `guild` must be on PATH (or
# `-DSONICCROSS_GUILD=/path/to/guild`). If it is missing, configuration fails.

find_program(SONICCROSS_GUILD_EXECUTABLE NAMES guild
  HINTS ${SONICCROSS_GUILD})

function(sonicboom_generate_native_torch ATEN_DEST_DIR)
  if(NOT SONICCROSS_GUILD_EXECUTABLE)
    message(FATAL_ERROR
      "SonicCross (`guild`) is required to generate native-torch sources at "
      "build time (matching upstream PyTorch's build-time torchgen). Install "
      "SonicCross (`./bootstrap && ./configure && make && make install`) or "
      "pass -DSONICCROSS_GUILD=/path/to/guild.")
  endif()

  # Skip regeneration when the generated tree is up-to-date. Re-running guild
  # on every configure rewrites every generated file with a fresh mtime, which
  # invalidates the whole aten_core object tree and forces a full rebuild even
  # when nothing changed. We stamp the output and only regenerate when the
  # generation inputs (the native_functions.yaml snapshot or the guild binary)
  # are newer than the stamp. Delete <ATEN_DEST_DIR>/.soniccross_stamp (or the
  # build dir) to force a regeneration.
  set(_soniccross_stamp "${ATEN_DEST_DIR}/.soniccross_stamp")
  set(_soniccross_yaml
    "${CMAKE_SOURCE_DIR}/tools/generation/native/native_functions.yaml")
  if(EXISTS "${_soniccross_stamp}" AND
     NOT "${_soniccross_yaml}" IS_NEWER_THAN "${_soniccross_stamp}" AND
     NOT "${SONICCROSS_GUILD_EXECUTABLE}" IS_NEWER_THAN "${_soniccross_stamp}")
    message(STATUS
      "SonicCross: native-torch sources up-to-date (${ATEN_DEST_DIR})")
    return()
  endif()

  message(STATUS
    "SonicCross: generating native-torch sources -> ${ATEN_DEST_DIR}")
  execute_process(
    COMMAND ${SONICCROSS_GUILD_EXECUTABLE} torchgen
      -s "${CMAKE_SOURCE_DIR}/tools/generation"
      -d "${ATEN_DEST_DIR}"
      --per-operator-headers
      --generate headers,sources
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    RESULT_VARIABLE _soniccross_result
    OUTPUT_VARIABLE _soniccross_out
    ERROR_VARIABLE _soniccross_err)
  if(NOT _soniccross_result EQUAL 0)
    message(FATAL_ERROR
      "SonicCross generation failed (exit ${_soniccross_result}):\n"
      "${_soniccross_err}\n${_soniccross_out}")
  endif()
  file(TOUCH "${_soniccross_stamp}")
endfunction()
