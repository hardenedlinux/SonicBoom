# SonicCross regeneration hook.
#
# SonicCross (guild torchgen) is an OPTIONAL build-time dependency. It is used
# only when a SonicBoom developer explicitly regenerates the native-torch C++
# sources. Normal builds compile the checked-in generated/ tree and never need
# SonicCross installed.
#
# Cache variables:
#   SONICCROSS_SOURCE_DIR  — path to a SonicCross source/build tree; sets
#                            GUILE_LOAD_PATH so the tree's modules are used
#                            instead of the installed ones.
#   SONICCROSS_GUILD       — explicit path to the `guild` executable
#                            (defaults to `guild` found on PATH).

find_program(GUILD_EXECUTABLE NAMES guild
  HINTS ${SONICCROSS_GUILD})

if(GUILD_EXECUTABLE)
  set(_torchgen_cmd
    ${GUILD_EXECUTABLE} torchgen
      -s "${CMAKE_SOURCE_DIR}/tools/generation"
      -d "${CMAKE_SOURCE_DIR}/third_party/native-torch/generated"
      --generate headers,sources)

  if(SONICCROSS_SOURCE_DIR)
    set(_torchgen_cmd
      ${CMAKE_COMMAND} -E env
        "GUILE_LOAD_PATH=${SONICCROSS_SOURCE_DIR}/modules:${SONICCROSS_SOURCE_DIR}/scripts"
        ${_torchgen_cmd})
  endif()

  # Not part of ALL: runs only when the developer requests regeneration.
  add_custom_target(generate-native-torch
    COMMAND ${_torchgen_cmd}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Regenerating native-torch sources with SonicCross (guild torchgen)"
    VERBATIM)
else()
  message(STATUS
    "SonicCross (guild) not found: 'generate-native-torch' target is "
    "unavailable. Using the checked-in generated native-torch sources.")
endif()
