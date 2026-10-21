# Writes the build-tag header (SB_BUILD_GIT, SB_BUILD_GIT_TAG).
#
# Runs in script mode (-P) twice: once at configure time to populate the header,
# and once per build through the `sonicboom_build_tag` custom target. Running it
# at build time is what keeps sb_build_info() honest after a commit: the tag is
# resolved from the working tree of the build in progress, not from the last
# configure.
#
# The header is rewritten ONLY when its content changes, so an unchanged
# revision leaves the file's timestamp alone and does not recompile capi.cpp or
# relink libsonicboom.so.
#
# Inputs: GIT_EXECUTABLE (may be unset/not-found), SOURCE_DIR, OUTPUT.

if(NOT DEFINED SOURCE_DIR OR NOT DEFINED OUTPUT)
  message(FATAL_ERROR "WriteBuildTag.cmake: SOURCE_DIR and OUTPUT are required")
endif()

set(_git "unknown")
set(_tag "unknown")

if(GIT_EXECUTABLE AND EXISTS "${GIT_EXECUTABLE}" AND EXISTS "${SOURCE_DIR}")
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse --short=12 HEAD
                  WORKING_DIRECTORY "${SOURCE_DIR}"
                  OUTPUT_VARIABLE _rev OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET RESULT_VARIABLE _rev_rc)
  if(_rev_rc EQUAL 0 AND _rev)
    set(_git "${_rev}")
    execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain
                            --untracked-files=no
                    WORKING_DIRECTORY "${SOURCE_DIR}"
                    OUTPUT_VARIABLE _dirty
                    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    if(_dirty)
      set(_git "${_git}-dirty")
    endif()
    # `git describe` names the release the revision belongs to; a bare commit
    # cannot say which one. --always keeps a tagless tree usable.
    execute_process(COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
                    WORKING_DIRECTORY "${SOURCE_DIR}"
                    OUTPUT_VARIABLE _desc OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE _desc_rc)
    if(_desc_rc EQUAL 0 AND _desc)
      set(_tag "${_desc}")
    endif()
  endif()
endif()

set(_content
"// Generated at build time by cmake/WriteBuildTag.cmake. Do not edit.
// The revision this library was built from; reported by sb_build_info().
#ifndef SONICBOOM_SB_BUILD_TAG_H
#define SONICBOOM_SB_BUILD_TAG_H

#define SB_BUILD_GIT \"${_git}\"
#define SB_BUILD_GIT_TAG \"${_tag}\"

#endif // SONICBOOM_SB_BUILD_TAG_H
")

set(_old "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" _old)
endif()
if(NOT _old STREQUAL _content)
  get_filename_component(_dir "${OUTPUT}" DIRECTORY)
  file(MAKE_DIRECTORY "${_dir}")
  file(WRITE "${OUTPUT}" "${_content}")
endif()
