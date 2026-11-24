# output-decay: Output Decay between two directories of per-prompt outputs.
# The tool depends on nothing in SonicBoom (no library link), so it can read
# any engine's outputs; the unit test drives the same core over hand-made
# fixtures.

add_executable(output-decay output-decay.cpp output_decay.cpp)
target_include_directories(output-decay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(output-decay PRIVATE cxx_std_20)

add_executable(test_output_decay test_output_decay.cpp output_decay.cpp)
target_include_directories(test_output_decay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(test_output_decay PRIVATE cxx_std_20)
target_compile_definitions(test_output_decay PRIVATE
  OD_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_test(NAME test_output_decay COMMAND $<TARGET_FILE:test_output_decay>)
