# Public benchmark tools. They use only the stable C ABI and link the published
# shared library, so they exercise the same boundary a language binding does.

add_executable(codeedit_bench codeedit_bench.c)
set_target_properties(codeedit_bench PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF)
target_include_directories(codeedit_bench PRIVATE
  ${CMAKE_SOURCE_DIR}/capi/include)
target_link_libraries(codeedit_bench PRIVATE sonicboom)
