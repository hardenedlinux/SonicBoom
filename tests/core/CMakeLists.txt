# M2 Layer 2 boundary tests (manifest M2 §tests). Five focused executables,
# one per architectural proof (A–E). They include only <sonicboom/...> headers.

foreach(test_name IN ITEMS test_schema test_value test_operator test_backend test_tensor)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
endforeach()
