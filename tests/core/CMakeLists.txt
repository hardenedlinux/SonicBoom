# M2 Layer 2 boundary tests (manifest M2 §tests). They include only
# <sonicboom/...> headers and link libsonicboom.so (the SHARED core library).
# test_mlir additionally proves the statically-linked MLIR infrastructure.
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

foreach(test_name IN ITEMS test_schema test_value test_operator test_backend test_tensor test_mlir test_s_expr test_s_expr_lowering test_s_expr_exec test_native_dispatch test_native_ops test_autograd)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  sb_add_cpp_test(${test_name})
endforeach()

# Real GPU closed-loop test. Only built when the CUDA build is enabled; at
# runtime it skips cleanly (exit 0) when no CUDA device is present.
if(SONICBOOM_USE_CUDA)
  add_executable(test_cuda test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE sonicboom)
  sb_add_cpp_test(test_cuda)
endif()

# Static execution planner tests (unit + integration). They include only
# <sonicboom/planner/...> + <sonicboom/sx/...> public headers.
foreach(test_name IN ITEMS test_planner_resource test_planner_cost_model test_planner_graph test_execution_plan test_planner test_memory_planner test_plan_validator test_transfer_scheduler test_runtime_executor test_coexecution test_coexecution_merge test_coexecution_region test_coexecution_benchmark test_planner_telemetry test_planner_benchmark test_runtime test_persistence)
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
foreach(test_name IN ITEMS test_s_expr_resnet18 test_s_expr_resnet18_mlir test_s_expr_resnet18_exec test_onnx_import_resnet18_exec test_planner_acceptance)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  target_compile_definitions(${test_name} PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
  sb_add_cpp_test(${test_name})
endforeach()

# C ABI test, compiled as C to prove <sonicboom/capi.h> is a genuine C header
# and libsonicboom.so exports its entry points with C linkage. Links libm for
# fabsf.
add_executable(test_capi test_capi.c)
target_link_libraries(test_capi PRIVATE sonicboom m)
sb_add_cpp_test(test_capi)
