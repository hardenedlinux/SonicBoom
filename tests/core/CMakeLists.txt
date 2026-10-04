# M2 Layer 2 boundary tests (manifest M2 §tests). They include only
# <sonicboom/...> headers and link libsonicboom.so (the SHARED core library).
# test_mlir additionally proves the statically-linked MLIR infrastructure.

foreach(test_name IN ITEMS test_schema test_value test_operator test_backend test_tensor test_mlir test_s_expr test_s_expr_lowering test_s_expr_exec)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
endforeach()

# Static execution planner tests (unit + integration). They include only
# <sonicboom/planner/...> + <sonicboom/sx/...> public headers.
foreach(test_name IN ITEMS test_planner_resource test_planner_cost_model test_planner_graph test_execution_plan test_planner test_memory_planner test_plan_validator test_transfer_scheduler test_runtime_executor test_planner_telemetry test_planner_benchmark)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
endforeach()

# Integration tests: parse the frozen ResNet-18 S-Expr v0.1 example document.
# They need the repository root to locate the example file at build/run time.
foreach(test_name IN ITEMS test_s_expr_resnet18 test_s_expr_resnet18_mlir test_s_expr_resnet18_exec test_onnx_import_resnet18_exec test_planner_acceptance)
  add_executable(${test_name} ${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE sonicboom)
  target_compile_definitions(${test_name} PRIVATE
    SONICBOOM_SRC_DIR="${CMAKE_SOURCE_DIR}")
endforeach()

# C ABI test, compiled as C to prove <sonicboom/capi.h> is a genuine C header
# and libsonicboom.so exports its entry points with C linkage. Links libm for
# fabsf.
add_executable(test_capi test_capi.c)
target_link_libraries(test_capi PRIVATE sonicboom m)
