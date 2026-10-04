# SonicBoom core — Layer 2 (nt::) public API + Layer 1 adapter.
#
# Layer boundaries (manifest M2):
#   - core/include/sonicboom/  — the public nt:: API. MUST NOT include
#     native-torch headers. This is the only include dir exposed PUBLICly.
#   - core/layer1/             — the adapter: the only SonicBoom-owned code that
#     uses c10/ATen headers (PRIVATE include dirs below).
#
# The adapter compiles against the M1 native-torch targets (aten_core + c10),
# which are built by third_party/native-torch/CMakeLists.txt (added first in the
# root CMakeLists.txt).

# Native-torch include roots, mirroring third_party/native-torch/CMakeLists.txt.
# (Its binary dir is ${CMAKE_BINARY_DIR}/third_party/native-torch, so the
# generated headers land under .../generated.)
set(SONICBOOM_NATIVE_TORCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/native-torch)

file(GLOB_RECURSE SONICBOOM_ADAPTER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/layer1/*.cpp)

# S-Expr v0.1 parser + IR (core/src/sx/). Pure SonicBoom C++: no native-torch,
# no MLIR, no Guile. It sits above Layer 2 and feeds the future MLIR lowering.
file(GLOB_RECURSE SONICBOOM_SX_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sx/*.cpp)

# libsonicboom.so is SonicBoom's final published core library (SHARED).
# Native-torch and MLIR are statically linked into it. The stable C ABI
# (sonicboom_capi object library) is folded in so its `extern "C"` symbols are
# exported from the one published library that FFI consumers load.
add_library(sonicboom SHARED
  ${SONICBOOM_ADAPTER_SOURCES}
  ${SONICBOOM_SX_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mlir/smoke.cpp
  $<TARGET_OBJECTS:sonicboom_capi>)

target_include_directories(sonicboom
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
         ${CMAKE_SOURCE_DIR}/capi/include
  PRIVATE
    ${SONICBOOM_NATIVE_TORCH_ROOT}
    ${SONICBOOM_NATIVE_TORCH_ROOT}/aten/src
    ${CMAKE_BINARY_DIR}/third_party/native-torch/generated)

# SonicBoom-owned code targets C++23 (global default); the native-torch targets
# carry their own C++17 constraint. Linking aten_core is PRIVATE so the
# native-torch dependency does not leak into Layer 2 consumers.
target_link_libraries(sonicboom PRIVATE aten_core)

# MLIRIR provides MLIRContext / ModuleOp / BuiltinDialect and the asm printer;
# its transitive LLVM dependencies (MLIRSupport, LLVMSupport, …) come via the
# official dependency graph. Linked PRIVATE so MLIR types never leak into
# Layer 2 public headers.
#
# The installed MLIR targets export only INTERFACE_LINK_LIBRARIES (no
# INTERFACE_INCLUDE_DIRECTORIES), so the include path must be added explicitly
# (this mirrors mlir/examples/standalone, which uses ${MLIR_INCLUDE_DIRS}).
# ${MLIR_INCLUDE_DIRS} = <install-prefix>/include, covering both mlir/ and llvm/.
if(SONICBOOM_ENABLE_MLIR)
  # S-Expr → MLIR lowering dialects (core/src/sx/lowering.cpp).
  target_link_libraries(sonicboom PRIVATE
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    # CPU JIT execution (core/src/sx/exec.cpp): the lowering pipeline dialects
    # and passes (bufferize → linalg-to-loops → … → LLVM dialect).
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRBufferizationToMemRef
    MLIRLinalgTransforms
    # One-shot-bufferize external models for arith/tensor (constants) and the
    # func.func boundary live in the *Transforms libraries below.
    MLIRArithTransforms
    MLIRTensorTransforms
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRAffineDialect
    MLIRAffineToStandard
    MLIRControlFlowDialect
    MLIRControlFlowToLLVM
    MLIRArithToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRLLVMDialect
    # MLIR ExecutionEngine (JIT) + the builtin/LLVM LLVM-IR translations it needs.
    MLIRExecutionEngine
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    # Native (x86-64) target asm parser for InitializeNativeTargetAsmParser;
    # the X86 target/desc/info libs come transitively via MLIRExecutionEngine.
    LLVMX86AsmParser)
  target_include_directories(sonicboom PRIVATE ${MLIR_INCLUDE_DIRS})
endif()
