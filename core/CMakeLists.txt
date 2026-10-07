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

# Static execution planner (core/src/planner/). Pure SonicBoom C++ that
# consumes the sx IR and the sx::Executable facade; no MLIR/LLVM/ATen/c10
# headers (the public facade in <sonicboom/sx/exec.h> is MLIR-free).
file(GLOB_RECURSE SONICBOOM_PLANNER_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/planner/*.cpp)
# sonic_cuda_backend.cpp is host-only orchestration that depends on the CUDA
# _dev primitives (cuda_resident.h / quant/cuda_resident.h) and <cuda_runtime.h>,
# so like model/cuda_block.cpp it is compiled by g++ (C++23) only when
# SONICBOOM_USE_CUDA is ON. Exclude it from the unconditional planner glob; it is
# re-added in the CUDA block below.
list(REMOVE_ITEM SONICBOOM_PLANNER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/planner/sonic_cuda_backend.cpp)

# Minimal autograd (core/src/autograd/). Layer 2 only: it dispatches through
# the nt:: adapter (no direct c10/ATen use), so it belongs to the same public
# library surface as the planner/sx code above.
file(GLOB_RECURSE SONICBOOM_AUTOGRAD_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/autograd/*.cpp)

# GGUF v3 reader (core/src/gguf/). Pure file-format reader: parses the header,
# KV metadata and tensor directory, and exposes raw (quantized) tensor bytes by
# offset + size. No ggml, llama.cpp, MLIR, or device backend — format constants
# only, re-declared by reference to the public GGUF v3 spec.
file(GLOB_RECURSE SONICBOOM_GGUF_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gguf/*.cpp)

# Quantized weight representation + reference dequantization (core/src/quant/).
# Backend-neutral storage semantics and the Q3_K/Q4_K/Q5_K scalar reference
# kernels. No ggml, llama.cpp, MLIR, or device backend; pure C++23.
file(GLOB_RECURSE SONICBOOM_QUANT_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/*.cpp)

# Transformer operator primitives (core/src/nn/). Pure C++23 float32 reference
# kernels (RMSNorm, SiLU/GELU, RoPE, attention, …) for the native inference
# path. No ggml, llama.cpp, MLIR, or device backend.
file(GLOB_RECURSE SONICBOOM_NN_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/*.cpp)

# Gemma 4 model subsystem (core/src/model/): the explicit model loader + config,
# weight binding, 42-layer plan, and the per-layer embedding / single-block
# reference execution path. Pure C++23, depends only on gguf + quant + nn.
file(GLOB_RECURSE SONICBOOM_MODEL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model/*.cpp)
# cuda_block.cpp is host-only orchestration (no device kernels) but depends on
# <cuda_runtime.h> and the CUDA _dev primitives, so it is compiled by g++
# (C++23) instead of nvcc (C++20) and only built when SONICBOOM_USE_CUDA is ON.
# Exclude it from the unconditional model glob; it is re-added in the CUDA block.
list(REMOVE_ITEM SONICBOOM_MODEL_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/model/cuda_block.cpp)

# CPU thread pool (core/src/thread_pool.cpp): a process-wide, SonicBoom-owned
# data-parallel executor used by the quantized matmul (and later the elementwise
# primitives) to split row/element loops across workers. Pure C++23.
file(GLOB_RECURSE SONICBOOM_THREADPOOL_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/thread_pool.cpp)

# libsonicboom.so is SonicBoom's final published core library (SHARED).
# Native-torch and MLIR are statically linked into it. The stable C ABI
# (sonicboom_capi object library) is folded in so its `extern "C"` symbols are
# exported from the one published library that FFI consumers load.
add_library(sonicboom SHARED
  ${SONICBOOM_ADAPTER_SOURCES}
  ${SONICBOOM_SX_SOURCES}
  ${SONICBOOM_PLANNER_SOURCES}
  ${SONICBOOM_AUTOGRAD_SOURCES}
  ${SONICBOOM_GGUF_SOURCES}
  ${SONICBOOM_QUANT_SOURCES}
  ${SONICBOOM_NN_SOURCES}
  ${SONICBOOM_MODEL_SOURCES}
  ${SONICBOOM_THREADPOOL_SOURCES}
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
#
# aten_core is a STATIC library whose generated Register{Schema,CPU,...}*.cpp
# register kernels via static initializers (TORCH_LIBRARY / TORCH_LIBRARY_IMPL).
# Without --whole-archive the linker drops those objects (no symbol is referenced
# from libsonicboom.so's own objects), so the dispatcher would find no operators
# at runtime. c10 is whole-archived alongside aten_core for the same reason:
# c10/cuda/impl/CUDAGuardImpl.cpp is a static-initializer-ONLY object
# (C10_REGISTER_GUARD_IMPL(CUDA, ...) — all CUDAGuardImpl methods are inline),
# so no symbol references it and the normal link drops it, leaving
# getDeviceGuardImpl(CUDA) unregistered ("not linked with support for cuda
# devices"). The --no-whole-archive immediately after closes the scope so the
# later MLIR/LLVM deps stay normally-linked.
target_link_libraries(sonicboom PRIVATE
  "-Wl,--whole-archive" aten_core c10 "-Wl,--no-whole-archive")

# CUDA: define the adapter's build-time switch (guards cuda_available() and the
# CUDA tensor factory in core/layer1/adapter/tensor.cpp) and link cudart for
# cudaGetDeviceCount. The heavier CUDA libs (cublas/cusolver/cusparse/cufft/
# nvrtc) flow through from aten_core's PRIVATE links (static lib → $<LINK_ONLY:>),
# and cudart is already PUBLIC on c10; this explicit link keeps the adapter's
# direct cudart use self-documenting.
if(SONICBOOM_USE_CUDA)
  target_compile_definitions(sonicboom PRIVATE SONICBOOM_USE_CUDA=1)
  target_link_libraries(sonicboom PRIVATE CUDA::cudart)
  # SonicBoom-owned CUDA K-quant matvec kernel (dequantize-on-the-fly gemv).
  # Compiled with nvcc (enable_language(CUDA) + CMAKE_CUDA_ARCHITECTURES=86 are
  # set in the root CMakeLists), statically linked into libsonicboom.so like the
  # other core sources. Its public header exposes only Layer 2 types.
  target_sources(sonicboom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/cuda_quant_matmul.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/cuda_elementwise.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/cuda_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/planner/sonic_cuda_backend.cpp)
endif()

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
