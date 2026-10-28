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

# CPU Boom executor (core/src/exec/): the CPU peer of the device training graph
# executor. Blade -> Boom IR -> BufferReusePass -> planner::Graph, then host
# buffers and CPU kernels (SonicBoom nn::/training::, native-torch CPU kernels
# through the Layer 1 direct-call adapter, and inline data movement). It backs
# the MLIR-free sx::Executable facade in <sonicboom/sx/exec.h>.
file(GLOB_RECURSE SONICBOOM_EXEC_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/exec/*.cpp)

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

# vulkan_backend.cpp is the community-backend stub (zero kernels, no Vulkan
# loader). Excluded from the unconditional planner glob and re-added only when
# SONICBOOM_USE_VULKAN is ON, so the shipped core never carries the stub.
list(REMOVE_ITEM SONICBOOM_PLANNER_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/planner/vulkan_backend.cpp)

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

# safetensors reader (core/src/safetensors/). Pure file-format reader: parses
# the 8-byte header length and the JSON tensor directory, validates every byte
# range, and exposes tensor bytes as spans over a read-only mmap. No dtype
# conversion, no device backend, no dependency on the torch safetensors library.
file(GLOB_RECURSE SONICBOOM_SAFETENSORS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/safetensors/*.cpp)

# Tokenizer vocabulary (core/src/vocab/). Layer 2 data layout + GGUF loading
# glue; the BPE/detokenize algorithms are header-only, MIT-derived from ggml
# (core/src/vocab/ggml_derived/bpe_tokenizer.h). No ggml runtime dependency.
file(GLOB_RECURSE SONICBOOM_VOCAB_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/vocab/*.cpp)

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

# Training (core/src/training/): the AdamW host reference and the device/host
# Param storage. Pure C++23 here; the fused device kernel (adamw.cu) is added
# in the CUDA block below.
file(GLOB_RECURSE SONICBOOM_TRAINING_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/src/training/*.cpp)

# libsonicboom.so is SonicBoom's final published core library (SHARED).
# Native-torch and MLIR are statically linked into it. The stable C ABI
# (sonicboom_capi object library) is folded in so its `extern "C"` symbols are
# exported from the one published library that FFI consumers load.
add_library(sonicboom SHARED
  ${SONICBOOM_ADAPTER_SOURCES}
  ${SONICBOOM_SX_SOURCES}
  ${SONICBOOM_EXEC_SOURCES}
  ${SONICBOOM_PLANNER_SOURCES}
  ${SONICBOOM_AUTOGRAD_SOURCES}
  ${SONICBOOM_GGUF_SOURCES}
  ${SONICBOOM_SAFETENSORS_SOURCES}
  ${SONICBOOM_VOCAB_SOURCES}
  ${SONICBOOM_QUANT_SOURCES}
  ${SONICBOOM_NN_SOURCES}
  ${SONICBOOM_MODEL_SOURCES}
  ${SONICBOOM_THREADPOOL_SOURCES}
  ${SONICBOOM_TRAINING_SOURCES}
  ${CMAKE_CURRENT_SOURCE_DIR}/src/mlir/smoke.cpp
  $<TARGET_OBJECTS:sonicboom_capi>)

# SONICBOOM_SANITIZE: instrument SonicBoom's own runtime code. The native-torch
# and MLIR static libraries linked below stay uninstrumented (root
# CMakeLists.txt explains why).
sonicboom_sanitize(sonicboom)

# Drop the static symbol table (and its string table) from the published
# library in Release builds. Dynamic linking reads .dynsym, which is kept, and
# the C ABI is exported there; the static table is a quarter of the file and
# nothing in the runtime reads it. A non-Release build keeps every symbol.
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_link_options(sonicboom PRIVATE -Wl,-s)
endif()

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
  # cuda_driver: the glue JIT (src/planner/glue_jit.cpp) loads compiler-generated
  # PTX with cuModuleLoadData and launches it on the planner's stream.
  # NVTX is header-only in nvtx3: the shim provides inline implementations, so
  # there is no library to link. CUDA >= 12.9 no longer ships libnvtx with the
  # toolkit; an old /usr/include/nvToolsExt.h can still be present and hide the
  # missing symbol until link time, which is why the old nvToolsExt link target
  # is gone.
  target_link_libraries(sonicboom PRIVATE CUDA::cudart CUDA::cublas
                                       CUDA::cuda_driver)

  # The nvtx3 shim lives in the toolkit's include directory. On CUDA 13.x
  # CUDAToolkit_INCLUDE_DIRS is a LIST (e.g. ".../include;.../include/cccl"),
  # so it is searched entry by entry: appending "/nvtx3" to the quoted list
  # would produce one literal-';' path that never matches and leave the include
  # to chance. EXISTS is used rather than find_path because a find in a CUDA
  # language is re-rooted through CMAKE_FIND_ROOT_PATH (NO_DEFAULT_PATH does
  # not bypass that), and because an absent entry is not an error here: a shim
  # on the compiler's own default include path needs no extra flag.
  set(_sb_nvtx3_include "")
  foreach(_sb_cuda_include IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(EXISTS "${_sb_cuda_include}/nvtx3/nvToolsExt.h")
      set(_sb_nvtx3_include "${_sb_cuda_include}")
      break()
    endif()
  endforeach()
  if(_sb_nvtx3_include)
    target_include_directories(sonicboom PRIVATE "${_sb_nvtx3_include}")
  endif()
  # SonicBoom-owned CUDA K-quant matvec kernel (dequantize-on-the-fly gemv).
  # Compiled with nvcc (enable_language(CUDA) + CMAKE_CUDA_ARCHITECTURES=86 are
  # set in the root CMakeLists), statically linked into libsonicboom.so like the
  # other core sources. Its public header exposes only Layer 2 types.
  target_sources(sonicboom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/cuda_quant_matmul.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/cuda_elementwise.cu
    # SonicBoom-owned prefill attention (boom_fattn), the prefill attention
    # implementation. Plain device code, no device lambdas, so it needs no
    # per-source flags.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/boom_fattn.cu
    # boom_fattn's strict-order path (d128/GQA 6): the scalar reference kernel
    # and the byte-identical strict-order tiled kernel it is gated against.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/boom_fattn_strict.cu
    # boom_fattn candidate selection and its runtime JIT (see src/nn/
    # fattn_tune.h). fattn_tune.cu is the autotuner, the per-device selection
    # cache and the startup report; fattn_jit.cpp compiles a candidate the AOT
    # cubin has no image for (other architectures) with NVRTC and caches the
    # PTX. Both are host-side: no device lambdas, no extra flags.
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/fattn_tune.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/fattn_jit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/training/adamw.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/training/backward_ops.cu
    # The training graph executor's forward + AdamW kernels (device_graph.cpp
    # dispatches them; the fused adjoints are backward_ops.cu above).
    ${CMAKE_CURRENT_SOURCE_DIR}/src/training/device_graph_kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/cuda_block.cpp
    # MOSS-Image DiT: Q8_0 weights + a device-resident forward (M2).
    ${CMAKE_CURRENT_SOURCE_DIR}/src/model/moss_dit_cuda.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/planner/sonic_cuda_backend.cpp)
  # ggml-derived MMQ at upstream build parity (step 9): llama.cpp compiles
  # ggml-cuda/mmq.cu with -use_fast_math -extended-lambda. The arch already
  # matches (both trees emit sm_86 cubins) and the standard stays C++20 for
  # SonicBoom sources. Per-source options are the finest granularity available:
  # the derived MMQ kernels are header-only (ggml_derived/mmq.cuh) and this TU
  # is the one that instantiates them, but it also carries SonicBoom's q8_1
  # quantization, rms_norm_quantize_q8_1, gemv and dense-GEMM code.
  #
  # -use_fast_math is NOT applied: step 8 measured it as codegen-neutral for
  # every Type/J here, and with it this TU's q8_1 scale/quantization math
  # changes low bits -- the out-of-tree test_cuda_quant's "q8_1 batched matches
  # integer CPU reference" (Q6_K, rel err 9.4e-4 / 2.9e-2), the out-of-tree
  # test_glue_codegen's
  # byte-identical rms_norm_quantize_q8_1 check against this TU's reference,
  # and the Qwen greedy streams versus their goldens all fail. The numerics
  # rule (tokens bit-identical) outranks flag parity. -extended-lambda has no
  # effect on this TU (no device lambdas in the derived tree) and is kept for
  # the upstream command line.
  #
  # The parity that matters is upstream's __launch_bounds__(256, 1), which
  # lets ptxas use 254 registers for Q3_K J=128 instead of 202 (exactly
  # llama.cpp's allocation; measured with cuobjdump). It lives in
  # ggml_derived/mmq.cuh, next to the kernel it constrains.
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/src/quant/cuda_quant_matmul.cu
    PROPERTIES COMPILE_OPTIONS "-extended-lambda")

  # boom_fattn's runtime JIT compiles the same device code the AOT translation
  # unit includes, so src/nn/boom_fattn_device.cuh is embedded verbatim as a
  # string here and handed to NVRTC. Reading it at configure time and listing it
  # as a configure dependency is what keeps the two from drifting: editing the
  # header regenerates the string on the next build.
  set(_sb_fattn_device
      "${CMAKE_CURRENT_SOURCE_DIR}/src/nn/boom_fattn_device.cuh")
  set(_sb_fattn_src_inc
      "${CMAKE_CURRENT_BINARY_DIR}/fattn_device_src.inc")
  file(READ "${_sb_fattn_device}" _sb_fattn_device_text)
  file(WRITE "${_sb_fattn_src_inc}"
       "// Generated from src/nn/boom_fattn_device.cuh by CMake. Do not edit.\n"
       "static const char kFattnDeviceSource[] = R\"SB_FATTN_DEVICE(\n")
  file(APPEND "${_sb_fattn_src_inc}" "${_sb_fattn_device_text}")
  file(APPEND "${_sb_fattn_src_inc}" ")SB_FATTN_DEVICE\";\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
               "${_sb_fattn_device}")
  target_include_directories(sonicboom PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

  # NVRTC searches no system path: the CUDA headers (<cuda_fp16.h>) have to be
  # passed explicitly, and the directory is a property of the toolkit this
  # library was built against. The JIT also probes the usual locations at
  # runtime, so a program run against a differently installed toolkit still
  # finds them.
  set(_sb_fattn_cuda_includes "")
  foreach(_sb_inc IN LISTS CUDAToolkit_INCLUDE_DIRS)
    if(_sb_inc)
      if(_sb_fattn_cuda_includes)
        string(APPEND _sb_fattn_cuda_includes ":${_sb_inc}")
      else()
        set(_sb_fattn_cuda_includes "${_sb_inc}")
      endif()
    endif()
  endforeach()
  target_compile_definitions(sonicboom PRIVATE
    SONICBOOM_VERSION_STRING="${PROJECT_VERSION}"
    FATTN_JIT_CUDA_INCLUDE="${_sb_fattn_cuda_includes}")
endif()

# Vulkan backend stub (Step 3): pure C++23, no Vulkan loader. OFF by default;
# building with SONICBOOM_USE_VULKAN=ON compiles the zero-kernel scaffold that
# backend authors use as a template (see docs/backend-author-guide.md).
if(SONICBOOM_USE_VULKAN)
  target_sources(sonicboom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/planner/vulkan_backend.cpp)
endif()

# The core itself now only touches MLIR through MLIRContext (the Blade → Boom
# and Boom → graph passes live in sonicboom_compiler, linked below with their
# own dialect/pass dependencies). MLIRIR provides MLIRContext and the builtin
# dialect; its transitive MLIRSupport/LLVMSupport come via the official
# dependency graph. Linked PRIVATE so MLIR types never leak into Layer 2 public
# headers.
#
# The installed MLIR targets export only INTERFACE_LINK_LIBRARIES (no
# INTERFACE_INCLUDE_DIRECTORIES), so the include path must be added explicitly
# (this mirrors mlir/examples/standalone, which uses ${MLIR_INCLUDE_DIRS}).
# ${MLIR_INCLUDE_DIRS} = <install-prefix>/include, covering both mlir/ and llvm/.
#
# MLIR is required (root CMakeLists.txt), so this is unconditional.
target_link_libraries(sonicboom PRIVATE MLIRIR)
target_include_directories(sonicboom PRIVATE ${MLIR_INCLUDE_DIRS})

# The compiler library: the Boom MLIR dialect and the Boom IR passes
# (core/compiler/). A separate static library, linked into the runtime because
# the Boom IR rewrite pipeline is the runtime's only rewrite path
# (<sonicboom/compiler/BoomPasses.h>, run by the plan emitter). PRIVATE, like
# MLIR itself — the passes are
# behind Layer 2, and no MLIR type reaches a public header.
add_subdirectory(compiler)
target_link_libraries(sonicboom PRIVATE sonicboom_compiler)
