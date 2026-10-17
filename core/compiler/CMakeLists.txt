# SonicBoom compiler — the MLIR side of the project (design/mlir-migration.md
# §6-§9): the `boom` dialect, the IR the planner's Graph is lifted into, and the
# passes that rewrite that IR.
#
# Deliberately a SEPARATE library from `sonicboom`, but linked into
# libsonicboom.so: the runtime's consumers are the plan emitter's IR builder
# (lib/BoomBuilder.cpp) and the Boom IR rewrite pipeline (lib/BoomPasses.cpp),
# which is the only rewrite path there is
# (<sonicboom/compiler/BoomPasses.h>).
#
# The whole directory is MLIR, which is a required dependency (root
# CMakeLists.txt).

# --- TableGen: the generated dialect code ------------------------------------
#
# MLIRConfig exports MLIR_TABLEGEN_EXE as the bare program name ("mlir-tblgen"),
# so the directory it was installed into comes from LLVMConfig's
# LLVM_TOOLS_BINARY_DIR. Going through mlir-tblgen directly rather than
# add_mlir_dialect also lets the generated files keep the name of the header
# that includes them (BoomOps.td -> BoomOps.h.inc), and avoids the
# mlir-headers target that a pre-built install does not export.
if(NOT EXISTS "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen")
  message(FATAL_ERROR
    "mlir-tblgen was not found in ${LLVM_TOOLS_BINARY_DIR}. The Boom dialect's "
    "generated code is built with the same MLIR install the runtime links "
    "against, so it must ship with the tools (MLIR_TABLEGEN_EXE=${MLIR_TABLEGEN_EXE}).")
endif()
set(BOOM_TBLGEN "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen")

set(BOOM_TD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/sonicboom/compiler)
set(BOOM_GEN_ROOT ${CMAKE_CURRENT_BINARY_DIR}/include)
set(BOOM_GEN_DIR ${BOOM_GEN_ROOT}/sonicboom/compiler)

# The .td files, in include order. Every tablegen rule depends on all of them:
# BoomOps.td includes BoomTypes.td includes BoomDialect.td, and tablegen sees no
# per-file granularity.
set(BOOM_TD_FILES
  ${BOOM_TD_DIR}/BoomDialect.td
  ${BOOM_TD_DIR}/BoomTypes.td
  ${BOOM_TD_DIR}/BoomOps.td)

# -I for each MLIR include root (the .td files include mlir/IR/OpBase.td and
# friends) and for the .td directory itself (their own includes are relative to
# the including file, which tablegen resolves either way).
set(BOOM_TBLGEN_INCLUDES)
foreach(dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND BOOM_TBLGEN_INCLUDES "-I" "${dir}")
endforeach()
list(APPEND BOOM_TBLGEN_INCLUDES "-I" "${BOOM_TD_DIR}")

# boom_tablegen(<output> <generator> <input .td>)
#
# `<output>` is the file name the hand-written headers include, so the rule that
# produces it is readable at the site that consumes it.
function(boom_tablegen output generator input)
  add_custom_command(
    OUTPUT ${BOOM_GEN_DIR}/${output}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${BOOM_GEN_DIR}
    COMMAND ${BOOM_TBLGEN} ${generator} ${BOOM_TBLGEN_INCLUDES}
            -o ${BOOM_GEN_DIR}/${output} ${BOOM_TD_DIR}/${input}
    DEPENDS ${BOOM_TD_FILES}
    COMMENT "TableGen: ${output} from ${input}"
    VERBATIM)
  set(BOOM_GEN_FILES ${BOOM_GEN_FILES} ${BOOM_GEN_DIR}/${output} PARENT_SCOPE)
endfunction()

boom_tablegen(BoomDialect.h.inc   -gen-dialect-decls  BoomDialect.td)
boom_tablegen(BoomDialect.cpp.inc -gen-dialect-defs   BoomDialect.td)
boom_tablegen(BoomAttrs.h.inc     -gen-attrdef-decls  BoomDialect.td)
boom_tablegen(BoomAttrs.cpp.inc   -gen-attrdef-defs   BoomDialect.td)
boom_tablegen(BoomTypes.h.inc     -gen-typedef-decls  BoomTypes.td)
boom_tablegen(BoomTypes.cpp.inc   -gen-typedef-defs   BoomTypes.td)
boom_tablegen(BoomOps.h.inc       -gen-op-decls       BoomOps.td)
boom_tablegen(BoomOps.cpp.inc     -gen-op-defs        BoomOps.td)

add_custom_target(boom_incgen DEPENDS ${BOOM_GEN_FILES})

# --- libsonicboom_compiler.a -------------------------------------------------

add_library(sonicboom_compiler STATIC
  lib/BoomDialect.cpp
  lib/BoomTypes.cpp
  lib/BoomVerify.cpp
  lib/BoomBuilder.cpp
  lib/BoomConversion.cpp
  lib/BladeToBoom.cpp
  lib/BoomPasses.cpp
  lib/GlueCodegen.cpp)

# Statically linked into libsonicboom.so once the runtime consumes Boom IR, and
# already read from the tests next to it, so PIC now rather than as a later fix.
set_target_properties(sonicboom_compiler PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The installed MLIR is built without RTTI, so `typeinfo for mlir::Dialect` does
# not exist anywhere. Defining BoomDialect (a subclass of mlir::Dialect) with
# RTTI on emits a vtable whose RTTI slot names that typeinfo, and the link
# fails; MLIR's own out-of-tree examples carry the same flag. Nothing in the
# dialect uses dynamic_cast or typeid.
target_compile_options(sonicboom_compiler PRIVATE -fno-rtti)

add_dependencies(sonicboom_compiler boom_incgen)

# PUBLIC: the dialect's public headers (BoomDialect.h, BoomTypes.h, BoomOps.h)
# include the .inc files by name, and the generated classes include real MLIR
# headers — the installed MLIR targets export no include directories, so a
# consumer needs ${MLIR_INCLUDE_DIRS} as well (the same reason core/CMakeLists
# spells it out for the runtime).
target_include_directories(sonicboom_compiler PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${BOOM_GEN_ROOT}
  ${MLIR_INCLUDE_DIRS})

# PRIVATE: the Layer 2 headers the type verifiers read the frozen sx::DType names
# and the quant format table from.
target_include_directories(sonicboom_compiler PRIVATE
  ${CMAKE_SOURCE_DIR}/core/include)

# MLIRIR carries MLIRContext and the builtin dialect whose types the op
# signatures use; MLIRSideEffectInterfaces carries the Resource/Effect bases the
# KV-cache memory effects are defined with. MLIRPass is the pass infrastructure
# (PassManager, PassWrapper) the Boom IR pipeline runs its passes through
# (BoomPasses.cpp). All three bring MLIRSupport and LLVMSupport with them, which
# is where SmallVector/DenseMap/StringSwitch come from.
target_link_libraries(sonicboom_compiler PRIVATE
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRPass
  # Glue codegen (lib/GlueCodegen.cpp): the GPU/NVVM dialect and the
  # gpu.module -> NVVM -> PTX pipeline. MLIRNVVMTarget owns the assembly
  # serialization; LLVMNVPTX* is the backend it runs in-process.
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRNVVMDialect
  MLIRNVVMTarget
  MLIRNVVMToLLVM
  # The nvvm dialect's LLVM IR translation interface (nvvm ops -> LLVM IR for
  # the NVPTX backend). Without it the target serializer has no translator.
  MLIRNVVMToLLVMIRTranslation
  # The gpu.module -> LLVM module translation (the serializer translates the
  # module before handing it to NVPTX).
  MLIRGPUToLLVMIRTranslation
  # The llvm dialect's own translation (llvm.mlir.constant et al.).
  MLIRLLVMIRToLLVMTranslation
  MLIRTargetLLVMIRExport
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRSCFToControlFlow
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRLLVMDialect
  LLVMNVPTXCodeGen
  LLVMNVPTXDesc
  LLVMNVPTXInfo)
