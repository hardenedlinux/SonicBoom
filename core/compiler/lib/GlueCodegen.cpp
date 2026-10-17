// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// The glue-region code generator (compiler/GlueCodegen.h).
//
// One region shape so far: decode rms_norm + q8_1 quantize. The kernel is
// emitted in the GPU dialect and lowered with the standard MLIR pipeline
//
//   scf -> cf, arith -> llvm, gpu.module -> nvvm, finalize memref -> llvm,
//   gpu.module -> binary (assembly, through the LLVM NVPTX backend)
//
// The text below is the region written out: a CUDA block (1024 threads) per
// norm row; each thread accumulates its strided share of sum(x^2) in f64; five
// `shfl.down` tree steps (f64 shuffled as two b32 halves); one value per warp
// through shared memory; warp 0 reduces those and publishes
// inv = 1/sqrt(t/n + eps); each warp then quantizes dim/32 blocks, exactly as
// quant::cuda::rms_norm_quantize_q8_1_kernel does (the reference this kernel is
// byte-compared against by tests/test_glue_codegen.cpp).
//
// Two details exist to keep that byte identity:
//   * the f64 shuffle is spelled as two b32 `gpu.shuffle down`s because the
//     NVVM lowering only accepts 32-bit shuffles (the same split nvcc emits for
//     __shfl_down_sync on a double);
//   * roundf's add is `nvvm.addf ... {rnd = rz}`: nvcc lowers roundf to
//     add.rz(x, copysign(0.5, x)) + cvt.rzi, and a plain fadd would round to
//     nearest and could cross an integer boundary where add.rz does not.

#include <sonicboom/compiler/GlueCodegen.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h>
#include <mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/Dialect/GPU/IR/GPUDialect.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/LLVMIR/NVVMDialect.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVM/NVVM/Target.h>
#include <mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h>

#include <llvm/Support/TargetSelect.h>

#include <cstdio>
#include <string>

namespace sonicboom::compiler {
namespace {

// rms_norm_quantize_q8_1: the decode region's kernel. `%CHIP%` is the NVVM
// target chip (sm_86), the only parameter the region's text depends on — the
// row length n is a runtime argument, so one compiled kernel serves every
// norm width of a model.
constexpr const char *kRmsNormQuantizeQ8_1 = R"MLIR(module attributes {gpu.container_module} {
  gpu.module @glue [#nvvm.target<chip = "%CHIP%">] {
    gpu.func @rms_norm_q8_1(%x: !llvm.ptr, %w: !llvm.ptr, %y: !llvm.ptr, %n: i32, %eps: f32) kernel {
      %tid = gpu.thread_id x
      %tid_i = arith.index_cast %tid : index to i32
      %c0i = arith.constant 0 : i32
      %cni = arith.constant 1024 : i32
      %c0 = arith.constant 0.0 : f64
      %c0f = arith.constant 0.0 : f32
      %c1f = arith.constant 1.0 : f32
      %c36 = arith.constant 36 : i32
      %c4 = arith.constant 4 : i32
      %s0 = scf.for %iv = %tid_i to %n step %cni iter_args(%acc = %c0) -> (f64) : i32 {
        %xp = llvm.getelementptr inbounds %x[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %v = llvm.load %xp : !llvm.ptr -> f32
        %sq = arith.mulf %v, %v : f32
        %sq64 = arith.extf %sq : f32 to f64
        %na = arith.addf %acc, %sq64 : f64
        scf.yield %na : f64
      }
      %c31 = arith.constant 31 : i32
      %c5 = arith.constant 5 : i32
      %lane = arith.andi %tid_i, %c31 : i32
      %warp = arith.shrui %tid_i, %c5 : i32
      %c32_64 = arith.constant 32 : i64
      %c16 = arith.constant 16 : i32
      %c8 = arith.constant 8 : i32
      %c2 = arith.constant 2 : i32
      %c1 = arith.constant 1 : i32
      %wid = arith.constant 32 : i32
      %b0 = arith.bitcast %s0 : f64 to i64
      %lo0 = arith.trunci %b0 : i64 to i32
      %h0 = arith.shrui %b0, %c32_64 : i64
      %hi0 = arith.trunci %h0 : i64 to i32
      %lo1, %vv1 = gpu.shuffle down %lo0, %c16, %wid : i32
      %hi1, %vv2 = gpu.shuffle down %hi0, %c16, %wid : i32
      %eq1 = arith.extui %lo1 : i32 to i64
      %eh1 = arith.extui %hi1 : i32 to i64
      %sh1 = arith.shli %eh1, %c32_64 : i64
      %bt1 = arith.ori %eq1, %sh1 : i64
      %f1 = arith.bitcast %bt1 : i64 to f64
      %s1 = arith.addf %s0, %f1 : f64
      %b1 = arith.bitcast %s1 : f64 to i64
      %lo1b = arith.trunci %b1 : i64 to i32
      %h1 = arith.shrui %b1, %c32_64 : i64
      %hi1b = arith.trunci %h1 : i64 to i32
      %lo2, %vv3 = gpu.shuffle down %lo1b, %c8, %wid : i32
      %hi2, %vv4 = gpu.shuffle down %hi1b, %c8, %wid : i32
      %eq2 = arith.extui %lo2 : i32 to i64
      %eh2 = arith.extui %hi2 : i32 to i64
      %sh2 = arith.shli %eh2, %c32_64 : i64
      %bt2 = arith.ori %eq2, %sh2 : i64
      %f2 = arith.bitcast %bt2 : i64 to f64
      %s2 = arith.addf %s1, %f2 : f64
      %b2 = arith.bitcast %s2 : f64 to i64
      %lo2b = arith.trunci %b2 : i64 to i32
      %h2 = arith.shrui %b2, %c32_64 : i64
      %hi2b = arith.trunci %h2 : i64 to i32
      %lo3, %vv5 = gpu.shuffle down %lo2b, %c4, %wid : i32
      %hi3, %vv6 = gpu.shuffle down %hi2b, %c4, %wid : i32
      %eq3 = arith.extui %lo3 : i32 to i64
      %eh3 = arith.extui %hi3 : i32 to i64
      %sh3 = arith.shli %eh3, %c32_64 : i64
      %bt3 = arith.ori %eq3, %sh3 : i64
      %f3 = arith.bitcast %bt3 : i64 to f64
      %s3 = arith.addf %s2, %f3 : f64
      %b3 = arith.bitcast %s3 : f64 to i64
      %lo3b = arith.trunci %b3 : i64 to i32
      %h3 = arith.shrui %b3, %c32_64 : i64
      %hi3b = arith.trunci %h3 : i64 to i32
      %lo4, %vv7 = gpu.shuffle down %lo3b, %c2, %wid : i32
      %hi4, %vv8 = gpu.shuffle down %hi3b, %c2, %wid : i32
      %eq4 = arith.extui %lo4 : i32 to i64
      %eh4 = arith.extui %hi4 : i32 to i64
      %sh4 = arith.shli %eh4, %c32_64 : i64
      %bt4 = arith.ori %eq4, %sh4 : i64
      %f4 = arith.bitcast %bt4 : i64 to f64
      %s4 = arith.addf %s3, %f4 : f64
      %b4 = arith.bitcast %s4 : f64 to i64
      %lo4b = arith.trunci %b4 : i64 to i32
      %h4 = arith.shrui %b4, %c32_64 : i64
      %hi4b = arith.trunci %h4 : i64 to i32
      %lo5, %vv9 = gpu.shuffle down %lo4b, %c1, %wid : i32
      %hi5, %vv10 = gpu.shuffle down %hi4b, %c1, %wid : i32
      %eq5 = arith.extui %lo5 : i32 to i64
      %eh5 = arith.extui %hi5 : i32 to i64
      %sh5 = arith.shli %eh5, %c32_64 : i64
      %bt5 = arith.ori %eq5, %sh5 : i64
      %f5 = arith.bitcast %bt5 : i64 to f64
      %s5 = arith.addf %s4, %f5 : f64
      %c0idx = arith.constant 0 : index
      %smem = gpu.dynamic_shared_memory : memref<?xi8, #gpu.address_space<workgroup>>
      %wpart = memref.view %smem[%c0idx][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<32xf64, #gpu.address_space<workgroup>>
      %is_zero_lane = arith.cmpi eq, %lane, %c0i : i32
      scf.if %is_zero_lane {
        %widx = arith.index_cast %warp : i32 to index
        memref.store %s5, %wpart[%widx] : memref<32xf64, #gpu.address_space<workgroup>>
      }
      gpu.barrier
      %lane_idx = arith.index_cast %lane : i32 to index
      %c32i = arith.constant 32 : i32
      %c256 = arith.constant 256 : index
      %invm = memref.view %smem[%c256][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
      %in_w0 = arith.cmpi slt, %tid_i, %c32i : i32
      scf.if %in_w0 {
        %t0 = memref.load %wpart[%lane_idx] : memref<32xf64, #gpu.address_space<workgroup>>
        %tb0 = arith.bitcast %t0 : f64 to i64
        %tlo0 = arith.trunci %tb0 : i64 to i32
        %th0 = arith.shrui %tb0, %c32_64 : i64
        %thi0 = arith.trunci %th0 : i64 to i32
        %tlo1, %tv1 = gpu.shuffle down %tlo0, %c16, %wid : i32
        %thi1, %tv2 = gpu.shuffle down %thi0, %c16, %wid : i32
        %te1 = arith.extui %tlo1 : i32 to i64
        %the1 = arith.extui %thi1 : i32 to i64
        %tsh1 = arith.shli %the1, %c32_64 : i64
        %tbt1 = arith.ori %te1, %tsh1 : i64
        %tf1 = arith.bitcast %tbt1 : i64 to f64
        %t1 = arith.addf %t0, %tf1 : f64
        %tb1 = arith.bitcast %t1 : f64 to i64
        %tlo1b = arith.trunci %tb1 : i64 to i32
        %th1 = arith.shrui %tb1, %c32_64 : i64
        %thi1b = arith.trunci %th1 : i64 to i32
        %tlo2, %tv3 = gpu.shuffle down %tlo1b, %c8, %wid : i32
        %thi2, %tv4 = gpu.shuffle down %thi1b, %c8, %wid : i32
        %te2 = arith.extui %tlo2 : i32 to i64
        %the2 = arith.extui %thi2 : i32 to i64
        %tsh2 = arith.shli %the2, %c32_64 : i64
        %tbt2 = arith.ori %te2, %tsh2 : i64
        %tf2 = arith.bitcast %tbt2 : i64 to f64
        %t2 = arith.addf %t1, %tf2 : f64
        %tb2 = arith.bitcast %t2 : f64 to i64
        %tlo2b = arith.trunci %tb2 : i64 to i32
        %th2 = arith.shrui %tb2, %c32_64 : i64
        %thi2b = arith.trunci %th2 : i64 to i32
        %tlo3, %tv5 = gpu.shuffle down %tlo2b, %c4, %wid : i32
        %thi3, %tv6 = gpu.shuffle down %thi2b, %c4, %wid : i32
        %te3 = arith.extui %tlo3 : i32 to i64
        %the3 = arith.extui %thi3 : i32 to i64
        %tsh3 = arith.shli %the3, %c32_64 : i64
        %tbt3 = arith.ori %te3, %tsh3 : i64
        %tf3 = arith.bitcast %tbt3 : i64 to f64
        %t3 = arith.addf %t2, %tf3 : f64
        %tb3 = arith.bitcast %t3 : f64 to i64
        %tlo3b = arith.trunci %tb3 : i64 to i32
        %th3 = arith.shrui %tb3, %c32_64 : i64
        %thi3b = arith.trunci %th3 : i64 to i32
        %tlo4, %tv7 = gpu.shuffle down %tlo3b, %c2, %wid : i32
        %thi4, %tv8 = gpu.shuffle down %thi3b, %c2, %wid : i32
        %te4 = arith.extui %tlo4 : i32 to i64
        %the4 = arith.extui %thi4 : i32 to i64
        %tsh4 = arith.shli %the4, %c32_64 : i64
        %tbt4 = arith.ori %te4, %tsh4 : i64
        %tf4 = arith.bitcast %tbt4 : i64 to f64
        %t4 = arith.addf %t3, %tf4 : f64
        %tb4 = arith.bitcast %t4 : f64 to i64
        %tlo4b = arith.trunci %tb4 : i64 to i32
        %th4 = arith.shrui %tb4, %c32_64 : i64
        %thi4b = arith.trunci %th4 : i64 to i32
        %tlo5, %tv9 = gpu.shuffle down %tlo4b, %c1, %wid : i32
        %thi5, %tv10 = gpu.shuffle down %thi4b, %c1, %wid : i32
        %te5 = arith.extui %tlo5 : i32 to i64
        %the5 = arith.extui %thi5 : i32 to i64
        %tsh5 = arith.shli %the5, %c32_64 : i64
        %tbt5 = arith.ori %te5, %tsh5 : i64
        %tf5 = arith.bitcast %tbt5 : i64 to f64
        %t5 = arith.addf %t4, %tf5 : f64
        %fn = arith.sitofp %n : i32 to f64
        %tn = arith.divf %t5, %fn : f64
        %t32 = arith.truncf %tn : f64 to f32
        %te = arith.addf %t32, %eps : f32
        %sq = nvvm.sqrt %te {rnd = #nvvm.fp_rnd_mode<rn>} : f32
        %one = arith.constant 1.0 : f32
        %inv = arith.divf %one, %sq : f32
        %is_l0 = arith.cmpi eq, %lane, %c0i : i32
        scf.if %is_l0 {
          %i0i = arith.constant 0 : index
          memref.store %inv, %invm[%i0i] : memref<4xf32, #gpu.address_space<workgroup>>
        }
      }
      gpu.barrier
      %i0b = arith.constant 0 : index
      %ivv = memref.load %invm[%i0b] : memref<4xf32, #gpu.address_space<workgroup>>
      %nb = arith.divsi %n, %c32i : i32
      scf.for %ib = %warp to %nb step %c32i : i32 {
        %base = arith.muli %ib, %c32i : i32
        %xi = arith.addi %base, %lane : i32
        %xp2 = llvm.getelementptr inbounds %x[%xi] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xv = llvm.load %xp2 : !llvm.ptr -> f32
        %wp2 = llvm.getelementptr inbounds %w[%xi] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %wv = llvm.load %wp2 : !llvm.ptr -> f32
        %p1 = arith.mulf %xv, %ivv : f32
        %nv = arith.mulf %p1, %wv : f32
        %nbits = arith.bitcast %nv : f32 to i32
        %signclr = arith.constant 2147483647 : i32
        %ab2 = arith.andi %nbits, %signclr : i32
        %abf = arith.bitcast %ab2 : i32 to f32
        %amax0 = arith.maximumf %c0f, %abf : f32
        %am1, %av1 = gpu.shuffle xor %amax0, %c16, %wid : f32
        %am1m = arith.maximumf %amax0, %am1 : f32
        %am2, %av2 = gpu.shuffle xor %am1m, %c8, %wid : f32
        %am2m = arith.maximumf %am1m, %am2 : f32
        %am3, %av3 = gpu.shuffle xor %am2m, %c4, %wid : f32
        %am3m = arith.maximumf %am2m, %am3 : f32
        %am4, %av4 = gpu.shuffle xor %am3m, %c2, %wid : f32
        %am4m = arith.maximumf %am3m, %am4 : f32
        %am5, %av5 = gpu.shuffle xor %am4m, %c1, %wid : f32
        %amaxf = arith.maximumf %am4m, %am5 : f32
        %c127 = arith.constant 127.0 : f32
        %d = arith.divf %amaxf, %c127 : f32
        %off = arith.muli %ib, %c36 : i32
        %dp = llvm.getelementptr inbounds %y[%off] : (!llvm.ptr, i32) -> !llvm.ptr, i8
        scf.if %is_zero_lane {
          llvm.store %d, %dp : f32, !llvm.ptr
        }
        %dzero = arith.cmpf oeq, %d, %c0f : f32
        %safed = arith.select %dzero, %c1f, %d : f32
        %ratio = arith.divf %nv, %safed : f32
        %rbits = arith.bitcast %ratio : f32 to i32
        %cmin = arith.constant -2147483648 : i32
        %rsign = arith.andi %rbits, %cmin : i32
        %chalf = arith.constant 1056964608 : i32
        %rhalf = arith.ori %rsign, %chalf : i32
        %rhf = arith.bitcast %rhalf : i32 to f32
        %rsum = nvvm.addf %ratio, %rhf {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %qi = arith.fptosi %rsum : f32 to i32
        %q8 = arith.trunci %qi : i32 to i8
        %zi8 = arith.constant 0 : i8
        %qsel = arith.select %dzero, %zi8, %q8 : i8
        %qoff = arith.addi %off, %c4 : i32
        %qoff2 = arith.addi %qoff, %lane : i32
        %qp = llvm.getelementptr inbounds %y[%qoff2] : (!llvm.ptr, i32) -> !llvm.ptr, i8
        llvm.store %qsel, %qp : i8, !llvm.ptr
      }
      gpu.return
    }
  }
}
)MLIR";

// The region text for `kind`, with the target chip substituted. Empty when the
// kind has no generator.
std::string region_text(GlueRegionKind kind, int sm_arch) {
  const char *text = nullptr;
  switch (kind) {
    case GlueRegionKind::RmsNormQuantizeQ8_1:
      text = kRmsNormQuantizeQ8_1;
      break;
  }
  if (text == nullptr) return {};

  const std::string chip = "sm_" + std::to_string(sm_arch);
  std::string out;
  out.reserve(std::char_traits<char>::length(text) + chip.size());
  const std::string_view view(text);
  const std::string_view placeholder = "%CHIP%";
  std::size_t pos = 0;
  while (true) {
    const std::size_t at = view.find(placeholder, pos);
    if (at == std::string_view::npos) {
      out.append(view.substr(pos));
      break;
    }
    out.append(view.substr(pos, at - pos));
    out.append(chip);
    pos = at + placeholder.size();
  }
  return out;
}

// The NVPTX target's serialization (gpu.module -> binary with the assembly
// format) runs the LLVM NVPTX backend inside this process, so its target
// description must be registered before the pass runs. The tool-level
// equivalent (mlir-opt) does this for every built target at startup.
void register_nvptx_target() {
  static const bool once = [] {
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
    return true;
  }();
  (void)once;
}

} // namespace

std::optional<GlueRegionKind> glue_region_kind_from_name(
    std::string_view name) {
  if (name == "rms_norm_quantize_q8_1")
    return GlueRegionKind::RmsNormQuantizeQ8_1;
  return std::nullopt;
}

const char *glue_region_name(GlueRegionKind kind) noexcept {
  switch (kind) {
    case GlueRegionKind::RmsNormQuantizeQ8_1:
      return "rms_norm_quantize_q8_1";
  }
  return "?";
}

const char *glue_kernel_entry(GlueRegionKind kind) noexcept {
  switch (kind) {
    case GlueRegionKind::RmsNormQuantizeQ8_1:
      return "rms_norm_q8_1";
  }
  return "?";
}

std::optional<std::string> generate_glue_ptx(GlueRegionKind kind,
                                             int sm_arch) {
  const std::string text = region_text(kind, sm_arch);
  if (text.empty()) {
    std::fprintf(stderr, "glue codegen: region '%s' has no generator\n",
                 glue_region_name(kind));
    return std::nullopt;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::memref::MemRefDialect, mlir::NVVM::NVVMDialect,
                  mlir::scf::SCFDialect>();
  // The `#nvvm.target` serialization and the nvvm-dialect LLVM IR translation
  // are external models: without them the module-to-binary pass has no
  // serializer to call. mlir-opt registers both at startup.
  mlir::NVVM::registerNVVMTargetInterfaceExternalModels(registry);
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  // The GPU-to-NVVM pass lowers memref ops through the memref dialect's
  // ConvertToLLVMPatternInterface (it walks the loaded dialects' interfaces).
  // mlir-opt registers it at startup; in-process it is an external model.
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::MLIRContext context(registry, mlir::MLIRContext::Threading::DISABLED);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  if (!module) {
    std::fprintf(stderr, "glue codegen: the %s module did not parse\n",
                 glue_region_name(kind));
    return std::nullopt;
  }

  register_nvptx_target();

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  // Everything ring-fenced by the conversion lives in the gpu.module (the
  // module holds no memref ops of its own), so the GPU conversion's memref
  // patterns — fed by the memref dialect's ConvertToLLVMPatternInterface — do
  // the whole lowering. Running finalize-memref-to-llvm here as well would only
  // fail on the workgroup address space it cannot map.
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createConvertGpuOpsToNVVMOps());
  // gpu.module is isolated from above, so the casts the GPU conversion leaves
  // behind need a reconcile inside it as well as at module level.
  pm.addNestedPass<mlir::gpu::GPUModuleOp>(
      mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  mlir::GpuModuleToBinaryPassOptions binary;
  binary.compilationTarget = "assembly";
  pm.addPass(mlir::createGpuModuleToBinaryPass(binary));

  // The NVVM serializer emits a "LibDevice path" error when the CUDA toolkit's
  // libdevice is not where it expects it, yet still produces the PTX: these
  // kernels call no libdevice function, so the link is a no-op. Capture the
  // diagnostics for this run and only surface them when lowering actually
  // fails, so a cold compile does not print a non-error at every launch.
  std::string diagnostics;
  mlir::ScopedDiagnosticHandler capture(
      &context, [&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream os(diagnostics);
        diag.print(os);
        os << '\n';
        return mlir::success();
      });

  if (mlir::failed(pm.run(*module))) {
    std::fprintf(stderr, "glue codegen: lowering the %s module failed\n%s",
                 glue_region_name(kind), diagnostics.c_str());
    return std::nullopt;
  }

  // The binary pass replaces the gpu.module with a gpu.binary holding one
  // object per target. This pipeline has one target, and it must be assembly.
  std::string ptx;
  module->walk([&](mlir::gpu::BinaryOp op) {
    for (mlir::Attribute entry : op.getObjects()) {
      auto object = mlir::dyn_cast<mlir::gpu::ObjectAttr>(entry);
      if (!object || object.getFormat() != mlir::gpu::CompilationTarget::Assembly)
        continue;
      ptx = object.getObject().getValue().str();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (ptx.empty()) {
    std::fprintf(stderr,
                 "glue codegen: the %s module produced no PTX object\n",
                 glue_region_name(kind));
    return std::nullopt;
  }
  return ptx;
}

} // namespace sonicboom::compiler
