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
// Three region shapes so far: decode rms_norm + q8_1 quantize, prefill
// residual add + RMS-norm + mmq q8_1 quantize, and prefill RMS-norm + residual
// add. The kernels are emitted in the
// GPU dialect and lowered with the standard MLIR pipeline
//
//   scf -> cf, arith -> llvm, gpu.module -> nvvm, finalize memref -> llvm,
//   gpu.module -> binary (assembly, through the LLVM NVPTX backend)
//
// The first text below is the decode region written out: a CUDA block (1024
// threads) per norm row; each thread accumulates its strided share of sum(x^2)
// in f64; five `shfl.down` tree steps (f64 shuffled as two b32 halves); one
// value per warp through shared memory; warp 0 reduces those and publishes
// inv = 1/sqrt(t/n + eps); each warp then quantizes dim/32 blocks, exactly as
// quant::cuda::rms_norm_quantize_q8_1_kernel does (the reference this kernel is
// byte-compared against by tests/test_glue_codegen.cpp).
//
// The second is the prefill region: a 256-thread block per token row. Phase A
// adds the residual, writes the sum as the region's second output and reduces
// sum(v^2) exactly as rms_norm_inv_kernel does; phase B quantizes the row into
// nblk dim/128 mmq blocks exactly as scale_quantize_mmq_q8_1_kernel does,
// writing the same 144-byte block_q8_1_mmq layout (d, optional ds half2, qs).
// Reference: nn::cuda::add_dev + quant::cuda::rms_norm_quantize_mmq_q8_1_dev.
//
// Details that exist to keep byte identity:
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

// add_rms_norm_quantize_mmq_q8_1: the prefill region. One 256-thread block per
// token: phase A writes the residual sum and accumulates sum(v^2) in f64 with
// rms_norm_inv_kernel's reduction tree; phase B applies inv and the norm weight
// in the quantize_mmq_q8_1 geometry (one 128-element block per warp, float4 per
// lane) and writes the block_q8_1_mmq activation. `ds` is a runtime argument:
// 2 selects the D4_Q8_1 quantize (d = amax/127), everything else the D4/DS4 one
// (d_inv = 127/amax); the DS4 layout additionally stores the exp-scaled sum as
// a half2, exactly as quantize_mmq_q8_1_block does.
constexpr const char *kAddRmsNormQuantizeMmqQ8_1 = R"MLIR(module attributes {gpu.container_module} {
  gpu.module @glue [#nvvm.target<chip = "%CHIP%">] {
    gpu.func @add_rms_norm_q8_1_mmq(%a: !llvm.ptr, %b: !llvm.ptr, %w: !llvm.ptr, %y: !llvm.ptr, %res: !llvm.ptr, %dim: i32, %n: i32, %ds: i32, %eps: f32) kernel {
      %token = gpu.block_id x
      %tid = gpu.thread_id x
      %tid_i = arith.index_cast %tid : index to i32
      %c0i = arith.constant 0 : i32
      %c1i = arith.constant 1 : i32
      %c2i = arith.constant 2 : i32
      %c3i = arith.constant 3 : i32
      %c4i = arith.constant 4 : i32
      %c5i = arith.constant 5 : i32
      %c7i = arith.constant 7 : i32
      %c8i = arith.constant 8 : i32
      %c16i = arith.constant 16 : i32
      %c24i = arith.constant 24 : i32
      %c31i = arith.constant 31 : i32
      %c32i = arith.constant 32 : i32
      %c127f = arith.constant 127.0 : f32
      %c128i = arith.constant 128 : i32
      %c144l = arith.constant 144 : i64
      %c16l = arith.constant 16 : i64
      %c4l = arith.constant 4 : i64
      %c32l = arith.constant 32 : i64
      %c0 = arith.constant 0.0 : f64
      %c0f = arith.constant 0.0 : f32
      %c1f = arith.constant 1.0 : f32
      %signclr = arith.constant 2147483647 : i32
      %cminsign = arith.constant -2147483648 : i32
      %chalf = arith.constant 1056964608 : i32
      %lane = arith.andi %tid_i, %c31i : i32
      %warp = arith.shrui %tid_i, %c5i : i32
      %wid = arith.constant 32 : i32
      %token64 = arith.index_cast %token : index to i64
      %dim64 = arith.extsi %dim : i32 to i64
      %rowoff = arith.muli %token64, %dim64 : i64
      %arow = llvm.getelementptr inbounds %a[%rowoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %brow = llvm.getelementptr inbounds %b[%rowoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %rrow = llvm.getelementptr inbounds %res[%rowoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %c256i = arith.constant 256 : i32
      %s0 = scf.for %iv = %tid_i to %dim step %c256i iter_args(%acc = %c0) -> (f64) : i32 {
        %ap = llvm.getelementptr inbounds %arow[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %av = llvm.load %ap : !llvm.ptr -> f32
        %bp = llvm.getelementptr inbounds %brow[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %bv = llvm.load %bp : !llvm.ptr -> f32
        %v = arith.addf %av, %bv : f32
        %rp = llvm.getelementptr inbounds %rrow[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        llvm.store %v, %rp : f32, !llvm.ptr
        %sq = arith.mulf %v, %v : f32
        %sq64 = arith.extf %sq : f32 to f64
        %na = arith.addf %acc, %sq64 : f64
        scf.yield %na : f64
      }
      %b0 = arith.bitcast %s0 : f64 to i64
      %lo0 = arith.trunci %b0 : i64 to i32
      %h0 = arith.shrui %b0, %c32l : i64
      %hi0 = arith.trunci %h0 : i64 to i32
      %lo1, %vv1 = gpu.shuffle down %lo0, %c16i, %wid : i32
      %hi1, %vv2 = gpu.shuffle down %hi0, %c16i, %wid : i32
      %eq1 = arith.extui %lo1 : i32 to i64
      %eh1 = arith.extui %hi1 : i32 to i64
      %sh1 = arith.shli %eh1, %c32l : i64
      %bt1 = arith.ori %eq1, %sh1 : i64
      %pa1 = arith.bitcast %bt1 : i64 to f64
      %s1 = arith.addf %s0, %pa1 : f64
      %b1 = arith.bitcast %s1 : f64 to i64
      %lo1b = arith.trunci %b1 : i64 to i32
      %h1 = arith.shrui %b1, %c32l : i64
      %hi1b = arith.trunci %h1 : i64 to i32
      %lo2, %vv3 = gpu.shuffle down %lo1b, %c8i, %wid : i32
      %hi2, %vv4 = gpu.shuffle down %hi1b, %c8i, %wid : i32
      %eq2 = arith.extui %lo2 : i32 to i64
      %eh2 = arith.extui %hi2 : i32 to i64
      %sh2 = arith.shli %eh2, %c32l : i64
      %bt2 = arith.ori %eq2, %sh2 : i64
      %pa2 = arith.bitcast %bt2 : i64 to f64
      %s2 = arith.addf %s1, %pa2 : f64
      %b2 = arith.bitcast %s2 : f64 to i64
      %lo2b = arith.trunci %b2 : i64 to i32
      %h2 = arith.shrui %b2, %c32l : i64
      %hi2b = arith.trunci %h2 : i64 to i32
      %lo3, %vv5 = gpu.shuffle down %lo2b, %c4i, %wid : i32
      %hi3, %vv6 = gpu.shuffle down %hi2b, %c4i, %wid : i32
      %eq3 = arith.extui %lo3 : i32 to i64
      %eh3 = arith.extui %hi3 : i32 to i64
      %sh3 = arith.shli %eh3, %c32l : i64
      %bt3 = arith.ori %eq3, %sh3 : i64
      %pa3 = arith.bitcast %bt3 : i64 to f64
      %s3 = arith.addf %s2, %pa3 : f64
      %b3 = arith.bitcast %s3 : f64 to i64
      %lo3b = arith.trunci %b3 : i64 to i32
      %h3 = arith.shrui %b3, %c32l : i64
      %hi3b = arith.trunci %h3 : i64 to i32
      %lo4, %vv7 = gpu.shuffle down %lo3b, %c2i, %wid : i32
      %hi4, %vv8 = gpu.shuffle down %hi3b, %c2i, %wid : i32
      %eq4 = arith.extui %lo4 : i32 to i64
      %eh4 = arith.extui %hi4 : i32 to i64
      %sh4 = arith.shli %eh4, %c32l : i64
      %bt4 = arith.ori %eq4, %sh4 : i64
      %f4 = arith.bitcast %bt4 : i64 to f64
      %s4 = arith.addf %s3, %f4 : f64
      %b4 = arith.bitcast %s4 : f64 to i64
      %lo4b = arith.trunci %b4 : i64 to i32
      %h4 = arith.shrui %b4, %c32l : i64
      %hi4b = arith.trunci %h4 : i64 to i32
      %lo5, %vv9 = gpu.shuffle down %lo4b, %c1i, %wid : i32
      %hi5, %vv10 = gpu.shuffle down %hi4b, %c1i, %wid : i32
      %eq5 = arith.extui %lo5 : i32 to i64
      %eh5 = arith.extui %hi5 : i32 to i64
      %sh5 = arith.shli %eh5, %c32l : i64
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
      %c256idx = arith.constant 256 : index
      %invm = memref.view %smem[%c256idx][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
      %in_w0 = arith.cmpi slt, %tid_i, %c32i : i32
      scf.if %in_w0 {
        %tl = memref.load %wpart[%lane_idx] : memref<32xf64, #gpu.address_space<workgroup>>
        %is_active = arith.cmpi slt, %lane, %c8i : i32
        %t0 = arith.select %is_active, %tl, %c0 : f64
        %tb0 = arith.bitcast %t0 : f64 to i64
        %tlo0 = arith.trunci %tb0 : i64 to i32
        %th0 = arith.shrui %tb0, %c32l : i64
        %thi0 = arith.trunci %th0 : i64 to i32
        %tlo1, %tv1 = gpu.shuffle down %tlo0, %c16i, %wid : i32
        %thi1, %tv2 = gpu.shuffle down %thi0, %c16i, %wid : i32
        %te1 = arith.extui %tlo1 : i32 to i64
        %the1 = arith.extui %thi1 : i32 to i64
        %tsh1 = arith.shli %the1, %c32l : i64
        %tbt1 = arith.ori %te1, %tsh1 : i64
        %tf1 = arith.bitcast %tbt1 : i64 to f64
        %t1 = arith.addf %t0, %tf1 : f64
        %tb1 = arith.bitcast %t1 : f64 to i64
        %tlo1b = arith.trunci %tb1 : i64 to i32
        %th1 = arith.shrui %tb1, %c32l : i64
        %thi1b = arith.trunci %th1 : i64 to i32
        %tlo2, %tv3 = gpu.shuffle down %tlo1b, %c8i, %wid : i32
        %thi2, %tv4 = gpu.shuffle down %thi1b, %c8i, %wid : i32
        %te2 = arith.extui %tlo2 : i32 to i64
        %the2 = arith.extui %thi2 : i32 to i64
        %tsh2 = arith.shli %the2, %c32l : i64
        %tbt2 = arith.ori %te2, %tsh2 : i64
        %tf2 = arith.bitcast %tbt2 : i64 to f64
        %t2 = arith.addf %t1, %tf2 : f64
        %tb2 = arith.bitcast %t2 : f64 to i64
        %tlo2b = arith.trunci %tb2 : i64 to i32
        %th2 = arith.shrui %tb2, %c32l : i64
        %thi2b = arith.trunci %th2 : i64 to i32
        %tlo3, %tv5 = gpu.shuffle down %tlo2b, %c4i, %wid : i32
        %thi3, %tv6 = gpu.shuffle down %thi2b, %c4i, %wid : i32
        %te3 = arith.extui %tlo3 : i32 to i64
        %the3 = arith.extui %thi3 : i32 to i64
        %tsh3 = arith.shli %the3, %c32l : i64
        %tbt3 = arith.ori %te3, %tsh3 : i64
        %tf3 = arith.bitcast %tbt3 : i64 to f64
        %t3 = arith.addf %t2, %tf3 : f64
        %tb3 = arith.bitcast %t3 : f64 to i64
        %tlo3b = arith.trunci %tb3 : i64 to i32
        %th3 = arith.shrui %tb3, %c32l : i64
        %thi3b = arith.trunci %th3 : i64 to i32
        %tlo4, %tv7 = gpu.shuffle down %tlo3b, %c2i, %wid : i32
        %thi4, %tv8 = gpu.shuffle down %thi3b, %c2i, %wid : i32
        %te4 = arith.extui %tlo4 : i32 to i64
        %the4 = arith.extui %thi4 : i32 to i64
        %tsh4 = arith.shli %the4, %c32l : i64
        %tbt4 = arith.ori %te4, %tsh4 : i64
        %tf4 = arith.bitcast %tbt4 : i64 to f64
        %t4 = arith.addf %t3, %tf4 : f64
        %tb4 = arith.bitcast %t4 : f64 to i64
        %tlo4b = arith.trunci %tb4 : i64 to i32
        %th4 = arith.shrui %tb4, %c32l : i64
        %thi4b = arith.trunci %th4 : i64 to i32
        %tlo5, %tv9 = gpu.shuffle down %tlo4b, %c1i, %wid : i32
        %thi5, %tv10 = gpu.shuffle down %thi4b, %c1i, %wid : i32
        %te5 = arith.extui %tlo5 : i32 to i64
        %the5 = arith.extui %thi5 : i32 to i64
        %tsh5 = arith.shli %the5, %c32l : i64
        %tbt5 = arith.ori %te5, %tsh5 : i64
        %tf5 = arith.bitcast %tbt5 : i64 to f64
        %t5 = arith.addf %t4, %tf5 : f64
        %fn = arith.sitofp %dim : i32 to f64
        %tn = arith.divf %t5, %fn : f64
        %t32 = arith.truncf %tn : f64 to f32
        %te = arith.addf %t32, %eps : f32
        %sq = nvvm.sqrt %te {rnd = #nvvm.fp_rnd_mode<rn>} : f32
        %inv = arith.divf %c1f, %sq : f32
        %is_l0 = arith.cmpi eq, %lane, %c0i : i32
        scf.if %is_l0 {
          %i0i = arith.constant 0 : index
          memref.store %inv, %invm[%i0i] : memref<4xf32, #gpu.address_space<workgroup>>
        }
      }
      gpu.barrier
      %i0b = arith.constant 0 : index
      %ivv = memref.load %invm[%i0b] : memref<4xf32, #gpu.address_space<workgroup>>
      %nblk = arith.divsi %dim, %c128i : i32
      %n64 = arith.extsi %n : i32 to i64
      scf.for %blk = %warp to %nblk step %c8i : i32 {
        %b128 = arith.muli %blk, %c128i : i32
        %l4 = arith.muli %lane, %c4i : i32
        %base = arith.addi %b128, %l4 : i32
        %i1 = arith.addi %base, %c1i : i32
        %i2 = arith.addi %base, %c2i : i32
        %i3 = arith.addi %base, %c3i : i32
        %xb0p = llvm.getelementptr inbounds %rrow[%base] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xb0 = llvm.load %xb0p : !llvm.ptr -> f32
        %xb1p = llvm.getelementptr inbounds %rrow[%i1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xb1 = llvm.load %xb1p : !llvm.ptr -> f32
        %xb2p = llvm.getelementptr inbounds %rrow[%i2] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xb2 = llvm.load %xb2p : !llvm.ptr -> f32
        %xb3p = llvm.getelementptr inbounds %rrow[%i3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xb3 = llvm.load %xb3p : !llvm.ptr -> f32
        %w0p = llvm.getelementptr inbounds %w[%base] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %w0 = llvm.load %w0p : !llvm.ptr -> f32
        %w1p = llvm.getelementptr inbounds %w[%i1] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %w1 = llvm.load %w1p : !llvm.ptr -> f32
        %w2p = llvm.getelementptr inbounds %w[%i2] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %w2 = llvm.load %w2p : !llvm.ptr -> f32
        %w3p = llvm.getelementptr inbounds %w[%i3] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %w3 = llvm.load %w3p : !llvm.ptr -> f32
        %p0 = arith.mulf %xb0, %ivv : f32
        %v0 = arith.mulf %p0, %w0 : f32
        %p1 = arith.mulf %xb1, %ivv : f32
        %v1 = arith.mulf %p1, %w1 : f32
        %p2 = arith.mulf %xb2, %ivv : f32
        %v2 = arith.mulf %p2, %w2 : f32
        %p3 = arith.mulf %xb3, %ivv : f32
        %v3 = arith.mulf %p3, %w3 : f32
        %v0b = arith.bitcast %v0 : f32 to i32
        %a0b = arith.andi %v0b, %signclr : i32
        %ab0 = arith.bitcast %a0b : i32 to f32
        %v1b = arith.bitcast %v1 : f32 to i32
        %a1b = arith.andi %v1b, %signclr : i32
        %ab1 = arith.bitcast %a1b : i32 to f32
        %v2b = arith.bitcast %v2 : f32 to i32
        %a2b = arith.andi %v2b, %signclr : i32
        %ab2 = arith.bitcast %a2b : i32 to f32
        %v3b = arith.bitcast %v3 : f32 to i32
        %a3b = arith.andi %v3b, %signclr : i32
        %ab3 = arith.bitcast %a3b : i32 to f32
        %am01 = arith.maximumf %ab0, %ab1 : f32
        %am012 = arith.maximumf %am01, %ab2 : f32
        %am0123 = arith.maximumf %am012, %ab3 : f32
        %am4, %av1 = gpu.shuffle xor %am0123, %c4i, %wid : f32
        %am4m = arith.maximumf %am0123, %am4 : f32
        %am8, %av2 = gpu.shuffle xor %am4m, %c2i, %wid : f32
        %am8m = arith.maximumf %am4m, %am8 : f32
        %am16, %av3 = gpu.shuffle xor %am8m, %c1i, %wid : f32
        %amaxf = arith.maximumf %am8m, %am16 : f32
        %su0 = arith.addf %v0, %v1 : f32
        %su1 = arith.addf %su0, %v2 : f32
        %su2 = arith.addf %su1, %v3 : f32
        %su4, %sv1 = gpu.shuffle xor %su2, %c4i, %wid : f32
        %su4a = arith.addf %su2, %su4 : f32
        %su8, %sv2 = gpu.shuffle xor %su4a, %c2i, %wid : f32
        %su8a = arith.addf %su4a, %su8 : f32
        %su16, %sv3 = gpu.shuffle xor %su8a, %c1i, %wid : f32
        %sumf = arith.addf %su8a, %su16 : f32
        %az = arith.cmpf oeq, %amaxf, %c0f : f32
        %d1 = arith.divf %amaxf, %c127f : f32
        %safed1 = arith.select %az, %c1f, %d1 : f32
        %dinv0 = arith.divf %c127f, %amaxf : f32
        %safedinv = arith.select %az, %c1f, %dinv0 : f32
        %d2 = arith.divf %c1f, %safedinv : f32
        %r10 = arith.divf %v0, %safed1 : f32
        %r11 = arith.divf %v1, %safed1 : f32
        %r12 = arith.divf %v2, %safed1 : f32
        %r13 = arith.divf %v3, %safed1 : f32
        %r20 = arith.mulf %v0, %safedinv : f32
        %r21 = arith.mulf %v1, %safedinv : f32
        %r22 = arith.mulf %v2, %safedinv : f32
        %r23 = arith.mulf %v3, %safedinv : f32
        %r10b = arith.bitcast %r10 : f32 to i32
        %r10s = arith.andi %r10b, %cminsign : i32
        %r10h = arith.ori %r10s, %chalf : i32
        %r10f = arith.bitcast %r10h : i32 to f32
        %r10r = nvvm.addf %r10, %r10f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q10i = arith.fptosi %r10r : f32 to i32
        %q10 = arith.trunci %q10i : i32 to i8
        %r11b = arith.bitcast %r11 : f32 to i32
        %r11s = arith.andi %r11b, %cminsign : i32
        %r11h = arith.ori %r11s, %chalf : i32
        %r11f = arith.bitcast %r11h : i32 to f32
        %r11r = nvvm.addf %r11, %r11f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q11i = arith.fptosi %r11r : f32 to i32
        %q11 = arith.trunci %q11i : i32 to i8
        %r12b = arith.bitcast %r12 : f32 to i32
        %r12s = arith.andi %r12b, %cminsign : i32
        %r12h = arith.ori %r12s, %chalf : i32
        %r12f = arith.bitcast %r12h : i32 to f32
        %r12r = nvvm.addf %r12, %r12f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q12i = arith.fptosi %r12r : f32 to i32
        %q12 = arith.trunci %q12i : i32 to i8
        %r13b = arith.bitcast %r13 : f32 to i32
        %r13s = arith.andi %r13b, %cminsign : i32
        %r13h = arith.ori %r13s, %chalf : i32
        %r13f = arith.bitcast %r13h : i32 to f32
        %r13r = nvvm.addf %r13, %r13f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q13i = arith.fptosi %r13r : f32 to i32
        %q13 = arith.trunci %q13i : i32 to i8
        %r20b = arith.bitcast %r20 : f32 to i32
        %r20s = arith.andi %r20b, %cminsign : i32
        %r20h = arith.ori %r20s, %chalf : i32
        %r20f = arith.bitcast %r20h : i32 to f32
        %r20r = nvvm.addf %r20, %r20f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q20i = arith.fptosi %r20r : f32 to i32
        %q20 = arith.trunci %q20i : i32 to i8
        %r21b = arith.bitcast %r21 : f32 to i32
        %r21s = arith.andi %r21b, %cminsign : i32
        %r21h = arith.ori %r21s, %chalf : i32
        %r21f = arith.bitcast %r21h : i32 to f32
        %r21r = nvvm.addf %r21, %r21f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q21i = arith.fptosi %r21r : f32 to i32
        %q21 = arith.trunci %q21i : i32 to i8
        %r22b = arith.bitcast %r22 : f32 to i32
        %r22s = arith.andi %r22b, %cminsign : i32
        %r22h = arith.ori %r22s, %chalf : i32
        %r22f = arith.bitcast %r22h : i32 to f32
        %r22r = nvvm.addf %r22, %r22f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q22i = arith.fptosi %r22r : f32 to i32
        %q22 = arith.trunci %q22i : i32 to i8
        %r23b = arith.bitcast %r23 : f32 to i32
        %r23s = arith.andi %r23b, %cminsign : i32
        %r23h = arith.ori %r23s, %chalf : i32
        %r23f = arith.bitcast %r23h : i32 to f32
        %r23r = nvvm.addf %r23, %r23f {rnd = #nvvm.fp_rnd_mode<rz>} : f32
        %q23i = arith.fptosi %r23r : f32 to i32
        %q23 = arith.trunci %q23i : i32 to i8
        %is_d4q8 = arith.cmpi eq, %ds, %c2i : i32
        %dsel = arith.select %is_d4q8, %d1, %d2 : f32
        %q0 = arith.select %is_d4q8, %q10, %q20 : i8
        %q1 = arith.select %is_d4q8, %q11, %q21 : i8
        %q2 = arith.select %is_d4q8, %q12, %q22 : i8
        %q3 = arith.select %is_d4q8, %q13, %q23 : i8
        %blk64 = arith.extsi %blk : i32 to i64
        %bn = arith.muli %blk64, %n64 : i64
        %bidx = arith.addi %bn, %token64 : i64
        %boff = arith.muli %bidx, %c144l : i64
        %sub = arith.shrui %lane, %c3i : i32
        %sub64 = arith.extui %sub : i32 to i64
        %soff0 = arith.muli %sub64, %c4l : i64
        %soff = arith.addi %boff, %soff0 : i64
        %sp = llvm.getelementptr inbounds %y[%soff] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %is_owner = arith.andi %lane, %c7i : i32
        %is_owner0 = arith.cmpi eq, %is_owner, %c0i : i32
        %is_ds4 = arith.cmpi eq, %ds, %c1i : i32
        scf.if %is_owner0 {
          scf.if %is_ds4 {
            %dh = arith.truncf %dsel : f32 to f16
            %sh = arith.truncf %sumf : f32 to f16
            %d16 = arith.bitcast %dh : f16 to i16
            %s16 = arith.bitcast %sh : f16 to i16
            %de = arith.extui %d16 : i16 to i32
            %se = arith.extui %s16 : i16 to i32
            %se16 = arith.shli %se, %c16i : i32
            %pk = arith.ori %de, %se16 : i32
            llvm.store %pk, %sp : i32, !llvm.ptr
          } else {
            llvm.store %dsel, %sp : f32, !llvm.ptr
          }
        }
        %q0e = arith.extui %q0 : i8 to i32
        %q1e = arith.extui %q1 : i8 to i32
        %q2e = arith.extui %q2 : i8 to i32
        %q3e = arith.extui %q3 : i8 to i32
        %q1s = arith.shli %q1e, %c8i : i32
        %q2s = arith.shli %q2e, %c16i : i32
        %q3s = arith.shli %q3e, %c24i : i32
        %or01 = arith.ori %q0e, %q1s : i32
        %or012 = arith.ori %or01, %q2s : i32
        %or0123 = arith.ori %or012, %q3s : i32
        %l4b = arith.muli %lane, %c4i : i32
        %l4_64 = arith.extui %l4b : i32 to i64
        %qo0 = arith.addi %boff, %c16l : i64
        %qo1 = arith.addi %qo0, %l4_64 : i64
        %qp = llvm.getelementptr inbounds %y[%qo1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        llvm.store %or0123, %qp : i32, !llvm.ptr
      }
      gpu.return
    }
  }
})MLIR";

// Prefill RMS-norm + residual add (Stage 5c, rms_norm_add). One 256-thread block
// per [dim] row. The reduction is rms_heads_fused_kernel's exactly: a strided
// f64 accumulation, five shuffle-down steps, the 32-entry warp-partial array,
// and the warp-0 reduction with lanes >= nwarps zero-filled, so `inv` is the
// same fp32 value the norm kernel computes. The apply then does that kernel's
// `x * iv * w` followed by add_kernel's `residual + v`, both at the same fp32
// rounding points, so the region's output is byte-identical to the pair.
constexpr const char *kRmsNormAdd = R"MLIR(module attributes {gpu.container_module} {
  gpu.module @glue [#nvvm.target<chip = "%CHIP%">] {
    gpu.func @rms_norm_add(%x: !llvm.ptr, %r: !llvm.ptr, %w: !llvm.ptr, %y: !llvm.ptr, %dim: i32, %eps: f32) kernel {
      %token = gpu.block_id x
      %tid = gpu.thread_id x
      %tid_i = arith.index_cast %tid : index to i32
      %c0i = arith.constant 0 : i32
      %c1i = arith.constant 1 : i32
      %c2i = arith.constant 2 : i32
      %c4i = arith.constant 4 : i32
      %c5i = arith.constant 5 : i32
      %c8i = arith.constant 8 : i32
      %c16i = arith.constant 16 : i32
      %c31i = arith.constant 31 : i32
      %c32i = arith.constant 32 : i32
      %c256i = arith.constant 256 : i32
      %c32l = arith.constant 32 : i64
      %c0 = arith.constant 0.0 : f64
      %c1f = arith.constant 1.0 : f32
      %lane = arith.andi %tid_i, %c31i : i32
      %warp = arith.shrui %tid_i, %c5i : i32
      %wid = arith.constant 32 : i32
      %token64 = arith.index_cast %token : index to i64
      %dim64 = arith.extsi %dim : i32 to i64
      %rowoff = arith.muli %token64, %dim64 : i64
      %xrow = llvm.getelementptr inbounds %x[%rowoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %rrow = llvm.getelementptr inbounds %r[%rowoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %yrow = llvm.getelementptr inbounds %y[%rowoff] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %s0 = scf.for %iv = %tid_i to %dim step %c256i iter_args(%acc = %c0) -> (f64) : i32 {
        %xp = llvm.getelementptr inbounds %xrow[%iv] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xv = llvm.load %xp : !llvm.ptr -> f32
        %sq = arith.mulf %xv, %xv : f32
        %sq64 = arith.extf %sq : f32 to f64
        %na = arith.addf %acc, %sq64 : f64
        scf.yield %na : f64
      }
      %b0 = arith.bitcast %s0 : f64 to i64
      %lo0 = arith.trunci %b0 : i64 to i32
      %h0 = arith.shrui %b0, %c32l : i64
      %hi0 = arith.trunci %h0 : i64 to i32
      %lo1, %vv1 = gpu.shuffle down %lo0, %c16i, %wid : i32
      %hi1, %vv2 = gpu.shuffle down %hi0, %c16i, %wid : i32
      %eq1 = arith.extui %lo1 : i32 to i64
      %eh1 = arith.extui %hi1 : i32 to i64
      %sh1 = arith.shli %eh1, %c32l : i64
      %bt1 = arith.ori %eq1, %sh1 : i64
      %pa1 = arith.bitcast %bt1 : i64 to f64
      %s1 = arith.addf %s0, %pa1 : f64
      %b1 = arith.bitcast %s1 : f64 to i64
      %lo1b = arith.trunci %b1 : i64 to i32
      %h1 = arith.shrui %b1, %c32l : i64
      %hi1b = arith.trunci %h1 : i64 to i32
      %lo2, %vv3 = gpu.shuffle down %lo1b, %c8i, %wid : i32
      %hi2, %vv4 = gpu.shuffle down %hi1b, %c8i, %wid : i32
      %eq2 = arith.extui %lo2 : i32 to i64
      %eh2 = arith.extui %hi2 : i32 to i64
      %sh2 = arith.shli %eh2, %c32l : i64
      %bt2 = arith.ori %eq2, %sh2 : i64
      %pa2 = arith.bitcast %bt2 : i64 to f64
      %s2 = arith.addf %s1, %pa2 : f64
      %b2 = arith.bitcast %s2 : f64 to i64
      %lo2b = arith.trunci %b2 : i64 to i32
      %h2 = arith.shrui %b2, %c32l : i64
      %hi2b = arith.trunci %h2 : i64 to i32
      %lo3, %vv5 = gpu.shuffle down %lo2b, %c4i, %wid : i32
      %hi3, %vv6 = gpu.shuffle down %hi2b, %c4i, %wid : i32
      %eq3 = arith.extui %lo3 : i32 to i64
      %eh3 = arith.extui %hi3 : i32 to i64
      %sh3 = arith.shli %eh3, %c32l : i64
      %bt3 = arith.ori %eq3, %sh3 : i64
      %pa3 = arith.bitcast %bt3 : i64 to f64
      %s3 = arith.addf %s2, %pa3 : f64
      %b3 = arith.bitcast %s3 : f64 to i64
      %lo3b = arith.trunci %b3 : i64 to i32
      %h3 = arith.shrui %b3, %c32l : i64
      %hi3b = arith.trunci %h3 : i64 to i32
      %lo4, %vv7 = gpu.shuffle down %lo3b, %c2i, %wid : i32
      %hi4, %vv8 = gpu.shuffle down %hi3b, %c2i, %wid : i32
      %eq4 = arith.extui %lo4 : i32 to i64
      %eh4 = arith.extui %hi4 : i32 to i64
      %sh4 = arith.shli %eh4, %c32l : i64
      %bt4 = arith.ori %eq4, %sh4 : i64
      %f4 = arith.bitcast %bt4 : i64 to f64
      %s4 = arith.addf %s3, %f4 : f64
      %b4 = arith.bitcast %s4 : f64 to i64
      %lo4b = arith.trunci %b4 : i64 to i32
      %h4 = arith.shrui %b4, %c32l : i64
      %hi4b = arith.trunci %h4 : i64 to i32
      %lo5, %vv9 = gpu.shuffle down %lo4b, %c1i, %wid : i32
      %hi5, %vv10 = gpu.shuffle down %hi4b, %c1i, %wid : i32
      %eq5 = arith.extui %lo5 : i32 to i64
      %eh5 = arith.extui %hi5 : i32 to i64
      %sh5 = arith.shli %eh5, %c32l : i64
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
      %c256idx = arith.constant 256 : index
      %invm = memref.view %smem[%c256idx][] : memref<?xi8, #gpu.address_space<workgroup>> to memref<4xf32, #gpu.address_space<workgroup>>
      %in_w0 = arith.cmpi slt, %tid_i, %c32i : i32
      scf.if %in_w0 {
        %tl = memref.load %wpart[%lane_idx] : memref<32xf64, #gpu.address_space<workgroup>>
        %is_active = arith.cmpi slt, %lane, %c8i : i32
        %t0 = arith.select %is_active, %tl, %c0 : f64
        %tb0 = arith.bitcast %t0 : f64 to i64
        %tlo0 = arith.trunci %tb0 : i64 to i32
        %th0 = arith.shrui %tb0, %c32l : i64
        %thi0 = arith.trunci %th0 : i64 to i32
        %tlo1, %tv1 = gpu.shuffle down %tlo0, %c16i, %wid : i32
        %thi1, %tv2 = gpu.shuffle down %thi0, %c16i, %wid : i32
        %te1 = arith.extui %tlo1 : i32 to i64
        %the1 = arith.extui %thi1 : i32 to i64
        %tsh1 = arith.shli %the1, %c32l : i64
        %tbt1 = arith.ori %te1, %tsh1 : i64
        %tf1 = arith.bitcast %tbt1 : i64 to f64
        %t1 = arith.addf %t0, %tf1 : f64
        %tb1 = arith.bitcast %t1 : f64 to i64
        %tlo1b = arith.trunci %tb1 : i64 to i32
        %th1 = arith.shrui %tb1, %c32l : i64
        %thi1b = arith.trunci %th1 : i64 to i32
        %tlo2, %tv3 = gpu.shuffle down %tlo1b, %c8i, %wid : i32
        %thi2, %tv4 = gpu.shuffle down %thi1b, %c8i, %wid : i32
        %te2 = arith.extui %tlo2 : i32 to i64
        %the2 = arith.extui %thi2 : i32 to i64
        %tsh2 = arith.shli %the2, %c32l : i64
        %tbt2 = arith.ori %te2, %tsh2 : i64
        %tf2 = arith.bitcast %tbt2 : i64 to f64
        %t2 = arith.addf %t1, %tf2 : f64
        %tb2 = arith.bitcast %t2 : f64 to i64
        %tlo2b = arith.trunci %tb2 : i64 to i32
        %th2 = arith.shrui %tb2, %c32l : i64
        %thi2b = arith.trunci %th2 : i64 to i32
        %tlo3, %tv5 = gpu.shuffle down %tlo2b, %c4i, %wid : i32
        %thi3, %tv6 = gpu.shuffle down %thi2b, %c4i, %wid : i32
        %te3 = arith.extui %tlo3 : i32 to i64
        %the3 = arith.extui %thi3 : i32 to i64
        %tsh3 = arith.shli %the3, %c32l : i64
        %tbt3 = arith.ori %te3, %tsh3 : i64
        %tf3 = arith.bitcast %tbt3 : i64 to f64
        %t3 = arith.addf %t2, %tf3 : f64
        %tb3 = arith.bitcast %t3 : f64 to i64
        %tlo3b = arith.trunci %tb3 : i64 to i32
        %th3 = arith.shrui %tb3, %c32l : i64
        %thi3b = arith.trunci %th3 : i64 to i32
        %tlo4, %tv7 = gpu.shuffle down %tlo3b, %c2i, %wid : i32
        %thi4, %tv8 = gpu.shuffle down %thi3b, %c2i, %wid : i32
        %te4 = arith.extui %tlo4 : i32 to i64
        %the4 = arith.extui %thi4 : i32 to i64
        %tsh4 = arith.shli %the4, %c32l : i64
        %tbt4 = arith.ori %te4, %tsh4 : i64
        %tf4 = arith.bitcast %tbt4 : i64 to f64
        %t4 = arith.addf %t3, %tf4 : f64
        %tb4 = arith.bitcast %t4 : f64 to i64
        %tlo4b = arith.trunci %tb4 : i64 to i32
        %th4 = arith.shrui %tb4, %c32l : i64
        %thi4b = arith.trunci %th4 : i64 to i32
        %tlo5, %tv9 = gpu.shuffle down %tlo4b, %c1i, %wid : i32
        %thi5, %tv10 = gpu.shuffle down %thi4b, %c1i, %wid : i32
        %te5 = arith.extui %tlo5 : i32 to i64
        %the5 = arith.extui %thi5 : i32 to i64
        %tsh5 = arith.shli %the5, %c32l : i64
        %tbt5 = arith.ori %te5, %tsh5 : i64
        %tf5 = arith.bitcast %tbt5 : i64 to f64
        %t5 = arith.addf %t4, %tf5 : f64
        %fn = arith.sitofp %dim : i32 to f64
        %tn = arith.divf %t5, %fn : f64
        %t32 = arith.truncf %tn : f64 to f32
        %te = arith.addf %t32, %eps : f32
        %sq = nvvm.sqrt %te {rnd = #nvvm.fp_rnd_mode<rn>} : f32
        %inv = arith.divf %c1f, %sq : f32
        %is_l0 = arith.cmpi eq, %lane, %c0i : i32
        scf.if %is_l0 {
          %i0i = arith.constant 0 : index
          memref.store %inv, %invm[%i0i] : memref<4xf32, #gpu.address_space<workgroup>>
        }
      }
      gpu.barrier
      %i0b = arith.constant 0 : index
      %ivv = memref.load %invm[%i0b] : memref<4xf32, #gpu.address_space<workgroup>>
      %dim_idx = arith.index_cast %dim : i32 to index
      %c256idx2 = arith.constant 256 : index
      scf.for %k = %tid to %dim_idx step %c256idx2 {
        %ki = arith.index_cast %k : index to i32
        %xp2 = llvm.getelementptr inbounds %xrow[%ki] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %xv2 = llvm.load %xp2 : !llvm.ptr -> f32
        %wp = llvm.getelementptr inbounds %w[%ki] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %wv = llvm.load %wp : !llvm.ptr -> f32
        %rp = llvm.getelementptr inbounds %rrow[%ki] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        %rv = llvm.load %rp : !llvm.ptr -> f32
        %p = arith.mulf %xv2, %ivv : f32
        %v = arith.mulf %p, %wv : f32
        %o = arith.addf %rv, %v : f32
        %yp = llvm.getelementptr inbounds %yrow[%ki] : (!llvm.ptr, i32) -> !llvm.ptr, f32
        llvm.store %o, %yp : f32, !llvm.ptr
      }
      gpu.return
    }
  }
})MLIR";

// The region text for `kind`, with the target chip substituted. Empty when the
// kind has no generator.
std::string region_text(GlueRegionKind kind, int sm_arch) {
  const char *text = nullptr;
  switch (kind) {
    case GlueRegionKind::RmsNormQuantizeQ8_1:
      text = kRmsNormQuantizeQ8_1;
      break;
    case GlueRegionKind::AddRmsNormQuantizeMmqQ8_1:
      text = kAddRmsNormQuantizeMmqQ8_1;
      break;
    case GlueRegionKind::RmsNormAdd:
      text = kRmsNormAdd;
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
  if (name == "add_rms_norm_quantize_mmq_q8_1")
    return GlueRegionKind::AddRmsNormQuantizeMmqQ8_1;
  if (name == "rms_norm_add")
    return GlueRegionKind::RmsNormAdd;
  return std::nullopt;
}

const char *glue_region_name(GlueRegionKind kind) noexcept {
  switch (kind) {
    case GlueRegionKind::RmsNormQuantizeQ8_1:
      return "rms_norm_quantize_q8_1";
    case GlueRegionKind::AddRmsNormQuantizeMmqQ8_1:
      return "add_rms_norm_quantize_mmq_q8_1";
    case GlueRegionKind::RmsNormAdd:
      return "rms_norm_add";
  }
  return "?";
}

const char *glue_kernel_entry(GlueRegionKind kind) noexcept {
  switch (kind) {
    case GlueRegionKind::RmsNormQuantizeQ8_1:
      return "rms_norm_q8_1";
    case GlueRegionKind::AddRmsNormQuantizeMmqQ8_1:
      return "add_rms_norm_q8_1_mmq";
    case GlueRegionKind::RmsNormAdd:
      return "rms_norm_add";
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
