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

#pragma once

// Compiler-generated glue kernels (v1 scope: "an MLIR GPU lowering path
// (linalg -> gpu -> nvvm -> cubin), JIT-compiled and cached by shape
// signature, for fused glue kernels ... between the hot kernels").
//
// A glue *region* is a maximal run of elementwise / row-reduction ops between
// ukernel calls (FuseGlueRegionsPass, BoomPasses.cpp). This file turns a region
// into PTX through MLIR: it builds the region's gpu.module, converts it to the
// NVVM dialect and asks the GPU module-to-binary pass for assembly, which the
// shipped LLVM NVPTX backend produces. The runtime loads the PTX with the CUDA
// driver (planner/glue_jit).
//
// The numerics contract is the hand-written kernel the region replaces: the
// generated kernel reproduces its reduction order (warp-shuffle down tree, f64
// accumulation, block reduction through shared memory) and its quantize
// sequence (max tree, d = amax/127, roundf as add.rz(sign|0.5) + cvt.rzi), so
// the region's output is byte-identical to the reference kernel's.

#include <optional>
#include <string>
#include <string_view>

namespace sonicboom::compiler {

// The region shapes the compiler has a code generator for. The enumerator name
// is the value of the node's `fused_region` string attribute, so the planner and
// the code generator agree on one name.
enum class GlueRegionKind {
  // Decode RMS-norm + q8_1 quantize (rms_norm_quantize_q8_1). One CUDA block of
  // 1024 threads normalizes one [dim] row and writes dim/32 q8_1 blocks.
  RmsNormQuantizeQ8_1,
  // Prefill residual add + RMS-norm + mmq q8_1 quantize
  // (add_rms_norm_quantize_mmq_q8_1). One CUDA block of 256 threads per [dim]
  // row: it writes the residual sum and the normalized dim/128 mmq blocks.
  AddRmsNormQuantizeMmqQ8_1,
  // Prefill RMS-norm + residual add (rms_norm_add). One CUDA block of 256
  // threads per [dim] row; the row's norm is the private intermediate, so the
  // region's only result is `residual + x * inv * w`.
  RmsNormAdd,
  // Attention prep (heads_norm_rope_cast): per-head RMS-norm + optional NeoX
  // RoPE + optional fp16 rounding. One CUDA block of 256 threads per head; the
  // norm output is staged in shared memory because the rotation reads the
  // partner element (i, i + head_dim/2). The runtime flags `do_rope` and
  // `do_cast` cover the three chains the attention prep uses: Q norm->rope,
  // K norm->rope->cast, V norm->cast.
  HeadsNormRopeCast,
  // Strided row slice + scale (slice_scale): one thread per output element,
  //   y[r*length + k] = x[offset + r*stride + k] * s.
  // Extracts a layer's per-layer-embedding column from each position and folds
  // the following scale into the same read. Byte-identical to slice_strided
  // followed by scale (the multiply is the reference kernel's).
  SliceScale,
  // Gated activation (gelu_mul): gelu_fp16's fp16-round-trip tanh-GELU on the
  // gate, multiplied by the up projection, one thread per element. The tanh
  // goes through libdevice, so this region uses the same checked-on-tolerance
  // pathway as the attention prep's trig.
  GeluMul,
};

// The region kind named `name`, or nullopt if the compiler has no generator for
// it.
std::optional<GlueRegionKind> glue_region_kind_from_name(std::string_view name);

const char* glue_region_name(GlueRegionKind kind) noexcept;

// The generated kernel's PTX entry symbol (gpu.func name).
const char* glue_kernel_entry(GlueRegionKind kind) noexcept;

// The launch geometry of the generated kernel. It is the reference kernel's:
// one 1024-thread block per region row, 512 bytes of dynamic shared memory
// (32 f64 warp partials + one f32 inverse RMS).
constexpr int kGlueBlockThreads = 1024;
constexpr int kGlueDynamicSharedBytes = 512;

// add_rms_norm_quantize_mmq_q8_1: the reference pair's geometry — a 256-thread
// block per row (rms_norm_inv_kernel's block), 512 bytes of dynamic shared
// memory (32 f64 warp partials + the f32 inverse RMS). The generated kernel
// does both phases in one block so it can also write the residual sum.
constexpr int kAddRmsNormBlockThreads = 256;
constexpr int kAddRmsNormSharedBytes = 512;

// rms_norm_add: the same geometry — rms_heads_fused_kernel's 256-thread block
// and 512 bytes of shared memory (32 f64 warp partials + the f32 inverse RMS).
constexpr int kRmsNormAddBlockThreads = 256;
constexpr int kRmsNormAddSharedBytes = 512;
// Contextual: how many rows (norm vectors) one launch covers is the node's
// element count / dim, computed at dispatch.

// heads_norm_rope_cast: rms_heads_fused_kernel's 256-thread block. Shared
// memory: 32 f64 warp partials at 0, the f32 inverse RMS at 256, and the staged
// norm output at byte 512. The stage holds one head; the reduction and apply
// loops stride by the block size, so the limit is the stage's 512 floats.
// Gemma's global layers have head_dim 512, its local layers 256.
constexpr int kHeadsNormRopeCastBlockThreads = 256;
constexpr int kHeadsNormRopeCastSharedBytes = 2560;
constexpr int kHeadsNormRopeCastMaxHeadDim = 512;

// slice_scale: the reference kernels' geometry — one thread per output element,
// 256 threads per block, no shared memory.
constexpr int kSliceScaleBlockThreads = 256;

// gelu_mul: gelu_fp16_kernel/mul_kernel's geometry — one thread per element,
// 256 threads per block, no shared memory.
constexpr int kGeluMulBlockThreads = 256;

// Generate the region's PTX for compute capability `sm_arch` (86 = sm_86).
// Returns nullopt, having printed the MLIR diagnostic, when the region has no
// generator, the module fails to verify, or the downstream lowering fails.
std::optional<std::string> generate_glue_ptx(GlueRegionKind kind, int sm_arch);

// Lower a textual GPU-dialect module (a container module holding one
// gpu.module) to PTX assembly through the NVVM pipeline above. `link_libdevice`
// converts math ops through libdevice and links it, for kernels that use exp /
// trig. `what` names the source in diagnostics. The glue-region generator and
// the attention generator (AttentionCodegen.h) share this one pipeline.
std::optional<std::string> lower_gpu_module_to_ptx(std::string_view mlir_text,
                                                   bool link_libdevice,
                                                   std::string_view what);

// The identity of the compiler that produces that PTX: the linked LLVM version
// plus this generator's revision. The JIT disk cache stamps it into every entry
// and ignores an entry written by anything else — a PTX file cached by a
// different compiler must not be loaded and run just because the region name
// and arch in its file name match.
std::string glue_codegen_version();

} // namespace sonicboom::compiler
