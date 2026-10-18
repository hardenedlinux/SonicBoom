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

// The generated dense bf16 GEMM (v1 Stage 6). Emitted in the GPU dialect and
// lowered through the same NVVM pipeline as the glue regions and the generated
// attention kernels:
//
//   gpu.func + nvgpu.mma.sync / nvgpu.ldmatrix / nvgpu.device_async_copy
//       -> nvvm -> PTX
//
// Operands and result (the strided dense projection's layout):
//   X  [N][K] bf16, token-major (K contiguous)
//   W  [M][K] bf16, row-major  (K contiguous)
//   Y  [N][M] f32,  token-major (M contiguous), y[token * M + row]
// so one launch computes C^T = W * X^T exactly as the cuBLASLt path's
// C (rows x n) column-major view does.
//
// Tiling: one block per (BM output rows x BN tokens). The K loop stages BK
// elements of both operands with cp.async double buffering, runs
// m16n8k16 bf16 -> f32 tensor-core mma over (BM/16) warps, and writes f32.
//
// Batch invariance: a block owns a fixed (row, token) tile, the K loop runs in
// ascending BK steps, and each accumulator sees the same sequence of mma
// instructions regardless of N or of the block's position in N: no column
// chunking, no split-K, no shape-dependent reduction order. A token's row is
// therefore bit-identical whether it is computed alone (N = 1) or as part of a
// full prefill (N = 512).
//
// Layer placement: compiler machinery behind Layer 2, like GlueCodegen. The
// runtime JIT half lives in planner/glue_jit (launch_dense_gemm).

#include <optional>
#include <string>

namespace sonicboom::compiler {

// One generated GEMM shape class. Candidates come from the autotune service
// (planner/autotune); the kernel text depends on nothing else.
struct DenseGemmShape {
  int bm = 64;  // output rows per block  (multiple of 16)
  int bn = 64;  // tokens per block       (multiple of 8)
  int bk = 64;  // K elements per stage   (multiple of 16)

  bool operator==(const DenseGemmShape &o) const {
    return bm == o.bm && bn == o.bn && bk == o.bk;
  }
};

// True when the generator covers the shape: bm/16 warps of 32 threads (at most
// 8 warps, so at most 256 threads), bn/8 f32 accumulator vectors per lane (at
// most 16), and the double-buffered staging (2 * (bm + bn) * bk * 2 bytes)
// within the 96 KiB dynamic shared-memory limit.
bool dense_gemm_shape_supported(const DenseGemmShape &shape);

// Launch geometry: (bm/16)*32 threads per block; grid (ceil(N/bn),
// ceil(M/bm)).
int dense_gemm_block_threads(const DenseGemmShape &shape);
int dense_gemm_shared_bytes(const DenseGemmShape &shape);

// The generated kernel's PTX entry symbol (gpu.func name).
constexpr const char *kDenseGemmKernelEntry = "gemm";

// Emit and lower the kernel for `shape` targeting compute capability `sm_arch`
// (86 = sm_86). Returns nullopt, having printed the MLIR diagnostic, when the
// shape has no generator or lowering fails.
std::optional<std::string> generate_dense_gemm_ptx(const DenseGemmShape &shape,
                                                   int sm_arch);

// The identity of the generator, stamped into the JIT disk cache entry (same
// contract as glue_codegen_version / attention_codegen_version).
std::string dense_gemm_codegen_version();

} // namespace sonicboom::compiler
