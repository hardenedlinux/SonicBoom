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

// The generated q8_1 x q8_1 int8 tensor-core prefill matmul (Wave 4, MMQ1).
//
// Status: experimental and default off (boom-fix1 C5). The greedy stream is not
// bit-identical to the ggml mmq path it replaces (first divergence at step 19
// of 30 on two prompts), and its q8_1 weight copies read 8.5 bit/weight against
// Q4_K's 4.5 — 3662 MiB for Gemma 4 E4B, over the 2048 MiB default budget. Its
// intended use is the fused prologue/epilogue seam (GGUF quantize, SwiGLU,
// residual, sibling merge), not standalone matmul speed, which the ggml kernel
// already provides.
//
// Emitted in the GPU dialect and lowered through the same NVVM pipeline as the
// dense GEMM and attention kernels:
//
//   gpu.func + nvgpu.mma.sync (m16n8k32.s8) -> nvvm -> PTX
//
// Operands (the prefill projection's layout):
//   A  activation q8_1, block_q8_1_mmq layout: one 144-byte block per
//      (128 K values, token), blocks indexed [k_block][token] (k_block*n + t),
//      scale layout ds=0 (float d4) or ds=1 (half2 ds4).
//   Wq int8 weights [rows][K], row-major, produced by the q8_1 weight
//      converter: each 32-value group is quantized with its own f32 scale.
//   Ws f32 scales [rows][K/32].
//   Y  f32 [tokens][rows], token-major (y[token*rows + row]) — the same layout
//      the ggml mmq kernel writes.
//
// Tiling: one block per (BN-token, BM-row) tile. Each warp owns BM/8/NW weight
// groups of 8 rows and loops over the tile's BN/16 token groups, running one
// m16n8k32 mma per (weight group, token group, 32-K chunk). The int32 dot of a
// 32-K chunk is scaled by the activation's and the weight's per-32 scales and
// accumulated in f32, which is the q8_1 arithmetic the ggml mmq path uses.
//
// Batch invariance: a block owns a fixed (token, row) tile, the K loop runs in
// ascending 32-element chunks, and every accumulator sees the same sequence of
// mma instructions regardless of the launch's token count: a token's output row
// is bit-identical whether it is computed alone or as part of a large prefill.

#include <optional>
#include <string>

namespace sonicboom::compiler {

struct MmqTcShape {
  int bm = 64;  // weight rows per block   (16, 32, 64 or 128)
  int bn = 64;  // tokens per block        (16, 32, 64 or 128)
  int ds = 0;   // activation scale layout: 0/2 = float d4, 1 = half2 ds4

  bool operator==(const MmqTcShape &o) const {
    return bm == o.bm && bn == o.bn && ds == o.ds;
  }
};

// True when the generator covers the shape: bm in {16,32,64,128}, bn in
// {16,32,64,128}, ds in {0,1,2}.
bool mmq_tc_shape_supported(const MmqTcShape &shape);

// Launch geometry: min(bm/8, 8) warps of 32 threads; grid (ceil(n/bn),
// ceil(m/bm)).
int mmq_tc_block_threads(const MmqTcShape &shape);
int mmq_tc_shared_bytes(const MmqTcShape &shape);

constexpr const char *kMmqTcKernelEntry = "mmq_tc";

// Emit and lower the kernel for `shape` targeting compute capability `sm_arch`
// (86 = sm_86). Returns nullopt, having printed the MLIR diagnostic, when the
// shape has no generator or lowering fails.
std::optional<std::string> generate_mmq_tc_ptx(const MmqTcShape &shape,
                                               int sm_arch);

// The identity of the generator, stamped into the JIT disk cache entry.
std::string mmq_tc_codegen_version();

} // namespace sonicboom::compiler
