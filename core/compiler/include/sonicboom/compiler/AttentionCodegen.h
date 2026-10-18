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

// The generated FlashAttention-2 kernel (v1 Stage 3). One generated kernel per
// attention shape class: (head_dim, GQA ratio, mask kind). The kernel is
// emitted in the GPU dialect and lowered through the same NVVM pipeline as the
// glue regions (GlueCodegen.h):
//
//   gpu.func + nvgpu.mma.sync / nvgpu.ldmatrix + scf
//       -> scf/cf/arith/gpu -> nvvm -> PTX (assembly)
//
// Tiling (Br query rows x Bc keys, one block per (query tile, query head)):
// Q is staged f16 in shared memory, K staged transposed, V staged natural. A
// key tile is loaded once, S = Q K^T runs on m16n8k16 tensor cores, the
// online softmax (running max, sum, rescaled O) runs in f32, and O += P V runs
// on tensor cores. The causal / sliding-window mask is applied per element
// before the max, so masked keys never contribute.
//
// The layout conventions (validated against mma/ldmatrix probes) are:
//   A (16x16) ldmatrix x4 non-trans, addr row=l&15, col=(l>>4)*8:
//     [0]=(A[r][2c],A[r][2c+1]) [1]=(A[r+8][2c],A[r+8][2c+1])
//     [2]=(A[r][2c+8],A[r][2c+9]) [3]=(A[r+8][2c+8],A[r+8][2c+9])
//   B (16x8) ldmatrix x2.trans, addr row=kbase+(l&15), col=nbase:
//     [0]=(B[2c][r],B[2c+1][r]) [1]=(B[2c+8][r],B[2c+9][r])
//   C (16x8 f32): [0][0]=C[r][2c] [0][1]=C[r][2c+1]
//                 [1][0]=C[r+8][2c] [1][1]=C[r+8][2c+1]
//
// Layer placement: this is compiler machinery behind Layer 2 (the runtime JIT
// lives in planner/glue_jit), like GlueCodegen.

#include <optional>
#include <string>

namespace sonicboom::compiler {

// One generated attention shape class. `window_mask` selects the sliding-window
// lower bound; the window length itself is a runtime kernel argument, so the
// same kernel serves a windowed call with any window and a causal call
// (window = 0) too.
struct AttentionShape {
  int head_dim = 0;
  int gqa = 0;        // n_heads_q / n_heads_kv
  bool window_mask = false;
  int br = 64;        // query rows per block (multiple of 16)
  int bc = 64;        // key columns per tile (multiple of 16)
  // Warps the head_dim is split across: warp w = rowchunk * dps + dchunk owns
  // query rows [rowchunk*16, +16) and head_dim columns [dchunk*DC, +DC), where
  // DC = head_dim/dps <= 128. The warps exchange partial S through shared
  // memory and sum in warp order (batch invariance).
  int dps = 1;

  bool operator==(const AttentionShape &o) const {
    return head_dim == o.head_dim && gqa == o.gqa &&
           window_mask == o.window_mask && br == o.br && bc == o.bc &&
           dps == o.dps;
  }
};

// True when the generator covers the shape. Each warp holds DC/8 output
// fragments of four f32 per thread (DC = head_dim/dps <= 128), so the register
// budget bounds the per-warp column slice, not head_dim; the staging is
// 2*(br*head_dim + 2*bc*head_dim) bytes plus the partial-S buffer when dps > 1.
bool attention_shape_supported(const AttentionShape &shape);

// Launch geometry of the generated kernel: (br/16)*dps warps of 32 threads,
// one block per (query tile, query head).
int attention_block_threads(const AttentionShape &shape);
int attention_shared_bytes(const AttentionShape &shape);

// The generated kernel's PTX entry symbol (gpu.func name).
constexpr const char *kAttentionKernelEntry = "fa";

// Emit and lower the kernel for `shape` targeting compute capability
// `sm_arch` (86 = sm_86). Returns nullopt, having printed the MLIR
// diagnostic, when the shape has no generator or lowering fails.
std::optional<std::string> generate_attention_ptx(const AttentionShape &shape,
                                                  int sm_arch);

// The identity of the generator, stamped into the JIT disk cache entry (same
// contract as glue_codegen_version).
std::string attention_codegen_version();

} // namespace sonicboom::compiler
