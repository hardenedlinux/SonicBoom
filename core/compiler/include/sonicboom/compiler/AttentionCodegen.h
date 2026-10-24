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
  // Flash-decoding (q_len = 1) split-KV kernels: `fa_decode` writes one
  // partial (m, l, O) per (head, key chunk) and `fa_combine` reduces the
  // partials in ascending chunk order. The chunk length is a runtime argument
  // (fixed per shape class by the caller, at most kAttentionDecodeMaxChunk),
  // so the generated text, br/bc/dps and the region identity do not depend on
  // it. In decode mode br/bc/dps are ignored.
  bool decode = false;

  // P-in-registers variant (prefill): the head_dim is not split across warps.
  // One warp owns one (query head, 16-row chunk) for its whole head_dim, so S
  // is produced entirely inside that warp, the softmax runs on its own
  // accumulated fragments, and P is converted to fp16 A-fragments in registers
  // for P V -- no partial-S exchange and no P round trip through shared
  // memory. A block holds `gqa` warps, one per query head of the same KV head,
  // so the K/V tile staged in shared memory is shared by all of them; the grid
  // is (query tiles, n_heads_kv) with br fixed at 16. The register budget caps
  // this variant at head_dim <= 256 (head_dim/2 f32 accumulators per lane;
  // wider heads keep the dps-split kernel). Its S reduction order differs from
  // the dps-split kernel's (one ascending k sweep instead of per-dps partials
  // summed in warp order), so it is not token-identical and stays an opt-in
  // arm.
  bool warp_head = false;

  // Query heads packed into one block (GQA packing), 1 <= gqa_pack <= gqa and
  // gqa % gqa_pack == 0. A block still covers one br-row query tile, but for
  // gqa_pack consecutive query heads that share the same K/V head, so the K/V
  // tile staged in shared memory is read from global memory once for all of
  // them instead of once per query head (traffic / gqa_pack). Each head's rows
  // keep their own warp roles and the same per-warp arithmetic, so the output
  // is bit-identical to the unpacked kernel. The grid's y dimension becomes
  // n_heads_q / gqa_pack. Warp-head blocks are already packed (one warp per
  // query head) and ignore this field.
  int gqa_pack = 1;

  // Keep the Q fragments in registers across the key-tile loop (default), or
  // reload them from the staged Q tile in every key tile. Reloading costs KT
  // ldmatrix per tile but frees the dc/4 registers the fragments occupy, which
  // is what lets a dc = 256 slice (dps = 2 at head_dim 512) run without
  // spilling. Same values either way, so the output is bit-identical.
  bool q_in_reg = true;

  // Grid order. Default: blockIdx.x = query tile, blockIdx.y = head, so
  // consecutive blocks walk the query tiles of one head. head_fast: the two
  // dimensions swap (blockIdx.x = head, blockIdx.y = query tile), so the
  // blocks launched together cover all heads of one query tile and read the
  // same K/V range at the same time (one DRAM read serves all of them through
  // L2). Which block does which tile is the only change -- the block's
  // arithmetic is untouched, so the output is bit-identical. It pays when a
  // block reads its K/V range from global memory once per head (small br or
  // packed blocks); it is neutral when enough rows share the tile already.
  bool head_fast = false;

  // cp.async pipeline depth for the K/V staging (0 = synchronous staging, the
  // default; 2 or 3 = pipeline stages). The KV cache and batch buffers are
  // f32 and the mma operands are f16, so cp.async cannot write the mma tiles
  // directly: each stage first copies the tile's f32 bytes into shared memory
  // with cp.async (16-byte chunks, overlapping the previous tile's compute),
  // then a vectorized in-shared conversion writes the f16 K/V tiles. The f32
  // prefetch buffers are the pipeline stages; their size is part of the
  // shape's shared-memory budget, and the tail of the pipeline is predicated
  // by clamping the source key (out-of-range keys are masked in the softmax,
  // so their staged values never contribute). Staging order and rounding are
  // identical to the synchronous path, so token streams are bit-identical.
  int async_stages = 0;

  // exp2 softmax: the scores are scaled by scale*log2(e) and the exponential
  // is math.exp2, so the online-softmax max and the running rescale factors
  // live in log2 units. This saves the per-score scale multiply and lets the
  // exponential lower to the hardware exp2 approximation, but it is not
  // bit-identical to the exp path (the folded scale rounds differently), so it
  // is an opt-in arm (SONICBOOM_ATTN_EXP2).
  bool exp2 = false;

  // Full attention: non-causal, no window, no KV cache. Every query attends
  // to every key in [0, n); the kernel reads the batch K/V planes only (no
  // cache arithmetic, no %startp load) and masks only the keys past n in the
  // final partial tile. This is a distinct mask kind, not a causal shape with
  // a large window: a causal class must never resolve to it (and vice versa),
  // so it is part of the region identity. Only the warp-head layout is
  // generated for it: br = 16*w gives one 16-row chunk per warp while the
  // block's warps share one K/V tile staged in shared memory, which is the
  // FA-2 layout (each warp owns distinct query rows and the whole head_dim;
  // S and P stay in registers).
  bool full_attn = false;

  // The KV cache is f16 instead of f32 (the standard numerics profile). The
  // batch K/V of the current chunk stay f32, so the staging reads the two
  // sources with different element types; the generated kernel then stages
  // synchronously (cp.async copies raw bytes and cannot convert a mixed
  // source, and an f16 cache tile needs a widening load anyway). The values
  // read are the same fp16-rounded numbers in both cases, so the kernel's
  // arithmetic is unchanged; only the staging path differs. Part of the
  // region identity (the JIT cache key).
  bool kv_f16 = false;

  // The KV cache is q8 instead of f32: each row is head_dim int8 values then
  // one f32 scale (max|x| / 127), so a row is addressed in bytes and every
  // load dequantizes (int8 -> f32 -> scale). Like kv_f16, the batch K/V of the
  // current chunk stay f32 (rounding to f16 on read where the cache would have
  // rounded), and the generated kernel stages synchronously. Mutually
  // exclusive with kv_f16. Part of the region identity (the JIT cache key).
  bool kv_q8 = false;

  bool operator==(const AttentionShape &o) const {
    return head_dim == o.head_dim && gqa == o.gqa &&
           window_mask == o.window_mask && br == o.br && bc == o.bc &&
           dps == o.dps && decode == o.decode && warp_head == o.warp_head &&
           gqa_pack == o.gqa_pack && q_in_reg == o.q_in_reg &&
           head_fast == o.head_fast &&
           async_stages == o.async_stages && exp2 == o.exp2 &&
           full_attn == o.full_attn && kv_f16 == o.kv_f16 &&
           kv_q8 == o.kv_q8;
  }
};

// True when the generator covers the shape. Each warp holds DC/8 output
// fragments of four f32 per thread (DC = head_dim/dps <= 128), so the register
// budget bounds the per-warp column slice, not head_dim; the staging is
// 2*(br*(head_dim+8) + head_dim*(bc+8) + bc*(head_dim+8)) bytes (the +8 is
// the bank-conflict row pad) plus the partial-S buffer when dps > 1.
// Warp-head shapes (warp_head) additionally require br % 16 == 0, dps == 1,
// gqa*(br/16) <= 16 warps, head_dim % 16 == 0 and head_dim <= 256 (the whole
// head_dim is accumulated in one warp), and their staging is
// 2*(gqa*br*(head_dim+8) + head_dim*(bc+8) + bc*(head_dim+8)).
// Full-attention shapes (full_attn) require the warp-head layout and no
// window; a causal or windowed class never resolves to them.
// Decode shapes need head_dim % 32 == 0 and head_dim / 32 <= 16 accumulator
// registers per lane (a single warp owns one head), and head_dim <= 1024 so the
// combine block fits the thread limit.
bool attention_shape_supported(const AttentionShape &shape);

// Launch geometry of the generated kernel: (br/16)*dps warps of 32 threads,
// one block per (query tile, query head). Decode shapes: one warp per block,
// with the chunk's scores staged in 4 * kAttentionDecodeMaxChunk bytes of
// dynamic shared memory (the combine kernel uses head_dim threads and no
// shared memory).
int attention_block_threads(const AttentionShape &shape);
int attention_shared_bytes(const AttentionShape &shape);

// The generated kernel's PTX entry symbol (gpu.func name).
constexpr const char *kAttentionKernelEntry = "fa";

// Flash-decoding entries (decode shapes only). `fa_decode` takes
//   q, k_cache, v_cache, start, n_slots, n_heads_q, n_heads_kv, window, scale,
//   partial, chunk
// with grid (n_chunks, n_heads_q) and one warp per block; `fa_combine` takes
//   partial, out, n_heads_q, n_chunks
// with grid (n_heads_q) and head_dim threads per block. The partial buffer is
// [n_chunks][n_heads_q][head_dim + 2] f32 (O, then m and l).
constexpr const char *kAttentionDecodeKernelEntry = "fa_decode";
constexpr const char *kAttentionCombineKernelEntry = "fa_combine";

// Largest key chunk the generated decoder supports: the chunk's scores live in
// a fixed 256-slot shared-memory view (the block is one warp and every lane
// needs each key's score for the P·V pass).
constexpr int kAttentionDecodeMaxChunk = 256;

// Emit and lower the kernel for `shape` targeting compute capability
// `sm_arch` (86 = sm_86). Returns nullopt, having printed the MLIR
// diagnostic, when the shape has no generator or lowering fails.
std::optional<std::string> generate_attention_ptx(const AttentionShape &shape,
                                                  int sm_arch);

// The identity of the generator, stamped into the JIT disk cache entry (same
// contract as glue_codegen_version).
std::string attention_codegen_version();

} // namespace sonicboom::compiler
