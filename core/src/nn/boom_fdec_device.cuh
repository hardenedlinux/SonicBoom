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

// Tensor-core decode/verify attention. One block serves a group of query rows
// for one KV head and one absolute key-axis chunk (see fdec_partition.h; the
// chunk size may grow with the distance from the query): scores go through
// m16n8k16 f16 mma, keys per tile are KT (a class constant), and the query
// rows of the block are the (query row, query head) pairs of one GQA group
// packed as mma rows. The online-softmax reduction order is fixed: per-row
// fragment max, 4-lane butterfly, then warps in ascending index; per 8-key
// n-tile the four lanes' own pairs are summed and folded with the 4-lane
// butterfly, the tile sums are written to shared, and one thread per row
// folds the n-tiles in ascending order, so the row sum does not depend on
// which warp owns which tile. The exponent is stored as f16 in shared memory
// for the P·V mma and the row sum keeps the f32 values. Partials are written
// in the layout
// `[row][chunk][head][head_dim + 2]` the multi-query merge kernel reads.
//
// This header is device-only and is included by the AOT translation unit that
// launches the kernel; it shares the mma / cp.async helpers with boom_fattn.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "boom_fattn_device.cuh"  // sb_mma_f32, sb_cp_commit, sb_cp_wait
#include "fdec_partition.h"

namespace sonicboom::nn::cuda {
namespace {

// Per-class shape constants: |D|KT|W|MT_max| = 64|64|4|7, 128|32|8|7,
// 256|16|8|4, 512|16|8|1. MT_max is the number of 16-row m-tiles the block
// can hold; the rows per block follow from the GQA group width.
constexpr __host__ __device__ int fdec_kt(int D) {
  return D == 64 ? 64 : D == 128 ? 32 : 16;
}

constexpr __host__ __device__ int fdec_warps(int D) {
  return D == 64 ? 4 : 8;
}

constexpr __host__ __device__ int fdec_mt_max(int D) {
  return D == 64 ? 7 : D == 128 ? 7 : D == 256 ? 4 : 1;
}

// Dynamic shared memory of one instantiation with `stages` buffered K/V tiles
// and MT m-tiles, bytes:
//   stages x (K + V) x KT x (D + 8) halves
//   + Q (MT*16 x (D + 8) halves)
//   + P (MT*16 x (KT + 8) halves)
//   + the per-warp per-row max exchange (W x MT*16 floats)
//   + the per-row per-n-tile sums (MT*16 x KT/8 floats)
//   + the per-row totals (MT*16 floats).
// MT defaults to the class maximum; a launch that serves fewer rows sizes it
// to ceil(rows*gqa/16).
template <int D, int KT, int W, int STAGES, int MT = fdec_mt_max(D)>
constexpr __host__ __device__ size_t fdec_smem_bytes_s() {
  constexpr size_t m = size_t(MT) * 16;
  constexpr size_t nts = size_t(KT) / 8;
  return (2 * size_t(STAGES) * size_t(KT) * size_t(D + 8) +
          m * size_t(D + 8) + m * size_t(KT + 8)) *
             2 +
         size_t(W) * m * 4 + m * nts * 4 + m * 4;
}

// Three buffered tiles where they fit the sm_86 block limit, two otherwise.
template <int D, int KT, int W>
constexpr __host__ __device__ int fdec_max_stages() {
  return fdec_smem_bytes_s<D, KT, W, 3>() <= 99 * 1024 ? 3 : 2;
}

template <int D, int KT, int W>
constexpr __host__ __device__ size_t fdec_smem_bytes() {
  return fdec_smem_bytes_s<D, KT, W, fdec_max_stages<D, KT, W>()>();
}

__device__ __forceinline__ float fdec_neg_inf() {
  return __int_as_float(0xff800000);
}

// 16-byte cp.async with a source-size operand: `in == false` copies no bytes
// and zero-fills the destination (the masked keys of a tile edge).
__device__ __forceinline__ void fdec_cp_async_16(void* dst, const void* src,
                                                 bool in) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  const uint32_t bytes = in ? 16u : 0u;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s),
               "l"(src), "r"(bytes));
}

// A fragment (16 rows x 16 k) of an f16 matrix with row stride `stride`
// halves, rows 0..15 of `base`, starting at column k0. ldmatrix.x4: matrix 0/1
// are the two 8-row halves at k0, matrix 2/3 the same at k0+8 -- the mma A
// layout {a0,a1,a2,a3}.
__device__ __forceinline__ void fdec_ldm_a(uint32_t a[4],
                                           const __half* base, int stride,
                                           int k0) {
  const int lane = threadIdx.x & 31;
  const int row = lane & 15;
  const int col = k0 + ((lane & 16) ? 8 : 0);
  const uint32_t addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(base + size_t(row) * size_t(stride) + col));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
               "{%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

// B fragment (8 keys x 16 k) of the K tile [key][D + 8]: rows key0..key0+7,
// matrix 0 at k0 and matrix 1 at k0+8 -- the mma B layout {b0,b1}.
__device__ __forceinline__ void fdec_ldm_k(uint32_t b[2],
                                           const __half* base, int stride,
                                           int k0, int key0) {
  const int lane = threadIdx.x & 31;
  const int row = key0 + (lane & 7);
  const int col = k0 + ((lane & 8) ? 8 : 0);
  const uint32_t addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(base + size_t(row) * size_t(stride) + col));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(addr));
}

// B fragment of the V tile [key][D + 8] for the P·V mma: matrix 0 is keys
// key0..key0+7 x dims n0..n0+7, matrix 1 the next 8 keys, both transposed --
// the layout `V[key][dim]` is read as B[dim][key].
__device__ __forceinline__ void fdec_ldm_v(uint32_t b[2],
                                           const __half* base, int stride,
                                           int n0, int key0) {
  const int lane = threadIdx.x & 31;
  const int row = key0 + (lane & 15);
  const uint32_t addr = static_cast<uint32_t>(
      __cvta_generic_to_shared(base + size_t(row) * size_t(stride) + n0));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, "
               "[%2];\n"
               : "=r"(b[0]), "=r"(b[1])
               : "r"(addr));
}

// Grid (n_chunks, n_heads_kv, row_groups); one block = one absolute chunk
// `chunk_of(first0) + blockIdx.x` of the key-axis partition, one KV head, one
// group of query rows.
// M rows are (query row r_local, query head h of the group) at index
// `r_local*gqa + h`, the group's rows are zero past `rows*gqa`, and the
// padded rows are never written to `partial`.
template <int D, int KT, int W, int STAGES, int MT = fdec_mt_max(D)>
__global__ void __launch_bounds__(W * 32) boom_fdec_kernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache, const uint64_t* __restrict__ pos,
    uint64_t n_slots, int n_heads_q, int n_heads_kv, int gqa,
    uint64_t sliding_window, FdecPartition part, int n_chunks, int n_q,
    int pos_offset0, int rows_per_block, float scale,
    float* __restrict__ partial) {
  constexpr int NPAD = D + 8;
  constexpr int PPAD = KT + 8;
  constexpr int MAXM = MT * 16;
  constexpr int NTS = KT / 8;             // 8-key n-tiles per K tile
  constexpr int SN = (NTS + W - 1) / W;   // score n-tiles per warp
  constexpr int ON = ((D / 8) + W - 1) / W;  // O dim-tiles per warp
  // Fewer n-tiles than warps: each warp owns one n-tile and a stride of the
  // m-tiles, so every warp computes scores.
  constexpr bool SPLIT_M = NTS < W;
  constexpr int MG = SPLIT_M ? W / NTS : 1;  // m-tile stride
  static_assert(!SPLIT_M || W % NTS == 0, "warps must divide into n-tile groups");
  static_assert(fdec_smem_bytes_s<D, KT, W, STAGES, MT>() <= 99 * 1024,
                "fdec shared memory exceeds the sm_86 per-block limit");
  static_assert(MT >= 1 && MT <= fdec_mt_max(D), "MT outside the class range");
  static_assert(NPAD % 8 == 0 && PPAD % 8 == 0, "16-byte row alignment");

  extern __shared__ __half smem[];
  __half* kbuf = smem;                                 // STAGES x KT*NPAD
  __half* vbuf = kbuf + STAGES * KT * NPAD;            // STAGES x KT*NPAD
  __half* qbuf = vbuf + STAGES * KT * NPAD;            // MAXM*NPAD
  __half* pbuf = qbuf + MAXM * NPAD;                   // MAXM*PPAD
  float* wmax = reinterpret_cast<float*>(pbuf + MAXM * PPAD);
  float* tsum = wmax + W * MAXM;                       // [row][n-tile]
  float* rsum = tsum + MAXM * NTS;                     // [row]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int g = lane >> 2;
  const int t4 = lane & 3;

  const int kv_head = blockIdx.y;
  const int row_base = int(blockIdx.z) * rows_per_block;
  const int rows =
      (n_q - row_base < rows_per_block) ? n_q - row_base : rows_per_block;
  if (rows <= 0) return;
  const int mrows = rows * gqa;
  const int mtiles = (mrows + 15) >> 4;

  // (1) positions, window rule, this block's key range. Chunk indices are
  // absolute and anchored at the launch's row 0 first chunk (the merge kernel
  // maps row r's chunks relative to that), so the split never depends on the
  // capacity or the position beyond what the keys themselves say. The block's
  // own rows start at row_base.
  const uint64_t p_first = *pos + uint64_t(pos_offset0);
  const uint64_t row0 = p_first + uint64_t(row_base);
  const uint64_t start0 = (sliding_window > 0 && p_first >= sliding_window)
                              ? p_first - sliding_window + 1
                              : 0;
  const uint64_t c = part.chunk_of(start0) + uint64_t(blockIdx.x);
  const uint64_t cb = part.chunk_start(c);
  const uint64_t clen = part.chunk_len(c);
  const uint64_t sfirst = (sliding_window > 0 && row0 >= sliding_window)
                              ? row0 - sliding_window + 1
                              : 0;
  const uint64_t plast = row0 + uint64_t(rows - 1);
  const uint64_t jlo = cb > sfirst ? cb : sfirst;
  const uint64_t jhi = (cb + clen < plast + 1) ? cb + clen : plast + 1;
  if (jlo >= jhi) return;
  const int t_lo = int((jlo - cb) / uint64_t(KT));
  const int t_hi = int((jhi - 1 - cb) / uint64_t(KT));
  const int ntiles = t_hi - t_lo + 1;

  // (2) Q: scaled and rounded to f16 once, one row per (row, head) pair.
  for (int i = tid; i < mtiles * 16 * D; i += W * 32) {
    const int rm = i / D;
    const int d = i - rm * D;
    float v = 0.0f;
    if (rm < mrows) {
      const int rl = rm / gqa;
      const int hh = rm - rl * gqa;
      v = q[(size_t(row_base + rl) * size_t(n_heads_q) +
             size_t(kv_head) * size_t(gqa) + size_t(hh)) *
                size_t(D) +
            size_t(d)] *
          scale;
    }
    qbuf[size_t(rm) * NPAD + d] = __float2half_rn(v);
  }

  // (3) K/V tiles, still anchored: one slot modulo per tile, then the key
  // index wraps. Keys outside the block range are zero-filled.
  constexpr int DCH = D / 8;
  constexpr int COPIES = KT * DCH;
  auto stage = [&](int ti, int buf) {
    const uint64_t base = cb + uint64_t(ti) * KT;
    const uint64_t s0 = base % n_slots;
    __half* kd = kbuf + size_t(buf) * (KT * NPAD);
    __half* vd = vbuf + size_t(buf) * (KT * NPAD);
    for (int i = tid; i < COPIES; i += W * 32) {
      const int key = i / DCH;
      const int dd = (i - key * DCH) * 8;
      const uint64_t j = base + uint64_t(key);
      const bool in = (j >= jlo) && (j < jhi);
      uint64_t slot = s0 + uint64_t(key);
      if (slot >= n_slots) slot -= n_slots;
      const size_t row =
          (size_t(slot) * size_t(n_heads_kv) + size_t(kv_head)) * size_t(D) +
          size_t(dd);
      fdec_cp_async_16(kd + size_t(key) * NPAD + dd, k_cache + row, in);
      fdec_cp_async_16(vd + size_t(key) * NPAD + dd, v_cache + row, in);
    }
    sb_cp_commit();
  };
  const int npre = ntiles < STAGES ? ntiles : STAGES;
  for (int i = 0; i < npre; ++i) stage(t_lo + i, i);
  if (npre == 1) {
    sb_cp_wait<0>();
  } else if (npre == 2) {
    sb_cp_wait<1>();
  } else {
    sb_cp_wait<2>();
  }
  __syncthreads();

  // Online-softmax state, one entry per (m-tile, row half). The O accumulator
  // is O[M][D] f32 in mma fragments.
  float mreg[MT][2], lreg[MT][2];
  float o[MT][ON][4];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    mreg[mt][0] = fdec_neg_inf();
    mreg[mt][1] = fdec_neg_inf();
    lreg[mt][0] = 0.0f;
    lreg[mt][1] = 0.0f;
#pragma unroll
    for (int on = 0; on < ON; ++on) {
      o[mt][on][0] = 0.0f;
      o[mt][on][1] = 0.0f;
      o[mt][on][2] = 0.0f;
      o[mt][on][3] = 0.0f;
    }
  }

  for (int tt = 0; tt < ntiles; ++tt) {
    const int buf = tt % STAGES;
    const uint64_t tbase = cb + uint64_t(t_lo + tt) * KT;
    const __half* kt = kbuf + size_t(buf) * (KT * NPAD);
    const __half* vt = vbuf + size_t(buf) * (KT * NPAD);
    float alpha[MT][2];
    // The warp's score n-tile when n-tiles are fewer than warps: tile
    // `warp % NTS` of the m-tiles `mt` with `mt % MG == warp / NTS`. With at
    // least as many n-tiles as warps, warp w owns tiles w, w + W, ...
    const int own_nt = SPLIT_M ? (warp % NTS) : 0;
    const int own_mod = SPLIT_M ? (warp / NTS) : 0;
    const auto owns_mt = [&](int mt) {
      return !SPLIT_M || (mt % MG) == own_mod;
    };

    // (4) scores: S[M][KT] over D ascending k-steps, one fragment per owned
    // (m-tile, n-tile). A warp without a tile of an m-tile still joins the
    // reductions below with a -inf entry.
    float s[MT][SN][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int ni = 0; ni < SN; ++ni)
        s[mt][ni][0] = s[mt][ni][1] = s[mt][ni][2] = s[mt][ni][3] = 0.0f;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= mtiles || !owns_mt(mt)) continue;
      for (int k0 = 0; k0 < D; k0 += 16) {
        uint32_t aq[4];
        fdec_ldm_a(aq, qbuf + size_t(mt) * 16 * NPAD, NPAD, k0);
#pragma unroll
        for (int ni = 0; ni < SN; ++ni) {
          const int nt = SPLIT_M ? own_nt : (warp + ni * W);
          if (nt >= NTS) continue;
          uint32_t bk[2];
          fdec_ldm_k(bk, kt, NPAD, k0, nt * 8);
          sb_mma_f32(s[mt][ni], aq, bk);
        }
      }
    }

    // (5) causal + window mask, (6) row max: fragment values, then the 4
    // lanes sharing the row, then the warps in ascending index.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= mtiles) continue;
      if (!owns_mt(mt)) {
        if (t4 == 0) {
          const int rm0 = mt * 16 + g;
          wmax[size_t(warp) * MAXM + rm0] = fdec_neg_inf();
          wmax[size_t(warp) * MAXM + rm0 + 8] = fdec_neg_inf();
        }
        continue;
      }
      const int rm0 = mt * 16 + g;
      const int rm1 = rm0 + 8;
      const bool real0 = rm0 < mrows;
      const bool real1 = rm1 < mrows;
      const uint64_t p_r0 = row0 + uint64_t(real0 ? rm0 / gqa : 0);
      const uint64_t p_r1 = row0 + uint64_t(real1 ? rm1 / gqa : 0);
      const uint64_t s_r0 = (sliding_window > 0 && p_r0 >= sliding_window)
                                ? p_r0 - sliding_window + 1
                                : 0;
      const uint64_t s_r1 = (sliding_window > 0 && p_r1 >= sliding_window)
                                ? p_r1 - sliding_window + 1
                                : 0;
      float mx0 = fdec_neg_inf();
      float mx1 = fdec_neg_inf();
#pragma unroll
      for (int ni = 0; ni < SN; ++ni) {
        const int nt = SPLIT_M ? own_nt : (warp + ni * W);
        if (nt >= NTS) continue;
        const uint64_t j0 = tbase + uint64_t(nt * 8 + 2 * t4);
        float* sv = s[mt][ni];
        sv[0] = (real0 && j0 >= s_r0 && j0 <= p_r0) ? sv[0] : fdec_neg_inf();
        sv[1] = (real0 && j0 + 1 >= s_r0 && j0 + 1 <= p_r0) ? sv[1]
                                                           : fdec_neg_inf();
        sv[2] = (real1 && j0 >= s_r1 && j0 <= p_r1) ? sv[2] : fdec_neg_inf();
        sv[3] = (real1 && j0 + 1 >= s_r1 && j0 + 1 <= p_r1) ? sv[3]
                                                           : fdec_neg_inf();
        mx0 = fmaxf(mx0, fmaxf(sv[0], sv[1]));
        mx1 = fmaxf(mx1, fmaxf(sv[2], sv[3]));
      }
      mx0 = sb_rmax4(mx0);
      mx1 = sb_rmax4(mx1);
      if (t4 == 0) {
        wmax[size_t(warp) * MAXM + rm0] = mx0;
        wmax[size_t(warp) * MAXM + rm1] = mx1;
      }
    }
    __syncthreads();

    // m_new / alpha for every real m-tile. The row max is a fold of
    // order-free maxima, so every warp derives the same value.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= mtiles) continue;
      const int rm0 = mt * 16 + g;
      const int rm1 = rm0 + 8;
      float tm0 = wmax[rm0];
      float tm1 = wmax[rm1];
#pragma unroll
      for (int wi = 1; wi < W; ++wi) {
        tm0 = fmaxf(tm0, wmax[size_t(wi) * MAXM + rm0]);
        tm1 = fmaxf(tm1, wmax[size_t(wi) * MAXM + rm1]);
      }
      const float mo0 = mreg[mt][0];
      const float mo1 = mreg[mt][1];
      const float mn0 = fmaxf(mo0, tm0);
      const float mn1 = fmaxf(mo1, tm1);
      const float b0 = (mn0 == fdec_neg_inf()) ? 0.0f : mn0;
      const float b1 = (mn1 == fdec_neg_inf()) ? 0.0f : mn1;
      const float al0 = (mo0 == mn0) ? 1.0f : expf(mo0 - b0);
      const float al1 = (mo1 == mn1) ? 1.0f : expf(mo1 - b1);
      alpha[mt][0] = al0;
      alpha[mt][1] = al1;
      mreg[mt][0] = mn0;
      mreg[mt][1] = mn1;
    }

    // The exponent as f16 in P and the per-n-tile row sums: the tile sum is
    // the four lanes' own pairs (c0+c1 / c2+c3), then the width-4 butterfly.
    // Each owned tile writes its sum to shared; the row total below folds the
    // n-tiles in ascending order, so it does not depend on which warp owns a
    // tile. P itself is per element, so the ownership split is invisible to it.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= mtiles || !owns_mt(mt)) continue;
      const int rm0 = mt * 16 + g;
      const int rm1 = rm0 + 8;
      // mreg now holds m_new for the tile: the exponent's bias.
      const float b0 = (mreg[mt][0] == fdec_neg_inf()) ? 0.0f : mreg[mt][0];
      const float b1 = (mreg[mt][1] == fdec_neg_inf()) ? 0.0f : mreg[mt][1];
#pragma unroll
      for (int ni = 0; ni < SN; ++ni) {
        const int nt = SPLIT_M ? own_nt : (warp + ni * W);
        if (nt >= NTS) continue;
        const int key0 = nt * 8 + 2 * t4;
        const float* sv = s[mt][ni];
        const float p00 = expf(sv[0] - b0);
        const float p01 = expf(sv[1] - b0);
        const float p10 = expf(sv[2] - b1);
        const float p11 = expf(sv[3] - b1);
        pbuf[size_t(mt * 16 + g) * PPAD + key0] = __float2half_rn(p00);
        pbuf[size_t(mt * 16 + g) * PPAD + key0 + 1] = __float2half_rn(p01);
        pbuf[size_t(mt * 16 + g + 8) * PPAD + key0] = __float2half_rn(p10);
        pbuf[size_t(mt * 16 + g + 8) * PPAD + key0 + 1] =
            __float2half_rn(p11);
        const float sum0 = sb_rsum4(p00 + p01);
        const float sum1 = sb_rsum4(p10 + p11);
        if (t4 == 0) {
          tsum[size_t(rm0) * NTS + nt] = sum0;
          tsum[size_t(rm1) * NTS + nt] = sum1;
        }
      }
    }
    __syncthreads();
    // One thread per row folds the n-tile sums in ascending order.
    for (int rm = tid; rm < mtiles * 16; rm += W * 32) {
      float acc = 0.0f;
#pragma unroll
      for (int nt = 0; nt < NTS; ++nt) acc += tsum[size_t(rm) * NTS + nt];
      rsum[rm] = acc;
    }
    __syncthreads();

    // (7) l = l*alpha + row sum, then O *= alpha and the P·V mma over the
    // tile's keys ascending. Every warp reads the same totals.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (mt >= mtiles) continue;
      const int rm0 = mt * 16 + g;
      const int rm1 = rm0 + 8;
      lreg[mt][0] = lreg[mt][0] * alpha[mt][0] + rsum[rm0];
      lreg[mt][1] = lreg[mt][1] * alpha[mt][1] + rsum[rm1];
#pragma unroll
      for (int on = 0; on < ON; ++on) {
        const int ot = warp + on * W;
        if (ot >= D / 8) continue;
        o[mt][on][0] *= alpha[mt][0];
        o[mt][on][1] *= alpha[mt][0];
        o[mt][on][2] *= alpha[mt][1];
        o[mt][on][3] *= alpha[mt][1];
      }
      for (int ks = 0; ks < KT / 16; ++ks) {
        uint32_t ap[4];
        fdec_ldm_a(ap, pbuf + size_t(mt) * 16 * PPAD, PPAD, ks * 16);
#pragma unroll
        for (int on = 0; on < ON; ++on) {
          const int ot = warp + on * W;
          if (ot >= D / 8) continue;
          uint32_t bv[2];
          fdec_ldm_v(bv, vt, NPAD, ot * 8, ks * 16);
          sb_mma_f32(o[mt][on], ap, bv);
        }
      }
    }
    __syncthreads();

    if (tt + 1 < ntiles) {
      if (tt + STAGES < ntiles) {
        stage(t_lo + tt + STAGES, buf);
        sb_cp_wait<STAGES - 1>();
      } else {
        sb_cp_wait<0>();
      }
      __syncthreads();
    }
  }

  // (8) partials: (m, l, O) per real row in natural-log units, in the
  // [row][chunk][head][D + 2] layout the multi-query merge reads.
  const size_t pstride = size_t(n_heads_q) * size_t(D + 2);
  for (int mt = 0; mt < MT; ++mt) {
    if (mt >= mtiles) continue;
    const int rm0 = mt * 16 + g;
    const int rm1 = rm0 + 8;
    float* pp[2] = {nullptr, nullptr};
    const int rms[2] = {rm0, rm1};
    for (int k = 0; k < 2; ++k) {
      if (rms[k] >= mrows) continue;
      const int rl = rms[k] / gqa;
      const int hh = rms[k] - rl * gqa;
      pp[k] = partial +
              (size_t(row_base + rl) * size_t(n_chunks) + size_t(blockIdx.x)) *
                  pstride +
              size_t(kv_head * gqa + hh) * size_t(D + 2);
    }
    if (warp == 0 && t4 == 0) {
      if (pp[0] != nullptr) {
        pp[0][0] = mreg[mt][0];
        pp[0][1] = lreg[mt][0];
      }
      if (pp[1] != nullptr) {
        pp[1][0] = mreg[mt][1];
        pp[1][1] = lreg[mt][1];
      }
    }
#pragma unroll
    for (int on = 0; on < ON; ++on) {
      const int ot = warp + on * W;
      if (ot >= D / 8) continue;
      const int col = 2 + ot * 8 + 2 * t4;
      if (pp[0] != nullptr) {
        pp[0][col] = o[mt][on][0];
        pp[0][col + 1] = o[mt][on][1];
      }
      if (pp[1] != nullptr) {
        pp[1][col] = o[mt][on][2];
        pp[1][col + 1] = o[mt][on][3];
      }
    }
  }
}

}  // namespace
}  // namespace sonicboom::nn::cuda
