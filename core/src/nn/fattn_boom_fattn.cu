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

// SonicBoom "GGML-inspired" chunked-prefill flash attention (`boom_fattn`).
//
// This is SonicBoom's own kernel: the algorithm was informed by the prefill
// path of the pinned llama.cpp commit (per-key f16 KV materialization, tiled
// MMA attention with an online softmax, GQA query grouping, cp.async staging,
// fp16 VKQ accumulation), but no ggml source is copied and the kernel does not
// call any ggml/llama.cpp code. It is a third, independent backend next to the
// native tiled/scalar kernels and the imported ggml kernel:
//
//   SONICBOOM_ATTN_IMPL=boom_fattn   (this file; "auto" never selects it)
//   SONICBOOM_ATTN_PREFILL=boom_fattn (prefill leg only)
//
// boom_fattn is SonicBoom's own implementation and its only name: selection
// values, configuration switches, kernel symbols and trace strings all use it.
//
// q8 KV storage and shapes without an instantiation decline (return false) and
// the caller falls through to the native path. Removing the backend is
// deleting this file, its declaration in cuda_resident.h, the three-line
// dispatch block in cuda_elementwise.cu and this file's line in
// core/CMakeLists.txt.
//
// Structure, and where it differs from the imported ggml kernel:
//
//   1. A gather kernel materializes the KV span this call needs once, as
//      [kv_head][key][dim] f16 for K and [kv_head][dim][key] f16 for V
//      (transposed, so the PV B-fragments are contiguous key runs). Reads of
//      the f32 batch buffer and of the f32/f16 cache are vectorized, and the
//      transpose is done through one 32x64 shared-memory tile per block. This
//      replaces ggml's scalar per-element cast and its large f16 additive mask.
//   2. The attention kernel derives the causal + sliding-window mask in
//      registers from the device start position, so there is no mask buffer
//      and no KV_max buffer, and the launch shape never depends on `start`
//      (CUDA-Graph capturable exactly like the native kernel).
//   3. Query tiles are 32 rows with all GQA heads of the KV head in one block
//      (ggml's prefill tiles are 16 queries), so the K/V traffic per key tile
//      is amortized over 32 query rows. A block is UNITS*NW = 32 warps (1024
//      threads); each unit is 16 query rows of one GQA group, split across NW
//      dim slices for the PV phase. Phase 1 (S and online softmax) runs on the
//      slice-0 warps and hands P (f16) and the per-sub-tile rescale factors
//      (f32) to the other slices through shared memory, as the native tiled
//      kernel does. This is the kernel selected by SONICBOOM_BOOM_FATTN_QREG=0;
//      its phase 1 is the serial part of the block and is why the imported
//      ggml kernel wins at long context. The d512 default instead uses the
//      query-resident variant below (mode 1), which splits phase 1 over the
//      unit's dim slices. The d256 default (the Gemma sliding-window head)
//      uses that variant too (SONICBOOM_BOOM_FATTN_D256=1): 16-key tiles
//      re-read Q twice as often as 32-key tiles, and phase 1 on 8 of 32 warps
//      was the prefill bottleneck. The variant is window-agnostic; the window
//      itself comes only from the per-layer metadata and may be 0 here.
//   4. K and V staging is cp.async double-buffered per key tile; the V tile is
//      padded to a 16-byte-aligned stride (BK+8 halves) so every cp.async is a
//      16-byte transfer.
//   5. Q is materialized as f16 once per call (a linear cast kernel) and read
//      from there by phase 1: the same __float2half_rn rounding as before, half
//      the bytes, no per-tile conversion. Measured +7.6% at pp4096 and +12% at
//      pp8192 on top of the f16-KV kernel.
//   6. The PV accumulation is fp16 (half2) by default, like ggml's prefill
//      kernel, and can be switched to fp32 with SONICBOOM_BOOM_FATTN_ACC=f32. The
//      fp32 variant halves the tile or widens the dim split to stay within the
//      64-register budget of a 1024-thread block.
//   7. A block's key loop stops at its own last query row, not at the end of
//      the chunk: causal prefill therefore does n^2/2 per chunk instead of n^2,
//      and a sliding-window block scans its window instead of the whole chunk.
//      Measured +5..8% end to end; the dropped tiles are fully masked.
//
// Batch invariance: key tiles are aligned to absolute multiples of BK, never
// to the chunk boundary, so a query's tile sequence (and therefore its
// softmax rescale order and bit pattern) is the same for any chunking of the
// same absolute positions.

#include "cuda_resident.h"

#include "cuda_stream.h"

#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <type_traits>

namespace sonicboom::nn::cuda {
namespace {

// ---------------------------------------------------------------------------
// MMA and fragment helpers (m16n8k16 row.col; same fragment layout as the
// native tiled kernel: C holds 16 queries x 8 keys / dims).
// ---------------------------------------------------------------------------

__device__ __forceinline__ __half2 sb_as_h2(uint32_t u) {
  return *reinterpret_cast<__half2*>(&u);
}

__device__ __forceinline__ uint32_t sb_as_u32(__half2 h) {
  return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ void sb_mma_f32(float c[4], const uint32_t a[4],
                                           const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void sb_mma_f16(uint32_t c[2], const uint32_t a[4],
                                           const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
      "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
      : "+r"(c[0]), "+r"(c[1])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A-fragment from f16 Q: rows qr..qr+15 (chunk-relative), cols k0..k0+15 of
// query head hq. Rows past the chunk read zeros (their output is not written).
//
// Q is read once per key tile by the phase-1 warp, so this is the hottest load
// in the kernel: at pp8192 a block re-reads its 16-row Q slice for every
// 16-key tile, and the block's Q working set (BR x heads x head_dim) does not
// fit in L1. Materializing Q as f16 once per call halves that traffic and
// removes the per-tile float->half conversion (the f16 rounding is the same
// __float2half_rn the old pack2 did, so the result is bit-identical).
__device__ __forceinline__ void sb_load_q(uint32_t a[4],
                                          const __half* __restrict__ q, int hq,
                                          int n_heads_q, int head_dim, int qr,
                                          int k0, int n) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t row_stride = (size_t)n_heads_q * head_dim;
  const int r0 = qr + g;
  const int r1 = qr + g + 8;
  const bool v0 = r0 < n;
  const bool v1 = r1 < n;
  const __half* p0 = q + (size_t)r0 * row_stride + (size_t)hq * head_dim + k0;
  const __half* p1 = q + (size_t)r1 * row_stride + (size_t)hq * head_dim + k0;
  a[0] = v0 ? *reinterpret_cast<const uint32_t*>(p0 + t * 2) : 0u;
  a[1] = v1 ? *reinterpret_cast<const uint32_t*>(p1 + t * 2) : 0u;
  a[2] = v0 ? *reinterpret_cast<const uint32_t*>(p0 + t * 2 + 8) : 0u;
  a[3] = v1 ? *reinterpret_cast<const uint32_t*>(p1 + t * 2 + 8) : 0u;
}

// B-fragment from the K tile [BK][head_dim]: n = keys key_off..+7, k = dims
// k0..k0+15.
__device__ __forceinline__ void sb_load_k(uint32_t b[2],
                                          const __half* __restrict__ k_tile,
                                          int head_dim, int k0, int key_off) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const __half* p = k_tile + (key_off + g) * head_dim + k0;
  b[0] = *reinterpret_cast<const uint32_t*>(p + t * 2);
  b[1] = *reinterpret_cast<const uint32_t*>(p + t * 2 + 8);
}

// B-fragment from the transposed V tile [head_dim][v_stride]: n = dims
// n0..n0+7, k = keys key_off..+15.
__device__ __forceinline__ void sb_load_v(uint32_t b[2],
                                          const __half* __restrict__ v_tile,
                                          int v_stride, int n0, int key_off) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const __half* p = v_tile + (n0 + g) * v_stride + key_off;
  b[0] = *reinterpret_cast<const uint32_t*>(p + t * 2);
  b[1] = *reinterpret_cast<const uint32_t*>(p + t * 2 + 8);
}

__device__ __forceinline__ float sb_rmax4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1, 4));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2, 4));
  return x;
}

__device__ __forceinline__ float sb_rsum4(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1, 4);
  x += __shfl_xor_sync(0xffffffffu, x, 2, 4);
  return x;
}

// ---------------------------------------------------------------------------
// cp.async (16-byte copy, one commit group per key tile)
// ---------------------------------------------------------------------------

__device__ __forceinline__ void sb_cp_async_16(void* dst, const void* src) {
  const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s),
               "l"(src));
}

__device__ __forceinline__ void sb_cp_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void sb_cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ---------------------------------------------------------------------------
// KV materialization: f16 K natural ([h][key][d]) and f16 V transposed
// ([h][d][key]), for the relative span [base, base + ne_pad).
//
// Grid: x = head_dim/64 dim tiles, y = ne_pad/32 key tiles, z = kv heads.
// Each thread loads 8 elements of one key row (K), writes them to the K span
// and to a 32x64 smem tile; the tile is then read transposed into the V span.
// Rows past the current key range are written as zeros. Every element of the
// span is written on every call, so the workspace needs no clearing.
//
// The span base is derived in-kernel from the device start position and
// aligned down to 32 keys; the attention kernel derives it the same way, so
// `start` never has to leave the device (CUDA-Graph capturable).
// ---------------------------------------------------------------------------

constexpr uint64_t kSpanAlign = 32;

__device__ __forceinline__ uint64_t sb_span_base(uint64_t start,
                                                 uint64_t window) {
  uint64_t b = (window > 0 && start + 1 >= window) ? start - (window - 1) : 0;
  b -= b % kSpanAlign;
  return b;
}

template <typename CACHE>
__device__ __forceinline__ void boom_fattn_copy8(__half dst[8], const CACHE* src) {
  if constexpr (std::is_same_v<CACHE, __half>) {
    const uint4 u = *reinterpret_cast<const uint4*>(src);
    const __half* h = reinterpret_cast<const __half*>(&u);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = h[i];
  } else {
    const float4 a = *reinterpret_cast<const float4*>(src);
    const float4 b = *reinterpret_cast<const float4*>(src + 4);
    dst[0] = __float2half_rn(a.x);
    dst[1] = __float2half_rn(a.y);
    dst[2] = __float2half_rn(a.z);
    dst[3] = __float2half_rn(a.w);
    dst[4] = __float2half_rn(b.x);
    dst[5] = __float2half_rn(b.y);
    dst[6] = __float2half_rn(b.z);
    dst[7] = __float2half_rn(b.w);
  }
}

union sb_half8 {
  __half h[8];
  uint4 u;
};

// Q materialization: [n][heads_q][head_dim] f32 -> f16, 8 elements per thread.
// Same rounding as the f32 fragment loads it replaces, so attention output is
// unchanged bit for bit.
__global__ void __launch_bounds__(256) boom_fattn_qcast_kernel(
    const float* __restrict__ q, int n_elems, __half* __restrict__ q16) {
  const int e = (blockIdx.x * 256 + threadIdx.x) * 8;
  if (e >= n_elems) return;
  const float4 a = *reinterpret_cast<const float4*>(q + e);
  const float4 b = *reinterpret_cast<const float4*>(q + e + 4);
  sb_half8 h;
  h.h[0] = __float2half_rn(a.x);
  h.h[1] = __float2half_rn(a.y);
  h.h[2] = __float2half_rn(a.z);
  h.h[3] = __float2half_rn(a.w);
  h.h[4] = __float2half_rn(b.x);
  h.h[5] = __float2half_rn(b.y);
  h.h[6] = __float2half_rn(b.z);
  h.h[7] = __float2half_rn(b.w);
  *reinterpret_cast<uint4*>(q16 + e) = h.u;
}

template <typename CACHE>
__global__ void __launch_bounds__(256) boom_fattn_gather_kernel(
    const float* __restrict__ k_batch, const float* __restrict__ v_batch,
    const CACHE* __restrict__ k_cache, const CACHE* __restrict__ v_cache,
    const uint64_t* __restrict__ start_ptr, uint64_t n_slots, int n,
    int n_heads_kv, int head_dim, int ne_pad, uint64_t window,
    __half* __restrict__ k_ws, __half* __restrict__ v_ws) {
  const uint64_t start = *start_ptr;
  const uint64_t key_end = start + (uint64_t)n;  // exclusive
  const uint64_t base = sb_span_base(start, window);
  const int h = blockIdx.z;
  const int key0 = blockIdx.y * 32;
  const int d0 = blockIdx.x * 64;

  __shared__ __half tile[32][72];  // 64 dims padded to a 16-byte row stride

  const int tid = threadIdx.x;
  const int key = tid >> 3;
  const int lane_d = tid & 7;
  const int d = d0 + lane_d * 8;
  const uint64_t abs = base + (uint64_t)(key0 + key);

  sb_half8 kv, vv;
  if (abs < key_end) {
    if (abs < start) {
      const uint64_t slot = abs % n_slots;
      const size_t cache_off =
          (slot * (uint64_t)n_heads_kv + (uint64_t)h) * (uint64_t)head_dim + d;
      boom_fattn_copy8(kv.h, k_cache + cache_off);
      boom_fattn_copy8(vv.h, v_cache + cache_off);
    } else {
      const uint64_t bi = abs - start;
      const size_t batch_off =
          (bi * (uint64_t)n_heads_kv + (uint64_t)h) * (uint64_t)head_dim + d;
      boom_fattn_copy8(kv.h, k_batch + batch_off);
      boom_fattn_copy8(vv.h, v_batch + batch_off);
    }
  } else {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      kv.h[i] = __float2half_rn(0.0f);
      vv.h[i] = __float2half_rn(0.0f);
    }
  }

  *reinterpret_cast<uint4*>(
      k_ws + ((size_t)h * (size_t)ne_pad + (size_t)(key0 + key)) *
                 (size_t)head_dim +
             d) = kv.u;
  *reinterpret_cast<uint4*>(&tile[key][lane_d * 8]) = vv.u;
  __syncthreads();

  const int dim = tid & 63;
  const int kc = tid >> 6;  // 4 groups of 8 keys
  sb_half8 tv;
#pragma unroll
  for (int i = 0; i < 8; ++i) tv.h[i] = tile[kc * 8 + i][dim];
  *reinterpret_cast<uint4*>(v_ws + ((size_t)h * (size_t)head_dim +
                                    (size_t)(d0 + dim)) *
                                       (size_t)ne_pad +
                                   (size_t)(key0 + kc * 8)) = tv.u;
}

// ---------------------------------------------------------------------------
// Attention kernel.
//
// Grid: x = ceil(n / BR) query tiles, y = kv heads. Block: UNITS*NW*32 threads.
//   unit u  = 16 consecutive query rows in one GQA group (rc = u / GQA,
//             qg = u % GQA, query head hq = kvh*GQA + qg).
//   warp w  = dim_slice (w / UNITS) and unit (w % UNITS).
// Phase 1 (S, online softmax) is computed by the slice-0 warps; phase 2 (P*V)
// by all NW slices of the unit.
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int GQA, int BR, int BK, int NW, bool F16ACC>
__global__ void __launch_bounds__(1024) boom_fattn_attn_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k_ws,
    const __half* __restrict__ v_ws, const uint64_t* __restrict__ start_ptr,
    uint64_t n_slots, uint64_t window, int n, int n_heads_q, int n_heads_kv,
    int ne_pad, float scale, float* __restrict__ out) {
  constexpr int NQ = BR / 16;
  constexpr int UNITS = NQ * GQA;
  constexpr int WARPS = UNITS * NW;
  constexpr int NT = (HEAD_DIM / NW) / 8;  // 8-dim output tiles per slice
  constexpr int BKP = BK + 8;              // 16-byte-aligned V tile stride
  constexpr int SUBS = BK / 16;
  constexpr int K_ELEMS = 2 * BK * HEAD_DIM;
  constexpr int V_ELEMS = 2 * HEAD_DIM * BKP;
  constexpr int P_ELEMS = UNITS * 16 * BK;
  constexpr int RS_ELEMS = UNITS * SUBS * 16;

  extern __shared__ __half smem[];
  __half* kbuf = smem;
  __half* vbuf = kbuf + K_ELEMS;
  __half* pbuf = vbuf + V_ELEMS;
  float* rsbuf = reinterpret_cast<float*>(pbuf + P_ELEMS);
  float* lbuf = rsbuf + RS_ELEMS;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int dim_slice = warp / UNITS;
  const int unit = warp - dim_slice * UNITS;
  const int qg = unit % GQA;
  const int rc = unit / GQA;
  const int kvh = blockIdx.y;
  const int hq = kvh * GQA + qg;
  const int qr = blockIdx.x * BR + rc * 16;
  const int dim0 = dim_slice * (HEAD_DIM / NW);

  const int p_base = unit * 16 * BK;
  const int rs_base = unit * SUBS * 16;

  const uint64_t start = *start_ptr;
  const uint64_t key_end = start + (uint64_t)n - 1;
  const int iq0 = qr + g;
  const int iq1 = qr + g + 8;
  const uint64_t p0 = start + (uint64_t)iq0;
  const uint64_t p1 = start + (uint64_t)iq1;
  uint64_t low0 = 0, low1 = 0;
  if (window > 0) {
    low0 = (p0 + 1 >= window) ? p0 - (window - 1) : 0;
    low1 = (p1 + 1 >= window) ? p1 - (window - 1) : 0;
  }

  // Key tile range: from the block's sliding-window low (absolute, aligned
  // down to BK so tile boundaries never depend on the chunk split) to the last
  // key of the chunk. Individual queries mask keys outside their own window.
  //
  // The range must be uniform across the block: the tile loop carries block-
  // wide barriers and block-wide cp.async staging. It is therefore derived
  // from the block's *first* query, not this warp's qr (which differs between
  // the rc groups of a BR=32 block). The extra leading tiles a later unit gets
  // are entirely below its window low, hence fully masked: m stays put, so the
  // rescale factor is exp(0)==1 and P is exactly 0, which makes them exact
  // no-ops -- including the bit pattern, so chunk-split invariance is intact.
  const uint64_t pblk = start + (uint64_t)blockIdx.x * (uint64_t)BR;
  uint64_t key_start = 0;
  if (window > 0) key_start = (pblk + 1 >= window) ? pblk - (window - 1) : 0;
  const uint64_t j0_start = key_start - key_start % (uint64_t)BK;
  // Last key any query in this block can see: its own last row (causal) or the
  // chunk end, whichever comes first. Without this cap every block would run
  // its key loop to the end of the chunk: causal prefill would do the full
  // n^2 instead of n^2/2, and a windowed block would scan the whole chunk
  // instead of its window. The dropped tiles are entirely above the block's
  // last query, hence fully masked -- m does not move, so rs == 1 and P == 0
  // and dropping them is bit-exact.
  const uint64_t last_q = start + (uint64_t)blockIdx.x * (uint64_t)BR + (uint64_t)(BR - 1);
  const uint64_t key_last = last_q < key_end ? last_q : key_end;
  const uint64_t base = sb_span_base(start, window);
  const uint64_t rel_first = j0_start - base;

  auto stage = [&](int b, uint64_t rel0) {
    __half* kd = kbuf + b * K_ELEMS / 2;
    for (int i = threadIdx.x; i < BK * HEAD_DIM / 8; i += WARPS * 32) {
      const int e = i * 8;
      const int key = e / HEAD_DIM;
      const int dd = e - key * HEAD_DIM;
      sb_cp_async_16(kd + key * HEAD_DIM + dd,
                     k_ws + ((size_t)kvh * (size_t)ne_pad +
                             (size_t)(rel0 + (uint64_t)key)) *
                                (size_t)HEAD_DIM +
                         dd);
    }
    __half* vd = vbuf + b * V_ELEMS / 2;
    for (int i = threadIdx.x; i < HEAD_DIM * BK / 8; i += WARPS * 32) {
      const int e = i * 8;
      const int dim = e / BK;
      const int koff = e - dim * BK;
      sb_cp_async_16(vd + dim * BKP + koff,
                     v_ws + ((size_t)kvh * (size_t)HEAD_DIM + (size_t)dim) *
                                (size_t)ne_pad +
                         rel0 + (uint64_t)koff);
    }
  };

  using ACC_T = std::conditional_t<F16ACC, uint32_t, float>;
  constexpr int ACC_N = F16ACC ? 2 : 4;
  ACC_T o[NT][ACC_N] = {};

  float m[2] = {-1e30f, -1e30f};
  float l[2] = {0.0f, 0.0f};

  stage(0, rel_first);
  sb_cp_commit();
  int buf = 0;
  for (uint64_t j0 = j0_start; j0 <= key_last; j0 += (uint64_t)BK) {
    __syncthreads();  // previous tile's compute finished; p_buf reusable
    const bool has_next = (j0 + (uint64_t)BK <= key_last);
    if (has_next) {
      stage(buf ^ 1, j0 + (uint64_t)BK - base);
      sb_cp_commit();
      sb_cp_wait<1>();
    } else {
      sb_cp_wait<0>();
    }
    __syncthreads();  // staged tile visible to every warp

    const __half* kt = kbuf + buf * K_ELEMS / 2;
    const __half* vt = vbuf + buf * V_ELEMS / 2;

    // ---- phase 1: S = Q K^T, mask, online softmax, P/rs to smem ----------
    if (dim_slice == 0) {
      for (int key_off = 0; key_off < BK; key_off += 16) {
        float s0[4] = {0.f, 0.f, 0.f, 0.f};
        float s1[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
          uint32_t aq[4], bb[2];
          sb_load_q(aq, q, hq, n_heads_q, HEAD_DIM, qr, k0, n);
          sb_load_k(bb, kt, HEAD_DIM, k0, key_off);
          sb_mma_f32(s0, aq, bb);
          sb_load_k(bb, kt, HEAD_DIM, k0, key_off + 8);
          sb_mma_f32(s1, aq, bb);
        }

        const uint64_t jb = j0 + (uint64_t)key_off + (uint64_t)t * 2;
        s0[0] = (jb > p0 || jb < low0) ? -1e30f : s0[0] * scale;
        s0[1] = (jb + 1 > p0 || jb + 1 < low0) ? -1e30f : s0[1] * scale;
        s1[0] = (jb + 8 > p0 || jb + 8 < low0) ? -1e30f : s1[0] * scale;
        s1[1] = (jb + 9 > p0 || jb + 9 < low0) ? -1e30f : s1[1] * scale;
        s0[2] = (jb > p1 || jb < low1) ? -1e30f : s0[2] * scale;
        s0[3] = (jb + 1 > p1 || jb + 1 < low1) ? -1e30f : s0[3] * scale;
        s1[2] = (jb + 8 > p1 || jb + 8 < low1) ? -1e30f : s1[2] * scale;
        s1[3] = (jb + 9 > p1 || jb + 9 < low1) ? -1e30f : s1[3] * scale;

        const float mx0 =
            sb_rmax4(fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1])));
        const float mx1 =
            sb_rmax4(fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3])));
        const float m_new0 = fmaxf(m[0], mx0);
        const float m_new1 = fmaxf(m[1], mx1);
        const float rs0 = expf(m[0] - m_new0);
        const float rs1 = expf(m[1] - m_new1);

        const float p00 = (s0[0] > -1e29f) ? expf(s0[0] - m_new0) : 0.0f;
        const float p01 = (s0[1] > -1e29f) ? expf(s0[1] - m_new0) : 0.0f;
        const float p08 = (s1[0] > -1e29f) ? expf(s1[0] - m_new0) : 0.0f;
        const float p09 = (s1[1] > -1e29f) ? expf(s1[1] - m_new0) : 0.0f;
        const float p10 = (s0[2] > -1e29f) ? expf(s0[2] - m_new1) : 0.0f;
        const float p11 = (s0[3] > -1e29f) ? expf(s0[3] - m_new1) : 0.0f;
        const float p18 = (s1[2] > -1e29f) ? expf(s1[2] - m_new1) : 0.0f;
        const float p19 = (s1[3] > -1e29f) ? expf(s1[3] - m_new1) : 0.0f;

        l[0] = l[0] * rs0 + sb_rsum4(p00 + p01 + p08 + p09);
        l[1] = l[1] * rs1 + sb_rsum4(p10 + p11 + p18 + p19);

        __half* pg = pbuf + p_base + g * BK + key_off;
        __half* pg8 = pg + 8 * BK;
        pg[t * 2] = __float2half_rn(p00);
        pg[t * 2 + 1] = __float2half_rn(p01);
        pg[t * 2 + 8] = __float2half_rn(p08);
        pg[t * 2 + 9] = __float2half_rn(p09);
        pg8[t * 2] = __float2half_rn(p10);
        pg8[t * 2 + 1] = __float2half_rn(p11);
        pg8[t * 2 + 8] = __float2half_rn(p18);
        pg8[t * 2 + 9] = __float2half_rn(p19);
        const int sub = key_off >> 4;
        rsbuf[rs_base + sub * 16 + g] = rs0;
        rsbuf[rs_base + sub * 16 + g + 8] = rs1;
        m[0] = m_new0;
        m[1] = m_new1;
      }
    }
    __syncthreads();

    // ---- phase 2: O = O * rs + P V, this warp's dim slice ----------------
    for (int key_off = 0; key_off < BK; key_off += 16) {
      const int sub = key_off >> 4;
      const float rs0 = rsbuf[rs_base + sub * 16 + g];
      const float rs1 = rsbuf[rs_base + sub * 16 + g + 8];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if constexpr (F16ACC) {
          const __half2 h0 = __float2half2_rn(rs0);
          const __half2 h1 = __float2half2_rn(rs1);
          o[i][0] = sb_as_u32(__hmul2(sb_as_h2(o[i][0]), h0));
          o[i][1] = sb_as_u32(__hmul2(sb_as_h2(o[i][1]), h1));
        } else {
          o[i][0] *= rs0;
          o[i][1] *= rs0;
          o[i][2] *= rs1;
          o[i][3] *= rs1;
        }
      }
      const __half* pg = pbuf + p_base + g * BK + key_off;
      const __half* pg8 = pg + 8 * BK;
      uint32_t ap[4];
      ap[0] = *reinterpret_cast<const uint32_t*>(pg + t * 2);
      ap[1] = *reinterpret_cast<const uint32_t*>(pg8 + t * 2);
      ap[2] = *reinterpret_cast<const uint32_t*>(pg + t * 2 + 8);
      ap[3] = *reinterpret_cast<const uint32_t*>(pg8 + t * 2 + 8);
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        uint32_t bv[2];
        sb_load_v(bv, vt, BKP, dim0 + i * 8, key_off);
        if constexpr (F16ACC) {
          sb_mma_f16(o[i], ap, bv);
        } else {
          sb_mma_f32(o[i], ap, bv);
        }
      }
    }
    buf ^= 1;
  }

  if (dim_slice == 0) {
    lbuf[unit * 16 + g] = l[0];
    lbuf[unit * 16 + g + 8] = l[1];
  }
  __syncthreads();

  const float inv0 = 1.0f / lbuf[unit * 16 + g];
  const float inv1 = 1.0f / lbuf[unit * 16 + g + 8];
  const size_t row_stride = (size_t)n_heads_q * HEAD_DIM;
  const int dd0 = dim0 + t * 2;
  float* out0 = out + (size_t)iq0 * row_stride + (size_t)hq * HEAD_DIM + dd0;
  float* out1 = out + (size_t)iq1 * row_stride + (size_t)hq * HEAD_DIM + dd0;
  const bool w0 = iq0 < n;
  const bool w1 = iq1 < n;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    if constexpr (F16ACC) {
      const float2 f0 = __half22float2(sb_as_h2(o[i][0]));
      const float2 f1 = __half22float2(sb_as_h2(o[i][1]));
      if (w0) {
        out0[i * 8] = f0.x * inv0;
        out0[i * 8 + 1] = f0.y * inv0;
      }
      if (w1) {
        out1[i * 8] = f1.x * inv1;
        out1[i * 8 + 1] = f1.y * inv1;
      }
    } else {
      if (w0) {
        out0[i * 8] = o[i][0] * inv0;
        out0[i * 8 + 1] = o[i][1] * inv0;
      }
      if (w1) {
        out1[i * 8] = o[i][2] * inv1;
        out1[i * 8 + 1] = o[i][3] * inv1;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Query-resident variant: the d512 f16 default (SONICBOOM_BOOM_FATTN_QREG=1)
// and the d256 f16 default (SONICBOOM_BOOM_FATTN_D256=1; NW and the tile
// geometry differ per head width). The 0 modes select the previous kernel
// above as the explicit fallback.
//
// The baseline kernel re-reads each unit's Q slice from global memory once per
// key tile: 16 KB per unit per 16-key tile, 128 KB per block per tile, which
// does not fit L1 and is the kernel's largest single cost (a constants-only Q
// diagnostic -- wrong results, timing only -- raised pp8192 from 1338 to 1848
// tok/s before f16 Q materialization). Here every warp keeps the
// HEAD_DIM/NW-dim slice it owns in registers for the whole key-tile loop, so Q
// is read from global exactly once per call, and phase 1 is split across the
// NW dim slices of the unit: each warp computes a partial S over its own dims,
// the partials are reduced through shared memory in a fixed slice order, and
// the slice-0 warp applies the mask and the online softmax (writing P and the
// rescale factors exactly as before). Phase 2 is unchanged.
//
// The smaller block is what makes the registers available: 512 threads with
// launch_bounds(512) lets ptxas use up to 128 registers instead of the 64 a
// 1024-thread block allows. The cost is half the resident warps per SM.
//
// The reduction changes the summation order of the QK dot product (NW partial
// chains summed in slice order) but not its determinism: the order depends
// only on the absolute tile, never on the chunk split, so the bitwise
// chunk-split invariance still holds.
// ---------------------------------------------------------------------------

template <int HEAD_DIM, int GQA, int BR, int BK, int NW, bool F16ACC,
          bool QRES = true>
__global__ void __launch_bounds__((BR / 16) * GQA * NW * 32)
boom_fattn_attn_qreg_kernel(
    const __half* __restrict__ q, const __half* __restrict__ k_ws,
    const __half* __restrict__ v_ws, const uint64_t* __restrict__ start_ptr,
    uint64_t n_slots, uint64_t window, int n, int n_heads_q, int n_heads_kv,
    int ne_pad, float scale, float* __restrict__ out) {
  constexpr int UNITS = (BR / 16) * GQA;
  constexpr int WARPS = UNITS * NW;
  constexpr int NT = (HEAD_DIM / NW) / 8;  // 8-dim output tiles per slice
  constexpr int KSTEPS = (HEAD_DIM / NW) / 16;  // Q k-fragments per slice
  constexpr int BKP = BK + 8;
  constexpr int SUBS = BK / 16;
  constexpr int K_ELEMS = 2 * BK * HEAD_DIM;
  constexpr int V_ELEMS = 2 * HEAD_DIM * BKP;
  constexpr int P_ELEMS = UNITS * 16 * BK;
  constexpr int RS_ELEMS = UNITS * SUBS * 16;

  extern __shared__ __half smem[];
  __half* kbuf = smem;
  __half* vbuf = kbuf + K_ELEMS;
  __half* pbuf = vbuf + V_ELEMS;
  float* rsbuf = reinterpret_cast<float*>(pbuf + P_ELEMS);
  float* lbuf = rsbuf + RS_ELEMS;
  // UNITS * NW * SUBS * 256 f32 partial-S planes; the host side sizes them.
  float* psbuf = lbuf + UNITS * 16;

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int dim_slice = warp / UNITS;
  const int unit = warp - dim_slice * UNITS;
  const int qg = unit % GQA;
  const int rc = unit / GQA;
  const int kvh = blockIdx.y;
  const int hq = kvh * GQA + qg;
  const int qr = blockIdx.x * BR + rc * 16;
  const int dim0 = dim_slice * (HEAD_DIM / NW);

  const int p_base = unit * 16 * BK;
  const int rs_base = unit * SUBS * 16;

  const uint64_t start = *start_ptr;
  const uint64_t key_end = start + (uint64_t)n - 1;
  const int iq0 = qr + g;
  const int iq1 = qr + g + 8;
  const uint64_t p0 = start + (uint64_t)iq0;
  const uint64_t p1 = start + (uint64_t)iq1;
  uint64_t low0 = 0, low1 = 0;
  if (window > 0) {
    low0 = (p0 + 1 >= window) ? p0 - (window - 1) : 0;
    low1 = (p1 + 1 >= window) ? p1 - (window - 1) : 0;
  }

  // Same block-uniform tile range as the baseline kernel (see its comment).
  const uint64_t pblk = start + (uint64_t)blockIdx.x * (uint64_t)BR;
  uint64_t key_start = 0;
  if (window > 0) key_start = (pblk + 1 >= window) ? pblk - (window - 1) : 0;
  const uint64_t j0_start = key_start - key_start % (uint64_t)BK;
  const uint64_t last_q = start + (uint64_t)blockIdx.x * (uint64_t)BR + (uint64_t)(BR - 1);
  const uint64_t key_last = last_q < key_end ? last_q : key_end;
  const uint64_t base = sb_span_base(start, window);
  const uint64_t rel_first = j0_start - base;

  // Q, once per call: this warp's dim slice of its unit, f16, 4 words per
  // 16-dim fragment. QRES=false is the control variant: same split warp
  // mapping and block, Q re-read from global once per key tile (isolates the
  // effect of holding Q in registers from the effect of splitting QK^T).
  uint32_t qreg[KSTEPS][4];
  if constexpr (QRES) {
#pragma unroll
    for (int j = 0; j < KSTEPS; ++j)
      sb_load_q(qreg[j], q, hq, n_heads_q, HEAD_DIM, qr, dim0 + j * 16, n);
  }

  auto stage = [&](int b, uint64_t rel0) {
    __half* kd = kbuf + b * K_ELEMS / 2;
    for (int i = threadIdx.x; i < BK * HEAD_DIM / 8; i += WARPS * 32) {
      const int e = i * 8;
      const int key = e / HEAD_DIM;
      const int dd = e - key * HEAD_DIM;
      sb_cp_async_16(kd + key * HEAD_DIM + dd,
                     k_ws + ((size_t)kvh * (size_t)ne_pad +
                             (size_t)(rel0 + (uint64_t)key)) *
                                (size_t)HEAD_DIM +
                         dd);
    }
    __half* vd = vbuf + b * V_ELEMS / 2;
    for (int i = threadIdx.x; i < HEAD_DIM * BK / 8; i += WARPS * 32) {
      const int e = i * 8;
      const int dim = e / BK;
      const int koff = e - dim * BK;
      sb_cp_async_16(vd + dim * BKP + koff,
                     v_ws + ((size_t)kvh * (size_t)HEAD_DIM + (size_t)dim) *
                                (size_t)ne_pad +
                         rel0 + (uint64_t)koff);
    }
  };

  using ACC_T = std::conditional_t<F16ACC, uint32_t, float>;
  constexpr int ACC_N = F16ACC ? 2 : 4;
  ACC_T o[NT][ACC_N] = {};

  float m[2] = {-1e30f, -1e30f};
  float l[2] = {0.0f, 0.0f};

  stage(0, rel_first);
  sb_cp_commit();
  int buf = 0;
  for (uint64_t j0 = j0_start; j0 <= key_last; j0 += (uint64_t)BK) {
    __syncthreads();  // previous tile's compute finished; p_buf reusable
    const bool has_next = (j0 + (uint64_t)BK <= key_last);
    if (has_next) {
      stage(buf ^ 1, j0 + (uint64_t)BK - base);
      sb_cp_commit();
      sb_cp_wait<1>();
    } else {
      sb_cp_wait<0>();
    }
    __syncthreads();  // staged tile visible to every warp

    const __half* kt = kbuf + buf * K_ELEMS / 2;
    const __half* vt = vbuf + buf * V_ELEMS / 2;

    // ---- phase 1a: this slice's partial S = Q_slice K^T -------------------
    for (int key_off = 0; key_off < BK; key_off += 16) {
      const int sub = key_off >> 4;
      float s0[4] = {0.f, 0.f, 0.f, 0.f};
      float s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int j = 0; j < KSTEPS; ++j) {
        uint32_t aql[4];
        const uint32_t* aq;
        if constexpr (QRES) {
          aq = qreg[j];
        } else {
          sb_load_q(aql, q, hq, n_heads_q, HEAD_DIM, qr, dim0 + j * 16, n);
          aq = aql;
        }
        uint32_t bb[2];
        sb_load_k(bb, kt, HEAD_DIM, dim0 + j * 16, key_off);
        sb_mma_f32(s0, aq, bb);
        sb_load_k(bb, kt, HEAD_DIM, dim0 + j * 16, key_off + 8);
        sb_mma_f32(s1, aq, bb);
      }
      float* p = psbuf + ((unit * NW + dim_slice) * SUBS + sub) * 256;
      p[g * 16 + t * 2 + 0] = s0[0];
      p[g * 16 + t * 2 + 1] = s0[1];
      p[g * 16 + t * 2 + 8] = s1[0];
      p[g * 16 + t * 2 + 9] = s1[1];
      p[(g + 8) * 16 + t * 2 + 0] = s0[2];
      p[(g + 8) * 16 + t * 2 + 1] = s0[3];
      p[(g + 8) * 16 + t * 2 + 8] = s1[2];
      p[(g + 8) * 16 + t * 2 + 9] = s1[3];
    }
    __syncthreads();  // partial S visible to the slice-0 warp

    // ---- phase 1b: reduce the slices, mask, online softmax, P/rs ----------
    if (dim_slice == 0) {
      for (int key_off = 0; key_off < BK; key_off += 16) {
        const int sub = key_off >> 4;
        float s0[4] = {0.f, 0.f, 0.f, 0.f};
        float s1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int sl = 0; sl < NW; ++sl) {
          const float* p = psbuf + ((unit * NW + sl) * SUBS + sub) * 256;
          s0[0] += p[g * 16 + t * 2 + 0];
          s0[1] += p[g * 16 + t * 2 + 1];
          s1[0] += p[g * 16 + t * 2 + 8];
          s1[1] += p[g * 16 + t * 2 + 9];
          s0[2] += p[(g + 8) * 16 + t * 2 + 0];
          s0[3] += p[(g + 8) * 16 + t * 2 + 1];
          s1[2] += p[(g + 8) * 16 + t * 2 + 8];
          s1[3] += p[(g + 8) * 16 + t * 2 + 9];
        }

        const uint64_t jb = j0 + (uint64_t)key_off + (uint64_t)t * 2;
        s0[0] = (jb > p0 || jb < low0) ? -1e30f : s0[0] * scale;
        s0[1] = (jb + 1 > p0 || jb + 1 < low0) ? -1e30f : s0[1] * scale;
        s1[0] = (jb + 8 > p0 || jb + 8 < low0) ? -1e30f : s1[0] * scale;
        s1[1] = (jb + 9 > p0 || jb + 9 < low0) ? -1e30f : s1[1] * scale;
        s0[2] = (jb > p1 || jb < low1) ? -1e30f : s0[2] * scale;
        s0[3] = (jb + 1 > p1 || jb + 1 < low1) ? -1e30f : s0[3] * scale;
        s1[2] = (jb + 8 > p1 || jb + 8 < low1) ? -1e30f : s1[2] * scale;
        s1[3] = (jb + 9 > p1 || jb + 9 < low1) ? -1e30f : s1[3] * scale;

        const float mx0 =
            sb_rmax4(fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1])));
        const float mx1 =
            sb_rmax4(fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3])));
        const float m_new0 = fmaxf(m[0], mx0);
        const float m_new1 = fmaxf(m[1], mx1);
        const float rs0 = expf(m[0] - m_new0);
        const float rs1 = expf(m[1] - m_new1);

        const float p00 = (s0[0] > -1e29f) ? expf(s0[0] - m_new0) : 0.0f;
        const float p01 = (s0[1] > -1e29f) ? expf(s0[1] - m_new0) : 0.0f;
        const float p08 = (s1[0] > -1e29f) ? expf(s1[0] - m_new0) : 0.0f;
        const float p09 = (s1[1] > -1e29f) ? expf(s1[1] - m_new0) : 0.0f;
        const float p10 = (s0[2] > -1e29f) ? expf(s0[2] - m_new1) : 0.0f;
        const float p11 = (s0[3] > -1e29f) ? expf(s0[3] - m_new1) : 0.0f;
        const float p18 = (s1[2] > -1e29f) ? expf(s1[2] - m_new1) : 0.0f;
        const float p19 = (s1[3] > -1e29f) ? expf(s1[3] - m_new1) : 0.0f;

        l[0] = l[0] * rs0 + sb_rsum4(p00 + p01 + p08 + p09);
        l[1] = l[1] * rs1 + sb_rsum4(p10 + p11 + p18 + p19);

        __half* pg = pbuf + p_base + g * BK + key_off;
        __half* pg8 = pg + 8 * BK;
        pg[t * 2] = __float2half_rn(p00);
        pg[t * 2 + 1] = __float2half_rn(p01);
        pg[t * 2 + 8] = __float2half_rn(p08);
        pg[t * 2 + 9] = __float2half_rn(p09);
        pg8[t * 2] = __float2half_rn(p10);
        pg8[t * 2 + 1] = __float2half_rn(p11);
        pg8[t * 2 + 8] = __float2half_rn(p18);
        pg8[t * 2 + 9] = __float2half_rn(p19);
        rsbuf[rs_base + sub * 16 + g] = rs0;
        rsbuf[rs_base + sub * 16 + g + 8] = rs1;
        m[0] = m_new0;
        m[1] = m_new1;
      }
    }
    __syncthreads();

    // ---- phase 2: O = O * rs + P V, this warp's dim slice ----------------
    for (int key_off = 0; key_off < BK; key_off += 16) {
      const int sub = key_off >> 4;
      const float rs0 = rsbuf[rs_base + sub * 16 + g];
      const float rs1 = rsbuf[rs_base + sub * 16 + g + 8];
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        if constexpr (F16ACC) {
          const __half2 h0 = __float2half2_rn(rs0);
          const __half2 h1 = __float2half2_rn(rs1);
          o[i][0] = sb_as_u32(__hmul2(sb_as_h2(o[i][0]), h0));
          o[i][1] = sb_as_u32(__hmul2(sb_as_h2(o[i][1]), h1));
        } else {
          o[i][0] *= rs0;
          o[i][1] *= rs0;
          o[i][2] *= rs1;
          o[i][3] *= rs1;
        }
      }
      const __half* pg = pbuf + p_base + g * BK + key_off;
      const __half* pg8 = pg + 8 * BK;
      uint32_t ap[4];
      ap[0] = *reinterpret_cast<const uint32_t*>(pg + t * 2);
      ap[1] = *reinterpret_cast<const uint32_t*>(pg8 + t * 2);
      ap[2] = *reinterpret_cast<const uint32_t*>(pg + t * 2 + 8);
      ap[3] = *reinterpret_cast<const uint32_t*>(pg8 + t * 2 + 8);
#pragma unroll
      for (int i = 0; i < NT; ++i) {
        uint32_t bv[2];
        sb_load_v(bv, vt, BKP, dim0 + i * 8, key_off);
        if constexpr (F16ACC) {
          sb_mma_f16(o[i], ap, bv);
        } else {
          sb_mma_f32(o[i], ap, bv);
        }
      }
    }
    buf ^= 1;
  }

  if (dim_slice == 0) {
    lbuf[unit * 16 + g] = l[0];
    lbuf[unit * 16 + g + 8] = l[1];
  }
  __syncthreads();

  const float inv0 = 1.0f / lbuf[unit * 16 + g];
  const float inv1 = 1.0f / lbuf[unit * 16 + g + 8];
  const size_t row_stride = (size_t)n_heads_q * HEAD_DIM;
  const int dd0 = dim0 + t * 2;
  float* out0 = out + (size_t)iq0 * row_stride + (size_t)hq * HEAD_DIM + dd0;
  float* out1 = out + (size_t)iq1 * row_stride + (size_t)hq * HEAD_DIM + dd0;
  const bool w0 = iq0 < n;
  const bool w1 = iq1 < n;
#pragma unroll
  for (int i = 0; i < NT; ++i) {
    if constexpr (F16ACC) {
      const float2 f0 = __half22float2(sb_as_h2(o[i][0]));
      const float2 f1 = __half22float2(sb_as_h2(o[i][1]));
      if (w0) {
        out0[i * 8] = f0.x * inv0;
        out0[i * 8 + 1] = f0.y * inv0;
      }
      if (w1) {
        out1[i * 8] = f1.x * inv1;
        out1[i * 8 + 1] = f1.y * inv1;
      }
    } else {
      if (w0) {
        out0[i * 8] = o[i][0] * inv0;
        out0[i * 8 + 1] = o[i][1] * inv0;
      }
      if (w1) {
        out1[i * 8] = o[i][2] * inv1;
        out1[i * 8 + 1] = o[i][3] * inv1;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host side: workspace, launch.
// ---------------------------------------------------------------------------

// KV workspace, reused across layers and grown on demand. The gather kernel
// writes every element of the current span, so no zeroing is needed on growth.
// Under CUDA-Graph capture a growth would need an illegal cudaMalloc: decline
// and let the native path run (capture happens after warm-up, so this is a
// fallback, not the normal route).
void* g_q16 = nullptr;
size_t g_q16_bytes = 0;
void* g_kv_k = nullptr;
size_t g_kv_k_bytes = 0;
void* g_kv_v = nullptr;
size_t g_kv_v_bytes = 0;

void* grow(void*& buf, size_t& have, size_t need) {
  if (have >= need) return buf;
  if (sonicboom::cuda::sb_capturing()) return nullptr;
  if (buf) cudaFree(buf);
  buf = nullptr;
  have = 0;
  if (cudaMalloc(&buf, need) != cudaSuccess) return nullptr;
  have = need;
  return buf;
}

void debug_check(const char* stage, cudaStream_t stream) {
  static const bool on = std::getenv("SONICBOOM_ATTN_DEBUG") != nullptr;
  if (!on) return;
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[boom_fattn] %s: %s\n", stage, cudaGetErrorString(e));
  }
}

// SONICBOOM_BOOM_FATTN_QREG selects the phase-1 variant (d512, f16
// accumulator):
//   1 (default) split QK^T across the NW dim slices, Q resident in registers
//   0           the previous kernel: BR=32 / 1024-thread block, phase 1 on the
//               slice-0 warps, Q re-read from global per key tile
//               (explicit fallback; the only mode that changes the query tile
//               and block shape, so it is kept as the revert path)
//   2           split QK^T only, Q still re-read from global per key tile
//               (control: isolates the register residency from the split)
//   3           the baseline kernel at BR=16 / 512 threads, no split
//               (control: isolates the split from the smaller query tile and
//               block, which modes 1 and 2 also change)
// Modes 0-3 apply to head_dim 512 with the default f16 accumulator only; the
// d256 configuration has its own variant and switch below, and the
// f32-accumulator configuration keeps its previous kernel regardless of this
// setting.
// Read per call so the variants can be benchmarked in one process.
int qreg_mode() {
  const char* e = std::getenv("SONICBOOM_BOOM_FATTN_QREG");
  if (e == nullptr) return 1;
  if (std::string_view(e) == "0") return 0;
  if (std::string_view(e) == "1") return 1;
  if (std::string_view(e) == "2") return 2;
  if (std::string_view(e) == "3") return 3;
  return 1;
}

// SONICBOOM_BOOM_FATTN_D256 selects the d256 head-width kernel (window-agnostic;
// the windowed traversal is driven by the per-layer window argument):
//   1 (default) query-resident, QK^T split over 4 dim slices, BR16/BK16,
//               512-thread block (the d512 mode-1 schedule at this head width)
//   0           the previous kernel: BR32/BK32, phase 1 on the slice-0 warps,
//               Q re-read from global per key tile (explicit fallback)
//   2           split QK^T only, Q still re-read per key tile (control)
//   3           BR16/BK16 without the split (control: isolates the geometry)
//   4           as 1 but split over 8 dim slices (1024-thread block; control)
// The d256 f32-accumulator configuration keeps its previous kernel regardless
// of this setting.
int d256_mode() {
  const char* e = std::getenv("SONICBOOM_BOOM_FATTN_D256");
  if (e == nullptr) return 1;
  if (std::string_view(e) == "0") return 0;
  if (std::string_view(e) == "1") return 1;
  if (std::string_view(e) == "2") return 2;
  if (std::string_view(e) == "3") return 3;
  if (std::string_view(e) == "4") return 4;
  return 1;
}

template <int HEAD_DIM, int GQA, int BR, int BK, int NW, bool F16ACC,
          bool QREG = false, bool QRES = true>
bool run_chunk(const float* q, const float* k_batch, const float* v_batch,
               const float* k_cache, const float* v_cache,
               const uint64_t* start, uint64_t n_slots, int n, int n_heads_q,
               int n_heads_kv, uint64_t window, float scale, float* out,
               bool kv_f16) {
  constexpr int UNITS = (BR / 16) * GQA;
  constexpr int WARPS = UNITS * NW;
  constexpr int BKP = BK + 8;
  constexpr int SUBS = BK / 16;
  constexpr size_t smem =
      (size_t)(2 * BK * HEAD_DIM + 2 * HEAD_DIM * BKP + UNITS * 16 * BK) *
          sizeof(__half) +
      (size_t)(UNITS * SUBS * 16 + UNITS * 16) * sizeof(float) +
      (QREG ? (size_t)UNITS * NW * SUBS * 256 * sizeof(float) : 0);
  static_assert(WARPS == 16 || WARPS == 32,
                "boom_fattn launch geometry is a 512- or 1024-thread block");

  using KFn = void (*)(const __half*, const __half*, const __half*,
                       const uint64_t*, uint64_t, uint64_t, int, int, int,
                       int, float, float*);
  KFn kernel;
  if constexpr (QREG)
    kernel =
        boom_fattn_attn_qreg_kernel<HEAD_DIM, GQA, BR, BK, NW, F16ACC, QRES>;
  else
    kernel = boom_fattn_attn_kernel<HEAD_DIM, GQA, BR, BK, NW, F16ACC>;
  static bool attr_ok = false;
  if (!attr_ok) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem) != cudaSuccess)
      return false;
    attr_ok = true;
  }

  // Relative span bound. Windowed: the span starts at most window-1 keys
  // below the first query, and the span base is aligned down by up to 31 keys;
  // the last key tile stages up to BK-1 keys past the final key. Global: base
  // is 0 and the largest span is n_slots-1+n. Rounding to 32 keeps the gather
  // grid and the V rows 16-byte aligned.
  const uint64_t span =
      (window > 0 ? window - 1 + (uint64_t)n + 31 + (uint64_t)BK
                  : n_slots + (uint64_t)n + (uint64_t)BK);
  const int ne_pad = (int)((span + 31) / 32 * 32);
  const size_t need =
      (size_t)ne_pad * (size_t)n_heads_kv * (size_t)HEAD_DIM * sizeof(__half);
  __half* k_ws = static_cast<__half*>(grow(g_kv_k, g_kv_k_bytes, need));
  if (!k_ws) return false;
  __half* v_ws = static_cast<__half*>(grow(g_kv_v, g_kv_v_bytes, need));
  if (!v_ws) return false;

  cudaStream_t stream = sonicboom::cuda::sb_current_stream();

  // Both kernels derive the span base from the device start position.
  const dim3 gather_grid(HEAD_DIM / 64, (unsigned)(ne_pad / 32),
                         (unsigned)n_heads_kv);
  SB_NOTE_LAUNCH_TP("boom_fattn_gather_kernel", kv_f16 ? "cache=f16" : "cache=f32",
                    gather_grid.x * gather_grid.y * gather_grid.z, 256, 0,
                    stream);
  if (kv_f16) {
    boom_fattn_gather_kernel<__half><<<gather_grid, 256, 0, stream>>>(
        k_batch, v_batch, reinterpret_cast<const __half*>(k_cache),
        reinterpret_cast<const __half*>(v_cache), start, n_slots, n, n_heads_kv,
        HEAD_DIM, ne_pad, window, k_ws, v_ws);
  } else {
    boom_fattn_gather_kernel<float><<<gather_grid, 256, 0, stream>>>(
        k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_kv,
        HEAD_DIM, ne_pad, window, k_ws, v_ws);
  }
  debug_check("gather", stream);

  const int q_elems = n * n_heads_q * HEAD_DIM;
  __half* q16 = static_cast<__half*>(
      grow(g_q16, g_q16_bytes, (size_t)q_elems * sizeof(__half)));
  if (!q16) return false;
  const int qcast_blocks = (q_elems / 8 + 255) / 256;
  SB_NOTE_LAUNCH(boom_fattn_qcast_kernel, qcast_blocks, 256, 0, stream);
  boom_fattn_qcast_kernel<<<qcast_blocks, 256, 0, stream>>>(q, q_elems, q16);
  debug_check("qcast", stream);

  const dim3 grid((unsigned)((n + BR - 1) / BR), (unsigned)n_heads_kv);
  if constexpr (QREG)
    SB_NOTE_LAUNCH(boom_fattn_attn_qreg_kernel, grid.x * grid.y, WARPS * 32, smem,
                   stream);
  else
    SB_NOTE_LAUNCH(boom_fattn_attn_kernel, grid.x * grid.y, WARPS * 32, smem, stream);
  kernel<<<grid, WARPS * 32, smem, stream>>>(
      q16, k_ws, v_ws, start, n_slots, window, n, n_heads_q, n_heads_kv, ne_pad,
      scale, out);
  debug_check("attention", stream);
  return cudaGetLastError() == cudaSuccess;
}

// Read per call (not cached) so the mode can be switched between launches in
// one process; getenv is negligible next to an attention launch.
bool acc_f16() {
  const char* e = std::getenv("SONICBOOM_BOOM_FATTN_ACC");
  return !(e != nullptr && std::string_view(e) == "f32");
}

template <int HEAD_DIM, int GQA, int BR, int BK, int NW, bool F16ACC,
          bool QREG = false, bool QRES = true>
void describe(int* br, int* bk, int* block, size_t* smem) {
  constexpr int UNITS = (BR / 16) * GQA;
  constexpr int WARPS = UNITS * NW;
  constexpr int BKP = BK + 8;
  constexpr int SUBS = BK / 16;
  if (br) *br = BR;
  if (bk) *bk = BK;
  if (block) *block = WARPS * 32;
  if (smem)
    *smem = (size_t)(2 * BK * HEAD_DIM + 2 * HEAD_DIM * BKP +
                     UNITS * 16 * BK) *
                sizeof(__half) +
            (size_t)(UNITS * SUBS * 16 + UNITS * 16) * sizeof(float) +
            (QREG ? (size_t)UNITS * NW * SUBS * 256 * sizeof(float) : 0);
}

}  // namespace

bool flash_attention_boom_fattn_chunk_dev_pos(
    const float* q, const float* k_batch, const float* v_batch,
    const float* k_cache, const float* v_cache, const uint64_t* start,
    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* out, bool kv_f16,
    bool kv_q8) {
  if (kv_q8 || !q || !k_batch || !v_batch || !k_cache || !v_cache || !start ||
      !out)
    return false;
  if (n <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 || n_slots == 0) return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  const int gqa = n_heads_q / n_heads_kv;
  if (gqa != 4) return false;  // instantiated classes only
  // A 1024-thread block caps ptxas at 64 registers/thread, so the per-warp PV
  // output slice is kept at <= 16 mma tiles (32 accumulator registers). The
  // fp32 accumulator doubles that per tile, which halves the query tile for
  // d512: BR16 x 8 dim slices instead of BR32 x 4.
  if (head_dim == 512) {
    if (acc_f16()) {
      const int mode = qreg_mode();
      if (mode == 1)
        return run_chunk<512, 4, 16, 16, 4, true, true, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 2)
        return run_chunk<512, 4, 16, 16, 4, true, true, false>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 3)
        return run_chunk<512, 4, 16, 16, 4, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      return run_chunk<512, 4, 32, 16, 4, true>(
          q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
          n_heads_kv, sliding_window, scale, out, kv_f16);
    }
    return run_chunk<512, 4, 16, 16, 8, false>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  }
  if (head_dim == 256) {
    if (acc_f16()) {
      const int mode = d256_mode();
      if (mode == 0)
        return run_chunk<256, 4, 32, 32, 4, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 2)
        return run_chunk<256, 4, 16, 16, 4, true, true, false>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 3)
        return run_chunk<256, 4, 16, 16, 4, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 4)
        return run_chunk<256, 4, 16, 16, 8, true, true, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      return run_chunk<256, 4, 16, 16, 4, true, true, true>(
          q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
          n_heads_kv, sliding_window, scale, out, kv_f16);
    }
    return run_chunk<256, 4, 32, 32, 4, false>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  }
  return false;
}

bool flash_attention_boom_fattn_config(int head_dim, int gqa, int* br, int* bk,
                                   int* block, size_t* smem) {
  if (gqa != 4) return false;
  if (head_dim == 512) {
    if (acc_f16()) {
      const int mode = qreg_mode();
      if (mode == 1)
        describe<512, 4, 16, 16, 4, true, true, true>(br, bk, block, smem);
      else if (mode == 2)
        describe<512, 4, 16, 16, 4, true, true, false>(br, bk, block, smem);
      else if (mode == 3)
        describe<512, 4, 16, 16, 4, true>(br, bk, block, smem);
      else
        describe<512, 4, 32, 16, 4, true>(br, bk, block, smem);
    } else {
      describe<512, 4, 16, 16, 8, false>(br, bk, block, smem);
    }
    return true;
  }
  if (head_dim == 256) {
    if (acc_f16()) {
      const int mode = d256_mode();
      if (mode == 0)
        describe<256, 4, 32, 32, 4, true>(br, bk, block, smem);
      else if (mode == 2)
        describe<256, 4, 16, 16, 4, true, true, false>(br, bk, block, smem);
      else if (mode == 3)
        describe<256, 4, 16, 16, 4, true>(br, bk, block, smem);
      else if (mode == 4)
        describe<256, 4, 16, 16, 8, true, true, true>(br, bk, block, smem);
      else
        describe<256, 4, 16, 16, 4, true, true, true>(br, bk, block, smem);
    } else {
      describe<256, 4, 32, 32, 4, false>(br, bk, block, smem);
    }
    return true;
  }
  return false;
}

}  // namespace sonicboom::nn::cuda
