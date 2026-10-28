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

// boom_fattn's strict-order path: the DeepSeek-R1-Distill-Qwen 1.5B prefill
// geometry (12 query heads, 2 KV heads, head_dim 128, causal, global).
//
// Tensor cores cannot serve this class: the strict profile's acceptance gate is
// a byte-for-byte differential against the scalar kernel (the 50/50 token gate
// follows from it), and a tensor-core reduction changes the FP association.
// The class therefore keeps the two strict kernels, moved here unchanged as
// part of boom_fattn so that boom_fattn is the only prefill-attention
// implementation:
//
//   flash_attention_chunk_pos_kernel   scalar fp32 reference (one warp per
//                                       (query, head), 32-key softmax tiles)
//   flash_attention_chunk_exact_kernel strict-order tiled kernel; same per-key
//                                       dot order, 32-lane butterfly, softmax
//                                       tiles and ascending PV FMA chain, so it
//                                       is byte-identical to the scalar arm
//
// The exact kernel is the default; SONICBOOM_ATTN_D128_EXACT=0 rolls back to
// the scalar arm (the differential test pins both). q8 KV storage is served by
// the scalar instantiation only — the exact kernel has no q8 staging, and the
// q8 cache dequantizes to the same values.

#include <sonicboom/nn/cuda_elementwise.h>

#include "cuda_resident.h"
#include "cuda_stream.h"

#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <type_traits>

namespace sonicboom::nn::cuda {

namespace {

// Current stream for device-resident kernels (see cuda_stream.h).
inline cudaStream_t cur_stream() { return ::sonicboom::cuda::sb_current_stream(); }

constexpr int FA_BK = 32;  // keys per online-softmax tile


// --- KV-cache element type --------------------------------------------------
//
// The KV cache is f32 under the strict profile and f16 under standard. The
// values were fp16-rounded before either storage form existed (the prep kernels
// cast K and V through f16), so f16 storage is exact: widening a stored f16 is
// the same value the f32 cache held. Every kernel that reads or writes the
// cache goes through these two helpers and is templated on the element type;
// the launch sites pick the instantiation from the cache's declared dtype.
__device__ inline float kv_load(const float* p) { return *p; }
__device__ inline float kv_load(const __half* p) { return __half2float(*p); }
// Batch (chunk) K/V rows are always f32. Under f16 KV storage the cache append
// rounds them to f16, so attention must read them through the same rounding:
// otherwise a fresh prefill (all keys from the batch) sees unrounded values
// while a cache-reuse prefill sees rounded ones, and the two take different
// tokens. The strict (f32) instantiation keeps the raw value.
template <typename KV>
__device__ inline float kv_load_batch(const float* p) {
  if constexpr (std::is_same_v<KV, float>) return *p;
  return __half2float(__float2half_rn(*p));
}
__device__ inline void kv_store(float* p, float v) { *p = v; }
__device__ inline void kv_store(__half* p, float v) {
  *p = __float2half_rn(v);
}

// q8 storage: a row is `head_dim` int8 values followed by one f32 scale (the
// magnitude the int8 values were divided by, i.e. max|x| / 127), so a row is
// addressed in bytes and the stride is head_dim + 4. kv_row returns row
// `index` of a cache laid out [n_slots, n_heads_kv, head_dim]; kv_load_at reads
// element i of a row through the cache's element type, dequantizing q8.
// The scale is the same for every element of the row, so a loop reading a
// whole row hoists the load; `head_dim` is a kernel argument or a compile-time
// constant at every call site.
template <typename KV>
__device__ inline const KV *kv_row(const KV *cache, uint64_t index,
                                   int head_dim) {
  if constexpr (std::is_same_v<KV, int8_t>)
    return cache + index * (uint64_t)(head_dim + 4);
  return cache + index * (uint64_t)head_dim;
}
template <typename KV>
__device__ inline float kv_load_at(const KV *row, int i, int head_dim) {
  if constexpr (std::is_same_v<KV, int8_t>) {
    return float(row[i]) * *reinterpret_cast<const float *>(row + head_dim);
  } else {
    return kv_load(row + i);
  }
}


// Device-position variant of flash_attention_chunk_kernel: `start` is a DEVICE
// uint64_t read in-kernel, so the chunk's absolute start position is an input to
// a capturable CUDA Graph rather than a value baked into the node at capture
// time. Identical numerics otherwise (the launch config is fixed by n/head
// shapes, never by start).
//
// HEAD_DIM_ = 0 is the generic instantiation (the original kernel). A nonzero
// value is the head_dim=128 specialization: the four per-lane dimensions are a
// compile-time count, so the invariant Q row can be held in registers across
// the key loop instead of being re-read through L1 on every visit (B0a). The
// per-key values, accumulation order, reduction tree and softmax are unchanged.
template <typename KV, int HEAD_DIM_ = 0>
__global__ void flash_attention_chunk_pos_kernel(
    const float* __restrict__ q, const float* __restrict__ k_batch,
    const float* __restrict__ v_batch, const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache, const uint64_t* __restrict__ start_ptr,
    uint64_t n_slots, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* __restrict__ out) {
  extern __shared__ float s_sh[];  // [n_heads_q * FA_BK]
  const int lane = threadIdx.x & 31;
  const int h = threadIdx.x >> 5;  // warp == query head
  const int iq = blockIdx.x;       // query position within the chunk (0-based)
  if (h >= n_heads_q) return;
  const int kv_h = h * n_heads_kv / n_heads_q;
  const uint64_t start = *start_ptr;
  const uint64_t p = start + (uint64_t)iq;  // absolute query position
  const float* qrow = q + ((uint64_t)iq * n_heads_q + h) * head_dim;
  float* tile = s_sh + h * FA_BK;

  // B0a (head_dim=128): Q is warp-invariant for the whole kernel, but the
  // compiler does not hoist it across the key loop. Hold this lane's four
  // floats in registers; the loads they replace are the same values.
  constexpr int kQRegs = (HEAD_DIM_ > 0) ? (HEAD_DIM_ / 32) : 1;
  float qreg[kQRegs];
  if constexpr (HEAD_DIM_ > 0) {
    #pragma unroll
    for (int i = 0; i < kQRegs; ++i) qreg[i] = qrow[lane + 32 * i];
  }

  float acc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  float m = -1e30f, l = 0.0f;

  const uint64_t start_key =
      (sliding_window > 0 && p >= sliding_window) ? p - sliding_window + 1 : 0;

  for (uint64_t j0 = start_key; j0 < p + 1; j0 += FA_BK) {
    const uint64_t jend = (p + 1 < j0 + FA_BK) ? (p + 1) : (j0 + FA_BK);
    const int jj_count = int(jend - j0);

    for (int jj = 0; jj < jj_count; ++jj) {
      const uint64_t j = j0 + (uint64_t)jj;
      float dot = 0.0f;
      if constexpr (std::is_same_v<KV, float>) {
        // Strict (f32) path: one pointer-select and one dot loop, textually the
        // pre-profile kernel. Any restructuring here changes FP contraction and
        // the fp32 result, so keep it verbatim.
        const KV* krow;
        if (j < start) {
          const uint64_t slot = j % n_slots;
          krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
        } else {
          const uint64_t bi = j - start;
          krow = k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        }
        if constexpr (HEAD_DIM_ > 0) {
          #pragma unroll
          for (int i = 0; i < kQRegs; ++i)
            dot += qreg[i] * krow[lane + 32 * i];
        } else {
          for (int d = lane; d < head_dim; d += 32) dot += qrow[d] * krow[d];
        }
      } else if (j < start) {
        const uint64_t slot = j % n_slots;
        const KV* krow =
            kv_row(k_cache, slot * (uint64_t)n_heads_kv + kv_h, head_dim);
        if constexpr (HEAD_DIM_ > 0) {
          #pragma unroll
          for (int i = 0; i < kQRegs; ++i)
            dot += qreg[i] * kv_load_at(krow, lane + 32 * i, head_dim);
        } else {
          for (int d = lane; d < head_dim; d += 32)
            dot += qrow[d] * kv_load_at(krow, d, head_dim);
        }
      } else {
        const uint64_t bi = j - start;
        const float* krow =
            k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        if constexpr (HEAD_DIM_ > 0) {
          #pragma unroll
          for (int i = 0; i < kQRegs; ++i)
            dot += qreg[i] * kv_load_batch<KV>(krow + lane + 32 * i);
        } else {
          for (int d = lane; d < head_dim; d += 32)
            dot += qrow[d] * kv_load_batch<KV>(krow + d);
        }
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, o);
      if (lane == 0) tile[jj] = dot * scale;
    }
    __syncwarp();

    float tile_max = -1e30f;
    for (int jj = lane; jj < jj_count; jj += 32)
      tile_max = fmaxf(tile_max, tile[jj]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, o));
    tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

    const float m_new = fmaxf(m, tile_max);
    const float rescale = expf(m - m_new);

    float tile_sum = 0.0f;
    for (int jj = lane; jj < jj_count; jj += 32) {
      const float pv = expf(tile[jj] - m_new);
      tile[jj] = pv;
      tile_sum += pv;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      tile_sum += __shfl_down_sync(0xffffffffu, tile_sum, o);
    tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
    l = l * rescale + tile_sum;

    __syncwarp();
    int di = 0;
    for (int d = lane; d < head_dim; d += 32, ++di) {
      float a = acc[di] * rescale;
      for (int jj = 0; jj < jj_count; ++jj) {
        const uint64_t j = j0 + (uint64_t)jj;
        if constexpr (std::is_same_v<KV, float>) {
          const KV* vrow;
          if (j < start) {
            const uint64_t slot = j % n_slots;
            vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
          } else {
            const uint64_t bi = j - start;
            vrow = v_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
          }
          a += tile[jj] * vrow[d];
        } else if (j < start) {
          const uint64_t slot = j % n_slots;
          const KV* vrow =
              kv_row(v_cache, slot * (uint64_t)n_heads_kv + kv_h, head_dim);
          // Explicit FMA: nvcc contracts the f32 instantiation's `a += t*v`
          // into FFMA but not the f16 one, which then rounds the V
          // accumulation 1 ulp differently from strict even though the values
          // are identical. __fmaf_rn pins both to the same FFMA.
          a = __fmaf_rn(tile[jj], kv_load_at(vrow, d, head_dim), a);
        } else {
          const uint64_t bi = j - start;
          const float* vrow =
              v_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
          a = __fmaf_rn(tile[jj], kv_load_batch<KV>(vrow + d), a);
        }
      }
      acc[di] = a;
    }
    m = m_new;
    __syncwarp();
  }

  const float inv = 1.0f / l;
  float* orow = out + ((uint64_t)iq * n_heads_q + h) * head_dim;
  int di = 0;
  for (int d = lane; d < head_dim; d += 32, ++di) orow[d] = acc[di] * inv;
}

// ---------------------------------------------------------------------------
// Strict-order shared-memory tiled flash attention (chunked prefill).
//
// The same per-query arithmetic as flash_attention_chunk_pos_kernel — the
// lane-strided QK partial sums reduced by the fixed 32-lane butterfly, 32-key
// online-softmax tiles, ascending fp32 PV FMA chains, final fp32
// normalization — but K/V are staged once per 32-key tile into shared memory
// and shared by all n_heads_q warps of the block instead of being re-read from
// global memory by every (position, head) warp. The block is one query
// position with one warp per query head, exactly the scalar launch shape, so
// every warp walks the same j0 tiles (window 0 only) and the block-wide
// staging barriers are uniform.
//
// The staged tile is lane-major: element ((kv*BK + jj)*32 + lane)*NP + i holds
// dim lane + 32*i, so the NP terms of one lane's dot are contiguous and become
// a single vector load. K and V reuse one buffer — K is dead once the tile's
// scores are in `tile`, and V is staged after the whole block finishes its QK
// pass — which keeps the footprint at BK*NHKV*HEAD_DIM floats.
//
// This is the strict-profile counterpart of the tensor-core tiled kernel
// below: it buys the K/V reuse and Q/accumulator register residency while
// keeping every FP association of the scalar kernel.
// ---------------------------------------------------------------------------

// One K or V tile for every KV head: [NHKV][BK][HEAD_DIM] fp32, read either
// from the f16/f32 ring cache (j < start) or from the f32 chunk batch (j >=
// start, rounded through the cache's element type exactly as kv_load_batch).
// One warp stages one row, which is one [16-byte] store per warp either way:
//   LANE_MAJOR=false (K): each lane takes four consecutive dims, a single
//     16-byte load.
//   LANE_MAJOR=true (V): each lane takes dim lane+32i, i.e. the four values
//     the PV loop consumes for that lane, so PV becomes one 16-byte shared
//     load per key instead of four. The four global loads are still fully
//     coalesced (one 128-byte transaction per i).
// The row loop is fully unrolled so several independent loads are in flight
// (the scalar kernel's staging cost was one exposed global-load latency per
// element).
template <typename KV, int HEAD_DIM, int NHKV, int NTHREADS,
          bool LANE_MAJOR = false>
__device__ __forceinline__ void fa_exact_stage_kv(
    float* __restrict__ dst, const float* __restrict__ batch,
    const KV* __restrict__ cache, uint64_t start, uint64_t n_slots, uint64_t j0,
    int jj_count) {
  constexpr int NP = HEAD_DIM / 32;         // dims per lane
  static_assert(NP == 4, "staging assumes four dims per lane");
  constexpr int ROWS = NHKV * FA_BK;        // rows per tile
  constexpr int RPI = NTHREADS / 32;        // rows in flight per iteration
  constexpr int ITERS = (ROWS + RPI - 1) / RPI;
  const int lane = threadIdx.x & 31;
  // Ring slot of the tile's first key, so a row's slot is base + jj with at
  // most one wrap instead of a 64-bit modulo per row: device division is a
  // reciprocal sequence (inline for a 32-bit dividend, a 65-instruction call
  // above it) and it stood in the per-row staging path. A tile that ends
  // before the ring (start == 0 during a first prefill) has no ring row at
  // all, so the division is skipped there.
  const uint64_t base_slot = (start > j0) ? j0 % n_slots : 0;
  #pragma unroll
  for (int it = 0; it < ITERS; ++it) {
    const int r = (threadIdx.x >> 5) + it * RPI;
    if (r >= ROWS) continue;
    const int kv = r / FA_BK;
    const int jj = r - kv * FA_BK;
    if (jj >= jj_count) continue;  // tail rows are never read
    const uint64_t j = j0 + (uint64_t)jj;
    const bool ring = j < start;
    const KV* crow = nullptr;
    const float* brow = nullptr;
    if (ring) {
      uint64_t slot = base_slot + (uint64_t)jj;
      while (slot >= n_slots) slot -= n_slots;  // jj < FA_BK: one step
      crow = kv_row(cache, slot * (uint64_t)NHKV + (uint64_t)kv, HEAD_DIM);
    } else {
      const uint64_t bi = j - start;
      brow = batch + (bi * (uint64_t)NHKV + (uint64_t)kv) * HEAD_DIM;
    }
    float v[NP];
    if constexpr (LANE_MAJOR) {
      if (ring) {
        #pragma unroll
        for (int i = 0; i < NP; ++i)
          v[i] = kv_load_at(crow, lane + i * 32, HEAD_DIM);
      } else {
        #pragma unroll
        for (int i = 0; i < NP; ++i)
          v[i] = kv_load_batch<KV>(brow + lane + i * 32);
      }
    } else if (ring) {
      // Four consecutive dims: one width-4 load, then widen.
      if constexpr (std::is_same_v<KV, float>) {
        const float4 x = *reinterpret_cast<const float4*>(crow + lane * NP);
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
      } else {
        const uint2 h = *reinterpret_cast<const uint2*>(crow + lane * NP);
        const __half* hp = reinterpret_cast<const __half*>(&h);
        #pragma unroll
        for (int i = 0; i < NP; ++i) v[i] = __half2float(hp[i]);
      }
    } else {
      const float* src = brow + lane * NP;
      if constexpr (std::is_same_v<KV, float>) {
        const float4 x = *reinterpret_cast<const float4*>(src);
        v[0] = x.x; v[1] = x.y; v[2] = x.z; v[3] = x.w;
      } else {
        #pragma unroll
        for (int i = 0; i < NP; ++i) v[i] = kv_load_batch<KV>(src + i);
      }
    }
    *reinterpret_cast<float4*>(dst + (size_t)r * HEAD_DIM + lane * NP) =
        make_float4(v[0], v[1], v[2], v[3]);
  }
}

// Block = NQ consecutive query positions x NHQ heads, one warp per head; each
// warp carries the NQ positions of its head through the tile loop together, so
// every staged K/V value feeds NQ dots before it is discarded. That halves the
// shared-memory read traffic per (position, head, key) visit, which is what the
// kernel is bound by: a visit needs the whole 128-dim K and V row (1 KiB) for
// only eight FMAs. The four per-tile barriers are shared by all positions too.
// One warp per head means 12 warps per block and 24 warps per SM with two
// resident blocks; four keys x NQ positions in flight per warp cover the
// dependent shuffle latency. HEAD_DIM is a multiple of 32 and NHQ/NHKV is the
// GQA ratio.
template <typename KV, int HEAD_DIM, int NHQ, int NHKV, int NQ>
__global__ void __launch_bounds__(NHQ * 32, 2)
    flash_attention_chunk_exact_kernel(
        const float* __restrict__ q, const float* __restrict__ k_batch,
        const float* __restrict__ v_batch, const KV* __restrict__ k_cache,
        const KV* __restrict__ v_cache, const uint64_t* __restrict__ start_ptr,
        uint64_t n_slots, int n_heads_kv, uint64_t sliding_window, int n,
        float scale, float* __restrict__ out) {
  constexpr int NP = HEAD_DIM / 32;  // dims per lane
  static_assert(NP == 4, "lane-major staging assumes four dims per lane");
  constexpr int KV_TILE = NHKV * FA_BK * HEAD_DIM;
  extern __shared__ float s_sh[];
  float* kv_sm = s_sh;                  // [NHKV][FA_BK][HEAD_DIM], lane-major
  float* tile_all = s_sh + KV_TILE;     // [NHQ][NQ][FA_BK]

  const int lane = threadIdx.x & 31;
  const int h = threadIdx.x >> 5;
  const int kv_h = h * NHKV / NHQ;
  const int iq = blockIdx.x;
  const uint64_t start = *start_ptr;
  const int pos0 = iq * NQ;
  const int npos = (n - pos0 < NQ) ? (n - pos0) : NQ;

  // Q and the output accumulators live in registers for every position: the
  // scalar kernel re-reads Q through L1 for each key and spills acc[] once the
  // loop index becomes dynamic. The values here are the same ones.
  float qreg[NQ][NP], acc[NQ][NP];
  float m[NQ], l[NQ];
  uint64_t p[NQ];
  float* tile = tile_all + h * (NQ * FA_BK);
  #pragma unroll
  for (int t = 0; t < NQ; ++t) {
    const bool ok = t < npos;
    const int posi = pos0 + (ok ? t : 0);
    const float* qrow = q + ((uint64_t)posi * NHQ + h) * HEAD_DIM;
    p[t] = start + (uint64_t)posi;
    m[t] = -1e30f;
    l[t] = 0.0f;
    #pragma unroll
    for (int i = 0; i < NP; ++i) {
      qreg[t][i] = ok ? qrow[lane + i * 32] : 0.0f;
      acc[t][i] = 0.0f;
    }
  }
  const uint64_t p_blk = p[npos - 1];

  // The loop runs to the block's last position; a position that is already
  // finished still reaches every barrier but skips its softmax and PV.
  for (uint64_t j0 = 0; j0 < p_blk + 1; j0 += FA_BK) {
    const uint64_t jstage_end =
        (p_blk + 1 < j0 + FA_BK) ? (p_blk + 1) : (j0 + FA_BK);
    const int jstage_cnt = int(jstage_end - j0);
    bool act[NQ];
    int jc[NQ];
    int jqk = 0;
    #pragma unroll
    for (int t = 0; t < NQ; ++t) {
      act[t] = (t < npos) && (j0 <= p[t]);
      const uint64_t jend =
          (p[t] + 1 < j0 + FA_BK) ? (p[t] + 1) : (j0 + FA_BK);
      jc[t] = act[t] ? int(jend - j0) : 0;
      jqk = (jc[t] > jqk) ? jc[t] : jqk;
    }

    // ---- stage K for every KV head, then compute this tile's scores --------
    fa_exact_stage_kv<KV, HEAD_DIM, NHKV, NHQ * 32, true>(
        kv_sm, k_batch, k_cache, start, n_slots, j0, jstage_cnt);
    __syncthreads();

    // Four keys in flight for every position: the five-step butterfly is a
    // long dependent shfl chain and one staged K row feeds all NQ dots. Each
    // key's dot order and reduction tree are exactly the scalar kernel's.
    int jj = 0;
    for (; jj + 3 < jqk; jj += 4) {
      const float* kp = kv_sm + (kv_h * FA_BK + jj) * HEAD_DIM + lane * NP;
      float k0[NP], k1[NP], k2[NP], k3[NP];
      *reinterpret_cast<float4*>(k0) = *reinterpret_cast<const float4*>(kp);
      *reinterpret_cast<float4*>(k1) =
          *reinterpret_cast<const float4*>(kp + HEAD_DIM);
      *reinterpret_cast<float4*>(k2) =
          *reinterpret_cast<const float4*>(kp + 2 * HEAD_DIM);
      *reinterpret_cast<float4*>(k3) =
          *reinterpret_cast<const float4*>(kp + 3 * HEAD_DIM);
      float d[NQ][4];
      #pragma unroll
      for (int t = 0; t < NQ; ++t) {
        d[t][0] = 0.0f; d[t][1] = 0.0f; d[t][2] = 0.0f; d[t][3] = 0.0f;
        #pragma unroll
        for (int i = 0; i < NP; ++i) {
          d[t][0] += qreg[t][i] * k0[i];
          d[t][1] += qreg[t][i] * k1[i];
          d[t][2] += qreg[t][i] * k2[i];
          d[t][3] += qreg[t][i] * k3[i];
        }
      }
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int t = 0; t < NQ; ++t) {
          d[t][0] += __shfl_down_sync(0xffffffffu, d[t][0], o);
          d[t][1] += __shfl_down_sync(0xffffffffu, d[t][1], o);
          d[t][2] += __shfl_down_sync(0xffffffffu, d[t][2], o);
          d[t][3] += __shfl_down_sync(0xffffffffu, d[t][3], o);
        }
      }
      if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < NQ; ++t) {
          tile[t * FA_BK + jj + 0] = d[t][0] * scale;
          tile[t * FA_BK + jj + 1] = d[t][1] * scale;
          tile[t * FA_BK + jj + 2] = d[t][2] * scale;
          tile[t * FA_BK + jj + 3] = d[t][3] * scale;
        }
      }
    }
    for (; jj < jqk; ++jj) {
      const float* kp = kv_sm + (kv_h * FA_BK + jj) * HEAD_DIM + lane * NP;
      float k[NP];
      *reinterpret_cast<float4*>(k) = *reinterpret_cast<const float4*>(kp);
      #pragma unroll
      for (int t = 0; t < NQ; ++t) {
        float dot = 0.0f;
        #pragma unroll
        for (int i = 0; i < NP; ++i) dot += qreg[t][i] * k[i];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          dot += __shfl_down_sync(0xffffffffu, dot, o);
        if (lane == 0) tile[t * FA_BK + jj] = dot * scale;
      }
    }
    __syncwarp();

    // ---- online softmax, identical to the scalar kernel -------------------
    float m_new[NQ], rescale[NQ];
    #pragma unroll
    for (int t = 0; t < NQ; ++t) {
      m_new[t] = m[t];
      rescale[t] = 1.0f;
      if (act[t]) {
        const int n_j = jc[t];
        float* tl = tile + t * FA_BK;
        float tile_max = -1e30f;
        for (int j = lane; j < n_j; j += 32) tile_max = fmaxf(tile_max, tl[j]);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          tile_max =
              fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, o));
        tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
        m_new[t] = fmaxf(m[t], tile_max);
        rescale[t] = expf(m[t] - m_new[t]);
        float tile_sum = 0.0f;
        for (int j = lane; j < n_j; j += 32) {
          const float pv = expf(tl[j] - m_new[t]);
          tl[j] = pv;
          tile_sum += pv;
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
          tile_sum += __shfl_down_sync(0xffffffffu, tile_sum, o);
        tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
        l[t] = l[t] * rescale[t] + tile_sum;
      }
    }

    // ---- stage V into the same buffer, then accumulate --------------------
    __syncthreads();  // every warp is done reading K
    fa_exact_stage_kv<KV, HEAD_DIM, NHKV, NHQ * 32, true>(
        kv_sm, v_batch, v_cache, start, n_slots, j0, jstage_cnt);
    __syncthreads();
    __syncwarp();

    #pragma unroll
    for (int t = 0; t < NQ; ++t) {
      if (!act[t]) continue;
      const float* tl = tile + t * FA_BK;
      // Loop order is interchanged (key outer, dim inner): each accumulator
      // sees exactly the scalar kernel's ascending sequence of FMAs, while the
      // key weight and the V row are loaded once for all four accumulators.
      #pragma unroll
      for (int i = 0; i < NP; ++i) acc[t][i] *= rescale[t];
      for (int j = 0; j < jc[t]; ++j) {
        const float tv = tl[j];
        const float4 v = *reinterpret_cast<const float4*>(
            kv_sm + (kv_h * FA_BK + j) * HEAD_DIM + lane * NP);
        acc[t][0] = __fmaf_rn(tv, v.x, acc[t][0]);
        acc[t][1] = __fmaf_rn(tv, v.y, acc[t][1]);
        acc[t][2] = __fmaf_rn(tv, v.z, acc[t][2]);
        acc[t][3] = __fmaf_rn(tv, v.w, acc[t][3]);
      }
    }
    #pragma unroll
    for (int t = 0; t < NQ; ++t) m[t] = m_new[t];
    __syncwarp();
    __syncthreads();  // every warp is done reading V before the next K stage
  }

  #pragma unroll
  for (int t = 0; t < NQ; ++t) {
    if (t >= npos) continue;
    const float inv = 1.0f / l[t];
    float* orow = out + ((uint64_t)(pos0 + t) * NHQ + h) * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < NP; ++i) orow[lane + i * 32] = acc[t][i] * inv;
  }
}

// ---------------------------------------------------------------------------
// Launchers.
// ---------------------------------------------------------------------------

// Strict-order shared-memory tiled kernel. Fixed block shape (NQ query
// positions x NHQ warps) and BK == FA_BK so its softmax tiles and reduction
// points are the scalar kernel's.
constexpr int kExactNQ = 2;

template <typename KV, int HEAD_DIM, int NHQ, int NHKV>
bool launch_exact_flash_attention(const float* q, const float* k_batch,
                                  const float* v_batch, const void* k_cache,
                                  const void* v_cache, const uint64_t* start,
                                  uint64_t n_slots, int n,
                                  uint64_t sliding_window, float scale,
                                  float* out) {
  constexpr int NQ = kExactNQ;
  constexpr int smem_bytes =
      (NHKV * FA_BK * HEAD_DIM + NQ * NHQ * FA_BK) * (int)sizeof(float);
  auto kfn = flash_attention_chunk_exact_kernel<KV, HEAD_DIM, NHQ, NHKV, NQ>;
  SB_NOTE_LAUNCH_TP("flash_attention_chunk_exact_kernel",
                    ::sonicboom::planner::plan_detail::tp_label(
                        {{"HEAD_DIM", HEAD_DIM}, {"NHQ", NHQ}, {"NHKV", NHKV}}),
                    (n + NQ - 1) / NQ, NHQ * 32, smem_bytes, cur_stream());
  kfn<<<(n + NQ - 1) / NQ, NHQ * 32, smem_bytes, cur_stream()>>>(
      q, k_batch, v_batch, static_cast<const KV*>(k_cache),
      static_cast<const KV*>(v_cache), start, n_slots, NHKV, sliding_window, n,
      scale, out);
  return cudaGetLastError() == cudaSuccess;
}

bool strict_exact_enabled() {
  const char* e = std::getenv("SONICBOOM_ATTN_D128_EXACT");
  return !(e != nullptr && std::string_view(e) == "0");
}

}  // namespace

// Strict-order chunked prefill for the d128/GQA-6 class. Returns false for any
// other shape; the caller keeps its own routing for those.
bool flash_attention_boom_fattn_strict_chunk_dev_pos(
    const float* q, const float* k_batch, const float* v_batch,
    const float* k_cache, const float* v_cache, const uint64_t* start,
    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* out, bool kv_f16,
    bool kv_q8) {
  if (head_dim != 128 || n_heads_q != 12 || n_heads_kv != 2 ||
      sliding_window != 0)
    return false;
  if (n <= 0 || n_slots == 0) return false;
  constexpr int NHQ = 12, NHKV = 2, kHeadDim = 128;

  if (!kv_q8 && strict_exact_enabled()) {
    const bool ok =
        kv_f16 ? launch_exact_flash_attention<__half, kHeadDim, NHQ, NHKV>(
                     q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
                     sliding_window, scale, out)
               : launch_exact_flash_attention<float, kHeadDim, NHQ, NHKV>(
                     q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
                     sliding_window, scale, out);
    if (ok) return true;
  }

  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  SB_NOTE_LAUNCH(flash_attention_chunk_pos_kernel, n, n_heads_q * 32, smem,
                 cur_stream());
  auto launch_scalar = [&](auto tag) {
    using KV = decltype(tag);
    flash_attention_chunk_pos_kernel<KV, kHeadDim>
        <<<n, NHQ * 32, smem, cur_stream()>>>(
            q, k_batch, v_batch, reinterpret_cast<const KV*>(k_cache),
            reinterpret_cast<const KV*>(v_cache), start, n_slots, n_heads_q,
            n_heads_kv, head_dim, sliding_window, scale, out);
  };
  if (kv_f16)
    launch_scalar(__half{});
  else if (kv_q8)
    launch_scalar(int8_t{});
  else
    launch_scalar(float{});
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace sonicboom::nn::cuda
