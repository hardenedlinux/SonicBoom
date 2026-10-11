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

#include <sonicboom/nn/cuda_elementwise.h>

#include "cuda_resident.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <unordered_map>

// CUDA kernels for the native-transformer elementwise / norm / RoPE ops and the
// dense gate matvecs (Phase 6a target 3, first slice). Each entry point mirrors
// a CPU reference in core/src/nn/*.cpp with the same signature: upload inputs,
// launch a grid-stride kernel, copy the result back. The arithmetic matches the
// CPU reference to fp32 tolerance — the CPU accumulates mean/matvec in double
// (ggml_float) while these kernels accumulate in fp32, the same relaxation the
// K-quant gemv in cuda_quant_matmul.cu makes. fp16 conversions use CUDA's
// round-to-nearest __float2half_rn / __half2float, matching std::float16_t.

namespace sonicboom::nn::cuda {

namespace {

// --- device helpers ---------------------------------------------------------

__device__ inline float bf16_to_float(uint16_t h) {
  return __uint_as_float(uint32_t(h) << 16);
}

// f32 -> fp16 -> f32 round-trip (matches std::float16_t, round to nearest even).
__device__ inline float half_roundtrip(float v) {
  return __half2float(__float2half_rn(v));
}

// f32 -> bf16, round-to-nearest-even, bit-identical to sonicboom::f32_to_bf16
// (core/include/sonicboom/dtype.h). NaN is quieted the same way.
__device__ inline uint16_t float_to_bf16(float f) {
  const uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffff) > 0x7f800000) return uint16_t((u >> 16) | 64);
  return uint16_t((u + (0x7fff + ((u >> 16) & 1))) >> 16);
}

// f32 -> bf16 -> f32 round-trip (matches bf16_to_f32(f32_to_bf16(v))).
__device__ inline float bf16_roundtrip(float v) {
  return __uint_as_float(uint32_t(float_to_bf16(v)) << 16);
}

// ggml's ggml_gelu_f32, reproduced verbatim (constants + operation order) so the
// tanh argument matches the CPU's gelu_tanh_f32. Device tanhf vs the CPU's
// std::tanh(double) differ by ~1e-7, well inside the fp32 tolerance.
__device__ inline float gelu_tanh_f32(float x) {
  constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
  constexpr float gelu_coef_a = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
}

// --- kernels ----------------------------------------------------------------

// Parallel sum-of-squares: warp-shuffle + shared-memory block reduction. Each
// thread accumulates a partial double sum over a strided slice of x, then the
// block reduces the partials. Accumulation stays in double and each term is
// still double(v * v) (float square) — identical per-term semantics to the CPU
// rms_norm's sequential double sum (norm.cpp). Only the *order* of the double
// adds changes, and that reordering is many orders below the fp32 rounding that
// rms_inv_kernel applies when it converts sum/n to float, so the downstream
// token stream is unchanged. This replaces a single-thread loop that serialized
// ~2560 dependent double adds (~127 us) with a ~256-way reduction.
__global__ void rms_sum_sq_kernel(const float* __restrict__ x, int n,
                                  double* __restrict__ out) {
  const int tid = threadIdx.x;
  double s = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = x[i];
    s += double(v * v);
  }
  // Warp-level butterfly reduction.
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ double warp_part[32];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) warp_part[warp] = s;
  __syncthreads();
  // First warp reduces the per-warp partials and writes the single result.
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    double t = (lane < nwarps) ? warp_part[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
    if (lane == 0) *out = t;
  }
}

__global__ void rms_apply_kernel(const float* __restrict__ x,
                                 const float* __restrict__ weight, int n, float inv,
                                 float* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] = weight ? x[i] * inv * weight[i] : x[i] * inv;
}

__global__ void gelu_fp16_kernel(const float* __restrict__ x, float* __restrict__ y,
                                 int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  float out;
  if (v <= -10.0f) {
    out = 0.0f;
  } else if (v >= 10.0f) {
    out = v;
  } else {
    out = half_roundtrip(gelu_tanh_f32(half_roundtrip(v)));
  }
  y[i] = out;
}

__global__ void mul_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

__global__ void add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

__global__ void scale_kernel(const float* __restrict__ x, float s,
                             float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * s;
}

__global__ void cast_fp16_kernel(float* __restrict__ v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = half_roundtrip(v[i]);
}

__global__ void cast_bf16_kernel(float* __restrict__ v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] = bf16_roundtrip(v[i]);
}

__global__ void layer_combine_kernel(const float* __restrict__ proj,
                                     const float* __restrict__ ple, float ple_scale,
                                     float combine_scale, float* __restrict__ y,
                                     int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = (proj[i] + ple[i] * ple_scale) * combine_scale;
}

__global__ void rope_neox_kernel(float* __restrict__ x, int half, int n, uint64_t pos,
                                 float freq_scale, float log_base,
                                 const float* __restrict__ freq_factors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= half) return;
  float theta = freq_scale * float(pos) *
                expf(-2.0f * float(i) / float(n) * log_base);
  if (freq_factors) theta /= freq_factors[i];
  const float c = cosf(theta);
  const float s = sinf(theta);
  const float x0 = x[i];
  const float x1 = x[i + half];
  x[i] = x0 * c - x1 * s;
  x[i + half] = x0 * s + x1 * c;
}

// Multi-head RoPE: rotate each contiguous head slice independently. `i` is the
// flat index; head = i/head_dim, off = i%head_dim. Only the low half of each head
// participates (off < head_dim/2), pairing off with off + head_dim/2.
__global__ void rope_neox_heads_kernel(float* __restrict__ x, int head_dim, int n,
                                       uint64_t pos, float freq_scale,
                                       float log_base,
                                       const float* __restrict__ freq_factors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int half = head_dim >> 1;
  const int off = i % head_dim;
  if (off >= half) return;
  float theta = freq_scale * float(pos) *
                expf(-2.0f * float(off) / float(head_dim) * log_base);
  if (freq_factors) theta /= freq_factors[off];
  const float c = cosf(theta);
  const float s = sinf(theta);
  const float x0 = x[i];
  const float x1 = x[i + half];
  x[i] = x0 * c - x1 * s;
  x[i + half] = x0 * s + x1 * c;
}

// y[j] = sum_i W[j*cols + i] * x[i], fp32 accumulation. W is row-major with
// `cols` contiguous (GGUF order).
__global__ void gemv_dense_f32(const float* __restrict__ W, const float* __restrict__ x,
                               float* __restrict__ y, int cols, int rows) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < rows;
       j += gridDim.x * blockDim.x) {
    const float* wrow = W + (size_t)j * cols;
    float acc = 0.0f;
    for (int i = 0; i < cols; ++i) acc += wrow[i] * x[i];
    y[j] = acc;
  }
}

// y[j] = sum_i bf16_to_float(W[j*cols + i]) * x[i]. W holds raw bf16 (uint16);
// x is already the bf16-rounded activation, reproducing the CPU's per-product
// rounding exactly.
__global__ void gemv_bf16(const uint16_t* __restrict__ W, const float* __restrict__ x,
                          float* __restrict__ y, int cols, int rows) {
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < rows;
       j += gridDim.x * blockDim.x) {
    const uint16_t* wrow = W + (size_t)j * cols;
    float acc = 0.0f;
    for (int i = 0; i < cols; ++i) acc += bf16_to_float(wrow[i]) * x[i];
    y[j] = acc;
  }
}

// --- vectorized warp-per-row dense gemv (Phase 2.3b) ------------------------
//
// Each warp computes a partial dot over a contiguous chunk of one row's
// float4/uint4 vectors, then __shfl-reduces. `k_split` warps cooperate per row:
// each chunk writes its partial into `dst[row * k_split + chunk]` (no atomics),
// and a second `gemv_reduce_kernel` sums the chunks in fixed chunk order so the
// accumulation order is deterministic across runs. k_split == 1 => dst == y and
// chunk 0 writes y[row] directly. The few-row shapes (inp_gate, 256 rows) still
// fill the SMs. W and x must be 16-byte aligned; cols must be a multiple of 4
// (f32) / 8 (bf16).

__global__ void gemv_dense_f32_v(const float* __restrict__ W,
                                 const float* __restrict__ x,
                                 float* __restrict__ dst, int cols, int rows,
                                 int k_split) {
  const int warp_id =
      blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int row = warp_id / k_split;
  const int chunk = warp_id % k_split;
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;

  const int nv = cols >> 2;                       // float4 per row
  const int chunk_nv = (nv + k_split - 1) / k_split;
  const int v_begin = chunk * chunk_nv;
  const int v_end = min(v_begin + chunk_nv, nv);

  const float4* wv = reinterpret_cast<const float4*>(W + (size_t)row * cols);
  const float4* xv = reinterpret_cast<const float4*>(x);

  float acc = 0.0f;
  for (int v = v_begin + lane; v < v_end; v += 32) {
    const float4 w = wv[v];
    const float4 xi = xv[v];
    acc += w.x * xi.x + w.y * xi.y + w.z * xi.z + w.w * xi.w;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) dst[(size_t)row * k_split + chunk] = acc;
}

__global__ void gemv_bf16_v(const uint16_t* __restrict__ W,
                            const float* __restrict__ x,
                            float* __restrict__ dst, int cols, int rows,
                            int k_split) {
  const int warp_id =
      blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int row = warp_id / k_split;
  const int chunk = warp_id % k_split;
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;

  const int nv = cols >> 3;                       // uint4 (8 bf16) per row
  const int chunk_nv = (nv + k_split - 1) / k_split;
  const int v_begin = chunk * chunk_nv;
  const int v_end = min(v_begin + chunk_nv, nv);

  const uint4* wv = reinterpret_cast<const uint4*>(W + (size_t)row * cols);
  const float4* xv = reinterpret_cast<const float4*>(x);

  float acc = 0.0f;
  for (int v = v_begin + lane; v < v_end; v += 32) {
    const uint4 w = wv[v];
    const float4 x0 = xv[(v << 1)];
    const float4 x1 = xv[(v << 1) + 1];
    acc += bf16_to_float(uint16_t(w.x & 0xffffu)) * x0.x +
           bf16_to_float(uint16_t(w.x >> 16)) * x0.y +
           bf16_to_float(uint16_t(w.y & 0xffffu)) * x0.z +
           bf16_to_float(uint16_t(w.y >> 16)) * x0.w +
           bf16_to_float(uint16_t(w.z & 0xffffu)) * x1.x +
           bf16_to_float(uint16_t(w.z >> 16)) * x1.y +
           bf16_to_float(uint16_t(w.w & 0xffffu)) * x1.z +
           bf16_to_float(uint16_t(w.w >> 16)) * x1.w;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, o);
  if (lane == 0) dst[(size_t)row * k_split + chunk] = acc;
}

// Second phase of the split-K dense gemv: sum each row's `k_split` partial
// chunks in fixed order (chunk 0, 1, ...) into y[row]. Deterministic by
// construction — no atomics, no scheduling-dependent accumulation order.
__global__ void gemv_reduce_kernel(const float* __restrict__ partial,
                                   float* __restrict__ y, int rows, int k_split) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const float* p = partial + (size_t)row * k_split;
  float acc = 0.0f;
  for (int c = 0; c < k_split; ++c) acc += p[c];
  y[row] = acc;
}

// --- device-resident rms_norm reduction (no D2H of the sum) -----------------

// Single-thread inv: reads the double sum and writes
// inv = 1/sqrt((double)sum/(double)n + eps), reproducing the host rms_norm's
// double mean division + float sqrt/recip exactly.
__global__ void rms_inv_kernel(const double* __restrict__ sum, int n, float eps,
                               float* __restrict__ inv) {
  const double s = *sum;
  *inv = 1.0f / sqrtf(float(double(s) / double(n)) + eps);
}

// rms apply variant that reads `inv` from a device scalar (device-resident path).
__global__ void rms_apply_dev_kernel(const float* __restrict__ x,
                                     const float* __restrict__ weight, int n,
                                     const float* __restrict__ inv,
                                     float* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float iv = *inv;
  y[i] = weight ? x[i] * iv * weight[i] : x[i] * iv;
}

// Per-head sum-of-squares, one thread per head accumulating in double in
// ascending index order — bit-identical to the CPU rms_norm_heads sequential
// double sum (norm.cpp).
__global__ void rms_heads_sum_kernel(const float* __restrict__ x, int head_dim,
                                     int heads, double* __restrict__ sums) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= heads) return;
  const float* __restrict__ xh = x + (size_t)h * head_dim;
  double s = 0.0;
  for (int i = 0; i < head_dim; ++i) {
    const float v = xh[i];
    s += double(v * v);
  }
  sums[h] = s;
}

__global__ void rms_heads_inv_kernel(const double* __restrict__ sums, int head_dim,
                                     float eps, float* __restrict__ inv, int heads) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= heads) return;
  const double s = sums[h];
  inv[h] = 1.0f / sqrtf(float(double(s) / double(head_dim)) + eps);
}

__global__ void rms_heads_apply_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       int head_dim, int heads,
                                       const float* __restrict__ inv,
                                       float* __restrict__ y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = head_dim * heads;
  if (i >= n) return;
  const int h = i / head_dim;
  const float iv = inv[h];
  y[i] = weight ? x[i] * iv * weight[i % head_dim] : x[i] * iv;
}

// GQA broadcast for n_k == 1 attention: dst[hq*d + i] = src[kv_h*d + i],
// kv_h = hq * n_kv / n_q.
__global__ void gqa_broadcast_kernel(const float* __restrict__ src,
                                     float* __restrict__ dst, int n_q, int n_kv,
                                     int head_dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = n_q * head_dim;
  if (i >= n) return;
  const int hq = i / head_dim;
  const int off = i % head_dim;
  const int kv_h = hq * n_kv / n_q;
  dst[i] = src[kv_h * head_dim + off];
}

// Incremental decode attention (mirrors nn::decode_attention): one WARP per
// query head (Gemma 4: n_heads_q=8 <= 32, so a single block of n_heads_q*32
// threads). The q·k dot is computed ONCE per key — 32 lanes each sum a strided
// head_dim/32 slice, then __shfl-reduce — and the scores land in dynamic shared
// memory so the dot is never recomputed (the prior 3-pass form recomputed it
// three times). max/sum use warp reductions, and exp is evaluated once (scores
// are overwritten with exp(score-max)) and reused by the V-weighted sum.
//
// fp32 KV is unchanged. The only numeric difference from the CPU reference is
// reduction order (lane-strided + tree, vs the CPU's sequential scalar sum) —
// fmaxf is exact, so the max is order-independent; the sum and dot carry a
// ~1e-4 relative error from reordering, well inside the fp32 tolerance.
__global__ void decode_attention_kernel(const float* __restrict__ q,
                                        const float* __restrict__ k_cache,
                                        const float* __restrict__ v_cache,
                                        uint64_t start, uint64_t n_keys,
                                        uint64_t n_slots,
                                        int n_heads_q, int n_heads_kv,
                                        int head_dim, float scale,
                                        float* __restrict__ out) {
  extern __shared__ float scores[];  // [n_heads_q * n_keys]
  const int lane = threadIdx.x & 31;
  const int h = threadIdx.x >> 5;    // warp id == query head
  if (h >= n_heads_q) return;
  const int kv_h = h * n_heads_kv / n_heads_q;
  const float* qrow = q + (uint64_t)h * head_dim;
  float* srow = scores + (uint64_t)h * n_keys;

  // Phase 1: one q·k dot per key, warp-cooperative.
  for (uint64_t idx = 0; idx < n_keys; ++idx) {
    const uint64_t slot = (start + idx) % n_slots;
    const float* krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    float partial = 0.0f;
    for (int d = lane; d < head_dim; d += 32) partial += qrow[d] * krow[d];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, o);
    if (lane == 0) srow[idx] = partial * scale;
  }
  __syncwarp();

  // Phase 2a: max score (fmaxf is exact, so the reduction order is irrelevant).
  float max_score = -1e30f;
  for (uint64_t idx = lane; idx < n_keys; idx += 32)
    max_score = fmaxf(max_score, srow[idx]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    max_score = fmaxf(max_score, __shfl_down_sync(0xffffffffu, max_score, o));
  max_score = __shfl_sync(0xffffffffu, max_score, 0);

  // Phase 2b: exp once, store back, warp-reduce the sum.
  float sum = 0.0f;
  for (uint64_t idx = lane; idx < n_keys; idx += 32) {
    const float e = expf(srow[idx] - max_score);
    srow[idx] = e;
    sum += e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, o);
  const float inv = 1.0f / __shfl_sync(0xffffffffu, sum, 0);
  __syncwarp();

  // Phase 3: V-weighted sum, warp-cooperative — each lane owns a strided slice
  // of the output row's dims, accumulating over keys in the same sequential
  // order as the CPU reference.
  float* orow = out + (uint64_t)h * head_dim;
  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    for (uint64_t idx = 0; idx < n_keys; ++idx) {
      const uint64_t slot = (start + idx) % n_slots;
      const float* vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
      acc += srow[idx] * inv * vrow[d];
    }
    orow[d] = acc;
  }
}

// --- flash attention (prefill) ----------------------------------------------
//
// Tiled online-softmax attention (the "flash" part: the [n_q, n_kv] scores
// matrix is never materialized; memory is O(n_q + n_kv)). One warp per (query
// position, query head), so a block of n_heads_q warps handles one query
// position. The KV sequence is walked in FA_BK-key tiles; within a tile each
// q·k dot is a warp-cooperative strided sum (same as decode_attention_kernel),
// then running-max / running-sum / V-accumulator are rescaled online-softmax
// style. Numerics mirror nn::scaled_dot_product_attention: scale, no softcap,
// causal + sliding-window mask, exp(s - max) / sum exp(s - max). The masked
// sentinel -1e30f is far below any real score (q/k are RMSNormed), so
// expf(-1e30 - max) == 0 exactly and an all-masked tile is detected by
// tile_max == -1e30f and skipped. fp32 KV is unchanged (the caller pre-rounds
// K/V to fp16 via cast_fp16, same as the decode path).
constexpr int FA_BK = 32;  // keys per online-softmax tile

__global__ void flash_attention_kernel(const float* __restrict__ q,
                                       const float* __restrict__ k,
                                       const float* __restrict__ v, int n_kv,
                                       int n_heads_q, int n_heads_kv,
                                       int head_dim, uint64_t sliding_window,
                                       float scale, float* __restrict__ out) {
  extern __shared__ float s_sh[];  // [n_heads_q * FA_BK]
  const int lane = threadIdx.x & 31;
  const int h = threadIdx.x >> 5;  // warp == query head
  const int iq = blockIdx.x;       // query position (0-based)
  if (h >= n_heads_q) return;
  const int kv_h = h * n_heads_kv / n_heads_q;
  const float* qrow = q + ((uint64_t)iq * n_heads_q + h) * head_dim;
  float* tile = s_sh + h * FA_BK;

  // acc[d] for d = lane, lane+32, ... (head_dim/32 dims per lane, <= 16).
  float acc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  float m = -1e30f, l = 0.0f;

  for (int j0 = 0; j0 < n_kv; j0 += FA_BK) {
    const int jj_count = (n_kv - j0 < FA_BK) ? (n_kv - j0) : FA_BK;

    // 1. warp-cooperative q·k dot per key; mask in-line; store score in tile.
    for (int jj = 0; jj < jj_count; ++jj) {
      const int j = j0 + jj;
      const float* krow = k + ((uint64_t)j * n_heads_kv + kv_h) * head_dim;
      float dot = 0.0f;
      for (int d = lane; d < head_dim; d += 32) dot += qrow[d] * krow[d];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, o);
      if (lane == 0) {
        const bool masked =
            (j > iq) || (sliding_window > 0 && iq >= j + sliding_window);
        tile[jj] = masked ? -1e30f : dot * scale;
      }
    }
    __syncwarp();

    // 2. tile max (== -1e30f iff every key in the tile is masked).
    float tile_max = -1e30f;
    for (int jj = lane; jj < jj_count; jj += 32)
      tile_max = fmaxf(tile_max, tile[jj]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, o));
    tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
    if (tile_max == -1e30f) { __syncwarp(); continue; }  // all keys masked

    // 3. online-softmax rescale (m == -1e30f on the first live tile => expf(-inf)=0).
    const float m_new = fmaxf(m, tile_max);
    const float rescale = expf(m - m_new);

    // 4. exp once, store p back into tile; warp-reduce the tile sum.
    float tile_sum = 0.0f;
    for (int jj = lane; jj < jj_count; jj += 32) {
      const float p = expf(tile[jj] - m_new);
      tile[jj] = p;
      tile_sum += p;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      tile_sum += __shfl_down_sync(0xffffffffu, tile_sum, o);
    tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
    l = l * rescale + tile_sum;

    // 5. V-weighted accumulate: acc[d] = acc[d]*rescale + sum_jj tile[jj]*v[j,d].
    __syncwarp();
    int di = 0;
    for (int d = lane; d < head_dim; d += 32, ++di) {
      float a = acc[di] * rescale;
      for (int jj = 0; jj < jj_count; ++jj) {
        const float* vrow = v + ((uint64_t)(j0 + jj) * n_heads_kv + kv_h) * head_dim;
        a += tile[jj] * vrow[d];
      }
      acc[di] = a;
    }
    m = m_new;
    __syncwarp();
  }

  // Finalize: out[d] = acc[d] / l.
  const float inv = 1.0f / l;
  float* orow = out + ((uint64_t)iq * n_heads_q + h) * head_dim;
  int di = 0;
  for (int d = lane; d < head_dim; d += 32, ++di) orow[d] = acc[di] * inv;
}

// --- device state (scratch + weight cache) ----------------------------------

struct DeviceWeight {
  void* ptr = nullptr;
  size_t bytes = 0;
};

// Key the weight cache by (host pointer, byte size) rather than pointer alone:
// the same host pointer can legitimately serve different views (e.g. the full
// per_layer_model_proj tensor vs. a contiguous leading per-block slice). Keying
// by pointer alone lets one view shadow the other — the size-mismatched upload
// misses forever and its freshly-allocated buffer leaks every decode step.
struct CacheKey {
  const void* host;
  size_t bytes;
  bool operator==(const CacheKey& o) const {
    return host == o.host && bytes == o.bytes;
  }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const noexcept {
    return std::hash<const void*>{}(k.host) ^
           (std::hash<size_t>{}(k.bytes) * 0x9e3779b97f4a7c15ull);
  }
};

std::unordered_map<CacheKey, DeviceWeight, CacheKeyHash> g_cache;
std::mutex g_cache_mutex;

void* g_dx = nullptr;
size_t g_dx_bytes = 0;
void* g_db = nullptr;
size_t g_db_bytes = 0;
void* g_dy = nullptr;
size_t g_dy_bytes = 0;
void* g_dff = nullptr;
size_t g_dff_bytes = 0;
void* g_partial = nullptr;      // split-K dense gemv per-chunk partials (rows*k_split floats)
size_t g_partial_bytes = 0;
double* g_sum = nullptr;        // device-resident rms_norm sum (double, matches CPU)
float* g_inv = nullptr;        // device-resident rms_norm inv scalar
double* g_head_sum = nullptr;   // device-resident per-head rms_norm sums (double)
float* g_head_inv = nullptr;   // device-resident per-head rms_norm inv
int g_head_cap = 0;            // number of heads allocated for g_head_sum/inv

bool g_available_checked = false;
bool g_available = false;

// Grow a device scratch buffer to at least `need` bytes, reusing it when big
// enough (nullptr on allocation failure).
void* grow_scratch(void*& buf, size_t& have, size_t need) {
  if (have >= need) return buf;
  if (buf) cudaFree(buf);
  buf = nullptr;
  have = 0;
  if (cudaMalloc(&buf, need) != cudaSuccess) return nullptr;
  have = need;
  return buf;
}

// Upload `bytes` from `host`, caching by (pointer, byte size) so repeated calls
// with the same weight (or the same contiguous slice of a larger tensor) reuse
// the device copy.
const void* upload_bytes(const void* host, size_t bytes) {
  std::lock_guard<std::mutex> lk(g_cache_mutex);
  const CacheKey key{host, bytes};
  auto it = g_cache.find(key);
  if (it != g_cache.end()) return it->second.ptr;

  void* dptr = nullptr;
  if (cudaMalloc(&dptr, bytes) != cudaSuccess) return nullptr;
  if (cudaMemcpy(dptr, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dptr);
    return nullptr;
  }
  g_cache.emplace(key, DeviceWeight{dptr, bytes});
  return dptr;
}

inline int grid_for(int n, int block = 256) { return (n + block - 1) / block; }

// Split-K selection for the vectorized dense gemv: choose enough warps to keep
// the SMs busy even for few-row shapes (e.g. inp_gate's 256 rows).
constexpr int kGemvWarpsPerBlock = 8;  // 256 threads
constexpr int kGemvTargetWarps = 1024;

inline int gemv_k_split(int rows) {
  return rows >= kGemvTargetWarps ? 1 : (kGemvTargetWarps + rows - 1) / rows;
}

inline int gemv_grid(int rows, int k_split) {
  const int warps = rows * k_split;
  return (warps + kGemvWarpsPerBlock - 1) / kGemvWarpsPerBlock;
}

// Greedy argmax over a device f32 vector (Phase 3.2): one block grid-strides the
// whole vector, each thread keeps the running max of its slice (ties broken to
// the lowest index via a strict `>`), then a shared-memory tree reduction picks
// the block winner. Single launch, deterministic, no atomics; mirrors nn::argmax.
__global__ void argmax_kernel(const float* __restrict__ x, float* __restrict__ out,
                              int n) {
  __shared__ float s_val[1024];
  __shared__ int s_idx[1024];
  const int tid = threadIdx.x;
  float mv = -INFINITY;
  int mi = -1;
  for (int i = tid; i < n; i += blockDim.x) {
    if (x[i] > mv) { mv = x[i]; mi = i; }
  }
  s_val[tid] = mv;
  s_idx[tid] = mi;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      if (s_val[tid + s] > s_val[tid]) {
        s_val[tid] = s_val[tid + s];
        s_idx[tid] = s_idx[tid + s];
      }
    }
    __syncthreads();
  }
  if (tid == 0) out[0] = (s_idx[0] < 0) ? -1.0f : float(s_idx[0]);
}

} // namespace

bool available() noexcept {
  if (!g_available_checked) {
    int n = 0;
    g_available = (cudaGetDeviceCount(&n) == cudaSuccess && n > 0);
    g_available_checked = true;
  }
  return g_available;
}

void clear_cache() {
  std::lock_guard<std::mutex> lk(g_cache_mutex);
  for (auto& kv : g_cache) cudaFree(kv.second.ptr);
  g_cache.clear();
}

bool rms_norm(std::span<const float> x, std::span<const float> weight, float eps,
              std::span<float> y) {
  if (!available()) return false;
  const uint64_t n = x.size();
  if (y.size() < n) return false;
  if (!weight.empty() && weight.size() < n) return false;

  void* d_x = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, n * sizeof(float));
  if (!d_x || !d_y) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(grow_scratch(g_db, g_db_bytes, n * sizeof(float)));
    if (!d_w) return false;
    if (cudaMemcpy(const_cast<float*>(d_w), weight.data(), n * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
  }
  if (cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  if (!g_sum && cudaMalloc(reinterpret_cast<void**>(&g_sum), sizeof(double)) != cudaSuccess)
    return false;
  if (cudaMemset(g_sum, 0, sizeof(double)) != cudaSuccess) return false;

  rms_sum_sq_kernel<<<1, 256>>>(static_cast<const float*>(d_x), int(n), g_sum);
  if (cudaGetLastError() != cudaSuccess) return false;
  double sum = 0.0;
  if (cudaMemcpy(&sum, g_sum, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;

  // Reproduce the CPU's mean/rsqrt in double (mean) then float (sqrt/recip),
  // matching norm.cpp bit-for-bit.
  const float inv = 1.0f / std::sqrt(float(sum / double(n)) + eps);
  rms_apply_kernel<<<grid_for(int(n)), 256>>>(static_cast<const float*>(d_x), d_w,
                                              int(n), inv, static_cast<float*>(d_y));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool gelu_fp16(std::span<const float> x, std::span<float> y) {
  if (!available()) return false;
  const uint64_t n = x.size();
  if (y.size() < n) return false;
  void* d_x = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, n * sizeof(float));
  if (!d_x || !d_y) return false;
  if (cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  gelu_fp16_kernel<<<grid_for(int(n)), 256>>>(static_cast<const float*>(d_x),
                                              static_cast<float*>(d_y), int(n));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool mul(std::span<const float> a, std::span<const float> b, std::span<float> y) {
  if (!available()) return false;
  const uint64_t n = a.size();
  if (b.size() != n || y.size() < n) return false;
  void* d_a = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  void* d_b = grow_scratch(g_db, g_db_bytes, n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, n * sizeof(float));
  if (!d_a || !d_b || !d_y) return false;
  if (cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  mul_kernel<<<grid_for(int(n)), 256>>>(static_cast<const float*>(d_a),
                                        static_cast<const float*>(d_b),
                                        static_cast<float*>(d_y), int(n));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool add(std::span<const float> a, std::span<const float> b, std::span<float> y) {
  if (!available()) return false;
  const uint64_t n = a.size();
  if (b.size() != n || y.size() < n) return false;
  void* d_a = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  void* d_b = grow_scratch(g_db, g_db_bytes, n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, n * sizeof(float));
  if (!d_a || !d_b || !d_y) return false;
  if (cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  add_kernel<<<grid_for(int(n)), 256>>>(static_cast<const float*>(d_a),
                                        static_cast<const float*>(d_b),
                                        static_cast<float*>(d_y), int(n));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool scale(std::span<const float> x, float s, std::span<float> y) {
  if (!available()) return false;
  const uint64_t n = x.size();
  if (y.size() < n) return false;
  void* d_x = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, n * sizeof(float));
  if (!d_x || !d_y) return false;
  if (cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  scale_kernel<<<grid_for(int(n)), 256>>>(static_cast<const float*>(d_x), s,
                                          static_cast<float*>(d_y), int(n));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool cast_fp16(std::span<float> v) {
  if (!available()) return false;
  const uint64_t n = v.size();
  void* d_v = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  if (!d_v) return false;
  if (cudaMemcpy(d_v, v.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  cast_fp16_kernel<<<grid_for(int(n)), 256>>>(static_cast<float*>(d_v), int(n));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(v.data(), d_v, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool rope_neox(std::span<float> x, uint64_t pos, float theta_base, float freq_scale,
               std::span<const float> freq_factors) {
  if (!available()) return false;
  const size_t n = x.size();
  if (n == 0 || (n & 1) != 0) return false;
  const size_t half = n / 2;
  if (!freq_factors.empty() && freq_factors.size() < half) return false;

  void* d_x = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  if (!d_x) return false;
  const float* d_ff = nullptr;
  if (!freq_factors.empty()) {
    d_ff = static_cast<const float*>(grow_scratch(g_dff, g_dff_bytes, half * sizeof(float)));
    if (!d_ff) return false;
    if (cudaMemcpy(const_cast<float*>(d_ff), freq_factors.data(),
                   half * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
  }
  if (cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  const float log_base = std::log(theta_base);  // matches CPU's float(std::log(base))
  rope_neox_kernel<<<grid_for(int(half)), 256>>>(static_cast<float*>(d_x), int(half),
                                                 int(n), pos, freq_scale, log_base, d_ff);
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool rope_neox_heads(std::span<float> x, uint64_t head_dim, uint64_t pos,
                     float theta_base, float freq_scale,
                     std::span<const float> freq_factors) {
  if (!available()) return false;
  const size_t n = x.size();
  if (head_dim == 0 || (head_dim & 1) != 0 || n == 0 || (n % head_dim) != 0)
    return false;
  const size_t half = head_dim / 2;
  if (!freq_factors.empty() && freq_factors.size() < half) return false;

  void* d_x = grow_scratch(g_dx, g_dx_bytes, n * sizeof(float));
  if (!d_x) return false;
  const float* d_ff = nullptr;
  if (!freq_factors.empty()) {
    d_ff = static_cast<const float*>(grow_scratch(g_dff, g_dff_bytes, half * sizeof(float)));
    if (!d_ff) return false;
    if (cudaMemcpy(const_cast<float*>(d_ff), freq_factors.data(),
                   half * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
  }
  if (cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  const float log_base = std::log(theta_base);
  rope_neox_heads_kernel<<<grid_for(int(n)), 256>>>(
      static_cast<float*>(d_x), int(head_dim), int(n), pos, freq_scale, log_base,
      d_ff);
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(x.data(), d_x, n * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool matvec_f32(std::span<const float> W, uint64_t cols, std::span<const float> x,
                std::span<float> y) {
  if (!available()) return false;
  const uint64_t rows = y.size();
  if (cols == 0 || W.size() < cols * rows || x.size() < cols) return false;
  const void* d_w = upload_bytes(W.data(), cols * rows * sizeof(float));
  if (!d_w) return false;
  void* d_x = grow_scratch(g_dx, g_dx_bytes, cols * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, rows * sizeof(float));
  if (!d_x || !d_y) return false;
  if (cudaMemcpy(d_x, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  if ((cols & 3) == 0) {
    const int r = int(rows);
    const int k_split = gemv_k_split(r);
    if (k_split > 1) {
      float* partial = static_cast<float*>(
          grow_scratch(g_partial, g_partial_bytes,
                       (size_t)r * k_split * sizeof(float)));
      if (!partial) return false;
      gemv_dense_f32_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32>>>(
          static_cast<const float*>(d_w), static_cast<const float*>(d_x),
          partial, int(cols), r, k_split);
      gemv_reduce_kernel<<<grid_for(r), 256>>>(partial,
                                               static_cast<float*>(d_y), r,
                                               k_split);
    } else {
      gemv_dense_f32_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32>>>(
          static_cast<const float*>(d_w), static_cast<const float*>(d_x),
          static_cast<float*>(d_y), int(cols), r, k_split);
    }
  } else {
    gemv_dense_f32<<<grid_for(int(rows)), 256>>>(static_cast<const float*>(d_w),
                                                 static_cast<const float*>(d_x),
                                                 static_cast<float*>(d_y), int(cols),
                                                 int(rows));
  }
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool matvec_bf16(std::span<const uint16_t> W, uint64_t cols, std::span<const float> x,
                 std::span<float> y) {
  if (!available()) return false;
  const uint64_t rows = y.size();
  if (cols == 0 || W.size() < cols * rows || x.size() < cols) return false;
  const void* d_w = upload_bytes(W.data(), cols * rows * sizeof(uint16_t));
  if (!d_w) return false;
  void* d_x = grow_scratch(g_dx, g_dx_bytes, cols * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, rows * sizeof(float));
  if (!d_x || !d_y) return false;
  if (cudaMemcpy(d_x, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  if ((cols & 7) == 0) {
    const int r = int(rows);
    const int k_split = gemv_k_split(r);
    if (k_split > 1) {
      float* partial = static_cast<float*>(
          grow_scratch(g_partial, g_partial_bytes,
                       (size_t)r * k_split * sizeof(float)));
      if (!partial) return false;
      gemv_bf16_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32>>>(
          static_cast<const uint16_t*>(d_w), static_cast<const float*>(d_x),
          partial, int(cols), r, k_split);
      gemv_reduce_kernel<<<grid_for(r), 256>>>(partial,
                                               static_cast<float*>(d_y), r,
                                               k_split);
    } else {
      gemv_bf16_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32>>>(
          static_cast<const uint16_t*>(d_w), static_cast<const float*>(d_x),
          static_cast<float*>(d_y), int(cols), r, k_split);
    }
  } else {
    gemv_bf16<<<grid_for(int(rows)), 256>>>(static_cast<const uint16_t*>(d_w),
                                            static_cast<const float*>(d_x),
                                            static_cast<float*>(d_y), int(cols), int(rows));
  }
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

bool flash_attention(std::span<const float> q, uint64_t n_q,
                     std::span<const float> k, std::span<const float> v,
                     uint64_t n_kv, uint64_t n_heads_q, uint64_t n_heads_kv,
                     uint64_t head_dim, uint64_t sliding_window, float scale,
                     std::span<float> out) {
  if (!available()) return false;
  if (n_q == 0 || n_kv == 0 || n_heads_q == 0 || n_heads_kv == 0 || head_dim == 0)
    return false;
  if (n_heads_q % n_heads_kv != 0 || head_dim % 32 != 0 || head_dim > 512 ||
      n_heads_q > 32)
    return false;
  const size_t qn = (size_t)n_q * n_heads_q * head_dim;
  const size_t kvn = (size_t)n_kv * n_heads_kv * head_dim;
  if (q.size() < qn || k.size() < kvn || v.size() < kvn || out.size() < qn)
    return false;
  void* d_q = grow_scratch(g_dx, g_dx_bytes, qn * sizeof(float));
  void* d_k = grow_scratch(g_db, g_db_bytes, kvn * sizeof(float));
  void* d_v = grow_scratch(g_dy, g_dy_bytes, kvn * sizeof(float));
  void* d_o = grow_scratch(g_dff, g_dff_bytes, qn * sizeof(float));
  if (!d_q || !d_k || !d_v || !d_o) return false;
  if (cudaMemcpy(d_q, q.data(), qn * sizeof(float), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_k, k.data(), kvn * sizeof(float), cudaMemcpyHostToDevice) !=
          cudaSuccess ||
      cudaMemcpy(d_v, v.data(), kvn * sizeof(float), cudaMemcpyHostToDevice) !=
          cudaSuccess)
    return false;
  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  flash_attention_kernel<<<(unsigned)n_q, (unsigned)(n_heads_q * 32), smem>>>(
      static_cast<const float*>(d_q), static_cast<const float*>(d_k),
      static_cast<const float*>(d_v), (int)n_kv, (int)n_heads_q, (int)n_heads_kv,
      (int)head_dim, sliding_window, scale, static_cast<float*>(d_o));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(out.data(), d_o, qn * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

// --- device-resident entry points (Phase 6a Option A) -----------------------
// These mirror the public wrappers above but take/return device pointers and
// never copy activations to/from the host. Weight operands are host spans,
// uploaded once and cached by upload_bytes. The caller owns the device buffers
// and chains these back-to-back so activations stay resident.

bool rms_norm_dev(std::span<const float> weight, const float* x, float eps,
                  float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  if (!weight.empty() && weight.size() < (size_t)n) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(upload_bytes(weight.data(), n * sizeof(float)));
    if (!d_w) return false;
  }
  if (!g_sum && cudaMalloc(reinterpret_cast<void**>(&g_sum), sizeof(double)) != cudaSuccess)
    return false;
  if (!g_inv && cudaMalloc(reinterpret_cast<void**>(&g_inv), sizeof(float)) != cudaSuccess)
    return false;
  if (cudaMemset(g_sum, 0, sizeof(double)) != cudaSuccess) return false;
  rms_sum_sq_kernel<<<1, 256>>>(x, n, g_sum);
  if (cudaGetLastError() != cudaSuccess) return false;
  rms_inv_kernel<<<1, 1>>>(g_sum, n, eps, g_inv);
  if (cudaGetLastError() != cudaSuccess) return false;
  rms_apply_dev_kernel<<<grid_for(n), 256>>>(x, d_w, n, g_inv, y);
  return cudaGetLastError() == cudaSuccess;
}

bool rms_norm_heads_dev(std::span<const float> weight, const float* x, float eps,
                        float* y, int head_dim, int heads) {
  if (!available()) return false;
  if (head_dim <= 0 || heads <= 0 || !x || !y) return false;
  if (!weight.empty() && weight.size() < (size_t)head_dim) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(upload_bytes(weight.data(), head_dim * sizeof(float)));
    if (!d_w) return false;
  }
  if (heads > g_head_cap) {
    if (g_head_sum) cudaFree(g_head_sum);
    if (g_head_inv) cudaFree(g_head_inv);
    g_head_sum = nullptr;
    g_head_inv = nullptr;
    g_head_cap = 0;
    if (cudaMalloc(reinterpret_cast<void**>(&g_head_sum), heads * sizeof(double)) != cudaSuccess)
      return false;
    if (cudaMalloc(reinterpret_cast<void**>(&g_head_inv), heads * sizeof(float)) != cudaSuccess)
      return false;
    g_head_cap = heads;
  }
  if (cudaMemset(g_head_sum, 0, heads * sizeof(double)) != cudaSuccess) return false;
  const int n = head_dim * heads;
  rms_heads_sum_kernel<<<grid_for(heads), 256>>>(x, head_dim, heads, g_head_sum);
  if (cudaGetLastError() != cudaSuccess) return false;
  rms_heads_inv_kernel<<<grid_for(heads), 256>>>(g_head_sum, head_dim, eps, g_head_inv, heads);
  if (cudaGetLastError() != cudaSuccess) return false;
  rms_heads_apply_kernel<<<grid_for(n), 256>>>(x, d_w, head_dim, heads, g_head_inv, y);
  return cudaGetLastError() == cudaSuccess;
}

bool gelu_fp16_dev(const float* x, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  gelu_fp16_kernel<<<grid_for(n), 256>>>(x, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool mul_dev(const float* a, const float* b, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !a || !b || !y) return false;
  mul_kernel<<<grid_for(n), 256>>>(a, b, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool add_dev(const float* a, const float* b, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !a || !b || !y) return false;
  add_kernel<<<grid_for(n), 256>>>(a, b, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool scale_dev(const float* x, float s, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  scale_kernel<<<grid_for(n), 256>>>(x, s, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool cast_fp16_dev(float* v, int n) {
  if (!available()) return false;
  if (n <= 0 || !v) return false;
  cast_fp16_kernel<<<grid_for(n), 256>>>(v, n);
  return cudaGetLastError() == cudaSuccess;
}

bool cast_bf16_dev(float* v, int n) {
  if (!available()) return false;
  if (n <= 0 || !v) return false;
  cast_bf16_kernel<<<grid_for(n), 256>>>(v, n);
  return cudaGetLastError() == cudaSuccess;
}

bool rope_neox_heads_dev(float* x, int head_dim, int heads, uint64_t pos,
                         float base, float freq_scale,
                         std::span<const float> freq_factors) {
  if (!available()) return false;
  if (head_dim <= 0 || heads <= 0 || !x) return false;
  const int n = head_dim * heads;
  const int half = head_dim >> 1;
  if ((head_dim & 1) != 0) return false;
  if (!freq_factors.empty() && freq_factors.size() < (size_t)half) return false;
  const float* d_ff = nullptr;
  if (!freq_factors.empty()) {
    d_ff = static_cast<const float*>(upload_bytes(freq_factors.data(), half * sizeof(float)));
    if (!d_ff) return false;
  }
  const float log_base = std::log(base);
  rope_neox_heads_kernel<<<grid_for(n), 256>>>(x, head_dim, n, pos, freq_scale, log_base, d_ff);
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_f32_dev(std::span<const float> W, int cols, const float* x,
                    float* y, int rows) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const float* d_w = static_cast<const float*>(upload_bytes(W.data(), (size_t)cols * rows * sizeof(float)));
  if (!d_w) return false;
  if ((cols & 3) == 0) {
    const int k_split = gemv_k_split(rows);
    if (k_split > 1) {
      float* partial = static_cast<float*>(
          grow_scratch(g_partial, g_partial_bytes,
                       (size_t)rows * k_split * sizeof(float)));
      if (!partial) return false;
      gemv_dense_f32_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32>>>(
          d_w, x, partial, cols, rows, k_split);
      gemv_reduce_kernel<<<grid_for(rows), 256>>>(partial, y, rows, k_split);
    } else {
      gemv_dense_f32_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32>>>(
          d_w, x, y, cols, rows, k_split);
    }
  } else {
    gemv_dense_f32<<<grid_for(rows), 256>>>(d_w, x, y, cols, rows);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_bf16_dev(std::span<const uint16_t> W, int cols, const float* x,
                     float* y, int rows) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const uint16_t* d_w = static_cast<const uint16_t*>(upload_bytes(W.data(), (size_t)cols * rows * sizeof(uint16_t)));
  if (!d_w) return false;
  if ((cols & 7) == 0) {
    const int k_split = gemv_k_split(rows);
    if (k_split > 1) {
      float* partial = static_cast<float*>(
          grow_scratch(g_partial, g_partial_bytes,
                       (size_t)rows * k_split * sizeof(float)));
      if (!partial) return false;
      gemv_bf16_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32>>>(
          d_w, x, partial, cols, rows, k_split);
      gemv_reduce_kernel<<<grid_for(rows), 256>>>(partial, y, rows, k_split);
    } else {
      gemv_bf16_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32>>>(
          d_w, x, y, cols, rows, k_split);
    }
  } else {
    gemv_bf16<<<grid_for(rows), 256>>>(d_w, x, y, cols, rows);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool gqa_broadcast_dev(const float* src, float* dst, int n_q, int n_kv, int head_dim) {
  if (!available()) return false;
  if (n_q <= 0 || n_kv <= 0 || head_dim <= 0 || !src || !dst) return false;
  const int n = n_q * head_dim;
  gqa_broadcast_kernel<<<grid_for(n), 256>>>(src, dst, n_q, n_kv, head_dim);
  return cudaGetLastError() == cudaSuccess;
}

bool decode_attention_dev(const float* q, const float* k_cache,
                          const float* v_cache, uint64_t pos, uint64_t n_slots,
                          int n_heads_q, int n_heads_kv, int head_dim,
                          uint64_t sliding_window, float scale, float* out) {
  if (!available()) return false;
  if (n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || !q || !k_cache ||
      !v_cache || !out || n_slots == 0)
    return false;
  const uint64_t start =
      (sliding_window > 0 && pos >= sliding_window) ? pos - sliding_window + 1 : 0;
  const uint64_t n_keys = pos - start + 1;
  // One warp per query head (n_heads_q <= 32 for Gemma 4). Scores for all heads
  // are staged in dynamic shared memory, sized to n_heads_q * n_keys floats.
  const size_t smem = (size_t)n_heads_q * n_keys * sizeof(float);
  decode_attention_kernel<<<1, n_heads_q * 32, smem>>>(
      q, k_cache, v_cache, start, n_keys, n_slots, n_heads_q, n_heads_kv,
      head_dim, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

bool flash_attention_dev(const float* q, const float* k, const float* v,
                         int n_q, int n_kv, int n_heads_q, int n_heads_kv,
                         int head_dim, uint64_t sliding_window, float scale,
                         float* out) {
  if (!available()) return false;
  if (n_q <= 0 || n_kv <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 ||
      head_dim <= 0 || !q || !k || !v || !out)
    return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  if (head_dim % 32 != 0 || head_dim > 512) return false;
  if (n_heads_q > 32) return false;  // one warp per query head (Gemma: 8)
  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  flash_attention_kernel<<<n_q, n_heads_q * 32, smem>>>(
      q, k, v, n_kv, n_heads_q, n_heads_kv, head_dim, sliding_window, scale,
      out);
  return cudaGetLastError() == cudaSuccess;
}

bool layer_combine_dev(const float* proj, const float* ple, float ple_scale,
                       float combine_scale, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !proj || !ple || !y) return false;
  layer_combine_kernel<<<grid_for(n), 256>>>(proj, ple, ple_scale, combine_scale,
                                             y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool argmax_dev(const float* x, int n, float* out) {
  if (!available()) return false;
  if (n <= 0 || !x || !out) return false;
  argmax_kernel<<<1, 1024>>>(x, out, n);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace sonicboom::nn::cuda
