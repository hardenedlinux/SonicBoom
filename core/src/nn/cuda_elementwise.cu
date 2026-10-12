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
#include "cuda_stream.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
#include <string_view>
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

// Current stream for device-resident kernels (see cuda_stream.h).
inline cudaStream_t cur_stream() { return ::sonicboom::cuda::sb_current_stream(); }

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

// Device-position variant of rope_neox_heads_kernel: `pos` is a DEVICE uint64_t
// read in-kernel, so the position is an input to a capturable CUDA Graph rather
// than a value baked into the node at capture time.
__global__ void rope_neox_heads_pos_kernel(float* __restrict__ x, int head_dim,
                                           int n,
                                           const uint64_t* __restrict__ pos,
                                           float freq_scale, float log_base,
                                           const float* __restrict__ freq_factors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int half = head_dim >> 1;
  const int off = i % head_dim;
  if (off >= half) return;
  float theta = freq_scale * float(*pos) *
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

// Single-pass fused rms_norm: one kernel computes the sum-of-squares (the same
// reduction algorithm as rms_sum_sq_kernel — strided double partials, warp
// shuffle, shared block reduction), derives inv = 1/sqrtf(float(sum/n) + eps),
// and applies it, in a single launch. This removes the separate rms_inv_kernel
// (1-thread) + rms_apply_dev_kernel launches and the g_sum/g_inv device scalars
// + memset from the decode path — for a ~2560-element vector the three-launch
// sequence is launch-latency-bound, so collapsing it to one launch is a net win.
// The 1024-thread block widens the reduction tree (32 warps vs the sum kernel's
// 8), but that reordering is many orders below the fp32 rounding in the inv
// conversion (same argument as rms_sum_sq_kernel), so the result is bit-identical
// in practice — confirmed by the unchanged 64-step token stream.
__global__ void rms_norm_fused_kernel(const float* __restrict__ x,
                                      const float* __restrict__ weight, int n,
                                      float eps, float* __restrict__ y) {
  const int tid = threadIdx.x;
  double s = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = x[i];
    s += double(v * v);
  }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ double warp_part[32];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) warp_part[warp] = s;
  __syncthreads();
  __shared__ float inv;
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    double t = (lane < nwarps) ? warp_part[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
    if (lane == 0) inv = 1.0f / sqrtf(float(t / double(n)) + eps);
  }
  __syncthreads();
  const float iv = inv;
  for (int i = tid; i < n; i += blockDim.x)
    y[i] = weight ? x[i] * iv * weight[i] : x[i] * iv;
}

// Per-head rms_norm fused (T4): one block per head. The sum-of-squares is a
// block-wide parallel reduction (each thread accumulates a strided slice of the
// head, then warp + block tree-reduce) instead of thread 0 serializing the whole
// head_dim — the serial form left the other 255 threads idle and was bound by
// the single thread's dependent double-add chain. The double accumulation and
// the fp32 rounding at inv = 1/sqrtf(float(sum/head_dim)+eps) keep the result
// within the fp32 tolerance of the CPU rms_norm_heads reference (the reordering
// is far below the fp32 rounding applied to sum/head_dim). The apply is
// elementwise so spreading it across the block does not change the result.
__global__ void rms_heads_fused_kernel(const float* __restrict__ x,
                                       const float* __restrict__ weight,
                                       int head_dim, float eps,
                                       float* __restrict__ y) {
  const int h = blockIdx.x;
  const float* __restrict__ xh = x + (size_t)h * head_dim;
  float* __restrict__ yh = y + (size_t)h * head_dim;
  const int tid = threadIdx.x;
  double s = 0.0;
  for (int i = tid; i < head_dim; i += blockDim.x) {
    const float v = xh[i];
    s += double(v * v);
  }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  __shared__ double warp_part[32];
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) warp_part[warp] = s;
  __syncthreads();
  __shared__ float inv;
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    double t = (lane < nwarps) ? warp_part[lane] : 0.0;
    for (int o = 16; o > 0; o >>= 1) t += __shfl_down_sync(0xffffffffu, t, o);
    if (lane == 0) inv = 1.0f / sqrtf(float(t / double(head_dim)) + eps);
  }
  __syncthreads();
  const float iv = inv;
  for (int i = tid; i < head_dim; i += blockDim.x)
    yh[i] = weight ? xh[i] * iv * weight[i] : xh[i] * iv;
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

// Split-KV incremental decode attention (flash-decoding). One block per
// (chunk, GQA group): the KV sequence is split into `chunk`-key slices, and
// within each slice the n_heads_kv GQA groups run as separate blocks so each KV
// head is read once and shared by its group_q query heads. Each block computes a
// partial (local max m, local sum l, V-weighted output o) per query head over
// its slice, and a second kernel merges the slice partials in fixed order.
// Splitting the KV sequence across blocks is what lets the decode step use more
// than one SM: the single-block form was memory-latency bound (63% of warp
// cycles stalled on the L1TEX K/V scoreboard, 16.7% achieved occupancy, 1 grid
// block for 20 SMs). The chunk size is computed at runtime (see
// decode_attention_dev) from n_keys and the SM count, so short context still
// fills every SM instead of running a handful of blocks.
//
// Numerics: fmaxf is exact, so each chunk's local max is order-independent; the
// merge rescales by exp(chunk_max - global_max) and sums the (l, o) partials in
// fixed chunk order, so the whole result is deterministic (no float atomics).
// It differs from the single-pass CPU reference only in fp32 summation order
// (~1e-4 relative), inside the existing fp32 tolerance.
__global__ void decode_attention_split_kernel(const float* __restrict__ q,
                                              const float* __restrict__ k_cache,
                                              const float* __restrict__ v_cache,
                                              const uint64_t* __restrict__ pos,
                                              uint64_t n_slots, uint64_t chunk,
                                              uint64_t sliding_window,
                                              int n_heads_q, int n_heads_kv,
                                              int head_dim, float scale,
                                              float* __restrict__ partial) {
  extern __shared__ float scores[];  // [group_q * chunk]
  const int group_q = n_heads_q / n_heads_kv;  // query heads sharing one KV head
  const int kv_group = blockIdx.x % n_heads_kv;
  const uint64_t chunk_idx = blockIdx.x / n_heads_kv;
  const int lane = threadIdx.x & 31;
  const int hg = threadIdx.x >> 5;   // query head within this GQA group
  if (hg >= group_q) return;
  const int h = kv_group * group_q + hg;  // global query head
  const int kv_h = kv_group;
  const float* qrow = q + (uint64_t)h * head_dim;
  float* srow = scores + (uint64_t)hg * chunk;

  // Device-resident position: the actual key window for this step, so a single
  // fixed launch config (grid/smem/chunk) replays for every position.
  const uint64_t p = *pos;
  const uint64_t start =
      (sliding_window > 0 && p >= sliding_window) ? p - sliding_window + 1 : 0;
  const uint64_t n_keys = p - start + 1;

  const uint64_t c0 = chunk_idx * chunk;
  if (c0 >= n_keys) return;  // neutral chunk: the merge only reads chunks < n_active
  const uint64_t cn = (c0 + chunk < n_keys) ? chunk : (n_keys - c0);

  // Phase 1: one q·k dot per key, warp-cooperative.
  for (uint64_t j = 0; j < cn; ++j) {
    const uint64_t slot = (start + c0 + j) % n_slots;
    const float* krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    float d = 0.0f;
    for (int i = lane; i < head_dim; i += 32) d += qrow[i] * krow[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      d += __shfl_down_sync(0xffffffffu, d, o);
    if (lane == 0) srow[j] = d * scale;
  }
  __syncwarp();

  // Phase 2: local max (exact), exp once, local sum.
  float m = -1e30f;
  for (uint64_t j = lane; j < cn; j += 32) m = fmaxf(m, srow[j]);
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
  m = __shfl_sync(0xffffffffu, m, 0);

  float l = 0.0f;
  for (uint64_t j = lane; j < cn; j += 32) {
    const float e = expf(srow[j] - m);
    srow[j] = e;
    l += e;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    l += __shfl_down_sync(0xffffffffu, l, o);
  l = __shfl_sync(0xffffffffu, l, 0);
  __syncwarp();

  // Phase 3: V-weighted sum over the chunk — each lane owns a strided slice of
  // the output row's dims.
  const int dims_per_lane = head_dim >> 5;  // head_dim/32, <= 16
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  for (uint64_t j = 0; j < cn; ++j) {
    const uint64_t slot = (start + c0 + j) % n_slots;
    const float* vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    const float p = srow[j];  // exp(score - local max)
    for (int i = 0; i < dims_per_lane; ++i) acc[i] += p * vrow[lane + i * 32];
  }

  // Write partial [m, l, o[0..head_dim-1]] for this head.
  float* pp = partial + chunk_idx * (uint64_t)n_heads_q * (head_dim + 2) +
              (uint64_t)h * (head_dim + 2);
  if (lane == 0) { pp[0] = m; pp[1] = l; }
  for (int i = 0; i < dims_per_lane; ++i) pp[2 + lane + i * 32] = acc[i];
}

// Merge the per-chunk partials in fixed order (flash-decoding online-softmax
// combine). One block per query head; threads split the head_dim outputs so each
// output element is accumulated independently. Deterministic: the chunk loop
// runs 0..n_chunks-1 in order and there are no atomics.
__global__ void decode_attention_merge_kernel(const float* __restrict__ partial,
                                              const uint64_t* __restrict__ pos,
                                              uint64_t chunk,
                                              uint64_t sliding_window,
                                              int n_chunks, int n_heads_q,
                                              int head_dim,
                                              float* __restrict__ out) {
  const int h = blockIdx.x;    // one block per query head
  const int t = threadIdx.x;   // one thread per output dim
  if (t >= head_dim) return;

  // Active chunk count from the device-resident position: the split kernel only
  // wrote chunks with c0 < n_keys, so merging only those needs no neutral-value
  // sentinel / partial memset. n_keys >= 1 always, so n_active >= 1.
  const uint64_t p = *pos;
  const uint64_t start =
      (sliding_window > 0 && p >= sliding_window) ? p - sliding_window + 1 : 0;
  const uint64_t n_keys = p - start + 1;
  const uint64_t n_active = (n_keys + chunk - 1) / chunk;
  const int nc = n_active < (uint64_t)n_chunks ? (int)n_active : n_chunks;

  // Global max over the active chunks. Every thread redundantly scans the same
  // max; the chunk count is small, so this is cheaper than a block reduction.
  float m = -1e30f;
  for (int c = 0; c < nc; ++c)
    m = fmaxf(m, partial[(uint64_t)c * n_heads_q * (head_dim + 2) +
                         (uint64_t)h * (head_dim + 2)]);

  float l = 0.0f;
  float acc = 0.0f;
  for (int c = 0; c < nc; ++c) {
    const float* pp = partial + (uint64_t)c * n_heads_q * (head_dim + 2) +
                      (uint64_t)h * (head_dim + 2);
    const float w = expf(pp[0] - m);
    l += w * pp[1];
    acc += w * pp[2 + t];
  }
  out[(uint64_t)h * head_dim + t] = acc / l;
}

// Append one token's K/V row (row_elems floats) into the flat KV cache at slot
// (pos % n_slots). Fixed grid == ceil(row_elems / 256); the slot offset is
// computed in-kernel from a device-resident pos, so it is CUDA-Graph-capturable.
__global__ void kv_append_kernel(float* __restrict__ dst,
                                 const float* __restrict__ src,
                                 const uint64_t* __restrict__ pos,
                                 uint64_t n_slots, int row_elems) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= row_elems) return;
  const uint64_t off = (*pos % n_slots) * (uint64_t)row_elems;
  dst[off + i] = src[i];
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

// Chunked-prefill flash attention: one warp per (query position, query head).
// The n queries are the chunk at absolute positions [start, start+n); query
// blockIdx.x == iq has absolute position p = start + iq and attends over
// [start_key, p] where start_key = max(0, p - window + 1). That range is split
// into two sources read in one online-softmax pass:
//   - [start_key, start): prior context already resident in the KV cache, slot
//     j % n_slots;
//   - [start, p]: the current chunk's own K/V in the batch buffer, index j-start.
// The chunk's cache append is deferred until after every layer (including the
// shared donors) has read it, so the cache still holds the prior context while
// this kernel runs. For start == 0 the prior range is empty and the accumulation
// order over the batch keys is identical to flash_attention_kernel (bit-identical).
__global__ void flash_attention_chunk_kernel(
    const float* __restrict__ q, const float* __restrict__ k_batch,
    const float* __restrict__ v_batch, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, uint64_t start, uint64_t n_slots,
    int n_heads_q, int n_heads_kv, int head_dim, uint64_t sliding_window,
    float scale, float* __restrict__ out) {
  extern __shared__ float s_sh[];  // [n_heads_q * FA_BK]
  const int lane = threadIdx.x & 31;
  const int h = threadIdx.x >> 5;  // warp == query head
  const int iq = blockIdx.x;       // query position within the chunk (0-based)
  if (h >= n_heads_q) return;
  const int kv_h = h * n_heads_kv / n_heads_q;
  const uint64_t p = start + (uint64_t)iq;  // absolute query position
  const float* qrow = q + ((uint64_t)iq * n_heads_q + h) * head_dim;
  float* tile = s_sh + h * FA_BK;

  float acc[16];
  #pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  float m = -1e30f, l = 0.0f;

  const uint64_t start_key =
      (sliding_window > 0 && p >= sliding_window) ? p - sliding_window + 1 : 0;

  for (uint64_t j0 = start_key; j0 < p + 1; j0 += FA_BK) {
    const uint64_t jend = (p + 1 < j0 + FA_BK) ? (p + 1) : (j0 + FA_BK);
    const int jj_count = int(jend - j0);

    // 1. warp-cooperative q·k dot per key; cache (prior) vs batch (chunk) source.
    for (int jj = 0; jj < jj_count; ++jj) {
      const uint64_t j = j0 + (uint64_t)jj;
      const float* krow;
      if (j < start) {
        const uint64_t slot = j % n_slots;
        krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
      } else {
        const uint64_t bi = j - start;
        krow = k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
      }
      float dot = 0.0f;
      for (int d = lane; d < head_dim; d += 32) dot += qrow[d] * krow[d];
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        dot += __shfl_down_sync(0xffffffffu, dot, o);
      if (lane == 0) tile[jj] = dot * scale;
    }
    __syncwarp();

    // 2. tile max (every key is unmasked, so a live tile is never all -1e30).
    float tile_max = -1e30f;
    for (int jj = lane; jj < jj_count; jj += 32)
      tile_max = fmaxf(tile_max, tile[jj]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, o));
    tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

    // 3. online-softmax rescale (m == -1e30f on the first tile => expf(-inf)=0).
    const float m_new = fmaxf(m, tile_max);
    const float rescale = expf(m - m_new);

    // 4. exp once, store p back into tile; warp-reduce the tile sum.
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

    // 5. V-weighted accumulate: acc[d] = acc[d]*rescale + sum_jj tile[jj]*v[j,d].
    __syncwarp();
    int di = 0;
    for (int d = lane; d < head_dim; d += 32, ++di) {
      float a = acc[di] * rescale;
      for (int jj = 0; jj < jj_count; ++jj) {
        const uint64_t j = j0 + (uint64_t)jj;
        const float* vrow;
        if (j < start) {
          const uint64_t slot = j % n_slots;
          vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
        } else {
          const uint64_t bi = j - start;
          vrow = v_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        }
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

// Device-position variant of flash_attention_chunk_kernel: `start` is a DEVICE
// uint64_t read in-kernel, so the chunk's absolute start position is an input to
// a capturable CUDA Graph rather than a value baked into the node at capture
// time. Identical numerics otherwise (the launch config is fixed by n/head
// shapes, never by start).
__global__ void flash_attention_chunk_pos_kernel(
    const float* __restrict__ q, const float* __restrict__ k_batch,
    const float* __restrict__ v_batch, const float* __restrict__ k_cache,
    const float* __restrict__ v_cache, const uint64_t* __restrict__ start_ptr,
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
      const float* krow;
      if (j < start) {
        const uint64_t slot = j % n_slots;
        krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
      } else {
        const uint64_t bi = j - start;
        krow = k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
      }
      float dot = 0.0f;
      for (int d = lane; d < head_dim; d += 32) dot += qrow[d] * krow[d];
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
        const float* vrow;
        if (j < start) {
          const uint64_t slot = j % n_slots;
          vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
        } else {
          const uint64_t bi = j - start;
          vrow = v_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        }
        a += tile[jj] * vrow[d];
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
// Tiled tensor-core flash attention (fp16 mma.m16n8k16) for chunked prefill.
//
// Replaces flash_attention_chunk_pos_kernel's one-warp-per-(position, head)
// scalar-fp32 walk (which re-reads K/V from global memory once per position and
// dots in scalar fp32) with a block-per-(KV head x QB-query-segment) tile:
//   - K and V are loaded into shared memory once per BK-key tile and reused by
//     every query head and every query position in the block;
//   - the q·k^T and p·v products use fp16 tensor cores (m16n8k16) with fp32
//     accumulation, so the per-position scalar dot becomes a matmul;
//   - the [n_q, n_kv] score matrix is still never materialized (online softmax);
//   - the ring slot j % n_slots is computed once per key load, not per dot.
//
// Block layout: blockIdx.y == KV head, blockIdx.x == QB-query segment (chunk
// relative). Each warp owns one m-tile (16 queries of one query head): with GQA
// query heads per KV head and QB/16 m-tiles per head there are GQA*(QB/16) m-
// tiles. NW warps per m-tile split the head_dim for the p·v product only
// (NW=1 for head_dim=256, NW=2 for head_dim=512) so the fp32 output accumulator
// stays at 16*(head_dim/NW)/32 = 128 floats/thread; with NW>1 each warp
// redundantly computes the full q·k^T + softmax (the score matrix is only 16x16
// per sub-tile) and skips any cross-warp reduction. Q is converted to fp16 in
// registers (round-to-nearest), matching the fp16-rounded K/V the caller already
// staged via cast_fp16; the running max / sum / output stay fp32.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t fa_pack2(float x, float y) {
  __half2 h = __floats2half2_rn(x, y);
  return *reinterpret_cast<uint32_t*>(&h);
}

// D(16x8 f32) += A(16x16 f16) x B(16x8 f16) — mma.m16n8k16.row.col.
__device__ __forceinline__ void fa_mma(float c[4], const uint32_t a[4],
                                       const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A-fragment (16x16 fp16) from the global Q buffer: rows = query positions
// qr..qr+15 (chunk-relative), cols = dim k0..k0+15, for query head hq. Rows
// beyond the chunk (>= n) are zero-filled so they read no OOB memory (their
// output is never written).
__device__ __forceinline__ void fa_load_q(uint32_t a[4], const float* __restrict__ q,
                                          int hq, int n_heads_q, int head_dim,
                                          int qr, int k0, int n) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const size_t row_stride = (size_t)n_heads_q * head_dim;
  const int r0 = qr + g;
  const int r1 = qr + g + 8;
  const bool v0 = r0 < n;
  const bool v1 = r1 < n;
  const float* p0 = q + (size_t)r0 * row_stride + (size_t)hq * head_dim + k0;
  const float* p1 = q + (size_t)r1 * row_stride + (size_t)hq * head_dim + k0;
  a[0] = fa_pack2(v0 ? p0[t * 2] : 0.0f, v0 ? p0[t * 2 + 1] : 0.0f);
  a[1] = fa_pack2(v1 ? p1[t * 2] : 0.0f, v1 ? p1[t * 2 + 1] : 0.0f);
  a[2] = fa_pack2(v0 ? p0[t * 2 + 8] : 0.0f, v0 ? p0[t * 2 + 9] : 0.0f);
  a[3] = fa_pack2(v1 ? p1[t * 2 + 8] : 0.0f, v1 ? p1[t * 2 + 9] : 0.0f);
}

// B-fragment (16x8 fp16) from the K tile [BK][head_dim]: the n-dimension is keys
// key_off..key_off+7 (g in the fragment), the k-dimension is head_dim k0..k0+15.
__device__ __forceinline__ void fa_load_k(uint32_t b[2], const __half* __restrict__ k_tile,
                                          int head_dim, int k0, int key_off) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const __half* p = k_tile + (key_off + g) * head_dim + k0;
  b[0] = *reinterpret_cast<const uint32_t*>(p + t * 2);
  b[1] = *reinterpret_cast<const uint32_t*>(p + t * 2 + 8);
}

// B-fragment (16x8 fp16) from the transposed V tile [head_dim][v_stride]: the
// n-dimension is head_dim n0..n0+7 (g), the k-dimension is keys key_off..+15.
__device__ __forceinline__ void fa_load_v(uint32_t b[2], const __half* __restrict__ v_tile,
                                          int v_stride, int n0, int key_off) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
  const __half* p = v_tile + (n0 + g) * v_stride + key_off;
  b[0] = *reinterpret_cast<const uint32_t*>(p + t * 2);
  b[1] = *reinterpret_cast<const uint32_t*>(p + t * 2 + 8);
}

// Butterfly max / sum over the 4 lanes of this thread's fragment group (the
// t = lane&3 lanes that collectively hold a 16-key row). width=4 segments the
// warp into the {4g..4g+3} groups.
__device__ __forceinline__ float fa_rmax4(float x) {
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1, 4));
  x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2, 4));
  return x;
}
__device__ __forceinline__ float fa_rsum4(float x) {
  x += __shfl_xor_sync(0xffffffffu, x, 1, 4);
  x += __shfl_xor_sync(0xffffffffu, x, 2, 4);
  return x;
}

template <int HEAD_DIM, int BK, int NW, int QB, int GQA>
__global__ void __launch_bounds__(NW * GQA * (QB / 16) * 32)
    flash_attention_tiled_kernel(
        const float* __restrict__ q, const float* __restrict__ k_batch,
        const float* __restrict__ v_batch, const float* __restrict__ k_cache,
        const float* __restrict__ v_cache, const uint64_t* __restrict__ start_ptr,
        uint64_t n_slots, int n_heads_q, int n_heads_kv, int n,
        uint64_t sliding_window, float scale, float* __restrict__ out) {
  constexpr int V_STRIDE = BK + 2;  // pad the transposed V tile against bank conflicts
  constexpr int N_TILES = HEAD_DIM / NW / 8;  // p·v n-tiles in this warp's dim slice
  extern __shared__ __half smem[];
  __half* k_tile = smem;                    // [BK][HEAD_DIM]
  __half* v_tile = smem + BK * HEAD_DIM;    // [HEAD_DIM][V_STRIDE] (transposed)

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;              // [0, NW*GQA*(QB/16))
  const int unit = warp / NW;                     // one (head, m-tile) unit
  const int head_idx = unit / (QB / 16);          // 0..GQA-1
  const int m_tile = unit % (QB / 16);            // 0..QB/16-1
  const int dim_slice = warp % NW;                // head_dim slice for p·v
  const int hq = blockIdx.y * GQA + head_idx;     // global query head
  const int qr = blockIdx.x * QB + m_tile * 16;   // chunk-relative query base
  const int dim0 = dim_slice * (HEAD_DIM / NW);

  const int g = lane >> 2;  // fragment group 0..7
  const int t = lane & 3;   // thread in group 0..3

  const uint64_t start = *start_ptr;

  // online-softmax state for the two rows this thread owns (g, g+8).
  float m[2] = {-1e30f, -1e30f};
  float l[2] = {0.0f, 0.0f};
  float o[N_TILES][4];
  #pragma unroll
  for (int i = 0; i < N_TILES; ++i)
    o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

  const uint64_t abs_pb = start + (uint64_t)blockIdx.x * QB;
  const int n_blk = (n - blockIdx.x * QB < QB) ? (n - blockIdx.x * QB) : QB;
  const uint64_t key_end = abs_pb + (uint64_t)n_blk - 1;  // last query (absolute)
  // key_start: earliest key any query in this segment attends.
  const uint64_t key_start =
      (sliding_window > 0 && abs_pb >= sliding_window)
          ? (abs_pb - sliding_window + 1)
          : 0;
  // Align the key-tile start down to a BK multiple so tile boundaries are global
  // (absolute-position) aligned, independent of how the prompt is split into
  // chunks. Otherwise the online-softmax rescaling points would shift with the
  // chunking, and prefix reuse would no longer reproduce the full prefill
  // bit-for-bit (the earlier keys a block loads before its earliest window are
  // masked and contribute 0, so the extra aligned tiles are no-ops).
  const uint64_t j0_start = (key_start / (uint64_t)BK) * (uint64_t)BK;

  const uint64_t p0 = start + (uint64_t)(qr + g);      // row g absolute position
  const uint64_t p1 = start + (uint64_t)(qr + g + 8);  // row g+8 absolute position
  const uint64_t low0 = (sliding_window > 0 && p0 >= sliding_window)
                            ? (p0 - sliding_window + 1)
                            : 0;
  const uint64_t low1 = (sliding_window > 0 && p1 >= sliding_window)
                            ? (p1 - sliding_window + 1)
                            : 0;

  for (uint64_t j0 = j0_start; j0 <= key_end; j0 += BK) {
    // ---- cooperative K/V tile load into smem (whole block) ----------------
    for (int idx = threadIdx.x; idx < BK * HEAD_DIM; idx += blockDim.x) {
      const int kk = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const uint64_t j = j0 + (uint64_t)kk;
      float kv = 0.0f;
      if (j <= key_end) {
        const float* row;
        if (j < start) {
          const uint64_t slot = j % n_slots;
          row = k_cache + (slot * (uint64_t)n_heads_kv + blockIdx.y) * HEAD_DIM;
        } else {
          const uint64_t bi = j - start;
          row = k_batch + (bi * (uint64_t)n_heads_kv + blockIdx.y) * HEAD_DIM;
        }
        kv = row[d];
      }
      k_tile[kk * HEAD_DIM + d] = __float2half_rn(kv);
    }
    for (int idx = threadIdx.x; idx < BK * HEAD_DIM; idx += blockDim.x) {
      const int kk = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const uint64_t j = j0 + (uint64_t)kk;
      float vv = 0.0f;
      if (j <= key_end) {
        const float* row;
        if (j < start) {
          const uint64_t slot = j % n_slots;
          row = v_cache + (slot * (uint64_t)n_heads_kv + blockIdx.y) * HEAD_DIM;
        } else {
          const uint64_t bi = j - start;
          row = v_batch + (bi * (uint64_t)n_heads_kv + blockIdx.y) * HEAD_DIM;
        }
        vv = row[d];
      }
      v_tile[d * V_STRIDE + kk] = __float2half_rn(vv);
    }
    __syncthreads();

    // ---- this warp's m-tile over the 16-key sub-tiles of the K tile -------
    for (int key_off = 0; key_off < BK; key_off += 16) {
      // 1. q·k^T over the full head_dim: S[16 queries][16 keys].
      float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
      for (int k0 = 0; k0 < HEAD_DIM; k0 += 16) {
        uint32_t aq[4];
        fa_load_q(aq, q, hq, n_heads_q, HEAD_DIM, qr, k0, n);
        uint32_t bk[2];
        fa_load_k(bk, k_tile, HEAD_DIM, k0, key_off);
        fa_mma(s0, aq, bk);
        fa_load_k(bk, k_tile, HEAD_DIM, k0, key_off + 8);
        fa_mma(s1, aq, bk);
      }

      // 2. scale + causal/window mask. Keys jb, jb+1 (s0) and jb+8, jb+9 (s1).
      const uint64_t jb = j0 + (uint64_t)key_off + (uint64_t)t * 2;
      s0[0] = (jb > p0 || jb < low0) ? -1e30f : s0[0] * scale;
      s0[1] = (jb + 1 > p0 || jb + 1 < low0) ? -1e30f : s0[1] * scale;
      s1[0] = (jb + 8 > p0 || jb + 8 < low0) ? -1e30f : s1[0] * scale;
      s1[1] = (jb + 9 > p0 || jb + 9 < low0) ? -1e30f : s1[1] * scale;
      s0[2] = (jb > p1 || jb < low1) ? -1e30f : s0[2] * scale;
      s0[3] = (jb + 1 > p1 || jb + 1 < low1) ? -1e30f : s0[3] * scale;
      s1[2] = (jb + 8 > p1 || jb + 8 < low1) ? -1e30f : s1[2] * scale;
      s1[3] = (jb + 9 > p1 || jb + 9 < low1) ? -1e30f : s1[3] * scale;

      // 3. online-softmax: row max over the 16 keys (reduce over the group).
      const float mx0 = fa_rmax4(fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1])));
      const float mx1 = fa_rmax4(fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3])));
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

      const float sum0 = fa_rsum4(p00 + p01 + p08 + p09);
      const float sum1 = fa_rsum4(p10 + p11 + p18 + p19);
      l[0] = l[0] * rs0 + sum0;
      l[1] = l[1] * rs1 + sum1;

      // 4. rescale the output accumulator, then p·v on this warp's dim slice.
      #pragma unroll
      for (int i = 0; i < N_TILES; ++i) {
        o[i][0] *= rs0;
        o[i][1] *= rs0;
        o[i][2] *= rs1;
        o[i][3] *= rs1;
      }
      uint32_t ap[4];
      ap[0] = fa_pack2(p00, p01);
      ap[1] = fa_pack2(p10, p11);
      ap[2] = fa_pack2(p08, p09);
      ap[3] = fa_pack2(p18, p19);
      #pragma unroll
      for (int i = 0; i < N_TILES; ++i) {
        uint32_t bv[2];
        fa_load_v(bv, v_tile, V_STRIDE, dim0 + i * 8, key_off);
        fa_mma(o[i], ap, bv);
      }
      m[0] = m_new0;
      m[1] = m_new1;
    }
    __syncthreads();
  }

  // ---- finalize: out[d] = o[d] / l, written only for in-chunk rows --------
  const float inv0 = 1.0f / l[0];
  const float inv1 = 1.0f / l[1];
  const int iq0 = qr + g;
  const int iq1 = qr + g + 8;
  if (iq0 < n) {
    float* orow = out + ((size_t)iq0 * n_heads_q + hq) * HEAD_DIM + dim0;
    #pragma unroll
    for (int i = 0; i < N_TILES; ++i) {
      orow[i * 8 + t * 2] = o[i][0] * inv0;
      orow[i * 8 + t * 2 + 1] = o[i][1] * inv0;
    }
  }
  if (iq1 < n) {
    float* orow = out + ((size_t)iq1 * n_heads_q + hq) * HEAD_DIM + dim0;
    #pragma unroll
    for (int i = 0; i < N_TILES; ++i) {
      orow[i * 8 + t * 2] = o[i][2] * inv1;
      orow[i * 8 + t * 2 + 1] = o[i][3] * inv1;
    }
  }
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
void* g_attn_partial = nullptr; // split-KV decode attention partials (n_chunks*n_heads_q*(head_dim+2) floats)
size_t g_attn_partial_bytes = 0;
void* g_xpack = nullptr;        // repacked activation column-block for the cuBLAS dense matvec ([cols, 128])
size_t g_xpack_bytes = 0;
void* g_outpack = nullptr;      // padded output column-block ([rows, 128])
size_t g_outpack_bytes = 0;
double* g_sum = nullptr;        // device-resident rms_norm sum (double, matches CPU)

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

// --- batched prefill kernels (the decode kernels generalized to n columns) ---

// Column rms_norm over a [dim, ncols] row-major activation (the batched prefill
// layout): normalizes each of the `ncols` columns over the `dim` rows. The
// previous form was one block per column with threads spanning dim, reading
// x[k*ncols+c] at stride ncols — every warp touched one cache line per lane, so
// the pass ran far above the memory-bound floor. This form makes consecutive
// lanes own consecutive columns (coalesced loads/stores) and splits the dim
// reduction across the 32 warps of a block (each warp sums a contiguous dim
// slice), then combines the per-slice partials in shared memory. Grid is
// (ncols+31)/32 blocks of 1024 threads; each block covers 32 columns. The double
// accumulation and fp32 rounding at inv keep each column within the fp32
// tolerance of the decode rms_norm_dev reference (the reorder is far below the
// fp32 rounding applied to sum/dim).
__global__ void rms_norm_cols_kernel(const float* __restrict__ x,
                                     const float* __restrict__ weight, int dim,
                                     int ncols, float eps,
                                     float* __restrict__ y) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int nwarps = blockDim.x >> 5;
  const int c = blockIdx.x * 32 + lane;
  const bool valid = c < ncols;
  // contiguous dim slice owned by this warp
  const int chunk = (dim + nwarps - 1) / nwarps;
  const int k0 = warp * chunk;
  const int k1 = (k0 + chunk < dim) ? (k0 + chunk) : dim;
  double s = 0.0;
  if (valid) {
    for (int k = k0; k < k1; ++k) {
      const float v = x[(size_t)k * ncols + c];
      s += double(v * v);
    }
  }
  // combine this warp's partial per-column sum across the dim-slice warps.
  __shared__ double warp_part[32][32];  // [nwarps][32 columns]
  warp_part[warp][lane] = s;
  __syncthreads();
  __shared__ float inv[32];
  if (warp == 0) {
    double t = warp_part[0][lane];
    for (int w = 1; w < nwarps; ++w) t += warp_part[w][lane];
    inv[lane] = 1.0f / sqrtf(float(t / double(dim)) + eps);
  }
  __syncthreads();
  if (!valid) return;
  const float iv = inv[lane];
  for (int k = k0; k < k1; ++k) {
    const size_t idx = (size_t)k * ncols + c;
    y[idx] = weight ? x[idx] * iv * weight[k] : x[idx] * iv;
  }
}

// Batched multi-head RoPE: like rope_neox_heads_kernel but the absolute position
// is per-position — pos = start_pos + i / (heads_per_pos * head_dim) — so one
// launch rotates every position of a [n_pos, heads_per_pos*head_dim] buffer. The
// per-element rotation (off = i % head_dim, paired with off + head_dim/2) is
// unchanged from the single-position kernel.
__global__ void rope_neox_heads_batched_kernel(float* __restrict__ x, int n,
                                               int head_dim, int heads_per_pos,
                                               uint64_t start_pos, float freq_scale,
                                               float log_base,
                                               const float* __restrict__ freq_factors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int half = head_dim >> 1;
  const int off = i % head_dim;
  if (off >= half) return;
  const uint64_t pos = start_pos + (uint64_t)(i / (heads_per_pos * head_dim));
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

// Device-position variant of rope_neox_heads_batched_kernel: `start_pos` is a
// DEVICE uint64_t read in-kernel, so the chunk's start position is an input to a
// capturable CUDA Graph rather than a value baked into the node at capture time.
__global__ void rope_neox_heads_batched_pos_kernel(
    float* __restrict__ x, int n, int head_dim, int heads_per_pos,
    const uint64_t* __restrict__ start_pos, float freq_scale, float log_base,
    const float* __restrict__ freq_factors) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int half = head_dim >> 1;
  const int off = i % head_dim;
  if (off >= half) return;
  const uint64_t pos = *start_pos + (uint64_t)(i / (heads_per_pos * head_dim));
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

// Tiled transpose of a [rows, cols] row-major matrix -> [cols, rows]:
// dst[c*rows + r] = src[r*cols + c]. Same +1-pad tile as the K-quant matmul's
// transpose_tiled, kept here so the prefill Transpose op has no quant-file
// dependency.
__global__ void transpose_tiled_dev(const float* __restrict__ src,
                                    float* __restrict__ dst, int rows, int cols) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = src[(size_t)y * cols + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  if (x < rows && y < cols) dst[(size_t)y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

// Dense gemv over n columns: one thread per (row, col) pair. Consecutive threads
// span columns, so the activation loads x[i*ncols + c] are coalesced and the
// weight row W[row*cols + i] is broadcast across ncols threads.
__global__ void gemv_dense_f32_batched(const float* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int cols, int rows,
                                       int ncols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * ncols) return;
  const int row = idx / ncols;
  const int c = idx - row * ncols;
  const float* wrow = W + (size_t)row * cols;
  float acc = 0.0f;
  for (int i = 0; i < cols; ++i) acc += wrow[i] * x[(size_t)i * ncols + c];
  y[idx] = acc;
}

__global__ void gemv_bf16_batched(const uint16_t* __restrict__ W,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, int cols, int rows,
                                  int ncols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * ncols) return;
  const int row = idx / ncols;
  const int c = idx - row * ncols;
  const uint16_t* wrow = W + (size_t)row * cols;
  float acc = 0.0f;
  for (int i = 0; i < cols; ++i) acc += bf16_to_float(wrow[i]) * x[(size_t)i * ncols + c];
  y[idx] = acc;
}

// Fixed column-block width for the cuBLAS dense matvec (Phase C Step 3). Every
// cuBLAS call is shaped (m=128, k=cols, lda=ldc=128) regardless of the batch n,
// so the per-column numerics are independent of both n and the column offset —
// the property prefix-cache reuse relies on (the chunker splits full and reuse
// prefills into differently-sized segments at the same 512-aligned boundaries).
constexpr int kDenseCublasWidth = 128;

// Repack a [cols, ncols] activation slice (columns [c0, c0+chunk_n)) into a
// tight [cols, 128] block, zero-filling the trailing padding columns. Every
// chunk of the same activation lands at the same stride (128), so the GEMM shape
// (m, k, lda) is identical across chunks and across n values.
__global__ void pack_f32_chunk_kernel(const float* __restrict__ x,
                                      float* __restrict__ dst, int cols,
                                      int ncols, int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = cols * kDenseCublasWidth;
  if (idx >= total) return;
  const int row = idx / kDenseCublasWidth;
  const int p = idx - row * kDenseCublasWidth;
  dst[idx] = (p < chunk_n) ? x[(size_t)row * ncols + (c0 + p)] : 0.0f;
}

__global__ void pack_bf16_chunk_kernel(const float* __restrict__ x,
                                       uint16_t* __restrict__ dst, int cols,
                                       int ncols, int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = cols * kDenseCublasWidth;
  if (idx >= total) return;
  const int row = idx / kDenseCublasWidth;
  const int p = idx - row * kDenseCublasWidth;
  dst[idx] = (p < chunk_n) ? float_to_bf16(x[(size_t)row * ncols + (c0 + p)])
                           : uint16_t(0);
}

// Copy the first `chunk_n` columns of a [rows, 128] result block back into the
// [rows, ncols] output at column offset c0 (the padding columns are dropped).
__global__ void copyback_chunk_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst, int rows,
                                      int ncols, int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = rows * chunk_n;
  if (idx >= total) return;
  const int r = idx / chunk_n;
  const int p = idx - r * chunk_n;
  dst[(size_t)r * ncols + (c0 + p)] = src[(size_t)r * kDenseCublasWidth + p];
}

// Append n consecutive rows into the flat KV cache. Each thread copies one
// element of one row; the slot is (start_pos + row) % n_slots (a wrap-free run,
// so consecutive). src is [n, row_elems] position-major.
__global__ void kv_append_batched_kernel(float* __restrict__ dst,
                                         const float* __restrict__ src,
                                         uint64_t start_pos, uint64_t n_slots,
                                         int row_elems, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_elems * n;
  if (idx >= total) return;
  const int row = idx / row_elems;
  const int e = idx - row * row_elems;
  const uint64_t slot = (start_pos + (uint64_t)row) % n_slots;
  dst[slot * (uint64_t)row_elems + (uint64_t)e] = src[idx];
}

// Device-position variant of kv_append_batched_kernel: `start_pos` is a DEVICE
// uint64_t read in-kernel (CUDA Graphs for chunked prefill).
__global__ void kv_append_batched_pos_kernel(float* __restrict__ dst,
                                             const float* __restrict__ src,
                                             const uint64_t* __restrict__ start_pos,
                                             uint64_t n_slots, int row_elems,
                                             int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_elems * n;
  if (idx >= total) return;
  const int row = idx / row_elems;
  const int e = idx - row * row_elems;
  const uint64_t slot = (*start_pos + (uint64_t)row) % n_slots;
  dst[slot * (uint64_t)row_elems + (uint64_t)e] = src[idx];
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

// Cached SM count for the current device (used to size the decode-attention
// launch grid so short context still fills every SM). Falls back to a sane
// default if the query fails, so it never makes launch config invalid.
int sm_count() noexcept {
  static int n = [] {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 20;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) !=
        cudaSuccess)
      return 20;
    return v > 0 ? v : 20;
  }();
  return n;
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

  rms_sum_sq_kernel<<<1, 256, 0, cur_stream()>>>(static_cast<const float*>(d_x), int(n), g_sum);
  if (cudaGetLastError() != cudaSuccess) return false;
  double sum = 0.0;
  if (cudaMemcpy(&sum, g_sum, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;

  // Reproduce the CPU's mean/rsqrt in double (mean) then float (sqrt/recip),
  // matching norm.cpp bit-for-bit.
  const float inv = 1.0f / std::sqrt(float(sum / double(n)) + eps);
  rms_apply_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(static_cast<const float*>(d_x), d_w,
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
  gelu_fp16_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(static_cast<const float*>(d_x),
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
  mul_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(static_cast<const float*>(d_a),
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
  add_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(static_cast<const float*>(d_a),
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
  scale_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(static_cast<const float*>(d_x), s,
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
  cast_fp16_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(static_cast<float*>(d_v), int(n));
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
  rope_neox_kernel<<<grid_for(int(half)), 256, 0, cur_stream()>>>(static_cast<float*>(d_x), int(half),
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
  rope_neox_heads_kernel<<<grid_for(int(n)), 256, 0, cur_stream()>>>(
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
      gemv_dense_f32_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const float*>(d_w), static_cast<const float*>(d_x),
          partial, int(cols), r, k_split);
      gemv_reduce_kernel<<<grid_for(r), 256, 0, cur_stream()>>>(partial,
                                               static_cast<float*>(d_y), r,
                                               k_split);
    } else {
      gemv_dense_f32_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const float*>(d_w), static_cast<const float*>(d_x),
          static_cast<float*>(d_y), int(cols), r, k_split);
    }
  } else {
    gemv_dense_f32<<<grid_for(int(rows)), 256, 0, cur_stream()>>>(static_cast<const float*>(d_w),
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
      gemv_bf16_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const uint16_t*>(d_w), static_cast<const float*>(d_x),
          partial, int(cols), r, k_split);
      gemv_reduce_kernel<<<grid_for(r), 256, 0, cur_stream()>>>(partial,
                                               static_cast<float*>(d_y), r,
                                               k_split);
    } else {
      gemv_bf16_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const uint16_t*>(d_w), static_cast<const float*>(d_x),
          static_cast<float*>(d_y), int(cols), r, k_split);
    }
  } else {
    gemv_bf16<<<grid_for(int(rows)), 256, 0, cur_stream()>>>(static_cast<const uint16_t*>(d_w),
                                            static_cast<const float*>(d_x),
                                            static_cast<float*>(d_y), int(cols), int(rows));
  }
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

// Batched dense matvec (the prefill shape): Y[r*n+c] = sum_i W[r*cols+i] *
// X[i*n+c]. X is host row-major [cols, n] (n innermost), Y is host row-major
// [rows, n]. Uploads X, runs the device batched matvec (cuBLAS fixed-width
// blocks for every n, including n == 1), and copies Y back. Same
// contracts/failure semantics as matvec_f32.
bool matmul_f32(std::span<const float> W, uint64_t cols, std::span<const float> X,
                std::span<float> Y, uint64_t n) {
  if (!available()) return false;
  if (cols == 0 || n == 0 || X.size() < cols * n || Y.size() < n) return false;
  const uint64_t rows = Y.size() / n;
  if (rows == 0 || W.size() < cols * rows) return false;
  void* d_x = grow_scratch(g_dx, g_dx_bytes, cols * n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, rows * n * sizeof(float));
  if (!d_x || !d_y) return false;
  if (cudaMemcpy(d_x, X.data(), cols * n * sizeof(float), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return false;
  if (!matvec_f32_batched_dev(W, int(cols), static_cast<const float*>(d_x),
                              static_cast<float*>(d_y), int(rows), int(n)))
    return false;
  return cudaMemcpy(Y.data(), d_y, rows * n * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

// bf16 twin of matmul_f32 (W is raw uint16_t bf16). The activation is rounded
// through bf16 inside the cuBLAS path's repack, matching the prefill spine.
bool matmul_bf16(std::span<const uint16_t> W, uint64_t cols,
                 std::span<const float> X, std::span<float> Y, uint64_t n) {
  if (!available()) return false;
  if (cols == 0 || n == 0 || X.size() < cols * n || Y.size() < n) return false;
  const uint64_t rows = Y.size() / n;
  if (rows == 0 || W.size() < cols * rows) return false;
  void* d_x = grow_scratch(g_dx, g_dx_bytes, cols * n * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, rows * n * sizeof(float));
  if (!d_x || !d_y) return false;
  if (cudaMemcpy(d_x, X.data(), cols * n * sizeof(float), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return false;
  if (!matvec_bf16_batched_dev(W, int(cols), static_cast<const float*>(d_x),
                               static_cast<float*>(d_y), int(rows), int(n)))
    return false;
  return cudaMemcpy(Y.data(), d_y, rows * n * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
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
  flash_attention_kernel<<<(unsigned)n_q, (unsigned)(n_heads_q * 32), smem, cur_stream()>>>(
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
  rms_norm_fused_kernel<<<1, 1024, 0, cur_stream()>>>(x, d_w, n, eps, y);
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
  // One block per head; thread 0 reduces the sum, the block applies. head_dim
  // ≤ 256 keeps the whole apply in one pass (Gemma 4 head_dim is 256).
  rms_heads_fused_kernel<<<heads, 256, 0, cur_stream()>>>(x, d_w, head_dim, eps, y);
  return cudaGetLastError() == cudaSuccess;
}

bool gelu_fp16_dev(const float* x, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  gelu_fp16_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool mul_dev(const float* a, const float* b, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !a || !b || !y) return false;
  mul_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(a, b, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool add_dev(const float* a, const float* b, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !a || !b || !y) return false;
  add_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(a, b, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool scale_dev(const float* x, float s, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  scale_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, s, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool cast_fp16_dev(float* v, int n) {
  if (!available()) return false;
  if (n <= 0 || !v) return false;
  cast_fp16_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(v, n);
  return cudaGetLastError() == cudaSuccess;
}

bool cast_bf16_dev(float* v, int n) {
  if (!available()) return false;
  if (n <= 0 || !v) return false;
  cast_bf16_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(v, n);
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
  rope_neox_heads_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, head_dim, n, pos, freq_scale, log_base, d_ff);
  return cudaGetLastError() == cudaSuccess;
}

bool rope_neox_heads_dev_pos(float* x, int head_dim, int heads,
                             const uint64_t* pos, float base, float freq_scale,
                             std::span<const float> freq_factors) {
  if (!available()) return false;
  if (head_dim <= 0 || heads <= 0 || !x || !pos) return false;
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
  rope_neox_heads_pos_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(
      x, head_dim, n, pos, freq_scale, log_base, d_ff);
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
      gemv_dense_f32_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, partial, cols, rows, k_split);
      gemv_reduce_kernel<<<grid_for(rows), 256, 0, cur_stream()>>>(partial, y, rows, k_split);
    } else {
      gemv_dense_f32_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, y, cols, rows, k_split);
    }
  } else {
    gemv_dense_f32<<<grid_for(rows), 256, 0, cur_stream()>>>(d_w, x, y, cols, rows);
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
      gemv_bf16_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, partial, cols, rows, k_split);
      gemv_reduce_kernel<<<grid_for(rows), 256, 0, cur_stream()>>>(partial, y, rows, k_split);
    } else {
      gemv_bf16_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, y, cols, rows, k_split);
    }
  } else {
    gemv_bf16<<<grid_for(rows), 256, 0, cur_stream()>>>(d_w, x, y, cols, rows);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool gqa_broadcast_dev(const float* src, float* dst, int n_q, int n_kv, int head_dim) {
  if (!available()) return false;
  if (n_q <= 0 || n_kv <= 0 || head_dim <= 0 || !src || !dst) return false;
  const int n = n_q * head_dim;
  gqa_broadcast_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(src, dst, n_q, n_kv, head_dim);
  return cudaGetLastError() == cudaSuccess;
}

bool decode_attention_dev(const float* q, const float* k_cache,
                          const float* v_cache, const uint64_t* pos,
                          uint64_t window, uint64_t n_slots, int n_heads_q,
                          int n_heads_kv, int head_dim, uint64_t sliding_window,
                          float scale, float* out) {
  if (!available()) return false;
  if (n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || !q || !k_cache ||
      !v_cache || !out || !pos || n_slots == 0 || window == 0)
    return false;

  // Split-KV: n_chunks × n_kv_heads blocks each compute a per-head partial over
  // its slice of the KV sequence, then n_heads_q merge blocks combine them in
  // fixed order. The launch config is sized from the FIXED `window` (max key
  // count this graph replays for: sliding_window for SWA, n_ctx for global), so
  // grid/smem/chunk are stable for CUDA Graph capture; the split kernel computes
  // the real start/n_keys from *pos and early-exits chunks beyond n_keys.
  const int group_q = n_heads_q / n_heads_kv;
  const uint64_t target_chunks =
      std::max<uint64_t>(1, (uint64_t)sm_count() * 4 / (uint64_t)n_heads_kv);
  // chunk = ceil(window / target_chunks), but capped so a large n_ctx does not
  // serialize the whole KV sequence onto one block. Without the cap chunk grows
  // linearly with n_ctx (820 at n_ctx=32768), so at a short depth only one block
  // per KV head stays active and the split kernel becomes latency-bound
  // (measured split 1.95 -> 9.32 -> 20.36 ms at n_ctx 2048/8192/32768). Capping
  // lets the grid scale with window instead — ceil(window/chunk)*n_heads_kv —
  // with blocks beyond the current n_keys early-exiting (see the split kernel).
  // The cap is sized for the SWA window (512): chunk 16 keeps depth-512
  // attention at 32 active chunks (64 blocks, ~3.2/SM) for any n_ctx, matching
  // the SWA layers' parallelism (chunk 13 -> 80 blocks). A cap of 32 is a
  // wave-fill local minimum (16 chunks -> 32 blocks, 1.6/SM) and measured
  // ~2.4x slower split (3.20 vs 1.29 ms at depth 512). The trade-off of the
  // smaller chunk is a larger partial buffer (n_chunks*8*322 floats) and a
  // longer merge loop at full depth, both negligible at the SWA depth.
  constexpr uint64_t kMaxChunk = 16;
  const uint64_t chunk = std::max<uint64_t>(
      1, std::min((window + target_chunks - 1) / target_chunks, kMaxChunk));
  const uint64_t n_chunks = (window + chunk - 1) / chunk;

  const size_t partial_floats =
      (size_t)n_chunks * (size_t)n_heads_q * (size_t)(head_dim + 2);
  float* partial = static_cast<float*>(
      grow_scratch(g_attn_partial, g_attn_partial_bytes,
                   partial_floats * sizeof(float)));

  const size_t smem = (size_t)group_q * chunk * sizeof(float);
  decode_attention_split_kernel<<<(unsigned)(n_chunks * (uint64_t)n_heads_kv),
                                  group_q * 32, smem, cur_stream()>>>(
      q, k_cache, v_cache, pos, n_slots, chunk, sliding_window, n_heads_q,
      n_heads_kv, head_dim, scale, partial);
  decode_attention_merge_kernel<<<(unsigned)n_heads_q, head_dim, 0, cur_stream()>>>(
      partial, pos, chunk, sliding_window, (int)n_chunks, n_heads_q, head_dim,
      out);
  return cudaGetLastError() == cudaSuccess;
}

bool kv_append_dev(float* dst_k, float* dst_v, const float* src_k,
                   const float* src_v, const uint64_t* pos, uint64_t n_slots,
                   int row_elems) {
  if (!available()) return false;
  if (row_elems <= 0 || !dst_k || !dst_v || !src_k || !src_v || !pos ||
      n_slots == 0)
    return false;
  const int grid = grid_for(row_elems);
  kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(dst_k, src_k, pos, n_slots,
                                                   row_elems);
  kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(dst_v, src_v, pos, n_slots,
                                                   row_elems);
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
  flash_attention_kernel<<<n_q, n_heads_q * 32, smem, cur_stream()>>>(
      q, k, v, n_kv, n_heads_q, n_heads_kv, head_dim, sliding_window, scale,
      out);
  return cudaGetLastError() == cudaSuccess;
}

bool flash_attention_chunk_dev(const float* q, const float* k_batch,
                               const float* v_batch, const float* k_cache,
                               const float* v_cache, uint64_t start,
                               uint64_t n_slots, int n, int n_heads_q,
                               int n_heads_kv, int head_dim,
                               uint64_t sliding_window, float scale,
                               float* out) {
  if (!available()) return false;
  if (n <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || n_slots == 0)
    return false;
  if (!q || !k_batch || !v_batch || !k_cache || !v_cache || !out) return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  if (head_dim % 32 != 0 || head_dim > 512) return false;
  if (n_heads_q > 32) return false;  // one warp per query head (Gemma: 8)
  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  flash_attention_chunk_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
      q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q,
      n_heads_kv, head_dim, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

template <int HEAD_DIM, int BK, int NW, int QB, int GQA>
bool launch_tiled_flash_attention(const float* q, const float* k_batch,
                                  const float* v_batch, const float* k_cache,
                                  const float* v_cache, const uint64_t* start,
                                  uint64_t n_slots, int n, int n_heads_q,
                                  int n_heads_kv, uint64_t sliding_window,
                                  float scale, float* out) {
  constexpr int threads = NW * GQA * (QB / 16) * 32;
  constexpr int smem_bytes = HEAD_DIM * (2 * BK + 2) * (int)sizeof(__half);
  auto kfn = flash_attention_tiled_kernel<HEAD_DIM, BK, NW, QB, GQA>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute((const void*)kfn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured = true;
  }
  const int grid_x = (n + QB - 1) / QB;
  kfn<<<dim3(grid_x, n_heads_kv), threads, smem_bytes, cur_stream()>>>(
      q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q, n_heads_kv,
      n, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

bool flash_attention_chunk_dev_pos(const float* q, const float* k_batch,
                                   const float* v_batch, const float* k_cache,
                                   const float* v_cache, const uint64_t* start,
                                   uint64_t n_slots, int n, int n_heads_q,
                                   int n_heads_kv, int head_dim,
                                   uint64_t sliding_window, float scale,
                                   float* out) {
  if (!available()) return false;
  if (n <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || n_slots == 0)
    return false;
  if (!q || !k_batch || !v_batch || !k_cache || !v_cache || !start || !out)
    return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  if (head_dim % 32 != 0 || head_dim > 512) return false;
  if (n_heads_q > 32) return false;  // one warp per query head (Gemma: 8)

  // Tiled tensor-core path: Gemma's GQA=4 with the SWA head (256) or the global
  // head (512). Anything else keeps the scalar-fp32 kernel.
  const int gqa = n_heads_q / n_heads_kv;
  if (gqa == 4 && head_dim == 256)
    return launch_tiled_flash_attention<256, 64, 1, 32, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
  if (gqa == 4 && head_dim == 512)
    return launch_tiled_flash_attention<512, 32, 2, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);

  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  flash_attention_chunk_pos_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
      q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q,
      n_heads_kv, head_dim, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

bool layer_combine_dev(const float* proj, const float* ple, float ple_scale,
                       float combine_scale, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !proj || !ple || !y) return false;
  layer_combine_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(proj, ple, ple_scale, combine_scale,
                                             y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool argmax_dev(const float* x, int n, float* out) {
  if (!available()) return false;
  if (n <= 0 || !x || !out) return false;
  argmax_kernel<<<1, 1024, 0, cur_stream()>>>(x, out, n);
  return cudaGetLastError() == cudaSuccess;
}

// --- batched prefill entry points -------------------------------------------

bool rms_norm_cols_dev(std::span<const float> weight, const float* x, float eps,
                       float* y, int dim, int ncols) {
  if (!available()) return false;
  if (dim <= 0 || ncols <= 0 || !x || !y) return false;
  if (!weight.empty() && weight.size() < (size_t)dim) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(upload_bytes(weight.data(), dim * sizeof(float)));
    if (!d_w) return false;
  }
  rms_norm_cols_kernel<<<(ncols + 31) / 32, 1024, 0, cur_stream()>>>(x, d_w, dim, ncols, eps, y);
  return cudaGetLastError() == cudaSuccess;
}

bool rope_neox_heads_batched_dev(float* x, int head_dim, int heads_per_pos,
                                 int n_pos, uint64_t start_pos, float base,
                                 float freq_scale,
                                 std::span<const float> freq_factors) {
  if (!available()) return false;
  if (head_dim <= 0 || heads_per_pos <= 0 || n_pos <= 0 || !x) return false;
  const int n = head_dim * heads_per_pos * n_pos;
  const int half = head_dim >> 1;
  if ((head_dim & 1) != 0) return false;
  if (!freq_factors.empty() && freq_factors.size() < (size_t)half) return false;
  const float* d_ff = nullptr;
  if (!freq_factors.empty()) {
    d_ff = static_cast<const float*>(
        upload_bytes(freq_factors.data(), half * sizeof(float)));
    if (!d_ff) return false;
  }
  const float log_base = std::log(base);
  rope_neox_heads_batched_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(
      x, n, head_dim, heads_per_pos, start_pos, freq_scale, log_base, d_ff);
  return cudaGetLastError() == cudaSuccess;
}

bool rope_neox_heads_batched_dev_pos(float* x, int head_dim, int heads_per_pos,
                                     int n_pos, const uint64_t* start_pos,
                                     float base, float freq_scale,
                                     std::span<const float> freq_factors) {
  if (!available()) return false;
  if (head_dim <= 0 || heads_per_pos <= 0 || n_pos <= 0 || !x || !start_pos)
    return false;
  const int n = head_dim * heads_per_pos * n_pos;
  const int half = head_dim >> 1;
  if ((head_dim & 1) != 0) return false;
  if (!freq_factors.empty() && freq_factors.size() < (size_t)half) return false;
  const float* d_ff = nullptr;
  if (!freq_factors.empty()) {
    d_ff = static_cast<const float*>(
        upload_bytes(freq_factors.data(), half * sizeof(float)));
    if (!d_ff) return false;
  }
  const float log_base = std::log(base);
  rope_neox_heads_batched_pos_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(
      x, n, head_dim, heads_per_pos, start_pos, freq_scale, log_base, d_ff);
  return cudaGetLastError() == cudaSuccess;
}

bool transpose_dev(const float* src, float* dst, int dim, int ncols, bool forward) {
  if (!available()) return false;
  if (dim <= 0 || ncols <= 0 || !src || !dst) return false;
  const dim3 tblock(32, 32);
  if (forward) {
    // src [dim, ncols] -> dst [ncols, dim]
    const dim3 tgrid((ncols + 31) / 32, (dim + 31) / 32);
    transpose_tiled_dev<<<tgrid, tblock, 0, cur_stream()>>>(src, dst, dim, ncols);
  } else {
    // src [ncols, dim] -> dst [dim, ncols]
    const dim3 tgrid((dim + 31) / 32, (ncols + 31) / 32);
    transpose_tiled_dev<<<tgrid, tblock, 0, cur_stream()>>>(src, dst, ncols, dim);
  }
  return cudaGetLastError() == cudaSuccess;
}

// cuBLAS path for the batched dense projection (Phase C Step 3). On by default;
// SONICBOOM_DENSE_CUBLAS=0 restores the per-(row,col) gemv for A/B measurement.
bool dense_cublas_enabled() noexcept {
  if (std::getenv("SONICBOOM_DENSE_CUBLAS") == nullptr) return true;
  return std::string_view(std::getenv("SONICBOOM_DENSE_CUBLAS")) != "0";
}

// n (batch) threshold at or above which the cuBLAS path is used. This must be 1,
// not 2: a prefix-reuse suffix whose start_pos leaves exactly one token before
// the next chunk boundary splits into a 1-column segment (gemma4_plan.cpp:1187),
// and that lone column must be reduced by the SAME cuBLAS kernel as the n >= 2
// segment the full prefill computes it in — the per-(row,col) gemv has a
// different reduction order and would make the reused KV diverge. The decode
// spine's single-column path is a different entry point (matvec_f32, not the
// batched one) and is unaffected. The fixed-width 128-column blocking makes the
// cuBLAS per-column numerics independent of n (including n == 1, which packs one
// real column plus 127 zero pads into the same (m=128, n=rows, k=cols) shape).
constexpr int kDenseCublasMinN = 1;
int dense_cublas_min_n() {
  static int v = [] {
    if (const char* e = std::getenv("SONICBOOM_DENSE_CUBLAS_MIN_N")) {
      const int x = std::atoi(e);
      if (x > 0) return x;
    }
    return kDenseCublasMinN;
  }();
  return v;
}

// Fixed workspace so cuBLAS never lazily allocates inside a CUDA Graph capture
// (stream capture forbids host-side allocations). Ample for a single 128-column
// fp32/bf16 GEMM with fp32 accumulate.
constexpr size_t kDenseCublasWorkspaceBytes = 4 * 1024 * 1024;

// Lazily create the process-wide cuBLAS handle for the dense projections, bound
// to CUBLAS_DEFAULT_MATH (full fp32 Sgemm, no TF32 tensor ops; bf16 GemmEx uses
// CUBLAS_COMPUTE_32F tensor cores explicitly and is unaffected). Returns nullptr
// on failure (the caller falls back to the per-(row,col) gemv).
cublasHandle_t dense_cublas_handle() {
  static cublasHandle_t h = nullptr;
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!h) {
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
      h = nullptr;
      return nullptr;
    }
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    void* ws = nullptr;
    if (cudaMalloc(&ws, kDenseCublasWorkspaceBytes) == cudaSuccess)
      cublasSetWorkspace(h, ws, kDenseCublasWorkspaceBytes);
  }
  return h;
}

// Blocked cuBLAS matvec for the dense f32 projection. x is [cols, ncols], y is
// [rows, ncols] (ncols innermost); W is [rows, cols] row-major on device. The n
// columns are repacked into fixed-width 128-column blocks so every cuBLAS call
// has shape (m=128, n=rows, k=cols, lda=ldc=128) — the same per-column numerics
// regardless of n or column offset.
bool matvec_f32_cublas_dev(const float* d_w, int cols, const float* x, float* y,
                           int rows, int ncols) {
  cublasHandle_t h = dense_cublas_handle();
  if (!h) return false;
  float* xpack = static_cast<float*>(grow_scratch(
      g_xpack, g_xpack_bytes, (size_t)cols * kDenseCublasWidth * sizeof(float)));
  float* outpack = static_cast<float*>(grow_scratch(
      g_outpack, g_outpack_bytes, (size_t)rows * kDenseCublasWidth * sizeof(float)));
  if (!xpack || !outpack) return false;

  cublasSetStream(h, cur_stream());
  const float alpha = 1.0f, beta = 0.0f;
  for (int c0 = 0; c0 < ncols; c0 += kDenseCublasWidth) {
    const int chunk_n = std::min(kDenseCublasWidth, ncols - c0);
    pack_f32_chunk_kernel<<<grid_for(cols * kDenseCublasWidth), 256, 0,
                            cur_stream()>>>(x, xpack, cols, ncols, c0, chunk_n);
    const cublasStatus_t st = cublasSgemm(
        h, CUBLAS_OP_N, CUBLAS_OP_N, kDenseCublasWidth, rows, cols, &alpha, xpack,
        kDenseCublasWidth, d_w, cols, &beta, outpack, kDenseCublasWidth);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    copyback_chunk_kernel<<<grid_for(rows * chunk_n), 256, 0, cur_stream()>>>(
        outpack, y, rows, ncols, c0, chunk_n);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_bf16_cublas_dev(const uint16_t* d_w, int cols, const float* x,
                            float* y, int rows, int ncols) {
  cublasHandle_t h = dense_cublas_handle();
  if (!h) return false;
  uint16_t* xpack = static_cast<uint16_t*>(grow_scratch(
      g_xpack, g_xpack_bytes,
      (size_t)cols * kDenseCublasWidth * sizeof(uint16_t)));
  float* outpack = static_cast<float*>(grow_scratch(
      g_outpack, g_outpack_bytes, (size_t)rows * kDenseCublasWidth * sizeof(float)));
  if (!xpack || !outpack) return false;

  cublasSetStream(h, cur_stream());
  const float alpha = 1.0f, beta = 0.0f;
  for (int c0 = 0; c0 < ncols; c0 += kDenseCublasWidth) {
    const int chunk_n = std::min(kDenseCublasWidth, ncols - c0);
    pack_bf16_chunk_kernel<<<grid_for(cols * kDenseCublasWidth), 256, 0,
                             cur_stream()>>>(x, xpack, cols, ncols, c0, chunk_n);
    const cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, kDenseCublasWidth, rows, cols, &alpha, xpack,
        CUDA_R_16BF, kDenseCublasWidth, d_w, CUDA_R_16BF, cols, &beta, outpack,
        CUDA_R_32F, kDenseCublasWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    copyback_chunk_kernel<<<grid_for(rows * chunk_n), 256, 0, cur_stream()>>>(
        outpack, y, rows, ncols, c0, chunk_n);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_f32_batched_dev(std::span<const float> W, int cols, const float* x,
                            float* y, int rows, int ncols) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || ncols <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const float* d_w = static_cast<const float*>(
      upload_bytes(W.data(), (size_t)cols * rows * sizeof(float)));
  if (!d_w) return false;
  if (dense_cublas_enabled() && ncols >= dense_cublas_min_n() &&
      matvec_f32_cublas_dev(d_w, cols, x, y, rows, ncols))
    return true;
  const int total = rows * ncols;
  gemv_dense_f32_batched<<<grid_for(total), 256, 0, cur_stream()>>>(
      d_w, x, y, cols, rows, ncols);
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_bf16_batched_dev(std::span<const uint16_t> W, int cols, const float* x,
                             float* y, int rows, int ncols) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || ncols <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const uint16_t* d_w = static_cast<const uint16_t*>(
      upload_bytes(W.data(), (size_t)cols * rows * sizeof(uint16_t)));
  if (!d_w) return false;
  if (dense_cublas_enabled() && ncols >= dense_cublas_min_n() &&
      matvec_bf16_cublas_dev(d_w, cols, x, y, rows, ncols))
    return true;
  const int total = rows * ncols;
  gemv_bf16_batched<<<grid_for(total), 256, 0, cur_stream()>>>(
      d_w, x, y, cols, rows, ncols);
  return cudaGetLastError() == cudaSuccess;
}

bool kv_append_batched_dev(float* dst_k, float* dst_v, const float* src_k,
                           const float* src_v, uint64_t start_pos,
                           uint64_t n_slots, int row_elems, int n) {
  if (!available()) return false;
  if (row_elems <= 0 || n <= 0 || !dst_k || !dst_v || !src_k || !src_v ||
      n_slots == 0)
    return false;
  const int grid = grid_for(row_elems * n);
  kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_k, src_k, start_pos, n_slots, row_elems, n);
  kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_v, src_v, start_pos, n_slots, row_elems, n);
  return cudaGetLastError() == cudaSuccess;
}

bool kv_append_batched_dev_pos(float* dst_k, float* dst_v, const float* src_k,
                               const float* src_v, const uint64_t* start_pos,
                               uint64_t n_slots, int row_elems, int n) {
  if (!available()) return false;
  if (row_elems <= 0 || n <= 0 || !dst_k || !dst_v || !src_k || !src_v ||
      !start_pos || n_slots == 0)
    return false;
  const int grid = grid_for(row_elems * n);
  kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_k, src_k, start_pos, n_slots, row_elems, n);
  kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_v, src_v, start_pos, n_slots, row_elems, n);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace sonicboom::nn::cuda
