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

#include <sonicboom/runtime_features.h>

#include "cuda_resident.h"
#include "cuda_stream.h"
#include "reduce_sum.cuh"

#include "../planner/autotune.h"  // per-GPU tuning cache (Step 7)
#include "../planner/glue_jit.h"  // generated attention kernel (Stage 3)
#include "../planner/mlir_autotune.h"  // Q3: prefill attention shape classes
#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)
#include "../planner/reduce_mode.h"  // ReduceMode selection (f64 / df32 / f32)

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

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
// thread accumulates a partial sum over a strided slice of x, then the block
// reduces the partials (reduce_sum.cuh). The accumulator is the device's
// ReduceMode: f64 is the historical double path (each term is still the float
// square widened to double, identical per-term semantics to the CPU rms_norm),
// df32 is a double-float pair with the same tree, f32 is the plain-float
// performance reference. Only the *order* of the adds changes across tree
// widths, and that reordering is many orders below the fp32 rounding that the
// inverse RMS applies when it converts sum/n to float.
template <typename R>
__global__ void rms_sum_sq_kernel(const float* __restrict__ x, int n,
                                  void* __restrict__ out) {
  __shared__ typename R::part_t wp[32];
  const typename R::acc_t t = block_sum_sq<R>(x, n, wp);
  if (threadIdx.x == 0) *static_cast<typename R::out_t*>(out) = R::to_out(t);
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

__global__ void silu_kernel(const float* __restrict__ x, float* __restrict__ y,
                            int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  // x / (1 + exp(-x)); matches nn::silu / ggml_silu. expf (not __expf) so the
  // f32 result agrees with the CPU reference on the values Qwen's SwiGLU gate
  // takes.
  y[i] = v / (1.0f + expf(-v));
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

__global__ void add_bias_kernel(const float* __restrict__ x,
                                const float* __restrict__ bias, int bias_n,
                                float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + bias[i % bias_n];
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
// `add` is the optional residual epilogue (fuse_residual_epilogue_pass): the
// add that consumed this norm's output is folded into the apply loop, so
// y[i] = norm(x)[i] + add[i] in one launch. The apply of a separate add_kernel
// is the same single f32 addition of the stored norm value, so the result is
// bit-identical; __fadd_rn keeps the compiler from contracting that addition
// into the preceding multiply. x/y/add may alias (the executor's buffer reuse
// can place the result over either input): every index is read before it is
// written, and the reduction completes before the apply.
template <typename R>
__global__ void rms_norm_fused_kernel(const float* x,
                                      const float* __restrict__ weight, int n,
                                      float eps, float* y,
                                      const float* add) {
  const int tid = threadIdx.x;
  __shared__ typename R::part_t wp[32];
  __shared__ float inv;
  const typename R::acc_t t = block_sum_sq<R>(x, n, wp);
  if (tid == 0) inv = 1.0f / sqrtf(R::mean_to_float(t, n) + eps);
  __syncthreads();
  const float iv = inv;
  if (add != nullptr) {
    for (int i = tid; i < n; i += blockDim.x) {
      const float nv = weight ? x[i] * iv * weight[i] : x[i] * iv;
      y[i] = __fadd_rn(nv, add[i]);
    }
  } else {
    for (int i = tid; i < n; i += blockDim.x)
      y[i] = weight ? x[i] * iv * weight[i] : x[i] * iv;
  }
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
// `add` is the optional residual epilogue, as in rms_norm_fused_kernel: the
// add folded into this norm's output is applied here, bit-identically.
template <typename R>
__global__ void rms_heads_fused_kernel(const float* x,
                                       const float* __restrict__ weight,
                                       int head_dim, float eps,
                                       float* y, const float* add) {
  const int h = blockIdx.x;
  const float* xh = x + (size_t)h * head_dim;
  float* yh = y + (size_t)h * head_dim;
  const int tid = threadIdx.x;
  __shared__ typename R::part_t wp[32];
  __shared__ float inv;
  const typename R::acc_t t = block_sum_sq<R>(xh, head_dim, wp);
  if (tid == 0) inv = 1.0f / sqrtf(R::mean_to_float(t, head_dim) + eps);
  __syncthreads();
  const float iv = inv;
  if (add != nullptr) {
    const float* __restrict__ ah = add + (size_t)h * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
      const float nv = weight ? xh[i] * iv * weight[i] : xh[i] * iv;
      yh[i] = __fadd_rn(nv, ah[i]);
    }
  } else {
    for (int i = tid; i < head_dim; i += blockDim.x)
      yh[i] = weight ? xh[i] * iv * weight[i] : xh[i] * iv;
  }
}

// Launch the reduction-mode instantiation of a norm kernel. The selection is
// the device's ReduceMode (decided once per process), so the same grid/block
// geometry runs for every mode. These are non-template functions around the
// templated kernels: a template-template kernel name cannot be launched from a
// template context portably (nvcc parses `Kernel<T><<<` as a type name there).
inline void launch_rms_sum_sq(dim3 grid, int block, cudaStream_t st,
                              const float* x, int n, void* out) {
  switch (::sonicboom::planner::device_reduce_mode()) {
    case ::sonicboom::compiler::ReduceMode::Df32:
      rms_sum_sq_kernel<SumDf32><<<grid, block, 0, st>>>(x, n, out);
      break;
    case ::sonicboom::compiler::ReduceMode::F32:
      rms_sum_sq_kernel<SumF32><<<grid, block, 0, st>>>(x, n, out);
      break;
    case ::sonicboom::compiler::ReduceMode::F64:
    default:
      rms_sum_sq_kernel<SumF64><<<grid, block, 0, st>>>(x, n, out);
      break;
  }
}

inline void launch_rms_norm_fused(dim3 grid, int block, cudaStream_t st,
                                  const float* x, const float* w, int n,
                                  float eps, float* y, const float* add) {
  switch (::sonicboom::planner::device_reduce_mode()) {
    case ::sonicboom::compiler::ReduceMode::Df32:
      rms_norm_fused_kernel<SumDf32>
          <<<grid, block, 0, st>>>(x, w, n, eps, y, add);
      break;
    case ::sonicboom::compiler::ReduceMode::F32:
      rms_norm_fused_kernel<SumF32>
          <<<grid, block, 0, st>>>(x, w, n, eps, y, add);
      break;
    case ::sonicboom::compiler::ReduceMode::F64:
    default:
      rms_norm_fused_kernel<SumF64>
          <<<grid, block, 0, st>>>(x, w, n, eps, y, add);
      break;
  }
}

inline void launch_rms_heads_fused(dim3 grid, int block, cudaStream_t st,
                                   const float* x, const float* w, int head_dim,
                                   float eps, float* y, const float* add) {
  switch (::sonicboom::planner::device_reduce_mode()) {
    case ::sonicboom::compiler::ReduceMode::Df32:
      rms_heads_fused_kernel<SumDf32>
          <<<grid, block, 0, st>>>(x, w, head_dim, eps, y, add);
      break;
    case ::sonicboom::compiler::ReduceMode::F32:
      rms_heads_fused_kernel<SumF32>
          <<<grid, block, 0, st>>>(x, w, head_dim, eps, y, add);
      break;
    case ::sonicboom::compiler::ReduceMode::F64:
    default:
      rms_heads_fused_kernel<SumF64>
          <<<grid, block, 0, st>>>(x, w, head_dim, eps, y, add);
      break;
  }
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

  // Slot of the chunk's first key: one 64-bit modulo per block (device division
  // is a software routine, and doing it per key was a large part of this loop's
  // cost), then a compare-and-reset step so every later key's slot is a single
  // increment. Identical values to (start + c0 + j) % n_slots.
  uint64_t slot = (start + c0) % n_slots;

  // Phase 1: one q·k dot per key, warp-cooperative. Unrolled by 4 so several
  // keys' K loads are in flight at once — the per-key warp reduction is a
  // serialized shuffle chain, so a single-key loop pays a full L2/DRAM latency
  // per key.
  #pragma unroll 4
  for (uint64_t j = 0; j < cn; ++j) {
    const float* krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    float d = 0.0f;
    for (int i = lane; i < head_dim; i += 32) d += qrow[i] * krow[i];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      d += __shfl_down_sync(0xffffffffu, d, o);
    if (lane == 0) srow[j] = d * scale;
    if (++slot == n_slots) slot = 0;
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
  // the output row's dims. The accumulators are indexed by compile-time
  // constants (the runtime bound becomes a predicate) so the row lives in
  // registers instead of a dynamically-indexed local array, which the previous
  // form spilled to local memory inside this loop.
  const int dims_per_lane = head_dim >> 5;  // head_dim/32, <= 16
  float acc[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) acc[i] = 0.0f;
  slot = (start + c0) % n_slots;
  #pragma unroll 4
  for (uint64_t j = 0; j < cn; ++j) {
    const float* vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    const float p = srow[j];  // exp(score - local max)
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (i < dims_per_lane) acc[i] += p * vrow[lane + i * 32];
    if (++slot == n_slots) slot = 0;
  }

  // Write partial [m, l, o[0..head_dim-1]] for this head.
  float* pp = partial + chunk_idx * (uint64_t)n_heads_q * (head_dim + 2) +
              (uint64_t)h * (head_dim + 2);
  if (lane == 0) { pp[0] = m; pp[1] = l; }
#pragma unroll
  for (int i = 0; i < 16; ++i)
    if (i < dims_per_lane) pp[2 + lane + i * 32] = acc[i];
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
// (NW=2 for head_dim=256, NW=4 for head_dim=512) so the fp32 output accumulator
// stays at 16*(head_dim/NW)/32 = 64 floats/thread. The kernel is two-phase:
//   - phase 1 (dim_slice==0 warps only) computes q·k^T + online softmax once
//     per m-tile, storing p (fp16) and the per-sub-tile rescale rs (fp32) in
//     smem, so the NW>1 warps do NOT redundantly recompute S;
//   - phase 2 (all warps) reads p/rs from smem and does p·v on its dim slice.
// Q is converted to fp16 in registers (round-to-nearest), matching the
// fp16-rounded K/V the caller already staged via cast_fp16; the running max /
// sum / output stay fp32.
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
  constexpr int P_ROWS = GQA * (QB / 16) * 16;             // rows in the p scratch
  constexpr int P_ELEMS = P_ROWS * BK;                     // fp16 p entries
  constexpr int RS_ELEMS = GQA * (QB / 16) * (BK / 16) * 16;  // rs per sub-tile per row
  extern __shared__ __half smem[];
  __half* k_tile = smem;                    // [BK][HEAD_DIM]
  __half* v_tile = smem + BK * HEAD_DIM;    // [HEAD_DIM][V_STRIDE] (transposed)
  __half* p_buf = v_tile + HEAD_DIM * V_STRIDE;  // [P_ROWS][BK] fp16 p scratch
  float* rs_buf = reinterpret_cast<float*>(p_buf + P_ELEMS);  // [GQA*(QB/16)*(BK/16)][16]
  float* l_buf = rs_buf + RS_ELEMS;                          // [P_ROWS] per-row softmax sum

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

  // online-softmax state for the two rows this thread owns (g, g+8), used only
  // by the dim_slice==0 warps (phase 1).
  float m[2] = {-1e30f, -1e30f};
  float l[2] = {0.0f, 0.0f};
  // p·v output accumulator for this warp's dim slice (phase 2, all warps).
  float o[N_TILES][4];
  #pragma unroll
  for (int i = 0; i < N_TILES; ++i)
    o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

  // scratch base offsets for this (head, m-tile).
  const int p_base = (head_idx * (QB / 16) + m_tile) * 16 * BK;
  const int rs_base = (head_idx * (QB / 16) + m_tile) * (BK / 16) * 16;
  const int l_base = (head_idx * (QB / 16) + m_tile) * 16;

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

    // ---- phase 1 (dim_slice==0 only): q·k^T over the full head_dim, online
    // softmax, then store p (fp16) and rs (fp32) so the other warps reuse them
    // without redundantly recomputing S. ------------------------------------
    if (dim_slice == 0) {
      for (int key_off = 0; key_off < BK; key_off += 16) {
        // 1. q·k^T: S[16 queries][16 keys].
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

        // 4. store p (fp16) and rs (fp32) for the phase-2 warps.
        __half* pg = p_buf + p_base + g * BK + key_off;
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
        rs_buf[rs_base + sub * 16 + g] = rs0;
        rs_buf[rs_base + sub * 16 + g + 8] = rs1;
        m[0] = m_new0;
        m[1] = m_new1;
      }
    }
    __syncthreads();

    // ---- phase 2 (all warps): p·v on this warp's dim slice, reading p/rs from
    // smem (no redundant q·k^T). --------------------------------------------
    for (int key_off = 0; key_off < BK; key_off += 16) {
      const int sub = key_off >> 4;
      const float rs0 = rs_buf[rs_base + sub * 16 + g];
      const float rs1 = rs_buf[rs_base + sub * 16 + g + 8];
      #pragma unroll
      for (int i = 0; i < N_TILES; ++i) {
        o[i][0] *= rs0;
        o[i][1] *= rs0;
        o[i][2] *= rs1;
        o[i][3] *= rs1;
      }
      __half* pg = p_buf + p_base + g * BK + key_off;
      __half* pg8 = pg + 8 * BK;
      uint32_t ap[4];
      ap[0] = *reinterpret_cast<const uint32_t*>(pg + t * 2);
      ap[1] = *reinterpret_cast<const uint32_t*>(pg8 + t * 2);
      ap[2] = *reinterpret_cast<const uint32_t*>(pg + t * 2 + 8);
      ap[3] = *reinterpret_cast<const uint32_t*>(pg8 + t * 2 + 8);
      #pragma unroll
      for (int i = 0; i < N_TILES; ++i) {
        uint32_t bv[2];
        fa_load_v(bv, v_tile, V_STRIDE, dim0 + i * 8, key_off);
        fa_mma(o[i], ap, bv);
      }
    }
    __syncthreads();
  }

  // ---- finalize: out[d] = o[d] / l, written only for in-chunk rows --------
  // l is computed by the phase-1 (dim_slice==0) warp; broadcast via smem.
  if (dim_slice == 0) {
    l_buf[l_base + g] = l[0];
    l_buf[l_base + g + 8] = l[1];
  }
  __syncthreads();
  const float inv0 = 1.0f / l_buf[l_base + g];
  const float inv1 = 1.0f / l_buf[l_base + g + 8];
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
void* g_argmax_keys = nullptr;  // per-block packed argmax candidates (grid unsigned long long)
size_t g_argmax_keys_bytes = 0;
void* g_xpack = nullptr;        // repacked activation column-block for the cuBLAS dense matvec ([cols, 128])
size_t g_xpack_bytes = 0;
void* g_outpack = nullptr;      // padded output column-block ([rows, 128])
size_t g_outpack_bytes = 0;
void* g_xb = nullptr;           // bf16-rounded position-major activation ([n, cols]) for the strided path
size_t g_xb_bytes = 0;
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

// Packed argmax candidate: value in the high word as a monotone unsigned key
// (so an unsigned compare orders floats including +/-inf), index in the low word
// as 0xFFFFFFFF - i (so the *largest* packed key is the largest value and, on a
// tie, the smallest index). mi < 0 (no candidate: every value was NaN) maps to
// low word 0, which the finalize turns back into -1.
__device__ __forceinline__ unsigned long long argmax_key(float v, int mi) {
  const unsigned bits = __float_as_uint(v);
  const unsigned ord = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  const unsigned low = (mi < 0) ? 0u : (0xFFFFFFFFu - (unsigned)mi);
  return ((unsigned long long)ord << 32) | low;
}

// Multi-block argmax. The single-block form above is latency-bound: one block
// scans the whole row with a single 4 B load per thread in flight, which measures
// ~47 us/step for a 152 k vocab. Each block here reduces a strided float4 slice
// (grid sized from the row length and the SM count) to one packed key, and a
// single-block finalize picks the winner. Per block the reduction is exactly the
// single-block one (compare with >, first occurrence wins, so NaN never wins).
__global__ void argmax_partial_kernel(const float* __restrict__ x, int n,
                                      unsigned long long* __restrict__ keys) {
  __shared__ float s_val[256];
  __shared__ int s_idx[256];
  const int tid = threadIdx.x;
  float mv = -INFINITY;
  int mi = -1;
  const float4* x4 = reinterpret_cast<const float4*>(x);
  const int n4 = n >> 2;
  for (int i = blockIdx.x * blockDim.x + tid; i < n4;
       i += gridDim.x * blockDim.x) {
    const float4 v = x4[i];
    if (v.x > mv) { mv = v.x; mi = 4 * i + 0; }
    if (v.y > mv) { mv = v.y; mi = 4 * i + 1; }
    if (v.z > mv) { mv = v.z; mi = 4 * i + 2; }
    if (v.w > mv) { mv = v.w; mi = 4 * i + 3; }
  }
  // Row tail (n % 4): block 0 thread 0 sweeps the leftover elements, so the
  // float4 path stays unconditional.
  if (blockIdx.x == 0 && tid == 0)
    for (int i = 4 * n4; i < n; ++i)
      if (x[i] > mv) { mv = x[i]; mi = i; }
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
  if (tid == 0) keys[blockIdx.x] = argmax_key(s_val[0], s_idx[0]);
}

__global__ void argmax_finalize_kernel(const unsigned long long* __restrict__ keys,
                                       int n_keys, float* __restrict__ out) {
  __shared__ unsigned long long s_key[256];
  const int tid = threadIdx.x;
  unsigned long long k = 0;
  for (int i = tid; i < n_keys; i += blockDim.x) {
    const unsigned long long c = keys[i];
    k = c > k ? c : k;
  }
  s_key[tid] = k;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      const unsigned long long c = s_key[tid + s];
      if (c > s_key[tid]) s_key[tid] = c;
    }
    __syncthreads();
  }
  if (tid == 0) {
    const unsigned low = (unsigned)(s_key[0] & 0xFFFFFFFFull);
    out[0] = (low == 0) ? -1.0f : float(0xFFFFFFFFu - low);
  }
}

// --- batched prefill kernels (the decode kernels generalized to n columns) ---

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

// Dense batched gemv (position-major): one thread per (position, row) pair.
// Consecutive threads span output rows within a position, so the activation
// vector x[c*cols + i] is broadcast across the `rows` threads of a position and
// the weight row W[row*cols + i] is streamed per row.
__global__ void gemv_dense_f32_batched(const float* __restrict__ W,
                                       const float* __restrict__ x,
                                       float* __restrict__ y, int cols, int rows,
                                       int ncols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * ncols) return;
  const int c = idx / rows;
  const int r = idx - c * rows;
  const float* wrow = W + (size_t)r * cols;
  const float* xrow = x + (size_t)c * cols;
  float acc = 0.0f;
  for (int i = 0; i < cols; ++i) acc += wrow[i] * xrow[i];
  y[idx] = acc;
}

__global__ void gemv_bf16_batched(const uint16_t* __restrict__ W,
                                  const float* __restrict__ x,
                                  float* __restrict__ y, int cols, int rows,
                                  int ncols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * ncols) return;
  const int c = idx / rows;
  const int r = idx - c * rows;
  const uint16_t* wrow = W + (size_t)r * cols;
  const float* xrow = x + (size_t)c * cols;
  float acc = 0.0f;
  for (int i = 0; i < cols; ++i) acc += bf16_to_float(wrow[i]) * xrow[i];
  y[idx] = acc;
}

// Fixed column-block width for the cuBLAS dense matvec (Phase C Step 3). Every
// cuBLAS call is shaped (m=128, k=cols, lda=ldc=128) regardless of the batch n,
// so the per-column numerics are independent of both n and the column offset —
// the property prefix-cache reuse relies on (the chunker splits full and reuse
// prefills into differently-sized segments at the same 512-aligned boundaries).
constexpr int kDenseCublasWidth = 128;

// Repack a [ncols, cols] position-major activation slice (positions [c0,
// c0+chunk_n)) into the column-major [cols, 128] block cuBLAS consumes
// (xpack[i*128 + p] == x[c0+p, i]), zero-filling the trailing padding positions.
// Every chunk lands at the same stride (128), so the GEMM shape (m, k, lda) is
// identical across chunks and across n values.
__global__ void pack_f32_chunk_kernel(const float* __restrict__ x,
                                      float* __restrict__ dst, int cols,
                                      int ncols, int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = cols * kDenseCublasWidth;
  if (idx >= total) return;
  const int p = idx % kDenseCublasWidth;
  const int i = idx / kDenseCublasWidth;
  dst[idx] = (p < chunk_n) ? x[(size_t)(c0 + p) * cols + i] : 0.0f;
}

__global__ void pack_bf16_chunk_kernel(const float* __restrict__ x,
                                       uint16_t* __restrict__ dst, int cols,
                                       int ncols, int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = cols * kDenseCublasWidth;
  if (idx >= total) return;
  const int p = idx % kDenseCublasWidth;
  const int i = idx / kDenseCublasWidth;
  dst[idx] = (p < chunk_n) ? float_to_bf16(x[(size_t)(c0 + p) * cols + i])
                           : uint16_t(0);
}

// Round a position-major [ncols, cols] activation slice through bf16 in place
// into a scratch buffer (same layout). The strided dense path feeds this scratch
// to cuBLAS directly, so it replaces the pack (transposing repack) plus the
// backend's own bf16 staging copy with one linear pass.
__global__ void cast_bf16_rows_kernel(const float* __restrict__ x,
                                      uint16_t* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = float_to_bf16(x[i]);
}

// Zero-padded 128-position tail block for the strided dense path, in the same
// position-major layout the full blocks read (dst[p*cols + i] = x[(c0+p)*cols +
// i] for p < chunk_n, zero beyond). The tail must go through the same GEMM shape
// as the full blocks: a position that lands in a full block in one segmentation
// (full prefill) and in a tail block in another (prefix reuse) has to see the
// same per-position numerics.
__global__ void pad_f32_tail_kernel(const float* __restrict__ x,
                                    float* __restrict__ dst, int cols,
                                    int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cols * kDenseCublasWidth) return;
  const int p = idx / cols;
  const int i = idx - p * cols;
  dst[idx] = (p < chunk_n) ? x[(size_t)(c0 + p) * cols + i] : 0.0f;
}

__global__ void pad_bf16_tail_kernel(const uint16_t* __restrict__ x,
                                     uint16_t* __restrict__ dst, int cols,
                                     int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= cols * kDenseCublasWidth) return;
  const int p = idx / cols;
  const int i = idx - p * cols;
  dst[idx] = (p < chunk_n) ? x[(size_t)(c0 + p) * cols + i] : uint16_t(0);
}

// Drop the padding columns of a [128, rows] position-major result block (the
// strided GEMM writes it with ldc=rows) into the position-major output.
__global__ void copyback_tail_kernel(const float* __restrict__ src,
                                     float* __restrict__ dst, int rows, int c0,
                                     int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= chunk_n * rows) return;
  const int p = idx / rows;
  const int r = idx - p * rows;
  dst[(size_t)(c0 + p) * rows + r] = src[(size_t)p * rows + r];
}

// gemv_bf16_batched reading a position-major bf16 activation (the strided
// path's fallback when cuBLAS is unavailable).
__global__ void gemv_bf16_batched_xb(const uint16_t* __restrict__ W,
                                     const uint16_t* __restrict__ x,
                                     float* __restrict__ y, int cols, int rows,
                                     int ncols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * ncols) return;
  const int c = idx / rows;
  const int r = idx - c * rows;
  const uint16_t* wrow = W + (size_t)r * cols;
  const uint16_t* xrow = x + (size_t)c * cols;
  float acc = 0.0f;
  for (int i = 0; i < cols; ++i)
    acc += bf16_to_float(wrow[i]) * bf16_to_float(xrow[i]);
  y[idx] = acc;
}

// Copy the first `chunk_n` positions of a [rows, 128] result block back into the
// [ncols, rows] position-major output at position offset c0 (the padding
// positions are dropped). src is column-major [128, rows] as cuBLAS wrote it, so
// src[p + r*128] == y[c0+p, r].
__global__ void copyback_chunk_kernel(const float* __restrict__ src,
                                      float* __restrict__ dst, int rows,
                                      int ncols, int c0, int chunk_n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = chunk_n * rows;
  if (idx >= total) return;
  const int p = idx / rows;
  const int r = idx - p * rows;
  dst[(size_t)(c0 + p) * rows + r] = src[p + (size_t)r * kDenseCublasWidth];
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

  SB_NOTE_LAUNCH(rms_sum_sq_kernel, 1, 256, 0, cur_stream());
  // The kernel writes the accumulator in its mode's representation (8 bytes
  // for f64 and df32, 4 for f32); the mean is formed the same way the kernel's
  // own mean_to_float does, so the apply sees one value either way.
  launch_rms_sum_sq(dim3(1), 256, cur_stream(),
                    static_cast<const float*>(d_x), int(n), g_sum);
  if (cudaGetLastError() != cudaSuccess) return false;
  const float mean = [&] {
    switch (::sonicboom::planner::device_reduce_mode()) {
      case ::sonicboom::compiler::ReduceMode::Df32: {
        float2 pair{};
        if (cudaMemcpy(&pair, g_sum, sizeof pair, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
          return 0.0f;
        return pair.x / float(n) + pair.y / float(n);
      }
      case ::sonicboom::compiler::ReduceMode::F32: {
        float s = 0.0f;
        if (cudaMemcpy(&s, g_sum, sizeof s, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
          return 0.0f;
        return s / float(n);
      }
      case ::sonicboom::compiler::ReduceMode::F64:
      default: {
        double s = 0.0;
        if (cudaMemcpy(&s, g_sum, sizeof s, cudaMemcpyDeviceToHost) !=
            cudaSuccess)
          return 0.0f;
        return float(s / double(n));
      }
    }
  }();
  // mean then float (sqrt/recip), the CPU reference's rounding order.
  const float inv = 1.0f / std::sqrt(mean + eps);
  SB_NOTE_LAUNCH(rms_apply_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(gelu_fp16_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(mul_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(add_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(scale_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(cast_fp16_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(rope_neox_kernel, grid_for(int(half)), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(rope_neox_heads_kernel, grid_for(int(n)), 256, 0, cur_stream());
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
      SB_NOTE_LAUNCH(gemv_dense_f32_v, gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_dense_f32_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const float*>(d_w), static_cast<const float*>(d_x),
          partial, int(cols), r, k_split);
      SB_NOTE_LAUNCH(gemv_reduce_kernel, grid_for(r), 256, 0, cur_stream());
      gemv_reduce_kernel<<<grid_for(r), 256, 0, cur_stream()>>>(partial,
                                               static_cast<float*>(d_y), r,
                                               k_split);
    } else {
      SB_NOTE_LAUNCH(gemv_dense_f32_v, gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_dense_f32_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const float*>(d_w), static_cast<const float*>(d_x),
          static_cast<float*>(d_y), int(cols), r, k_split);
    }
  } else {
    SB_NOTE_LAUNCH(gemv_dense_f32, grid_for(int(rows)), 256, 0, cur_stream());
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
      SB_NOTE_LAUNCH(gemv_bf16_v, gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_bf16_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const uint16_t*>(d_w), static_cast<const float*>(d_x),
          partial, int(cols), r, k_split);
      SB_NOTE_LAUNCH(gemv_reduce_kernel, grid_for(r), 256, 0, cur_stream());
      gemv_reduce_kernel<<<grid_for(r), 256, 0, cur_stream()>>>(partial,
                                               static_cast<float*>(d_y), r,
                                               k_split);
    } else {
      SB_NOTE_LAUNCH(gemv_bf16_v, gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_bf16_v<<<gemv_grid(r, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          static_cast<const uint16_t*>(d_w), static_cast<const float*>(d_x),
          static_cast<float*>(d_y), int(cols), r, k_split);
    }
  } else {
    SB_NOTE_LAUNCH(gemv_bf16, grid_for(int(rows)), 256, 0, cur_stream());
    gemv_bf16<<<grid_for(int(rows)), 256, 0, cur_stream()>>>(static_cast<const uint16_t*>(d_w),
                                            static_cast<const float*>(d_x),
                                            static_cast<float*>(d_y), int(cols), int(rows));
  }
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost) ==
         cudaSuccess;
}

// Batched dense matvec (the prefill shape): Y[c*rows+r] = sum_i W[r*cols+i] *
// X[c*cols+i]. X is host position-major [n, cols], Y is host position-major
// [n, rows]. Uploads X, runs the device batched matvec (cuBLAS fixed-width
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
  SB_NOTE_LAUNCH(flash_attention_kernel, (unsigned)n_q, (unsigned)(n_heads_q * 32), smem, cur_stream());
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
                  float* y, int n, const float* add) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  if (!weight.empty() && weight.size() < (size_t)n) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(upload_bytes(weight.data(), n * sizeof(float)));
    if (!d_w) return false;
  }
  SB_NOTE_LAUNCH(rms_norm_fused_kernel, 1, 1024, 0, cur_stream());
  launch_rms_norm_fused(dim3(1), 1024, cur_stream(), x, d_w, n, eps, y, add);
  return cudaGetLastError() == cudaSuccess;
}

bool rms_norm_heads_dev(std::span<const float> weight, const float* x, float eps,
                        float* y, int head_dim, int heads, const float* add) {
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
  SB_NOTE_LAUNCH(rms_heads_fused_kernel, heads, 256, 0, cur_stream());
  launch_rms_heads_fused(heads, 256, cur_stream(), x, d_w, head_dim, eps, y,
                         add);
  return cudaGetLastError() == cudaSuccess;
}

bool gelu_fp16_dev(const float* x, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  SB_NOTE_LAUNCH(gelu_fp16_kernel, grid_for(n), 256, 0, cur_stream());
  gelu_fp16_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool silu_dev(const float* x, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  SB_NOTE_LAUNCH(silu_kernel, grid_for(n), 256, 0, cur_stream());
  silu_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool mul_dev(const float* a, const float* b, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !a || !b || !y) return false;
  SB_NOTE_LAUNCH(mul_kernel, grid_for(n), 256, 0, cur_stream());
  mul_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(a, b, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool add_dev(const float* a, const float* b, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !a || !b || !y) return false;
  SB_NOTE_LAUNCH(add_kernel, grid_for(n), 256, 0, cur_stream());
  add_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(a, b, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool add_bias_dev(std::span<const float> bias, const float* x, float* y, int n) {
  if (!available()) return false;
  const int bias_n = int(bias.size());
  // Same contract as nn::add_bias: a whole number of bias rows, so `bias_n`
  // divides `n`. A biasless architecture never emits this node.
  if (bias_n <= 0 || n <= 0 || n % bias_n != 0 || !x || !y) return false;
  const float* d_b = static_cast<const float*>(
      upload_bytes(bias.data(), size_t(bias_n) * sizeof(float)));
  if (!d_b) return false;
  SB_NOTE_LAUNCH(add_bias_kernel, grid_for(n), 256, 0, cur_stream());
  add_bias_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, d_b, bias_n, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool scale_dev(const float* x, float s, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y) return false;
  SB_NOTE_LAUNCH(scale_kernel, grid_for(n), 256, 0, cur_stream());
  scale_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, s, y, n);
  return cudaGetLastError() == cudaSuccess;
}

bool cast_fp16_dev(float* v, int n) {
  if (!available()) return false;
  if (n <= 0 || !v) return false;
  SB_NOTE_LAUNCH(cast_fp16_kernel, grid_for(n), 256, 0, cur_stream());
  cast_fp16_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(v, n);
  return cudaGetLastError() == cudaSuccess;
}

bool cast_bf16_dev(float* v, int n) {
  if (!available()) return false;
  if (n <= 0 || !v) return false;
  SB_NOTE_LAUNCH(cast_bf16_kernel, grid_for(n), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(rope_neox_heads_kernel, grid_for(n), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(rope_neox_heads_pos_kernel, grid_for(n), 256, 0, cur_stream());
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
      SB_NOTE_LAUNCH(gemv_dense_f32_v, gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_dense_f32_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, partial, cols, rows, k_split);
      SB_NOTE_LAUNCH(gemv_reduce_kernel, grid_for(rows), 256, 0, cur_stream());
      gemv_reduce_kernel<<<grid_for(rows), 256, 0, cur_stream()>>>(partial, y, rows, k_split);
    } else {
      SB_NOTE_LAUNCH(gemv_dense_f32_v, gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_dense_f32_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, y, cols, rows, k_split);
    }
  } else {
    SB_NOTE_LAUNCH(gemv_dense_f32, grid_for(rows), 256, 0, cur_stream());
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
      SB_NOTE_LAUNCH(gemv_bf16_v, gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_bf16_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, partial, cols, rows, k_split);
      SB_NOTE_LAUNCH(gemv_reduce_kernel, grid_for(rows), 256, 0, cur_stream());
      gemv_reduce_kernel<<<grid_for(rows), 256, 0, cur_stream()>>>(partial, y, rows, k_split);
    } else {
      SB_NOTE_LAUNCH(gemv_bf16_v, gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream());
      gemv_bf16_v<<<gemv_grid(rows, k_split), kGemvWarpsPerBlock * 32, 0, cur_stream()>>>(
          d_w, x, y, cols, rows, k_split);
    }
  } else {
    SB_NOTE_LAUNCH(gemv_bf16, grid_for(rows), 256, 0, cur_stream());
    gemv_bf16<<<grid_for(rows), 256, 0, cur_stream()>>>(d_w, x, y, cols, rows);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool gqa_broadcast_dev(const float* src, float* dst, int n_q, int n_kv, int head_dim) {
  if (!available()) return false;
  if (n_q <= 0 || n_kv <= 0 || head_dim <= 0 || !src || !dst) return false;
  const int n = n_q * head_dim;
  SB_NOTE_LAUNCH(gqa_broadcast_kernel, grid_for(n), 256, 0, cur_stream());
  gqa_broadcast_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(src, dst, n_q, n_kv, head_dim);
  return cudaGetLastError() == cudaSuccess;
}

// --- attention work trace (SONICBOOM_ATTN_TRACE=1) -------------------------
//
// Off by default. A trace line reports, per call, the shape, the implementation
// and config chosen, the absolute KV extent the kernel iterates, and the key
// tiles visited versus the minimum the causal+window mask requires. Reading the
// device position needs a D2H copy, so the trace is a diagnostic, not a timing
// mode; it is skipped during stream capture (where a D2H copy is illegal).
namespace {

bool attn_trace_on() {
  static const bool on = std::getenv("SONICBOOM_ATTN_TRACE") != nullptr;
  return on;
}

int &attn_trace_layer_slot() {
  static thread_local int layer = -1;
  return layer;
}

// Absolute KV start for this call, or kNoPos when unavailable/capturing.
constexpr uint64_t kNoPos = ~uint64_t(0);
uint64_t attn_trace_read_start(const uint64_t *start) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (start == nullptr) return kNoPos;
  if (cudaStreamIsCapturing(cur_stream(), &st) != cudaSuccess ||
      st != cudaStreamCaptureStatusNone)
    return kNoPos;
  uint64_t v = 0;
  if (cudaMemcpy(&v, start, sizeof(v), cudaMemcpyDeviceToHost) != cudaSuccess)
    return kNoPos;
  return v;
}

// Number of key tiles the (query tile, key tile) loop visits, and the minimum
// the causal+window mask admits. Tiles are aligned to ABSOLUTE positions.
void attn_trace_tile_counts(uint64_t start, int q_len, int tile_q, int tile_k,
                            uint64_t window, uint64_t *visited,
                            uint64_t *minimal, std::string *per_q) {
  uint64_t v = 0, m = 0;
  for (int t = 0; (uint64_t)t * (uint64_t)tile_q < (uint64_t)q_len; ++t) {
    const uint64_t abs_pb = start + (uint64_t)t * (uint64_t)tile_q;
    const uint64_t n_blk =
        std::min<uint64_t>((uint64_t)tile_q, (uint64_t)q_len - abs_pb + start);
    const uint64_t key_end = abs_pb + n_blk - 1;
    const uint64_t key_start =
        (window > 0 && abs_pb >= window) ? (abs_pb - window + 1) : 0;
    const uint64_t j0 = (key_start / (uint64_t)tile_k) * (uint64_t)tile_k;
    const uint64_t vt = (key_end + 1 - j0 + tile_k - 1) / (uint64_t)tile_k;
    const uint64_t mt =
        (key_end - key_start + 1 + tile_k - 1) / (uint64_t)tile_k;
    v += vt; m += mt;
    if (!per_q->empty()) per_q->push_back(',');
    per_q->append(std::to_string(vt));
  }
  *visited = v;
  *minimal = m;
}

void attn_trace_prefill(uint64_t start, int q_len, int head_dim, int gqa,
                        uint64_t window, const char *impl, const char *cfg,
                        int tile_q, int tile_k, int grid_x, int grid_y,
                        int block, size_t smem) {
  if (!attn_trace_on()) return;
  std::string per_q;
  uint64_t visited = 0, minimal = 0;
  attn_trace_tile_counts(start, q_len, tile_q, tile_k, window, &visited,
                         &minimal, &per_q);
  if (start == kNoPos)
    std::fprintf(stderr,
                 "[attn-trace] layer=%d impl=%s d=%d gqa=%d win=%llu q=%d "
                 "start=dev L=dev cfg=%s grid=%dx%d block=%d smem=%zu "
                 "tiles=%llu minimal=%llu per_q=[%s]\n",
                 attn_trace_layer_slot(), impl, head_dim, gqa,
                 (unsigned long long)window, q_len, cfg, grid_x, grid_y, block,
                 smem, (unsigned long long)visited,
                 (unsigned long long)minimal, per_q.c_str());
  else
    std::fprintf(stderr,
                 "[attn-trace] layer=%d impl=%s d=%d gqa=%d win=%llu q=%d "
                 "start=%llu L=%llu cfg=%s grid=%dx%d block=%d smem=%zu "
                 "tiles=%llu minimal=%llu per_q=[%s]\n",
                 attn_trace_layer_slot(), impl, head_dim, gqa,
                 (unsigned long long)window, q_len,
                 (unsigned long long)start,
                 (unsigned long long)(start + (uint64_t)q_len), cfg, grid_x,
                 grid_y, block, smem, (unsigned long long)visited,
                 (unsigned long long)minimal, per_q.c_str());
}

// Trace line for one of the hand-written tiled instantiations. The numbers
// mirror launch_tiled_flash_attention's launch geometry.
template <int HEAD_DIM, int BK, int NW, int QB, int GQA>
void attn_trace_own(const uint64_t *start, int n, int n_heads_kv, int head_dim,
                    int gqa, uint64_t window) {
  if (!attn_trace_on()) return;
  constexpr int threads = NW * GQA * (QB / 16) * 32;
  constexpr int P_ELEMS = GQA * (QB / 16) * 16 * BK;
  constexpr int RS_ELEMS = GQA * (QB / 16) * (BK / 16) * 16;
  constexpr int L_ELEMS = GQA * (QB / 16) * 16;
  constexpr int smem = HEAD_DIM * (2 * BK + 2) * (int)sizeof(__half) +
                       P_ELEMS * (int)sizeof(__half) +
                       (RS_ELEMS + L_ELEMS) * (int)sizeof(float);
  char cfg[64];
  std::snprintf(cfg, sizeof(cfg), "QB%d,BK%d,NW%d", QB, BK, NW);
  attn_trace_prefill(attn_trace_read_start(start), n, head_dim, gqa, window,
                     "own", cfg, QB, BK, (n + QB - 1) / QB, n_heads_kv, threads,
                     (size_t)smem);
}

void attn_trace_decode(uint64_t pos, uint64_t window, uint64_t chunk,
                       uint64_t n_chunks, int head_dim, int gqa,
                       const char *impl, int grid_x, int block, size_t smem) {
  if (!attn_trace_on()) return;
  if (pos == kNoPos) {
    std::fprintf(stderr,
                 "[attn-trace] layer=%d impl=%s d=%d gqa=%d "
                 "win=%llu q=1 start=dev chunk=%llu launched_chunks=%llu "
                 "grid=%d block=%d smem=%zu\n",
                 attn_trace_layer_slot(), impl, head_dim, gqa,
                 (unsigned long long)window, (unsigned long long)chunk,
                 (unsigned long long)n_chunks, grid_x, block, smem);
    return;
  }
  const uint64_t start =
      (window > 0 && pos >= window) ? pos - window + 1 : 0;
  const uint64_t n_keys = pos - start + 1;
  const uint64_t active = (n_keys + chunk - 1) / chunk;
  std::fprintf(stderr,
               "[attn-trace] layer=%d impl=%s d=%d gqa=%d "
               "win=%llu q=1 start=%llu L=%llu chunk=%llu launched_chunks=%llu "
               "active_chunks=%llu grid=%d block=%d smem=%zu\n",
               attn_trace_layer_slot(), impl, head_dim, gqa,
               (unsigned long long)window, (unsigned long long)start,
               (unsigned long long)(pos + 1), (unsigned long long)chunk,
               (unsigned long long)n_chunks, (unsigned long long)active, grid_x,
               block, smem);
}

} // namespace

void attention_trace_set_layer(int layer) { attn_trace_layer_slot() = layer; }

namespace {
// Defined with the Step 7 tuners below; declared here because the decode path
// is emitted before them.
uint64_t attn_decode_chunk_resolve(int head_dim, int gqa, int n_heads_q,
                                   int n_heads_kv, uint64_t max_window,
                                   uint64_t sliding_window,
                                   uint64_t default_chunk);
} // namespace

bool decode_attention_dev(const float* q, const float* k_cache,
                          const float* v_cache, const uint64_t* pos,
                          uint64_t window, uint64_t n_slots, int n_heads_q,
                          int n_heads_kv, int head_dim, uint64_t sliding_window,
                          float scale, float* out, int pos_offset) {
  if (!available()) return false;
  if (n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || head_dim > 512 ||
      !q || !k_cache || !v_cache || !out || !pos || n_slots == 0 || window == 0 ||
      pos_offset < 0)
    return false;
  // B3 verify: row `pos_offset` of the verify step is the n == 1 call at its
  // position, so the whole launch is the decode call with a shifted slot.
  pos += pos_offset;

  const int group_q = n_heads_q / n_heads_kv;

  // Chunk cap shared by both decode paths (see the own split-KV launch below):
  // it bounds how far a single block's key slice can grow with n_ctx.
  constexpr uint64_t kMaxChunk = 16;

  // Generated flash-decoding (codegen) or the hand-written split-KV (own).
  // Codegen is the default: since attn8 its rounding and reduction order match
  // the hand-written pair exactly and both split the window with the same
  // chunk rule, so it is token-identical. SONICBOOM_ATTN_DECODE=own selects
  // the hand-written pair (the ablation arm); SONICBOOM_ATTN_DECODE=codegen
  // is still accepted and SONICBOOM_ATTN_IMPL=own forces own.
  //
  // Measured per shape class (attention_bench, RTX 3050, us/call): windowed
  // d256 at a 512-key window 23.2 (codegen, chunk 16) vs 31.3 (own); causal
  // d512 44.8 vs 50.5 at 513 keys but 279+ vs 245.5 at 4097 keys -- own's
  // blocks share one KV group's read across query heads, the generated
  // one-warp-per-(head, chunk) blocks re-read it per query head. End to end
  // (all classes switched) that is +1.3% at depth 512 and -4.7% at depth
  // 4096 (47.1 vs 49.4 tok/s).
  {
    const char *adec = std::getenv("SONICBOOM_ATTN_DECODE");
    const char *aimpl = std::getenv("SONICBOOM_ATTN_IMPL");
    bool want_codegen = adec == nullptr || std::string_view(adec) != "own";
    if (aimpl != nullptr && std::string_view(aimpl) == "own")
      want_codegen = false;
    compiler::AttentionShape dsh;
    dsh.head_dim = head_dim;
    dsh.gqa = group_q;
    dsh.window_mask = sliding_window != 0;
    dsh.decode = true;
    dsh.exp2 = std::getenv("SONICBOOM_ATTN_EXP2") != nullptr;
    if (want_codegen && compiler::attention_shape_supported(dsh) &&
        n_heads_q % n_heads_kv == 0) {
      // Fixed chunk length per shape class: the chunk count scales with the
      // max valid KV extent (`window`), not with this launch's key count. The
      // absolute alignment makes a chunk's keys independent of the step.
      //
      // Token identity with the hand-written pair requires the same split, so
      // the default reproduces its chunk rule; the Step 7 chunk tuner changes
      // the split and is therefore part of the non-token-identical arm.
      // SONICBOOM_ATTN_DECODE_CHUNK overrides either way (ablation).
      const uint64_t gen_target_chunks =
          std::max<uint64_t>(1, (uint64_t)sm_count() * 4 / (uint64_t)n_heads_kv);
      const uint64_t gen_no_split_chunk =
          std::max<uint64_t>(1, (48 * 1024 / sizeof(float)) / (uint64_t)group_q);
      uint64_t kDecodeChunk =
          runtime::feature_enabled(runtime::Feature::SplitKv)
              ? std::max<uint64_t>(
                    1, std::min((window + gen_target_chunks - 1) /
                                    gen_target_chunks,
                                kMaxChunk))
              : std::min(window, gen_no_split_chunk);
      if (const char *e = std::getenv("SONICBOOM_ATTN_DECODE_CHUNK")) {
        const long v = std::atol(e);
        if (v > 0) kDecodeChunk = (uint64_t)v;
      } else if (std::getenv("SONICBOOM_ATTN_TUNE_NONIDENTICAL") != nullptr) {
        // Step 7 runtime tuner (cached per GPU + shape class; the identity
        // chunk is the fallback when tuning is off or inside a capture).
        kDecodeChunk = attn_decode_chunk_resolve(
            head_dim, group_q, n_heads_q, n_heads_kv, window, sliding_window,
            kDecodeChunk);
      }
      // The generated decoder stages one chunk's scores in a fixed shared
      // view; a larger chunk is outside what the generated text supports.
      if (kDecodeChunk > (uint64_t)compiler::kAttentionDecodeMaxChunk)
        kDecodeChunk = compiler::kAttentionDecodeMaxChunk;
      const uint64_t n_chunks = (window + kDecodeChunk - 1) / kDecodeChunk;
      const size_t partial_floats =
          (size_t)n_chunks * (size_t)n_heads_q * (size_t)(head_dim + 2);
      float *partial = static_cast<float *>(
          grow_scratch(g_attn_partial, g_attn_partial_bytes,
                       partial_floats * sizeof(float)));
      if (partial != nullptr) {
        const float *qa = q;
        const float *kca = k_cache;
        const float *vca = v_cache;
        const uint64_t *startp = pos;
        uint64_t nslots_arg = n_slots;
        uint64_t win_arg = sliding_window;
        int nhq_arg = n_heads_q;
        int nhkv_arg = n_heads_kv;
        float scale_arg = scale;
        uint64_t chunk_arg = kDecodeChunk;
        float *out_arg = out;
        int nch_arg = (int)n_chunks;
        void *args[13] = {&qa,       &kca,      &vca,       &startp,
                          &nslots_arg, &nhq_arg, &nhkv_arg, &win_arg,
                          &scale_arg, &partial, &chunk_arg, &out_arg,
                          &nch_arg};
        if (planner::glue::launch_attention_decode(dsh, args, (int)n_chunks,
                                                   n_heads_q)) {
          if (attn_trace_on())
            attn_trace_decode(attn_trace_read_start(pos), window,
                              kDecodeChunk, n_chunks, head_dim, group_q,
                              "codegen-split-decode",
                              (int)(n_chunks * (uint64_t)n_heads_q), 32, 0);
          return true;
        }
      }
    }
  }

  // Split-KV: n_chunks × n_kv_heads blocks each compute a per-head partial over
  // its slice of the KV sequence, then n_heads_q merge blocks combine them in
  // fixed order. The launch config is sized from the FIXED `window` (max key
  // count this graph replays for: sliding_window for SWA, n_ctx for global), so
  // grid/smem/chunk are stable for CUDA Graph capture; the split kernel computes
  // the real start/n_keys from *pos and early-exits chunks beyond n_keys.
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
  // Feature split_kv: off collapses to one chunk per KV head (no split-KV
  // partials; the split kernel still runs its n_chunks == 1 path and the merge
  // is a single-chunk copy). The reduction order differs from the split
  // variant, so the ablation arm is a performance probe, not a numerics
  // guarantee.
  // The no-split arm is bounded by shared memory: the split kernel holds
  // group_q*chunk floats, and dynamic shared memory is 48 KiB here. A window
  // larger than that still splits, but into the fewest chunks that fit (vs the
  // kMaxChunk-capped split above), which is the "less split-KV" comparison.
  const uint64_t no_split_chunk =
      std::max<uint64_t>(1, (48 * 1024 / sizeof(float)) / (uint64_t)group_q);
  const uint64_t chunk =
      runtime::feature_enabled(runtime::Feature::SplitKv)
          ? std::max<uint64_t>(1, std::min((window + target_chunks - 1) /
                                               target_chunks,
                                           kMaxChunk))
          : std::min(window, no_split_chunk);
  const uint64_t n_chunks = (window + chunk - 1) / chunk;

  const size_t partial_floats =
      (size_t)n_chunks * (size_t)n_heads_q * (size_t)(head_dim + 2);
  float* partial = static_cast<float*>(
      grow_scratch(g_attn_partial, g_attn_partial_bytes,
                   partial_floats * sizeof(float)));

  const size_t smem = (size_t)group_q * chunk * sizeof(float);
  SB_NOTE_LAUNCH(decode_attention_split_kernel, (unsigned)(n_chunks * (uint64_t)n_heads_kv), group_q * 32, smem, cur_stream());
  decode_attention_split_kernel<<<(unsigned)(n_chunks * (uint64_t)n_heads_kv),
                                  group_q * 32, smem, cur_stream()>>>(
      q, k_cache, v_cache, pos, n_slots, chunk, sliding_window, n_heads_q,
      n_heads_kv, head_dim, scale, partial);
  SB_NOTE_LAUNCH(decode_attention_merge_kernel, (unsigned)n_heads_q, head_dim, 0, cur_stream());
  decode_attention_merge_kernel<<<(unsigned)n_heads_q, head_dim, 0, cur_stream()>>>(
      partial, pos, chunk, sliding_window, (int)n_chunks, n_heads_q, head_dim,
      out);
  if (attn_trace_on())
    attn_trace_decode(attn_trace_read_start(pos), window, chunk, n_chunks,
                      head_dim, group_q, "own-split-decode",
                      int(n_chunks * (uint64_t)n_heads_kv), group_q * 32, smem);
  return cudaGetLastError() == cudaSuccess;
}

bool kv_append_dev(float* dst_k, float* dst_v, const float* src_k,
                   const float* src_v, const uint64_t* pos, uint64_t n_slots,
                   int row_elems, int pos_offset) {
  if (!available()) return false;
  if (row_elems <= 0 || !dst_k || !dst_v || !src_k || !src_v || !pos ||
      n_slots == 0 || pos_offset < 0)
    return false;
  pos += pos_offset;  // B3 verify: append row `pos_offset` at its position
  const int grid = grid_for(row_elems);
  SB_NOTE_LAUNCH(kv_append_kernel, grid, 256, 0, cur_stream());
  kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(dst_k, src_k, pos, n_slots,
                                                   row_elems);
  SB_NOTE_LAUNCH(kv_append_kernel, grid, 256, 0, cur_stream());
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
  if (n_heads_q > 32) return false;  // one warp per query head; 32 warps = block limit
  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  SB_NOTE_LAUNCH(flash_attention_kernel, n_q, n_heads_q * 32, smem, cur_stream());
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
  if (n_heads_q > 32) return false;  // one warp per query head; 32 warps = block limit
  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  SB_NOTE_LAUNCH(flash_attention_chunk_kernel, n, n_heads_q * 32, smem, cur_stream());
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
  constexpr int P_ELEMS = GQA * (QB / 16) * 16 * BK;
  constexpr int RS_ELEMS = GQA * (QB / 16) * (BK / 16) * 16;
  constexpr int L_ELEMS = GQA * (QB / 16) * 16;
  constexpr int smem_bytes = HEAD_DIM * (2 * BK + 2) * (int)sizeof(__half) +
                             P_ELEMS * (int)sizeof(__half) +
                             (RS_ELEMS + L_ELEMS) * (int)sizeof(float);
  auto kfn = flash_attention_tiled_kernel<HEAD_DIM, BK, NW, QB, GQA>;
  static bool configured = false;
  if (!configured) {
    cudaFuncSetAttribute((const void*)kfn,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    configured = true;
  }
  const int grid_x = (n + QB - 1) / QB;
  SB_NOTE_LAUNCH_TP("flash_attention_tiled_kernel",
                    ::sonicboom::planner::plan_detail::tp_label({{"HEAD_DIM", HEAD_DIM},
                                                                 {"BK", BK}, {"NW", NW},
                                                                 {"QB", QB}, {"GQA", GQA}}),
                    dim3(grid_x, n_heads_kv), threads, smem_bytes, cur_stream());
  kfn<<<dim3(grid_x, n_heads_kv), threads, smem_bytes, cur_stream()>>>(
      q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q, n_heads_kv,
      n, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

namespace {

// --- Stage 3c: attention implementation selection --------------------------
//
// Two implementations serve the same contract: the native tiled kernel
// ("own") and the compiler-generated FA-2 kernel ("codegen").
// SONICBOOM_ATTN_IMPL pins one of them; "auto" uses the measured
// per-shape-class table below. The imported ggml mma kernel was retired; the
// old "ggml" value is accepted as an alias for "auto" (with a warning) so
// existing scripts keep working. The table key is (head_dim, gqa,
// mask kind) only — never the batch size or sequence length — for the same
// batch-invariance reason as the MMQ decomposition policy: a reused prefix
// must take the same path as the full prefill, and a different implementation
// has a different reduction order.
enum class AttnImpl { Own, Codegen };

// The generated kernel's default tile config for a shape class, without the
// SONICBOOM_ATTN_CODEGEN pin; empty when the generator does not cover the
// shape. These are the per-class starting points of the Step 7 runtime tuner:
// the tuner times them against every supported (br, bc, dps) and keeps the
// default unless a candidate is faster by the switch margin.
// exp2 softmax arm (C6, opt-in): the generated kernel folds log2(e) into the
// score scale and exponentiates with exp2. The folded scale rounds
// differently from the exp path, so it is not token-identical and stays
// behind an explicit switch.
bool attn_exp2_enabled() {
  return std::getenv("SONICBOOM_ATTN_EXP2") != nullptr;
}

// Build a prefill shape field by field. nvcc's host front end stops applying
// designated initializers at the first skipped member: a shape written as
// `.head_dim = ..., ..., .async_stages = ...` leaves `decode` at its default
// and silently drops every initializer after it (measured: warp_head and
// async_stages stayed 0 while the other fields were set). Assign explicitly.
compiler::AttentionShape attn_prefill_shape(int head_dim, int gqa, bool window,
                                            int br, int bc, int dps, bool wh,
                                            int as) {
  compiler::AttentionShape s;
  s.head_dim = head_dim;
  s.gqa = gqa;
  s.window_mask = window;
  s.br = br;
  s.bc = bc;
  s.dps = dps;
  s.warp_head = wh;
  s.async_stages = as;
  s.exp2 = attn_exp2_enabled();
  return s;
}

std::optional<compiler::AttentionShape> attn_codegen_default_shape(int head_dim,
                                                                   int gqa,
                                                                   bool window) {
  compiler::AttentionShape shape{
      .head_dim = head_dim, .gqa = gqa, .window_mask = window};
  shape.exp2 = attn_exp2_enabled();
  // Tile configs from the Stage 3c sweep (test_cuda_attention_codegen_wide
  // --bench; RTX 3050/sm_86). Each warp keeps DC = head_dim/dps <= 128 output
  // columns; wider tiles buy occupancy only until the dynamic-smem limit
  // (96 KiB) or the 32-warp block limit is reached.
  if (gqa == 7 && head_dim == 64) {
    // br = 64, bc = 16, dps = 1, two-buffer cp.async staging (one warp, 8
    // KiB). Since attn9 the generated prefill is bit-identical to the
    // hand-written tiled kernel at dps == 1 (attn_choice_bit_identical), so
    // the default is the measured winner over that identical set: pp512 Qwen
    // on the 3050 is 16933 tok/s here against 15057 for the synchronous
    // 64x32 tile (3-run prefill_bench, SONICBOOM_AUTOTUNE=0). bc = 32 and the
    // synchronous depth stay in the tuner's candidate set.
    shape.br = 64;
    shape.bc = 16;
    shape.dps = 1;
    shape.async_stages = 2;
  } else if (gqa == 4 && head_dim == 256) {
    shape.br = 64;
    shape.bc = 16;
    shape.dps = 4;  // 16 warps, 64 KiB
  } else if (gqa == 4 && head_dim == 512) {
    shape.br = 32;
    shape.bc = 16;
    shape.dps = 8;  // 16 warps, 80 KiB
  } else {
    return std::nullopt;
  }
  if (!compiler::attention_shape_supported(shape)) return std::nullopt;
  return shape;
}

// Shape class -> generator shape, with SONICBOOM_ATTN_CODEGEN="br,bc,dps"
// pinning the tile config (Stage 3c sweep). An unsupported pin is rejected
// like an unsupported shape.
std::optional<compiler::AttentionShape> attn_codegen_shape(int head_dim,
                                                           int gqa,
                                                           bool window) {
  std::optional<compiler::AttentionShape> shape =
      attn_codegen_default_shape(head_dim, gqa, window);
  if (!shape) return std::nullopt;
  if (const char *e = std::getenv("SONICBOOM_ATTN_CODEGEN")) {
    int br = 0, bc = 0, dps = 0, as = 0;
    if (std::sscanf(e, "wh,%d,%d,%d", &br, &bc, &as) >= 2 && br > 0 && bc > 0) {
      // P-in-registers (warp-head) arm: br must be 16, dps is implicit. The
      // optional third value is the cp.async pipeline depth.
      shape->warp_head = true;
      shape->br = br;
      shape->bc = bc;
      shape->dps = 1;
      shape->async_stages = as;
    } else if (std::sscanf(e, "%d,%d,%d,%d", &br, &bc, &dps, &as) >= 3 &&
               br > 0 && bc > 0 && dps > 0) {
      shape->br = br;
      shape->bc = bc;
      shape->dps = dps;
      shape->async_stages = as;
    }
  }
  if (!compiler::attention_shape_supported(*shape)) return std::nullopt;
  return *shape;
}

bool attn_launch_codegen(const compiler::AttentionShape &shape,
                         const float *q, const float *k_batch,
                         const float *v_batch, const float *k_cache,
                         const float *v_cache, const uint64_t *start,
                         uint64_t n_slots, int n, int n_heads_q,
                         int n_heads_kv, uint64_t sliding_window, float scale,
                         float *out) {
  const float *q_p = q, *kb_p = k_batch, *vb_p = v_batch, *kc_p = k_cache,
              *vc_p = v_cache;
  const uint64_t *start_p = start;
  uint64_t nslots_p = n_slots, window_p = sliding_window;
  int nhq_p = n_heads_q, nhkv_p = n_heads_kv, n_p = n;
  float scale_p = scale;
  float *out_p = out;
  void *args[] = {&q_p,      &kb_p,    &vb_p,   &kc_p,   &vc_p,  &start_p,
                  &nslots_p, &nhq_p,   &nhkv_p, &n_p,    &window_p,
                  &scale_p,  &out_p};
  const int grid_x = (n + shape.br - 1) / shape.br;
  // Warp-head blocks are (query tile, KV head): each warp covers one of the
  // KV head's query heads.
  const int grid_y = shape.warp_head ? n_heads_kv : n_heads_q;
  return planner::glue::launch_attention(shape, args, grid_x, grid_y);
}

// Prefill implementation choice for a shape class: the hand-written kernel
// ("own") or the generated FA-2 kernel at a (br, bc, dps) tile config.
struct AttnChoice {
  AttnImpl impl = AttnImpl::Own;
  int br = 0;
  int bc = 0;
  int dps = 0;
  // P-in-registers variant: one warp per (query head, 16-row chunk), no d
  // split, S/P never in shared memory. Not token-identical to the dps split
  // (different S reduction association), so only an opt-in arm.
  bool wh = false;
  // cp.async staging pipeline depth (0 = synchronous staging). Token streams
  // are bit-identical to the synchronous staging.
  int as = 0;
};

bool attn_same_choice(const AttnChoice &a, const AttnChoice &b) {
  if (a.impl != b.impl) return false;
  if (a.wh != b.wh) return false;
  return a.impl == AttnImpl::Own ||
         (a.br == b.br && a.bc == b.bc && a.dps == b.dps && a.as == b.as);
}

std::string attn_choice_name(const AttnChoice &c) {
  if (c.impl == AttnImpl::Own) return "own";
  std::string n = std::string(c.wh ? "codegen-wh:" : "codegen:") +
                  std::to_string(c.br) + "," + std::to_string(c.bc) + "," +
                  std::to_string(c.dps);
  if (c.as >= 2) n += ",as" + std::to_string(c.as);
  return n;
}

// Does `c` reproduce the hand-written tiled kernel's token stream?
//
// At dps == 1 the S dot product is one warp's mma chain over the full
// head_dim, exactly as in the own kernel; every other generated-kernel lever
// moves no arithmetic between lanes:
//   - br/bc only choose which 16-key sub-tile a key lands in, and sub-tile
//     boundaries are absolute 16-key multiples in both kernels (own aligns its
//     BK tiles to the same multiples), so the sub-tile sequence over the valid
//     keys is the same; masked leading keys contribute exactly zero,
//   - warp_head (wh) keeps the same per-row reduction, just without the dps
//     split (dps is implicitly 1 there),
//   - async_stages stages the same values through cp.async (a copy, no math),
//   - exp2 is excluded from this function's callers entirely (see
//     attn_exp2_enabled: it changes exp to exp2, an explicit numerics arm).
// Splitting head_dim across dps warps sums per-warp partial dots, and that
// association differs from the mma chain; measured on sm_86, d64/g7, causal
// and windowed, start 0..200: every dps == 1 arm had 0 bitwise differences
// against own, every dps > 1 arm differed on 30-80% of elements.
//
// The d256/d512 classes default to a dps > 1 kernel, i.e. they are already on
// the non-identical arm relative to own; for them the conservative rule stays
// (only as, and br under causal, may move) so a class never changes its
// numbers twice at once.
bool attn_choice_bit_identical(int head_dim, int gqa, bool window,
                               const AttnChoice &def, const AttnChoice &c) {
  if (c.impl == AttnImpl::Own) return true;  // it is the reference kernel
  if (std::getenv("SONICBOOM_ATTN_TUNE_NONIDENTICAL") != nullptr) return true;
  if (head_dim == 64 && gqa == 7) return c.dps == def.dps;
  return c.impl == def.impl && c.bc == def.bc && c.dps == def.dps &&
         c.wh == def.wh && (!window || c.br == def.br);
}

// The static per-shape-class choice: what auto used before the runtime tuner
// existed, and what it still uses when tuning is off (SONICBOOM_AUTOTUNE=0)
// or unavailable (graph capture). The tuner measures these same defaults
// first and keeps them unless a candidate wins by the switch margin.
AttnChoice attn_default_choice(int head_dim, int gqa, bool window) {
  // d64 (Qwen2.5, GQA 7): generated prefill is the default since attn9 made
  // its softmax bookkeeping bit-identical to the hand-written tiled kernel
  // (left-to-right row sum, fused l update). The ablation arm
  // SONICBOOM_ATTN_PREFILL_D64=own pins the hand-written kernel.
  //
  // Decode is separately keyed: the generated split-KV pair is token-identical
  // there too (C7), but at d64 the hand-written split-KV kernel is faster
  // (513 keys: own 21.8 us vs generated 38.6; 4097 keys: 156.7 vs 181.4), so
  // this choice is the prefill table only.
  // D0: at dps == 1 every generated lever (br, bc, wh, async depth) is
  // bit-identical to own (attn_choice_bit_identical), so the static default is
  // the measured winner over that identical set: the async-staged 64x16 tile
  // (pp512 Qwen, 3050: 16933 tok/s vs 15057 for the synchronous 64x32).
  if (gqa == 7 && head_dim == 64) {
    if (const char *e = std::getenv("SONICBOOM_ATTN_PREFILL_D64");
        e != nullptr && std::string_view(e) == "own")
      return {AttnImpl::Own, 0, 0, 0};
    if (const auto s = attn_codegen_default_shape(head_dim, gqa, window))
      return {AttnImpl::Codegen, s->br, s->bc, s->dps, false, s->async_stages};
    return {AttnImpl::Own, 0, 0, 0};
  }
  // d256: codegen with br=64, bc=16, dps=4 is faster on both legs
  // (0.318/0.231 ms vs own 0.424/0.243).
  // d512: codegen with br=32, bc=16, dps=8 wins decode (0.566 vs 0.704);
  // own keeps a prefill lead at q >= 128 (1.62 vs 1.86 ms per call at
  // pp512), but the class must stay on one kernel: a prefix-reuse suffix
  // prefill must be bit-identical to the same rows of a full prefill, and
  // the two kernels are not (q=64: 57.44 ms generated vs 58.94 own). Gating
  // own at q >= 128 mixed them by n and diverged test_prefix_cache, so the
  // static table keeps codegen here and the tune-time measurement decides
  // the generated tile for the prefill leg.
  if (gqa == 4 && (head_dim == 256 || head_dim == 512)) {
    if (const auto s = attn_codegen_default_shape(head_dim, gqa, window))
      return {AttnImpl::Codegen, s->br, s->bc, s->dps, false, s->async_stages};
  }
  return {AttnImpl::Own, 0, 0, 0};
}

// Launch the hand-written tiled kernel for the tuned shape classes, so the
// tuner can time it with the same synthetic buffers as the generated kernel.
bool attn_launch_own(int head_dim, int gqa, const float *q,
                     const float *k_batch, const float *v_batch,
                     const float *k_cache, const float *v_cache,
                     const uint64_t *start, uint64_t n_slots, int n,
                     int n_heads_q, int n_heads_kv, uint64_t sliding_window,
                     float scale, float *out) {
  if (gqa == 4 && head_dim == 256)
    return launch_tiled_flash_attention<256, 64, 2, 32, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
  if (gqa == 4 && head_dim == 512)
    return launch_tiled_flash_attention<512, 32, 4, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
  if (gqa == 7 && head_dim == 64)
    return launch_tiled_flash_attention<64, 64, 2, 32, 7>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
  return false;
}

// --- Step 7: per-GPU runtime tuning -----------------------------------
//
// The class default is timed once per (shape class, GPU) and every supported
// generated (br, bc, dps) plus "own" is timed against it; the winner is
// persisted in the autotune cache (planner/autotune.h) and reused. The
// synthetic call is a canonical prefill chunk at position 0, so the search
// never sees the caller's batch size or position and a reused prefix resolves
// to the same config as a full prefill (batch invariance).
constexpr int kAttnTuneQueries = 256;
constexpr uint64_t kAttnTuneSlots = 256;
// A candidate must beat the class default by this fraction to replace it.
// The defaults are the measured 3050 winners; the margin keeps a re-measurement
// of equal kernels from changing the default token stream.
constexpr double kAttnSwitchMargin = 0.03;

bool attn_capturing() {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(cur_stream(), &st) != cudaSuccess ||
         st != cudaStreamCaptureStatusNone;
}

struct AttnTuneBufs {
  float *q = nullptr, *kb = nullptr, *vb = nullptr, *kc = nullptr,
        *vc = nullptr, *out = nullptr;
  uint64_t *start = nullptr;
  ~AttnTuneBufs() {
    for (float *p : {q, kb, vb, kc, vc, out})
      if (p) cudaFree(p);
    if (start) cudaFree(start);
  }
};

AttnChoice attn_tune_prefill(int head_dim, int gqa, int n_heads_q,
                             int n_heads_kv, uint64_t window_len, bool window,
                             const AttnChoice &def, double &tune_s,
                             int &candidate_count) {
  const auto t0 = std::chrono::steady_clock::now();
  candidate_count = 0;

  // Search space from the device limits: the generator's own shape rules
  // (16-row sub-tiles, dps <= 8, DC <= 128, 32-warp blocks, 96 KiB) plus the
  // device's shared-memory and thread limits.
  int dev = 0, max_smem = 96 * 1024, max_threads = 1024;
  if (cudaGetDevice(&dev) == cudaSuccess) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               dev) == cudaSuccess &&
        v > 0)
      max_smem = v;
    v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerBlock, dev) ==
            cudaSuccess &&
        v > 0)
      max_threads = v;
  }

  std::vector<AttnChoice> cands{def};
  auto add_candidate = [&](const compiler::AttentionShape &s, const AttnChoice &c) {
    if (!compiler::attention_shape_supported(s)) return;
    if (compiler::attention_shared_bytes(s) > max_smem) return;
    if (compiler::attention_block_threads(s) > max_threads) return;
    // Only candidates whose token stream matches the hand-written kernel (or
    // the non-identical arm, when explicitly enabled) enter the search; see
    // attn_choice_bit_identical for which levers qualify per shape class.
    if (!attn_choice_bit_identical(head_dim, gqa, window, def, c)) return;
    for (const AttnChoice &x : cands)
      if (attn_same_choice(x, c)) return;
    cands.push_back(c);
  };
  for (int as : {0, 2, 3})
    for (int dps : {1, 2, 4, 8})
      for (int bc : {16, 32, 48, 64})
        for (int br : {16, 32, 48, 64}) {
          const compiler::AttentionShape s = attn_prefill_shape(
              head_dim, gqa, window, br, bc, dps, false, as);
          add_candidate(s, {AttnImpl::Codegen, br, bc, dps, false, as});
        }
  // P-in-registers arm (warp per query head, no d split). Its dps is 1, so it
  // is bit-identical for the d64 class and opt-in elsewhere (see
  // attn_choice_bit_identical); add_candidate applies that rule.
  for (int bc : {16, 32, 48, 64}) {
    const compiler::AttentionShape s =
        attn_prefill_shape(head_dim, gqa, window, 16, bc, 1, true, 0);
    add_candidate(s, {AttnImpl::Codegen, 16, bc, 1, true});
  }

  AttnTuneBufs b;
  const size_t q_n = size_t(kAttnTuneQueries) * n_heads_q * head_dim;
  const size_t kv_n = size_t(kAttnTuneQueries) * n_heads_kv * head_dim;
  const size_t c_n = size_t(kAttnTuneSlots) * n_heads_kv * head_dim;
  bool ok = cudaMalloc(reinterpret_cast<void **>(&b.q), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kc), c_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vc), c_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.out), q_n * 4) == cudaSuccess;
  ok = ok &&
       cudaMalloc(reinterpret_cast<void **>(&b.start), sizeof(uint64_t)) ==
           cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    tune_s = 0.0;
    return def;
  }
  // Zeroed inputs: the kernel's work is data-independent (no early exits), and
  // zeros keep denormals out of the timing.
  cudaMemsetAsync(b.q, 0, q_n * 4, cur_stream());
  cudaMemsetAsync(b.kb, 0, kv_n * 4, cur_stream());
  cudaMemsetAsync(b.vb, 0, kv_n * 4, cur_stream());
  cudaMemsetAsync(b.kc, 0, c_n * 4, cur_stream());
  cudaMemsetAsync(b.vc, 0, c_n * 4, cur_stream());
  cudaMemsetAsync(b.out, 0, q_n * 4, cur_stream());
  const uint64_t zero = 0;
  ok = cudaMemcpyAsync(b.start, &zero, sizeof(zero), cudaMemcpyHostToDevice,
                       cur_stream()) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    tune_s = 0.0;
    return def;
  }

  const float scale = 1.0f / std::sqrt(float(head_dim));
  const uint64_t w = window ? window_len : 0;
  auto launch = [&](const AttnChoice &c) {
    if (c.impl == AttnImpl::Own)
      return attn_launch_own(head_dim, gqa, b.q, b.kb, b.vb, b.kc, b.vc,
                             b.start, kAttnTuneSlots, kAttnTuneQueries,
                             n_heads_q, n_heads_kv, w, scale, b.out);
    const compiler::AttentionShape s = attn_prefill_shape(
        head_dim, gqa, window, c.br, c.bc, c.dps, c.wh, c.as);
    return attn_launch_codegen(s, b.q, b.kb, b.vb, b.kc, b.vc, b.start,
                               kAttnTuneSlots, kAttnTuneQueries, n_heads_q,
                               n_heads_kv, w, scale, b.out);
  };

  candidate_count = int(cands.size());
  // The timing launches are not part of any run: the first execution of a
  // prefill may be the one being recorded, and a recorded plan must name the
  // chosen kernel's real grid, not the tuning pass's synthetic candidates
  // (same rule as the dense tuner).
  planner::PlanRecorder *saved_rec = planner::recorder();
  planner::set_thread_recorder(nullptr);
  const double def_ms =
      ::sonicboom::tune::median_launch_ms([&] { return launch(def); },
                                          cur_stream(), 2, 5);
  AttnChoice best = def;
  double best_ms = def_ms;
  for (size_t i = 1; i < cands.size(); ++i) {
    const double ms =
        ::sonicboom::tune::median_launch_ms([&] { return launch(cands[i]); },
                                            cur_stream(), 2, 5);
    if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr)
      std::fprintf(stderr, "[autotune]   attn cand %s %.4f ms (best %.4f)\n",
                   attn_choice_name(cands[i]).c_str(), ms, best_ms);
    if (ms > 0.0 && (best_ms <= 0.0 || ms < best_ms)) {
      best = cands[i];
      best_ms = ms;
    }
  }
  if (best_ms <= 0.0)
    best = def;
  else if (!attn_same_choice(best, def) && def_ms > 0.0 &&
           !(best_ms < def_ms * (1.0 - kAttnSwitchMargin)))
    best = def;  // within the noise margin: keep the measured default
  planner::set_thread_recorder(saved_rec);

  tune_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count();
  return best;
}

// --- Q3: prefill attention shape classes ----------------------------------
//
// The generated prefill kernel's tile config is (br, bc, dps, as). Q3 exposes
// (br, bc, as) in real units: tile_m = br (query rows per block), tile_n = dc
// = head_dim/dps (head columns per warp), tile_k = bc (key columns per tile),
// unroll = as (cp.async pipeline depth). dps stays at the production shape's
// value: another split changes the S reduction order, so it would fail the
// runner's bit-identity gate. The production config (the shape the dispatcher
// chose) is always candidates[0], the bit-identity reference and the time the
// winner must beat by more than the measured burst-to-burst noise.
//
// The class key is (head_dim, seq_len bucket, query-head bucket) and the
// choice lives in the mlir_tune per-SM cache under "attn_prefill", gated by
// the autotune switch (on by default, SONICBOOM_AUTOTUNE=0 disables).
//
// Returns a shape with the winner's knobs applied, or nullopt to keep `base`
// (gate off, capture, probe failure, no bit-identical winner).
static std::optional<compiler::AttentionShape> attn_prefill_tile_autotune(
    int head_dim, int gqa, int n_heads_q, int n_heads_kv, int n,
    const compiler::AttentionShape &base, uint64_t sliding_window,
    float scale) {
  using mlir_tune::TileConfig;
  if (n <= 0 || base.warp_head) return std::nullopt;
  if (!mlir_tune::attn_prefill_tunable(head_dim)) return std::nullopt;
  if (!mlir_tune::enabled()) return std::nullopt;

  auto apply = [&](const TileConfig &cfg)
      -> std::optional<compiler::AttentionShape> {
    if (cfg.tile_m <= 0 || cfg.tile_n <= 0 || head_dim % cfg.tile_n != 0)
      return std::nullopt;
    if (cfg.tile_k < 16 || cfg.tile_k > 64 || cfg.tile_k % 16 != 0)
      return std::nullopt;
    if (cfg.unroll != 0 && cfg.unroll != 2 && cfg.unroll != 3)
      return std::nullopt;
    compiler::AttentionShape s = base;
    s.br = cfg.tile_m;
    s.bc = cfg.tile_k;
    s.dps = head_dim / cfg.tile_n;
    s.async_stages = cfg.unroll;
    s.warp_head = false;
    if (!compiler::attention_shape_supported(s)) return std::nullopt;
    if (compiler::attention_shared_bytes(s) > 96 * 1024) return std::nullopt;
    return s;
  };
  // The dispatcher's own shape: the bit-identity reference and the default we
  // keep unless a candidate clearly beats it.
  const TileConfig production{base.br, head_dim / base.dps, base.bc,
                              base.async_stages, 1};

  const mlir_tune::AttnPrefillChoice choice =
      mlir_tune::attn_prefill_choice(head_dim, n, n_heads_q);
  if (choice.resolved()) return apply(choice.cfg);
  if (attn_capturing()) return std::nullopt;  // host timing is illegal there
  // Out of cold budget: keep the dispatcher's shape without allocating the
  // probe buffers (the choice function memoizes a Skipped record, so this is
  // reached once per class).
  if (mlir_tune::budget_exhausted()) return std::nullopt;

  // Probe at the caller's query length (capped so the scratch stays small; a
  // class is a bucket, and any length inside it is a valid probe).
  const int probe_n = std::min(n, 1024);
  const size_t q_n = size_t(probe_n) * size_t(n_heads_q) * size_t(head_dim);
  const size_t kv_n = size_t(probe_n) * size_t(n_heads_kv) * size_t(head_dim);
  const size_t c_n = size_t(kAttnTuneSlots) * size_t(n_heads_kv) *
                     size_t(head_dim);
  AttnTuneBufs b;
  bool ok = cudaMalloc(reinterpret_cast<void **>(&b.q), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kc), c_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vc), c_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.out), q_n * 4) == cudaSuccess;
  ok = ok &&
       cudaMalloc(reinterpret_cast<void **>(&b.start), sizeof(uint64_t)) ==
           cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    return std::nullopt;
  }
  // Non-uniform Q/K/V: the bit-identity gate is only meaningful on data that
  // exercises the softmax and the P·V product (zeroed operands make every
  // tile config produce the same zeros). The kernel's work is
  // data-independent, so this does not bias the timing. The cache is unread
  // at start 0 and stays zeroed.
  auto fill = [](std::vector<float> &v) {
    uint32_t s = 12345u;
    for (float &x : v) {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
      x = float(s % 2000u) / 1000.0f - 1.0f;
    }
  };
  std::vector<float> hq(q_n), hkv(kv_n);
  fill(hq);
  fill(hkv);
  ok = cudaMemcpyAsync(b.q, hq.data(), q_n * 4, cudaMemcpyHostToDevice,
                       cur_stream()) == cudaSuccess;
  ok = ok && cudaMemcpyAsync(b.kb, hkv.data(), kv_n * 4,
                             cudaMemcpyHostToDevice, cur_stream()) ==
                 cudaSuccess;
  ok = ok && cudaMemcpyAsync(b.vb, hkv.data(), kv_n * 4,
                             cudaMemcpyHostToDevice, cur_stream()) ==
                 cudaSuccess;
  cudaMemsetAsync(b.kc, 0, c_n * 4, cur_stream());
  cudaMemsetAsync(b.vc, 0, c_n * 4, cur_stream());
  cudaMemsetAsync(b.out, 0, q_n * 4, cur_stream());
  if (!ok) {
    (void)cudaGetLastError();
    return std::nullopt;
  }
  const uint64_t zero = 0;
  ok = cudaMemcpyAsync(b.start, &zero, sizeof(zero), cudaMemcpyHostToDevice,
                       cur_stream()) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    return std::nullopt;
  }

  std::vector<float> host_out(q_n);
  bool launch_ok = true;
  constexpr int kQ3ProbeReps = 16;
  auto make_launch = [&](const compiler::AttentionShape &s) {
    return mlir_tune::AutotuneRunner::Launch([&, s] {
      bool ran = true;
      for (int i = 0; i < kQ3ProbeReps && ran; ++i)
        ran = attn_launch_codegen(s, b.q, b.kb, b.vb, b.kc, b.vc, b.start,
                                  kAttnTuneSlots, probe_n, n_heads_q,
                                  n_heads_kv, sliding_window, scale, b.out);
      launch_ok = ran;
    });
  };

  mlir_tune::AutotuneRunner::Problem p;
  p.candidates = mlir_tune::attn_prefill_candidates(head_dim, production);
  p.compile = [&](const TileConfig &cfg)
      -> std::optional<mlir_tune::AutotuneRunner::Launch> {
    const auto s = apply(cfg);
    if (!s) return std::nullopt;
    return make_launch(*s);
  };
  p.output = host_out.data();
  p.output_bytes = host_out.size() * sizeof(float);
  p.sync = [&] {
    if (!launch_ok) {
      // A failed candidate must not look like a bit-identical fast one.
      std::memset(host_out.data(), 0xFF, p.output_bytes);
      (void)cudaGetLastError();
      return;
    }
    cudaError_t e = cudaMemcpyAsync(host_out.data(), b.out, p.output_bytes,
                                    cudaMemcpyDeviceToHost, cur_stream());
    if (e == cudaSuccess) e = cudaStreamSynchronize(cur_stream());
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[autotune] attn_prefill probe: %s\n",
                   cudaGetErrorString(e));
      std::memset(host_out.data(), 0xFF, p.output_bytes);
      (void)cudaGetLastError();
    }
  };
  // Timing does not need the bit-identity window on the host: the ~900 KiB
  // copy is identical for every candidate and was larger than the tile
  // difference being measured. The check runs and the final winner run still
  // use the copying sync above.
  p.timing_sync = [&] { (void)cudaStreamSynchronize(cur_stream()); };
  // 8 timed bursts of 16 launches, min: the tile differences are small and the
  // choice is persisted, so the measurement must be stable.
  p.iters = 8;

  // The timing launches are not part of any run: a recorded prefill must name
  // the chosen kernel's real grid, not the probe's (same rule as the dense
  // and MMQ tuners).
  planner::PlanRecorder *saved_rec = planner::recorder();
  planner::set_thread_recorder(nullptr);
  std::optional<TileConfig> best = mlir_tune::AutotuneRunner::run(p);

  // The probe's per-burst minimum is noisy at this size, so a candidate
  // replaces production only when it beats it by more than a fixed resolution.
  // Three untimed warm-up bursts bring the memory clock up (its ramp inflated
  // the old two-sample spread), then five timed pairs in alternating order keep
  // a residual drift common to both configs; each side keeps its minimum.
  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  auto timed_burst = [&](const TileConfig &cfg) -> std::optional<double> {
    const auto s = apply(cfg);
    if (!s) return std::nullopt;
    auto launch = make_launch(*s);
    double best_s = 1e30;
    for (int i = 0; i < 2; ++i) {
      if (cudaStreamSynchronize(cur_stream()) != cudaSuccess)
        return std::nullopt;
      const auto t0 = std::chrono::steady_clock::now();
      launch();
      if (cudaStreamSynchronize(cur_stream()) != cudaSuccess)
        return std::nullopt;
      const auto t1 = std::chrono::steady_clock::now();
      best_s = std::min(
          best_s, std::chrono::duration<double>(t1 - t0).count());
    }
    return launch_ok ? std::optional<double>(best_s) : std::nullopt;
  };
  TileConfig winner = production;
  // The confirmation re-measurement is part of the same cold budget. When the
  // runner used it up, keep production rather than take a pick the confirmation
  // never reproduced.
  if (best && !(*best == production) && !mlir_tune::budget_exhausted()) {
    for (int i = 0; i < 3; ++i) {
      if (mlir_tune::budget_exhausted()) break;
      (void)timed_burst(production);
    }
    constexpr int kRounds = 5;
    double refs[kRounds] = {}, wins[kRounds] = {};
    bool ok = !mlir_tune::budget_exhausted();
    for (int i = 0; i < kRounds && ok; ++i) {
      if (mlir_tune::budget_exhausted()) {
        ok = false;
        break;
      }
      if ((i & 1) == 0) {
        const auto r = timed_burst(production);
        const auto w = timed_burst(*best);
        ok = r.has_value() && w.has_value();
        if (ok) {
          refs[i] = *r;
          wins[i] = *w;
        }
      } else {
        const auto w = timed_burst(*best);
        const auto r = timed_burst(production);
        ok = r.has_value() && w.has_value();
        if (ok) {
          refs[i] = *r;
          wins[i] = *w;
        }
      }
    }
    if (ok) {
      double ref = refs[0], win = wins[0], spread = 0;
      for (int i = 1; i < kRounds; ++i) {
        ref = std::min(ref, refs[i]);
        win = std::min(win, wins[i]);
        spread = std::max(spread, std::fabs(refs[i] - refs[0]));
      }
      constexpr double kResolution = 0.005;
      if (win < ref * (1.0 - kResolution)) winner = *best;
      if (verbose)
        std::fprintf(stderr,
                     "[autotune] attn_prefill probe: prod=%.4f ms "
                     "best=%.4f ms spread=%.2f%% -> %s\n",
                     ref * 1e3, win * 1e3, spread / refs[0] * 100.0,
                     winner == *best ? "take" : "keep production");
    }
  }
  planner::set_thread_recorder(saved_rec);

  const bool stored =
      mlir_tune::attn_prefill_choice_store(head_dim, n, n_heads_q, winner);
  if (verbose)
    std::fprintf(stderr,
                 "[autotune] %s: br=%d dc=%d bc=%d as=%d%s\n",
                 ("attn_prefill/" +
                  mlir_tune::attn_prefill_shape_class(head_dim, n, n_heads_q))
                     .c_str(),
                 winner.tile_m, winner.tile_n, winner.tile_k, winner.unroll,
                 stored ? "" : " [not persisted]");
  return apply(winner);
}

std::mutex g_attn_choice_mutex;
std::unordered_map<std::string, AttnChoice> g_attn_choice;

std::string attn_tune_key(int head_dim, int gqa, int n_heads_q, int n_heads_kv,
                          bool window, uint64_t window_len) {
  std::string k = compiler::attention_codegen_version();
  k += "|prefill-d" + std::to_string(head_dim) + "-g" + std::to_string(gqa) +
       "-h" + std::to_string(n_heads_q) + "-kv" + std::to_string(n_heads_kv) +
       "-m" + (window ? "window" : "causal");
  if (window) k += "-w" + std::to_string(window_len);
  if (attn_exp2_enabled()) k += "-x2";
  return k;
}

bool attn_choice_from_values(const std::map<std::string, int> &v, int head_dim,
                             int gqa, bool window, AttnChoice &out) {
  const auto get = [&v](const char *name, int &dst) {
    const auto it = v.find(name);
    if (it == v.end()) return false;
    dst = it->second;
    return true;
  };
  int impl = 0;
  if (!get("impl", impl)) return false;
  if (impl == 0) {
    out = {AttnImpl::Own, 0, 0, 0};
    return true;
  }
  int br = 0, bc = 0, dps = 0, wh = 0, as = 0;
  if (!get("br", br) || !get("bc", bc) || !get("dps", dps)) return false;
  (void)get("wh", wh);
  (void)get("as", as);
  const compiler::AttentionShape s = attn_prefill_shape(
      head_dim, gqa, window, br, bc, dps, wh != 0, as);
  if (!compiler::attention_shape_supported(s)) return false;
  out = {AttnImpl::Codegen, br, bc, dps, wh != 0, as};
  return true;
}

// Resolve the prefill implementation for a shape class: cached choice, else a
// tuning run on synthetic inputs, else the static default. The key is the
// shape class (head count included) and the generated-code version; it never
// includes the batch size or position.
AttnChoice attn_resolve(int head_dim, int gqa, int n_heads_q, int n_heads_kv,
                        bool window, uint64_t window_len) {
  const AttnChoice def = attn_default_choice(head_dim, gqa, window);
  if (!::sonicboom::tune::enabled()) return def;
  // Only the classes the generator covers can be tuned; the rest keep own.
  if (!attn_codegen_default_shape(head_dim, gqa, window)) return def;
  // A class whose measured default is the hand-written kernel keeps it,
  // except d64, whose generated prefill has matched the hand-written kernel
  // bitwise since attn9. The guard stays for any other own class: there the
  // generated S reduction association still differs, so moving the class
  // would change the token stream (the opt-in non-identical arm).
  if (def.impl == AttnImpl::Own &&
      std::getenv("SONICBOOM_ATTN_TUNE_NONIDENTICAL") == nullptr)
    return def;

  const std::string key =
      attn_tune_key(head_dim, gqa, n_heads_q, n_heads_kv, window, window_len);
  {
    std::lock_guard<std::mutex> lk(g_attn_choice_mutex);
    const auto it = g_attn_choice.find(key);
    if (it != g_attn_choice.end()) return it->second;
  }

  AttnChoice choice = def;
  bool have = false;
  const std::map<std::string, int> cached =
      ::sonicboom::tune::lookup("attention-prefill", key);
  // A cached choice is still subject to the token-identity rule: a cache
  // written while SONICBOOM_ATTN_TUNE_NONIDENTICAL was set must not pin a
  // divergent config once the arm is off. A rejected entry forces a retune.
  if (!cached.empty() &&
      attn_choice_from_values(cached, head_dim, gqa, window, choice) &&
      attn_choice_bit_identical(head_dim, gqa, window, def, choice))
    have = true;
  else
    choice = def;

  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  if (!have) {
    if (attn_capturing()) return def;  // host timing is illegal in a capture
    double tune_s = 0.0;
    int candidate_count = 0;
    choice = attn_tune_prefill(head_dim, gqa, n_heads_q, n_heads_kv, window_len,
                               window, def, tune_s, candidate_count);
    std::map<std::string, int> values{
        {"impl", choice.impl == AttnImpl::Codegen ? 1 : 0},
        {"br", choice.br},
        {"bc", choice.bc},
        {"dps", choice.dps},
        {"wh", choice.wh ? 1 : 0},
        {"as", choice.as}};
    ::sonicboom::tune::store("attention-prefill", key, values);
    if (verbose)
      std::fprintf(stderr,
                   "[autotune] attention-prefill %s default=%s chosen=%s "
                   "candidates=%d tune=%.2f s\n",
                   key.c_str(), attn_choice_name(def).c_str(),
                   attn_choice_name(choice).c_str(), candidate_count, tune_s);
  } else if (verbose) {
    std::fprintf(stderr, "[autotune] attention-prefill %s cached=%s\n",
                 key.c_str(), attn_choice_name(choice).c_str());
  }

  {
    std::lock_guard<std::mutex> lk(g_attn_choice_mutex);
    g_attn_choice[key] = choice;
  }
  return choice;
}

// --- Step 7: decode chunk length for the generated split-KV pair ----------
//
// The generated flash-decoding pair is the default decode path. Its default
// chunk rule reproduces the hand-written kernel's split, which is what makes
// the two bit-identical; the tuner below trades grid width against per-block
// work and therefore changes the split, so it belongs to the
// non-token-identical arm (SONICBOOM_ATTN_TUNE_NONIDENTICAL). The chunk
// length is a pure runtime argument (generated text unchanged).
// The choice is keyed by shape class + window extent, never by position.
constexpr uint64_t kAttnDecodeTuneChunks[] = {8, 16, 32, 64};
// The synthetic extent: the full window when it is small, capped so the
// partial buffer stays bounded (a 32k-key extent would need hundreds of MB of
// tuning scratch). The choice is made once per class; the kernel itself still
// runs the real extent.
constexpr uint64_t kAttnDecodeTuneMaxExtent = 4096;

std::mutex g_attn_decode_key_mutex;
std::unordered_map<std::string, uint64_t> g_attn_decode_chunk;

uint64_t attn_decode_chunk_resolve(int head_dim, int gqa, int n_heads_q,
                                   int n_heads_kv, uint64_t max_window,
                                   uint64_t sliding_window,
                                   uint64_t default_chunk) {
  if (!::sonicboom::tune::enabled()) return default_chunk;
  compiler::AttentionShape dsh;
  dsh.head_dim = head_dim;
  dsh.gqa = gqa;
  dsh.window_mask = sliding_window != 0;
  dsh.decode = true;
  dsh.exp2 = attn_exp2_enabled();
  if (!compiler::attention_shape_supported(dsh)) return default_chunk;

  std::string key = compiler::attention_codegen_version();
  key += "|decode-d" + std::to_string(head_dim) + "-g" + std::to_string(gqa) +
         "-h" + std::to_string(n_heads_q) + "-kv" +
         std::to_string(n_heads_kv) + "-m" +
         (sliding_window != 0 ? "window" : "causal");
  key += "-w" + std::to_string(max_window);
  if (attn_exp2_enabled()) key += "-x2";

  {
    std::lock_guard<std::mutex> lk(g_attn_decode_key_mutex);
    const auto it = g_attn_decode_chunk.find(key);
    if (it != g_attn_decode_chunk.end()) return it->second;
  }

  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  if (const std::map<std::string, int> cached =
          ::sonicboom::tune::lookup("attention-decode", key);
      !cached.empty()) {
    const auto it = cached.find("chunk");
    if (it != cached.end() && it->second > 0 &&
        (uint64_t)it->second <= (uint64_t)compiler::kAttentionDecodeMaxChunk) {
      const uint64_t c = (uint64_t)it->second;
      std::lock_guard<std::mutex> lk(g_attn_decode_key_mutex);
      g_attn_decode_chunk[key] = c;
      if (verbose)
        std::fprintf(stderr, "[autotune] attention-decode %s cached=chunk %llu\n",
                     key.c_str(), (unsigned long long)c);
      return c;
    }
  }
  if (attn_capturing()) return default_chunk;

  const uint64_t extent =
      std::min<uint64_t>(std::max<uint64_t>(max_window, 1),
                         kAttnDecodeTuneMaxExtent);
  const uint64_t slots = std::min<uint64_t>(extent, 512);
  AttnTuneBufs b;
  const size_t q_n = size_t(n_heads_q) * head_dim;
  bool ok =
      cudaMalloc(reinterpret_cast<void **>(&b.q), q_n * 4) == cudaSuccess;
  ok = ok &&
       cudaMalloc(reinterpret_cast<void **>(&b.kc), slots * n_heads_kv * head_dim * 4) ==
           cudaSuccess;
  ok = ok &&
       cudaMalloc(reinterpret_cast<void **>(&b.vc), slots * n_heads_kv * head_dim * 4) ==
           cudaSuccess;
  ok = ok &&
       cudaMalloc(reinterpret_cast<void **>(&b.out), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.start), sizeof(uint64_t)) ==
                 cudaSuccess;
  // Partial scratch sized for the smallest chunk (the most chunks).
  uint64_t max_chunks = 0;
  for (uint64_t c : kAttnDecodeTuneChunks)
    max_chunks = std::max<uint64_t>(max_chunks, (extent + c - 1) / c);
  max_chunks = std::max<uint64_t>(max_chunks, (extent + default_chunk - 1) /
                                                  default_chunk);
  float *partial = nullptr;
  const size_t partial_floats =
      size_t(max_chunks) * size_t(n_heads_q) * size_t(head_dim + 2);
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&partial),
                        partial_floats * 4) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    if (partial) cudaFree(partial);
    return default_chunk;
  }
  // Zeroed inputs: data-independent work, no denormals in the timing.
  cudaMemsetAsync(b.q, 0, q_n * 4, cur_stream());
  cudaMemsetAsync(b.kc, 0, slots * n_heads_kv * head_dim * 4, cur_stream());
  cudaMemsetAsync(b.vc, 0, slots * n_heads_kv * head_dim * 4, cur_stream());
  cudaMemsetAsync(b.out, 0, q_n * 4, cur_stream());
  cudaMemsetAsync(partial, 0, partial_floats * 4, cur_stream());
  // One full window of keys is valid: start = 0 when the extent does not
  // exceed the window, so the whole synthetic ring is exercised.
  const uint64_t pos_host = extent - 1;
  ok = cudaMemcpyAsync(b.start, &pos_host, sizeof(pos_host),
                       cudaMemcpyHostToDevice, cur_stream()) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    cudaFree(partial);
    return default_chunk;
  }

  const auto t0 = std::chrono::steady_clock::now();
  auto launch = [&](uint64_t c) {
    const uint64_t n_chunks = (extent + c - 1) / c;
    const float *qa = b.q;
    const float *kca = b.kc;
    const float *vca = b.vc;
    const uint64_t *startp = b.start;
    uint64_t nslots_arg = slots;
    int nhq_arg = n_heads_q;
    int nhkv_arg = n_heads_kv;
    uint64_t win_arg = sliding_window;
    float scale_arg = 1.0f / std::sqrt(float(head_dim));
    uint64_t chunk_arg = c;
    float *out_arg = b.out;
    int nch_arg = int(n_chunks);
    void *args[13] = {&qa,        &kca,       &vca,        &startp,
                      &nslots_arg, &nhq_arg,  &nhkv_arg,   &win_arg,
                      &scale_arg, &partial,   &chunk_arg,  &out_arg,
                      &nch_arg};
    return planner::glue::launch_attention_decode(dsh, args, int(n_chunks),
                                                  n_heads_q);
  };

  // Same recorder rule as the prefill tuner: a decode being recorded must not
  // carry the chunk candidates' launches.
  planner::PlanRecorder *saved_rec = planner::recorder();
  planner::set_thread_recorder(nullptr);
  const double def_ms =
      ::sonicboom::tune::median_launch_ms([&] { return launch(default_chunk); },
                                          cur_stream(), 2, 5);
  uint64_t best = default_chunk;
  double best_ms = def_ms;
  for (uint64_t c : kAttnDecodeTuneChunks) {
    if (c == default_chunk || c > (uint64_t)compiler::kAttentionDecodeMaxChunk)
      continue;
    const double ms =
        ::sonicboom::tune::median_launch_ms([&] { return launch(c); },
                                            cur_stream(), 2, 5);
    if (ms > 0.0 && (best_ms <= 0.0 || ms < best_ms)) {
      best = c;
      best_ms = ms;
    }
  }
  if (best_ms <= 0.0)
    best = default_chunk;
  else if (best != default_chunk && def_ms > 0.0 &&
           !(best_ms < def_ms * (1.0 - kAttnSwitchMargin)))
    best = default_chunk;
  planner::set_thread_recorder(saved_rec);
  cudaFree(partial);

  ::sonicboom::tune::store("attention-decode", key, {{"chunk", int(best)}});
  if (verbose)
    std::fprintf(stderr,
                 "[autotune] attention-decode %s default_chunk=%llu "
                 "chosen_chunk=%llu default=%.4f ms chosen=%.4f ms tune=%.2f s\n",
                 key.c_str(), (unsigned long long)default_chunk,
                 (unsigned long long)best, def_ms, best_ms,
                 std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                               t0)
                     .count());

  {
    std::lock_guard<std::mutex> lk(g_attn_decode_key_mutex);
    g_attn_decode_chunk[key] = best;
  }
  return best;
}

} // namespace

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
  if (n_heads_q > 32) return false;  // one warp per query head; 32 warps = block limit

  // Tiled tensor-core path, selected by shape (head_dim, gqa): the tile config
  // is a pure function of those two, so a shape either has an instantiation
  // here or falls through to the scalar-fp32 warp kernel below (correct for
  // every shape, just per-warp redundant on K/V and scalar on the dot). The
  // block is NW*GQA*(QB/16) warps (<= 32 => <= 1024 threads) and its smem is
  // sized in the launcher, which also raises the dynamic-smem attribute.
  //   - Gemma-family: gqa 4, SWA head 256 / global head 512.
  //   - SwiGLU families (Qwen2.5-0.5B, and the same 7:1 group ratio at the
  //     wider head): gqa 7, head_dim 64.
  const int gqa = n_heads_q / n_heads_kv;

  // Attention implementation switch (v1 Stages 2-3). "auto" (default)
  // resolves the shape class through the runtime tuner (attn_resolve, Step 7)
  // or its static defaults when tuning is off; its key is the shape class and
  // not the batch size, so a reused prefix takes the same path as a full
  // prefill. "own" pins the native tiled kernel below, "codegen" the
  // compiler-generated FA-2 kernel; the pinned implementations fall back to
  // the native kernel for shapes they do not cover. "ggml" was the imported
  // mma kernel, now retired: it maps to "auto" (once, with a warning) rather
  // than failing, so an old script still runs the measured choice instead of
  // silently changing behavior.
  const char* attn_impl_env = std::getenv("SONICBOOM_ATTN_IMPL");
  std::string_view attn_impl =
      attn_impl_env ? std::string_view(attn_impl_env) : std::string_view("auto");
  if (attn_impl == "ggml") {
    static const bool warned = [] {
      std::fprintf(stderr,
                   "[attention] SONICBOOM_ATTN_IMPL=ggml is retired; using "
                   "auto\n");
      return true;
    }();
    (void)warned;
    attn_impl = "auto";
  }
  // Prefill-only ablation switch (D0): SONICBOOM_ATTN_PREFILL=own reverts this
  // leg to the hand-written tiled kernel for every shape class, leaving the
  // decode leg's implementation untouched. "codegen" pins the generated
  // default shape, "auto" (or unset) keeps the normal resolution.
  if (const char *e = std::getenv("SONICBOOM_ATTN_PREFILL"); e != nullptr &&
      (std::string_view(e) == "own" || std::string_view(e) == "codegen"))
    attn_impl = e;
  std::optional<compiler::AttentionShape> shape;
  if (attn_impl == "codegen") {
    shape = attn_codegen_shape(head_dim, gqa, sliding_window != 0);
  } else if (attn_impl == "auto") {
    const bool windowed = sliding_window != 0;
    if (std::getenv("SONICBOOM_ATTN_CODEGEN") != nullptr &&
        attn_default_choice(head_dim, gqa, windowed).impl == AttnImpl::Codegen) {
      // Explicit tile pin (the Stage 3c sweep): codegen at that config. The
      // pin does not move a class whose measured default is own.
      shape = attn_codegen_shape(head_dim, gqa, windowed);
    } else {
      const AttnChoice c = attn_resolve(head_dim, gqa, n_heads_q, n_heads_kv,
                                        windowed, sliding_window);
      if (c.impl == AttnImpl::Codegen) {
        const compiler::AttentionShape s = attn_prefill_shape(
            head_dim, gqa, windowed, c.br, c.bc, c.dps, c.wh, c.as);
        if (compiler::attention_shape_supported(s)) shape = s;
      }
    }
  }

  // Q3: when the class resolves to the generated kernel, tune its
  // (tile_seq, tile_head) once per (head_dim, seq_len bucket, head bucket).
  // An explicit tile pin (the Stage 3c sweep) bypasses the tuner, so the
  // sweep measures exactly the config it names; an implementation pin
  // (SONICBOOM_ATTN_IMPL / SONICBOOM_ATTN_PREFILL) still lets Q3 choose the
  // generated kernel's tile.
  const bool tile_pinned = std::getenv("SONICBOOM_ATTN_CODEGEN") != nullptr;
  if (shape && !tile_pinned) {
    if (auto tuned =
            attn_prefill_tile_autotune(head_dim, gqa, n_heads_q, n_heads_kv, n,
                                       *shape, sliding_window, scale))
      shape = *tuned;
  }

  // Stage 3 generated FA-2 kernel, for the shape classes the generator covers.
  // A shape without a generator (or a failed compile) falls through to the
  // hand-written kernels below.
  if (shape &&
      attn_launch_codegen(*shape, q, k_batch, v_batch, k_cache, v_cache, start,
                          n_slots, n, n_heads_q, n_heads_kv, sliding_window,
                          scale, out)) {
    if (attn_trace_on()) {
      std::string cfg =
          (shape->warp_head ? "wh," : "") + std::to_string(shape->br) + "," +
          std::to_string(shape->bc) + "," + std::to_string(shape->dps);
      if (shape->async_stages >= 2)
        cfg += ",as" + std::to_string(shape->async_stages);
      if (shape->exp2) cfg += ",x2";
      attn_trace_prefill(
          attn_trace_read_start(start), n, head_dim, gqa, sliding_window,
          "codegen", cfg.c_str(), shape->br, shape->bc,
          (n + shape->br - 1) / shape->br, n_heads_q,
          compiler::attention_block_threads(*shape),
          (size_t)compiler::attention_shared_bytes(*shape));
    }
    return true;
  }

  if (gqa == 4 && head_dim == 256) {
    const bool ok = launch_tiled_flash_attention<256, 64, 2, 32, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
    if (ok)
      attn_trace_own<256, 64, 2, 32, 4>(start, n, n_heads_kv, head_dim, gqa,
                           sliding_window);
    return ok;
  }
  if (gqa == 4 && head_dim == 512) {
    const bool ok = launch_tiled_flash_attention<512, 32, 4, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
    if (ok)
      attn_trace_own<512, 32, 4, 16, 4>(start, n, n_heads_kv, head_dim, gqa,
                           sliding_window);
    return ok;
  }
  if (gqa == 7 && head_dim == 64) {
    const bool ok = launch_tiled_flash_attention<64, 64, 2, 32, 7>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
    if (ok)
      attn_trace_own<64, 64, 2, 32, 7>(start, n, n_heads_kv, head_dim, gqa,
                           sliding_window);
    return ok;
  }

  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  SB_NOTE_LAUNCH(flash_attention_chunk_pos_kernel, n, n_heads_q * 32, smem, cur_stream());
  flash_attention_chunk_pos_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
      q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q,
      n_heads_kv, head_dim, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

bool layer_combine_dev(const float* proj, const float* ple, float ple_scale,
                       float combine_scale, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !proj || !ple || !y) return false;
  SB_NOTE_LAUNCH(layer_combine_kernel, grid_for(n), 256, 0, cur_stream());
  layer_combine_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(proj, ple, ple_scale, combine_scale,
                                             y, n);
  return cudaGetLastError() == cudaSuccess;
}

// Batched (position-major) layer combine: proj/y are [n, per], ple is
// [n, total_per], and the layer's slice is read with a total_per stride (the
// per-layer embedding row is position-major, so each position's per-slice is
// strided, not contiguous). One thread per (position, per) output element.
__global__ void layer_combine_strided_kernel(const float* __restrict__ proj,
                                             const float* __restrict__ ple,
                                             float ple_scale, float combine_scale,
                                             float* __restrict__ y, int per,
                                             int n, int col, int total_per) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n * per) return;
  const int c = i / per;
  const int k = i - c * per;
  y[i] = (proj[i] + ple[(size_t)c * total_per + col + k] * ple_scale) * combine_scale;
}

bool layer_combine_batched_dev(const float* proj, const float* ple, float ple_scale,
                               float combine_scale, float* y, int per, int n,
                               int col, int total_per) {
  if (!available()) return false;
  if (per <= 0 || n <= 0 || total_per <= 0 || !proj || !ple || !y) return false;
  SB_NOTE_LAUNCH(layer_combine_strided_kernel, grid_for(n * per), 256, 0, cur_stream());
  layer_combine_strided_kernel<<<grid_for(n * per), 256, 0, cur_stream()>>>(
      proj, ple, ple_scale, combine_scale, y, per, n, col, total_per);
  return cudaGetLastError() == cudaSuccess;
}

bool argmax_dev(const float* x, int n, float* out) {
  if (!available()) return false;
  if (n <= 0 || !x || !out) return false;
  constexpr int kThreads = 256;
  const int n4 = n >> 2;
  // Enough float4s for at least two blocks, and a 16 B-aligned base (the pool
  // allocations are, but the general case has to degrade gracefully).
  int grid = (n4 + kThreads - 1) / kThreads;
  if (grid > sm_count() * 8) grid = sm_count() * 8;
  if (grid < 2 || (reinterpret_cast<uintptr_t>(x) & 15u) != 0) {
    SB_NOTE_LAUNCH(argmax_kernel, 1, 1024, 0, cur_stream());
    argmax_kernel<<<1, 1024, 0, cur_stream()>>>(x, out, n);
    return cudaGetLastError() == cudaSuccess;
  }
  unsigned long long* keys = static_cast<unsigned long long*>(grow_scratch(
      g_argmax_keys, g_argmax_keys_bytes,
      (size_t)grid * sizeof(unsigned long long)));
  if (!keys) return false;
  SB_NOTE_LAUNCH(argmax_partial_kernel, grid, kThreads, 0, cur_stream());
  argmax_partial_kernel<<<grid, kThreads, 0, cur_stream()>>>(x, n, keys);
  SB_NOTE_LAUNCH(argmax_finalize_kernel, 1, 256, 0, cur_stream());
  argmax_finalize_kernel<<<1, 256, 0, cur_stream()>>>(keys, grid, out);
  return cudaGetLastError() == cudaSuccess;
}

// Strided row slice (Section C.5): copies `rows` contiguous `length`-element
// rows out of a position-major source, y[r*length + k] = x[offset + r*stride + k].
// Used to extract a layer's per-layer-embedding column [offset, offset+length)
// from each of the n positions of the full [n, total_per] row. One thread per
// output element.
__global__ void slice_strided_kernel(const float* __restrict__ x,
                                     float* __restrict__ y, int rows, int length,
                                     int offset, int stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * length) return;
  const int r = i / length;
  const int k = i - r * length;
  y[i] = x[(size_t)offset + (size_t)r * stride + k];
}

bool slice_strided_dev(const float* x, float* y, int rows, int length, int offset,
                       int stride) {
  if (!available()) return false;
  if (rows <= 0 || length <= 0 || stride <= 0 || !x || !y) return false;
  SB_NOTE_LAUNCH(slice_strided_kernel, grid_for(rows * length), 256, 0, cur_stream());
  slice_strided_kernel<<<grid_for(rows * length), 256, 0, cur_stream()>>>(
      x, y, rows, length, offset, stride);
  return cudaGetLastError() == cudaSuccess;
}

// --- batched prefill entry points -------------------------------------------

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
  SB_NOTE_LAUNCH(rope_neox_heads_batched_kernel, grid_for(n), 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(rope_neox_heads_batched_pos_kernel, grid_for(n), 256, 0, cur_stream());
  rope_neox_heads_batched_pos_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(
      x, n, head_dim, heads_per_pos, start_pos, freq_scale, log_base, d_ff);
  return cudaGetLastError() == cudaSuccess;
}

// cuBLAS path for the batched dense projection (Phase C Step 3). On by default;
// SONICBOOM_DENSE_CUBLAS=0 restores the per-(row,col) gemv for A/B measurement.
bool dense_cublas_enabled() noexcept {
  if (std::getenv("SONICBOOM_DENSE_CUBLAS") == nullptr) return true;
  return std::string_view(std::getenv("SONICBOOM_DENSE_CUBLAS")) != "0";
}

// SONICBOOM_DENSE_LT=0 restores the cublasGemmEx default algorithm for the
// strided bf16 dense projection (the Stage 4b A/B switch).
bool dense_lt_enabled() noexcept {
  if (std::getenv("SONICBOOM_DENSE_LT") == nullptr) return true;
  return std::string_view(std::getenv("SONICBOOM_DENSE_LT")) != "0";
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

// The workspace block behind that budget: shared by the cuBLAS handle and the
// cuBLASLt dense GEMM.
void *g_dense_ws = nullptr;

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
    if (cudaMalloc(&g_dense_ws, kDenseCublasWorkspaceBytes) == cudaSuccess)
      cublasSetWorkspace(h, g_dense_ws, kDenseCublasWorkspaceBytes);
  }
  return h;
}

// cuBLASLt bf16 tensor-core GEMM for the strided dense projection (Stage 4b):
// C^T (rows x 128) = W (rows x cols) * X^T (cols x 128), fp32 accumulate. The
// heuristic runs once per (rows, cols) shape class and the chosen algorithm is
// cached: the same shape always runs the same algorithm, so a position's
// numerics do not depend on the batch size or column offset (batch invariance),
// and no heuristic runs inside CUDA graph capture.
struct DenseLtKernel {
  cublasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
};

cublasLtHandle_t dense_lt_handle() {
  static cublasLtHandle_t h = nullptr;
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!h && cublasLtCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
  return h;
}

// The cached algorithm for (rows, cols), chosen on first use. Returns false when
// cuBLASLt has no candidate for the shape (the caller falls back to GemmEx).
bool dense_lt_kernel(cublasLtHandle_t lt, cublasLtMatmulDesc_t op,
                     cublasLtMatrixLayout_t ad, cublasLtMatrixLayout_t bd,
                     cublasLtMatrixLayout_t cd, int rows, int cols,
                     DenseLtKernel &out) {
  static std::mutex mu;
  static std::map<std::pair<int, int>, std::optional<DenseLtKernel>> cache;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find({rows, cols});
  if (it != cache.end()) {
    if (!it->second) return false;
    out = *it->second;
    return true;
  }
  cublasLtMatmulPreference_t pref = nullptr;
  cublasLtMatmulHeuristicResult_t heur{};
  int found = 0;
  std::optional<DenseLtKernel> picked;
  if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
    size_t max_ws = kDenseCublasWorkspaceBytes;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, ad, bd, cd, cd, pref, 1, &heur,
                                       &found) == CUBLAS_STATUS_SUCCESS &&
        found > 0) {
      DenseLtKernel k;
      k.algo = heur.algo;
      k.workspace = g_dense_ws ? std::min(heur.workspaceSize,
                                          kDenseCublasWorkspaceBytes)
                               : 0;
      picked = k;
      if (std::getenv("SONICBOOM_DENSE_LT_DEBUG"))
        std::fprintf(stderr,
                     "[dense-lt] rows=%d cols=%d workspace=%zu selected\n", rows,
                     cols, k.workspace);
    }
    cublasLtMatmulPreferenceDestroy(pref);
  }
  cache.insert({{rows, cols}, picked});
  if (!picked) return false;
  out = *picked;
  return true;
}

// A is the position-major activation block (read as a column-major k x 128
// matrix), W the row-major weight (a column-major k x m matrix under TRANSA).
bool matvec_bf16_lt_dev(cublasLtHandle_t lt, const uint16_t *d_w, int cols,
                        const uint16_t *a, float *c, int rows) {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
  bool ok = false;
  if (cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  const cublasOperation_t tr_a = CUBLAS_OP_T, tr_b = CUBLAS_OP_N;
  const cudaDataType_t bf16 = CUDA_R_16BF, f32 = CUDA_R_32F;
  if (cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tr_a,
                                     sizeof(tr_a)) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tr_b,
                                     sizeof(tr_b)) == CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&ad, bf16, cols, rows, cols) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&bd, bf16, cols, kDenseCublasWidth, cols) ==
          CUBLAS_STATUS_SUCCESS &&
      cublasLtMatrixLayoutCreate(&cd, f32, rows, kDenseCublasWidth, rows) ==
          CUBLAS_STATUS_SUCCESS) {
    DenseLtKernel k;
    if (dense_lt_kernel(lt, op, ad, bd, cd, rows, cols, k)) {
      const float alpha = 1.0f, beta = 0.0f;
      ok = cublasLtMatmul(lt, op, &alpha, d_w, ad, a, bd, &beta, c, cd, c, cd,
                          &k.algo, g_dense_ws, k.workspace, cur_stream()) ==
           CUBLAS_STATUS_SUCCESS;
    }
  }
  if (ad) cublasLtMatrixLayoutDestroy(ad);
  if (bd) cublasLtMatrixLayoutDestroy(bd);
  if (cd) cublasLtMatrixLayoutDestroy(cd);
  cublasLtMatmulDescDestroy(op);
  return ok;
}

// Blocked cuBLAS matvec for the dense f32 projection. x is [cols, ncols], y is
// [rows, ncols] (ncols innermost); W is [rows, cols] row-major on device. The n
// columns are repacked into fixed-width 128-column blocks so every cuBLAS call
// has shape (m=128, n=rows, k=cols, lda=ldc=128) — the same per-column numerics
// regardless of n or column offset.
bool matvec_f32_cublas_dev(const float* d_w, int cols, const float* x, float* y,
                           int rows, int ncols, bool strided) {
  cublasHandle_t h = dense_cublas_handle();
  if (!h) return false;
  float* xpack = nullptr;
  float* outpack = nullptr;
  auto need_pack = [&] {
    if (xpack) return true;
    xpack = static_cast<float*>(grow_scratch(
        g_xpack, g_xpack_bytes, (size_t)cols * kDenseCublasWidth * sizeof(float)));
    outpack = static_cast<float*>(grow_scratch(
        g_outpack, g_outpack_bytes,
        (size_t)rows * kDenseCublasWidth * sizeof(float)));
    return xpack && outpack;
  };

  cublasSetStream(h, cur_stream());
  const float alpha = 1.0f, beta = 0.0f;
  for (int c0 = 0; c0 < ncols; c0 += kDenseCublasWidth) {
    const int chunk_n = std::min(kDenseCublasWidth, ncols - c0);
    cublasStatus_t st;
    if (strided) {
      // Read the producer's position-major block directly and write the
      // consumer's position-major result directly. In column-major terms both
      // operands are already stored in the layout cuBLAS wants:
      //   C^T (rows x 128) = W (rows x cols) * X^T (cols x 128)
      // with C^T row-major == C column-major (m=rows, n=128, ldc=rows), which
      // is exactly y[c0*rows ...]; W row-major (lda=cols) is the OP_T operand;
      // X block row-major (cols x 128 with ldb=cols) is the OP_N operand.
      // Tail blocks run the same shape over a zero-padded copy: the per-position
      // numerics must not depend on where a segmentation splits.
      const bool full = chunk_n == kDenseCublasWidth;
      const float* a = x + (size_t)c0 * cols;
      float* c = y + (size_t)c0 * rows;
      if (!full) {
        if (!need_pack()) return false;
        SB_NOTE_LAUNCH(pad_f32_tail_kernel, grid_for(cols * kDenseCublasWidth), 256, 0, cur_stream());
        pad_f32_tail_kernel<<<grid_for(cols * kDenseCublasWidth), 256, 0,
                              cur_stream()>>>(x, xpack, cols, c0, chunk_n);
        a = xpack;
        c = outpack;
      }
      st = cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, kDenseCublasWidth,
                       cols, &alpha, d_w, cols, a, cols, &beta, c, rows);
      if (st == CUBLAS_STATUS_SUCCESS && !full) {
        SB_NOTE_LAUNCH(copyback_tail_kernel, grid_for(rows * chunk_n), 256, 0, cur_stream());
        copyback_tail_kernel<<<grid_for(rows * chunk_n), 256, 0, cur_stream()>>>(
            outpack, y, rows, c0, chunk_n);
      }
    } else {
      if (!need_pack()) return false;
      SB_NOTE_LAUNCH(pack_f32_chunk_kernel, grid_for(cols * kDenseCublasWidth), 256, 0, cur_stream());
      pack_f32_chunk_kernel<<<grid_for(cols * kDenseCublasWidth), 256, 0,
                              cur_stream()>>>(x, xpack, cols, ncols, c0, chunk_n);
      st = cublasSgemm(
          h, CUBLAS_OP_N, CUBLAS_OP_N, kDenseCublasWidth, rows, cols, &alpha, xpack,
          kDenseCublasWidth, d_w, cols, &beta, outpack, kDenseCublasWidth);
      if (st == CUBLAS_STATUS_SUCCESS) {
        SB_NOTE_LAUNCH(copyback_chunk_kernel, grid_for(rows * chunk_n), 256, 0, cur_stream());
        copyback_chunk_kernel<<<grid_for(rows * chunk_n), 256, 0, cur_stream()>>>(
            outpack, y, rows, ncols, c0, chunk_n);
      }
    }
    if (st != CUBLAS_STATUS_SUCCESS) return false;
  }
  if (alpha != 1.0f) {
    SB_NOTE_LAUNCH(scale_kernel, grid_for(rows * ncols), 256, 0, cur_stream());
    scale_kernel<<<grid_for(rows * ncols), 256, 0, cur_stream()>>>(
        y, alpha, y, rows * ncols);
  }
  return cudaGetLastError() == cudaSuccess;
}

// --- Step 10: generated dense GEMM selection ---------------------------------
//
// The generated GEMM (compiler/GemmCodegen.h, launched through
// planner/glue_jit.h) computes the same C^T = W X^T as the cuBLASLt path in one
// launch, with no column chunking: a token's row depends only on its own bf16
// activation and W, so batch invariance holds by construction. The choice
// between it and cuBLASLt is per (rows, cols) shape class — the model's
// projection dimensions, never the batch size — decided by timing both on
// synthetic zero data at a canonical 256-token chunk and cached in the autotune
// cache (op "dense-gemm"). SONICBOOM_DENSE_GEMM=0 forces the cuBLASLt path
// (the ablation arm).
bool dense_gemm_enabled() {
  static const bool enabled = [] {
    const char *e = std::getenv("SONICBOOM_DENSE_GEMM");
    return !(e != nullptr && std::string_view(e) == "0");
  }();
  return enabled;
}

struct DenseChoice {
  bool codegen = false;
  compiler::DenseGemmShape shape;
};

// Candidate tiles: block shape and warp count only, so the choice is a property
// of (shape class, GPU). All are covered by dense_gemm_shape_supported().
// One tile per K-depth: after the shared-stage padding, m64 n64 k32 and k64
// measure within noise of each other, and two candidates that close let the
// recorded tile flap between runs; k64 (half the loop trips) is the listed one.
const compiler::DenseGemmShape kDenseGemmCandidates[] = {
    {64, 64, 64},   {32, 64, 64},   {64, 32, 64},
    {128, 32, 32},  {32, 128, 32},  {128, 64, 32}, {128, 64, 64},
    {128, 128, 32}, {128, 128, 64}, {64, 128, 32}};

// The tuning chunk is the canonical prefill segment: the generated kernel is
// only on this (prefill, batched) path, and the choice it produces is applied
// to every batch size, so the measurement must represent the largest one.
constexpr int kDenseTuneTokens = 512;
// A candidate must beat cuBLASLt by this fraction to replace it (the same noise
// margin the attention tuner uses).
constexpr double kDenseSwitchMargin = 0.03;
// A later candidate must also beat the current best tile by this fraction to
// take its place. Candidates that measure within noise of each other then
// resolve to the earlier (list order) one on every run: the plan record
// contains the chosen tile, and a record made without a tune cache has to land
// on the same one as a record made with it.
constexpr double kDenseCandidateMargin = 0.03;

bool dense_capturing() {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(cur_stream(), &st) != cudaSuccess ||
         st != cudaStreamCaptureStatusNone;
}

std::string dense_choice_name(const DenseChoice &c) {
  if (!c.codegen) return "lt";
  return "gemm-m" + std::to_string(c.shape.bm) + "n" +
         std::to_string(c.shape.bn) + "k" + std::to_string(c.shape.bk);
}

bool dense_choice_from_values(const std::map<std::string, int> &v,
                              DenseChoice &out) {
  const auto impl = v.find("impl");
  if (impl == v.end()) return false;
  if (impl->second == 0) {
    out = {};
    return true;
  }
  int bm = 0, bn = 0, bk = 0;
  const auto get = [&v](const char *n, int &dst) {
    const auto i = v.find(n);
    if (i == v.end()) return false;
    dst = i->second;
    return true;
  };
  if (!get("bm", bm) || !get("bn", bn) || !get("bk", bk)) return false;
  const compiler::DenseGemmShape s{bm, bn, bk};
  if (!compiler::dense_gemm_shape_supported(s)) return false;
  out = {true, s};
  return true;
}

// The strided bf16 dense projection: the fixed 128-column cuBLASLt blocking
// over an already-bf16 position-major activation. Split from
// matvec_bf16_cublas_dev so the dense-GEMM autotuner times this exact loop
// against the generated kernel (both pay the same bf16 cast). The activation
// block is position-major (ldb=cols); tail blocks run the same shape over a
// zero-padded copy so a position's numerics do not depend on where a
// segmentation splits.
bool dense_lt_bf16_strided(const uint16_t* xb, const uint16_t* d_w, float* y,
                           int rows, int cols, int ncols) {
  cublasHandle_t h = dense_cublas_handle();
  if (!h) return false;
  cublasLtHandle_t lt = dense_lt_handle();
  uint16_t* xpack = nullptr;
  float* outpack = nullptr;
  auto need_pack = [&] {
    if (xpack) return true;
    xpack = static_cast<uint16_t*>(grow_scratch(
        g_xpack, g_xpack_bytes,
        (size_t)cols * kDenseCublasWidth * sizeof(uint16_t)));
    outpack = static_cast<float*>(grow_scratch(
        g_outpack, g_outpack_bytes,
        (size_t)rows * kDenseCublasWidth * sizeof(float)));
    return xpack && outpack;
  };

  cublasSetStream(h, cur_stream());
  const float alpha = 1.0f, beta = 0.0f;
  for (int c0 = 0; c0 < ncols; c0 += kDenseCublasWidth) {
    const int chunk_n = std::min(kDenseCublasWidth, ncols - c0);
    // Same transposed formulation as the f32 path; the activation block is
    // already bf16 in position-major order (ldb=cols).
    const bool full = chunk_n == kDenseCublasWidth;
    const uint16_t* a = xb + (size_t)c0 * cols;
    float* c = y + (size_t)c0 * rows;
    if (!full) {
      if (!need_pack()) return false;
      SB_NOTE_LAUNCH(pad_bf16_tail_kernel, grid_for(cols * kDenseCublasWidth), 256, 0, cur_stream());
      pad_bf16_tail_kernel<<<grid_for(cols * kDenseCublasWidth), 256, 0,
                             cur_stream()>>>(xb, xpack, cols, c0, chunk_n);
      a = xpack;
      c = outpack;
    }
    // cuBLASLt bf16 tensor-core GEMM with the shape class's cached algorithm;
    // GemmEx stays as the fallback when Lt has no candidate.
    cublasStatus_t st =
        (lt && dense_lt_enabled() && matvec_bf16_lt_dev(lt, d_w, cols, a, c, rows))
            ? CUBLAS_STATUS_SUCCESS
            : cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, rows, kDenseCublasWidth,
                           cols, &alpha, d_w, CUDA_R_16BF, cols, a, CUDA_R_16BF,
                           cols, &beta, c, CUDA_R_32F, rows, CUBLAS_COMPUTE_32F,
                           CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS && !full) {
      SB_NOTE_LAUNCH(copyback_tail_kernel, grid_for(rows * chunk_n), 256, 0, cur_stream());
      copyback_tail_kernel<<<grid_for(rows * chunk_n), 256, 0, cur_stream()>>>(
          outpack, y, rows, c0, chunk_n);
    }
    if (st != CUBLAS_STATUS_SUCCESS) return false;
  }
  if (alpha != 1.0f) {
    SB_NOTE_LAUNCH(scale_kernel, grid_for(rows * ncols), 256, 0, cur_stream());
    scale_kernel<<<grid_for(rows * ncols), 256, 0, cur_stream()>>>(
        y, alpha, y, rows * ncols);
  }
  return cudaGetLastError() == cudaSuccess;
}

// Time cuBLASLt and every candidate tile on zeroed synthetic operands. Both
// arms consume the same bf16 activation, so the measurement isolates the GEMM.
DenseChoice dense_tune(int rows, int cols, double &tune_s,
                       int &candidate_count) {
  DenseChoice best;
  candidate_count = 0;
  const size_t w_n = size_t(rows) * size_t(cols);
  const size_t x_n = size_t(kDenseTuneTokens) * size_t(cols);
  const size_t y_n = size_t(kDenseTuneTokens) * size_t(rows);
  uint16_t *w = nullptr, *x = nullptr;
  float *y = nullptr;
  if (cudaMalloc(&w, w_n * 2) != cudaSuccess) return best;
  if (cudaMalloc(&x, x_n * 2) != cudaSuccess) {
    cudaFree(w);
    return best;
  }
  if (cudaMalloc(&y, y_n * 4) != cudaSuccess) {
    cudaFree(x);
    cudaFree(w);
    return best;
  }
  // Zeroed inputs: the work is data-independent and denormals stay out of the
  // timing.
  cudaMemsetAsync(w, 0, w_n * 2, cur_stream());
  cudaMemsetAsync(x, 0, x_n * 2, cur_stream());
  cudaMemsetAsync(y, 0, y_n * 4, cur_stream());

  const auto t0 = std::chrono::steady_clock::now();
  const double lt_ms = ::sonicboom::tune::median_launch_ms(
      [&] {
        return dense_lt_bf16_strided(x, w, y, rows, cols, kDenseTuneTokens);
      },
      cur_stream(), 2, 5);
  double best_ms = 0.0;  // best candidate so far; cuBLASLt is compared below
  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  if (verbose)
    std::fprintf(stderr, "[autotune] dense-gemm m=%d k=%d n=%d lt=%.4f ms\n",
                 rows, cols, kDenseTuneTokens, lt_ms);
  for (const compiler::DenseGemmShape &s : kDenseGemmCandidates) {
    if (!compiler::dense_gemm_shape_supported(s)) continue;
    ++candidate_count;
    const double ms = ::sonicboom::tune::median_launch_ms(
        [&] {
          return planner::glue::launch_dense_gemm(s, x, w, y, rows, cols,
                                                  kDenseTuneTokens);
        },
        cur_stream(), 2, 5);
    if (verbose)
      std::fprintf(stderr, "[autotune]   gemm m%d n%d k%d = %.4f ms\n", s.bm,
                   s.bn, s.bk, ms);
    if (ms > 0.0 &&
        (best_ms <= 0.0 || ms < best_ms * (1.0 - kDenseCandidateMargin))) {
      best = {true, s};
      best_ms = ms;
    }
  }
  // Keep cuBLASLt unless a generated tile wins by the noise margin.
  if (best.codegen &&
      !(lt_ms > 0.0 && best_ms < lt_ms * (1.0 - kDenseSwitchMargin)))
    best = {};
  tune_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  cudaFree(y);
  cudaFree(x);
  cudaFree(w);
  return best;
}

// Resolve the dense projection implementation for a (rows, cols) shape class:
// cached choice, else a tuning run, else cuBLASLt. The key never includes the
// batch size or position.
DenseChoice dense_resolve(int rows, int cols) {
  const DenseChoice def;
  if (!::sonicboom::tune::enabled()) return def;
  const std::string key = compiler::dense_gemm_codegen_version() + "|m" +
                          std::to_string(rows) + "k" + std::to_string(cols);
  static std::mutex mu;
  static std::unordered_map<std::string, DenseChoice> memo;
  {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = memo.find(key);
    if (it != memo.end()) return it->second;
  }

  DenseChoice choice = def;
  bool have = false;
  const std::map<std::string, int> cached =
      ::sonicboom::tune::lookup("dense-gemm", key);
  if (!cached.empty() && dense_choice_from_values(cached, choice)) have = true;

  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  if (!have) {
    if (dense_capturing()) return def;  // host timing is illegal in a capture
    // The timing launches are not part of any run: the first execution of a
    // dense step may be the one being recorded, and a recorded plan must name
    // the projection's real grid, not the tuning pass's synthetic one.
    planner::PlanRecorder *saved = planner::recorder();
    planner::set_thread_recorder(nullptr);
    double tune_s = 0.0;
    int candidate_count = 0;
    choice = dense_tune(rows, cols, tune_s, candidate_count);
    planner::set_thread_recorder(saved);
    std::map<std::string, int> values{{"impl", choice.codegen ? 1 : 0}};
    if (choice.codegen) {
      values["bm"] = choice.shape.bm;
      values["bn"] = choice.shape.bn;
      values["bk"] = choice.shape.bk;
    }
    ::sonicboom::tune::store("dense-gemm", key, values);
    if (verbose)
      std::fprintf(stderr,
                   "[autotune] dense-gemm %s default=lt chosen=%s "
                   "candidates=%d tune=%.2f s\n",
                   key.c_str(), dense_choice_name(choice).c_str(),
                   candidate_count, tune_s);
  } else if (verbose) {
    std::fprintf(stderr, "[autotune] dense-gemm %s cached=%s\n", key.c_str(),
                 dense_choice_name(choice).c_str());
  }

  {
    std::lock_guard<std::mutex> lk(mu);
    memo[key] = choice;
  }
  return choice;
}

bool matvec_bf16_cublas_dev(const uint16_t* d_w, int cols, const float* x,
                            float* y, int rows, int ncols, bool strided,
                            float alpha) {
  cublasHandle_t h = dense_cublas_handle();
  if (!h) return false;
  uint16_t* xpack = nullptr;
  float* outpack = nullptr;
  auto need_pack = [&] {
    if (xpack) return true;
    xpack = static_cast<uint16_t*>(grow_scratch(
        g_xpack, g_xpack_bytes,
        (size_t)cols * kDenseCublasWidth * sizeof(uint16_t)));
    outpack = static_cast<float*>(grow_scratch(
        g_outpack, g_outpack_bytes,
        (size_t)rows * kDenseCublasWidth * sizeof(float)));
    return xpack && outpack;
  };

  // The strided path consumes a position-major bf16 activation: one contiguous
  // rounding pass, no transposing repack for full blocks. Its own scratch, so
  // the tail padding cannot alias it. Both the generated GEMM and the cuBLASLt
  // loop read that same bf16 staging, so this cast is common to both arms.
  if (strided) {
    uint16_t* xb = static_cast<uint16_t*>(grow_scratch(
        g_xb, g_xb_bytes, (size_t)cols * ncols * sizeof(uint16_t)));
    if (!xb) return false;
    SB_NOTE_LAUNCH(cast_bf16_rows_kernel, grid_for(cols * ncols), 256, 0, cur_stream());
    cast_bf16_rows_kernel<<<grid_for(cols * ncols), 256, 0, cur_stream()>>>(
        x, xb, cols * ncols);
    // One launch per projection when the shape class selected the generated
    // kernel (no 128-column chunks, per-token numerics independent of N). A
    // failed launch falls through to the cuBLASLt path.
    if (dense_gemm_enabled()) {
      const DenseChoice choice = dense_resolve(rows, cols);
      if (choice.codegen &&
          planner::glue::launch_dense_gemm(choice.shape, xb, d_w, y, rows, cols,
                                           ncols, alpha))
        return true;
    }
    if (!dense_lt_bf16_strided(xb, d_w, y, rows, cols, ncols)) return false;
    // cuBLASLt has no epilogue here; the folded scale runs as its own pass, so
    // the ablation arm produces the same output as the generated kernel's.
    if (alpha != 1.0f) {
      SB_NOTE_LAUNCH(scale_kernel, grid_for(rows * ncols), 256, 0, cur_stream());
      scale_kernel<<<grid_for(rows * ncols), 256, 0, cur_stream()>>>(
          y, alpha, y, rows * ncols);
    }
    return cudaGetLastError() == cudaSuccess;
  }

  // The non-strided path transposes each 128-column block into the cuBLAS
  // layout first.
  cublasSetStream(h, cur_stream());
  const float one = 1.0f, zero = 0.0f;
  for (int c0 = 0; c0 < ncols; c0 += kDenseCublasWidth) {
    const int chunk_n = std::min(kDenseCublasWidth, ncols - c0);
    if (!need_pack()) return false;
    SB_NOTE_LAUNCH(pack_bf16_chunk_kernel, grid_for(cols * kDenseCublasWidth), 256, 0, cur_stream());
    pack_bf16_chunk_kernel<<<grid_for(cols * kDenseCublasWidth), 256, 0,
                             cur_stream()>>>(x, xpack, cols, ncols, c0, chunk_n);
    cublasStatus_t st = cublasGemmEx(
        h, CUBLAS_OP_N, CUBLAS_OP_N, kDenseCublasWidth, rows, cols, &one, xpack,
        CUDA_R_16BF, kDenseCublasWidth, d_w, CUDA_R_16BF, cols, &zero, outpack,
        CUDA_R_32F, kDenseCublasWidth, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) {
      SB_NOTE_LAUNCH(copyback_chunk_kernel, grid_for(rows * chunk_n), 256, 0, cur_stream());
      copyback_chunk_kernel<<<grid_for(rows * chunk_n), 256, 0, cur_stream()>>>(
          outpack, y, rows, ncols, c0, chunk_n);
    }
    if (st != CUBLAS_STATUS_SUCCESS) return false;
  }
  if (alpha != 1.0f) {
    SB_NOTE_LAUNCH(scale_kernel, grid_for(rows * ncols), 256, 0, cur_stream());
    scale_kernel<<<grid_for(rows * ncols), 256, 0, cur_stream()>>>(
        y, alpha, y, rows * ncols);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_f32_batched_dev(std::span<const float> W, int cols, const float* x,
                            float* y, int rows, int ncols, bool strided) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || ncols <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const float* d_w = static_cast<const float*>(
      upload_bytes(W.data(), (size_t)cols * rows * sizeof(float)));
  if (!d_w) return false;
  if (dense_cublas_enabled() && ncols >= dense_cublas_min_n() &&
      matvec_f32_cublas_dev(d_w, cols, x, y, rows, ncols, strided))
    return true;
  // The gemv reads the same position-major activation in either layout.
  const int total = rows * ncols;
  SB_NOTE_LAUNCH(gemv_dense_f32_batched, grid_for(total), 256, 0, cur_stream());
  gemv_dense_f32_batched<<<grid_for(total), 256, 0, cur_stream()>>>(
      d_w, x, y, cols, rows, ncols);
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_bf16_batched_dev(std::span<const uint16_t> W, int cols, const float* x,
                             float* y, int rows, int ncols, bool strided,
                             float alpha) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || ncols <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const uint16_t* d_w = static_cast<const uint16_t*>(
      upload_bytes(W.data(), (size_t)cols * rows * sizeof(uint16_t)));
  if (!d_w) return false;
  if (dense_cublas_enabled() && ncols >= dense_cublas_min_n() &&
      matvec_bf16_cublas_dev(d_w, cols, x, y, rows, ncols, strided, alpha))
    return true;
  // The gemv fallbacks below have no epilogue: the folded scale is a pass.
  const auto scale_if = [&] {
    if (alpha == 1.0f) return;
    SB_NOTE_LAUNCH(scale_kernel, grid_for(rows * ncols), 256, 0, cur_stream());
    scale_kernel<<<grid_for(rows * ncols), 256, 0, cur_stream()>>>(
        y, alpha, y, rows * ncols);
  };
  if (strided) {
    // The caller skipped its bf16 staging copy: round position-major here so
    // the fallback consumes exactly the same bf16-rounded activation.
    uint16_t* xb = static_cast<uint16_t*>(grow_scratch(
        g_xpack, g_xpack_bytes, (size_t)cols * ncols * sizeof(uint16_t)));
    if (!xb) return false;
    SB_NOTE_LAUNCH(cast_bf16_rows_kernel, grid_for(cols * ncols), 256, 0, cur_stream());
    cast_bf16_rows_kernel<<<grid_for(cols * ncols), 256, 0, cur_stream()>>>(
        x, xb, cols * ncols);
    const int total = rows * ncols;
    SB_NOTE_LAUNCH(gemv_bf16_batched_xb, grid_for(total), 256, 0, cur_stream());
    gemv_bf16_batched_xb<<<grid_for(total), 256, 0, cur_stream()>>>(
        d_w, xb, y, cols, rows, ncols);
    scale_if();
    return cudaGetLastError() == cudaSuccess;
  }
  const int total = rows * ncols;
  SB_NOTE_LAUNCH(gemv_bf16_batched, grid_for(total), 256, 0, cur_stream());
  gemv_bf16_batched<<<grid_for(total), 256, 0, cur_stream()>>>(
      d_w, x, y, cols, rows, ncols);
  scale_if();
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
  SB_NOTE_LAUNCH(kv_append_batched_kernel, grid, 256, 0, cur_stream());
  kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_k, src_k, start_pos, n_slots, row_elems, n);
  SB_NOTE_LAUNCH(kv_append_batched_kernel, grid, 256, 0, cur_stream());
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
  SB_NOTE_LAUNCH(kv_append_batched_pos_kernel, grid, 256, 0, cur_stream());
  kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_k, src_k, start_pos, n_slots, row_elems, n);
  SB_NOTE_LAUNCH(kv_append_batched_pos_kernel, grid, 256, 0, cur_stream());
  kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
      dst_v, src_v, start_pos, n_slots, row_elems, n);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace sonicboom::nn::cuda
