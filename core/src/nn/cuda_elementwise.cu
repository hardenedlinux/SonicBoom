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
#include "../planner/decision_registry.h"  // attribute-keyed attention classes
#include "../planner/glue_jit.h"  // generated attention kernel (Stage 3)
#include "../planner/mlir_autotune.h"  // Q3: prefill attention shape classes
#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)
#include "../planner/reduce_mode.h"  // ReduceMode selection (f64 / df32 / f32)

#include <cuda_fp16.h>
#include <type_traits>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <algorithm>
#include <atomic>
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
#include <unordered_set>
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

// Final-logit softcap, y = cap * tanh(x / cap), the device form of nn::softcap.
// One multiply by the reciprocal, as on the host.
__global__ void softcap_kernel(const float* __restrict__ x, float cap,
                               float* __restrict__ y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = cap * tanhf(x[i] * (1.0f / cap));
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
template <typename KV>
__global__ void decode_attention_split_kernel(const float* __restrict__ q,
                                              const KV* __restrict__ k_cache,
                                              const KV* __restrict__ v_cache,
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
  // Chunk indices are absolute: chunk k covers key positions [k*C, (k+1)*C),
  // so a key always lands in the same chunk whatever the window length, the
  // cache capacity or how the context was built. Blocks are launched from the
  // window's first chunk (block 0) in ascending absolute index; a block whose
  // chunk holds no valid key exits without writing, and the merge stops at the
  // last active block.
  const uint64_t first = start / chunk;
  const uint64_t c0 = (first + chunk_idx) * chunk;
  if (c0 > p || c0 + chunk <= start) return;
  const uint64_t jlo = c0 > start ? c0 : start;
  const uint64_t jhi = (c0 + chunk < p + 1) ? (c0 + chunk) : (p + 1);
  const uint64_t cn = jhi - jlo;

  // Slot of the chunk's first key: one 64-bit modulo per block (device division
  // is a software routine, and doing it per key was a large part of this loop's
  // cost), then a compare-and-reset step so every later key's slot is a single
  // increment. Identical values to (start + c0 + j) % n_slots.
  uint64_t slot = jlo % n_slots;

  // Phase 1: one q·k dot per key, warp-cooperative. Unrolled by 4 so several
  // keys' K loads are in flight at once — the per-key warp reduction is a
  // serialized shuffle chain, so a single-key loop pays a full L2/DRAM latency
  // per key.
  #pragma unroll 4
  for (uint64_t j = 0; j < cn; ++j) {
    const KV* krow =
        kv_row(k_cache, slot * (uint64_t)n_heads_kv + kv_h, head_dim);
    float d = 0.0f;
    for (int i = lane; i < head_dim; i += 32)
      d += qrow[i] * kv_load_at(krow, i, head_dim);
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
  slot = jlo % n_slots;
  #pragma unroll 4
  for (uint64_t j = 0; j < cn; ++j) {
    const KV* vrow =
        kv_row(v_cache, slot * (uint64_t)n_heads_kv + kv_h, head_dim);
    const float p = srow[j];  // exp(score - local max)
#pragma unroll
    for (int i = 0; i < 16; ++i)
      if (i < dims_per_lane)
        acc[i] += p * kv_load_at(vrow, lane + i * 32, head_dim);
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

  // Active chunk count from the device-resident position. Block bx covers the
  // absolute chunk start/chunk + bx, so the last block holding a valid key is
  // p/chunk - start/chunk; blocks after it exited unwritten and must not be
  // read. n_keys >= 1 always, so n_active >= 1.
  const uint64_t p = *pos;
  const uint64_t start =
      (sliding_window > 0 && p >= sliding_window) ? p - sliding_window + 1 : 0;
  const uint64_t n_active = p / chunk - start / chunk + 1;
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
template <typename KV>
__global__ void kv_append_kernel(KV* __restrict__ dst,
                                 const float* __restrict__ src,
                                 const uint64_t* __restrict__ pos,
                                 uint64_t n_slots, int row_elems) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= row_elems) return;
  const uint64_t off = (*pos % n_slots) * (uint64_t)row_elems;
  kv_store(dst + off + i, src[i]);
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
template <typename KV>
__global__ void flash_attention_chunk_kernel(
    const float* __restrict__ q, const float* __restrict__ k_batch,
    const float* __restrict__ v_batch, const KV* __restrict__ k_cache,
    const KV* __restrict__ v_cache, uint64_t start, uint64_t n_slots,
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
      float dot = 0.0f;
      if constexpr (std::is_same_v<KV, float>) {
        // Strict (f32) path: textually the pre-profile kernel, so the fp32
        // result stays bit-identical (restructuring changes FP contraction).
        const KV* krow;
        if (j < start) {
          const uint64_t slot = j % n_slots;
          krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
        } else {
          const uint64_t bi = j - start;
          krow = k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        }
        for (int d = lane; d < head_dim; d += 32) dot += qrow[d] * krow[d];
      } else if (j < start) {
        const uint64_t slot = j % n_slots;
        const KV* krow =
            kv_row(k_cache, slot * (uint64_t)n_heads_kv + kv_h, head_dim);
        for (int d = lane; d < head_dim; d += 32)
          dot += qrow[d] * kv_load_at(krow, d, head_dim);
      } else {
        const uint64_t bi = j - start;
        const float* krow =
            k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        for (int d = lane; d < head_dim; d += 32)
          dot += qrow[d] * kv_load_batch<KV>(krow + d);
      }
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
template <typename KV>
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
        for (int d = lane; d < head_dim; d += 32) dot += qrow[d] * krow[d];
      } else if (j < start) {
        const uint64_t slot = j % n_slots;
        const KV* krow =
            kv_row(k_cache, slot * (uint64_t)n_heads_kv + kv_h, head_dim);
        for (int d = lane; d < head_dim; d += 32)
          dot += qrow[d] * kv_load_at(krow, d, head_dim);
      } else {
        const uint64_t bi = j - start;
        const float* krow =
            k_batch + (bi * (uint64_t)n_heads_kv + kv_h) * head_dim;
        for (int d = lane; d < head_dim; d += 32)
          dot += qrow[d] * kv_load_batch<KV>(krow + d);
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
      const uint64_t slot = j % n_slots;
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

// CAUSAL = true is the LLM prefill contract: a query at absolute position p
// attends to keys in [max(0, p - window + 1), p]. CAUSAL = false is dense full
// attention (no mask, no window): every query attends to all keys in
// [start, start + n), which is the image-token block of a DiT. The key range
// and the mask are compile-time, so the full path carries no mask arithmetic.
template <typename KV, int HEAD_DIM, int BK, int NW, int QB, int GQA,
          bool CAUSAL = true>
__global__ void __launch_bounds__(NW * GQA * (QB / 16) * 32)
    flash_attention_tiled_kernel(
        const float* __restrict__ q, const float* __restrict__ k_batch,
        const float* __restrict__ v_batch, const KV* __restrict__ k_cache,
        const KV* __restrict__ v_cache, const uint64_t* __restrict__ start_ptr,
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

  // The full-attention entry has no KV cache and no absolute position: start
  // is 0 and start_ptr is null, so it must never be dereferenced.
  const uint64_t start = CAUSAL ? *start_ptr : 0;

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
  // Causal: the last key any query in this segment attends is its last query.
  // Full: every query attends to the whole key range [start, start + n).
  const uint64_t key_end = CAUSAL ? abs_pb + (uint64_t)n_blk - 1
                                  : start + (uint64_t)n - 1;
  // key_start: earliest key any query in this segment attends.
  const uint64_t key_start =
      (CAUSAL && sliding_window > 0 && abs_pb >= sliding_window)
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
      // Both sources are fp16-rounded: the cache under f16 storage is already
      // __half, the f32 batch row is converted here.
      __half kv = __float2half_rn(0.0f);
      if (j <= key_end) {
        if (j < start) {
          const uint64_t slot = j % n_slots;
          kv = __float2half_rn(kv_load_at(
              kv_row(k_cache, slot * (uint64_t)n_heads_kv + blockIdx.y,
                     HEAD_DIM),
              d, HEAD_DIM));
        } else {
          const uint64_t bi = j - start;
          kv = __float2half_rn(
              k_batch[(bi * (uint64_t)n_heads_kv + blockIdx.y) * HEAD_DIM + d]);
        }
      }
      k_tile[kk * HEAD_DIM + d] = kv;
    }
    for (int idx = threadIdx.x; idx < BK * HEAD_DIM; idx += blockDim.x) {
      const int kk = idx / HEAD_DIM;
      const int d = idx % HEAD_DIM;
      const uint64_t j = j0 + (uint64_t)kk;
      __half vv = __float2half_rn(0.0f);
      if (j <= key_end) {
        if (j < start) {
          const uint64_t slot = j % n_slots;
          vv = __float2half_rn(kv_load_at(
              kv_row(v_cache, slot * (uint64_t)n_heads_kv + blockIdx.y,
                     HEAD_DIM),
              d, HEAD_DIM));
        } else {
          const uint64_t bi = j - start;
          vv = __float2half_rn(
              v_batch[(bi * (uint64_t)n_heads_kv + blockIdx.y) * HEAD_DIM + d]);
        }
      }
      v_tile[d * V_STRIDE + kk] = vv;
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
        if constexpr (CAUSAL) {
          const uint64_t jb = j0 + (uint64_t)key_off + (uint64_t)t * 2;
          s0[0] = (jb > p0 || jb < low0) ? -1e30f : s0[0] * scale;
          s0[1] = (jb + 1 > p0 || jb + 1 < low0) ? -1e30f : s0[1] * scale;
          s1[0] = (jb + 8 > p0 || jb + 8 < low0) ? -1e30f : s1[0] * scale;
          s1[1] = (jb + 9 > p0 || jb + 9 < low0) ? -1e30f : s1[1] * scale;
          s0[2] = (jb > p1 || jb < low1) ? -1e30f : s0[2] * scale;
          s0[3] = (jb + 1 > p1 || jb + 1 < low1) ? -1e30f : s0[3] * scale;
          s1[2] = (jb + 8 > p1 || jb + 8 < low1) ? -1e30f : s1[2] * scale;
          s1[3] = (jb + 9 > p1 || jb + 9 < low1) ? -1e30f : s1[3] * scale;
        } else {
          // Full attention: every key in [start, start + n) is attended, so the
          // only mask is the end of the key range. The loop steps the keys in
          // BK tiles, so the final tile may be partial: keys past key_end load a
          // zero-filled K row, and their score of 0 would otherwise enter the
          // softmax denominator (inflating l and shrinking the output). In the
          // causal path j <= key_end already implies j <= p, so no such tail
          // exists there.
          const uint64_t jb = j0 + (uint64_t)key_off + (uint64_t)t * 2;
          s0[0] = (jb > key_end) ? -1e30f : s0[0] * scale;
          s0[1] = (jb + 1 > key_end) ? -1e30f : s0[1] * scale;
          s1[0] = (jb + 8 > key_end) ? -1e30f : s1[0] * scale;
          s1[1] = (jb + 9 > key_end) ? -1e30f : s1[1] * scale;
          s0[2] = (jb > key_end) ? -1e30f : s0[2] * scale;
          s0[3] = (jb + 1 > key_end) ? -1e30f : s0[3] * scale;
          s1[2] = (jb + 8 > key_end) ? -1e30f : s1[2] * scale;
          s1[3] = (jb + 9 > key_end) ? -1e30f : s1[3] * scale;
        }

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
void* g_select_keys = nullptr;  // per-token packed sampling keys (vocab unsigned long long, r7)
size_t g_select_keys_bytes = 0;
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

// Grow a scratch that a captured CUDA graph may embed (the sampling select
// kernel's key buffer): the old allocation is retained instead of freed, so a
// graph captured against it keeps a valid pointer. Growth happens before the
// capture that uses the new buffer.
void* grow_scratch_keep(void*& buf, size_t& have, size_t need) {
  if (have >= need) return buf;
  void* nb = nullptr;
  if (cudaMalloc(&nb, need) != cudaSuccess) return nullptr;
  buf = nb;
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

// Constrained decoding's mask test: bit i of the bitmap at mask[i >> 3], least
// significant bit first. The host guarantees the mask covers the whole row, so
// there is no bounds test here.
__device__ __forceinline__ bool mask_bit(const uint8_t* __restrict__ mask, int i) {
  return ((mask[i >> 3] >> (i & 7)) & 1u) != 0;
}

// Masked single-block argmax: the single-block form above restricted to ids the
// grammar mask allows. Disallowed lanes stay -INFINITY, so an all-disallowed row
// reduces to the same -1 output as an all-NaN one.
__global__ void argmax_masked_kernel(const float* __restrict__ x,
                                     const uint8_t* __restrict__ mask,
                                     float* __restrict__ out, int n) {
  __shared__ float s_val[1024];
  __shared__ int s_idx[1024];
  const int tid = threadIdx.x;
  float mv = -INFINITY;
  int mi = -1;
  for (int i = tid; i < n; i += blockDim.x) {
    if (!mask_bit(mask, i)) continue;
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

// --- deterministic sampling candidate selection (r7) ------------------------
// One block: penalties -> mask -> z = l/T -> top kSampleCandidates by the
// packed key (z descending, id ascending) -> write (id, z). The arithmetic is
// the same single-rounding sequence as nn::select_candidates_host, so the
// output is bitwise identical to the host selector.
//
// The window is small (<= kSampleSelectMaxWindow), so its per-token counts are
// built once in a shared lock-free linear-probe hash table; the logits buffer
// stays read-only. Keys are materialized in a global scratch buffer (8 B per
// token) so the eight radix passes only scan keys.

__device__ __forceinline__ float select_adjust_logit(float v, unsigned count,
                                                     float rp, float pres,
                                                     float freq) {
  if (rp != 1.0f) v = v > 0.0f ? __fdiv_rn(v, rp) : fmaf(v, rp, 0.0f);
  const float adjust = fmaf(freq, float(count), pres);
  return fmaf(-1.0f, adjust, v);
}

// Order-preserving float -> uint32 (monotone unsigned), with -0.0 folded onto
// +0.0 so sign-zero ties order by id like the host comparator. The low word is
// 0xFFFFFFFF - id, so a larger key is the better candidate.
__device__ __forceinline__ unsigned long long select_key(float z, int id) {
  unsigned bits = __float_as_uint(z);
  if ((bits << 1) == 0u) bits = 0u;
  const unsigned ord = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
  return (static_cast<unsigned long long>(ord) << 32) |
         (0xFFFFFFFFu - static_cast<unsigned>(id));
}

constexpr int kSelectThreads = 1024;
constexpr int kSelectHashSize = 2048;  // power of two, >= 2 * window cap
constexpr unsigned long long kSelectEmpty = ~0ull;

__device__ __forceinline__ unsigned select_hash(uint32_t token) {
  return (token * 2654435761u) >> 21;  // 11 bits -> [0, 2048)
}

// Insert token into the window hash; the low word becomes its occurrence count.
__device__ __forceinline__ void select_hash_insert(unsigned long long* ht,
                                                   uint32_t token) {
  unsigned h = select_hash(token);
  unsigned long long old = ht[h];
  while (true) {
    const unsigned key = unsigned(old >> 32);
    if (key == 0xFFFFFFFFu) {
      const unsigned long long want =
          (static_cast<unsigned long long>(token) << 32) | 1ull;
      const unsigned long long prev = atomicCAS(&ht[h], old, want);
      if (prev == old) return;
      old = prev;
    } else if (key == token) {
      const unsigned long long prev = atomicCAS(&ht[h], old, old + 1ull);
      if (prev == old) return;
      old = prev;
    } else {
      h = (h + 1u) & (kSelectHashSize - 1);
      old = ht[h];
    }
  }
}

__device__ __forceinline__ int select_hash_count(const unsigned long long* ht,
                                                 uint32_t token) {
  unsigned h = select_hash(token);
  while (true) {
    const unsigned long long e = ht[h];
    const unsigned key = unsigned(e >> 32);
    if (key == 0xFFFFFFFFu) return 0;
    if (key == token) return int(e & 0xFFFFFFFFu);
    h = (h + 1u) & (kSelectHashSize - 1);
  }
}

// z for token i. The window lookup is skipped when the window is empty; neutral
// penalties reproduce v exactly (fmaf(0, c, 0) == 0, fmaf(-1, 0, v) == v).
__device__ __forceinline__ float select_z(const float* __restrict__ logits, int i,
                                          const unsigned long long* ht,
                                          int window_len, float temperature,
                                          float rp, float pres, float freq) {
  float v = logits[i];
  if (window_len > 0) {
    const int c = select_hash_count(ht, uint32_t(i));
    if (c > 0 && (rp != 1.0f || pres != 0.0f || freq != 0.0f))
      v = select_adjust_logit(v, unsigned(c), rp, pres, freq);
  }
  return __fdiv_rn(v, temperature);
}

__global__ void sample_select_kernel(
    const float* __restrict__ logits, int vocab,
    const uint8_t* __restrict__ mask, const int32_t* __restrict__ window,
    const SampleSelectParams* __restrict__ dyn, int window_len,
    float temperature, float repeat_penalty, float presence, float frequency,
    unsigned long long* __restrict__ keys, Candidate* __restrict__ out,
    int* __restrict__ out_count) {
  if (dyn) {
    window_len = dyn->window_len;
    temperature = dyn->temperature;
    repeat_penalty = dyn->repeat_penalty;
    presence = dyn->presence_penalty;
    frequency = dyn->frequency_penalty;
  }
  const int tid = threadIdx.x;

  __shared__ int32_t s_window[kSampleSelectMaxWindow];
  __shared__ unsigned long long s_hash[kSelectHashSize];
  __shared__ unsigned s_hist[256];
  __shared__ unsigned long long s_keys[kSampleCandidates];
  __shared__ float s_z[kSampleCandidates];
  __shared__ int s_id[kSampleCandidates];
  __shared__ unsigned s_total;
  __shared__ int s_fill;
  __shared__ unsigned long long s_prefix;
  __shared__ int s_rank;

  // Window + its (token -> count) hash. Slots outside the vocabulary never
  // adjust a logit, matching the host's in-vocabulary filter.
  for (int i = tid; i < window_len; i += blockDim.x) s_window[i] = window[i];
  for (int i = tid; i < kSelectHashSize; i += blockDim.x) s_hash[i] = kSelectEmpty;
  if (tid == 0) s_total = 0u;
  __syncthreads();
  for (int i = tid; i < window_len; i += blockDim.x) {
    const int t = s_window[i];
    if (t >= 0 && t < vocab) select_hash_insert(s_hash, uint32_t(t));
  }
  __syncthreads();

  // z per token, packed into a key; masked ids carry the sentinel key 0 (a real
  // key always has a non-zero low word or ordinal, and the sentinel is checked
  // explicitly in the passes below).
  for (int i = tid; i < vocab; i += blockDim.x) {
    if (mask && !mask_bit(mask, i)) {
      keys[i] = 0ull;
      continue;
    }
    const float z =
        select_z(logits, i, s_hash, window_len, temperature, repeat_penalty,
                 presence, frequency);
    keys[i] = select_key(z, i);
    atomicAdd(&s_total, 1u);
  }
  __syncthreads();

  const int total = int(s_total);
  if (total == 0) {
    if (tid == 0) *out_count = 0;
    return;
  }

  // Deterministic radix select of the kSampleCandidates-th largest key: 8-bit
  // digits from the top, largest bucket first. Integer counts make each pass
  // order-independent under any thread schedule.
  unsigned long long threshold = 0ull;
  if (total >= kSampleCandidates) {
    unsigned rank = unsigned(kSampleCandidates);
    unsigned long long prefix = 0ull;
    for (int pass = 0; pass < 8; ++pass) {
      const int shift = 56 - 8 * pass;
      for (int d = tid; d < 256; d += blockDim.x) s_hist[d] = 0u;
      __syncthreads();
      for (int i = tid; i < vocab; i += blockDim.x) {
        const unsigned long long k = keys[i];
        if (k == 0ull) continue;
        if (pass > 0 && (k >> (shift + 8)) != (prefix >> (shift + 8)))
          continue;
        atomicAdd(&s_hist[unsigned((k >> shift) & 0xFFull)], 1u);
      }
      __syncthreads();
      if (tid == 0) {
        unsigned acc = 0u;
        int bucket = 0;
        for (int d = 255; d >= 0; --d) {
          const unsigned c = s_hist[d];
          if (acc + c >= rank) {
            bucket = d;
            break;
          }
          acc += c;
        }
        s_prefix = prefix | (static_cast<unsigned long long>(unsigned(bucket))
                             << shift);
        s_rank = int(rank - acc);
      }
      __syncthreads();
      prefix = s_prefix;
      rank = unsigned(s_rank);
    }
    threshold = prefix;
  }

  // Gather keys >= threshold. With at least 256 candidates there are exactly
  // 256 of them (keys are unique: one per token id); otherwise every key.
  // The gather order is nondeterministic; the sort below makes the output
  // deterministic.
  if (tid == 0) s_fill = 0;
  __syncthreads();
  for (int i = tid; i < vocab; i += blockDim.x) {
    const unsigned long long k = keys[i];
    if (k == 0ull) continue;
    if (total >= kSampleCandidates && k < threshold) continue;
    const int slot = atomicAdd(&s_fill, 1);
    if (slot < kSampleCandidates) {
      s_keys[slot] = k;
      s_z[slot] = select_z(logits, i, s_hash, window_len, temperature,
                           repeat_penalty, presence, frequency);
      s_id[slot] = i;
    }
  }
  __syncthreads();

  const int count = total < kSampleCandidates ? total : kSampleCandidates;
  for (int i = tid; i < kSampleCandidates; i += blockDim.x) {
    if (i >= count) {
      s_keys[i] = 0ull;
      s_z[i] = 0.0f;
      s_id[i] = -1;
    }
  }
  __syncthreads();

  // Bitonic sort, descending key (ids are unique, so the order is total).
  for (int size = 2; size <= kSampleCandidates; size <<= 1) {
    for (int stride = size >> 1; stride > 0; stride >>= 1) {
      for (int i = tid; i < kSampleCandidates; i += blockDim.x) {
        const int j = i ^ stride;
        if (j <= i) continue;
        const bool descending = ((i & size) == 0);
        const bool swap =
            descending ? (s_keys[i] < s_keys[j]) : (s_keys[i] > s_keys[j]);
        if (swap) {
          const unsigned long long tk = s_keys[i];
          s_keys[i] = s_keys[j];
          s_keys[j] = tk;
          const float tz = s_z[i];
          s_z[i] = s_z[j];
          s_z[j] = tz;
          const int ti = s_id[i];
          s_id[i] = s_id[j];
          s_id[j] = ti;
        }
      }
      __syncthreads();
    }
  }

  if (tid < count) out[tid] = Candidate{s_id[tid], s_z[tid]};
  if (tid == 0) *out_count = count;
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

// Masked multi-block argmax: structure and reduction order identical to
// argmax_partial_kernel, with each lane skipped when the mask disallows it.
__global__ void argmax_partial_masked_kernel(const float* __restrict__ x,
                                             const uint8_t* __restrict__ mask,
                                             int n,
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
    const int b = 4 * i;
    if (mask_bit(mask, b + 0) && v.x > mv) { mv = v.x; mi = b + 0; }
    if (mask_bit(mask, b + 1) && v.y > mv) { mv = v.y; mi = b + 1; }
    if (mask_bit(mask, b + 2) && v.z > mv) { mv = v.z; mi = b + 2; }
    if (mask_bit(mask, b + 3) && v.w > mv) { mv = v.w; mi = b + 3; }
  }
  if (blockIdx.x == 0 && tid == 0)
    for (int i = 4 * n4; i < n; ++i)
      if (mask_bit(mask, i) && x[i] > mv) { mv = x[i]; mi = i; }
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
template <typename KV>
__global__ void kv_append_batched_kernel(KV* __restrict__ dst,
                                         const float* __restrict__ src,
                                         uint64_t start_pos, uint64_t n_slots,
                                         int row_elems, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = row_elems * n;
  if (idx >= total) return;
  const int row = idx / row_elems;
  const int e = idx - row * row_elems;
  const uint64_t slot = (start_pos + (uint64_t)row) % n_slots;
  kv_store(dst + slot * (uint64_t)row_elems + (uint64_t)e, src[idx]);
}

// Device-position variant of kv_append_batched_kernel: `start_pos` is a DEVICE
// uint64_t read in-kernel (CUDA Graphs for chunked prefill).
template <typename KV>
__global__ void kv_append_batched_pos_kernel(KV* __restrict__ dst,
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
  kv_store(dst + slot * (uint64_t)row_elems + (uint64_t)e, src[idx]);
}

// --- q8 KV cache append -----------------------------------------------------
//
// A q8 cache row is head_dim int8 values then that row's f32 scale at byte
// offset head_dim, i.e. head_dim + 4 bytes; the row index is
// slot * n_heads_kv + head. One block quantizes one (position, KV head) row:
// the scale is the row's amax / 127, so the whole row must be seen at once
// (an element-wise append cannot compute it). Quantization is symmetric and
// rounds to nearest; the source rows are the fp16-rounded f32 values the
// f16/f32 appends store, so only the int8 rounding is new.
__device__ __forceinline__ void kv_q8_store_row(int8_t *__restrict__ dst,
                                                const float *__restrict__ srow,
                                                uint64_t pos, uint64_t n_slots,
                                                int head, int n_heads_kv,
                                                int head_dim) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const uint64_t slot = pos % n_slots;
  int8_t *drow = dst + (slot * (uint64_t)n_heads_kv + (uint64_t)head) *
                          (uint64_t)(head_dim + 4);
  float amax = 0.0f;
  for (int i = tid; i < head_dim; i += (int)blockDim.x)
    amax = fmaxf(amax, fabsf(srow[i]));
  for (int o = 16; o > 0; o >>= 1)
    amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
  __shared__ float swarp[32];
  __shared__ float sscale;
  if (lane == 0) swarp[warp] = amax;
  __syncthreads();
  if (tid == 0) {
    float m = swarp[0];
    const int nw = ((int)blockDim.x + 31) >> 5;
    for (int w = 1; w < nw; ++w) m = fmaxf(m, swarp[w]);
    sscale = m / 127.0f;
  }
  __syncthreads();
  const float scale = sscale;
  const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
  for (int i = tid; i < head_dim; i += (int)blockDim.x) {
    int q = __float2int_rn(srow[i] * inv);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    drow[i] = (int8_t)q;
  }
  if (tid == 0) *reinterpret_cast<float *>(drow + head_dim) = scale;
}

// One position; the slot's position is a DEVICE value (graph-capturable).
__global__ void kv_append_q8_kernel(int8_t *__restrict__ dst,
                                    const float *__restrict__ src,
                                    const uint64_t *__restrict__ pos,
                                    uint64_t n_slots, int n_heads_kv,
                                    int head_dim) {
  const int head = blockIdx.x;
  kv_q8_store_row(dst, src + (uint64_t)head * head_dim, *pos, n_slots, head,
                  n_heads_kv, head_dim);
}

// n positions from a host start position; grid = n * n_heads_kv.
__global__ void kv_append_batched_q8_kernel(int8_t *__restrict__ dst,
                                            const float *__restrict__ src,
                                            uint64_t start_pos,
                                            uint64_t n_slots, int n_heads_kv,
                                            int head_dim, int n) {
  const int head = blockIdx.x % n_heads_kv;
  const int row = blockIdx.x / n_heads_kv;
  if (row >= n) return;
  const uint64_t row_elems = (uint64_t)n_heads_kv * head_dim;
  kv_q8_store_row(
      dst, src + (uint64_t)row * row_elems + (uint64_t)head * head_dim,
      start_pos + (uint64_t)row, n_slots, head, n_heads_kv, head_dim);
}

// Device-position variant of kv_append_batched_q8_kernel.
__global__ void kv_append_batched_pos_q8_kernel(
    int8_t *__restrict__ dst, const float *__restrict__ src,
    const uint64_t *__restrict__ start_pos, uint64_t n_slots, int n_heads_kv,
    int head_dim, int n) {
  const int head = blockIdx.x % n_heads_kv;
  const int row = blockIdx.x / n_heads_kv;
  if (row >= n) return;
  const uint64_t row_elems = (uint64_t)n_heads_kv * head_dim;
  kv_q8_store_row(
      dst, src + (uint64_t)row * row_elems + (uint64_t)head * head_dim,
      *start_pos + (uint64_t)row, n_slots, head, n_heads_kv, head_dim);
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

bool softcap_dev(const float* x, float cap, float* y, int n) {
  if (!available()) return false;
  if (n <= 0 || !x || !y || !(cap > 0.0f)) return false;
  SB_NOTE_LAUNCH(softcap_kernel, grid_for(n), 256, 0, cur_stream());
  softcap_kernel<<<grid_for(n), 256, 0, cur_stream()>>>(x, cap, y, n);
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

// R10 Stage B: opt in to the strict-order shared-memory tiled kernel for the
// d128/gqa6 class. Default off: the class keeps the scalar kernel until the
// tiled path clears the numerical and performance gates.
bool attn_own_d128_exact() {
  const char* e = std::getenv("SONICBOOM_ATTN_D128_EXACT");
  return e != nullptr &&
         (std::string_view(e) == "1" || std::string_view(e) == "smem");
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
// Defined with the decode-schedule tuner below; declared here because the
// decode path is emitted before it.
uint64_t attn_decode_class_chunk(int n_heads_kv, uint64_t sliding_window);
// The chunk a call runs with: the plan's attribute, else the env override (for
// A/B and for callers that drive the kernel directly), else the class default.
uint64_t attn_decode_requested_chunk(uint64_t chunk_attr, int n_heads_kv,
                                     uint64_t sliding_window);
} // namespace

// The identity split's chunk cap, shared by both decode paths. A tuned chunk
// size may exceed it (up to kAttentionDecodeMaxChunk); when the plan carries
// no `decode_chunk` attribute, the class default is capped by it.
constexpr uint64_t kMaxChunk = 16;

bool decode_attention_dev(const float* q, const float* k_cache,
                          const float* v_cache, const uint64_t* pos,
                          uint64_t window, uint64_t n_slots, int n_heads_q,
                          int n_heads_kv, int head_dim, uint64_t sliding_window,
                          float scale, float* out, int pos_offset, bool kv_f16,
                          uint64_t chunk_attr, bool kv_q8) {
  if (!available()) return false;
  if (n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || head_dim > 512 ||
      !q || !k_cache || !v_cache || !out || !pos || n_slots == 0 || window == 0 ||
      pos_offset < 0)
    return false;
  // B3 verify: row `pos_offset` of the verify step is the n == 1 call at its
  // position, so the whole launch is the decode call with a shifted slot.
  pos += pos_offset;

  const int group_q = n_heads_q / n_heads_kv;

  // Generated flash-decoding (codegen) or the hand-written split-KV (own).
  // Codegen is the default: since attn8 its rounding and reduction order match
  // the hand-written pair exactly and both split the window with the same
  // chunk rule, so it is token-identical. SONICBOOM_ATTN_DECODE=own selects
  // the hand-written pair (the ablation arm); SONICBOOM_ATTN_DECODE=codegen
  // is still accepted and SONICBOOM_ATTN_IMPL=own forces own.
  //
  // Measured per shape class (attention microbenchmark, RTX 3050, us/call): windowed
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
    dsh.kv_f16 = kv_f16;
    dsh.kv_q8 = kv_q8;
    dsh.exp2 = std::getenv("SONICBOOM_ATTN_EXP2") != nullptr;
    if (want_codegen && compiler::attention_shape_supported(dsh) &&
        n_heads_q % n_heads_kv == 0) {
      // The chunk size is the plan's `decode_chunk` attribute: a per-class
      // constant resolved (and tuned) when the graph was built. It never
      // depends on `window`/capacity -- chunk boundaries are absolute key
      // positions -- so the split is the same for any n_ctx and any depth.
      // A graph without the attribute (built before the schedule existed, or
      // by a caller that does not resolve it) uses the class default, which
      // both decode paths share (token identity with the hand-written pair).
      const uint64_t gen_no_split_chunk =
          std::max<uint64_t>(1, (48 * 1024 / sizeof(float)) / (uint64_t)group_q);
      uint64_t kDecodeChunk =
          runtime::feature_enabled(runtime::Feature::SplitKv)
              ? attn_decode_requested_chunk(chunk_attr, n_heads_kv,
                                            sliding_window)
              : std::min(window, gen_no_split_chunk);
      // The generated decoder stages one chunk's scores in a fixed shared
      // view; a larger chunk is outside what the generated text supports.
      if (kDecodeChunk > (uint64_t)compiler::kAttentionDecodeMaxChunk)
        kDecodeChunk = compiler::kAttentionDecodeMaxChunk;
      // Absolute chunk alignment: a window of `window` keys starts and ends
      // mid-chunk when C does not divide it, so it can straddle one more
      // absolute chunk than window / C + 1 (a 512-key SWA window at C == 13
      // touches 41 chunks, not 40). Missing the last block would drop the
      // newest keys -- including the query's own -- from the merge.
      const uint64_t n_chunks = (window - 1) / kDecodeChunk + 2;
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
  //
  // The chunk size is NOT derived from the window: it is the plan's
  // decode_chunk attribute, or the shared class default. Chunk boundaries are
  // absolute (key position / chunk), and the merge order is ascending chunk
  // index, so a query's reduction tree depends only on absolute key positions —
  // not on total length, n_ctx/capacity or prefix reuse.
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
  // Same schedule as the generated pair: the plan's chunk attribute when it
  // has one, else the shared class default. Only the no-split ablation arm
  // (feature off) derives anything from the window.
  const uint64_t chunk =
      runtime::feature_enabled(runtime::Feature::SplitKv)
          ? attn_decode_requested_chunk(chunk_attr, n_heads_kv,
                                        sliding_window)
          : std::min(window, no_split_chunk);
  // Absolute chunk alignment (same rule as the generated pair): the chunk
  // index is a function of the key's absolute position, so the merge order
  // never depends on the window length or the cache capacity. A window that
  // C does not divide straddles floor((window - 1) / C) + 2 absolute chunks.
  const uint64_t n_chunks = (window - 1) / chunk + 2;

  const size_t partial_floats =
      (size_t)n_chunks * (size_t)n_heads_q * (size_t)(head_dim + 2);
  float* partial = static_cast<float*>(
      grow_scratch(g_attn_partial, g_attn_partial_bytes,
                   partial_floats * sizeof(float)));

  const size_t smem = (size_t)group_q * chunk * sizeof(float);
  SB_NOTE_LAUNCH(decode_attention_split_kernel, (unsigned)(n_chunks * (uint64_t)n_heads_kv), group_q * 32, smem, cur_stream());
  const unsigned split_grid =
      (unsigned)(n_chunks * (uint64_t)n_heads_kv);
  if (kv_f16)
    decode_attention_split_kernel<<<split_grid, group_q * 32, smem,
                                    cur_stream()>>>(
        q, reinterpret_cast<const __half*>(k_cache),
        reinterpret_cast<const __half*>(v_cache), pos, n_slots, chunk,
        sliding_window, n_heads_q, n_heads_kv, head_dim, scale, partial);
  else if (kv_q8)
    decode_attention_split_kernel<<<split_grid, group_q * 32, smem,
                                    cur_stream()>>>(
        q, reinterpret_cast<const int8_t*>(k_cache),
        reinterpret_cast<const int8_t*>(v_cache), pos, n_slots, chunk,
        sliding_window, n_heads_q, n_heads_kv, head_dim, scale, partial);
  else
    decode_attention_split_kernel<<<split_grid, group_q * 32, smem,
                                    cur_stream()>>>(
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
                   int row_elems, int pos_offset, bool dst_f16, bool dst_q8,
                   int head_dim) {
  if (!available()) return false;
  if (row_elems <= 0 || !dst_k || !dst_v || !src_k || !src_v || !pos ||
      n_slots == 0 || pos_offset < 0)
    return false;
  pos += pos_offset;  // B3 verify: append row `pos_offset` at its position
  if (dst_q8) {
    if (head_dim <= 0 || row_elems % head_dim != 0) return false;
    const int nhkv = row_elems / head_dim;
    SB_NOTE_LAUNCH(kv_append_q8_kernel, nhkv, 128, 0, cur_stream());
    kv_append_q8_kernel<<<nhkv, 128, 0, cur_stream()>>>(
        reinterpret_cast<int8_t*>(dst_k), src_k, pos, n_slots, nhkv, head_dim);
    SB_NOTE_LAUNCH(kv_append_q8_kernel, nhkv, 128, 0, cur_stream());
    kv_append_q8_kernel<<<nhkv, 128, 0, cur_stream()>>>(
        reinterpret_cast<int8_t*>(dst_v), src_v, pos, n_slots, nhkv, head_dim);
    return cudaGetLastError() == cudaSuccess;
  }
  const int grid = grid_for(row_elems);
  SB_NOTE_LAUNCH(kv_append_kernel, grid, 256, 0, cur_stream());
  if (dst_f16) {
    kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(
        reinterpret_cast<__half*>(dst_k), src_k, pos, n_slots, row_elems);
    SB_NOTE_LAUNCH(kv_append_kernel, grid, 256, 0, cur_stream());
    kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(
        reinterpret_cast<__half*>(dst_v), src_v, pos, n_slots, row_elems);
  } else {
    kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(dst_k, src_k, pos,
                                                     n_slots, row_elems);
    SB_NOTE_LAUNCH(kv_append_kernel, grid, 256, 0, cur_stream());
    kv_append_kernel<<<grid, 256, 0, cur_stream()>>>(dst_v, src_v, pos,
                                                     n_slots, row_elems);
  }
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
                               float* out, bool kv_f16, bool kv_q8) {
  if (!available()) return false;
  if (n <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 || n_slots == 0)
    return false;
  if (!q || !k_batch || !v_batch || !k_cache || !v_cache || !out) return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  if (head_dim % 32 != 0 || head_dim > 512) return false;
  if (n_heads_q > 32) return false;  // one warp per query head; 32 warps = block limit
  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  SB_NOTE_LAUNCH(flash_attention_chunk_kernel, n, n_heads_q * 32, smem, cur_stream());
  if (kv_f16)
    flash_attention_chunk_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
        q, k_batch, v_batch, reinterpret_cast<const __half*>(k_cache),
        reinterpret_cast<const __half*>(v_cache), start, n_slots, n_heads_q,
        n_heads_kv, head_dim, sliding_window, scale, out);
  else if (kv_q8)
    flash_attention_chunk_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
        q, k_batch, v_batch, reinterpret_cast<const int8_t*>(k_cache),
        reinterpret_cast<const int8_t*>(v_cache), start, n_slots, n_heads_q,
        n_heads_kv, head_dim, sliding_window, scale, out);
  else
    flash_attention_chunk_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q,
        n_heads_kv, head_dim, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

template <typename KV, int HEAD_DIM, int BK, int NW, int QB, int GQA,
          bool CAUSAL = true>
bool launch_tiled_flash_attention(const float* q, const float* k_batch,
                                  const float* v_batch, const void* k_cache,
                                  const void* v_cache, const uint64_t* start,
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
  auto kfn =
      flash_attention_tiled_kernel<KV, HEAD_DIM, BK, NW, QB, GQA, CAUSAL>;
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
      q, k_batch, v_batch, static_cast<const KV*>(k_cache),
      static_cast<const KV*>(v_cache), start, n_slots, n_heads_q, n_heads_kv,
      n, sliding_window, scale, out);
  return cudaGetLastError() == cudaSuccess;
}

// f32 vs f16 KV cache arm for the hand-written tiled kernel.
template <int HEAD_DIM, int BK, int NW, int QB, int GQA>
bool launch_tiled_flash_attention_kv(const float* q, const float* k_batch,
                                     const float* v_batch,
                                     const float* k_cache,
                                     const float* v_cache,
                                     const uint64_t* start, uint64_t n_slots,
                                     int n, int n_heads_q, int n_heads_kv,
                                     uint64_t sliding_window, float scale,
                                     float* out, bool kv_f16, bool kv_q8) {
  if (kv_f16)
    return launch_tiled_flash_attention<__half, HEAD_DIM, BK, NW, QB, GQA>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
  if (kv_q8)
    return launch_tiled_flash_attention<int8_t, HEAD_DIM, BK, NW, QB, GQA>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out);
  return launch_tiled_flash_attention<float, HEAD_DIM, BK, NW, QB, GQA>(
      q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
      n_heads_kv, sliding_window, scale, out);
}

// Strict-order shared-memory tiled kernel. Fixed block shape (NQ query
// positions x NHQ warps) and BK == FA_BK so its softmax tiles and reduction
// points are the scalar kernel's.
constexpr int kExactNQ = 2;

template <typename KV, int HEAD_DIM, int NHQ, int NHKV>
bool launch_exact_flash_attention(const float* q, const float* k_batch,
                                  const float* v_batch, const void* k_cache,
                                  const void* v_cache, const uint64_t* start,
                                  uint64_t n_slots, int n, uint64_t sliding_window,
                                  float scale, float* out) {
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

// Host-facing counterpart of flash_attention_full_dev. q/k/v/out are plain
// [n, n_heads, head_dim] host spans; the inputs are staged into the shared
// scratch buffers and the result is copied back. This is the entry the
// differential test uses, so the tiled full-attention kernel is exercised
// through the same device pointers the DiT path hands it.
bool flash_attention_full(std::span<const float> q, std::span<const float> k,
                          std::span<const float> v, uint64_t n,
                          uint64_t n_heads, uint64_t head_dim, float scale,
                          std::span<float> out) {
  if (!available()) return false;
  if (n == 0 || n_heads == 0 || head_dim == 0) return false;
  const size_t total = (size_t)n * n_heads * head_dim;
  if (q.size() < total || k.size() < total || v.size() < total ||
      out.size() < total)
    return false;
  const size_t bytes = total * sizeof(float);
  void* d_q = grow_scratch(g_dx, g_dx_bytes, bytes);
  void* d_k = grow_scratch(g_db, g_db_bytes, bytes);
  void* d_v = grow_scratch(g_dy, g_dy_bytes, bytes);
  void* d_o = grow_scratch(g_dff, g_dff_bytes, bytes);
  if (!d_q || !d_k || !d_v || !d_o) return false;
  if (cudaMemcpy(d_q, q.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_k, k.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_v, v.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  if (!flash_attention_full_dev(static_cast<const float*>(d_q),
                                static_cast<const float*>(d_k),
                                static_cast<const float*>(d_v), (int)n,
                                (int)n_heads, (int)head_dim, scale,
                                static_cast<float*>(d_o)))
    return false;
  return cudaMemcpy(out.data(), d_o, bytes, cudaMemcpyDeviceToHost) ==
         cudaSuccess;
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
                                            int as, int pack = 1,
                                            bool qreg = true, int hf = 0) {
  compiler::AttentionShape s;
  s.head_dim = head_dim;
  s.gqa = gqa;
  s.window_mask = window;
  s.br = br;
  s.bc = bc;
  s.dps = dps;
  s.warp_head = wh;
  s.async_stages = as;
  s.gqa_pack = pack;
  s.q_in_reg = qreg;
  s.head_fast = hf != 0;
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
    // T4 measured (wide bench, 3050): packing all four query heads of a KV
    // head into one block and pipelining the K/V staging beats every unpacked
    // tile on both query lengths -- pp512 0.325 ms (0.635 for the previous
    // 64x16 sync tile), 2048 queries 1.652 ms (3.447). bc/dps are unchanged
    // from the previous default, so the reduction order and the token stream
    // are unchanged.
    shape.br = 16;
    shape.bc = 16;
    shape.dps = 4;  // 16 warps, 53 KiB
    shape.async_stages = 2;
    shape.gqa_pack = 4;
  } else if (gqa == 4 && head_dim == 512) {
    // T4 measured (wide bench, 3050): the hand-written kernel wins this
    // class, and no legal generated tile beats it. Counting only shapes that
    // really generate -- a tile over the shared-memory gate silently falls
    // back to own, which makes several rows look like wins -- the best
    // generated shape, 16x16 pack2 dps8, is 1.717 ms at q=512 and 27.19 ms
    // at q=2048 against own 1.603/20.25; 32x16 dps8 is 1.883/28.22. The
    // 96 KiB budget is what binds: gqa_pack >= 3 needs 107.8 KiB (q 65.0 +
    // k 24.0 + v 16.3), and the cp.async f32 prefetch buffer is 64 KiB,
    // which cannot alias the 33 KiB Q tile at pack 2 / dps 8 (140 KiB), so
    // every legal d512 shape stages synchronously. The dispatcher resolves
    // the class to own (attn_default_choice); this shape is the reference
    // the explicit SONICBOOM_ATTN_IMPL=codegen arm and the tuner measure
    // against. dps 8 keeps dc = 64 (the previously shipped d512 split) and
    // packing is bit-identical per head, so the codegen arm's token stream
    // is unchanged from the old 32x16 dps8 default.
    shape.br = 16;
    shape.bc = 16;
    shape.dps = 8;       // dc = 64, 16 warps
    shape.gqa_pack = 2;  // q rows 32 -> 74.5 KiB, the largest legal pack
  } else if (gqa == 6 && head_dim == 128) {
    // d128/gqa6 (DeepSeek-R1-Distill-Qwen-1.5B): measurement arm only, not a
    // class default. The class has no hand-written tiled specialization, so
    // prefill takes the scalar warp-per-(position, head) kernel (M3 nsys,
    // pp512 RTX 3050: 48.7% of prefill kernel time at ~1 TFLOP/s). This tile
    // is 1.92x faster there (pp512 2841 -> 5450 tok/s, attention 562 -> 55 ms
    // per capture), but its S/P association order moves DeepSeek's first
    // greedy token on 8 of the 50 standard-profile gate prompts -- the
    // acceptance gate requires 50/50 -- so auto keeps the scalar kernel in
    // both profiles. Reachable through SONICBOOM_ATTN_IMPL=codegen. dps 2
    // keeps dc = 64; packing all six query heads of a KV head per block is 12
    // warps and 36.6 KiB, staged synchronously (f16 KV rules out cp.async).
    shape.br = 16;
    shape.bc = 16;
    shape.dps = 2;
    shape.gqa_pack = 6;
    shape.async_stages = 0;
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
    int br = 0, bc = 0, dps = 0, as = 0, pack = 1, qs = 0, hf = 0;
    if (std::sscanf(e, "wh,%d,%d,%d", &br, &bc, &as) >= 2 && br > 0 && bc > 0) {
      // P-in-registers (warp-head) arm: br must be 16, dps is implicit. The
      // optional third value is the cp.async pipeline depth.
      shape->warp_head = true;
      shape->br = br;
      shape->bc = bc;
      shape->dps = 1;
      shape->async_stages = as;
    } else if (std::sscanf(e, "%d,%d,%d,%d,%d,%d,%d", &br, &bc, &dps, &as,
                           &pack, &qs, &hf) >= 3 &&
               br > 0 && bc > 0 && dps > 0) {
      // Optional fourth value: cp.async depth; fifth: GQA packing; sixth:
      // nonzero reloads Q fragments per key tile instead of holding them in
      // registers; seventh: nonzero puts the head in blockIdx.x (head-fast
      // grid order).
      shape->br = br;
      shape->bc = bc;
      shape->dps = dps;
      shape->async_stages = as;
      shape->gqa_pack = pack;
      shape->q_in_reg = qs == 0;
      shape->head_fast = hf != 0;
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
  // `shape` already carries kv_f16: the caller builds it from the profile, so
  // the region identity, the staging path and the shared-memory budget agree.
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
  const int tiles = (n + shape.br - 1) / shape.br;
  // Warp-head blocks are (query tile, KV head): each warp covers one of the
  // KV head's query heads. Packed blocks are (query tile, query-head group).
  const int heads = shape.warp_head
                        ? n_heads_kv
                        : n_heads_q / std::max(1, shape.gqa_pack);
  // head_fast puts the head in blockIdx.x so the blocks launched together
  // share one K/V range; the kernel reads the two ids the same way either
  // way.
  return planner::glue::launch_attention(
      shape, args, shape.head_fast ? heads : tiles,
      shape.head_fast ? tiles : heads);
}

// --- Full-attention op class (non-causal, no KV cache, no prefix reuse) ----
//
// Class selection is keyed on the op's attributes and the device, never on the
// model that issued the call. The attributes of this class are causal = 0,
// kv_cache = absent, prefix_reuse = impossible (there is no cache to reuse)
// and window = 0; head_dim, the GQA ratio and the head count complete the key.
// A causal or cached op -- every LLM prefill/decode shape, including DeepSeek's
// d128 GQA 6 -- cannot resolve here: the runtime only reaches this entry from
// the full-attention op, and the generator's support rules reject a causal,
// windowed or cached shape for the full mask. full_attn is part of the region
// identity, so a causal class can never be served from a full-attention JIT
// cache entry (and a full class can never select the causal kernel).
//
// Two regimes, split by score-plane capacity (the documented R1 exception in
// docs/decision-registry.md):
//   full.score-gemm  n <= kFullScoreMaxTokens: the per-head [n, n] f16 score
//                    planes fit the scratch budget; S and P are batched cuBLAS
//                    f16 GEMMs with a row softmax between them.
//   full.tiled       above the boundary: online-softmax tiling, no score
//                    planes. The generated FA-2 warp-head layout is the
//                    default (one warp per 16 query rows and the whole
//                    head_dim, P in registers), with the hand-written tiled
//                    kernel as an autotune candidate.
// The boundary is a capacity constant of the device, not a live n: a
// resolution maps to exactly one regime, deterministically and reproducibly.
constexpr int kFullScoreMaxTokens = 2048;

// Declared here (defined with the causal prefill tuner below): a host-timed
// probe is illegal while the current stream is capturing.
bool attn_capturing();

// The tiled regime's decision probe: a fixed synthetic length, never the
// caller's n, so the class resolves the same way on every call.
constexpr int kFullTuneTokens = 2048;
// A tiling candidate must beat the class default by this fraction to replace
// it (same rule as the causal prefill tuner).
constexpr double kFullSwitchMargin = 0.03;

// [tokens][heads][hd] f32 -> three [heads][tdst][hd] f16 planes. `tdst` is
// the padded token count that keeps every GEMM leading dimension a multiple of
// 8 (a tensor-core requirement); only the real rows are written, and only
// those are ever read (the GEMM dims are the real token count).
__global__ void full_pack_qkv_f16_kernel(const float* __restrict__ q,
                                         const float* __restrict__ k,
                                         const float* __restrict__ v,
                                         __half* __restrict__ qh,
                                         __half* __restrict__ kh,
                                         __half* __restrict__ vh, int tokens,
                                         int tdst, int heads, int hd) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  const long total = (long)tokens * heads * hd;
  if (i >= total) return;
  const long per_token = (long)heads * hd;
  const long t = i / per_token;
  const int rem = int(i - t * per_token);
  const int h = rem / hd;
  const int d = rem - h * hd;
  const long dst = (long(h) * tdst + t) * hd + d;
  qh[dst] = __float2half_rn(q[i]);
  kh[dst] = __float2half_rn(k[i]);
  vh[dst] = __float2half_rn(v[i]);
}

// The two reductions use separate slots so the value one returns can never be
// overwritten by the other's first shared-memory store.
__device__ __forceinline__ float full_block_reduce_max(float v, float* red) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
  if (lane == 0) red[warp] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nw) ? red[threadIdx.x] : -1e30f;
  if (warp == 0) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    if (lane == 0) red[0] = v;
  }
  __syncthreads();
  return red[0];
}

__device__ __forceinline__ float full_block_reduce_sum(float v, float* red) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) red[32 + warp] = v;
  __syncthreads();
  const int nw = (blockDim.x + 31) >> 5;
  v = (threadIdx.x < nw) ? red[32 + threadIdx.x] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) red[32] = v;
  }
  __syncthreads();
  return red[32];
}

// Per-head attention planes are [head][token][dim]; the caller's buffer is
// [token][head][dim]. Gather the real rows out of the plane.
__global__ void full_attn_interleave_kernel(const float* __restrict__ plane,
                                            float* __restrict__ out,
                                            int tokens, int plane_tokens,
                                            int heads, int hd) {
  const long total = (long)tokens * heads * hd;
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int d = int(i % hd);
  const long r = i / hd;
  const int h = int(r % heads);
  const long q = r / heads;
  out[i] = plane[(h * (long)plane_tokens + q) * hd + d];
}

// One block of 256 threads per row. The row is read once into registers, so
// the scale and both softmax passes touch it only once.
__global__ void full_softmax_f16_rows_kernel(const __half* __restrict__ s,
                                             __half* __restrict__ p, int cols,
                                             int row_stride, float scale) {
  __shared__ float red[64];
  const long row = blockIdx.x;
  const __half* src = s + row * row_stride;
  __half* dst = p + row * row_stride;
  float v[8];
  float m = -1e30f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = threadIdx.x + k * 256;
    v[k] = (i < cols) ? __half2float(src[i]) * scale : -1e30f;
    m = fmaxf(m, v[k]);
  }
  m = full_block_reduce_max(m, red);
  float sum = 0.0f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    v[k] = (v[k] > -1e29f) ? __expf(v[k] - m) : 0.0f;
    sum += v[k];
  }
  sum = full_block_reduce_sum(sum, red);
  const float inv = 1.0f / sum;
  #pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int i = threadIdx.x + k * 256;
    if (i < cols) dst[i] = __float2half_rn(v[k] * inv);
  }
}

// Score-plane scratch of the full.score-gemm regime. Sized for the largest
// token count the regime admits; released at the caller's teardown.
float* g_full_scratch = nullptr;
size_t g_full_scratch_bytes = 0;

bool ensure_full_scratch(size_t bytes) {
  if (bytes <= g_full_scratch_bytes) return true;
  if (g_full_scratch != nullptr) cudaFree(g_full_scratch);
  g_full_scratch = nullptr;
  g_full_scratch_bytes = 0;
  void* p = nullptr;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return false;
  g_full_scratch = static_cast<float*>(p);
  g_full_scratch_bytes = bytes;
  return true;
}

void release_full_scratch() {
  if (g_full_scratch != nullptr) cudaFree(g_full_scratch);
  g_full_scratch = nullptr;
  g_full_scratch_bytes = 0;
}

cublasHandle_t full_attn_cublas() {
  static cublasHandle_t h = nullptr;
  if (h == nullptr && cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) h = nullptr;
  return h;
}

size_t full_score_scratch_bytes(int tokens, int heads, int head_dim) {
  const size_t tpad = (size_t(tokens) + 7) & ~size_t(7);
  const size_t kv_elems = tpad * size_t(heads) * size_t(head_dim);
  const size_t s_elems = size_t(heads) * tpad * tpad;
  const size_t s_pad = (s_elems + 1) & ~size_t(1);  // keep the f32 tail aligned
  return 2 * (3 * kv_elems + 2 * s_pad) + 4 * kv_elems;
}

// full.score-gemm: S = scale * Q K^T, P = softmax_rows(S), O = P V, all as
// batched f16 GEMMs (fp32 accumulate) over the per-head planes. False when the
// shape or the scratch does not fit the regime; the caller then takes the
// tiled regime.
bool full_attention_score_gemm(const float* q, const float* k, const float* v,
                               int tokens, int heads, int head_dim, float scale,
                               float* out) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return false;
  if (tokens > kFullScoreMaxTokens) return false;
  if (head_dim % 32 != 0 || head_dim > 32 * 8) return false;
  cublasHandle_t h = full_attn_cublas();
  if (h == nullptr) return false;

  // Tensor-core GEMMs require every leading dimension to be a multiple of 8,
  // so the per-head planes are padded to `tpad` tokens. The tile dimensions
  // stay at the real token count: the padded rows are never read.
  const int tpad = (tokens + 7) & ~7;
  const size_t T = size_t(tokens), H = size_t(heads), D = size_t(head_dim);
  const size_t TP = size_t(tpad);
  const size_t kv_elems = TP * H * D;
  const size_t s_elems = H * TP * TP;
  const size_t s_pad = (s_elems + 1) & ~size_t(1);  // keep the f32 tail aligned
  const size_t bytes = 2 * (3 * kv_elems + 2 * s_pad) + 4 * kv_elems;
  if (!ensure_full_scratch(bytes)) return false;

  __half* qh = reinterpret_cast<__half*>(g_full_scratch);
  __half* kh = qh + kv_elems;
  __half* vh = kh + kv_elems;
  __half* sh = vh + kv_elems;
  __half* ph = sh + s_pad;
  float* od = reinterpret_cast<float*>(ph + s_pad);  // [head][token][dim]

  full_pack_qkv_f16_kernel<<<grid_for(int(T * H * D)), 256, 0, cur_stream()>>>(
      q, k, v, qh, kh, vh, tokens, tpad, heads, head_dim);

  // S_cm (M=key, N=query) per head is the row-major [query][key] plane the
  // softmax walks. Both operands are contiguous [token][dim] planes, so the
  // column-major leading dimensions are just head_dim.
  const long long stride_kv = (long long)tpad * head_dim;
  const long long stride_s = (long long)tpad * tpad;
  const float alpha = 1.0f, beta = 0.0f, one = 1.0f;  // scale is in softmax
  cublasSetStream(h, cur_stream());

  // S = Q K^T: C(m=k, n=q) = sum_d kh(k,d) qh(q,d). A row-major [token][dim]
  // plane is the *transpose* of its column-major view, so A takes the T flag
  // and B (also [token][dim]) is already the K-dim-by-N form.
  if (cublasGemmStridedBatchedEx(
          h, CUBLAS_OP_T, CUBLAS_OP_N, tokens, tokens, head_dim, &alpha, kh,
          CUDA_R_16F, head_dim, stride_kv, qh, CUDA_R_16F, head_dim,
          stride_kv, &beta, sh, CUDA_R_16F, tpad, stride_s, heads,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) !=
      CUBLAS_STATUS_SUCCESS)
    return false;

  // One block per padded plane row (the padded query rows are computed too so
  // the block index maps straight to a plane row; those rows are never read).
  full_softmax_f16_rows_kernel<<<unsigned(H * TP), 256, 0, cur_stream()>>>(
      sh, ph, tokens, tpad, scale);

  // O^T = V^T P^T: C(m=d, n=q) = sum_k vh(k,d) ph(q,k). `out` is
  // [query][head][dim], so its per-head C view is column-major (M=dim, N=query);
  // vh's column-major view is already V^T (dim by token) and ph's is P^T, so
  // both operands take the N flag.
  if (cublasGemmStridedBatchedEx(
          h, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, tokens, tokens, &one, vh,
          CUDA_R_16F, head_dim, stride_kv, ph, CUDA_R_16F, tpad, stride_s,
          &beta, od, CUDA_R_32F, head_dim, stride_kv, heads,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  full_attn_interleave_kernel<<<grid_for(int(T * H * D)), 256, 0,
                                cur_stream()>>>(od, out, tokens, tpad, heads,
                                                head_dim);
  return cudaGetLastError() == cudaSuccess;
}

// --- full.tiled: resolution, autotune and launch ---------------------------

struct FullTile {
  int br = 0;
  int bc = 0;
  int as = 0;
};

// The generated full-attention shape: the FA-2 warp-head layout (one warp per
// 16 query rows of one head, whole head_dim, no dim split), no window, no
// cache, no exp2 (the exp path keeps the same softmax arithmetic the reference
// kernel uses, so a re-measurement never changes the numbers).
compiler::AttentionShape full_attn_shape(int head_dim, int gqa,
                                         const FullTile& t) {
  compiler::AttentionShape s;
  s.head_dim = head_dim;
  s.gqa = gqa;
  s.full_attn = true;
  s.warp_head = true;
  s.br = t.br;
  s.bc = t.bc;
  s.dps = 1;
  s.async_stages = t.as;
  return s;
}

bool full_tile_ok(int head_dim, int gqa, const FullTile& t, int max_smem) {
  const compiler::AttentionShape s = full_attn_shape(head_dim, gqa, t);
  if (!compiler::attention_shape_supported(s)) return false;
  if (compiler::attention_shared_bytes(s) > max_smem) return false;
  // 16 warps is the largest block the register file holds (see the Q3 apply).
  if (compiler::attention_block_threads(s) > 512) return false;
  return true;
}

// Candidate tilings, in preference order. The first supported entry is the
// static default: br = 128 (8 warps, 256 threads) holds the whole head_dim
// accumulator plus the Q fragments without approaching the 128-register cap of
// a 512-thread block. The wider br = 192/256 tiles halve the K/V traffic per
// query row but need ~110+ registers per thread; the tuner launches them and
// keeps one only if it really runs and wins.
std::vector<FullTile> full_tile_candidates(int head_dim, int gqa) {
  int dev = 0, max_smem = 96 * 1024;
  if (cudaGetDevice(&dev) == cudaSuccess) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               dev) == cudaSuccess &&
        v > 0)
      max_smem = v;
  }
  std::vector<FullTile> out;
  for (const FullTile t : {FullTile{128, 32, 2}, FullTile{128, 16, 2},
                           FullTile{192, 16, 2}, FullTile{256, 32, 2},
                           FullTile{256, 32, 3}, FullTile{192, 32, 0},
                           FullTile{128, 32, 0}, FullTile{64, 32, 2},
                           FullTile{256, 16, 2}}) {
    if (full_tile_ok(head_dim, gqa, t, max_smem)) out.push_back(t);
  }
  return out;
}

enum class FullClass { ScoreGemm, Tiled };
enum class FullImpl { Own, Codegen };

struct FullResolution {
  FullClass cls = FullClass::Tiled;
  FullImpl impl = FullImpl::Codegen;
  FullTile tile;
};

std::string full_choice_name(const FullResolution& r) {
  if (r.cls == FullClass::ScoreGemm) return "full.score-gemm";
  if (r.impl == FullImpl::Own) return "full.tiled:own";
  return "full.tiled:codegen-wh:" + std::to_string(r.tile.br) + "," +
         std::to_string(r.tile.bc) + ",as" + std::to_string(r.tile.as);
}

// The op's attribute key (device is the registry's own dimension). The mask
// kind, the cache state and prefix reuse are attributes of the class, not of
// the model that issued the call.
std::string full_attn_attrs(int head_dim, int gqa) {
  return "causal0-kvcache0-prefixreuse0-window0-d" + std::to_string(head_dim) +
         "-g" + std::to_string(gqa);
}

// Class scope. The two regimes are two scopes: the capacity boundary is a
// device constant, so neither scope ever contains a live n.
std::string full_attn_scope(int head_dim, int gqa, bool score_gemm) {
  return "full-" + full_attn_attrs(head_dim, gqa) +
         (score_gemm ? "-cap2048" : "-over2048");
}

// The tiled regime's tuner entry is regime-independent: the same kernel serves
// the tiled class whether it was reached at n > 2048 or by a shape at n <=
// 2048 that the score-GEMM regime cannot serve.
std::string full_tiled_scope(int head_dim, int gqa) {
  return "full-" + full_attn_attrs(head_dim, gqa) + "-tiled";
}

enum class FullClassName { Auto, ScoreGemm, Tiled, Own, Codegen, Scalar };

struct FullOverride {
  FullClassName cls = FullClassName::Auto;
  const char* raw = nullptr;
};

void full_warn_once(const std::string& msg) {
  static std::mutex mu;
  static std::unordered_set<std::string> seen;
  {
    std::lock_guard<std::mutex> lk(mu);
    if (!seen.insert(msg).second) return;
  }
  std::fprintf(stderr, "[attention] %s\n", msg.c_str());
}

// SONICBOOM_ATTN_CLASS overrides the class for an A/B run; SONICBOOM_DIT_ATTN
// is kept as its alias (the old spelling). Values that do not describe this
// op's attributes are refused: an LLM class on a non-causal, cache-less op
// would silently select a causal kernel, which is exactly what the class key
// exists to prevent.
FullOverride full_class_override() {
  const char* e = std::getenv("SONICBOOM_ATTN_CLASS");
  if (e == nullptr || *e == '\0') e = std::getenv("SONICBOOM_DIT_ATTN");
  FullOverride o;
  if (e == nullptr || *e == '\0') return o;
  o.raw = e;
  const std::string_view v(e);
  if (v == "auto") o.cls = FullClassName::Auto;
  else if (v == "full.score-gemm") o.cls = FullClassName::ScoreGemm;
  else if (v == "full.tiled") o.cls = FullClassName::Tiled;
  else if (v == "full.own") o.cls = FullClassName::Own;
  else if (v == "full.codegen") o.cls = FullClassName::Codegen;
  else if (v == "scalar" || v == "full.scalar") o.cls = FullClassName::Scalar;
  else {
    full_warn_once("SONICBOOM_ATTN_CLASS=" + std::string(v) +
                   " is not a class for a non-causal, cache-less attention "
                   "op; using auto");
    o.cls = FullClassName::Auto;
  }
  return o;
}

// Process memo: the resolution and the source that produced it, so a repeat
// call neither retimes nor misreports where the class came from.
struct FullMemo {
  FullResolution res;
  planner::decisions::Source src = planner::decisions::Source::Default;
};

std::mutex g_full_choice_mutex;
std::unordered_map<std::string, FullMemo> g_full_choice;

// Class records are written once per (name, scope, value): a second value for
// the same scope must still reach the registry, where it is counted as a
// conflict instead of being hidden by a per-scope flag.
std::mutex g_full_rec_mutex;
std::unordered_set<std::string> g_full_rec_seen;

bool full_record_once(planner::decisions::Entry e) {
  const std::string k = e.name + "\x1f" + e.scope + "\x1f" + e.value;
  {
    std::lock_guard<std::mutex> lk(g_full_rec_mutex);
    if (!g_full_rec_seen.insert(k).second) return false;
  }
  planner::decisions::record(std::move(e));
  return true;
}

void full_record_class(const FullResolution& r, int head_dim, int gqa,
                       int n_heads, int score_gemm_vram, bool fallback,
                       planner::decisions::Source src) {
  planner::decisions::Entry e;
  e.name = "attention.class";
  e.scope = full_attn_scope(head_dim, gqa, r.cls == FullClass::ScoreGemm);
  e.value = full_choice_name(r);
  e.source = src;
  e.env = "SONICBOOM_ATTN_CLASS";
  e.inputs = full_attn_attrs(head_dim, gqa) + " heads=" +
             std::to_string(n_heads) + " device=sm_86";
  e.candidates = "full.score-gemm|full.tiled";
  if (r.cls == FullClass::ScoreGemm)
    e.reason = "n <= 2048: the per-head score planes fit the scratch budget "
               "(R1 capacity exception; no prefix reuse, deterministic)";
  else if (fallback)
    e.reason = "score-GEMM regime unavailable for this shape; tiled regime "
               "(R1 capacity exception; no prefix reuse, deterministic)";
  else
    e.reason = "n > 2048: score planes exceed the scratch budget; online "
               "tiling (R1 capacity exception; no prefix reuse, deterministic)";
  e.numerics = r.cls == FullClass::ScoreGemm
                   ? "f16 GEMM operands, fp32 accumulate, row softmax in f32"
                   : "f16 K/V, fp32 accumulate, deterministic per resolution";
  e.vram_bytes = r.cls == FullClass::ScoreGemm ? score_gemm_vram : 0;
  full_record_once(std::move(e));
}

bool full_attn_launch_own(int head_dim, const float* q, const float* k,
                          const float* v, int n, int n_heads, float scale,
                          float* out) {
  if (head_dim != 128) return false;
  // The hand-written tiled kernel in dense full-attention mode: identical
  // geometry to the causal prefill specialization, with the mask compile-time
  // off (no cache, no start position, no window).
  return launch_tiled_flash_attention<float, 128, 64, 2, 128, 1, false>(
      q, k, v, nullptr, nullptr, nullptr, /*n_slots=*/0, n, n_heads, n_heads,
      /*sliding_window=*/0, scale, out);
}

// Launch the generated full-attention kernel. Falls back to the hand-written
// kernel only when the generator cannot launch (JIT failure); the production
// resolution always names the implementation that ran.
bool full_attn_launch(const FullResolution& r, int head_dim, int gqa,
                      const float* q, const float* k, const float* v, int n,
                      int n_heads, float scale, float* out,
                      FullImpl& ran) {
  if (r.impl == FullImpl::Codegen) {
    const compiler::AttentionShape s = full_attn_shape(head_dim, gqa, r.tile);
    if (attn_launch_codegen(s, q, k, v, nullptr, nullptr, nullptr, 0, n,
                            n_heads, n_heads, 0, scale, out)) {
      ran = FullImpl::Codegen;
      return true;
    }
    full_warn_once("full-attention kernel launch failed for " +
                   full_choice_name(r) + "; using the hand-written kernel");
  }
  ran = FullImpl::Own;
  return full_attn_launch_own(head_dim, q, k, v, n, n_heads, scale, out);
}

// Static tiled default: the first supported generated tiling, else the
// hand-written kernel.
FullResolution full_tiled_default(int head_dim, int gqa) {
  FullResolution r;
  r.cls = FullClass::Tiled;
  const std::vector<FullTile> cands = full_tile_candidates(head_dim, gqa);
  if (!cands.empty()) {
    r.impl = FullImpl::Codegen;
    r.tile = cands.front();
  } else {
    r.impl = FullImpl::Own;
  }
  return r;
}

std::string full_tune_key(int head_dim, int gqa, int n_heads) {
  // The generated-code version and the default the pick was measured against
  // are decision inputs: changing either retimes the class. The probe length,
  // head dimension, GQA ratio and head count are the class; the caller's n is
  // never part of the key.
  std::string k = compiler::attention_codegen_version();
  k += "|" + full_tiled_scope(head_dim, gqa) + "|h" + std::to_string(n_heads) +
       "|wl" + std::to_string(kFullTuneTokens) + "|dflt-" +
       full_choice_name(full_tiled_default(head_dim, gqa));
  return k;
}

std::optional<FullResolution>
full_choice_from_values(const std::map<std::string, int>& v, int head_dim,
                        int gqa) {
  const auto get = [&v](const char* name, int& dst) {
    const auto it = v.find(name);
    if (it == v.end()) return false;
    dst = it->second;
    return true;
  };
  int impl = 0;
  if (!get("impl", impl)) return std::nullopt;
  FullResolution r;
  r.cls = FullClass::Tiled;
  if (impl == 0) {
    if (head_dim != 128) return std::nullopt;  // no hand-written instantiation
    r.impl = FullImpl::Own;
    return r;
  }
  int br = 0, bc = 0, as = 0;
  if (!get("br", br) || !get("bc", bc)) return std::nullopt;
  (void)get("as", as);
  // A cached tile must still be a supported, launchable tile on this device.
  for (const FullTile& c : full_tile_candidates(head_dim, gqa))
    if (c.br == br && c.bc == bc && c.as == as) {
      r.impl = FullImpl::Codegen;
      r.tile = c;
      return r;
    }
  return std::nullopt;
}

// Time every supported tiling plus the hand-written kernel on one fixed
// synthetic full-attention call. The probe length is a constant of the class,
// so the ranking never depends on the caller's n.
FullResolution full_tune_tiled(int head_dim, int gqa, int n_heads,
                               const FullResolution& def, double& tune_s,
                               int& candidate_count) {
  const auto t0 = std::chrono::steady_clock::now();
  candidate_count = 0;
  std::vector<FullResolution> cands{def};
  auto add = [&](const FullResolution& c) {
    for (const FullResolution& x : cands)
      if (x.cls == c.cls && x.impl == c.impl && x.tile.br == c.tile.br &&
          x.tile.bc == c.tile.bc && x.tile.as == c.tile.as)
        return;
    cands.push_back(c);
  };
  {
    FullResolution own;
    own.cls = FullClass::Tiled;
    own.impl = FullImpl::Own;
    if (head_dim == 128) add(own);
  }
  for (const FullTile& t : full_tile_candidates(head_dim, gqa)) {
    FullResolution c;
    c.cls = FullClass::Tiled;
    c.impl = FullImpl::Codegen;
    c.tile = t;
    add(c);
  }

  const int n = kFullTuneTokens;
  const size_t elems = size_t(n) * size_t(n_heads) * size_t(head_dim);
  float *q = nullptr, *k = nullptr, *v = nullptr, *out = nullptr;
  bool ok = cudaMalloc(reinterpret_cast<void**>(&q), elems * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void**>(&k), elems * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void**>(&v), elems * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void**>(&out), elems * 4) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    for (float* p : {q, k, v, out})
      if (p) cudaFree(p);
    tune_s = 0.0;
    return def;
  }
  // Zeroed inputs: the kernel's work is data-independent (no early exits).
  for (float* p : {q, k, v, out}) cudaMemsetAsync(p, 0, elems * 4, cur_stream());

  const float scale = 1.0f / std::sqrt(float(head_dim));
  auto run = [&](const FullResolution& r) {
    if (r.impl == FullImpl::Own)
      return full_attn_launch_own(head_dim, q, k, v, n, n_heads, scale, out);
    return attn_launch_codegen(full_attn_shape(head_dim, gqa, r.tile), q, k, v,
                               nullptr, nullptr, nullptr, 0, n, n_heads,
                               n_heads, 0, scale, out);
  };

  // The timing launches are not part of any run (same rule as the causal
  // prefill tuner: a recorded plan must name the real kernel's grid).
  planner::PlanRecorder* saved_rec = planner::recorder();
  planner::set_thread_recorder(nullptr);
  candidate_count = int(cands.size());
  auto time_one = [&](const FullResolution& c) {
    if (!run(c)) {  // a tile that cannot launch cannot be chosen
      (void)cudaGetLastError();
      return -1.0;
    }
    const double ms = ::sonicboom::tune::median_launch_ms(
        [&] { return run(c); }, cur_stream(), 2, 5);
    if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr)
      std::fprintf(stderr, "[autotune]   full-attn cand %s %.4f ms\n",
                   full_choice_name(c).c_str(), ms);
    return ms;
  };
  // The default is timed first; it is the reference the winner must beat by
  // the switch margin.
  const double def_ms = time_one(def);
  FullResolution best = def;
  double best_ms = def_ms;
  for (size_t i = 1; i < cands.size(); ++i) {
    const double ms = time_one(cands[i]);
    if (ms > 0.0 && (best_ms <= 0.0 || ms < best_ms)) {
      best = cands[i];
      best_ms = ms;
    }
  }
  planner::set_thread_recorder(saved_rec);
  for (float* p : {q, k, v, out}) cudaFree(p);

  if (best_ms <= 0.0)
    best = def;
  else if (def_ms > 0.0 &&
           !(best_ms < def_ms * (1.0 - kFullSwitchMargin)))
    best = def;  // within the noise margin: keep the measured default
  tune_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count();
  return best;
}

// Resolve the tiled regime for the class: the env pin, else the process memo,
// else the per-device cached pick, else a tuning run, else the static default.
// `src` reports where the value came from, for the registry.
FullResolution full_resolve_tiled(int head_dim, int gqa, int n_heads,
                                  const FullOverride& ov,
                                  planner::decisions::Source& src,
                                  double& tune_s, int& candidate_count) {
  const FullResolution def = full_tiled_default(head_dim, gqa);
  src = planner::decisions::Source::Default;
  tune_s = 0.0;
  candidate_count = 0;

  if (ov.cls == FullClassName::Own || ov.cls == FullClassName::Codegen) {
    FullResolution r;
    r.cls = FullClass::Tiled;
    if (ov.cls == FullClassName::Own) {
      if (head_dim != 128) {
        full_warn_once("SONICBOOM_ATTN_CLASS=full.own has no instantiation for "
                       "this head width; using the default class");
        return def;
      }
      r.impl = FullImpl::Own;
    } else {
      if (def.impl != FullImpl::Codegen) {
        full_warn_once("SONICBOOM_ATTN_CLASS=full.codegen has no supported "
                       "tiling for this shape; using the default class");
        return def;
      }
      r.impl = FullImpl::Codegen;
      r.tile = def.tile;
    }
    src = planner::decisions::Source::Env;
    return r;
  }

  const std::string key = full_tune_key(head_dim, gqa, n_heads);
  {
    std::lock_guard<std::mutex> lk(g_full_choice_mutex);
    const auto it = g_full_choice.find(key);
    if (it != g_full_choice.end()) {
      src = it->second.src;
      tune_s = 0.0;
      return it->second.res;
    }
  }

  FullResolution choice = def;
  bool have = false;
  if (::sonicboom::tune::enabled()) {
    // Host timing is illegal inside a graph capture, so a capture that misses
    // the cache takes the static default (the class is still recorded).
    if (!attn_capturing()) {
      const std::map<std::string, int> cached =
          ::sonicboom::tune::lookup("attention-full", key);
      if (!cached.empty()) {
        if (const auto c = full_choice_from_values(cached, head_dim, gqa)) {
          choice = *c;
          have = true;
        }
      }
      if (!have) {
        choice = full_tune_tiled(head_dim, gqa, n_heads, def, tune_s,
                                 candidate_count);
        const std::map<std::string, int> values{
            {"impl", choice.impl == FullImpl::Codegen ? 1 : 0},
            {"br", choice.tile.br},
            {"bc", choice.tile.bc},
            {"as", choice.tile.as}};
        ::sonicboom::tune::store("attention-full", key, values);
      }
    }
  }
  src = have ? planner::decisions::Source::Cached
             : (candidate_count > 0 ? planner::decisions::Source::Measured
                                    : planner::decisions::Source::Default);

  if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr)
    std::fprintf(stderr,
                 "[autotune] attention-full %s %s default=%s candidates=%d "
                 "tune=%.2f s\n",
                 key.c_str(),
                 std::string(planner::decisions::source_name(src)).c_str(),
                 full_choice_name(def).c_str(), candidate_count, tune_s);

  {
    std::lock_guard<std::mutex> lk(g_full_choice_mutex);
    g_full_choice[key] = FullMemo{choice, src};
  }
  return choice;
}

// The tiled regime's tuner decision: one entry per attribute key, with the
// candidates considered, the measured probe and where the value came from.
void full_record_tune(const FullResolution& r, int head_dim, int gqa,
                      int n_heads, planner::decisions::Source src,
                      int candidate_count, double tune_s) {
  planner::decisions::Entry e;
  e.name = "tune/attention-full";
  e.scope = full_tiled_scope(head_dim, gqa);
  e.value = full_choice_name(r);
  e.source = src;
  e.env = "SONICBOOM_ATTN_CLASS";
  e.inputs = full_attn_attrs(head_dim, gqa) + " heads=" +
             std::to_string(n_heads) + " probe=" +
             std::to_string(kFullTuneTokens) + "-tokens dev=sm_86";
  std::string cands = "full.tiled:own|";
  for (const FullTile& t : full_tile_candidates(head_dim, gqa))
    cands += "full.tiled:codegen-wh:" + std::to_string(t.br) + "," +
             std::to_string(t.bc) + ",as" + std::to_string(t.as) + "|";
  e.candidates = std::move(cands);
  e.reason = "FA-2 warp-head kernel timed once per device on a fixed synthetic "
             "full-attention call; the default is kept unless a candidate wins "
             "by the switch margin";
  e.numerics =
      "f16 K/V, fp32 accumulate; deterministic per resolution, not "
      "token-identical to the hand-written kernel's S/P association";
  e.tuning_ms = int64_t(std::llround(tune_s * 1000.0));
  full_record_once(std::move(e));
}

// Resolve the whole class for one call: regime first (capacity boundary), then
// the tiled regime's implementation. `src` is the origin of the value.
FullResolution full_resolve(int n, int n_heads, int head_dim, int gqa,
                            const FullOverride& ov,
                            planner::decisions::Source& src, double& tune_s,
                            int& candidate_count) {
  bool score_gemm = n <= kFullScoreMaxTokens;
  if (ov.cls == FullClassName::ScoreGemm) score_gemm = true;
  if (ov.cls == FullClassName::Tiled || ov.cls == FullClassName::Own ||
      ov.cls == FullClassName::Codegen)
    score_gemm = false;
  if (score_gemm) {
    FullResolution r;
    r.cls = FullClass::ScoreGemm;
    r.impl = FullImpl::Own;
    src = ov.cls == FullClassName::ScoreGemm
              ? planner::decisions::Source::Env
              : planner::decisions::Source::Default;
    tune_s = 0.0;
    candidate_count = 0;
    return r;
  }
  return full_resolve_tiled(head_dim, gqa, n_heads, ov, src, tune_s,
                            candidate_count);
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
  // GQA packing: query heads per block (bit-identical; see AttentionShape).
  int pack = 1;
  // Reload Q fragments per key tile instead of holding them in registers
  // (bit-identical; see AttentionShape::q_in_reg).
  int qs = 0;
  // Head-fast grid order (bit-identical; see AttentionShape::head_fast).
  int hf = 0;
};

bool attn_same_choice(const AttnChoice &a, const AttnChoice &b) {
  if (a.impl != b.impl) return false;
  if (a.wh != b.wh) return false;
  return a.impl == AttnImpl::Own ||
         (a.br == b.br && a.bc == b.bc && a.dps == b.dps && a.as == b.as &&
          a.pack == b.pack && a.qs == b.qs && a.hf == b.hf);
}

std::string attn_choice_name(const AttnChoice &c) {
  if (c.impl == AttnImpl::Own) return "own";
  std::string n = std::string(c.wh ? "codegen-wh:" : "codegen:") +
                  std::to_string(c.br) + "," + std::to_string(c.bc) + "," +
                  std::to_string(c.dps);
  if (c.as >= 2) n += ",as" + std::to_string(c.as);
  if (c.pack > 1) n += ",p" + std::to_string(c.pack);
  if (c.qs) n += ",qs";
  if (c.hf) n += ",hf";
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
      return {AttnImpl::Codegen, s->br,          s->bc,      s->dps, false,
              s->async_stages, s->gqa_pack, s->q_in_reg ? 0 : 1,
              s->head_fast ? 1 : 0};
    return {AttnImpl::Own, 0, 0, 0};
  }
  // d256: T4 measured codegen (16x16, dps 4, pack 4, double-buffered) at
  // 0.325 ms per call at q=512 and 1.652 at q=2048 against own 0.753/3.563.
  if (gqa == 4 && head_dim == 256) {
    if (const auto s = attn_codegen_default_shape(head_dim, gqa, window))
      return {AttnImpl::Codegen, s->br,          s->bc,      s->dps, false,
              s->async_stages, s->gqa_pack, s->q_in_reg ? 0 : 1,
              s->head_fast ? 1 : 0};
  }
  // d128/gqa6 (DeepSeek-R1-Distill-Qwen-1.5B): boom-r5 M1 re-evaluation arm.
  // The generated tile is 1.92x the scalar kernel, but its S/P association
  // order moved DeepSeek's first greedy token on 10 of the r4 gate prompts.
  // Gate v2 judges the full-corpus distribution rather than 50 first tokens,
  // so the arm is measured with SONICBOOM_ATTN_D128=codegen; it becomes the
  // standard-profile default only if the KL/same-top gate admits it. Strict
  // keeps the scalar kernel either way.
  if (gqa == 6 && head_dim == 128) {
    const char *e = std::getenv("SONICBOOM_ATTN_D128");
    const bool on = e != nullptr && std::string_view(e) == "codegen";
    if (on) {
      if (const auto s = attn_codegen_default_shape(head_dim, gqa, window))
        return {AttnImpl::Codegen, s->br,          s->bc,      s->dps, false,
                s->async_stages, s->gqa_pack, s->q_in_reg ? 0 : 1,
                s->head_fast ? 1 : 0};
    }
    return {AttnImpl::Own, 0, 0, 0};
  }
  // d512: own wins the prefill leg, and no legal generated tile beats it
  // (best generated 1.717/27.19 ms vs own 1.603/20.25 at q=512/2048; see
  // attn_codegen_default_shape for why the 96 KiB budget rules out packing
  // and cp.async at this head width). The class is uniform over the query
  // length -- one kernel per shape class -- so a prefix-reuse suffix prefill
  // uses the same kernel as the full prefill it extends. The generated
  // kernel stays reachable through SONICBOOM_ATTN_IMPL=codegen for
  // measurement. The token stream moves relative to the previous generated
  // default: flagged intended in the T4 report. Every other class keeps the
  // hand-written kernel too.
  return {AttnImpl::Own, 0, 0, 0};
}

// Launch the hand-written tiled kernel for the tuned shape classes, so the
// tuner can time it with the same synthetic buffers as the generated kernel.
bool attn_launch_own(int head_dim, int gqa, const float *q,
                     const float *k_batch, const float *v_batch,
                     const float *k_cache, const float *v_cache,
                     const uint64_t *start, uint64_t n_slots, int n,
                     int n_heads_q, int n_heads_kv, uint64_t sliding_window,
                     float scale, float *out, bool kv_f16 = false,
                     bool kv_q8 = false) {
  if (gqa == 4 && head_dim == 256)
    return launch_tiled_flash_attention_kv<256, 64, 2, 32, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16, kv_q8);
  if (gqa == 4 && head_dim == 512)
    return launch_tiled_flash_attention_kv<512, 32, 4, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16, kv_q8);
  if (gqa == 7 && head_dim == 64)
    return launch_tiled_flash_attention_kv<64, 64, 2, 32, 7>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16, kv_q8);
  return false;
}

// --- Step 7: per-GPU runtime tuning -----------------------------------
//
// The class default is timed once per (shape class, GPU) and every supported
// generated (br, bc, dps) plus "own" is timed against it; the winner is
// persisted in the autotune cache (planner/autotune.h) and reused. The
// synthetic call is a causal prefill at position 0, so the search never sees
// the caller's batch size or position and a reused prefix resolves to the
// same config as a full prefill (batch invariance).
//
// Two canonical lengths are timed and their times summed for one decision per
// class. The tile ranking is not monotone in length: on sm_86 the d64 class
// favours GQA packing at 256 queries (16x32, pack 7: 0.0431 vs 0.0451 ms) and
// the default 64x16 tile at 512 (0.0973 vs 0.1075), so a 256-only tuning
// picked a tile that lost at pp512. Summing keeps one kernel per class while
// measuring it at both ends of the prefill range.
constexpr int kAttnTuneLengths[] = {256, 512};
constexpr int kAttnTuneMaxQueries = 512;
constexpr size_t kAttnTuneLengthCount =
    sizeof(kAttnTuneLengths) / sizeof(kAttnTuneLengths[0]);
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
                             const AttnChoice &def, bool kv_f16, double &tune_s,
                             int &candidate_count, bool kv_q8 = false) {
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
    // 16 warps is the largest block the register file holds (see the Q3
    // apply()); a wider candidate would only time a failed launch.
    if (compiler::attention_block_threads(s) > 512) return;
    // Only candidates whose token stream matches the hand-written kernel (or
    // the non-identical arm, when explicitly enabled) enter the search; see
    // attn_choice_bit_identical for which levers qualify per shape class.
    if (!attn_choice_bit_identical(head_dim, gqa, window, def, c)) return;
    for (const AttnChoice &x : cands)
      if (attn_same_choice(x, c)) return;
    cands.push_back(c);
  };
  // f16 KV forces synchronous staging (the generated kernel cannot cp.async a
  // mixed f16-cache / f32-batch tile), so only the as = 0 candidates exist.
  const std::vector<int> stages = (kv_f16 || kv_q8) ? std::vector<int>{0}
                                         : std::vector<int>{0, 2, 3};
  for (int as : stages)
    for (int dps : {1, 2, 4, 8})
      for (int bc : {16, 32, 48, 64})
        for (int br : {16, 32, 48, 64})
          for (int pack : {1, gqa})
            for (int qs : {0, 1}) {
              const compiler::AttentionShape s = attn_prefill_shape(
                  head_dim, gqa, window, br, bc, dps, false, as, pack,
                  qs == 0);
              add_candidate(
                  s, {AttnImpl::Codegen, br, bc, dps, false, as, pack, qs});
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
  const size_t q_n = size_t(kAttnTuneMaxQueries) * n_heads_q * head_dim;
  const size_t kv_n = size_t(kAttnTuneMaxQueries) * n_heads_kv * head_dim;
  const size_t c_n = size_t(kAttnTuneMaxQueries) * n_heads_kv * head_dim;
  // A q8 row is head_dim + 4 bytes instead of head_dim elements.
  const size_t c_bytes = kv_q8 ? size_t(kAttnTuneMaxQueries) * n_heads_kv *
                                     size_t(head_dim + 4)
                               : c_n * 4;
  bool ok = cudaMalloc(reinterpret_cast<void **>(&b.q), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kc), c_bytes) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vc), c_bytes) == cudaSuccess;
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
  cudaMemsetAsync(b.kc, 0, c_bytes, cur_stream());
  cudaMemsetAsync(b.vc, 0, c_bytes, cur_stream());
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
  auto launch = [&](const AttnChoice &c, uint64_t slots, int queries) {
    if (c.impl == AttnImpl::Own)
      return attn_launch_own(head_dim, gqa, b.q, b.kb, b.vb, b.kc, b.vc,
                             b.start, slots, queries, n_heads_q, n_heads_kv, w,
                             scale, b.out, kv_f16, kv_q8);
    compiler::AttentionShape s = attn_prefill_shape(
        head_dim, gqa, window, c.br, c.bc, c.dps, c.wh, c.as, c.pack,
        c.qs == 0, c.hf);
    s.kv_f16 = kv_f16;
    s.kv_q8 = kv_q8;
    return attn_launch_codegen(s, b.q, b.kb, b.vb, b.kc, b.vc, b.start, slots,
                               queries, n_heads_q, n_heads_kv, w, scale, b.out);
  };
  // The sum over the canonical lengths; a candidate that cannot launch at one
  // of them cannot be chosen (-1).
  auto time_choice = [&](const AttnChoice &c) {
    double total = 0.0;
    for (int n : kAttnTuneLengths) {
      const double ms = ::sonicboom::tune::median_launch_ms(
          [&] { return launch(c, uint64_t(n), n); }, cur_stream(), 2, 5);
      if (!(ms > 0.0)) return -1.0;
      total += ms;
    }
    return total;
  };

  candidate_count = int(cands.size());
  // The timing launches are not part of any run: the first execution of a
  // prefill may be the one being recorded, and a recorded plan must name the
  // chosen kernel's real grid, not the tuning pass's synthetic candidates
  // (same rule as the dense tuner).
  planner::PlanRecorder *saved_rec = planner::recorder();
  planner::set_thread_recorder(nullptr);
  const double def_ms = time_choice(def);
  AttnChoice best = def;
  double best_ms = def_ms;
  for (size_t i = 1; i < cands.size(); ++i) {
    const double ms = time_choice(cands[i]);
    if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr)
      std::fprintf(stderr,
                   "[autotune]   attn cand %s %.4f ms summed over lengths "
                   "(best %.4f)\n",
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

  // The decision inputs of the class's static default: the tile knobs plus
  // the packing/grid order the candidates inherit from it. Part of the cache
  // key, so changing the default retimes the class instead of leaving a pick
  // measured against the old reference in place. The query length is not.
  const std::string prod_key =
      "b" + std::to_string(base.br) + "x" + std::to_string(base.bc) + "-dps" +
      std::to_string(base.dps) + "-as" + std::to_string(base.async_stages) +
      "-p" + std::to_string(base.gqa_pack) + (base.head_fast ? "-hf" : "") +
      (base.q_in_reg ? "" : "-qs");

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
    // The generator's support rules allow 32-warp blocks, but this device's
    // register file does not hold them: at 24-32 warps a block needs more
    // registers than the SM's 64K (measured: 768-thread d512 and 1024-thread
    // d256 pack4 cuLaunchKernel back with CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES).
    // 16 warps is the largest block that launches, so a stored choice from an
    // older grid cannot turn a class into a silent hand-written fallback.
    if (compiler::attention_block_threads(s) > 512) return std::nullopt;
    return s;
  };
  // The dispatcher's own shape: the bit-identity reference and the default we
  // keep unless a candidate clearly beats it.
  const TileConfig production{base.br, head_dim / base.dps, base.bc,
                              base.async_stages, 1};

  const mlir_tune::AttnPrefillChoice choice =
      mlir_tune::attn_prefill_choice(head_dim, n_heads_q, prod_key);
  if (choice.resolved()) return apply(choice.cfg);
  if (attn_capturing()) return std::nullopt;  // host timing is illegal there
  // Out of cold budget: keep the dispatcher's shape without allocating the
  // probe buffers (the choice function memoizes a Skipped record, so this is
  // reached once per class).
  if (mlir_tune::budget_exhausted()) return std::nullopt;

  // Probe at a fixed query length, not the caller's: the class key has no
  // query length (one kernel per class, R1), so the first caller's n must not
  // decide what every later launch of the class runs. 512 rows is a prefill-
  // sized burst (the class is the prefill kernel) and keeps the scratch small.
  constexpr int kQ3ProbeQueries = 512;
  // Fixed synthetic cache extent, unchanged by the d64 workload fix above: the
  // Q3 runner only ranks tile shapes against each other and its picks are
  // gated bit-identical, so changing its extent would be an unrelated retune.
  constexpr uint64_t kQ3ProbeSlots = 256;
  const int probe_n = kQ3ProbeQueries;
  const size_t q_n = size_t(probe_n) * size_t(n_heads_q) * size_t(head_dim);
  const size_t kv_n = size_t(probe_n) * size_t(n_heads_kv) * size_t(head_dim);
  const size_t c_n = size_t(kQ3ProbeSlots) * size_t(n_heads_kv) *
                     size_t(head_dim);
  // A q8 row is head_dim + 4 bytes instead of head_dim elements.
  const size_t c_bytes = base.kv_q8
                             ? size_t(kQ3ProbeSlots) * size_t(n_heads_kv) *
                                   size_t(head_dim + 4)
                             : c_n * 4;
  AttnTuneBufs b;
  bool ok = cudaMalloc(reinterpret_cast<void **>(&b.q), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vb), kv_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.kc), c_bytes) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.vc), c_bytes) == cudaSuccess;
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
  cudaMemsetAsync(b.kc, 0, c_bytes, cur_stream());
  cudaMemsetAsync(b.vc, 0, c_bytes, cur_stream());
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
                                  kQ3ProbeSlots, probe_n, n_heads_q,
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
      mlir_tune::attn_prefill_choice_store(head_dim, n_heads_q, prod_key,
                                          winner);
  if (verbose)
    std::fprintf(stderr,
                 "[autotune] %s: br=%d dc=%d bc=%d as=%d%s\n",
                 ("attn_prefill/" +
                  mlir_tune::attn_prefill_shape_class(head_dim, n_heads_q,
                                                      prod_key))
                     .c_str(),
                 winner.tile_m, winner.tile_n, winner.tile_k, winner.unroll,
                 stored ? "" : " [not persisted]");
  return apply(winner);
}

std::mutex g_attn_choice_mutex;
std::unordered_map<std::string, AttnChoice> g_attn_choice;

std::string attn_tune_key(int head_dim, int gqa, int n_heads_q, int n_heads_kv,
                          bool window, uint64_t window_len, bool kv_f16,
                          bool kv_q8, const AttnChoice &def) {
  std::string k = compiler::attention_codegen_version();
  k += "|prefill-d" + std::to_string(head_dim) + "-g" + std::to_string(gqa) +
       "-h" + std::to_string(n_heads_q) + "-kv" + std::to_string(n_heads_kv) +
       "-m" + (window ? "window" : "causal");
  if (window) k += "-w" + std::to_string(window_len);
  // The f16-KV variant is a distinct generated kernel (it widens __half and
  // stores f16), so its tile ranking is its own. The marker is appended only
  // under the standard profile, which keeps every strict key byte-identical to
  // the pre-profile ones.
  if (kv_f16) k += "-kv16";
  if (kv_q8) k += "-kv8";
  if (attn_exp2_enabled()) k += "-x2";
  // The default the choice was measured against is a decision input: a new
  // default (a different tile, packing or grid order) must retune the class
  // rather than be shadowed by the old pick. The batch size and position are
  // still never part of the key.
  k += "|dflt-" + attn_choice_name(def);
  // The tuning workload is a decision input too: the d64 class ranks its
  // candidates differently at 256 and 512 queries, so a pick measured over a
  // different set of lengths must not shadow the retune.
  k += "|wl";
  for (size_t i = 0; i < kAttnTuneLengthCount; ++i) {
    if (i != 0) k += "+";
    k += std::to_string(kAttnTuneLengths[i]);
  }
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
  int br = 0, bc = 0, dps = 0, wh = 0, as = 0, pack = 1, qs = 0, hf = 0;
  if (!get("br", br) || !get("bc", bc) || !get("dps", dps)) return false;
  (void)get("wh", wh);
  (void)get("as", as);
  (void)get("pack", pack);
  (void)get("qs", qs);
  (void)get("hf", hf);
  const compiler::AttentionShape s = attn_prefill_shape(
      head_dim, gqa, window, br, bc, dps, wh != 0, as, pack, qs == 0,
      hf);
  if (!compiler::attention_shape_supported(s)) return false;
  out = {AttnImpl::Codegen, br, bc, dps, wh != 0, as, pack, qs, hf};
  return true;
}

// Resolve the prefill implementation for a shape class: cached choice, else a
// tuning run on synthetic inputs, else the static default. The key is the
// shape class (head count included) and the generated-code version; it never
// includes the batch size or position.
AttnChoice attn_resolve(int head_dim, int gqa, int n_heads_q, int n_heads_kv,
                        bool window, uint64_t window_len, bool kv_f16,
                        bool kv_q8 = false) {
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

  const std::string key = attn_tune_key(head_dim, gqa, n_heads_q, n_heads_kv,
                                        window, window_len, kv_f16, kv_q8,
                                        def);
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
                               window, def, kv_f16, tune_s, candidate_count,
                               kv_q8);
    std::map<std::string, int> values{
        {"impl", choice.impl == AttnImpl::Codegen ? 1 : 0},
        {"br", choice.br},
        {"bc", choice.bc},
        {"dps", choice.dps},
        {"wh", choice.wh ? 1 : 0},
        {"as", choice.as},
        {"pack", choice.pack},
        {"hf", choice.hf},
        {"qs", choice.qs}};
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

// --- Decode split-KV schedule: the chunk size C ---------------------------
//
// The schedule is (C, combine order). C is a per-(device, shape class)
// constant and is NOT derived from the window length or the cache capacity:
// chunk k covers the ABSOLUTE key positions [k*C, (k+1)*C), and the combine
// walks the grid in ascending absolute chunk index. A key therefore always
// belongs to the same chunk, so every query's reduction tree is a function of
// key positions alone -- independent of depth, n_ctx capacity, and how much of
// the context came from a reused prefix. That is what makes a decode at depth
// N bit-identical whether it followed a full prefill or a prefix reuse at any
// capacity, and it keeps R1 (one fixed split per class) honest.
//
// The number of splits still adapts: the grid is sized from the capacity
// (floor(window/C) + 1 chunks) so it covers any position, and chunks beyond
// the current key count write a neutral partial. Larger C => fewer, longer
// splits; smaller C => more parallelism.
//
// C is timed once per class on synthetic KV of the class's shape (same
// head_dim/GQA/mask/kv type, zeroed data) at the depths the workload covers,
// and the pick minimizes the workload-weighted expected step time. The default
// weights favour long context: short depths are dominated by the weight
// stream and barely see C, while at long depth attention is a large share of
// the step. SONICBOOM_ATTN_DECODE_WEIGHTS overrides the weights.
constexpr uint64_t kAttnDecodeTuneChunks[] = {8, 16, 32, 64, 128, 256};
// Depths the pick is measured at. The last tier is the model's context
// capacity, capped so a cold tune stays inside the cold budget (the ring is
// only 512 slots, so a longer extent costs DRAM replay, not memory).
constexpr uint64_t kAttnDecodeTuneDepths[] = {1024, 4096, 16384};
constexpr uint64_t kAttnDecodeTuneCapacityCap = 32768;
constexpr double kAttnDecodeTuneWeights[] = {0.10, 0.20, 0.30, 0.40};

std::mutex g_attn_decode_key_mutex;
std::unordered_map<std::string, uint64_t> g_attn_decode_chunk;

// The class default: the generated and hand-written decoders share it, which
// keeps them token-identical. For a windowed class the window is a model
// property (the SWA ring), so deriving the split from it is class-fixed; a
// causal class uses kMaxChunk whatever its capacity.
uint64_t attn_decode_class_chunk(int n_heads_kv, uint64_t sliding_window) {
  if (sliding_window == 0) return kMaxChunk;
  const uint64_t target_chunks =
      std::max<uint64_t>(1, (uint64_t)sm_count() * 4 / (uint64_t)n_heads_kv);
  return std::max<uint64_t>(
      1, std::min((sliding_window + target_chunks - 1) / target_chunks,
                  kMaxChunk));
}

uint64_t attn_decode_requested_chunk(uint64_t chunk_attr, int n_heads_kv,
                                     uint64_t sliding_window) {
  if (chunk_attr != 0) return chunk_attr;
  if (const char *e = std::getenv("SONICBOOM_ATTN_DECODE_CHUNK")) {
    const long v = std::atol(e);
    if (v > 0 && (uint64_t)v <= (uint64_t)compiler::kAttentionDecodeMaxChunk)
      return (uint64_t)v;
  }
  return attn_decode_class_chunk(n_heads_kv, sliding_window);
}

} // namespace

// Public so the model plans can write the schedule into the Boom IR attention
// op (the chunk size and merge order are plan-visible attributes, not a
// dispatch-time surprise).
uint64_t attn_decode_plan_chunk(int head_dim, int gqa, int n_heads_q,
                                int n_heads_kv, uint64_t sliding_window,
                                uint64_t capacity, bool kv_f16, bool kv_q8) {
  const uint64_t default_chunk = attn_decode_class_chunk(n_heads_kv, sliding_window);
  compiler::AttentionShape dsh;
  dsh.head_dim = head_dim;
  dsh.gqa = gqa;
  dsh.window_mask = sliding_window != 0;
  dsh.decode = true;
  dsh.kv_f16 = kv_f16;
  dsh.kv_q8 = kv_q8;
  dsh.exp2 = attn_exp2_enabled();
  if (!compiler::attention_shape_supported(dsh)) return default_chunk;

  // Class key: no capacity and no position. A pick is valid for every n_ctx.
  std::string key = compiler::attention_codegen_version();
  key += "|decode-d" + std::to_string(head_dim) + "-g" + std::to_string(gqa) +
         "-h" + std::to_string(n_heads_q) + "-kv" +
         std::to_string(n_heads_kv) + "-m" +
         (sliding_window != 0 ? "window" : "causal") +
         (kv_f16 ? "-kv16" : kv_q8 ? "-kv8" : "-kv32");
  if (attn_exp2_enabled()) key += "-x2";

  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  const auto remember = [&](uint64_t c) {
    std::lock_guard<std::mutex> lk(g_attn_decode_key_mutex);
    g_attn_decode_chunk[key] = c;
    return c;
  };
  const auto record = [&](uint64_t c, planner::decisions::Source src,
                          const std::string &inputs, const std::string &reason,
                          int64_t tuning_ms) {
    planner::decisions::Entry e;
    e.name = "attention.decode-chunk";
    e.scope = key;
    e.value = std::to_string(c);
    e.source = src;
    e.env = "SONICBOOM_ATTN_DECODE_CHUNK";
    e.inputs = inputs;
    e.candidates = "8|16|32|64|128|256";
    e.reason = reason;
    e.numerics =
        "chunk boundaries at absolute key positions; fixed per class, "
        "independent of depth/capacity/prefix reuse";
    e.tuning_ms = tuning_ms;
    planner::decisions::record(std::move(e));
  };

  {
    std::lock_guard<std::mutex> lk(g_attn_decode_key_mutex);
    const auto it = g_attn_decode_chunk.find(key);
    if (it != g_attn_decode_chunk.end()) return it->second;
  }

  if (const char *e = std::getenv("SONICBOOM_ATTN_DECODE_CHUNK")) {
    const long v = std::atol(e);
    if (v > 0 && (uint64_t)v <= (uint64_t)compiler::kAttentionDecodeMaxChunk) {
      const uint64_t c = (uint64_t)v;
      record(c, planner::decisions::Source::Env, "env override",
             "SONICBOOM_ATTN_DECODE_CHUNK set", 0);
      return remember(c);
    }
  }

  if (const std::map<std::string, int> cached =
          ::sonicboom::tune::lookup("attention-decode", key);
      !cached.empty()) {
    const auto it = cached.find("chunk");
    if (it != cached.end() && it->second > 0 &&
        (uint64_t)it->second <= (uint64_t)compiler::kAttentionDecodeMaxChunk) {
      const uint64_t c = (uint64_t)it->second;
      record(c, planner::decisions::Source::Cached,
             "chunk=" + std::to_string(c) + " (per-device cache)",
             "cached pick", 0);
      if (verbose)
        std::fprintf(stderr, "[autotune] attention-decode %s cached=chunk %llu\n",
                     key.c_str(), (unsigned long long)c);
      return remember(c);
    }
  }
  if (attn_capturing() || !::sonicboom::tune::enabled()) {
    record(default_chunk, planner::decisions::Source::Default,
           "no measurement", "class default", 0);
    return remember(default_chunk);
  }
  // A plan being recorded (the plan golden / plan dump path) must be a
  // function of the plan, not of a measurement taken on whatever device is
  // recording it: the goldens pin the static class default. Normal runs still
  // tune; the default is not remembered for this process, so a plan built
  // after recording in the same process still tunes.
  if (planner::recorder() != nullptr) {
    record(default_chunk, planner::decisions::Source::Default, "plan recording",
           "static class default while a plan record is installed", 0);
    return default_chunk;
  }

  // Measurement tiers: 1K, 4K, 16K, and the model capacity (capped). A tier at
  // or below the previous one is dropped so short-capacity models are not
  // measured twice at the same extent.
  std::vector<uint64_t> depths;
  const uint64_t cap =
      std::max<uint64_t>(1, std::min<uint64_t>(capacity, kAttnDecodeTuneCapacityCap));
  for (uint64_t d : kAttnDecodeTuneDepths)
    if (depths.empty() || d > depths.back()) depths.push_back(d);
  if (depths.empty() || cap > depths.back()) depths.push_back(cap);

  // The synthetic KV is sized to the deepest tier, not to a small ring: a
  // 512-slot ring times every depth with the cache resident in L2, so the pick
  // would be blind to the DRAM traffic that dominates a real long context. One
  // buffer per class; the worst case (Gemma's global class: 2 KV heads x 512
  // dims x 32K keys at f16) is ~67 MiB per side. A sliding-window class touches
  // at most window+1 distinct keys at any depth, so its buffer stops there —
  // that ring is the real working set, not a cache-residency shortcut.
  const uint64_t slots = std::max<uint64_t>(
      1, sliding_window ? std::min<uint64_t>(depths.back(), sliding_window + 1)
                        : depths.back());
  const size_t q_n = size_t(n_heads_q) * head_dim;
  const size_t kv_elems = size_t(slots) * n_heads_kv * head_dim;
  const size_t kv_bytes =
      kv_q8 ? size_t(slots) * n_heads_kv * (head_dim + 4)
            : kv_elems * (kv_f16 ? 2 : 4);
  AttnTuneBufs b;
  void *kc_raw = nullptr, *vc_raw = nullptr;
  bool ok =
      cudaMalloc(reinterpret_cast<void **>(&b.q), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(&kc_raw, kv_bytes) == cudaSuccess;
  ok = ok && cudaMalloc(&vc_raw, kv_bytes) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.out), q_n * 4) == cudaSuccess;
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&b.start), sizeof(uint64_t)) ==
                 cudaSuccess;
  // Partial scratch: the largest grid any candidate produces at the deepest
  // tier (the smallest candidate).
  uint64_t max_chunks = 0;
  for (uint64_t c : kAttnDecodeTuneChunks)
    max_chunks = std::max<uint64_t>(max_chunks, depths.back() / c + 1);
  float *partial = nullptr;
  const size_t partial_floats =
      size_t(max_chunks) * size_t(n_heads_q) * size_t(head_dim + 2);
  ok = ok && cudaMalloc(reinterpret_cast<void **>(&partial),
                        partial_floats * 4) == cudaSuccess;
  if (!ok) {
    (void)cudaGetLastError();
    if (partial) cudaFree(partial);
    if (kc_raw) cudaFree(kc_raw);
    if (vc_raw) cudaFree(vc_raw);
    record(default_chunk, planner::decisions::Source::Default,
           "scratch allocation failed", "class default", 0);
    return remember(default_chunk);
  }
  // Zeroed inputs: data-independent work, no denormals in the timing.
  cudaMemsetAsync(b.q, 0, q_n * 4, cur_stream());
  cudaMemsetAsync(kc_raw, 0, kv_bytes, cur_stream());
  cudaMemsetAsync(vc_raw, 0, kv_bytes, cur_stream());
  cudaMemsetAsync(b.out, 0, q_n * 4, cur_stream());
  cudaMemsetAsync(partial, 0, partial_floats * 4, cur_stream());

  const float *kca = reinterpret_cast<const float *>(kc_raw);
  const float *vca = reinterpret_cast<const float *>(vc_raw);
  float *out_d = b.out;
  uint64_t *start_d = b.start;

  auto launch = [&](uint64_t c, uint64_t extent) {
    // The grid a real decode launch would use at this depth: the window is the
    // sliding window when there is one, else the cache extent (n_ctx). Sizing
    // an SWA class from the extent would launch thousands of blocks that exit
    // on their first compare, which is not what the model pays.
    const uint64_t win = sliding_window ? sliding_window : extent;
    const uint64_t n_chunks = (win - 1) / c + 2;
    const uint64_t *startp = start_d;
    uint64_t nslots_arg = slots;
    int nhq_arg = n_heads_q;
    int nhkv_arg = n_heads_kv;
    uint64_t win_arg = sliding_window;
    float scale_arg = 1.0f / std::sqrt(float(head_dim));
    uint64_t chunk_arg = c;
    float *out_arg = out_d;
    int nch_arg = int(n_chunks);
    const float *q_arg = b.q;
    void *args[13] = {&q_arg,      &kca,      &vca,       &startp,
                      &nslots_arg, &nhq_arg, &nhkv_arg,  &win_arg,
                      &scale_arg,  &partial, &chunk_arg, &out_arg,
                      &nch_arg};
    return planner::glue::launch_attention_decode(dsh, args, int(n_chunks),
                                                  n_heads_q);
  };
  const auto time_at = [&](uint64_t c, uint64_t extent) {
    const uint64_t pos_host = extent - 1;  // one full extent of valid keys
    if (cudaMemcpyAsync(start_d, &pos_host, sizeof(pos_host),
                        cudaMemcpyHostToDevice, cur_stream()) != cudaSuccess) {
      (void)cudaGetLastError();
      return -1.0;
    }
    return ::sonicboom::tune::median_launch_ms(
        [&] { return launch(c, extent); }, cur_stream(), 2, 5);
  };

  // Weights: default favour long context; an env override is comma separated.
  std::vector<double> weights;
  if (const char *e = std::getenv("SONICBOOM_ATTN_DECODE_WEIGHTS")) {
    std::string s(e), item;
    for (char ch : s + ",") {
      if (ch == ',') {
        if (!item.empty()) weights.push_back(std::atof(item.c_str()));
        item.clear();
      } else {
        item.push_back(ch);
      }
    }
  }
  if (weights.size() != depths.size()) {
    weights.assign(depths.size(), 0.0);
    for (size_t i = 0; i < depths.size(); ++i)
      weights[i] = kAttnDecodeTuneWeights[std::min<size_t>(
          i, sizeof(kAttnDecodeTuneWeights) / sizeof(double) - 1)];
  }

  // Analytic prune: a candidate whose smallest tier cannot fill the device
  // with blocks is not measured (it cannot beat a wider grid).
  const uint64_t sms = (uint64_t)sm_count();
  std::vector<uint64_t> cands;
  for (uint64_t c : kAttnDecodeTuneChunks) {
    if (c > (uint64_t)compiler::kAttentionDecodeMaxChunk) continue;
    const uint64_t blocks = (depths.front() / c + 1) * (uint64_t)n_heads_q;
    if (c == default_chunk || blocks >= sms) cands.push_back(c);
  }
  if (std::find(cands.begin(), cands.end(), default_chunk) == cands.end())
    cands.insert(cands.begin(), default_chunk);

  // Same recorder rule as the prefill tuner: a decode being planned must not
  // carry the chunk candidates' launches.
  planner::PlanRecorder *saved_rec = planner::recorder();
  planner::set_thread_recorder(nullptr);
  const auto t0 = std::chrono::steady_clock::now();
  double best_score = -1.0;
  uint64_t best = default_chunk;
  std::vector<std::pair<uint64_t, std::string>> timings;  // "C:ms,ms,..."
  double total_w = 0.0;
  for (double w : weights) total_w += w;
  for (uint64_t c : cands) {
    double score = 0.0;
    bool valid = true;
    std::string row = std::to_string(c) + ":";
    for (size_t i = 0; i < depths.size(); ++i) {
      const double ms = time_at(c, depths[i]);
      if (ms < 0.0) {
        valid = false;
        break;
      }
      score += (weights[i] / (total_w > 0.0 ? total_w : 1.0)) * ms;
      row += (i ? "," : "") + std::to_string(ms);
    }
    if (!valid) continue;
    timings.emplace_back(c, row);
    if (best_score < 0.0 || score < best_score) {
      best_score = score;
      best = c;
    }
  }
  const double def_score = [&] {
    for (const auto &t : timings)
      if (t.first == default_chunk) {
        // Recompute is cheaper than storing a parallel score: the row holds
        // the per-tier times.
        std::string s = t.second.substr(t.second.find(':') + 1);
        double sc = 0.0, wsum = 0.0;
        size_t i = 0;
        std::string item;
        for (char ch : s + ",") {
          if (ch == ',') {
            if (i < depths.size() && i < weights.size()) {
              sc += weights[i] * std::atof(item.c_str());
              wsum += weights[i];
            }
            ++i;
            item.clear();
          } else {
            item.push_back(ch);
          }
        }
        return wsum > 0.0 ? sc / wsum : -1.0;
      }
    return -1.0;
  }();
  // Keep the default unless a candidate wins by the switch margin. A different
  // C is a different (still position-only) reduction tree, so the margin also
  // keeps a noisy measurement from moving the blessed split.
  if (best_score < 0.0 || def_score < 0.0 ||
      !(best_score < def_score * (1.0 - kAttnSwitchMargin)))
    best = default_chunk;
  planner::set_thread_recorder(saved_rec);
  const double tune_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  cudaFree(partial);
  cudaFree(kc_raw);
  cudaFree(vc_raw);

  ::sonicboom::tune::store("attention-decode", key, {{"chunk", int(best)}});
  std::string inputs = "depths=";
  for (size_t i = 0; i < depths.size(); ++i)
    inputs += (i ? "|" : "") + std::to_string(depths[i]);
  inputs += " weights=";
  for (size_t i = 0; i < weights.size(); ++i)
    inputs += (i ? "|" : "") + std::to_string(weights[i]);
  inputs += " device=" + std::to_string(int(sms)) + "SMs";
  std::string cand_times;
  for (const auto &t : timings) cand_times += t.second + "|";
  record(best, planner::decisions::Source::Measured, inputs,
         "min weighted mean decode-attention ms at synthetic depths; " +
             cand_times + " default kept unless beaten by the switch margin",
         int64_t(std::llround(tune_s * 1000.0)));
  if (verbose)
    std::fprintf(stderr,
                 "[autotune] attention-decode %s default_chunk=%llu "
                 "chosen_chunk=%llu score=%.4f default_score=%.4f tune=%.2f s\n",
                 key.c_str(), (unsigned long long)default_chunk,
                 (unsigned long long)best, best_score, def_score, tune_s);
  return remember(best);
}

// Dense full attention (no mask, no window, no KV cache): every query attends
// to every key in [0, n). The op's attributes (causal 0, kv_cache absent,
// prefix_reuse impossible, window 0) are the class key; the class and its
// implementation are resolved through the decision registry (full_resolve) and
// recorded as `attention.class` / `tune/attention-full`. No model name and no
// caller identity participates in the choice.
//
// The score-GEMM regime returns false when the shape or the scratch does not
// fit (the caller then falls back to its own scalar kernel); every other
// regime falls back to the hand-written tiled kernel instead. false means no
// class of this op can run, never "this caller is the DiT".
bool flash_attention_full_dev(const float* q, const float* k, const float* v,
                              int n, int n_heads, int head_dim, float scale,
                              float* out) {
  if (!available()) return false;
  if (n <= 0 || n_heads <= 0 || head_dim <= 0) return false;
  if (!q || !k || !v || !out) return false;
  if (n_heads > 32) return false;  // one warp per 16-row chunk, 16-warps cap

  // This op has one K/V head per query head: GQA 1 is an attribute of the
  // class, not an assumption about the caller.
  const int gqa = 1;
  const FullOverride ov = full_class_override();
  if (ov.cls == FullClassName::Scalar) return false;

  planner::decisions::Source src = planner::decisions::Source::Default;
  double tune_s = 0.0;
  int candidate_count = 0;
  FullResolution r = full_resolve(n, n_heads, head_dim, gqa, ov, src, tune_s,
                                  candidate_count);
  bool fell_back = false;

  if (r.cls == FullClass::ScoreGemm) {
    const int vram = int(full_score_scratch_bytes(n, n_heads, head_dim));
    if (full_attention_score_gemm(q, k, v, n, n_heads, head_dim, scale, out)) {
      full_record_class(r, head_dim, gqa, n_heads, vram, /*fallback=*/false,
                        src);
      return true;
    }
    // Pinned to the score-GEMM regime: do not silently take another class.
    if (ov.cls == FullClassName::ScoreGemm) return false;
    // The shape/scratch cannot serve this regime; the tiled regime is the
    // class's remaining implementation.
    full_warn_once(
        "score-GEMM regime unavailable at this shape; using the tiled "
        "full-attention class");
    src = planner::decisions::Source::Default;
    tune_s = 0.0;
    candidate_count = 0;
    r = full_resolve_tiled(head_dim, gqa, n_heads, ov, src, tune_s,
                           candidate_count);
    fell_back = true;
  }

  full_record_tune(r, head_dim, gqa, n_heads, src, candidate_count, tune_s);
  FullImpl ran = FullImpl::Own;
  if (!full_attn_launch(r, head_dim, gqa, q, k, v, n, n_heads, scale, out,
                        ran))
    return false;
  if (ran != r.impl) {
    // The generated kernel did not launch: record what actually ran, under the
    // same scope, so the registry counts the deviation as a conflict.
    r.impl = ran;
    r.tile = FullTile{0, 0, 0};
  }
  full_record_class(r, head_dim, gqa, n_heads, 0, fell_back, src);
  return true;
}

void flash_attention_full_release_scratch() { release_full_scratch(); }

bool flash_attention_chunk_dev_pos(const float* q, const float* k_batch,
                                   const float* v_batch, const float* k_cache,
                                   const float* v_cache, const uint64_t* start,
                                   uint64_t n_slots, int n, int n_heads_q,
                                   int n_heads_kv, int head_dim,
                                   uint64_t sliding_window, float scale,
                                   float* out, bool kv_f16, bool kv_q8) {
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

  // R10 Stage B: strict-order shared-memory tiled kernel for the d128/gqa6
  // class (window-less causal KV-cache prefill). It reproduces the scalar
  // kernel's FP associations — the differential test compares it bit-for-bit
  // at every required length, both KV dtypes and with/without a cached
  // prefix — so it is a pure performance route, not a numerical one. Opt-in
  // until the benchmark gates pass; any other shape, or a launch failure,
  // falls through to the scalar kernel.
  if (head_dim == 128 && gqa == 6 && n_heads_q == 12 && n_heads_kv == 2 &&
      sliding_window == 0 && !kv_q8 && attn_own_d128_exact()) {
    const bool ok =
        kv_f16 ? launch_exact_flash_attention<__half, 128, 12, 2>(
                     q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
                     sliding_window, scale, out)
               : launch_exact_flash_attention<float, 128, 12, 2>(
                     q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
                     sliding_window, scale, out);
    if (ok) {
      const size_t smem =
          (2 * FA_BK * 128 + kExactNQ * 12 * FA_BK) * sizeof(float);
      attn_trace_prefill(attn_trace_read_start(start), n, head_dim, gqa,
                         sliding_window,
                         kv_f16 ? "own-tiled-exact-kv16" : "own-tiled-exact",
                         "smem/head", 1, FA_BK, n, n_heads_q, 12 * 32, smem);
      return true;
    }
  }

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
  auto with_kv = [&](std::optional<compiler::AttentionShape> s) {
    if (s) {
      s->kv_f16 = kv_f16;
      s->kv_q8 = kv_q8;
    }
    return s;
  };
  if (attn_impl == "codegen") {
    shape = with_kv(attn_codegen_shape(head_dim, gqa, sliding_window != 0));
  } else if (attn_impl == "auto") {
    const bool windowed = sliding_window != 0;
    if (std::getenv("SONICBOOM_ATTN_CODEGEN") != nullptr &&
        attn_default_choice(head_dim, gqa, windowed).impl == AttnImpl::Codegen) {
      // Explicit tile pin (the Stage 3c sweep): codegen at that config. The
      // pin does not move a class whose measured default is own.
      shape = with_kv(attn_codegen_shape(head_dim, gqa, windowed));
    } else {
      const AttnChoice c =
          attn_resolve(head_dim, gqa, n_heads_q, n_heads_kv, windowed,
                       sliding_window, kv_f16, kv_q8);
      if (c.impl == AttnImpl::Codegen) {
        compiler::AttentionShape s = attn_prefill_shape(
            head_dim, gqa, windowed, c.br, c.bc, c.dps, c.wh, c.as, c.pack,
            c.qs == 0, c.hf);
        s.kv_f16 = kv_f16;
        s.kv_q8 = kv_q8;
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
  // hand-written kernels below. A shape that compiles but cannot launch (a
  // tile whose register demand exceeds the block's budget) lands here too,
  // which would otherwise be invisible in a timing run: say so once.
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
      if (shape->gqa_pack > 1) cfg += ",p" + std::to_string(shape->gqa_pack);
      if (shape->head_fast) cfg += ",hf";
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
  if (shape) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true))
      std::fprintf(stderr,
                   "[attention] generated prefill kernel b%dx%d-dps%d-p%d%s "
                   "did not launch; falling back to the hand-written kernel\n",
                   shape->br, shape->bc, shape->dps, shape->gqa_pack,
                   shape->head_fast ? "-hf" : "");
  }

  if (gqa == 4 && head_dim == 256) {
    const bool ok = launch_tiled_flash_attention_kv<256, 64, 2, 32, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16, kv_q8);
    if (ok)
      attn_trace_own<256, 64, 2, 32, 4>(start, n, n_heads_kv, head_dim, gqa,
                           sliding_window);
    return ok;
  }
  if (gqa == 4 && head_dim == 512) {
    const bool ok = launch_tiled_flash_attention_kv<512, 32, 4, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16, kv_q8);
    if (ok)
      attn_trace_own<512, 32, 4, 16, 4>(start, n, n_heads_kv, head_dim, gqa,
                           sliding_window);
    return ok;
  }
  if (gqa == 7 && head_dim == 64) {
    const bool ok = launch_tiled_flash_attention_kv<64, 64, 2, 32, 7>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16, kv_q8);
    if (ok)
      attn_trace_own<64, 64, 2, 32, 7>(start, n, n_heads_kv, head_dim, gqa,
                           sliding_window);
    return ok;
  }

  const size_t smem = (size_t)n_heads_q * FA_BK * sizeof(float);
  SB_NOTE_LAUNCH(flash_attention_chunk_pos_kernel, n, n_heads_q * 32, smem, cur_stream());
  if (kv_f16)
    flash_attention_chunk_pos_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
        q, k_batch, v_batch, reinterpret_cast<const __half*>(k_cache),
        reinterpret_cast<const __half*>(v_cache), start, n_slots, n_heads_q,
        n_heads_kv, head_dim, sliding_window, scale, out);
  else if (kv_q8)
    flash_attention_chunk_pos_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
        q, k_batch, v_batch, reinterpret_cast<const int8_t*>(k_cache),
        reinterpret_cast<const int8_t*>(v_cache), start, n_slots, n_heads_q,
        n_heads_kv, head_dim, sliding_window, scale, out);
  else
    flash_attention_chunk_pos_kernel<<<n, n_heads_q * 32, smem, cur_stream()>>>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n_heads_q,
        n_heads_kv, head_dim, sliding_window, scale, out);
  const bool ok_scalar = cudaGetLastError() == cudaSuccess;
  if (ok_scalar)
    attn_trace_prefill(attn_trace_read_start(start), n, head_dim, gqa,
                       sliding_window,
                       kv_f16 ? "own-scalar-kv16"
                              : kv_q8 ? "own-scalar-kv8" : "own-scalar",
                       "warp/head", 1, FA_BK, n, n_heads_q, n_heads_q * 32,
                       smem);
  return ok_scalar;
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

bool argmax_masked_dev(const float* x, int n, const uint8_t* mask, float* out) {
  if (!available()) return false;
  if (n <= 0 || !x || !mask || !out) return false;
  constexpr int kThreads = 256;
  const int n4 = n >> 2;
  int grid = (n4 + kThreads - 1) / kThreads;
  if (grid > sm_count() * 8) grid = sm_count() * 8;
  if (grid < 2 || (reinterpret_cast<uintptr_t>(x) & 15u) != 0) {
    SB_NOTE_LAUNCH(argmax_masked_kernel, 1, 1024, 0, cur_stream());
    argmax_masked_kernel<<<1, 1024, 0, cur_stream()>>>(x, mask, out, n);
    return cudaGetLastError() == cudaSuccess;
  }
  unsigned long long* keys = static_cast<unsigned long long*>(grow_scratch(
      g_argmax_keys, g_argmax_keys_bytes,
      (size_t)grid * sizeof(unsigned long long)));
  if (!keys) return false;
  SB_NOTE_LAUNCH(argmax_partial_masked_kernel, grid, kThreads, 0, cur_stream());
  argmax_partial_masked_kernel<<<grid, kThreads, 0, cur_stream()>>>(x, mask, n, keys);
  SB_NOTE_LAUNCH(argmax_finalize_kernel, 1, 256, 0, cur_stream());
  argmax_finalize_kernel<<<1, 256, 0, cur_stream()>>>(keys, grid, out);
  return cudaGetLastError() == cudaSuccess;
}

namespace {

// Shared launcher for the direct and captured-graph forms. The key scratch is
// allocated (never freed) before the launch, so a capture always sees a buffer
// that is large enough.
bool sample_select_launch(const float* logits, int vocab, const uint8_t* mask,
                          const int32_t* window, const SampleSelectParams* dyn,
                          int window_len, float temperature,
                          float repeat_penalty, float presence, float frequency,
                          Candidate* out, int* out_count) {
  if (!available()) return false;
  if (vocab <= 0 || !logits || !out || !out_count) return false;
  unsigned long long* keys = static_cast<unsigned long long*>(
      grow_scratch_keep(g_select_keys, g_select_keys_bytes,
                        size_t(vocab) * sizeof(unsigned long long)));
  if (!keys) return false;
  SB_NOTE_LAUNCH(sample_select_kernel, 1, kSelectThreads, 0, cur_stream());
  sample_select_kernel<<<1, kSelectThreads, 0, cur_stream()>>>(
      logits, vocab, mask, window, dyn, window_len, temperature, repeat_penalty,
      presence, frequency, keys, out, out_count);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool sample_select_dev(const float* logits, int vocab, const uint8_t* mask,
                       const int32_t* window, int window_len,
                       float temperature, float repeat_penalty, float presence,
                       float frequency, Candidate* out, int* out_count) {
  if (window_len < 0 || window_len > kSampleSelectMaxWindow) return false;
  if (window_len > 0 && !window) return false;
  if (!(temperature > 0.0f)) return false;
  return sample_select_launch(logits, vocab, mask, window, nullptr, window_len,
                              temperature, repeat_penalty, presence, frequency,
                              out, out_count);
}

bool sample_select_dev_dyn(const float* logits, int vocab, const uint8_t* mask,
                           const int32_t* window,
                           const SampleSelectParams* params_dev,
                           Candidate* out, int* out_count) {
  if (!params_dev || !window) return false;
  return sample_select_launch(logits, vocab, mask, window, params_dev, 0, 0.0f,
                              0.0f, 0.0f, 0.0f, out, out_count);
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
// regardless of n or column offset. This is the ablation/default arm of
// matvec_f32_cublas_dev below, which prefers the full-batch cuBLASLt arm.
bool matvec_f32_sgemm_chunked(const float* d_w, int cols, const float* x,
                              float* y, int rows, int ncols, bool strided) {
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

// --- f32 dense: full-batch cuBLASLt with a per-class algorithm --------------
//
// The f32 dense projection as one cuBLASLt matmul over every token, instead of
// the fixed 128-column cublasSgemm chunks of matvec_f32_sgemm_chunked. The
// chunking existed to make a token's numerics independent of the batch size:
// every call had the same (m=rows, n=128, k=cols) shape, so cuBLAS selected
// the same kernel and reduced in the same order for a one-token suffix and the
// segment of a 512-token prefill. A fixed Lt algorithm gives the same
// guarantee without the chunking -- the algorithm is chosen once per
// (rows, cols) shape class and every batch size runs it, so a row's reduction
// does not depend on n (measured on the 3050: n=130 vs n=512, 0/33280 and
// 0/332800 elements differ for the two PLE shapes) -- and it removes the
// per-128-token launch, the split-K reduce kernel, and the tail
// pad/copyback passes.
//
// The algorithm is measured, not assumed: the heuristic's candidates are timed
// against the chunked arm at the canonical 512-token chunk, once per shape
// class per device, and the winner goes to the tune store (op "dense-f32") like
// every other tuner choice. A candidate is only eligible if its fp64-reference
// error does not exceed the chunked arm's, so a faster reduction order with
// worse numerics cannot displace the current arm.
// SONICBOOM_AUTOTUNE=0 keeps the chunked arm (the pre-T5 path; the static
// default); SONICBOOM_DENSE_F32=lt|sgemm|tf32 pins one arm.
//
// TF32 (CUBLAS_COMPUTE_32F_FAST_TF32) is a measurement arm only, reachable
// through SONICBOOM_DENSE_F32=tf32; it is not admitted to either profile. On
// the reference 3050 it is 1.5-2x the fp32 arm on both PLE shape classes
// (0.100 vs 0.205 ms at m256k2560, 0.099 vs 0.154 at m2560k256) and worth
// +1.8% on Gemma pp512 (interleaved, 3 runs). It is not a default because the
// f32 dense arm is one of the near-ties the standard gate resolves by bit
// identity: the gate run that admitted TF32 flipped one greedy token on gemma
// prompt 3, and a control with the fp32 arm pinned flipped the same token, so
// the flip was not attributable to TF32 and the arm never got a clean gate of
// its own. Admit it again only with a gate run that isolates it (see the
// boom-r4 M5 notes).
struct DenseF32Lt {
  cublasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
  bool valid = false;
  // TF32 tensor-core arm; reachable only through the SONICBOOM_DENSE_F32=tf32
  // pin, never part of the tuned list (see dense_f32_resolve).
  bool tf32 = false;
};

// -1 unset (tuned), 0 = sgemm chunks, 1 = fp32 Lt, 2 = Lt/TF32 (pinned arms).
int dense_f32_pin() {
  static const int v = [] {
    const char *e = std::getenv("SONICBOOM_DENSE_F32");
    if (!e) return -1;
    const std::string_view s(e);
    if (s == "sgemm") return 0;
    if (s == "tf32") return 2;
    return 1;
  }();
  return v;
}

// A: W (cols x rows column-major, OP_T); B: x (cols x n, OP_N); C/D: y
// (rows x n). Same formulation and layouts as the chunked arm.
bool dense_f32_lt_descs(cublasLtMatmulDesc_t *op, cublasLtMatrixLayout_t *ad,
                        cublasLtMatrixLayout_t *bd,
                        cublasLtMatrixLayout_t *cd, int rows, int cols,
                        int ncols, bool tf32 = false) {
  const cublasComputeType_t compute =
      tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
  if (cublasLtMatmulDescCreate(op, compute, CUDA_R_32F) !=
      CUBLAS_STATUS_SUCCESS)
    return false;
  const cublasOperation_t tr_a = CUBLAS_OP_T, tr_b = CUBLAS_OP_N;
  return cublasLtMatmulDescSetAttribute(*op, CUBLASLT_MATMUL_DESC_TRANSA,
                                        &tr_a, sizeof(tr_a)) ==
             CUBLAS_STATUS_SUCCESS &&
         cublasLtMatmulDescSetAttribute(*op, CUBLASLT_MATMUL_DESC_TRANSB,
                                        &tr_b, sizeof(tr_b)) ==
             CUBLAS_STATUS_SUCCESS &&
         cublasLtMatrixLayoutCreate(ad, CUDA_R_32F, cols, rows, cols) ==
             CUBLAS_STATUS_SUCCESS &&
         cublasLtMatrixLayoutCreate(bd, CUDA_R_32F, cols, ncols, cols) ==
             CUBLAS_STATUS_SUCCESS &&
         cublasLtMatrixLayoutCreate(cd, CUDA_R_32F, rows, ncols, rows) ==
             CUBLAS_STATUS_SUCCESS;
}

bool dense_f32_lt_launch(const DenseF32Lt &k, const float *d_w, int cols,
                         const float *x, float *y, int rows, int ncols) {
  cublasLtHandle_t lt = dense_lt_handle();
  if (!lt || !k.valid) return false;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
  bool ok =
      dense_f32_lt_descs(&op, &ad, &bd, &cd, rows, cols, ncols, k.tf32);
  if (ok) {
    const float alpha = 1.0f, beta = 0.0f;
    ok = cublasLtMatmul(lt, op, &alpha, d_w, ad, x, bd, &beta, y, cd, y, cd,
                        &k.algo, g_dense_ws, k.workspace, cur_stream()) ==
         CUBLAS_STATUS_SUCCESS;
  }
  if (ad) cublasLtMatrixLayoutDestroy(ad);
  if (bd) cublasLtMatrixLayoutDestroy(bd);
  if (cd) cublasLtMatrixLayoutDestroy(cd);
  if (op) cublasLtMatmulDescDestroy(op);
  return ok;
}

// The heuristic's candidates for the class, built at the canonical chunk. The
// order is the heuristic's timing model order and is stable for a given
// device/driver, which is what lets the tune store remember a pick as an index.
std::vector<DenseF32Lt> dense_f32_lt_candidates(cublasLtHandle_t lt, int rows,
                                                int cols, bool tf32 = false) {
  std::vector<DenseF32Lt> out;
  if (!g_dense_ws) return out;
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t ad = nullptr, bd = nullptr, cd = nullptr;
  if (!dense_f32_lt_descs(&op, &ad, &bd, &cd, rows, cols, kDenseTuneTokens,
                          tf32)) {
    if (ad) cublasLtMatrixLayoutDestroy(ad);
    if (bd) cublasLtMatrixLayoutDestroy(bd);
    if (cd) cublasLtMatrixLayoutDestroy(cd);
    if (op) cublasLtMatmulDescDestroy(op);
    return out;
  }
  cublasLtMatmulPreference_t pref = nullptr;
  if (cublasLtMatmulPreferenceCreate(&pref) == CUBLAS_STATUS_SUCCESS) {
    size_t max_ws = kDenseCublasWorkspaceBytes;
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_ws, sizeof(max_ws));
    cublasLtMatmulHeuristicResult_t heurs[8];
    int found = 0;
    if (cublasLtMatmulAlgoGetHeuristic(lt, op, ad, bd, cd, cd, pref, 8, heurs,
                                       &found) == CUBLAS_STATUS_SUCCESS) {
      for (int i = 0; i < found; ++i) {
        if (heurs[i].state != CUBLAS_STATUS_SUCCESS ||
            heurs[i].workspaceSize > kDenseCublasWorkspaceBytes)
          continue;
        out.push_back({heurs[i].algo, heurs[i].workspaceSize, true, tf32});
      }
    }
    cublasLtMatmulPreferenceDestroy(pref);
  }
  if (ad) cublasLtMatrixLayoutDestroy(ad);
  if (bd) cublasLtMatrixLayoutDestroy(bd);
  if (cd) cublasLtMatrixLayoutDestroy(cd);
  if (op) cublasLtMatmulDescDestroy(op);
  return out;
}

struct DenseF32Choice {
  int impl = 1;  // 0 = full-batch Lt, 1 = chunked cublasSgemm (pre-T5)
  DenseF32Lt lt;
};

// Deterministic operand fill for the tuning run: timing does not depend on the
// values, but the accuracy gate needs non-zero operands, and a fixed pattern
// keeps the tuning repeatable.
std::vector<float> dense_f32_tune_operands(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  for (float &f : v) {
    seed ^= seed << 13;
    seed ^= seed >> 17;
    seed ^= seed << 5;
    f = float(seed % 2000) / 1000.0f - 1.0f;
  }
  return v;
}

__global__ void f32_to_f64_tune_kernel(const float *src, double *dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = (double)src[i];
}

struct DenseF32Err {
  double maxabs = 0.0;
  double rms = 0.0;
};

// fp64 reference for the tuning operands, fetched to host (token-major). Built
// with a DGEMM on double copies of w/x, so the error of an arm is measured
// against the same reference the report's numerics table uses.
bool dense_f32_tune_reference(const float *w, const float *x, int rows,
                              int cols, std::vector<double> &ref) {
  const size_t w_n = size_t(rows) * size_t(cols);
  const size_t x_n = size_t(cols) * size_t(kDenseTuneTokens);
  const size_t y_n = size_t(rows) * size_t(kDenseTuneTokens);
  double *dw = nullptr, *dx = nullptr, *dy = nullptr;
  bool ok = cudaMalloc(&dw, w_n * 8) == cudaSuccess &&
            cudaMalloc(&dx, x_n * 8) == cudaSuccess &&
            cudaMalloc(&dy, y_n * 8) == cudaSuccess;
  if (ok) {
    const int tpb = 256;
    f32_to_f64_tune_kernel<<<int((w_n + tpb - 1) / tpb), tpb, 0, cur_stream()>>>(
        w, dw, int(w_n));
    f32_to_f64_tune_kernel<<<int((x_n + tpb - 1) / tpb), tpb, 0, cur_stream()>>>(
        x, dx, int(x_n));
    const double one = 1.0, zero = 0.0;
    ok = cublasDgemm(dense_cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, rows,
                     kDenseTuneTokens, cols, &one, dw, cols, dx, cols, &zero,
                     dy, rows) == CUBLAS_STATUS_SUCCESS;
  }
  if (ok) {
    ref.resize(y_n);
    cudaStreamSynchronize(cur_stream());
    ok = cudaMemcpy(ref.data(), dy, y_n * 8, cudaMemcpyDeviceToHost) ==
         cudaSuccess;
  }
  if (dy) cudaFree(dy);
  if (dx) cudaFree(dx);
  if (dw) cudaFree(dw);
  return ok;
}

// Error of the arm whose result is currently in y, against the reference.
DenseF32Err dense_f32_error(const float *y, size_t y_n,
                            const std::vector<double> &ref) {
  DenseF32Err e;
  std::vector<float> yh(y_n);
  cudaStreamSynchronize(cur_stream());
  if (cudaMemcpy(yh.data(), y, y_n * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
    return e;
  double acc = 0.0;
  for (size_t i = 0; i < y_n; ++i) {
    const double d = std::fabs(double(yh[i]) - ref[i]);
    if (d > e.maxabs) e.maxabs = d;
    acc += d * d;
  }
  e.rms = std::sqrt(acc / double(y_n));
  return e;
}

// Time both arms at the canonical 512-token chunk and pick the fastest among
// those that are no less accurate than the chunked arm. A Lt candidate only has
// to measure faster than the chunked arm: that arm pays a launch per
// kDenseCublasWidth column block plus the split-K reduction and tail passes,
// so replacing it with one call removes both the kernel time it measures and
// the launches it costs (pp512: 7 launches per PLE projection). A later Lt
// candidate must still beat the current best by kDenseCandidateMargin, so the
// pick among Lt candidates — the ones whose times sit within noise of each
// other — stays stable across cold tunes.
//
// Accuracy gate: the fp64-reference error of the chosen arm must not exceed
// the chunked arm's (maxabs and rms both). The fastest Lt candidate is not
// necessarily the one that wins: e.g. on the 3050 the down-projection winner
// by time reduces in a different order and is ~1.85x the chunked arm's rms,
// while the next candidate is both faster than the chunked arm and slightly
// more accurate. A pinned Lt run (the ablation) skips the gate.
DenseF32Choice dense_f32_tune(int rows, int cols,
                              const std::vector<DenseF32Lt> &cands,
                              bool lt_only, double &tune_s,
                              int &candidate_count) {
  DenseF32Choice best;
  candidate_count = 0;
  const size_t w_n = size_t(rows) * size_t(cols);
  const size_t x_n = size_t(cols) * size_t(kDenseTuneTokens);
  const size_t y_n = size_t(rows) * size_t(kDenseTuneTokens);
  const auto t0 = std::chrono::steady_clock::now();
  float *w = nullptr, *x = nullptr, *y = nullptr;
  if (cudaMalloc(&w, w_n * 4) != cudaSuccess) return best;
  if (cudaMalloc(&x, x_n * 4) != cudaSuccess) {
    cudaFree(w);
    return best;
  }
  if (cudaMalloc(&y, y_n * 4) != cudaSuccess) {
    cudaFree(x);
    cudaFree(w);
    return best;
  }
  const std::vector<float> w_h = dense_f32_tune_operands(w_n, 12345);
  const std::vector<float> x_h = dense_f32_tune_operands(x_n, 67890);
  cudaMemcpyAsync(w, w_h.data(), w_n * 4, cudaMemcpyHostToDevice,
                  cur_stream());
  cudaMemcpyAsync(x, x_h.data(), x_n * 4, cudaMemcpyHostToDevice,
                  cur_stream());
  cudaMemsetAsync(y, 0, y_n * 4, cur_stream());
  std::vector<double> ref;
  if (!dense_f32_tune_reference(w, x, rows, cols, ref)) ref.clear();
  planner::PlanRecorder *saved = planner::recorder();
  planner::set_thread_recorder(nullptr);
  // With lt_only (a pinned Lt measurement) the chunked arm is not the
  // incumbent and the fastest Lt candidate wins without a margin.
  double best_ms = -1.0;
  DenseF32Err base_err;
  if (!lt_only)
    best_ms = ::sonicboom::tune::median_launch_ms(
        [&] {
          return matvec_f32_sgemm_chunked(w, cols, x, y, rows, kDenseTuneTokens,
                                          true);
        },
        cur_stream(), 2, 5);
  if (!lt_only && !ref.empty()) base_err = dense_f32_error(y, y_n, ref);
  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  if (verbose && !lt_only)
    std::fprintf(stderr,
                 "[autotune] dense-f32 m=%d k=%d n=%d sgemm=%.4f ms "
                 "maxabs=%.3e rms=%.3e\n",
                 rows, cols, kDenseTuneTokens, best_ms, base_err.maxabs,
                 base_err.rms);
  bool best_is_lt = false;
  for (size_t i = 0; i < cands.size(); ++i) {
    ++candidate_count;
    // A longer sample than the other tuners': the contenders here sit within a
    // few percent of the incumbent and the pick flips the plan record, so the
    // decision must not follow the clock-ramp of the first milliseconds.
    const double ms = ::sonicboom::tune::median_launch_ms(
        [&] {
          return dense_f32_lt_launch(cands[i], w, cols, x, y, rows,
                                     kDenseTuneTokens);
        },
        cur_stream(), 4, 9);
    DenseF32Err err;
    if (!ref.empty()) err = dense_f32_error(y, y_n, ref);
    const double margin = best_is_lt ? kDenseCandidateMargin : 0.0;
    const bool faster =
        ms > 0.0 && (best_ms < 0.0 || ms < best_ms * (1.0 - margin));
    // Without a reference (allocation failed) fall back to the time-only pick.
    // A TF32 candidate is not comparable to the fp32 incumbent's error (it
    // rounds the operands to 10 bits), so it is exempt; TF32 only appears in
    // the pinned measurement list, which skips the gate anyway.
    const bool accurate =
        lt_only || ref.empty() || cands[i].tf32 ||
        (err.maxabs <= base_err.maxabs && err.rms <= base_err.rms);
    if (verbose)
      std::fprintf(stderr,
                   "[autotune]   lt[%zu]%s ws=%zu = %.4f ms maxabs=%.3e "
                   "rms=%.3e%s\n",
                   i, cands[i].tf32 ? " tf32" : "", cands[i].workspace, ms,
                   err.maxabs, err.rms,
                   !faster   ? " (not faster)"
                   : !accurate ? " [rejected: less accurate than sgemm]"
                               : "");
    if (!faster || !accurate) continue;
    best = {0, cands[i]};
    best_ms = ms;
    best_is_lt = true;
  }
  planner::set_thread_recorder(saved);
  tune_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
          .count();
  cudaFree(y);
  cudaFree(x);
  cudaFree(w);
  return best;
}

// Resolve the f32 dense arm for a (rows, cols) shape class: env pin, cached
// pick, else a tuning run, else the chunked arm. The key never includes the
// batch size. Tuning is skipped inside a capture (host timing is illegal
// there) and the class is left unmemoized so a later launch can decide it.
DenseF32Choice dense_f32_resolve(int rows, int cols) {
  const DenseF32Choice def;  // chunked
  const int pin = dense_f32_pin();
  if (pin == 0) return def;
  if (!::sonicboom::tune::enabled() && pin < 0) return def;
  cublasLtHandle_t lt = dense_lt_handle();
  if (!lt || !g_dense_ws) return def;

  // The trailing number is the policy version: T9 stopped requiring a Lt
  // candidate to beat the chunked arm by kDenseCandidateMargin, so a pick
  // cached under the old rule (e.g. "down stays chunked") is not a decision
  // this code would make. Bumping it retimes once per class per device. The
  // pinned TF32 measurement arm gets its own key so it never becomes the
  // class's decision.
  const std::string key =
      pin == 2
          ? "dense-f32-tf32-pin|m" + std::to_string(rows) + "k" +
                std::to_string(cols)
          : "dense-f32-3|m" + std::to_string(rows) + "k" +
                std::to_string(cols);
  // The heuristic's order is stable, so the cached (impl, algo) pair indexes
  // this list. The list is the fp32 candidates for the tuned decision; the TF32
  // list exists only under the pin.
  const auto build_cands = [&] {
    return dense_f32_lt_candidates(lt, rows, cols);
  };
  static std::mutex mu;
  static std::unordered_map<std::string, DenseF32Choice> memo;
  {
    std::lock_guard<std::mutex> lk(mu);
    const auto it = memo.find(key);
    if (it != memo.end()) return it->second;
  }

  // A pinned TF32 measurement (the ablation): TF32-only list, time-only pick,
  // never persisted -- it is not the class's decision and works under either
  // profile so the two arms can be compared directly.
  if (pin == 2) {
    const std::vector<DenseF32Lt> c =
        dense_f32_lt_candidates(lt, rows, cols, true);
    if (c.empty() || dense_capturing()) return def;
    double tune_s = 0.0;
    int candidate_count = 0;
    dense_f32_tune(rows, cols, c, true, tune_s, candidate_count);
    if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE"))
      std::fprintf(stderr,
                   "[autotune] dense-f32 m=%dk%d tf32-only candidates=%d "
                   "tune=%.2f s [pinned]\n",
                   rows, cols, candidate_count, tune_s);
    const DenseF32Choice pinned{0, c.front()};
    std::lock_guard<std::mutex> lk(mu);
    memo[key] = pinned;
    return pinned;
  }

  DenseF32Choice choice = def;
  bool have = false;
  if (pin < 0) {
    const std::map<std::string, int> cached =
        ::sonicboom::tune::lookup("dense-f32", key);
    const auto impl = cached.find("impl");
    const auto algo = cached.find("algo");
    if (impl != cached.end() && impl->second == 1) {
      have = true;  // the chunked arm, measured as the winner
    } else if (impl != cached.end() && impl->second == 0 &&
               algo != cached.end()) {
      const std::vector<DenseF32Lt> cands = build_cands();
      if (algo->second >= 0 && algo->second < int(cands.size())) {
        choice = {0, cands[algo->second]};
        have = true;
      }
    }
  }

  if (!have) {
    if (dense_capturing()) return def;  // decide on a later, uncaptured launch
    std::vector<DenseF32Lt> cands = build_cands();
    if (cands.empty()) {
      std::lock_guard<std::mutex> lk(mu);
      memo[key] = choice;
      return choice;
    }
    double tune_s = 0.0;
    int candidate_count = 0;
    choice = dense_f32_tune(rows, cols, cands, pin == 1, tune_s,
                            candidate_count);
    if (pin == 1 && choice.impl != 0) {
      // The pin forces the fp32 Lt arm (the ablation); a pinned measurement is
      // not the class's decision, so it is not persisted.
      choice = DenseF32Choice{0, cands.front()};
    }
    // The winner is an index into the candidate list: the cached record
    // remembers the pick, never the opaque algorithm bytes. The compute type is
    // part of the identity -- the heuristic can name the same algorithm for the
    // fp32 and TF32 lists.
    int idx = 0;
    if (choice.impl == 0)
      for (size_t i = 0; i < cands.size(); ++i)
        if (std::memcmp(&cands[i].algo, &choice.lt.algo,
                        sizeof(choice.lt.algo)) == 0 &&
            cands[i].tf32 == choice.lt.tf32)
          idx = int(i);
    if (pin < 0) {
      std::map<std::string, int> values{{"impl", choice.impl},
                                        {"algo", idx},
                                        {"tf32", choice.lt.tf32 ? 1 : 0}};
      ::sonicboom::tune::store("dense-f32", key, values);
    }
    if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE"))
      std::fprintf(stderr,
                   "[autotune] dense-f32 %s default=sgemm chosen=%s "
                   "algo=%d candidates=%d tune=%.2f s%s\n",
                   key.c_str(),
                   choice.impl == 0 ? (choice.lt.tf32 ? "lt/tf32" : "lt")
                                    : "sgemm",
                   idx,
                   candidate_count, tune_s, pin < 0 ? "" : " [pinned]");
  }

  {
    std::lock_guard<std::mutex> lk(mu);
    memo[key] = choice;
  }
  return choice;
}

bool matvec_f32_cublas_dev(const float* d_w, int cols, const float* x, float* y,
                           int rows, int ncols, bool strided) {
  cublasHandle_t h = dense_cublas_handle();
  if (!h) return false;
  // The full-batch arm reads the position-major activation in place; the
  // non-strided layout keeps the chunked repack below.
  if (strided) {
    const DenseF32Choice c = dense_f32_resolve(rows, cols);
    if (c.impl == 0 && dense_f32_lt_launch(c.lt, d_w, cols, x, y, rows, ncols))
      return true;
  }
  return matvec_f32_sgemm_chunked(d_w, cols, x, y, rows, ncols, strided);
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
                           uint64_t n_slots, int row_elems, int n,
                           bool dst_f16, bool dst_q8, int head_dim) {
  if (!available()) return false;
  if (row_elems <= 0 || n <= 0 || !dst_k || !dst_v || !src_k || !src_v ||
      n_slots == 0)
    return false;
  if (dst_q8) {
    if (head_dim <= 0 || row_elems % head_dim != 0) return false;
    const int nhkv = row_elems / head_dim;
    const int grid = n * nhkv;
    SB_NOTE_LAUNCH(kv_append_batched_q8_kernel, grid, 128, 0, cur_stream());
    kv_append_batched_q8_kernel<<<grid, 128, 0, cur_stream()>>>(
        reinterpret_cast<int8_t*>(dst_k), src_k, start_pos, n_slots, nhkv,
        head_dim, n);
    SB_NOTE_LAUNCH(kv_append_batched_q8_kernel, grid, 128, 0, cur_stream());
    kv_append_batched_q8_kernel<<<grid, 128, 0, cur_stream()>>>(
        reinterpret_cast<int8_t*>(dst_v), src_v, start_pos, n_slots, nhkv,
        head_dim, n);
    return cudaGetLastError() == cudaSuccess;
  }
  const int grid = grid_for(row_elems * n);
  SB_NOTE_LAUNCH(kv_append_batched_kernel, grid, 256, 0, cur_stream());
  if (dst_f16) {
    kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
        reinterpret_cast<__half*>(dst_k), src_k, start_pos, n_slots, row_elems,
        n);
    SB_NOTE_LAUNCH(kv_append_batched_kernel, grid, 256, 0, cur_stream());
    kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
        reinterpret_cast<__half*>(dst_v), src_v, start_pos, n_slots, row_elems,
        n);
  } else {
    kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
        dst_k, src_k, start_pos, n_slots, row_elems, n);
    SB_NOTE_LAUNCH(kv_append_batched_kernel, grid, 256, 0, cur_stream());
    kv_append_batched_kernel<<<grid, 256, 0, cur_stream()>>>(
        dst_v, src_v, start_pos, n_slots, row_elems, n);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool kv_append_batched_dev_pos(float* dst_k, float* dst_v, const float* src_k,
                               const float* src_v, const uint64_t* start_pos,
                               uint64_t n_slots, int row_elems, int n,
                               bool dst_f16, bool dst_q8, int head_dim) {
  if (!available()) return false;
  if (row_elems <= 0 || n <= 0 || !dst_k || !dst_v || !src_k || !src_v ||
      !start_pos || n_slots == 0)
    return false;
  if (dst_q8) {
    if (head_dim <= 0 || row_elems % head_dim != 0) return false;
    const int nhkv = row_elems / head_dim;
    const int grid = n * nhkv;
    SB_NOTE_LAUNCH(kv_append_batched_pos_q8_kernel, grid, 128, 0,
                   cur_stream());
    kv_append_batched_pos_q8_kernel<<<grid, 128, 0, cur_stream()>>>(
        reinterpret_cast<int8_t*>(dst_k), src_k, start_pos, n_slots, nhkv,
        head_dim, n);
    SB_NOTE_LAUNCH(kv_append_batched_pos_q8_kernel, grid, 128, 0,
                   cur_stream());
    kv_append_batched_pos_q8_kernel<<<grid, 128, 0, cur_stream()>>>(
        reinterpret_cast<int8_t*>(dst_v), src_v, start_pos, n_slots, nhkv,
        head_dim, n);
    return cudaGetLastError() == cudaSuccess;
  }
  const int grid = grid_for(row_elems * n);
  SB_NOTE_LAUNCH(kv_append_batched_pos_kernel, grid, 256, 0, cur_stream());
  if (dst_f16) {
    kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
        reinterpret_cast<__half*>(dst_k), src_k, start_pos, n_slots, row_elems,
        n);
    SB_NOTE_LAUNCH(kv_append_batched_pos_kernel, grid, 256, 0, cur_stream());
    kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
        reinterpret_cast<__half*>(dst_v), src_v, start_pos, n_slots, row_elems,
        n);
  } else {
    kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
        dst_k, src_k, start_pos, n_slots, row_elems, n);
    SB_NOTE_LAUNCH(kv_append_batched_pos_kernel, grid, 256, 0, cur_stream());
    kv_append_batched_pos_kernel<<<grid, 256, 0, cur_stream()>>>(
        dst_v, src_v, start_pos, n_slots, row_elems, n);
  }
  return cudaGetLastError() == cudaSuccess;
}

} // namespace sonicboom::nn::cuda
