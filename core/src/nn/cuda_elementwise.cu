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

// Single-thread sum-of-squares, accumulated in double in ascending index order.
// This reproduces the CPU rms_norm's sequential double sum (norm.cpp) bit-for-bit,
// unlike a parallel fp32 atomicAdd whose rounding differs enough to flip the
// argmax on the sensitive per-layer gate.
__global__ void rms_sum_sq_kernel(const float* __restrict__ x, int n,
                                  double* __restrict__ out) {
  double s = 0.0;
  for (int i = 0; i < n; ++i) {
    const float v = x[i];
    s += double(v * v);
  }
  *out = s;
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

// Incremental decode attention (mirrors nn::decode_attention): one thread per
// query head, a scalar loop over the valid key slots. Three passes recompute the
// q·k dot so no variable-size scores array is needed (the dot is deterministic,
// so each pass yields identical scores — bit-identical to the CPU kernel's
// single-pass-plus-storage, up to expf vs std::exp rounding).
__global__ void decode_attention_kernel(const float* __restrict__ q,
                                        const float* __restrict__ k_cache,
                                        const float* __restrict__ v_cache,
                                        uint64_t pos, uint64_t n_slots,
                                        int n_heads_q, int n_heads_kv,
                                        int head_dim, uint64_t sliding_window,
                                        float scale, float* __restrict__ out) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= n_heads_q) return;
  const int kv_h = h * n_heads_kv / n_heads_q;
  const uint64_t start =
      (sliding_window > 0 && pos >= sliding_window) ? pos - sliding_window + 1 : 0;
  const uint64_t n_keys = pos - start + 1;
  const float* qrow = q + (uint64_t)h * head_dim;

  // Pass 1: max score (scores are unit-norm cosine similarities ~ [-1,1]).
  float max_score = -1e30f;
  for (uint64_t idx = 0; idx < n_keys; ++idx) {
    const uint64_t slot = (start + idx) % n_slots;
    const float* krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
    max_score = fmaxf(max_score, dot * scale);
  }

  // Pass 2: sum of exp(score - max).
  float sum = 0.0f;
  for (uint64_t idx = 0; idx < n_keys; ++idx) {
    const uint64_t slot = (start + idx) % n_slots;
    const float* krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
    sum += expf(dot * scale - max_score);
  }
  const float inv = 1.0f / sum;

  // Pass 3: weighted sum of V.
  float* orow = out + (uint64_t)h * head_dim;
  for (int d = 0; d < head_dim; ++d) orow[d] = 0.0f;
  for (uint64_t idx = 0; idx < n_keys; ++idx) {
    const uint64_t slot = (start + idx) % n_slots;
    const float* krow = k_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    const float* vrow = v_cache + (slot * (uint64_t)n_heads_kv + kv_h) * head_dim;
    float dot = 0.0f;
    for (int d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
    const float prob = expf(dot * scale - max_score) * inv;
    for (int d = 0; d < head_dim; ++d) orow[d] += prob * vrow[d];
  }
}

// --- device state (scratch + weight cache) ----------------------------------

struct DeviceWeight {
  void* ptr = nullptr;
  size_t bytes = 0;
};

std::unordered_map<const void*, DeviceWeight> g_cache;
std::mutex g_cache_mutex;

void* g_dx = nullptr;
size_t g_dx_bytes = 0;
void* g_db = nullptr;
size_t g_db_bytes = 0;
void* g_dy = nullptr;
size_t g_dy_bytes = 0;
void* g_dff = nullptr;
size_t g_dff_bytes = 0;
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
  auto it = g_cache.find(host);
  if (it != g_cache.end() && it->second.bytes == bytes) return it->second.ptr;

  void* dptr = nullptr;
  if (cudaMalloc(&dptr, bytes) != cudaSuccess) return nullptr;
  if (cudaMemcpy(dptr, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dptr);
    return nullptr;
  }
  g_cache.emplace(host, DeviceWeight{dptr, bytes});
  return dptr;
}

inline int grid_for(int n, int block = 256) { return (n + block - 1) / block; }

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

  rms_sum_sq_kernel<<<1, 1>>>(static_cast<const float*>(d_x), int(n), g_sum);
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
  gemv_dense_f32<<<grid_for(int(rows)), 256>>>(static_cast<const float*>(d_w),
                                               static_cast<const float*>(d_x),
                                               static_cast<float*>(d_y), int(cols),
                                               int(rows));
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
  gemv_bf16<<<grid_for(int(rows)), 256>>>(static_cast<const uint16_t*>(d_w),
                                          static_cast<const float*>(d_x),
                                          static_cast<float*>(d_y), int(cols), int(rows));
  if (cudaGetLastError() != cudaSuccess) return false;
  return cudaMemcpy(y.data(), d_y, rows * sizeof(float), cudaMemcpyDeviceToHost) ==
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
  rms_sum_sq_kernel<<<1, 1>>>(x, n, g_sum);
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
  gemv_dense_f32<<<grid_for(rows), 256>>>(d_w, x, y, cols, rows);
  return cudaGetLastError() == cudaSuccess;
}

bool matvec_bf16_dev(std::span<const uint16_t> W, int cols, const float* x,
                     float* y, int rows) {
  if (!available()) return false;
  if (cols <= 0 || rows <= 0 || !x || !y) return false;
  if (W.size() < (size_t)cols * (size_t)rows) return false;
  const uint16_t* d_w = static_cast<const uint16_t*>(upload_bytes(W.data(), (size_t)cols * rows * sizeof(uint16_t)));
  if (!d_w) return false;
  gemv_bf16<<<grid_for(rows), 256>>>(d_w, x, y, cols, rows);
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
  decode_attention_kernel<<<grid_for(n_heads_q), 256>>>(
      q, k_cache, v_cache, pos, n_slots, n_heads_q, n_heads_kv, head_dim,
      sliding_window, scale, out);
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

} // namespace sonicboom::nn::cuda
