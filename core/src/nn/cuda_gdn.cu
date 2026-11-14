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

// The gated DeltaNet scan, CUDA (nn/cuda_gdn.h).
//
// Two launches per node, whatever n is:
//   gdn_prep   (a) the convolution over the last K raw rows and SiLU, one
//              thread per (row, channel); (b) the split; (c) the per-key-head
//              normalization, one warp per (row, key head); (d) the gates.
//              Then the histogram is rewritten with the last K-1 raw rows of
//              (old hist ++ batch), by whichever block finishes last: every
//              block's history reads happen before it raises the counter.
//   gdn_recur  (e) the memory update: one warp per (h, j), lane l holding
//              M[h][j][i] for i = r*32 + l in registers, block (32, 4),
//              grid (heads_v, dim/4); the warp walks the rows in order.
//
// The reductions are fixed: a lane sums its dim/32 products in ascending r,
// then the warp reduces with __shfl_xor_sync at 16, 8, 4, 2, 1. No fast-math
// is used, so a row's result does not depend on n or the launch shape.

#include <sonicboom/nn/cuda_gdn.h>

#include <cuda_runtime.h>

#include "cuda_stream.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <utility>

namespace sonicboom::nn::cuda {

namespace {

constexpr int kPrepThreads = 256;
constexpr float kMaxSharedBytes = 96 * 1024;  // sm_86's opt-in dynamic limit

// --- weight upload cache ----------------------------------------------------

struct DeviceWeight {
  void* ptr = nullptr;
  size_t bytes = 0;
};

std::mutex g_mutex;
std::map<std::pair<const void*, size_t>, DeviceWeight> g_weights;

// Upload `w` once, cached by (data pointer, byte size), like the f32 weight
// path in cuda_elementwise.cu. Returns nullptr on a device failure.
const float* upload_f32(std::span<const float> w) {
  if (w.empty()) return nullptr;
  const std::pair<const void*, size_t> key{w.data(),
                                           w.size() * sizeof(float)};
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_weights.find(key);
  if (it != g_weights.end()) return static_cast<const float*>(it->second.ptr);
  void* dptr = nullptr;
  if (cudaMalloc(&dptr, key.second) != cudaSuccess) return nullptr;
  if (cudaMemcpy(dptr, w.data(), key.second, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    cudaFree(dptr);
    return nullptr;
  }
  g_weights.emplace(key, DeviceWeight{dptr, key.second});
  return static_cast<const float*>(dptr);
}

// --- per-call scratch -------------------------------------------------------

struct Scratch {
  void* ptr = nullptr;
  size_t bytes = 0;
};

Scratch g_scratch[6];  // qn, kn, v, beta, eg, the finished-block counter

// Grow a scratch buffer to at least `need` bytes. The old buffer is kept (a
// graph that was captured against it, if any, stays valid) and the process
// runs one model at a time, so this is bounded by the model's geometry.
void* grow_scratch(int slot, size_t need) {
  Scratch& s = g_scratch[slot];
  if (s.bytes >= need) return s.ptr;
  void* p = nullptr;
  if (cudaMalloc(&p, need) != cudaSuccess) return nullptr;
  s.ptr = p;
  s.bytes = need;
  return p;
}

// --- the scan ---------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

// The raw row of the scan at `p` positions before this batch's row 0: negative
// positions come from the incoming histogram (row -1 is its last slot).
__device__ __forceinline__ const float* raw_row(const float* __restrict__ qkv,
                                                const float* __restrict__ hist,
                                                int K, int C, int p) {
  return p < 0 ? hist + (size_t)(K - 1 + p) * C : qkv + (size_t)p * C;
}

// (a)-(d) for one row, plus the histogram rewrite by the last block.
__global__ void gdn_prep_kernel(
    const float* __restrict__ qkv, const float* __restrict__ beta_logit,
    const float* __restrict__ alpha_logit, const float* __restrict__ conv,
    const float* __restrict__ dt, const float* __restrict__ a,
    const float* __restrict__ hist_in, float* __restrict__ hist_out,
    float* __restrict__ qn, float* __restrict__ kn, float* __restrict__ vbuf,
    float* __restrict__ beta_out, float* __restrict__ eg_out,
    unsigned int* __restrict__ counter, int n, int hk, int hv, int S, int K,
    float eps, float inv_sqrt_S) {
  const int t = blockIdx.x;
  const int key_dim = hk * S;
  const int value_dim = hv * S;
  const int C = 2 * key_dim + value_dim;
  extern __shared__ float y[];

  // (a) Convolution over the last K raw rows, oldest tap first, then SiLU.
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
      const float x = raw_row(qkv, hist_in, K, C, t - (K - 1) + i)[c];
      sum += x * conv[c * K + i];
    }
    y[c] = sum / (1.0f + expf(-sum));
  }
  __syncthreads();  // every history read of this block is done

  // The last block to get here rewrites the histogram. It reads qkv and the
  // incoming histogram only; all other blocks have finished reading them.
  __shared__ int is_last;
  if (threadIdx.x == 0)
    is_last = (atomicAdd(counter, 1u) == (unsigned)(gridDim.x - 1));
  __syncthreads();

  // (b) The value part of the split.
  for (int idx = threadIdx.x; idx < value_dim; idx += blockDim.x)
    vbuf[(size_t)t * value_dim + idx] = y[2 * key_dim + idx];

  // (c) Per-key-head normalization, one warp per (row, key head).
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  for (int hidx = warp; hidx < 2 * hk; hidx += warps) {
    const int half = hidx / hk;
    const int g = hidx % hk;
    float* x = y + half * key_dim + g * S;
    double sumsq = 0.0;
    for (int i = lane; i < S; i += 32) sumsq += (double)x[i] * (double)x[i];
    for (int off = 16; off > 0; off >>= 1)
      sumsq += __shfl_xor_sync(0xffffffffu, sumsq, off);
    const double mean = sumsq / (double)S;
    const float inv =
        (float)(1.0 / sqrt(mean + (double)eps / (double)S));
    float* dst = (half == 0 ? qn : kn) + (size_t)t * key_dim + g * S;
    for (int i = lane; i < S; i += 32) {
      float xv = x[i] * inv;
      xv = xv * inv_sqrt_S;
      dst[i] = xv;
    }
  }

  // (d) The gates, one thread per value head.
  for (int h = threadIdx.x; h < hv; h += blockDim.x) {
    const float beta = 1.0f / (1.0f + expf(-beta_logit[t * hv + h]));
    const float t_gate = alpha_logit[t * hv + h] + dt[h];
    const float sp = t_gate > 20.0f ? t_gate : logf(1.0f + expf(t_gate));
    beta_out[(size_t)t * hv + h] = beta;
    eg_out[(size_t)t * hv + h] = expf(sp * a[h]);
  }

  // The histogram keeps RAW rows: the last K-1 of (old hist ++ batch).
  // Column-wise ascending j: a column's source index is its destination index
  // plus at least one, so a thread never reads a slot it has overwritten.
  if (is_last && K > 1) {
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
      for (int j = 0; j < K - 1; ++j) {
        const int src = j - (K - 1) + n;
        hist_out[(size_t)j * C + c] =
            src < 0 ? hist_in[(size_t)(K - 1 + src) * C + c]
                    : qkv[(size_t)src * C + c];
      }
    }
  }
}

// (e) The memory update for one (h, j), R = S/32 values per lane.
template <int R>
__global__ void gdn_recur_kernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const float* __restrict__ vbuf, const float* __restrict__ beta_in,
    const float* __restrict__ eg_in, const float* __restrict__ mem_in,
    float* __restrict__ mem_out, float* __restrict__ out, int n, int hk, int hv,
    int S, float inv_sqrt_S) {
  const int h = blockIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int g = h % hk;
  const int key_dim = hk * S;
  const int value_dim = hv * S;
  const size_t row = (size_t)(h * S + j) * (size_t)S;

  float m[R];
#pragma unroll
  for (int r = 0; r < R; ++r) m[r] = mem_in[row + r * 32 + lane];

  for (int t = 0; t < n; ++t) {
    const float eg = eg_in[(size_t)t * hv + h];
    const float beta = beta_in[(size_t)t * hv + h];
    const float vl = vbuf[(size_t)t * value_dim + h * S + j];
    const float* krow = kn + (size_t)t * key_dim + g * S;
    const float* qrow = qn + (size_t)t * key_dim + g * S;

#pragma unroll
    for (int r = 0; r < R; ++r) m[r] *= eg;
    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) acc += m[r] * krow[r * 32 + lane];
    const float sk = warp_sum(acc);
    const float delta = (vl - sk) * beta;
    float acc2 = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      m[r] += krow[r * 32 + lane] * delta;
      acc2 += m[r] * qrow[r * 32 + lane];
    }
    const float o = warp_sum(acc2);
    if (lane == 0) out[(size_t)t * value_dim + h * S + j] = o * inv_sqrt_S;
  }

#pragma unroll
  for (int r = 0; r < R; ++r) mem_out[row + r * 32 + lane] = m[r];
}

// The same update for a head size past the register instantiation, with the
// lane's row in local memory.
__global__ void gdn_recur_generic_kernel(
    const float* __restrict__ qn, const float* __restrict__ kn,
    const float* __restrict__ vbuf, const float* __restrict__ beta_in,
    const float* __restrict__ eg_in, const float* __restrict__ mem_in,
    float* __restrict__ mem_out, float* __restrict__ out, int n, int hk, int hv,
    int S, float inv_sqrt_S) {
  const int h = blockIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  const int lane = threadIdx.x;
  const int g = h % hk;
  const int key_dim = hk * S;
  const int value_dim = hv * S;
  const int R = S / 32;
  const size_t row = (size_t)(h * S + j) * (size_t)S;

  float m[64];
  for (int r = 0; r < R; ++r) m[r] = mem_in[row + r * 32 + lane];

  for (int t = 0; t < n; ++t) {
    const float eg = eg_in[(size_t)t * hv + h];
    const float beta = beta_in[(size_t)t * hv + h];
    const float vl = vbuf[(size_t)t * value_dim + h * S + j];
    const float* krow = kn + (size_t)t * key_dim + g * S;
    const float* qrow = qn + (size_t)t * key_dim + g * S;

    for (int r = 0; r < R; ++r) m[r] *= eg;
    float acc = 0.0f;
    for (int r = 0; r < R; ++r) acc += m[r] * krow[r * 32 + lane];
    const float sk = warp_sum(acc);
    const float delta = (vl - sk) * beta;
    float acc2 = 0.0f;
    for (int r = 0; r < R; ++r) {
      m[r] += krow[r * 32 + lane] * delta;
      acc2 += m[r] * qrow[r * 32 + lane];
    }
    const float o = warp_sum(acc2);
    if (lane == 0) out[(size_t)t * value_dim + h * S + j] = o * inv_sqrt_S;
  }

  for (int r = 0; r < R; ++r) mem_out[row + r * 32 + lane] = m[r];
}

template <int R>
bool launch_recur(const float* qn, const float* kn, const float* vbuf,
                  const float* beta, const float* eg, const float* mem_in,
                  float* mem_out, float* out, int n, int hk, int hv, int S,
                  float inv_sqrt_S, cudaStream_t stream) {
  const dim3 grid(hv, S / 4);
  const dim3 block(32, 4);
  gdn_recur_kernel<R><<<grid, block, 0, stream>>>(
      qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n, hk, hv, S,
      inv_sqrt_S);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace

bool gdn_scan_dev(const GdnDims& dims, const GdnWeights& weights,
                  const float* qkv, const float* beta_logit,
                  const float* alpha_logit, int n, const float* hist_in,
                  float* hist_out, const float* mem_in, float* mem_out,
                  float* out) {
  const uint64_t hk = dims.heads_k;
  const uint64_t hv = dims.heads_v;
  const uint64_t S = dims.dim;
  const uint64_t K = dims.kernel;
  if (hk == 0 || hv == 0 || S == 0 || K == 0 || hv % hk != 0 || S % 32 != 0 ||
      S / 32 > 64)
    return false;
  const uint64_t key_dim = hk * S;
  const uint64_t value_dim = hv * S;
  const uint64_t C = 2 * key_dim + value_dim;
  if (C > 0x7fffffff) return false;
  if (weights.conv.size() != C * K || weights.dt.size() != hv ||
      weights.a.size() != hv)
    return false;
  if (n < 0) return false;
  if (n == 0) return true;
  if (!qkv || !beta_logit || !alpha_logit || !mem_in || !mem_out || !out)
    return false;
  if (K > 1 && (!hist_in || !hist_out)) return false;
  const size_t smem = (size_t)C * sizeof(float);
  if (smem > kMaxSharedBytes) return false;

  const float* d_conv = upload_f32(weights.conv);
  const float* d_dt = upload_f32(weights.dt);
  const float* d_a = upload_f32(weights.a);
  if (!d_conv || !d_dt || !d_a) return false;

  std::lock_guard<std::mutex> lock(g_mutex);
  void* d_qn = grow_scratch(0, (size_t)n * key_dim * sizeof(float));
  void* d_kn = grow_scratch(1, (size_t)n * key_dim * sizeof(float));
  void* d_v = grow_scratch(2, (size_t)n * value_dim * sizeof(float));
  void* d_beta = grow_scratch(3, (size_t)n * hv * sizeof(float));
  void* d_eg = grow_scratch(4, (size_t)n * hv * sizeof(float));
  void* d_counter = grow_scratch(5, sizeof(unsigned int));
  if (!d_qn || !d_kn || !d_v || !d_beta || !d_eg || !d_counter) return false;

  const cudaStream_t stream = sonicboom::cuda::sb_current_stream();
  const float inv_sqrt_S = 1.0f / std::sqrt((float)S);
  if (smem > 48 * 1024 &&
      cudaFuncSetAttribute(gdn_prep_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           (int)smem) != cudaSuccess)
    return false;
  if (cudaMemsetAsync(d_counter, 0, sizeof(unsigned int), stream) !=
      cudaSuccess)
    return false;

  gdn_prep_kernel<<<n, kPrepThreads, smem, stream>>>(
      qkv, beta_logit, alpha_logit, d_conv, d_dt, d_a,
      K > 1 ? hist_in : nullptr, hist_out, static_cast<float*>(d_qn),
      static_cast<float*>(d_kn), static_cast<float*>(d_v),
      static_cast<float*>(d_beta), static_cast<float*>(d_eg),
      static_cast<unsigned int*>(d_counter), n, (int)hk, (int)hv, (int)S,
      (int)K, dims.eps, inv_sqrt_S);
  if (cudaGetLastError() != cudaSuccess) return false;

  const float* qn = static_cast<const float*>(d_qn);
  const float* kn = static_cast<const float*>(d_kn);
  const float* vbuf = static_cast<const float*>(d_v);
  const float* beta = static_cast<const float*>(d_beta);
  const float* eg = static_cast<const float*>(d_eg);
  switch (S / 32) {
    case 1:
      return launch_recur<1>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 2:
      return launch_recur<2>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 3:
      return launch_recur<3>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 4:
      return launch_recur<4>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 5:
      return launch_recur<5>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 6:
      return launch_recur<6>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 7:
      return launch_recur<7>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    case 8:
      return launch_recur<8>(qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n,
                             (int)hk, (int)hv, (int)S, inv_sqrt_S, stream);
    default: {
      const dim3 grid((unsigned)hv, (unsigned)(S / 4));
      const dim3 block(32, 4);
      gdn_recur_generic_kernel<<<grid, block, 0, stream>>>(
          qn, kn, vbuf, beta, eg, mem_in, mem_out, out, n, (int)hk, (int)hv,
          (int)S, inv_sqrt_S);
      return cudaGetLastError() == cudaSuccess;
    }
  }
}

} // namespace sonicboom::nn::cuda
