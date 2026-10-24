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

// MOSS-Image DiT kernels (nvcc, C++20). The host orchestration lives in
// moss_dit_cuda.cpp; this TU only launches kernels and wraps the CUDA API so
// that no C++23 header (safetensors Reader, std::expected) reaches nvcc.

#include "moss_dit_kernels.h"

#include "../nn/cuda_resident.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

namespace sonicboom::moss::cuda_dit {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float situ_gate(float g) {
  return (4.0f * tanhf(g / 4.0f) * (1.0f / (1.0f + expf(-g))));
}

__device__ __forceinline__ float situ_value(float h) {
  return 25.0f * tanhf(h / 25.0f);
}

// y[r][i] = rms_norm(x[r])[i] * w[i], one block per row.
__global__ void rms_norm_rows_kernel(const float* __restrict__ x,
                                     const float* __restrict__ w, float eps,
                                     float* __restrict__ y, int dim) {
  __shared__ float s[32];
  const int r = blockIdx.x;
  const int tid = threadIdx.x;
  const float* row = x + (long)r * dim;
  float* dst = y + (long)r * dim;
  float ss = 0.0f;
  for (int i = tid; i < dim; i += blockDim.x) ss += row[i] * row[i];
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
  if ((tid & 31) == 0) s[tid >> 5] = ss;
  __syncthreads();
  if (tid == 0) {
    float sum = 0.0f;
    for (int i = 0; i < (int)(blockDim.x >> 5); ++i) sum += s[i];
    s[0] = rsqrtf(sum / (float)dim + eps);
  }
  __syncthreads();
  const float scale = s[0];
  for (int i = tid; i < dim; i += blockDim.x) dst[i] = row[i] * scale * w[i];
}

// Per-head RMSNorm over rows of `heads` heads; one warp per (row, head).
__global__ void rms_norm_heads_kernel(float* __restrict__ v,
                                      const float* __restrict__ w, float eps,
                                      int hd) {
  const int lane = threadIdx.x & 31;
  const int r = blockIdx.x;
  float* row = v + (long)r * hd;
  float ss = 0.0f;
  for (int d = lane; d < hd; d += 32) ss += row[d] * row[d];
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
  ss = __shfl_sync(0xffffffffu, ss, 0);
  const float scale = rsqrtf(ss / (float)hd + eps);
  for (int d = lane; d < hd; d += 32) row[d] = row[d] * scale * w[d];
}

// y = y * (1 + scale) + shift, per row of `dim`.
__global__ void modulate_kernel(float* __restrict__ y,
                                const float* __restrict__ shift,
                                const float* __restrict__ scale, int dim,
                                long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = (int)(i % dim);
  y[i] = y[i] * (1.0f + scale[c]) + shift[c];
}

// x += add * gate.
__global__ void gated_add_kernel(float* __restrict__ x,
                                 const float* __restrict__ add,
                                 const float* __restrict__ gate, int dim,
                                 long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  x[i] += add[i] * gate[i % dim];
}

__global__ void add_kernel(float* __restrict__ x,
                           const float* __restrict__ add, long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  x[i] += add[i];
}

// y += bias, broadcast over rows of `width`.
__global__ void bias_add_kernel(float* __restrict__ y,
                                const float* __restrict__ b, int width,
                                long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  y[i] += b[i % width];
}

// act = situ(gate) * situ_value(up) for one [rows, ffn] tile; the fused
// variant reads the [rows, 2*ffn] up_gate projection (up first).
__global__ void situ_glu_fused_kernel(const float* __restrict__ ug,
                                      float* __restrict__ act, int ffn,
                                      long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long row = i / ffn;
  const long c = i - row * ffn;
  const float* base = ug + row * 2 * ffn;
  act[i] = situ_gate(base[ffn + c]) * situ_value(base[c]);
}

__global__ void situ_glu_split_kernel(const float* __restrict__ up,
                                      const float* __restrict__ gate,
                                      float* __restrict__ act, long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  act[i] = situ_gate(gate[i]) * situ_value(up[i]);
}

// Half-split 4-axis RoPE with a per-token (cos, sin) table [rows, hd/2].
__global__ void rope_table_kernel(float* __restrict__ v,
                                  const float* __restrict__ rcos,
                                  const float* __restrict__ rsin, int heads,
                                  int hd, long total) {
  const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int half = hd >> 1;
  const int i = (int)(idx % half);
  const long rest = idx / half;
  const int h = (int)(rest % heads);
  const long t = rest / heads;
  float* row = v + (t * heads + h) * hd;
  const float a = row[i], b = row[i + half];
  const float c = rcos[t * half + i], s = rsin[t * half + i];
  row[i] = a * c - b * s;
  row[i + half] = b * c + a * s;
}

// Full (non-causal) SDPA. A tiled flash-style kernel: one CTA per
// (head, 64-query tile), K and V streamed through shared memory in 32-key tiles
// so every K/V element is read once per 64 queries instead of once per query
// (the warp-per-query form was ~80% of DiT time, purely on K/V traffic).
//
// Layout: 8 warps, each owning 8 queries. Within a key tile each lane owns one
// key and computes its complete dot product (no shuffles in the QK path); the
// scores are exchanged through shared memory for the PV accumulation, where
// each lane owns head-dim/32 output slots. K is shared transposed with a
// (BK+1) stride so both the write and the read patterns are bank-conflict free.
constexpr int DIT_BK = 32;
constexpr int DIT_WQ = 8;
constexpr int DIT_TQ = 64;

template <int ACC>
__global__ void sdpa_full_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v, int tokens,
                                 int n_heads, int head_dim, float scale,
                                 float* __restrict__ out) {
  extern __shared__ float s_sh[];
  float* sK = s_sh;                       // [head_dim][DIT_BK + 1]
  float* sV = sK + head_dim * (DIT_BK + 1);  // [DIT_BK][head_dim]
  float* sP = sV + DIT_BK * head_dim;     // [8][DIT_BK]
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int h = blockIdx.y;
  const int q0 = blockIdx.x * DIT_TQ;

  int qidx[DIT_WQ];
  float m[DIT_WQ], l[DIT_WQ];
  float acc[DIT_WQ][ACC];
  #pragma unroll
  for (int i = 0; i < DIT_WQ; ++i) {
    qidx[i] = q0 + warp * DIT_WQ + i;
    m[i] = -1e30f;
    l[i] = 0.0f;
    #pragma unroll
    for (int d = 0; d < ACC; ++d) acc[i][d] = 0.0f;
  }
  float* pbuf = sP + warp * DIT_BK;

  for (int j0 = 0; j0 < tokens; j0 += DIT_BK) {
    for (int idx = threadIdx.x; idx < DIT_BK * head_dim;
         idx += blockDim.x) {
      const int jj = idx / head_dim;
      const int d = idx - jj * head_dim;
      const int j = j0 + jj;
      float kv = 0.0f, vv = 0.0f;
      if (j < tokens) {
        const long src = (long(j) * n_heads + h) * head_dim + d;
        kv = k[src];
        vv = v[src];
      }
      sK[d * (DIT_BK + 1) + jj] = kv;
      sV[idx] = vv;
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < DIT_WQ; ++i) {
      if (qidx[i] >= tokens) continue;
      const float* qrow = q + ((long)qidx[i] * n_heads + h) * head_dim;
      // Lane == key: four independent accumulation chains for ILP.
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      int d = 0;
      for (; d + 3 < head_dim; d += 4) {
        a0 += qrow[d] * sK[(d + 0) * (DIT_BK + 1) + lane];
        a1 += qrow[d + 1] * sK[(d + 1) * (DIT_BK + 1) + lane];
        a2 += qrow[d + 2] * sK[(d + 2) * (DIT_BK + 1) + lane];
        a3 += qrow[d + 3] * sK[(d + 3) * (DIT_BK + 1) + lane];
      }
      for (; d < head_dim; ++d)
        a0 += qrow[d] * sK[d * (DIT_BK + 1) + lane];
      float s = (a0 + a1) + (a2 + a3);
      s = (j0 + lane < tokens) ? s * scale : -1e30f;

      float tile_max = s;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffu, tile_max, o));
      tile_max = __shfl_sync(0xffffffffu, tile_max, 0);
      const float m_new = fmaxf(m[i], tile_max);
      const float rescale = expf(m[i] - m_new);
      float p = (tile_max == -1e30f) ? 0.0f : expf(s - m_new);
      float psum = p;
      #pragma unroll
      for (int o = 16; o > 0; o >>= 1)
        psum += __shfl_down_sync(0xffffffffu, psum, o);
      psum = __shfl_sync(0xffffffffu, psum, 0);
      l[i] = l[i] * rescale + psum;
      m[i] = m_new;
      #pragma unroll
      for (int di = 0; di < ACC; ++di) acc[i][di] *= rescale;
      pbuf[lane] = p;
      __syncwarp();
      #pragma unroll
      for (int di = 0; di < ACC; ++di) {
        const int dd = lane + di * 32;
        float a = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < DIT_BK; ++jj)
          a += pbuf[jj] * sV[jj * head_dim + dd];
        acc[i][di] += a;
      }
      __syncwarp();
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < DIT_WQ; ++i) {
    if (qidx[i] >= tokens) continue;
    float* orow = out + ((long)qidx[i] * n_heads + h) * head_dim;
    const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
    #pragma unroll
    for (int di = 0; di < ACC; ++di) orow[lane + di * 32] = acc[i][di] * inv;
  }
}

// [C][HW] -> [HW][C] and back.
__global__ void video_to_tokens_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst, int channels,
                                       int hw) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)channels * hw) return;
  const int c = (int)(i / hw);
  const int p = (int)(i % hw);
  dst[(long)p * channels + c] = src[i];
}

__global__ void tokens_to_video_kernel(const float* __restrict__ src,
                                       float* __restrict__ dst, int channels,
                                       int hw) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (long)channels * hw) return;
  const int c = (int)(i / hw);
  const int p = (int)(i % hw);
  dst[i] = src[(long)p * channels + c];
}

// Split a fused [rows, 3*dim] qkv projection into three [rows, dim] buffers.
__global__ void qkv_split_kernel(const float* __restrict__ qkv,
                                 float* __restrict__ q, float* __restrict__ k,
                                 float* __restrict__ v, int dim, long total) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const long row = i / dim;
  const int c = (int)(i - row * dim);
  const float* src = qkv + row * 3 * dim + c;
  q[i] = src[0];
  k[i] = src[dim];
  v[i] = src[2 * dim];
}

inline unsigned grid_for(long total, int threads = kThreads) {
  return (unsigned)((total + threads - 1) / threads);
}

// The fp32 CUDA-core path, used when the f16 score planes would not fit.
bool sdpa_full_fused(const float* q, const float* k, const float* v,
                     int tokens, int heads, int head_dim, float scale,
                     float* out) {
  if (head_dim % 32 != 0 || head_dim > 32 * 8) return false;
  const int acc = head_dim / 32;
  const unsigned smem = unsigned(head_dim * (DIT_BK + 1) + DIT_BK * head_dim +
                                 8 * DIT_BK) * 4u;
  const dim3 grid(unsigned((tokens + DIT_TQ - 1) / DIT_TQ), unsigned(heads));
  switch (acc) {
    case 1: sdpa_full_kernel<1><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    case 2: sdpa_full_kernel<2><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    case 3: sdpa_full_kernel<3><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    case 4: sdpa_full_kernel<4><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    case 5: sdpa_full_kernel<5><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    case 6: sdpa_full_kernel<6><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    case 7: sdpa_full_kernel<7><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
    default: sdpa_full_kernel<8><<<grid, 256, smem>>>(q, k, v, tokens, heads, head_dim, scale, out); break;
  }
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace

// --- entry points ----------------------------------------------------------

bool device_available() {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

void* device_alloc(size_t bytes) {
  void* p = nullptr;
  return cudaMalloc(&p, bytes) == cudaSuccess ? p : nullptr;
}
void device_free(void* p) {
  if (p != nullptr) cudaFree(p);
}
bool upload(void* dev, const void* host, size_t bytes) {
  return cudaMemcpy(dev, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}
bool download(void* host, const void* dev, size_t bytes) {
  return cudaMemcpy(host, dev, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}
bool copy_d2d(void* dst, const void* src, size_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice) == cudaSuccess;
}
bool upload_async(void* dev, const void* pinned_host, size_t bytes,
                  void* stream) {
  return cudaMemcpyAsync(dev, pinned_host, bytes, cudaMemcpyHostToDevice,
                         static_cast<cudaStream_t>(stream)) == cudaSuccess;
}
void* pinned_alloc(size_t bytes) {
  void* p = nullptr;
  return cudaHostAlloc(&p, bytes, cudaHostAllocDefault) == cudaSuccess ? p
                                                                      : nullptr;
}
void pinned_free(void* p) {
  if (p != nullptr) cudaFreeHost(p);
}
void* stream_create() {
  cudaStream_t s = nullptr;
  return cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess
             ? static_cast<void*>(s)
             : nullptr;
}
void stream_destroy(void* stream) {
  if (stream != nullptr) cudaStreamDestroy(static_cast<cudaStream_t>(stream));
}
bool stream_sync(void* stream) {
  return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess;
}
const char* last_error() {
  const cudaError_t e = cudaGetLastError();
  return e == cudaSuccess ? "" : cudaGetErrorString(e);
}

bool rms_norm_rows(const float* x, const float* w, float eps, float* y,
                   int dim, long rows) {
  if (rows <= 0) return true;
  rms_norm_rows_kernel<<<(unsigned)rows, kThreads>>>(x, w, eps, y, dim);
  return cudaGetLastError() == cudaSuccess;
}

bool rms_norm_heads(float* v, const float* w, float eps, int head_dim,
                    long rows_heads) {
  if (rows_heads <= 0) return true;
  rms_norm_heads_kernel<<<(unsigned)rows_heads, 32>>>(v, w, eps, head_dim);
  return cudaGetLastError() == cudaSuccess;
}

bool modulate(float* y, const float* shift, const float* scale, int dim,
              long total) {
  if (total <= 0) return true;
  modulate_kernel<<<grid_for(total), kThreads>>>(y, shift, scale, dim, total);
  return cudaGetLastError() == cudaSuccess;
}

bool gated_add(float* x, const float* add, const float* gate, int dim,
               long total) {
  if (total <= 0) return true;
  gated_add_kernel<<<grid_for(total), kThreads>>>(x, add, gate, dim, total);
  return cudaGetLastError() == cudaSuccess;
}

bool add(float* x, const float* addend, long total) {
  if (total <= 0) return true;
  add_kernel<<<grid_for(total), kThreads>>>(x, addend, total);
  return cudaGetLastError() == cudaSuccess;
}

bool bias_add(float* y, const float* b, int width, long total) {
  if (total <= 0 || b == nullptr) return true;
  bias_add_kernel<<<grid_for(total), kThreads>>>(y, b, width, total);
  return cudaGetLastError() == cudaSuccess;
}

bool situ_glu_fused(const float* ug, float* act, int ffn, long total) {
  if (total <= 0) return true;
  situ_glu_fused_kernel<<<grid_for(total), kThreads>>>(ug, act, ffn, total);
  return cudaGetLastError() == cudaSuccess;
}

bool situ_glu_split(const float* up, const float* gate, float* act,
                    long total) {
  if (total <= 0) return true;
  situ_glu_split_kernel<<<grid_for(total), kThreads>>>(up, gate, act, total);
  return cudaGetLastError() == cudaSuccess;
}

bool rope_table(float* v, const float* rcos, const float* rsin, int heads,
                int head_dim, long total) {
  if (total <= 0) return true;
  rope_table_kernel<<<grid_for(total), kThreads>>>(v, rcos, rsin, heads,
                                                   head_dim, total);
  return cudaGetLastError() == cudaSuccess;
}

void sdpa_release_scratch() {
  nn::cuda::flash_attention_full_release_scratch();
}

// Full attention of the DiT block. The op's attributes (non-causal, no KV
// cache, no prefix reuse, no window) are the class key; nn/cuda resolves the
// class (full.score-gemm / full.tiled) and its implementation through the
// decision registry. No model-name check and no token-count branch live here:
// this function hands the op to the core dispatch and keeps the fused fp32
// kernel only as the fallback for a shape the core has no class for.
bool sdpa_full(const float* q, const float* k, const float* v, int tokens,
               int heads, int head_dim, float scale, float* out) {
  if (tokens <= 0) return true;
  if (nn::cuda::flash_attention_full_dev(q, k, v, tokens, heads, head_dim,
                                         scale, out))
    return true;
  return sdpa_full_fused(q, k, v, tokens, heads, head_dim, scale, out);
}

bool video_to_tokens(const float* src, float* dst, int channels, int hw) {
  video_to_tokens_kernel<<<grid_for(long(channels) * hw), kThreads>>>(src, dst,
                                                                      channels, hw);
  return cudaGetLastError() == cudaSuccess;
}

bool tokens_to_video(const float* src, float* dst, int channels, int hw) {
  tokens_to_video_kernel<<<grid_for(long(channels) * hw), kThreads>>>(src, dst,
                                                                      channels, hw);
  return cudaGetLastError() == cudaSuccess;
}

bool qkv_split(const float* qkv, float* q, float* k, float* v, int dim,
               long total) {
  qkv_split_kernel<<<grid_for(total), kThreads>>>(qkv, q, k, v, dim, total);
  return cudaGetLastError() == cudaSuccess;
}

}  // namespace sonicboom::moss::cuda_dit
