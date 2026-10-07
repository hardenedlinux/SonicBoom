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

#include <sonicboom/quant/quantized_matmul_cuda.h>

#include "cuda_resident.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>

// CUDA dequantize-on-the-fly gemv for Q3_K/Q4_K/Q5_K weights @ f32 activation.
//
// The packed weight lives in device memory (uploaded once, cached). Each thread
// handles one super-block (256 elements): it dequantizes the block and dots it
// against the matching 256-element slice of the activation, then atomically
// accumulates its partial into the output row. Consecutive threads walk
// consecutive super-blocks (consecutive packed bytes), so the weight stream is
// coalesced; the kernel is bandwidth-bound on the weight, which is the whole
// point for decode. The dequant arithmetic reproduces the scalar reference in
// core/src/quant/dequant.cpp (same reconstruction formulas and block layouts).

namespace sonicboom::quant::cuda {

namespace {

// --- device helpers (mirror core/src/quant/dequant.cpp) --------------------

__device__ inline uint16_t ld_u16(const uint8_t* p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

__device__ inline uint32_t ld_u32(const uint8_t* p) {
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) |
         (uint32_t(p[3]) << 24);
}

// IEEE 754 binary16 -> binary32 (identical to ggml's GGML_FP16_TO_FP32).
__device__ inline float half_to_float(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exp = uint32_t(h >> 10) & 0x1Fu;
  uint32_t mant = uint32_t(h) & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  return __uint_as_float(bits);
}

// Unpack the j-th 6-bit (scale, min) pair from a q4_K/q5_K scales block.
__device__ inline void get_scale_min_k4(int j, const uint8_t* scales, uint8_t& d,
                                        uint8_t& m) {
  if (j < 4) {
    d = scales[j] & 63;
    m = scales[j + 4] & 63;
  } else {
    d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// Unpack the q3_K 12-byte scales block into 16 int8 values.
__device__ inline void unpack_q3_K_scales(const uint8_t* src, int8_t* scales) {
  const uint32_t a0 = ld_u32(src + 0);
  const uint32_t a1 = ld_u32(src + 4);
  const uint32_t tmp = ld_u32(src + 8);
  const uint32_t kmask1 = 0x03030303u;
  const uint32_t kmask2 = 0x0f0f0f0fu;
  uint32_t b[4];
  b[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
  b[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
  b[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  b[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  memcpy(scales, b, 16);
}

// --- gemv kernels ----------------------------------------------------------

// Grid-stride, one thread per super-block (256 elements): thread `i` dequantizes
// block `i` on the fly and dots it against the matching 256-element activation
// slice, then atomically accumulates into the output row. Consecutive threads walk
// consecutive blocks, so the weight stream is coalesced and no lane sits idle
// (a warp-per-row split wastes 31/32 lanes when blocks_per_row < 32). The dequant
// arithmetic mirrors the scalar reference in core/src/quant/dequant.cpp. The
// per-block 256-deep FMA chain is split across 8 independent accumulators so the
// kernel is not latency-bound on a single serial dependency chain.
__global__ void gemv_q3_K(const uint8_t* __restrict__ w, const float* __restrict__ x,
                          float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks; i += stride) {
    const int row = i / blocks_per_row;
    const int b = i % blocks_per_row;
    const uint8_t* blk = w + (size_t)i * 110;
    const uint8_t* hm = blk;
    const uint8_t* q3 = blk + 32;
    const float d = half_to_float(ld_u16(blk + 108));
    int8_t scales[16];
    unpack_q3_K_scales(blk + 96, scales);
    const float* xb = x + (size_t)b * 256;

    float acc[8];
    #pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.0f;
    int is = 0;
    #pragma unroll
    for (int n = 0; n < 2; ++n) {
      const float* xn = xb + n * 128;
      int shift = 0;
      uint8_t m = uint8_t(1u << (n * 4));
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float* xq = xn + j * 32;
        const float dl = d * float(scales[is] - 32);
        const float dl2 = d * float(scales[is + 1] - 32);
        is += 2;
        const int a = n * 4 + j;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
          acc[a] += dl * float(int((q3[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)) * xq[l];
          acc[a] += dl2 * float(int((q3[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)) * xq[16 + l];
        }
        shift += 2;
        m <<= 1;
      }
      q3 += 32;
    }
    float s = acc[0];
    #pragma unroll
    for (int k = 1; k < 8; ++k) s += acc[k];
    atomicAdd(&y[row], s);
  }
}

__global__ void gemv_q4_K(const uint8_t* __restrict__ w, const float* __restrict__ x,
                          float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks; i += stride) {
    const int row = i / blocks_per_row;
    const int b = i % blocks_per_row;
    const uint8_t* blk = w + (size_t)i * 144;
    const float d = half_to_float(ld_u16(blk + 0));
    const float dmin = half_to_float(ld_u16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;
    const float* xb = x + (size_t)b * 256;

    float acc[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] = 0.0f;
    int is = 0;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint8_t sc0, mi0, sc1, mi1;
      get_scale_min_k4(is, scales, sc0, mi0);
      get_scale_min_k4(is + 1, scales, sc1, mi1);
      const float d1 = d * float(sc0);
      const float m1 = dmin * float(mi0);
      const float d2 = d * float(sc1);
      const float m2 = dmin * float(mi1);
      const float* xq = xb + j * 64;
      #pragma unroll
      for (int l = 0; l < 32; ++l)
        acc[j] += (d1 * float(q[l] & 0xF) - m1) * xq[l];
      #pragma unroll
      for (int l = 0; l < 32; ++l)
        acc[j] += (d2 * float(q[l] >> 4) - m2) * xq[32 + l];
      q += 32;
      is += 2;
    }
    float s = acc[0];
    #pragma unroll
    for (int k = 1; k < 4; ++k) s += acc[k];
    atomicAdd(&y[row], s);
  }
}

__global__ void gemv_q5_K(const uint8_t* __restrict__ w, const float* __restrict__ x,
                          float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks; i += stride) {
    const int row = i / blocks_per_row;
    const int b = i % blocks_per_row;
    const uint8_t* blk = w + (size_t)i * 176;
    const float d = half_to_float(ld_u16(blk + 0));
    const float dmin = half_to_float(ld_u16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* ql = blk + 48;
    const float* xb = x + (size_t)b * 256;

    float acc[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) acc[k] = 0.0f;
    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint8_t sc0, mi0, sc1, mi1;
      get_scale_min_k4(is, scales, sc0, mi0);
      get_scale_min_k4(is + 1, scales, sc1, mi1);
      const float d1 = d * float(sc0);
      const float m1 = dmin * float(mi0);
      const float d2 = d * float(sc1);
      const float m2 = dmin * float(mi1);
      const float* xq = xb + j * 64;
      #pragma unroll
      for (int l = 0; l < 32; ++l) {
        const int v = int(ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
        acc[j] += (d1 * float(v) - m1) * xq[l];
      }
      #pragma unroll
      for (int l = 0; l < 32; ++l) {
        const int v = int(ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
        acc[j] += (d2 * float(v) - m2) * xq[32 + l];
      }
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
    float s = acc[0];
    #pragma unroll
    for (int k = 1; k < 4; ++k) s += acc[k];
    atomicAdd(&y[row], s);
  }
}

// --- device weight cache + scratch -----------------------------------------

struct DeviceWeight {
  void* ptr = nullptr;
  size_t bytes = 0;
};

std::unordered_map<const void*, DeviceWeight> g_cache;
std::mutex g_cache_mutex;

void* g_dx = nullptr;
size_t g_dx_bytes = 0;
void* g_dy = nullptr;
size_t g_dy_bytes = 0;

bool g_available_checked = false;
bool g_available = false;

// Grow a device scratch buffer to at least `need` bytes, returning the pointer
// (or nullptr on allocation failure). Reuses the existing buffer when big enough.
void* grow_scratch(void*& buf, size_t& have, size_t need) {
  if (have >= need) return buf;
  if (buf) cudaFree(buf);
  buf = nullptr;
  have = 0;
  if (cudaMalloc(&buf, need) != cudaSuccess) return nullptr;
  have = need;
  return buf;
}

// Upload W's packed bytes to the device, caching by (data pointer, byte size)
// so repeated calls with the same weight reuse the device copy.
const void* upload_weight(const QuantizedTensor& W) {
  std::lock_guard<std::mutex> lk(g_cache_mutex);
  const void* host = W.data().data();
  const size_t bytes = W.byte_size();
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

// --- temporary profiling (remove after measuring) ---------------------------
long g_prof_calls = 0;
double g_prof_kernel_ms = 0.0;
double g_prof_copy_ms = 0.0;
struct ProfDump {
  ~ProfDump() {
    if (g_prof_calls)
      std::fprintf(stderr,
                   "[cuda-matvec-prof] %ld calls: kernel %.2f ms, copies %.2f ms\n",
                   g_prof_calls, g_prof_kernel_ms, g_prof_copy_ms);
  }
} g_prof_dump;

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

bool matvec_f32(const QuantizedTensor& W, std::span<const float> x,
                std::span<float> y) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (x.size() < cols || y.size() < rows) return false;
  const uint32_t blocks_per_row = uint32_t(cols / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = upload_weight(W);
  if (!d_w) return false;

  void* d_x = grow_scratch(g_dx, g_dx_bytes, cols * sizeof(float));
  void* d_y = grow_scratch(g_dy, g_dy_bytes, rows * sizeof(float));
  if (!d_x || !d_y) return false;

  const auto t0 = std::chrono::steady_clock::now();
  if (cudaMemcpy(d_x, x.data(), cols * sizeof(float), cudaMemcpyHostToDevice) !=
      cudaSuccess)
    return false;
  if (cudaMemset(d_y, 0, rows * sizeof(float)) != cudaSuccess) return false;

  // Grid-stride over super-blocks: one thread per 256-element block, coalesced
  // weight stream, atomicAdd into the output row. grid spans all rows' blocks.
  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  const float* xp = static_cast<const float*>(d_x);
  float* yp = static_cast<float*>(d_y);
  const int num_blocks = int(rows * blocks_per_row);
  const int grid = (num_blocks + 255) / 256;
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  switch (W.type()) {
    case QuantType::Q3_K:
      gemv_q3_K<<<grid, 256>>>(wp, xp, yp, int(blocks_per_row), num_blocks);
      break;
    case QuantType::Q4_K:
      gemv_q4_K<<<grid, 256>>>(wp, xp, yp, int(blocks_per_row), num_blocks);
      break;
    case QuantType::Q5_K:
      gemv_q5_K<<<grid, 256>>>(wp, xp, yp, int(blocks_per_row), num_blocks);
      break;
  }
  cudaEventRecord(ev1);
  if (cudaGetLastError() != cudaSuccess) return false;

  const bool ok = cudaMemcpy(y.data(), d_y, rows * sizeof(float),
                             cudaMemcpyDeviceToHost) == cudaSuccess;
  const auto t1 = std::chrono::steady_clock::now();
  cudaEventSynchronize(ev1);
  float kms = 0.0f;
  cudaEventElapsedTime(&kms, ev0, ev1);
  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  g_prof_kernel_ms += kms;
  g_prof_copy_ms += std::chrono::duration<double, std::milli>(t1 - t0).count() - kms;
  ++g_prof_calls;
  return ok;
}

// Device-resident variant (Phase 6a Option A): same gemv, but x and y are device
// buffers and no host copy happens — the caller keeps the activation resident and
// chains this back-to-back with other device kernels. W is uploaded+cached exactly
// as in matvec_f32.
bool matvec_f32_dev(const QuantizedTensor& W, const float* x, float* y) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (!x || !y) return false;
  const uint32_t blocks_per_row = uint32_t(cols / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  if (cudaMemset(y, 0, rows * sizeof(float)) != cudaSuccess) return false;

  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  const int num_blocks = int(rows * blocks_per_row);
  const int grid = (num_blocks + 255) / 256;
  switch (W.type()) {
    case QuantType::Q3_K:
      gemv_q3_K<<<grid, 256>>>(wp, x, y, int(blocks_per_row), num_blocks);
      break;
    case QuantType::Q4_K:
      gemv_q4_K<<<grid, 256>>>(wp, x, y, int(blocks_per_row), num_blocks);
      break;
    case QuantType::Q5_K:
      gemv_q5_K<<<grid, 256>>>(wp, x, y, int(blocks_per_row), num_blocks);
      break;
  }
  return cudaGetLastError() == cudaSuccess;
}

} // namespace sonicboom::quant::cuda
