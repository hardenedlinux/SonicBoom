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

#include <sm_32_intrinsics.h>  // __vsub4
#include <sm_61_intrinsics.h>  // __dp4a

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

// CUDA dequantize-on-the-fly gemv for Q3_K/Q4_K/Q5_K weights @ f32 activation.
//
// The packed weight lives in device memory (uploaded once, cached). Two kernel
// families share the file and the launcher dispatches on blocks-per-row:
//
//   _coop — cooperative iqs-slice warp-per-row: the 32 lanes cover one row's
//   super-blocks via the iqs-slice mapping (one lane = one 16/32-element slice
//   sharing a scale), so the packed-weight loads coalesce/broadcast within the
//   warp and a __shfl reduction produces the row's dot (no per-block atomicAdd).
//   This wins for large blocks-per-row (ffn_down has 40), where the alternative's
//   per-block atomic fan-in and x working set (bpr × 1 KB) spill L1.
//
//   _base — one thread per super-block with 8-way ILP accumulators. It wins for
//   small blocks-per-row (ffn_gate_up/attn_* have 8-10), where the x slice stays
//   in L1 and the atomic fan-in is only ~10-way.
//
// The dequant arithmetic in both reproduces the scalar reference in
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

// --- gemv kernels: cooperative (large) + one-thread-per-block (small) -------
//
// Two kernel families:
//
//   _coop — cooperative iqs-slice warp-per-row. Each warp computes one row; the
//   32 lanes cover the row's super-blocks via the iqs-slice mapping (llama.cpp's
//   vec_dot_*_K_q8_K layout), so the packed-weight loads coalesce/broadcast
//   within the warp instead of striding 110/144/176 bytes apart. This removes
//   the L1 sector waste (ncu: L1/TEX ~92-95%, lg_throttle) that limits the
//   one-thread-per-superblock layout, a ~1.6x win on the large FFN shapes.
//
//   _base — one thread per super-block with 8-way ILP accumulators. Fewer total
//   threads but each carries 8 independent FMA chains, which hides the block-load
//   latency for the small attention shapes (few blocks -> few warps, where the
//   _coop warp-per-row layout under-utilizes the GPU).
//
// matvec_f32 / matvec_f32_dev dispatch on total block count (see below).

__global__ void gemv_q3_K_base(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;
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

__global__ void gemv_q4_K_base(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;
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

__global__ void gemv_q5_K_base(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;
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

__global__ void gemv_q3_K_coop(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_rows) {
  const int lane = threadIdx.x;   // 0..31
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;    // warp-uniform: all 32 lanes share one row

  const int iqs = lane & 15;      // slice 0..15 within the block
  const int ib = lane >> 4;       // which of the 2 blocks this lane covers (0/1)

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int b = ib; b < blocks_per_row; b += 2) {
    const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 110;
    const uint8_t* hm = blk;
    const uint8_t* qs = blk + 32;
    const float d = half_to_float(ld_u16(blk + 108));
    int8_t scales[16];
    unpack_q3_K_scales(blk + 96, scales);

    // Slice iqs = 16 elements sharing scale scales[iqs]. The 16 packed q bytes
    // are contiguous; slices iqs & ~1 differ only in the 2-bit shift, so the same
    // bytes are fetched 4-ways across the warp (and the hm bytes 8-ways).
    const float dl = d * float(scales[iqs] - 32);
    const int qsh = ((iqs >> 1) & 3) << 1;
    const int hmm = 1 << (((iqs >> 3) << 2) | ((iqs >> 1) & 3));
    const uint8_t* qq = qs + ((iqs >> 3) << 5) + ((iqs & 1) << 4);
    const uint8_t* hh = hm + ((iqs & 1) << 4);
    const float* xx = x + (size_t)b * 256 + (iqs << 4);

    #pragma unroll
    for (int k = 0; k < 16; k += 4) {
      a0 += dl * float(int((qq[k + 0] >> qsh) & 3) - ((hh[k + 0] & hmm) ? 0 : 4)) * xx[k + 0];
      a1 += dl * float(int((qq[k + 1] >> qsh) & 3) - ((hh[k + 1] & hmm) ? 0 : 4)) * xx[k + 1];
      a2 += dl * float(int((qq[k + 2] >> qsh) & 3) - ((hh[k + 2] & hmm) ? 0 : 4)) * xx[k + 2];
      a3 += dl * float(int((qq[k + 3] >> qsh) & 3) - ((hh[k + 3] & hmm) ? 0 : 4)) * xx[k + 3];
    }
  }

  float sum = (a0 + a1) + (a2 + a3);
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) y[row] = sum;
}

__global__ void gemv_q4_K_coop(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;

  const int iqs = lane & 7;       // slice 0..7 (32 elements, one scale/min pair)
  const int ib = lane >> 3;       // 0..3 (4 blocks per warp)

  float sum = 0.0f;
  for (int b = ib; b < blocks_per_row; b += 4) {
    const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 144;
    const float d = half_to_float(ld_u16(blk + 0));
    const float dmin = half_to_float(ld_u16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* q = blk + 16;

    uint8_t sc, mi;
    get_scale_min_k4(iqs, scales, sc, mi);
    const float dl = d * float(sc);
    const float m = dmin * float(mi);
    const int nib = iqs & 1;
    const uint8_t* qq = q + (iqs >> 1) * 32;
    const float* xx = x + (size_t)b * 256 + (iqs << 5);

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
      const uint8_t byte = qq[k];
      const int v = nib ? int(byte >> 4) : int(byte & 0xF);
      acc += (dl * float(v) - m) * xx[k];
    }
    sum += acc;
  }

  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) y[row] = sum;
}

__global__ void gemv_q5_K_coop(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;

  const int iqs = lane & 7;       // slice 0..7 (32 elements, one scale/min pair)
  const int ib = lane >> 3;       // 0..3 (4 blocks per warp)

  float sum = 0.0f;
  for (int b = ib; b < blocks_per_row; b += 4) {
    const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 176;
    const float d = half_to_float(ld_u16(blk + 0));
    const float dmin = half_to_float(ld_u16(blk + 2));
    const uint8_t* scales = blk + 4;
    const uint8_t* qh = blk + 16;
    const uint8_t* ql = blk + 48;

    uint8_t sc, mi;
    get_scale_min_k4(iqs, scales, sc, mi);
    const float dl = d * float(sc);
    const float m = dmin * float(mi);
    const int nib = iqs & 1;
    const int qhbit = iqs;
    const uint8_t* qlp = ql + (iqs >> 1) * 32;
    const float* xx = x + (size_t)b * 256 + (iqs << 5);

    float acc = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
      const uint8_t byte = qlp[k];
      const int lo = nib ? int(byte >> 4) : int(byte & 0xF);
      const int v = lo + (((qh[k] >> qhbit) & 1) ? 16 : 0);
      acc += (dl * float(v) - m) * xx[k];
    }
    sum += acc;
  }

  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) y[row] = sum;
}

// --- q8_1 (int8-activation) mmvq: llama.cpp's dp4a path --------------------
//
// The fp32 kernels above are L1/LSU-throughput bound (ncu: L1/TEX ~97%,
// math_pipe ~0.2%): the fp32 activation x is re-read `rows` times, so x load
// traffic dominates the L1 data path. Quantizing x to q8_1 int8 (1 byte/elem
// instead of 4) cuts that traffic 4x, and __dp4a consumes the int8 without
// converting back to fp32. This is llama.cpp's mmvq (vecdotq.cuh) structure,
// ported to the same GGUF byte layouts SonicBoom reads. The dot reproduces the
// fp32 reference to ~q8_1 rounding (amax/127), within model-level tolerance.
// Off by default: SONICBOOM_GEMV_Q8_1=1 selects it.

namespace q8_1 {

constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QI8_1 = QK8_1 / 4;                    // 8 int32 groups per block
constexpr int QR3_K = 4, QI3_K = QK_K / (4 * QR3_K);  // 4, 16
constexpr int QR4_K = 2;                              // 8x32-elem blocks/row
constexpr int QR5_K = 2;                              // 8x32-elem blocks/row

struct block_q8_1 {
  float d;        // amax / 127 (kept fp32; llama.cpp packs d+s into a half2)
  int8_t qs[32];  // quants
};

__device__ __forceinline__ int get_int_b2(const void* x, int i) {
  const uint16_t* x16 = static_cast<const uint16_t*>(x);
  return int(x16[2 * i]) | (int(x16[2 * i + 1]) << 16);
}
__device__ __forceinline__ int get_int_b4(const void* x, int i) {
  return static_cast<const int*>(x)[i];
}

// One thread per 32-element block; amax/127 quantization (== llama.cpp's
// quantize_q8_1, which warp-reduces over the same 32 elements).
__global__ void quantize_q8_1(const float* __restrict__ x,
                              block_q8_1* __restrict__ y, int nblocks) {
  const int ib = blockIdx.x * blockDim.x + threadIdx.x;
  if (ib >= nblocks) return;
  const float* xb = x + (size_t)ib * 32;
  float amax = 0.0f;
  #pragma unroll
  for (int k = 0; k < 32; ++k) amax = fmaxf(amax, fabsf(xb[k]));
  const float d = amax / 127.0f;
  y[ib].d = d;
  #pragma unroll
  for (int k = 0; k < 32; ++k)
    y[ib].qs[k] = (d == 0.0f) ? int8_t(0) : int8_t(roundf(xb[k] / d));
}

// --- dot products (ported from llama.cpp vecdotq.cuh) ----------------------

__device__ __forceinline__ float vec_dot_q3_K_q8_1(const uint8_t* wb,
                                                   const block_q8_1* yb, int iqs) {
  // wb: hmask[32]@0, qs(2-bit)[64]@32, scales(6-bit)[12]@96, d(half)@108.
  const uint8_t* qs = wb + 32;
  const uint8_t* hmask = wb;
  const uint8_t* scales = wb + 96;
  const float d = half_to_float(ld_u16(wb + 108));

  const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));      // 4*(iqs/8)
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

  const int vl = get_int_b2(qs, iqs);
  const int vh = ~get_int_b2(hmask, iqs % (QI3_K / 2)) >> bq8_offset;

  int u[QR3_K];
  float d8[QR3_K];
  #pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    u[i] = get_int_b4(yb[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = yb[bq8_offset + i].d;
  }

  float sumf = 0.0f;
  #pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    const int isc = scale_offset + 2 * i;
    const int isc_low = isc % (QK_K / 32);
    const int sc_shift_low = 4 * (isc / (QK_K / 32));
    const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
    const int isc_high = isc % (QK_K / 64);
    const int sc_shift_high = 2 * (isc / (QK_K / 64));
    const int sc_high = ((scales[QK_K / 32 + isc_high] >> sc_shift_high) & 3) << 4;
    const int sc = (sc_low | sc_high) - 32;

    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = __vsub4(vil, vih);
    sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc);
  }
  return d * sumf;
}

__device__ __forceinline__ float vec_dot_q4_K_q8_1(const uint8_t* wb,
                                                   const block_q8_1* yb, int iqs) {
  // wb: d@0, dmin@2, scales[12]@4, qs(nibble)[128]@16.
  const uint8_t* scales = wb + 4;
  const uint8_t* qs = wb + 16;
  const float d = half_to_float(ld_u16(wb + 0));
  const float dmin = half_to_float(ld_u16(wb + 2));

  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];

  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int* q4 = reinterpret_cast<const int*>(qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];

  const uint16_t* sc16 = reinterpret_cast<const uint16_t*>(scales);
  const int j = bq8_offset / 2;
  const int jm = j & 1;
  const uint32_t s0 = sc16[jm + 0];
  const uint32_t s2 = sc16[jm + 2];
  const uint32_t s4 = sc16[jm + 4];
  const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
  uint16_t aux[2];
  aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
  aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
  const uint8_t* m = sc + 2;

  #pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1* bq8i = yb + bq8_offset + i;
    d8[i] = bq8i->d;
    const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }

  float sumf_d = 0.0f, sumf_m = 0.0f;
  #pragma unroll
  for (int i = 0; i < QR4_K; ++i) {
    const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
    const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
    const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i + 0], 0));
    const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i + 0], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  return d * sumf_d - dmin * sumf_m;
}

__device__ __forceinline__ float vec_dot_q5_K_q8_1(const uint8_t* wb,
                                                   const block_q8_1* yb, int iqs) {
  // wb: d@0, dmin@2, scales[12]@4, qh[32]@16, ql(nibble)[128]@48.
  const uint8_t* scales = wb + 4;
  const uint8_t* qh = wb + 16;
  const uint8_t* ql = wb + 48;
  const float d = half_to_float(ld_u16(wb + 0));
  const float dmin = half_to_float(ld_u16(wb + 2));

  int vl[2], vh[2];
  int u[2 * QR5_K];
  float d8[QR5_K];

  const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
  const int* ql32 = reinterpret_cast<const int*>(ql + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  const int* qh32 = reinterpret_cast<const int*>(qh + 4 * ((iqs / 2) % 4));
  vl[0] = ql32[0];
  vl[1] = ql32[4];
  vh[0] = qh32[0] >> bq8_offset;
  vh[1] = qh32[4] >> bq8_offset;

  const uint16_t* sc16 = reinterpret_cast<const uint16_t*>(scales);
  const int j = bq8_offset / 2;
  const int jm = j & 1;
  const uint32_t s0 = sc16[jm + 0];
  const uint32_t s2 = sc16[jm + 2];
  const uint32_t s4 = sc16[jm + 4];
  const uint32_t hi = (uint32_t)-(int32_t)(j >= 2);
  uint16_t aux[2];
  aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
  aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
  const uint8_t* sc = reinterpret_cast<const uint8_t*>(aux);
  const uint8_t* m = sc + 2;

  #pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const block_q8_1* bq8i = yb + bq8_offset + i;
    d8[i] = bq8i->d;
    const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }

  float sumf_d = 0.0f, sumf_m = 0.0f;
  #pragma unroll
  for (int i = 0; i < QR5_K; ++i) {
    const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
    const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
    const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
    const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
    const int v0i = vl0i | vh0i;
    const int v1i = vl1i | vh1i;
    const int dot1 = __dp4a(v0i, u[2 * i + 0], __dp4a(v1i, u[2 * i + 1], 0));
    const int dot2 = __dp4a(0x01010101, u[2 * i + 0], __dp4a(0x01010101, u[2 * i + 1], 0));
    sumf_d += d8[i] * (dot1 * sc[i]);
    sumf_m += d8[i] * (dot2 * m[i]);
  }
  return d * sumf_d - dmin * sumf_m;
}

// --- warp-per-row mmvq kernels ---------------------------------------------

__global__ void gemv_q3_K_q8_1(const uint8_t* __restrict__ w,
                               const block_q8_1* __restrict__ y,
                               float* __restrict__ dst, int blocks_per_row,
                               int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = lane & 15;   // QI3_K=16 slices
  const int ib = lane >> 4;    // 2 blocks per warp
  float sum = 0.0f;
  for (int b = ib; b < blocks_per_row; b += 2) {
    const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 110;
    const block_q8_1* yb = y + (size_t)b * (QK_K / QK8_1);
    sum += vec_dot_q3_K_q8_1(wb, yb, iqs);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) dst[row] = sum;
}

__global__ void gemv_q4_K_q8_1(const uint8_t* __restrict__ w,
                               const block_q8_1* __restrict__ y,
                               float* __restrict__ dst, int blocks_per_row,
                               int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = 2 * (lane & 15);  // QI4_K=32, vdr=2 -> iqs even 0..30
  const int ib = lane >> 4;
  float sum = 0.0f;
  for (int b = ib; b < blocks_per_row; b += 2) {
    const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 144;
    const block_q8_1* yb = y + (size_t)b * (QK_K / QK8_1);
    sum += vec_dot_q4_K_q8_1(wb, yb, iqs);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) dst[row] = sum;
}

__global__ void gemv_q5_K_q8_1(const uint8_t* __restrict__ w,
                               const block_q8_1* __restrict__ y,
                               float* __restrict__ dst, int blocks_per_row,
                               int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = 2 * (lane & 15);  // QI5_K=32, vdr=2 -> iqs even 0..30
  const int ib = lane >> 4;
  float sum = 0.0f;
  for (int b = ib; b < blocks_per_row; b += 2) {
    const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 176;
    const block_q8_1* yb = y + (size_t)b * (QK_K / QK8_1);
    sum += vec_dot_q5_K_q8_1(wb, yb, iqs);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) dst[row] = sum;
}

} // namespace q8_1

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
void* g_q8 = nullptr;
size_t g_q8_bytes = 0;
void* g_col = nullptr;
size_t g_col_bytes = 0;
void* g_row = nullptr;
size_t g_row_bytes = 0;

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

// Batched-matmul column gather/scatter: the K-quant gemv kernels take a
// contiguous single-column activation, so the prefill matmul gathers column c
// (X[i*n + c]) into a contiguous buffer, runs the gemv, and scatters the row
// result back (Y[r*n + c]).
__global__ void gather_col(const float* __restrict__ X, float* __restrict__ dst,
                           int cols, int n, int c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < cols) dst[i] = X[(size_t)i * n + c];
}
__global__ void scatter_col(const float* __restrict__ src, float* __restrict__ Y,
                            int rows, int n, int c) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r < rows) Y[(size_t)r * n + c] = src[r];
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

// Blocks-per-row threshold above which the cooperative warp-per-row (_coop)
// kernel is used instead of the one-thread-per-block (_base) kernel.
//
// The one-thread-per-block kernel reads one super-block per thread, so a warp's x
// working set is blocks_per_row × 1 KB and every block in a row atomically
// accumulates into the same y[row]. For the Gemma 4 shapes this is only cheap at
// small blocks_per_row (ffn_gate_up/attn_* have 8-10 blocks/row: ~281 us for the
// 102400-block FFN gate_up), where the x slice fits in L1 and the atomic fan-in is
// ~10-way. ffn_down has 40 blocks/row, where the same kernel spills x out of L1
// and serializes a 40-way atomic fan-in (788 us); there the cooperative kernel,
// which hands a whole row to one warp and reduces with __shfl, wins (496 us).
// The breakpoint sits between those clusters; it is tunable.
constexpr int kCoopMinBlocksPerRow = 20;

// Tunable override for the dispatch breakpoint (0 = all cooperative, a huge value
// = all one-thread-per-block), used to A/B the two kernel families without
// recompiling. Dev-time only; defaults to kCoopMinBlocksPerRow.
int coop_min_blocks_per_row() {
  static int v = [] {
    if (const char* e = std::getenv("SONICBOOM_GEMV_COOP_MIN_BPR")) {
      const int x = std::atoi(e);
      if (x >= 0) return x;
    }
    return kCoopMinBlocksPerRow;
  }();
  return v;
}

// Launch the K-quant gemv against device-resident x/y, dispatching on
// blocks-per-row (see above). `_coop` writes each row exactly once (no zero-fill
// needed); `_base` atomically accumulates, so y is zero-filled first.
void launch_gemv(QuantType qt, const uint8_t* wp, const float* xp, float* yp,
                 int blocks_per_row, int num_rows) {
  if (blocks_per_row >= coop_min_blocks_per_row()) {
    const dim3 block(32, 8);
    const dim3 grid(1, (num_rows + 7) / 8);
    switch (qt) {
      case QuantType::Q3_K:
        gemv_q3_K_coop<<<grid, block>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q4_K:
        gemv_q4_K_coop<<<grid, block>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q5_K:
        gemv_q5_K_coop<<<grid, block>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
    }
  } else {
    cudaMemset(yp, 0, (size_t)num_rows * sizeof(float));
    const int num_blocks = num_rows * blocks_per_row;
    const dim3 block(256);
    const dim3 grid((num_blocks + 255) / 256);
    switch (qt) {
      case QuantType::Q3_K:
        gemv_q3_K_base<<<grid, block>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q4_K:
        gemv_q4_K_base<<<grid, block>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q5_K:
        gemv_q5_K_base<<<grid, block>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
    }
  }
}

// q8_1 mmvq selection (dev-time gate). Off by default: the fp32 hybrid is the
// tested path. When enabled, matvec_f32 / matvec_f32_dev quantize x to q8_1 and
// run the dp4a warp-per-row kernels instead of the fp32 base/coop hybrid.
bool g_q8_enabled() {
  static const bool v = std::getenv("SONICBOOM_GEMV_Q8_1") != nullptr;
  return v;
}

// Quantize x -> q8_1 (int8 + per-32-block scale) into the device scratch, then
// run the warp-per-row dp4a kernels (each row is written once by lane 0, so no
// zero-fill). This mirrors llama.cpp's mul_mat_vec_q path.
bool launch_gemv_q8_1(QuantType qt, const uint8_t* wp, const float* xp, float* yp,
                      int blocks_per_row, int num_rows) {
  const size_t nblocks = (size_t)blocks_per_row * (q8_1::QK_K / q8_1::QK8_1);
  q8_1::block_q8_1* d_q8 = static_cast<q8_1::block_q8_1*>(
      grow_scratch(g_q8, g_q8_bytes, nblocks * sizeof(q8_1::block_q8_1)));
  if (!d_q8) return false;

  q8_1::quantize_q8_1<<<(nblocks + 255) / 256, 256>>>(xp, d_q8, int(nblocks));

  const dim3 block(32, 8);
  const dim3 grid(1, (num_rows + 7) / 8);
  switch (qt) {
    case QuantType::Q3_K:
      q8_1::gemv_q3_K_q8_1<<<grid, block>>>(wp, d_q8, yp, blocks_per_row, num_rows);
      break;
    case QuantType::Q4_K:
      q8_1::gemv_q4_K_q8_1<<<grid, block>>>(wp, d_q8, yp, blocks_per_row, num_rows);
      break;
    case QuantType::Q5_K:
      q8_1::gemv_q5_K_q8_1<<<grid, block>>>(wp, d_q8, yp, blocks_per_row, num_rows);
      break;
  }
  return cudaGetLastError() == cudaSuccess;
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
  // Dispatch: cooperative warp-per-row for the large FFN weights, one-thread-per-
  // block for the small attention projections (see launch_gemv).
  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  const float* xp = static_cast<const float*>(d_x);
  float* yp = static_cast<float*>(d_y);
  const int num_rows = int(rows);
  cudaEvent_t ev0 = nullptr, ev1 = nullptr;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  cudaEventRecord(ev0);
  if (g_q8_enabled())
    launch_gemv_q8_1(W.type(), wp, xp, yp, int(blocks_per_row), num_rows);
  else
    launch_gemv(W.type(), wp, xp, yp, int(blocks_per_row), num_rows);
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
  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  const int num_rows = int(rows);
  if (g_q8_enabled())
    return launch_gemv_q8_1(W.type(), wp, x, y, int(blocks_per_row), num_rows);
  launch_gemv(W.type(), wp, x, y, int(blocks_per_row), num_rows);
  return cudaGetLastError() == cudaSuccess;
}

// Batched (prefill) matmul: Y = W @ X for X [cols, n] (n innermost) and Y
// [rows, n] (n innermost). The K-quant gemv kernels take a contiguous
// single-column activation, so each of the n columns is gathered into a
// contiguous scratch buffer, run through the (unchanged) gemv, and scattered
// back — the per-column arithmetic is bit-identical to matvec_f32_dev, so this
// matches the CPU matmul_f32 to the same fp32 tolerance. n == 1 falls back to
// the single-column path.
bool matmul_f32_dev(const QuantizedTensor& W, const float* X, float* Y, int n) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  if (n <= 0 || !X || !Y) return false;
  if (n == 1) return matvec_f32_dev(W, X, Y);
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  const uint32_t blocks_per_row = uint32_t(cols / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = upload_weight(W);
  if (!d_w) return false;

  float* col = static_cast<float*>(
      grow_scratch(g_col, g_col_bytes, cols * sizeof(float)));
  float* row = static_cast<float*>(
      grow_scratch(g_row, g_row_bytes, rows * sizeof(float)));
  if (!col || !row) return false;

  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  const int ic = int(cols), ir = int(rows);
  for (int c = 0; c < n; ++c) {
    gather_col<<<(ic + 255) / 256, 256>>>(X, col, ic, n, c);
    launch_gemv(W.type(), wp, col, row, int(blocks_per_row), ir);
    scatter_col<<<(ir + 255) / 256, 256>>>(row, Y, ir, n, c);
  }
  return cudaGetLastError() == cudaSuccess;
}

// Public host-span batched matmul: uploads X, runs the device-resident matmul,
// and copies Y back (the H2D/D2H pattern of matvec_f32).
bool matmul_f32(const QuantizedTensor& W, std::span<const float> X,
                std::span<float> Y, uint64_t n) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (n == 0) return false;
  if (X.size() < cols * n || Y.size() < rows * n) return false;
  if (n == 1) return matvec_f32(W, X, Y);

  void* d_X = grow_scratch(g_dx, g_dx_bytes, cols * n * sizeof(float));
  void* d_Y = grow_scratch(g_dy, g_dy_bytes, rows * n * sizeof(float));
  if (!d_X || !d_Y) return false;
  if (cudaMemcpy(d_X, X.data(), cols * n * sizeof(float),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  if (!matmul_f32_dev(W, static_cast<const float*>(d_X),
                      static_cast<float*>(d_Y), int(n)))
    return false;
  return cudaMemcpy(Y.data(), d_Y, rows * n * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

} // namespace sonicboom::quant::cuda
