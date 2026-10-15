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
//
// ---------------------------------------------------------------------------
// Derived from ggml (llama.cpp), MIT-licensed. The q8_1 mmvq kernels below
// (quantize_q8_1, vec_dot_q3/q4/q5/q6_K_q8_1, gemv_q3/q4/q5/q6_K_q8_1, get_int_b2,
// get_int_b4, and the block_q8_1 layout) are ported from the following
// upstream sources:
//
//   ggml/src/ggml-cuda/vecdotq.cuh  — vec_dot_q3/q4/q5/q6_K_q8_1 (mmvq impls)
//   ggml/src/ggml-cuda/mmvq.cu      — mul_mat_vec_q3/q4/q5/q6_K_q8_1 (warp gemv)
//   ggml/src/ggml-cuda/quantize.cu  — quantize_q8_1
//
// Upstream commit: llama.cpp 43fe9c64281ef735046adc025e9e7559a1f659a5
//
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// SonicBoom's own additions (the ld_u16 / half_to_float helpers below, the fp32
// `d` in block_q8_1 in place of ggml's packed half2, and the launch glue that
// stays in cuda_quant_matmul.cu) are GPL-3.0-or-later as above.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <sm_32_intrinsics.h>  // __vsub4
#include <sm_61_intrinsics.h>  // __dp4a

#include <cstdint>

namespace sonicboom::quant::cuda::q8_1 {

// --- self-contained device helpers (SonicBoom's own, not ggml-derived) -----

__device__ inline uint16_t ld_u16(const uint8_t* p) {
  return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

// Alignment-agnostic 32-bit load. The legacy block layouts put their quant
// arrays at a 2-byte offset (q8_0) or a 6-byte offset (q5_0), so a plain
// reinterpret_cast<const int*> would be misaligned.
__device__ inline uint32_t ld_u32(const uint8_t* p) {
  return uint32_t(ld_u16(p)) | (uint32_t(ld_u16(p + 2)) << 16);
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

// --- q8_1 (int8-activation) mmvq: llama.cpp's dp4a path --------------------

constexpr int QK_K = 256;
constexpr int QK8_1 = 32;
constexpr int QI8_1 = QK8_1 / 4;                    // 8 int32 groups per block
constexpr int QR3_K = 4, QI3_K = QK_K / (4 * QR3_K);  // 4, 16
constexpr int QR4_K = 2;                              // 8x32-elem blocks/row
constexpr int QR5_K = 2;                              // 8x32-elem blocks/row
constexpr int QR6_K = 2;
constexpr int QI6_K = QK_K / (4 * QR6_K);

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
  // scale_offset == iqs - iqs%8 + (iqs%8)/4, but expressed without the div/mod:
  // {0..3}->0, {4..7}->1, {8..11}->8, {12..15}->9.
  const int scale_offset = (iqs & 8) | ((iqs >> 2) & 1);

  // Cooperative 6-bit scale unpack (SonicBoom-owned, beyond the llama.cpp port):
  // the 16 lanes of a super-block group each need 4 of the block's 16 scales,
  // so the stock per-lane unpack recomputes every scale 4x. Here each lane
  // unpacks *one* scale — scale[iqs] — and the 4 it actually needs are gathered
  // from its group via __shfl. Bit-identical to the stock unpack (same integer
  // sc values, same dot accumulation order); it only removes redundant ALU work
  // from the instruction/compute-bound Q3_K kernel.
  const int lane = threadIdx.x;                            // 0..31
  const int group_base = (lane >> 4) << 4;                 // 0 or 16
  const int my_sc = ((scales[iqs & 7] >> (4 * (iqs >> 3))) & 0xF)
                  | (((scales[8 + (iqs & 3)] >> (2 * (iqs >> 2))) & 3) << 4);
  int sc[QR3_K];
  #pragma unroll
  for (int i = 0; i < QR3_K; ++i)
    sc[i] = __shfl_sync(0xffffffffu, my_sc, group_base + scale_offset + 2 * i);

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
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = __vsub4(vil, vih);
    sumf += d8[i] * (__dp4a(vi, u[i], 0) * (sc[i] - 32));
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

__device__ __forceinline__ float vec_dot_q6_K_q8_1(const uint8_t* wb,
                                                   const block_q8_1* yb, int iqs) {
  const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K / 2)) +
                        (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) +
                           (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = get_int_b2(wb, iqs);
  const int vh = get_int_b2(wb + 128,
      (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;
  const int8_t* scales = reinterpret_cast<const int8_t*>(wb + 192) + scale_offset;
  const float d = half_to_float(ld_u16(wb + 208));
  int u[QR6_K];
  float d8[QR6_K];
  #pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    u[i] = get_int_b4(yb[bq8_offset + 2 * i].qs, iqs % QI8_1);
    d8[i] = yb[bq8_offset + 2 * i].d;
  }
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int ql = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int qh = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int q = __vsubss4(ql | qh, 0x20202020);
    sum += d8[i] * (__dp4a(q, u[i], 0) * scales[4 * i]);
  }
  return d * sum;
}

// --- legacy 32-element blocks (Q5_0 / Q8_0) --------------------------------
//
// A legacy block is exactly one q8_1 activation block wide, so these dots take
// no `iqs` slice index: each call consumes the whole 32-element block pair
// (the K-quant dots above each cover one 16-element slice of a 256-element
// super-block instead). Both dots are integer dp4a products of the aligned
// 4-byte groups, scaled by the two block scales.
//
// q5_0 stores weight elements 0..15 in the low nibbles of qs[16] and 16..31 in
// the high nibbles; bit j of the little-endian 4-byte qh word is element j's
// 5th bit. As in ggml the 5-bit field is biased: value = 5bit - 16, so the
// correction -16 * sum(q8) is folded in through the same sum-of-quants dp4a
// trick that vec_dot_q4_K_q8_1 uses for its -dmin term.
//
// Layouts: q5_0 = {fp16 d; uint8 qh[4]; uint8 qs[16]}, q8_0 = {fp16 d; int8 qs[32]}.

__device__ __forceinline__ float vec_dot_q8_0_q8_1(const uint8_t* wb,
                                                   const block_q8_1* yb) {
  int sumi = 0;
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    sumi = __dp4a(int(ld_u32(wb + 2 + 4 * i)), get_int_b4(yb->qs, i), sumi);
  return half_to_float(ld_u16(wb)) * yb->d * float(sumi);
}

__device__ __forceinline__ float vec_dot_q5_0_q8_1(const uint8_t* wb,
                                                   const block_q8_1* yb) {
  const uint32_t xh = ld_u32(wb + 2);  // 5th bits of elements 0..31
  int sumi = 0, sumu = 0;
  #pragma unroll
  for (int g = 0; g < 4; ++g) {
    const uint32_t nib = ld_u32(wb + 6 + 4 * g);
    // Low nibbles -> elements 4g..4g+3; 5th bits are xh bits 4g..4g+3.
    const uint32_t lo = xh >> (4 * g);
    int vi = int(nib & 0x0F0F0F0Fu);
    vi |= int((lo << 4) & 0x00000010u);
    vi |= int((lo << 11) & 0x00001000u);
    vi |= int((lo << 18) & 0x00100000u);
    vi |= int((lo << 25) & 0x10000000u);
    // High nibbles -> elements 16+4g..19+4g; 5th bits are the next four.
    const uint32_t hi = xh >> (4 * g + 16);
    int wi = int((nib >> 4) & 0x0F0F0F0Fu);
    wi |= int((hi << 4) & 0x00000010u);
    wi |= int((hi << 11) & 0x00001000u);
    wi |= int((hi << 18) & 0x00100000u);
    wi |= int((hi << 25) & 0x10000000u);

    const int u0 = get_int_b4(yb->qs, g);      // elements 4g..4g+3
    const int u1 = get_int_b4(yb->qs, 4 + g);  // elements 16+4g..19+4g
    sumi = __dp4a(vi, u0, sumi);
    sumi = __dp4a(wi, u1, sumi);
    sumu = __dp4a(0x01010101, u0, sumu);
    sumu = __dp4a(0x01010101, u1, sumu);
  }
  return half_to_float(ld_u16(wb)) * yb->d * float(sumi - 16 * sumu);
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

__global__ void gemv_q6_K_q8_1(const uint8_t* __restrict__ w,
                               const block_q8_1* __restrict__ y,
                               float* __restrict__ dst, int blocks_per_row,
                               int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = lane;  // Q6_K MMVQ uses VDR_Q6_K_Q8_1_MMVQ = 1.
  float sum = 0.0f;
  for (int b = 0; b < blocks_per_row; ++b) {
    const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 210;
    const block_q8_1* yb = y + (size_t)b * (QK_K / QK8_1);
    sum += vec_dot_q6_K_q8_1(wb, yb, iqs);
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) dst[row] = sum;
}

// Legacy mmvq. Because one weight block is one q8_1 block, a lane can simply
// consume whole blocks (stride 32 lanes) instead of a fixed 16-element slice of
// a 256-element super-block; the activation index is then the flat block index
// `b`, with the tile offset folded in for the batched form.

#define SONICBOOM_DEFINE_LEGACY_MMVQ(suffix, blkbytes, dot)                       \
  __global__ void gemv_##suffix##_q8_1(const uint8_t* __restrict__ w,             \
                                       const block_q8_1* __restrict__ y,          \
                                       float* __restrict__ dst, int blocks_per_row, \
                                       int num_rows) {                            \
    const int lane = threadIdx.x;                                                 \
    const int row = blockIdx.y * blockDim.y + threadIdx.y;                        \
    if (row >= num_rows) return;                                                  \
    float sum = 0.0f;                                                             \
    for (int b = lane; b < blocks_per_row; b += 32) {                             \
      const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * blkbytes;      \
      sum += dot(wb, y + b);                                                      \
    }                                                                             \
    _Pragma("unroll") for (int o = 16; o > 0; o >>= 1)                            \
        sum += __shfl_down_sync(0xffffffffu, sum, o);                             \
    if (lane == 0) dst[row] = sum;                                                \
  }                                                                               \
  template <int T>                                                                \
  __global__ void gemv_##suffix##_q8_1_batched(                                   \
      const uint8_t* __restrict__ w, const block_q8_1* __restrict__ y,            \
      float* __restrict__ dst, int blocks_per_row, int num_rows, int n) {         \
    const int lane = threadIdx.x;                                                 \
    const int row = blockIdx.y * blockDim.y + threadIdx.y;                        \
    if (row >= num_rows) return;                                                  \
    for (int p0 = 0; p0 < n; p0 += T) {                                           \
      const int nt = (n - p0 < T) ? (n - p0) : T;                                 \
      float sum[T];                                                               \
      _Pragma("unroll") for (int t = 0; t < T; ++t) sum[t] = 0.0f;                \
      for (int b = lane; b < blocks_per_row; b += 32) {                           \
        const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * blkbytes;    \
        _Pragma("unroll") for (int t = 0; t < T; ++t)                             \
            if (t < nt) sum[t] += dot(                                            \
                wb, y + (size_t)(p0 + t) * blocks_per_row + b);                   \
      }                                                                           \
      _Pragma("unroll") for (int t = 0; t < T; ++t) {                             \
        if (t < nt) {                                                             \
          float s = sum[t];                                                       \
          _Pragma("unroll") for (int o = 16; o > 0; o >>= 1)                      \
              s += __shfl_down_sync(0xffffffffu, s, o);                           \
          if (lane == 0) dst[(size_t)(p0 + t) * num_rows + row] = s;              \
        }                                                                         \
      }                                                                           \
    }                                                                             \
  }

SONICBOOM_DEFINE_LEGACY_MMVQ(q5_0, 22, vec_dot_q5_0_q8_1)
SONICBOOM_DEFINE_LEGACY_MMVQ(q8_0, 34, vec_dot_q8_0_q8_1)
#undef SONICBOOM_DEFINE_LEGACY_MMVQ

// --- batched warp-per-row mmvq (prefill) ------------------------------------
//
// Y[row, pos] = sum_b vec_dot(W[row,b], Xq[pos,b]) for pos in [0, n).
// Same one-warp-per-row decomposition as the single-column kernels, but the
// weight block W[row,b] is read *once* and reused across a tile of T positions
// (T register accumulators), so the weight's DRAM traffic is amortized over the
// batch instead of re-read once per column. `y` is the q8_1 activation laid out
// position-major: [pos * blocks_per_row + b] * (QK_K / QK8_1) blocks. `dst` is
// [n, num_rows] position-major (matches the [n, rows] prefill layout).

constexpr int GEMV_BATCH_T = 8;

template <int T>
__global__ void gemv_q3_K_q8_1_batched(const uint8_t* __restrict__ w,
                                       const block_q8_1* __restrict__ y,
                                       float* __restrict__ dst,
                                       int blocks_per_row, int num_rows, int n) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = lane & 15;   // QI3_K=16 slices
  const int ib = lane >> 4;    // 2 blocks per warp
  for (int p0 = 0; p0 < n; p0 += T) {
    const int nt = (n - p0 < T) ? (n - p0) : T;
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.0f;
    for (int b = ib; b < blocks_per_row; b += 2) {
      const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 110;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t < nt) {
          const block_q8_1* yb =
              y + ((size_t)(p0 + t) * blocks_per_row + b) * (QK_K / QK8_1);
          sum[t] += vec_dot_q3_K_q8_1(wb, yb, iqs);
        }
      }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < nt) {
        float s = sum[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) dst[(size_t)(p0 + t) * num_rows + row] = s;
      }
    }
  }
}

template <int T>
__global__ void gemv_q4_K_q8_1_batched(const uint8_t* __restrict__ w,
                                       const block_q8_1* __restrict__ y,
                                       float* __restrict__ dst,
                                       int blocks_per_row, int num_rows, int n) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = 2 * (lane & 15);  // QI4_K=32, vdr=2 -> iqs even 0..30
  const int ib = lane >> 4;
  for (int p0 = 0; p0 < n; p0 += T) {
    const int nt = (n - p0 < T) ? (n - p0) : T;
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.0f;
    for (int b = ib; b < blocks_per_row; b += 2) {
      const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 144;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t < nt) {
          const block_q8_1* yb =
              y + ((size_t)(p0 + t) * blocks_per_row + b) * (QK_K / QK8_1);
          sum[t] += vec_dot_q4_K_q8_1(wb, yb, iqs);
        }
      }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < nt) {
        float s = sum[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) dst[(size_t)(p0 + t) * num_rows + row] = s;
      }
    }
  }
}

template <int T>
__global__ void gemv_q5_K_q8_1_batched(const uint8_t* __restrict__ w,
                                       const block_q8_1* __restrict__ y,
                                       float* __restrict__ dst,
                                       int blocks_per_row, int num_rows, int n) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = 2 * (lane & 15);  // QI5_K=32, vdr=2 -> iqs even 0..30
  const int ib = lane >> 4;
  for (int p0 = 0; p0 < n; p0 += T) {
    const int nt = (n - p0 < T) ? (n - p0) : T;
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.0f;
    for (int b = ib; b < blocks_per_row; b += 2) {
      const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 176;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t < nt) {
          const block_q8_1* yb =
              y + ((size_t)(p0 + t) * blocks_per_row + b) * (QK_K / QK8_1);
          sum[t] += vec_dot_q5_K_q8_1(wb, yb, iqs);
        }
      }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < nt) {
        float s = sum[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) dst[(size_t)(p0 + t) * num_rows + row] = s;
      }
    }
  }
}

template <int T>
__global__ void gemv_q6_K_q8_1_batched(const uint8_t* __restrict__ w,
                                       const block_q8_1* __restrict__ y,
                                       float* __restrict__ dst,
                                       int blocks_per_row, int num_rows, int n) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int iqs = lane;  // Q6_K MMVQ uses VDR_Q6_K_Q8_1_MMVQ = 1.
  for (int p0 = 0; p0 < n; p0 += T) {
    const int nt = (n - p0 < T) ? (n - p0) : T;
    float sum[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) sum[t] = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
      const uint8_t* wb = w + ((size_t)row * blocks_per_row + b) * 210;
      #pragma unroll
      for (int t = 0; t < T; ++t) {
        if (t < nt) {
          const block_q8_1* yb = y + ((size_t)(p0 + t) * blocks_per_row + b) * (QK_K / QK8_1);
          sum[t] += vec_dot_q6_K_q8_1(wb, yb, iqs);
        }
      }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
      if (t < nt) {
        float s = sum[t];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) dst[(size_t)(p0 + t) * num_rows + row] = s;
      }
    }
  }
}

} // namespace sonicboom::quant::cuda::q8_1
