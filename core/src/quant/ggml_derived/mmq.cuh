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
// Derived from ggml (llama.cpp), MIT-licensed. The int8-tensor-core "mmq"
// quantized matmul below (Q3_K / Q4_K / Q5_K / Q6_K shared-memory tiled dequant into
// int8 + mma) is ported from the following upstream sources:
//
//   ggml/src/ggml-cuda/mma.cuh                — tile<> + load_ldmatrix + mma
//   ggml/src/ggml-cuda/mmq.cuh                — block_q8_1_mmq, sram_stride,
//                                                process_tile, mul_mat_q,
//                                                write_back_mma
//   ggml/src/ggml-cuda/mmq-config-ampere.cuh  — J set (fallback = false)
//   ggml/src/ggml-cuda/mmq-load-tiles.cuh     — load_tiles_q3/q4/q5/q6_K,
//                                                load_tiles_q5_0/q8_0,
//                                                unpack_scales_q45_K
//   ggml/src/ggml-cuda/mmq-vec-dot.cuh        — vec_dot_q8_0_16_q8_1_mma,
//                                                vec_dot_q8_1_q8_1_mma,
//                                                vec_dot_q8_0_q8_1_mma
//   ggml/src/ggml-cuda/quantize.cu            — quantize_mmq_q8_1
//   ggml/src/ggml-cuda/vecdotq.cuh            — get_int_b2 / get_int_b4
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
// SonicBoom's own additions (dropping the MoE/channels/samples/ids_dst/fallback
// branches — Gemma has M%128==0 and no MoE — plus the host-side launchers below)
// are GPL-3.0-or-later as above.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>    // __half2float, __float2half, __half22float2, __halves2half2
#include <cuda_fp16.hpp>  // half / half2 types, make_half2, arithmetic operators
#include <sm_32_intrinsics.h>  // __vsub4

#include <cstdint>

namespace sonicboom::quant::cuda::mmq {

// --- constants -------------------------------------------------------------

constexpr int QK_K          = 256;
constexpr int QI8_1         = 8;
constexpr int QK8_1_MMQ     = 128;
constexpr int QI3_K         = 16;
constexpr int QR3_K         = 4;
constexpr int QI4_K         = 32;
constexpr int QR4_K         = 2;
constexpr int QI5_K         = 32;
constexpr int QR5_K         = 2;
constexpr int QI6_K         = 32;
constexpr int QK5_0         = 32;
constexpr int QI5_0         = 4;
constexpr int QR5_0         = 2;
constexpr int QK8_0         = 32;
constexpr int QI8_0         = 8;
constexpr int QR8_0         = 1;
constexpr int VDR_Q3_K_Q8_1_MMQ = 2;
constexpr int MMQ_TILE_NE_K = 32;
constexpr int MMQ_TILE_Y_K  = 36;
constexpr int MMQ_ITER_K    = 256;
constexpr int I             = 128;
constexpr int NWARPS        = 8;
constexpr int WARP_SIZE     = 32;
constexpr int CUDA_QUANTIZE_BLOCK_SIZE_MMQ = 128;

// Shared-memory stride (in int32 units) of one row of the weight tile.
// Q3_K: 2*32 qs int32 + 16 float dequant scales + 4 pad = 84.
// Q4_K/Q5_K: 2*32 qs int32 + 8 half2 dequant scales + 4 pad = 76.
// Q5_0/Q8_0: 2*32 qs int32 + 8 float dequant scales + 4 pad = 76 (ggml's
//   GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, whose value coincides with Q8_1's).
// Q6_K: 2*32 qs int32 + 1 float d + 4 int32 packed signed scales + 7 pad = 76.
constexpr int SRAM_STRIDE_Q3_K = 2*MMQ_TILE_NE_K + 16 + 4;
constexpr int SRAM_STRIDE_Q8_1 = 2*MMQ_TILE_NE_K +  8 + 4;
constexpr int SRAM_STRIDE_LEGACY_32 = 2*MMQ_TILE_NE_K + 2*MMQ_TILE_NE_K/QI8_0 + 4;
constexpr int SRAM_STRIDE_Q6_K = 2*MMQ_TILE_NE_K + 1 + 4 + 7;
static_assert(SRAM_STRIDE_LEGACY_32 % 8 == 4, "legacy-32 sram stride padding");

enum MMQType { MMQ_Q3_K = 0, MMQ_Q4_K = 1, MMQ_Q5_K = 2, MMQ_Q6_K = 3,
               MMQ_Q5_0 = 4, MMQ_Q8_0 = 5 };
enum DsLayout { D4 = 0, DS4 = 1, D4_Q8_1 = 2 };

__host__ __device__ constexpr int PAD(int x, int n) { return ((x) + (n) - 1) / (n) * (n); }

// --- quantized weight blocks (raw GGUF byte layout = ggml struct layout) ---

struct block_q3_K {
  uint8_t hmask[32];
  uint8_t qs[64];
  uint8_t scales[12];
  half d;
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K must be 110 bytes");

struct block_q4_K {
  half2 dm;              // { d, dmin }
  uint8_t scales[12];
  uint8_t qs[128];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K must be 144 bytes");

struct block_q5_K {
  half2 dm;              // { d, dmin }
  uint8_t scales[12];
  uint8_t qh[32];
  uint8_t qs[128];
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K must be 176 bytes");

struct block_q6_K {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  half d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K must be 210 bytes");

// Legacy 32-element formats (one fp16 scale per 32 values, no min).
struct block_q5_0 {
  half d;
  uint8_t qh[4];
  uint8_t qs[QK5_0 / 2];
};
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 must be 22 bytes");

struct block_q8_0 {
  half d;
  int8_t qs[QK8_0];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 must be 34 bytes");

struct block_q8_1_mmq {
  union { float d4[4]; half2 ds4[4]; };
  int8_t qs[128];
};
static_assert(sizeof(block_q8_1_mmq) == 144, "block_q8_1_mmq must be 144 bytes");

// --- byte reads (ggml vecdotq.cuh) ----------------------------------------

__device__ __forceinline__ int get_int_b2(const void* x, const int i32) {
  const uint16_t* x16 = (const uint16_t*)x;  // assume at least 2-byte alignment
  int x32 = x16[2*i32 + 0] << 0;
  x32    |= x16[2*i32 + 1] << 16;
  return x32;
}

__device__ __forceinline__ int get_int_b4(const void* x, const int i32) {
  return ((const int*)x)[i32];  // assume at least 4-byte alignment
}

// --- mma tile + primitives (ggml mma.cuh, CUDA path, int32/int8) ----------

template <int I_, int J_>
struct tile {
  static constexpr int I = I_;
  static constexpr int J = J_;
  static constexpr int ne = I * J / 32;
  int x[ne] = {0};

  static __device__ __forceinline__ int get_i(const int l) {
    if constexpr (I == 8 && J == 4) {
      return threadIdx.x / 4;
    } else if constexpr (I == 8 && J == 8) {
      return threadIdx.x / 4;
    } else if constexpr (I == 16 && J == 8) {
      return ((l / 2) * 8) + (threadIdx.x / 4);
    } else {
      return -1;  // unreachable for the tiles used here
    }
  }

  static __device__ __forceinline__ int get_j(const int l) {
    if constexpr (I == 8 && J == 4) {
      return threadIdx.x % 4;
    } else if constexpr (I == 8 && J == 8) {
      return (l * 4) + (threadIdx.x % 4);
    } else if constexpr (I == 16 && J == 8) {
      return ((threadIdx.x % 4) * 2) + (l % 2);
    } else {
      return -1;  // unreachable for the tiles used here
    }
  }
};

template <int I, int J>
__device__ __forceinline__ void load_generic(tile<I, J>& t, const int* xs0, int stride) {
#pragma unroll
  for (int l = 0; l < t.ne; ++l) {
    t.x[l] = xs0[t.get_i(l) * stride + t.get_j(l)];
  }
}

__device__ __forceinline__ void load_ldmatrix(tile<16, 8>& t, const int* xs0, int stride) {
  int* xi = (int*)t.x;
  const int* xs = xs0 + (threadIdx.x % t.I) * stride + (threadIdx.x / t.I) * (t.J / 2);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3])
               : "l"(xs));
}

__device__ __forceinline__ void load_ldmatrix(tile<16, 4>& t, const int* xs0, int stride) {
  int* xi = t.x;
  const int* xs = xs0 + (threadIdx.x % 16) * stride;
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.b16 {%0,%1}, [%2];"
               : "=r"(xi[0]), "=r"(xi[1])
               : "l"(xs));
}

// Ampere (SM_80+) int8 tensor-core mma.
__device__ __forceinline__ void mma(tile<16, 8>& D, const tile<16, 4>& A, const tile<8, 4>& B) {
  asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(B.x[0]));
}

__device__ __forceinline__ void mma(tile<16, 8>& D, const tile<16, 8>& A, const tile<8, 8>& B) {
  asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+r"(D.x[0]), "+r"(D.x[1]), "+r"(D.x[2]), "+r"(D.x[3])
      : "r"(A.x[0]), "r"(A.x[1]), "r"(A.x[2]), "r"(A.x[3]), "r"(B.x[0]), "r"(B.x[1]));
}

// --- Q4_K / Q5_K scale unpack (ggml mmq-load-tiles.cuh) -------------------

__device__ __forceinline__ int unpack_scales_q45_K(const int* scales, const int ksc) {
  return ((scales[(ksc % 2) + (ksc != 0)] >> (4 * (ksc & (ksc / 2)))) & 0x0F0F0F0F)
       | ((scales[ksc / 2] >> (2 * (ksc % 2))) & 0x30303030);
}

// --- load_tiles (ggml mmq-load-tiles.cuh, MMA path, fallback = false) -----

template <int J>
__device__ __forceinline__ void load_tiles_q3_K(const char* x, int* x_tile, int kbx0, int stride) {
  constexpr int sram_stride = SRAM_STRIDE_Q3_K;

  int*   x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2*MMQ_TILE_NE_K);

  constexpr int threads_per_row = MMQ_ITER_K / (4 * QR3_K);  // 16
  constexpr int nrows = WARP_SIZE / threads_per_row;         // 2
  const int kqsx = threadIdx.x % threads_per_row;

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += nrows*NWARPS) {
    int i = i0 + threadIdx.y*nrows + threadIdx.x/threads_per_row;
    const block_q3_K* bxi = (const block_q3_K*)x + kbx0 + i*stride;
    const int x_ql_0 = get_int_b2(bxi->qs, kqsx);
    const int x_qh_0 = get_int_b2(bxi->hmask, kqsx % (QI3_K/2)) >> (4 * (kqsx / (QI3_K/2)));
#pragma unroll
    for (int l = 0; l < QR3_K; ++l) {
      const int k = (kqsx/8)*32 + l*8 + kqsx % 8;
      const int x_ql_k =  (x_ql_0 >> (2*l))       & 0x03030303;
      const int x_qh_k = ((x_qh_0 >>    l)  << 2) & 0x04040404;
      const int x_qs_k = __vsub4(x_ql_k | x_qh_k, 0x04040404);
      x_qs[i*sram_stride + k] = x_qs_k;
    }
  }

  constexpr int rows_per_warp = WARP_SIZE / 4;  // 8
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS*rows_per_warp) {
    int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/4;
    const block_q3_K* bxi = (const block_q3_K*)x + kbx0 + i*stride;
    const int ksc = threadIdx.x % 4;
    const int sc_low = (get_int_b2(bxi->scales, ksc % (QI3_K/8)) >> (4 * (ksc / (QI3_K/8)))) & 0x0F0F0F0F;
    const int sc_high = ((get_int_b2(bxi->scales, QI3_K/8) >> (2*ksc)) << 4) & 0x30303030;
    const int sc = __vsub4(sc_low | sc_high, 0x20202020);
    const int8_t* sc8 = (const int8_t*)&sc;
    const float d = bxi->d;
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
      x_df[i*sram_stride + sizeof(int)*ksc + l] = d*sc8[l];
    }
  }
}

template <int J>
__device__ __forceinline__ void load_tiles_q4_K(const char* x, int* x_tile, int kbx0, int stride) {
  constexpr int sram_stride = SRAM_STRIDE_Q8_1;

  int*   x_qs = (int*)x_tile;
  half2* x_dm = (half2*)(x_qs + 2*MMQ_TILE_NE_K);

  constexpr int threads_per_row = MMQ_ITER_K / (4 * QR4_K);  // 32
  constexpr int nrows = WARP_SIZE / threads_per_row;         // 1
  const int txi = WARP_SIZE > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += nrows*NWARPS) {
    int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
    const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i*stride;
    const int qs0 = get_int_b4(bxi->qs, txi);
    x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
    x_qs[i*sram_stride + 16*(txi/8) + txi % 8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
  }

  constexpr int rows_per_warp = WARP_SIZE / 2;  // 16
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS*rows_per_warp) {
    int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
    const block_q4_K* bxi = (const block_q4_K*)x + kbx0 + i*stride;
    const int* scales = (const int*)bxi->scales;
    const int ksc = threadIdx.x % 2;
    const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
    const int m32  = unpack_scales_q45_K(scales, ksc + 2);
    const uint8_t* sc8 = (const uint8_t*)&sc32;
    const uint8_t* m8  = (const uint8_t*)&m32;
    const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
      x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
    }
  }
}

template <int J>
__device__ __forceinline__ void load_tiles_q5_K(const char* x, int* x_tile, int kbx0, int stride) {
  constexpr int sram_stride = SRAM_STRIDE_Q8_1;

  int*   x_qs = (int*)x_tile;
  half2* x_dm = (half2*)(x_qs + 2*MMQ_TILE_NE_K);

  constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);  // 32
  constexpr int nrows = WARP_SIZE / threads_per_row;         // 1
  const int txi = WARP_SIZE > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += nrows*NWARPS) {
    int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
    const block_q5_K* bxi = (const block_q5_K*)x + kbx0 + i*stride;
    const int ky = QR5_K*txi;                 // 2*txi
    const int ql = get_int_b4(bxi->qs, txi);
    const int ql0 = (ql >> 0) & 0x0F0F0F0F;
    const int ql1 = (ql >> 4) & 0x0F0F0F0F;
    const int qh = get_int_b4(bxi->qh, txi % (QI5_K/4));
    const int qh0 = ((qh >> (2 * (txi / (QI5_K/4)) + 0)) << 4) & 0x10101010;
    const int qh1 = ((qh >> (2 * (txi / (QI5_K/4)) + 1)) << 4) & 0x10101010;
    const int kq0 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + 0;
    const int kq1 = ky - ky % (QI5_K/2) + txi % (QI5_K/4) + QI5_K/4;
    x_qs[i*sram_stride + kq0] = ql0 | qh0;
    x_qs[i*sram_stride + kq1] = ql1 | qh1;
  }

  constexpr int rows_per_warp = WARP_SIZE / 2;  // 16
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS*rows_per_warp) {
    int i = (i0 + threadIdx.y*rows_per_warp + threadIdx.x/2) % I;
    const block_q5_K* bxi = (const block_q5_K*)x + kbx0 + i*stride;
    const int* scales = (const int*)bxi->scales;
    const int ksc = threadIdx.x % 2;
    const int sc32 = unpack_scales_q45_K(scales, ksc + 0);
    const int m32  = unpack_scales_q45_K(scales, ksc + 2);
    const uint8_t* sc8 = (const uint8_t*)&sc32;
    const uint8_t* m8  = (const uint8_t*)&m32;
    const half2 dm = bxi->dm * make_half2(1.0f, -1.0f);
#pragma unroll
    for (int l = 0; l < int(sizeof(int)); ++l) {
      x_dm[i*sram_stride + sizeof(int)*ksc + l] = dm*make_half2(sc8[l], m8[l]);
    }
  }
}

template <int J>
__device__ __forceinline__ void load_tiles_q6_K(const char* x, int* x_tile,
                                                 int kbx0, int stride) {
  constexpr int sram_stride = SRAM_STRIDE_Q6_K;
  int* x_qs = x_tile;
  float* x_df = reinterpret_cast<float*>(x_qs + 2 * MMQ_TILE_NE_K);
  int* x_sc = reinterpret_cast<int*>(x_df + 1);
  constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_K);
  constexpr int nrows = WARP_SIZE / threads_per_row;
  const int txi = WARP_SIZE > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += nrows * NWARPS) {
    const int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y * nrows + threadIdx.x / threads_per_row);
    const block_q6_K* bxi = reinterpret_cast<const block_q6_K*>(x) + kbx0 + i * stride;
    const int ql = get_int_b2(bxi->ql, txi);
    const int ql0 = (ql >> 0) & 0x0F0F0F0F;
    const int ql1 = (ql >> 4) & 0x0F0F0F0F;
    const int qh = get_int_b2(bxi->qh, (QI6_K / 4) * (txi / (QI6_K / 2)) + txi % (QI6_K / 4));
    const int qh0 = ((qh >> ((txi & 0x08) >> 2)) << 4) & 0x30303030;
    const int qh1 = (qh >> ((txi & 0x08) >> 2)) & 0x30303030;
    const int kq0 = 2 * txi - txi % (QI6_K / 2);
    const int kq1 = kq0 + QI6_K / 2;
    x_qs[i * sram_stride + kq0] = __vsub4(ql0 | qh0, 0x20202020);
    x_qs[i * sram_stride + kq1] = __vsub4(ql1 | qh1, 0x20202020);
  }

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS * WARP_SIZE) {
    const int i = (i0 + threadIdx.y * WARP_SIZE + threadIdx.x) % I;
    const block_q6_K* bxi = reinterpret_cast<const block_q6_K*>(x) + kbx0 + i * stride;
    x_df[i * sram_stride] = float(bxi->d);
  }

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS * (WARP_SIZE / 4)) {
    const int i = (i0 + threadIdx.y * (WARP_SIZE / 4) + threadIdx.x / 4) % I;
    const block_q6_K* bxi = reinterpret_cast<const block_q6_K*>(x) + kbx0 + i * stride;
    const int scale_word = threadIdx.x % 4;
    // block_q6_K is 210 bytes, so every other block starts 2 mod 4. Use the
    // two-byte-aligned loader for its packed scale words instead of get_int_b4.
    x_sc[i * sram_stride + scale_word] = get_int_b2(bxi->scales, scale_word);
  }
}

// Q5_0 (ggml mmq-load-tiles.cuh, Turing/Ampere MMA path). One 256-value K-block
// is 8 ggml blocks of 32 values; block kbx owns sram ints [kbx*8, kbx*8+8) and
// its scale lands in x_df[kbxd], kbxd == the block index within the row.
template <int J>
__device__ __forceinline__ void load_tiles_q5_0(const char* x, int* x_tile,
                                                int kbx0, int stride) {
  constexpr int sram_stride = SRAM_STRIDE_LEGACY_32;
  int*   x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2*MMQ_TILE_NE_K);

  constexpr int threads_per_row = MMQ_ITER_K / (4 * QR5_0);  // 32
  constexpr int nrows = WARP_SIZE / threads_per_row;         // 1
  const int txi  = WARP_SIZE > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
  const int kbx  = txi / QI5_0;
  const int kqsx = txi % QI5_0;

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += nrows*NWARPS) {
    const int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
    const block_q5_0* bxi = (const block_q5_0*)x + kbx0 + i*stride + kbx;

    const int ql = get_int_b2(bxi->qs, kqsx);            // 16 low nibbles
    const int qh = get_int_b2(bxi->qh, 0) >> (4 * kqsx); // 4 high bits

    int qs0 = (ql >>  0)   & 0x0F0F0F0F;
    qs0    |= (qh <<  4)   & 0x00000010;
    qs0    |= (qh << 11)   & 0x00001000;
    qs0    |= (qh << 18)   & 0x00100000;
    qs0    |= (qh << 25)   & 0x10000000;
    qs0     = __vsub4(qs0, 0x10101010);                  // subtract 16

    int qs1 = (ql >>  4)   & 0x0F0F0F0F;                 // 16 high nibbles
    qs1    |= (qh >> 12)   & 0x00000010;
    qs1    |= (qh >>  5)   & 0x00001000;
    qs1    |= (qh <<  2)   & 0x00100000;
    qs1    |= (qh <<  9)   & 0x10000000;
    qs1     = __vsub4(qs1, 0x10101010);

    x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + 0]     = qs0;
    x_qs[i*sram_stride + kbx*(2*QI5_0) + kqsx + QI5_0] = qs1;
  }

  constexpr int blocks_per_tile_x_row = MMQ_TILE_NE_K / QI5_0;  // 8
  constexpr int rows_per_warp = WARP_SIZE / blocks_per_tile_x_row;  // 4
  const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS*rows_per_warp) {
    const int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/blocks_per_tile_x_row;
    const block_q5_0* bxi = (const block_q5_0*)x + kbx0 + i*stride + kbxd;
    x_df[i*sram_stride + kbxd] = float(bxi->d);
  }
}

// Q8_0 (ggml mmq-load-tiles.cuh, Turing/Ampere MMA path). Same row geometry as
// Q5_0, but the two x_qs halves hold blocks [0,3] and [4,7] respectively, so a
// k00 == MMQ_TILE_NE_K call reads the second half's ints.
template <int J>
__device__ __forceinline__ void load_tiles_q8_0(const char* x, int* x_tile,
                                                int kbx0, int stride) {
  constexpr int sram_stride = SRAM_STRIDE_LEGACY_32;
  int*   x_qs = (int*)x_tile;
  float* x_df = (float*)(x_qs + 2*MMQ_TILE_NE_K);

  constexpr int threads_per_row = 32;  // MMQ_ITER_K/(4*QR8_0) would be 64
  constexpr int nrows = WARP_SIZE / threads_per_row;  // 1
  const int txi  = WARP_SIZE > threads_per_row ? threadIdx.x % threads_per_row : threadIdx.x;
  const int kbx  = txi / QI8_0;
  const int kqsx = txi % QI8_0;

#pragma unroll
  for (int i0 = 0; i0 < I; i0 += nrows*NWARPS) {
    const int i = i0 + (nrows == 1 ? threadIdx.y : threadIdx.y*nrows + threadIdx.x/threads_per_row);
    const block_q8_0* bxi = (const block_q8_0*)x + kbx0 + i*stride + kbx;
    x_qs[i*sram_stride + 0              + txi] = get_int_b2(bxi[0].qs, kqsx);
    x_qs[i*sram_stride + MMQ_TILE_NE_K  + txi] = get_int_b2(bxi[MMQ_TILE_NE_K/QI8_0].qs, kqsx);
  }

  constexpr int blocks_per_tile_x_row = 2*MMQ_TILE_NE_K / QI8_0;  // 8
  constexpr int rows_per_warp = WARP_SIZE / blocks_per_tile_x_row;  // 4
  const int kbxd = threadIdx.x % blocks_per_tile_x_row;
#pragma unroll
  for (int i0 = 0; i0 < I; i0 += NWARPS*rows_per_warp) {
    const int i = i0 + threadIdx.y*rows_per_warp + threadIdx.x/blocks_per_tile_x_row;
    const block_q8_0* bxi = (const block_q8_0*)x + kbx0 + i*stride + kbxd;
    x_df[i*sram_stride + kbxd] = float(bxi->d);
  }
}

template <int Type, int J>
__device__ __forceinline__ void load_tiles(const char* x, int* x_tile, int kbx0, int stride) {
  if constexpr (Type == MMQ_Q3_K) {
    load_tiles_q3_K<J>(x, x_tile, kbx0, stride);
  } else if constexpr (Type == MMQ_Q4_K) {
    load_tiles_q4_K<J>(x, x_tile, kbx0, stride);
  } else if constexpr (Type == MMQ_Q5_K) {
    load_tiles_q5_K<J>(x, x_tile, kbx0, stride);
  } else if constexpr (Type == MMQ_Q5_0) {
    load_tiles_q5_0<J>(x, x_tile, kbx0, stride);
  } else if constexpr (Type == MMQ_Q8_0) {
    load_tiles_q8_0<J>(x, x_tile, kbx0, stride);
  } else {
    load_tiles_q6_K<J>(x, x_tile, kbx0, stride);
  }
}

// --- vec_dot (ggml mmq-vec-dot.cuh, Turing/Ampere mma path) ---------------

// Q3_K weights -> activation quantized to q8_0_16 (d4 float scales).
template <int J>
__device__ __forceinline__ void vec_dot_q8_0_16_q8_1_mma(const int* x, const int* y, float* sum, int k00) {
  typedef tile<16, 4> tile_A;
  typedef tile<16, 8> tile_A_8;
  typedef tile<8, 4> tile_B;
  typedef tile<16, 8> tile_C;

  constexpr int sram_stride = SRAM_STRIDE_Q3_K;
  constexpr int rows_per_warp = (J >= 48 && J % 16 == 0) ? 32 : 16;
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

  const int*   x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + 2*MMQ_TILE_NE_K;
  const int*   y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_A::I);

  tile_A A[ntx][8];
  float dA[ntx][tile_C::ne/2][8];

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
      const int k0 = k00 + k01;
      load_ldmatrix(((tile_A_8*)A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
    }
#pragma unroll
    for (int l = 0; l < tile_C::ne/2; ++l) {
      const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);
#pragma unroll
      for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;
        dA[n][l][k01/4] = x_df[i*sram_stride + k0/4];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
      tile_B B[2];
      float dB[tile_C::ne/2];
      load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0), MMQ_TILE_Y_K);
      load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);
#pragma unroll
      for (int l = 0; l < tile_C::ne/2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
      }
#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C[2];
        mma(C[0], A[n][k01/4 + 0], B[0]);
        mma(C[1], A[n][k01/4 + 1], B[1]);
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0/tile_C::J + n)*tile_C::ne + l] +=
              dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
        }
      }
    }
  }
}

// Q4_K / Q5_K weights -> activation quantized to q8_1 (ds4 half2 scales).
template <int J>
__device__ __forceinline__ void vec_dot_q8_1_q8_1_mma(const int* x, const int* y, float* sum, int k00) {
  typedef tile<16, 8> tile_A;
  typedef tile<8, 8> tile_B;
  typedef tile<16, 8> tile_C;

  constexpr int sram_stride = SRAM_STRIDE_Q8_1;
  constexpr int rows_per_warp = (J >= 48 && J % 16 == 0) ? 32 : 16;
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

  const int*   x_qs = (const int*)x;
  const half2* x_dm = (const half2*)x_qs + 2*MMQ_TILE_NE_K;
  const int*   y_qs = (const int*)y + 4;
  const half2* y_dm = (const half2*)y;

  tile_A A[ntx][MMQ_TILE_NE_K/QI8_1];
  float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
      const int k0 = k00 + k01;
      load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
    }
#pragma unroll
    for (int l = 0; l < tile_C::ne/2; ++l) {
      const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
#pragma unroll
      for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;
        dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
      tile_B B;
      float2 dsB[tile_C::ne/2];
      load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
#pragma unroll
      for (int l = 0; l < tile_C::ne/2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
      }
#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01/QI8_1], B);
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0/tile_C::J + n)*tile_C::ne + l] +=
              dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l] + dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
        }
      }
    }
  }
}

template <int J>
__device__ __forceinline__ void vec_dot_q6_K_q8_1_mma(const int* x, const int* y,
                                                        float* sum, int k00) {
  typedef tile<16, 4> tile_A;
  typedef tile<8, 4> tile_B;
  typedef tile<16, 8> tile_C;
  constexpr int sram_stride = SRAM_STRIDE_Q6_K;
  constexpr int rows_per_warp = (J >= 48 && J % 16 == 0) ? 32 : 16;
  constexpr int ntx = rows_per_warp / tile_C::I;
  y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);
  const int* x_qs = x;
  const float* x_df = reinterpret_cast<const float*>(x_qs + 2 * MMQ_TILE_NE_K);
  const int* x_sc = reinterpret_cast<const int*>(x_df + 1);
  const int* y_qs = y + 4;
  const float* y_df = reinterpret_cast<const float*>(y);
  const int i0 = (threadIdx.y / ntx) * (ntx * tile_A::I);
  tile_A A[ntx][8];
  int scA[ntx][tile_C::ne / 2][8];
  float dA[ntx][tile_C::ne / 2];

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
      const int k0 = k00 + k01;
      load_ldmatrix(A[n][k01 / 4 + 0], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
      load_ldmatrix(A[n][k01 / 4 + 1], x_qs + (i0 + n * tile_A::I) * sram_stride + k0 + tile_A::J, sram_stride);
    }
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
      const int k0 = k00 + k01;
#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int i = i0 + n * tile_C::I + tile_C::get_i(2 * l);
        const int packed = x_sc[i * sram_stride + k0 / 16];
        const int8_t* sc = reinterpret_cast<const int8_t*>(&packed);
#pragma unroll
        for (int ksc = 0; ksc < 4; ++ksc) scA[n][l][k01 / 4 + ksc] = sc[ksc];
      }
    }
#pragma unroll
    for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_C::I + tile_C::get_i(2 * l);
      dA[n][l] = x_df[i * sram_stride];
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
    float tmp[ntx][tile_C::ne] = {};
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
      tile_B B[2];
      float dB[tile_C::ne / 2];
      load_generic(B[0], y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
      load_generic(B[1], y_qs + j0 * MMQ_TILE_Y_K + k01 + tile_B::J, MMQ_TILE_Y_K);
#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
      }
#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C[2];
        mma(C[0], A[n][k01 / 4 + 0], B[0]);
        mma(C[1], A[n][k01 / 4 + 1], B[1]);
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l)
          tmp[n][l] += (C[0].x[l] * scA[n][l / 2][k01 / 4 + 0] +
                        C[1].x[l] * scA[n][l / 2][k01 / 4 + 1]) * dB[l % 2];
      }
    }
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l)
        sum[(j0 / tile_C::J + n) * tile_C::ne + l] += tmp[n][l] * dA[n][l / 2];
    }
  }
}

// Q5_0 / Q8_0 weights -> activation quantized to block_q8_1_mmq with float
// (D4) scales (ggml vec_dot_q8_0_q8_1_mma, ds_layout = MMQ_Q8_1_DS_LAYOUT_D4).
// Same mma shape as the Q4_K/Q5_K path; the weight side contributes one float
// scale per 32 values and no min term, and the activation side contributes one
// float per 32 values instead of a half2 {d, s}.
template <int J>
__device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(const int* x, const int* y,
                                                      float* sum, int k00) {
  typedef tile<16, 8> tile_A;
  typedef tile<8, 8> tile_B;
  typedef tile<16, 8> tile_C;

  constexpr int sram_stride = SRAM_STRIDE_LEGACY_32;
  constexpr int rows_per_warp = (J >= 48 && J % 16 == 0) ? 32 : 16;
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

  const int*   x_qs = x;
  const float* x_df = (const float*)x_qs + 2*MMQ_TILE_NE_K;
  const int*   y_qs = y + 4;
  const float* y_df = (const float*)y;

  tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
  float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int n = 0; n < ntx; ++n) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
      const int k0 = k00 + k01;
      load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
    }
#pragma unroll
    for (int l = 0; l < tile_C::ne/2; ++l) {
      const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);
#pragma unroll
      for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;
        dA[n][l][k01/QI8_0] = x_df[i*sram_stride + k0/QI8_0];
      }
    }
  }

#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
      tile_B B;
      float dB[tile_C::ne/2];
      load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);
#pragma unroll
      for (int l = 0; l < tile_C::ne/2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
      }
#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01/QI8_0], B);
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0/tile_C::J + n)*tile_C::ne + l] +=
              C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
        }
      }
    }
  }
}

template <int Type, int J>
__device__ __forceinline__ void vec_dot(const int* x, const int* y, float* sum, int k00) {
  if constexpr (Type == MMQ_Q3_K) {
    vec_dot_q8_0_16_q8_1_mma<J>(x, y, sum, k00);
  } else if constexpr (Type == MMQ_Q6_K) {
    vec_dot_q6_K_q8_1_mma<J>(x, y, sum, k00);
  } else if constexpr (Type == MMQ_Q5_0 || Type == MMQ_Q8_0) {
    vec_dot_q8_0_q8_1_mma<J>(x, y, sum, k00);
  } else {
    vec_dot_q8_1_q8_1_mma<J>(x, y, sum, k00);
  }
}

// --- write_back (ggml mmq.cuh, no ids_dst, fallback = false) --------------

template <int J>
__device__ __forceinline__ void write_back_mma(const float* sum, float* dst, int stride, int j_max) {
  typedef tile<16, 8> tile_C;
  constexpr int rows_per_warp = (J >= 48 && J % 16 == 0) ? 32 : 16;
  constexpr int ntx = rows_per_warp / tile_C::I;
  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx)*tile_C::J + tile_C::get_j(l);
        if (j > j_max) continue;
        const int i = i0 + n*tile_C::I + tile_C::get_i(l);
        dst[j*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
      }
    }
  }
}

// --- activation quantization to block_q8_1_mmq (ggml quantize.cu) ---------

template <int ds>  // D4 (ggml), D4_Q8_1 (Q6 fallback), or DS4 (Q4_K/Q5_K)
__global__ void quantize_mmq_q8_1(const float* __restrict__ x, block_q8_1_mmq* __restrict__ y,
                                  const int K, const int n) {
  constexpr int vals_per_scale = 32;

  const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
  if (i0 >= K) return;

  const int64_t base_idx = (int64_t)blockIdx.x * K;
  const float4* x4 = (const float4*)x;

  const int64_t k_block = i0 / QK8_1_MMQ;
  const int64_t iqs     = i0 % QK8_1_MMQ;

  const float4 xi = x4[(base_idx + i0)/4];
  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));

#pragma unroll
  for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
  }

  float sum = 0.0f;
  if constexpr (ds == DS4) {
    sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
      sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
    }
  }

  char4 q;
  float d;
  if constexpr (ds == D4_Q8_1) {
    d = amax / 127.0f;
    q.x = roundf(xi.x / d);
    q.y = roundf(xi.y / d);
    q.z = roundf(xi.z / d);
    q.w = roundf(xi.w / d);
  } else {
    const float d_inv = 127.0f / amax;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    d = 1.0f / d_inv;
  }

  const int64_t ib = k_block*n + blockIdx.x;
  char4* yqs4 = (char4*)y[ib].qs;
  yqs4[iqs/4] = q;

  if (iqs % 32 == 0) {
    if constexpr (ds == DS4) {
      y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
      y[ib].d4[iqs/32] = d;
    }
  }
}

// --- matmul kernel (ggml mmq.cuh, non-stream-k, fallback = false) ---------

// Values per weight block: 256 for the K-quants (QK_K), 32 for the legacy
// 32-element formats. A 256-value iteration is therefore 1 weight block or 8.
template <int Type>
__host__ __device__ constexpr int qk_of() {
  return (Type == MMQ_Q5_0 || Type == MMQ_Q8_0) ? QK5_0 : QK_K;
}

template <int Type, int J>
__global__ void mul_mat_q(const char* __restrict__ x, const int* __restrict__ y,
                          float* __restrict__ dst,
                          const int blocks_per_row, const int ncols_dst, const int ncols_y,
                          const int stride_col_dst) {
  extern __shared__ int data_mul_mat_q[];
  int* tile_y = data_mul_mat_q;
  int* tile_x = tile_y + PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS);

  constexpr int sz = (int)(sizeof(block_q8_1_mmq) / sizeof(int));  // 36
  constexpr int qk = qk_of<Type>();
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;   // 1 (K-quant) or 8 (legacy)

  const int jt = blockIdx.y;
  const int it = blockIdx.x;

  const int offset_y   = jt * J * sz;
  const int offset_dst = jt * J * stride_col_dst + it * I;
  const int offset_x   = it * I * blocks_per_row;

  const int tile_y_max_j = ncols_dst - jt*J - 1;

  float sum[J*I/(WARP_SIZE*NWARPS)] = {0.0f};

  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += blocks_per_iter) {
    load_tiles<Type, J>(x, tile_x, offset_x + kb0, blocks_per_row);
    {
      const int* by0 = y + offset_y + ncols_y * (kb0 * qk / QK8_1_MMQ) * sz;
#pragma unroll
      for (int l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += WARP_SIZE*NWARPS) {
        const int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }
    __syncthreads();
    vec_dot<Type, J>(tile_x, tile_y, sum, 0);
    __syncthreads();
    {
      const int* by0 = y + offset_y + ncols_y * ((kb0 * qk / QK8_1_MMQ) * sz + sz);
#pragma unroll
      for (int l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += WARP_SIZE*NWARPS) {
        const int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }
    __syncthreads();
    vec_dot<Type, J>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
  }

  write_back_mma<J>(sum, dst + offset_dst, stride_col_dst, tile_y_max_j);
}

// --- matmul kernel, stream-k work partitioning (ggml mmq.cuh) -------------
//
// ggml's "stream-k" partitioning (arXiv:2301.03598), imported with the
// stream_k_fixup kernel. A block owns a contiguous range of the continuous
// (it, jt, kb0) index space `kbc`; every iteration that covers a whole output
// tile writes dst directly, and a block's last iteration, if it covers only
// part of a tile, writes its partial sums to tmp_fixup[blockIdx.x*J*I]
// instead (two blocks writing the same tile element would race). The fixup
// kernel then adds the preceding blocks' partials into dst, walking back over
// the blocks that end where this one starts.
//
// SonicBoom's adaptation (same as the tiled kernel above): the MoE
// (ids_dst/expert_bounds), channels and samples dimensions are dropped — this
// runtime's matmul is a plain [M,K] x [K,n] product — so col_low/col_high/
// col_diff and the ids table reduce to ncols_dst and the identity, and the
// kbc space is (it, jt, kb0) over nty*ntx*blocks_per_row.

// One tile's contribution over a K range [kb0_start, kb0_stop). When `fixup`
// is true the sums go to tmp[blockIdx.x*J*I] instead of dst (the caller's last,
// possibly partial, tile); the layout of that buffer is write_back_mma's dst
// layout at stride I, which is what the fixup kernel reads.
template <int Type, int J>
__device__ __forceinline__ void mul_mat_q_process_tile(
    const char* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ dst, float* __restrict__ fixup_dst,
    const int blocks_per_row, const int ncols_dst, const int ncols_y,
    const int stride_col_dst, const int it, const int jt,
    const int kb0_start, const int kb0_stop, const bool fixup) {
  extern __shared__ int data_mul_mat_q[];
  int* tile_y = data_mul_mat_q;
  int* tile_x = tile_y + PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS);

  constexpr int sz = (int)(sizeof(block_q8_1_mmq) / sizeof(int));  // 36
  constexpr int qk = qk_of<Type>();
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;   // 1 (K-quant) or 8 (legacy)

  const int offset_y   = jt * J * sz;
  const int offset_x   = it * I * blocks_per_row;

  float sum[J*I/(WARP_SIZE*NWARPS)] = {0.0f};

  for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles<Type, J>(x, tile_x, offset_x + kb0, blocks_per_row);
    {
      const int* by0 = y + offset_y + ncols_y * (kb0 * qk / QK8_1_MMQ) * sz;
#pragma unroll
      for (int l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += WARP_SIZE*NWARPS) {
        const int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }
    __syncthreads();
    vec_dot<Type, J>(tile_x, tile_y, sum, 0);
    __syncthreads();
    {
      const int* by0 = y + offset_y + ncols_y * ((kb0 * qk / QK8_1_MMQ) * sz + sz);
#pragma unroll
      for (int l0 = 0; l0 < J*MMQ_TILE_Y_K; l0 += WARP_SIZE*NWARPS) {
        const int l = l0 + threadIdx.y*WARP_SIZE + threadIdx.x;
        tile_y[l] = by0[l];
      }
    }
    __syncthreads();
    vec_dot<Type, J>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
  }

  if (fixup) {
    // The whole J x I sum block, at the tile-contiguous stride I: every output
    // element is written, including the columns past ncols_dst (they are zero
    // and the fixup reads only j <= j_max). `fixup_dst` is this call's partial
    // slot — per block for upstream stream-k, per work unit for the fixed
    // split-K schedule below.
    write_back_mma<J>(sum, fixup_dst, I, J - 1);
  } else {
    const int offset_dst = jt * J * stride_col_dst + it * I;
    const int tile_y_max_j = ncols_dst - jt*J - 1;
    write_back_mma<J>(sum, dst + offset_dst, stride_col_dst, tile_y_max_j);
  }
}

template <int Type, int J>
__global__ void mul_mat_q_stream_k(const char* __restrict__ x,
                                   const int* __restrict__ y,
                                   float* __restrict__ dst,
                                   float* __restrict__ tmp_fixup,
                                   const int blocks_per_row, const int ncols_dst,
                                   const int ncols_y, const int stride_col_dst,
                                   const int ntx, const int nty) {
  constexpr int qk = qk_of<Type>();
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;

  // kbc == k block continuous, current index in continuous ijk space.
  const int64_t total = int64_t(ntx) * nty * blocks_per_row;
  int kbc      = int(int64_t(blockIdx.x)     * total / gridDim.x);
  int kbc_stop = int(int64_t(blockIdx.x + 1) * total / gridDim.x);

  kbc      -= (kbc      % blocks_per_row) % blocks_per_iter;
  kbc_stop -= (kbc_stop % blocks_per_row) % blocks_per_iter;

  // kb0 == k index when doing the matrix multiplication for an output tile.
  int kb0_start = kbc % blocks_per_row;
  int kb0_stop  = min(blocks_per_row, kb0_start + (kbc_stop - kbc));
  while (kbc < kbc_stop && kb0_stop == blocks_per_row) {
    const int tmp = kbc / blocks_per_row;
    const int jt = tmp % ntx;
    const int it = tmp / ntx;
    mul_mat_q_process_tile<Type, J>(x, y, dst,
                                    tmp_fixup + int64_t(blockIdx.x)*(J*I),
                                    blocks_per_row, ncols_dst, ncols_y,
                                    stride_col_dst, it, jt, kb0_start, kb0_stop,
                                    /*fixup=*/false);
    kbc += blocks_per_row;
    kbc -= kbc % blocks_per_row;
    kb0_start = 0;
    kb0_stop  = min(blocks_per_row, kbc_stop - kbc);
  }

  if (kbc >= kbc_stop) return;

  const int tmp = kbc / blocks_per_row;
  const int jt = tmp % ntx;
  const int it = tmp / ntx;
  mul_mat_q_process_tile<Type, J>(x, y, dst,
                                  tmp_fixup + int64_t(blockIdx.x)*(J*I),
                                  blocks_per_row, ncols_dst, ncols_y,
                                  stride_col_dst, it, jt, kb0_start, kb0_stop,
                                  /*fixup=*/true);
}

// --- N-independent fixed split-K schedule ----------------------------------
//
// Upstream stream-k's split points are derived from the continuous index total
// = ntx*nty*blocks_per_row, so they move when the number of columns (hence
// nty) changes. That breaks batch invariance: the same activation rows
// prefilled in one segment and in several prefix-cache segments would be
// reduced in different orders. The schedule below keeps the work decomposition
// fixed per *weight shape*: every output tile's K range is divided into the
// same `parts` pieces, chosen from (M, K) and the SM count only. Work units
// (it, jt, part) are then distributed over a 1D grid, so the K partition — and
// therefore the per-row result — does not depend on n.

template <int Type, int J>
__global__ void mul_mat_q_split_k(const char* __restrict__ x,
                                  const int* __restrict__ y,
                                  float* __restrict__ dst,
                                  float* __restrict__ tmp_partial,
                                  const int blocks_per_row, const int ncols_dst,
                                  const int ncols_y, const int stride_col_dst,
                                  const int ntx, const int nty, const int parts,
                                  const int bpr_aligned) {
  constexpr int qk = qk_of<Type>();
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;
  const int64_t units = int64_t(ntx) * nty * parts;
  for (int64_t u = blockIdx.x; u < units; u += gridDim.x) {
    const int it = int(u % nty);
    const int64_t r = u / nty;
    const int jt = int(r % ntx);
    const int p  = int(r / ntx);
    int kb0 = (int((int64_t(p)     * bpr_aligned) / parts) / blocks_per_iter) * blocks_per_iter;
    int kb1 = (int((int64_t(p + 1) * bpr_aligned) / parts) / blocks_per_iter) * blocks_per_iter;
    if (p + 1 == parts) kb1 = bpr_aligned;
    if (kb0 == kb1) continue;
    mul_mat_q_process_tile<Type, J>(x, y, dst,
                                    tmp_partial + u*(J*I), blocks_per_row,
                                    ncols_dst, ncols_y, stride_col_dst, it, jt,
                                    kb0, kb1, /*fixup=*/true);
  }
}

template <int Type, int J>
__global__ void mul_mat_q_split_k_fixup(float* __restrict__ dst,
                                        const float* __restrict__ tmp_partial,
                                        const int ncols_dst,
                                        const int stride_col_dst, const int ntx,
                                        const int nty, const int parts) {
  constexpr int NWARPS_F = NWARPS / 2;
  const int t = blockIdx.x;
  const int it = t % nty;
  const int jt = t / nty;
  const int i = blockIdx.y*WARP_SIZE + threadIdx.x;
  const int tiles = ntx * nty;

  float sum[J / NWARPS_F] = {0.0f};
  // Partial slot of (tile t, part p) is work unit u = p*tiles + t.
  // Sum the parts in a fixed order (ascending K), independent of scheduling.
  for (int p = 0; p < parts; ++p) {
    const float* s = tmp_partial + int64_t(p)*tiles*(J*I) + int64_t(t)*(J*I);
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += NWARPS_F) {
      const int j = j0 + threadIdx.y;
      sum[j0/NWARPS_F] += s[j*I + i];
    }
  }

  dst += jt*J*stride_col_dst + it*I;
  const int j_max = ncols_dst - jt*J - 1;
#pragma unroll
  for (int j0 = 0; j0 < J; j0 += NWARPS_F) {
    const int j = j0 + threadIdx.y;
    if (j > j_max) return;
    dst[j*stride_col_dst + i] = sum[j0/NWARPS_F];
  }
}

template <int Type, int J>
__global__ void mul_mat_q_stream_k_fixup(
    float* __restrict__ dst, const float* __restrict__ tmp_last_tile,
    const int blocks_per_row, const int ncols_dst, const int stride_col_dst,
    const int ntx, const int nty) {
  constexpr int NWARPS_F = NWARPS / 2;  // the fixup's block is half the warps
  constexpr int qk = qk_of<Type>();
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;

  float sum[J / NWARPS_F] = {0.0f};
  const int i = blockIdx.y*WARP_SIZE + threadIdx.x;

  const int64_t total = int64_t(ntx) * nty * blocks_per_row;
  int kbc0      = int(int64_t(blockIdx.x)     * total / gridDim.x);
  int kbc0_stop = int(int64_t(blockIdx.x + 1) * total / gridDim.x);

  kbc0      -= (kbc0      % blocks_per_row) % blocks_per_iter;
  kbc0_stop -= (kbc0_stop % blocks_per_row) % blocks_per_iter;

  const bool did_not_have_any_data   = kbc0 == kbc0_stop;
  const bool wrote_beginning_of_tile = (kbc0 % blocks_per_row) == 0;
  const bool did_not_write_last =
      (kbc0 / blocks_per_row) == (kbc0_stop / blocks_per_row) &&
      (kbc0_stop % blocks_per_row) != 0;
  if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last)
    return;

  bool any_fixup = false;

  // Iterate over previous blocks and sum up the partial sums they wrote.
  int bidx = blockIdx.x - 1;
  int kbc_stop = kbc0;
  while (true) {
    int kbc = int(int64_t(bidx) * total / gridDim.x);
    kbc -= (kbc % blocks_per_row) % blocks_per_iter;

    if (kbc == kbc_stop) {  // Did not have any data.
      bidx--;
      kbc_stop = kbc;
      continue;
    }

    any_fixup = true;

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += NWARPS_F) {
      const int j = j0 + threadIdx.y;
      sum[j0/NWARPS_F] += tmp_last_tile[bidx*(J*I) + j*I + i];
    }

    // If this block started in a previous tile we are done and do not need to
    // combine additional partial results.
    if ((kbc % blocks_per_row) == 0 ||
        (kbc / blocks_per_row) < (kbc0 / blocks_per_row))
      break;
    bidx--;
    kbc_stop = kbc;
  }

  if (!any_fixup) return;

  const int tmp = kbc0 / blocks_per_row;
  const int jt = tmp % ntx;
  const int it = tmp / ntx;

  dst += jt*J*stride_col_dst + it*I;

  const int j_max = ncols_dst - jt*J - 1;
#pragma unroll
  for (int j0 = 0; j0 < J; j0 += NWARPS_F) {
    const int j = j0 + threadIdx.y;
    if (j > j_max) return;
    dst[j*stride_col_dst + i] += sum[j0/NWARPS_F];
  }
}

// --- host-side launchers (SonicBoom's own) ---------------------------------

// Number of 128-value block planes an activation row must have for mul_mat_q:
// the kernel always consumes whole 256-value K-iterations (two planes, see the
// kb0 loop and the two vec_dot calls), so an *odd* plane count gets one trailing
// zero plane. That plane's scale is 0, which zeroes the tail iteration's second
// half whatever the weight-side over-read returns. ggml reaches the same state
// by padding ne10 to MATRIX_ROW_PADDING (512) and zero-filling the pad in
// quantize_mmq_q8_1. Even counts are unchanged, so the K-quant formats (their K
// is always a multiple of 256, e.g. Gemma) keep today's exact layout.
__host__ __device__ constexpr int mmq_q8_planes(int64_t K) {
  const int64_t p = K / QK8_1_MMQ;
  return int(p + (p & 1));
}

// Number of block_q8_1_mmq elements needed for a [K/128 -> padded][n] activation
// buffer, including J_max tail padding (over-read guard, mirrors ggml's src1
// sizing). K must be a multiple of 128 (the kernels and the plan enforce it).
__host__ inline size_t mmq_q8_blocks(int K, int n) {
  return (size_t)mmq_q8_planes(K) * n + I;  // I == J_max == 128
}

// Zero the trailing plane(s) mmq_q8_blocks adds over the planes a quantizer
// writes. Only the legacy 32-element formats can need it (K not a multiple of
// MMQ_ITER_K == 256); their D4 layout stores a float block scale, so a zeroed
// block is a zero scale. No-op for a K that is a multiple of 256.
__host__ inline void zero_pad_planes(const block_q8_1_mmq* y, int K, int n,
                                     cudaStream_t stream) {
  const int64_t used = K / QK8_1_MMQ;
  const int64_t planes = mmq_q8_planes(K);
  if (planes == used) return;
  cudaMemsetAsync(const_cast<void*>(static_cast<const void*>(y) +
                                     used * n * sizeof(block_q8_1_mmq)),
                  0, size_t(planes - used) * n * sizeof(block_q8_1_mmq), stream);
}

// Quantize an already-transposed activation Xt [n, K] into block_q8_1_mmq.
template <int ds>
__host__ void quantize_activation(const float* x, block_q8_1_mmq* y, int K, int n, cudaStream_t stream) {
  const int64_t block_num_y = (K + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
  const dim3 num_blocks(n, block_num_y, 1);
  const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
  quantize_mmq_q8_1<ds><<<num_blocks, block_size, 0, stream>>>(x, y, K, n);
}

// J selection (fallback = false J set from mmq-config-ampere.cuh). Picks the
// first (smallest) J achieving the minimal number of column tiles ceil(n/J).
__host__ inline int select_J(int n) {
  static constexpr int Js[] = {8, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128};
  int best = 128;
  int min_ntiles = (n + 127) / 128;
  for (int J : Js) {
    const int ntiles = (n + J - 1) / J;
    if (ntiles < min_ntiles) { min_ntiles = ntiles; best = J; }
  }
  return best;
}

template <int Type, int J>
__host__ void launch_mul_mat_q(const char* w, const int* y_q8, float* dst,
                               int M, int n, int blocks_per_row, cudaStream_t stream) {
  constexpr int sram_stride = Type == MMQ_Q3_K ? SRAM_STRIDE_Q3_K :
                              Type == MMQ_Q6_K ? SRAM_STRIDE_Q6_K :
                              (Type == MMQ_Q5_0 || Type == MMQ_Q8_0) ? SRAM_STRIDE_LEGACY_32
                                                                    : SRAM_STRIDE_Q8_1;
  constexpr int nbytes_shared = (PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS) + I*sram_stride) * 4;
  cudaFuncSetAttribute((const void*)mul_mat_q<Type, J>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
  const int nty = (M + I - 1) / I;
  const int ntx = (n + J - 1) / J;
  mul_mat_q<Type, J><<<dim3(nty, ntx, 1), dim3(WARP_SIZE, NWARPS, 1), nbytes_shared, stream>>>(
      w, y_q8, dst, blocks_per_row, n, n, M);
}

// Stream-k launch geometry (ggml's mul_mat_q host launch, fallback = false).
// ntx*nty tiles are spread over `nsm` blocks when a whole-tile-per-block
// schedule would leave the SMs under-used (tiling efficiency < 90%), and the
// fixup is needed exactly when the tile count does not divide the block count.
struct StreamKGeometry {
  int ntx;
  int nty;
  int nblocks;
  bool fixup;
};

__host__ inline StreamKGeometry stream_k_geometry(int M, int n, int J, int nsm) {
  StreamKGeometry g{};
  g.nty = (M + I - 1) / I;
  g.ntx = (n + J - 1) / J;
  const int ntiles = g.ntx * g.nty;
  if (nsm <= 0) nsm = 1;
  const int rounds = (ntiles + nsm - 1) / nsm;
  const int tiles_efficiency_percent = 100 * ntiles / (nsm * rounds);
  g.nblocks = tiles_efficiency_percent >= 90 ? ntiles : nsm;
  g.fixup = (ntiles % g.nblocks) != 0;
  return g;
}

// Stream-k launch. `tmp_fixup` must hold >= nblocks*J*I floats when
// stream_k_geometry(M, n, J, nsm).fixup is true; it is not touched otherwise.
template <int Type, int J>
__host__ void launch_mul_mat_q_stream_k(const char* w, const int* y_q8, float* dst,
                                        float* tmp_fixup, int M, int n,
                                        int blocks_per_row, int nsm,
                                        cudaStream_t stream) {
  constexpr int sram_stride = Type == MMQ_Q3_K ? SRAM_STRIDE_Q3_K :
                              Type == MMQ_Q6_K ? SRAM_STRIDE_Q6_K :
                              (Type == MMQ_Q5_0 || Type == MMQ_Q8_0) ? SRAM_STRIDE_LEGACY_32
                                                                    : SRAM_STRIDE_Q8_1;
  constexpr int nbytes_shared = (PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS) + I*sram_stride) * 4;
  cudaFuncSetAttribute((const void*)mul_mat_q_stream_k<Type, J>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
  const StreamKGeometry g = stream_k_geometry(M, n, J, nsm);
  mul_mat_q_stream_k<Type, J>
      <<<g.nblocks, dim3(WARP_SIZE, NWARPS, 1), nbytes_shared, stream>>>(
          w, y_q8, dst, tmp_fixup, blocks_per_row, n, n, M, g.ntx, g.nty);
  if (!g.fixup) return;
  mul_mat_q_stream_k_fixup<Type, J>
      <<<dim3(g.nblocks, I/WARP_SIZE, 1), dim3(WARP_SIZE, NWARPS/2, 1), 0, stream>>>(
          dst, tmp_fixup, blocks_per_row, n, M, g.ntx, g.nty);
}

// Number of K pieces the fixed split-K schedule uses for a weight shape. It is
// a function of (M, K, device SM count) only — never of the batch size n — so
// the K partition (and therefore every output element's rounding) is identical
// however the activation batch is segmented. Returns 1 when splitting cannot
// help or is not possible (1 K iteration, or the row tiles already fill the
// device), which callers treat as "use the tiled kernel".
// `qk` is QK_K (K-quants) or QK5_0 (legacy), as in qk_of.
__host__ inline int split_k_parts(int M, int blocks_per_row, int qk, int nsm) {
  constexpr int max_parts = 4;
  const int nty = M / I;
  if (nty <= 0 || M % I != 0) return 1;
  if (nsm <= 0) nsm = 1;
  const int blocks_per_iter = MMQ_ITER_K / qk;
  const int iters = (blocks_per_row + blocks_per_iter - 1) / blocks_per_iter;
  int parts = (nsm + nty - 1) / nty;
  if (parts > max_parts) parts = max_parts;
  if (parts > iters) parts = iters;
  return parts < 2 ? 1 : parts;
}

// Scratch (in floats) the split-K launch needs: one J*I partial per work unit.
__host__ inline int64_t split_k_scratch_floats(int M, int n, int J, int parts) {
  return int64_t((n + J - 1) / J) * (M / I) * parts * (J*I);
}

// Fixed split-K launch. `tmp_partial` must hold
// split_k_scratch_floats(M, n, J, parts) floats.
template <int Type, int J>
__host__ void launch_mul_mat_q_split_k(const char* w, const int* y_q8, float* dst,
                                       float* tmp_partial, int M, int n,
                                       int blocks_per_row, int nsm, int parts,
                                       cudaStream_t stream) {
  constexpr int sram_stride = Type == MMQ_Q3_K ? SRAM_STRIDE_Q3_K :
                              Type == MMQ_Q6_K ? SRAM_STRIDE_Q6_K :
                              (Type == MMQ_Q5_0 || Type == MMQ_Q8_0) ? SRAM_STRIDE_LEGACY_32
                                                                    : SRAM_STRIDE_Q8_1;
  constexpr int nbytes_shared = (PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS) + I*sram_stride) * 4;
  cudaFuncSetAttribute((const void*)mul_mat_q_split_k<Type, J>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
  constexpr int qk = qk_of<Type>();
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;
  const int iters = (blocks_per_row + blocks_per_iter - 1) / blocks_per_iter;
  const int bpr_aligned = iters * blocks_per_iter;
  const int ntx = (n + J - 1) / J;
  const int nty = M / I;
  const int64_t units = int64_t(ntx) * nty * parts;
  if (nsm <= 0) nsm = 1;
  const int nblocks = int(units < nsm ? units : nsm);
  mul_mat_q_split_k<Type, J>
      <<<nblocks, dim3(WARP_SIZE, NWARPS, 1), nbytes_shared, stream>>>(
          w, y_q8, dst, tmp_partial, blocks_per_row, n, n, M, ntx, nty, parts,
          bpr_aligned);
  mul_mat_q_split_k_fixup<Type, J>
      <<<dim3(ntx*nty, I/WARP_SIZE, 1), dim3(WARP_SIZE, NWARPS/2, 1), 0, stream>>>(
          dst, tmp_partial, n, M, ntx, nty, parts);
}

// Entry point: batched int8-tensor-core matmul, writing dst [M, n] column-major.
// w   = raw GGUF quantized weight bytes (block_q3/q4/q5_K).
// y   = quantized activation block_q8_1_mmq [K/128][n].
// M   = weight rows, n = columns. blocks_per_row = K/256.
// Returns false only if J dispatch failed (should not happen).
template <int Type>
__host__ bool mul_mat_q(const char* w, const int* y_q8, float* dst,
                        int M, int n, int blocks_per_row, cudaStream_t stream) {
  switch (select_J(n)) {
    case 8:   launch_mul_mat_q<Type, 8>  (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 16:  launch_mul_mat_q<Type, 16> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 24:  launch_mul_mat_q<Type, 24> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 32:  launch_mul_mat_q<Type, 32> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 40:  launch_mul_mat_q<Type, 40> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 48:  launch_mul_mat_q<Type, 48> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 64:  launch_mul_mat_q<Type, 64> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 80:  launch_mul_mat_q<Type, 80> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 96:  launch_mul_mat_q<Type, 96> (w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 112: launch_mul_mat_q<Type, 112>(w, y_q8, dst, M, n, blocks_per_row, stream); break;
    case 128: launch_mul_mat_q<Type, 128>(w, y_q8, dst, M, n, blocks_per_row, stream); break;
    default: return false;
  }
  return true;
}

}  // namespace sonicboom::quant::cuda::mmq
