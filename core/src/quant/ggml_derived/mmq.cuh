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
// quantized matmul below (Q3_K / Q4_K / Q5_K shared-memory tiled dequant into
// int8 + mma) is ported from the following upstream sources:
//
//   ggml/src/ggml-cuda/mma.cuh                — tile<> + load_ldmatrix + mma
//   ggml/src/ggml-cuda/mmq.cuh                — block_q8_1_mmq, sram_stride,
//                                                process_tile, mul_mat_q,
//                                                write_back_mma
//   ggml/src/ggml-cuda/mmq-config-ampere.cuh  — J set (fallback = false)
//   ggml/src/ggml-cuda/mmq-load-tiles.cuh     — load_tiles_q3/q4/q5_K,
//                                                unpack_scales_q45_K
//   ggml/src/ggml-cuda/mmq-vec-dot.cuh        — vec_dot_q8_0_16_q8_1_mma,
//                                                vec_dot_q8_1_q8_1_mma
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
constexpr int SRAM_STRIDE_Q3_K = 2*MMQ_TILE_NE_K + 16 + 4;
constexpr int SRAM_STRIDE_Q8_1 = 2*MMQ_TILE_NE_K +  8 + 4;

enum MMQType { MMQ_Q3_K = 0, MMQ_Q4_K = 1, MMQ_Q5_K = 2 };
enum DsLayout { D4 = 0, DS4 = 1 };

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

template <int Type, int J>
__device__ __forceinline__ void load_tiles(const char* x, int* x_tile, int kbx0, int stride) {
  if constexpr (Type == MMQ_Q3_K) {
    load_tiles_q3_K<J>(x, x_tile, kbx0, stride);
  } else if constexpr (Type == MMQ_Q4_K) {
    load_tiles_q4_K<J>(x, x_tile, kbx0, stride);
  } else {
    load_tiles_q5_K<J>(x, x_tile, kbx0, stride);
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

template <int Type, int J>
__device__ __forceinline__ void vec_dot(const int* x, const int* y, float* sum, int k00) {
  if constexpr (Type == MMQ_Q3_K) {
    vec_dot_q8_0_16_q8_1_mma<J>(x, y, sum, k00);
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

template <int ds>  // D4 (Q3_K) or DS4 (Q4_K/Q5_K)
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

  const float d_inv = 127.0f / amax;
  char4 q;
  q.x = roundf(xi.x*d_inv);
  q.y = roundf(xi.y*d_inv);
  q.z = roundf(xi.z*d_inv);
  q.w = roundf(xi.w*d_inv);
  const float d = 1.0f / d_inv;

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

template <int Type, int J>
__global__ void mul_mat_q(const char* __restrict__ x, const int* __restrict__ y,
                          float* __restrict__ dst,
                          const int blocks_per_row, const int ncols_dst, const int ncols_y,
                          const int stride_col_dst) {
  extern __shared__ int data_mul_mat_q[];
  int* tile_y = data_mul_mat_q;
  int* tile_x = tile_y + PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS);

  constexpr int sz = (int)(sizeof(block_q8_1_mmq) / sizeof(int));  // 36
  constexpr int blocks_per_iter = MMQ_ITER_K / QK_K;               // 1

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
      const int* by0 = y + offset_y + ncols_y * (kb0 * QK_K / QK8_1_MMQ) * sz;
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
      const int* by0 = y + offset_y + ncols_y * ((kb0 * QK_K / QK8_1_MMQ) * sz + sz);
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

// --- host-side launchers (SonicBoom's own) ---------------------------------

// Number of block_q8_1_mmq elements needed for a [K/128][n] activation buffer,
// including J_max tail padding (over-read guard, mirrors ggml's src1 sizing).
__host__ inline size_t mmq_q8_blocks(int K, int n) {
  return (size_t)(K / QK8_1_MMQ) * n + I;  // I == J_max == 128
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
  constexpr int sram_stride = (Type == MMQ_Q3_K) ? SRAM_STRIDE_Q3_K : SRAM_STRIDE_Q8_1;
  constexpr int nbytes_shared = (PAD(J*MMQ_TILE_Y_K, WARP_SIZE*NWARPS) + I*sram_stride) * 4;
  cudaFuncSetAttribute((const void*)mul_mat_q<Type, J>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, nbytes_shared);
  const int nty = (M + I - 1) / I;
  const int ntx = (n + J - 1) / J;
  mul_mat_q<Type, J><<<dim3(nty, ntx, 1), dim3(WARP_SIZE, NWARPS, 1), nbytes_shared, stream>>>(
      w, y_q8, dst, blocks_per_row, n, n, M);
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
