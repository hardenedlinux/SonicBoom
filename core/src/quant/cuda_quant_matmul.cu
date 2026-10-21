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

#include <sonicboom/quant/dequant.h>
#include <sonicboom/runtime_features.h>

#include "cuda_resident.h"

#include "../nn/cuda_stream.h"
#include "../nn/reduce_sum.cuh"
#include "../planner/reduce_mode.h"

#include "../planner/autotune.h"  // per-GPU tuning cache (Step 7)
#include "../planner/glue_jit.h"  // the generated MMQ1 tensor-core kernel
#include "../planner/mlir_autotune.h"  // Q1 tile-config cache (opt-in)
#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)

#include "ggml_derived/mmvq_q8_1.cuh"
#include "ggml_derived/mmq.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <sm_32_intrinsics.h>  // __vsub4
#include <sm_61_intrinsics.h>  // __dp4a

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
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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

// Current stream for device-resident kernels (see cuda_stream.h).
inline cudaStream_t cur_stream() { return ::sonicboom::cuda::sb_current_stream(); }

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

__device__ inline float dequant_q6_K_value(const uint8_t* blk, int l) {
  const int n = l >> 7;
  const int r = l & 127;
  const int group = r >> 5;
  const int lane = r & 31;
  const int qlo = (n << 6) + ((group & 1) ? 32 : 0) + lane;
  const uint8_t qbyte = blk[qlo];
  const int low = group < 2;
  const int q = (low ? int(qbyte & 0x0F) : int(qbyte >> 4)) |
                (((blk[128 + (n << 5) + lane] >> (2 * group)) & 3) << 4);
  const int8_t* scales = reinterpret_cast<const int8_t*>(blk + 192);
  const float d = half_to_float(ld_u16(blk + 208));
  return d * scales[n * 8 + group * 2 + (lane >> 4)] * (q - 32);
}

// Legacy 32-element blocks (Stage 2 / Qwen2). Q5_0 is 22 bytes
// {fp16 d; uint8 qh[4]; uint8 qs[16]}; Q8_0 is 34 bytes {fp16 d; int8 qs[32]}.
// Q5_0 element j is ((nibble) | (5th bit << 4)) - 16, where the low nibbles of
// qs carry elements 0..15, the high nibbles carry 16..31, and bit j of the
// little-endian qh word is the 5th bit of element j. This is the same
// arithmetic as dequantize_q5_0/q8_0 in ggml_derived/legacy_quants_q5_0_q8_0.h.
__device__ inline float dequant_q5_0_value(const uint8_t* blk, int l) {
  const uint32_t xh = uint32_t(blk[2]) | (uint32_t(blk[3]) << 8) |
                      (uint32_t(blk[4]) << 16) | (uint32_t(blk[5]) << 24);
  const int nib = l < 16 ? (blk[6 + l] & 0x0F) : (blk[6 + l - 16] >> 4);
  const int q = nib | (int((xh >> l) & 1u) << 4);
  return half_to_float(ld_u16(blk)) * float(q - 16);
}

__device__ inline float dequant_q8_0_value(const uint8_t* blk, int l) {
  return half_to_float(ld_u16(blk)) *
         float(reinterpret_cast<const int8_t*>(blk + 2)[l]);
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

// f32 -> fp16 -> f32 round-trip (matches nn::cuda's half_roundtrip in
// cuda_elementwise.cu; duplicated here because the fused gelu quantize kernel
// lives in this translation unit).
__device__ inline float half_roundtrip(float v) {
  return __half2float(__float2half_rn(v));
}

// ggml's ggml_gelu_f32 (mirrors nn::cuda::gelu_tanh_f32, same constants and
// operation order, so the fused kernel is bit-identical to gelu_fp16_dev).
__device__ inline float gelu_tanh_f32(float x) {
  constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
  constexpr float gelu_coef_a = 0.044715f;
  return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
}

// nn::gelu_fp16's clamping + fp16 round-trip, mirroring gelu_fp16_kernel in
// cuda_elementwise.cu (v clamped to [-10,10], fp16-rounded before and after the
// tanh gelu).
__device__ inline float gelu_fp16(float v) {
  if (v <= -10.0f) return 0.0f;
  if (v >= 10.0f) return v;
  return half_roundtrip(gelu_tanh_f32(half_roundtrip(v)));
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

__global__ void gemv_q6_K_base(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_blocks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_blocks) return;
  const int row = i / blocks_per_row;
  const int b = i % blocks_per_row;
  const uint8_t* blk = w + (size_t)i * 210;
  const float* xb = x + (size_t)b * 256;
  float sum = 0.0f;
  for (int l = 0; l < 256; ++l) sum += dequant_q6_K_value(blk, l) * xb[l];
  atomicAdd(&y[row], sum);
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


__global__ void gemv_q6_K_coop(const uint8_t* __restrict__ w, const float* __restrict__ x,
                               float* __restrict__ y, int blocks_per_row, int num_rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= num_rows) return;
  const int slice = lane & 15;
  const int first_block = lane >> 4;
  float sum = 0.0f;
  for (int b = first_block; b < blocks_per_row; b += 2) {
    const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 210;
    const float* xb = x + (size_t)b * 256 + slice * 16;
    #pragma unroll
    for (int k = 0; k < 16; ++k)
      sum += dequant_q6_K_value(blk, slice * 16 + k) * xb[k];
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o);
  if (lane == 0) y[row] = sum;
}

// --- device weight cache + scratch -----------------------------------------

struct DeviceWeight {
  void* ptr = nullptr;
  size_t bytes = 0;
};

struct CacheKey {
  const void* host;
  size_t bytes;
  bool operator==(const CacheKey& o) const { return host == o.host && bytes == o.bytes; }
};

struct CacheKeyHash {
  size_t operator()(const CacheKey& k) const noexcept {
    return std::hash<const void*>{}(k.host) ^ (std::hash<size_t>{}(k.bytes) * 0x9e3779b97f4a7c15ull);
  }
};

std::unordered_map<CacheKey, DeviceWeight, CacheKeyHash> g_cache;
std::mutex g_cache_mutex;

void* g_dx = nullptr;
size_t g_dx_bytes = 0;
void* g_dy = nullptr;
size_t g_dy_bytes = 0;
void* g_q8 = nullptr;
size_t g_q8_bytes = 0;
void* g_row = nullptr;
size_t g_row_bytes = 0;
void* g_gather = nullptr;
size_t g_gather_bytes = 0;
void* g_gather_out = nullptr;
size_t g_gather_out_bytes = 0;
void* g_tokens = nullptr;
size_t g_tokens_bytes = 0;
// fp16 dequant scratch for the cuBLAS batched matmul (C.1): the dequantized
// weight (row-major [rows, cols]) and the fp16 activation ([n, cols],
// position-major). Reused across layers, sized to the largest weight/activation
// seen.
void* g_wf16 = nullptr;
size_t g_wf16_bytes = 0;
void* g_xf16 = nullptr;
size_t g_xf16_bytes = 0;
// MMQ int8 tensor-core scratch (C.2): the block_q8_1_mmq quantized activation
// [K/128][n] (+ J_max tail over-read). Reused across layers.
void* g_mmq_q8 = nullptr;
size_t g_mmq_q8_bytes = 0;
void* g_mmq_q6_x_f32 = nullptr;
size_t g_mmq_q6_x_f32_bytes = 0;
// Fused rms_norm+quantize (C.4): one float per token, the per-row `inv` factor
// produced by phase A and consumed by phase B of
// rms_norm_quantize_mmq_q8_1_dev. Reused across layers.
void* g_rms_inv = nullptr;
size_t g_rms_inv_bytes = 0;

// Process-wide cuBLAS handle + stream-capture workspace (C.1). The handle is
// created lazily and the workspace is set so cublasGemmEx is stream-capturable
// into the prefill CUDA Graph (B.1). The handle is never destroyed (lives for
// the process lifetime, like g_cache); the workspace is freed by clear_cache().
std::mutex g_cublas_mutex;
cublasHandle_t g_cublas_handle = nullptr;
void* g_cublas_workspace = nullptr;
constexpr size_t kCublasWorkspaceBytes = 16 * 1024 * 1024;  // ample for fp16 GEMM

// Zero-copy per_layer_token_embd gather (Phase 3.2 B3): the packed tensor is
// pinned + mapped in place (cudaHostRegisterMapped) once, so the gather kernel
// reads the token's row straight out of host memory over PCIe — the per-token
// sync H2D disappears entirely. Single slot: only per_layer_token_embd uses
// this path (token_embd is already device-resident via upload_weight, B2).
struct MappedEmbedding {
  void* dev = nullptr;         // device-visible pointer (cudaHostGetDevicePointer)
  const void* host = nullptr;  // the registered host pointer (== W.data().data())
  size_t bytes = 0;
};
MappedEmbedding g_ple_mapped;

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

// The mmq kernels read whole 256-value K-iterations, so a row whose K is not a
// multiple of 256 over-reads up to 4 legacy blocks (4*34 = 136 bytes) past its
// last block. Those values enter the tail iteration's A tile, so they must be
// finite: that iteration's activation scale is 0, but (C.x * dA) * 0 is still
// NaN when dA is. An over-read into the next row is finite by construction; the
// last row needs this zeroed tail pad (ggml pads weight rows the same way).
constexpr size_t kWeightTailPad = 256;

// Upload W's packed bytes to the device, caching by (data pointer, byte size)
// so repeated calls with the same weight reuse the device copy.
const void* upload_weight(const QuantizedTensor& W) {
  std::lock_guard<std::mutex> lk(g_cache_mutex);
  const void* host = W.data().data();
  const size_t bytes = W.byte_size();
  CacheKey key{host, bytes};
  auto it = g_cache.find(key);
  if (it != g_cache.end()) return it->second.ptr;

  void* dptr = nullptr;
  if (cudaMalloc(&dptr, bytes + kWeightTailPad) != cudaSuccess) return nullptr;
  if (cudaMemset(static_cast<char*>(dptr) + bytes, 0, kWeightTailPad) != cudaSuccess ||
      cudaMemcpy(dptr, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dptr);
    return nullptr;
  }
  g_cache.emplace(key, DeviceWeight{dptr, bytes});
  return dptr;
}

// Upload a raw host byte range to the device, cached by (pointer, byte size) via
// the same cache as upload_weight. Used for the f32 norm weights of the fused
// rms_norm+quantize kernel (T3), which are plain f32 vectors, not packed K-quant
// tensors.
const void* upload_bytes(const void* host, size_t bytes) {
  std::lock_guard<std::mutex> lk(g_cache_mutex);
  CacheKey key{host, bytes};
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

// --- MMQ1 tensor-core q8_1 weight copy --------------------------------------
//
// The generated int8 tensor-core kernel takes an int8 weight with one f32 scale
// per 32 values, so the packed K-quant weight is converted once: dequantized
// row by row (quant::dequantize_row_f32, the same reference every other
// consumer reproduces), then each 32-value group is quantized to [-127, 127]
// with scale amax/127. The copy is int8 [rows][cols] followed by f32 scales
// [rows][cols/32], uploaded once and cached by (data pointer, byte size).
//
// The conversion is the pass's budgeted cost (SONICBOOM_MMQ_TC_MAX_MB): it runs
// on first dispatch, which — for a captured prefill segment — is the uncaptured
// warm-up run, so no allocation happens inside CUDA Graph capture.
struct TcWeight {
  void* q = nullptr;  // int8 [rows][cols]
  void* s = nullptr;  // f32 [rows][cols/32]
  size_t bytes = 0;
};

std::unordered_map<CacheKey, TcWeight, CacheKeyHash> g_tc_cache;

TcWeight upload_tc_weight(const QuantizedTensor& W, int rows, int cols) {
  if (rows <= 0 || cols <= 0 || cols % 32 != 0) return {};
  std::lock_guard<std::mutex> lk(g_cache_mutex);
  CacheKey key{W.data().data(), W.byte_size()};
  auto it = g_tc_cache.find(key);
  if (it != g_tc_cache.end()) return it->second;

  const size_t groups = size_t(cols) / 32;
  std::vector<int8_t> qh(size_t(rows) * size_t(cols));
  std::vector<float> sh(size_t(rows) * groups);
  std::vector<float> row_buf(static_cast<size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    if (!quant::dequantize_row_f32(W, uint64_t(r), row_buf.data(),
                                   uint64_t(cols)))
      return {};
    int8_t* qr = qh.data() + size_t(r) * size_t(cols);
    float* sr = sh.data() + size_t(r) * groups;
    for (size_t b = 0; b < groups; ++b) {
      const float* x = row_buf.data() + b * 32;
      float amax = 0.0f;
      for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[i]));
      const float d = amax / 127.0f;
      const float id = d > 0.0f ? 1.0f / d : 0.0f;
      sr[b] = d;
      for (int i = 0; i < 32; ++i) {
        const float v = std::roundf(x[i] * id);
        qr[b * 32 + i] = int8_t(std::fmin(127.0f, std::fmax(-127.0f, v)));
      }
    }
  }

  const size_t qbytes = qh.size();
  const size_t sbytes = sh.size() * sizeof(float);
  void* dq = nullptr;
  if (cudaMalloc(&dq, qbytes + sbytes) != cudaSuccess) return {};
  void* ds = static_cast<char*>(dq) + qbytes;
  if (cudaMemcpy(dq, qh.data(), qbytes, cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(ds, sh.data(), sbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
    cudaFree(dq);
    return {};
  }
  TcWeight w{dq, ds, qbytes + sbytes};
  g_tc_cache.emplace(key, w);
  return w;
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

// --- single-row embedding gather --------------------------------------------
//
// The decode preamble dequantizes token_embd[token] (Q3_K) and
// per_layer_token_embd[token] (Q4_K) to device f32. These are plain
// dequantizations (no dot product), one thread per element. Element i falls in
// super-block b = i >> 8 at intra-block index l = i & 255; the same block
// layout / reconstruction formulas as dequantize_row_f32 (dequant.cpp), so the
// result is bit-identical to the CPU embedding gather.

__global__ void gather_q3_K_row(const uint8_t* __restrict__ w, int64_t row_host,
                                const float* __restrict__ row_dev,
                                float* __restrict__ out, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (i >= width) return;
  const int64_t row = row_dev ? (int64_t)row_dev[0] : row_host;
  const int b = i >> 8;
  const int l = i & 255;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 110;
  const uint8_t* hm = blk;
  const uint8_t* q = blk + 32;
  const float d = half_to_float(ld_u16(blk + 108));
  int8_t scales[16];
  unpack_q3_K_scales(blk + 96, scales);

  const int n = l >> 7;            // 0 or 1 (two 128-element halves)
  const int r = l & 127;           // element within the 128 half
  const int j = r >> 5;            // 32-element group (0..3)
  const int l2 = r & 31;           // element within the 32 group
  const int qv = int((q[n * 32 + l2] >> (j << 1)) & 3);
  const int hmset = (hm[l2] & (1 << (j + (n << 2)))) ? 0 : 4;
  const int sidx = (n << 3) + (j << 1) + (l2 >> 4);  // 8n + 2j + (l2 >= 16)
  out[i] = d * float(scales[sidx] - 32) * float(qv - hmset);
}

__global__ void gather_q4_K_row(const uint8_t* __restrict__ w, int64_t row_host,
                                const float* __restrict__ row_dev,
                                float* __restrict__ out, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (i >= width) return;
  const int64_t row = row_dev ? (int64_t)row_dev[0] : row_host;
  const int b = i >> 8;
  const int l = i & 255;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 144;
  const float d = half_to_float(ld_u16(blk + 0));
  const float dmin = half_to_float(ld_u16(blk + 2));
  const uint8_t* scales = blk + 4;
  const uint8_t* q = blk + 16;

  const int j = l >> 6;            // 64-element group (0..3)
  const int r = l & 63;
  const int low = (r < 32) ? 1 : 0;
  const int l2 = r & 31;
  const uint8_t qb = q[j * 32 + l2];
  uint8_t sc, mi;
  get_scale_min_k4((j << 1) + (low ? 0 : 1), scales, sc, mi);
  const int v = low ? (qb & 0xF) : (qb >> 4);
  out[i] = d * float(sc) * float(v) - dmin * float(mi);
}

__global__ void gather_q5_K_row(const uint8_t* __restrict__ w, int64_t row_host,
                                const float* __restrict__ row_dev,
                                float* __restrict__ out, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (i >= width) return;
  const int64_t row = row_dev ? (int64_t)row_dev[0] : row_host;
  const int b = i >> 8;
  const int l = i & 255;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 176;
  const float d = half_to_float(ld_u16(blk + 0));
  const float dmin = half_to_float(ld_u16(blk + 2));
  const uint8_t* scales = blk + 4;
  const uint8_t* qh = blk + 16;
  const uint8_t* ql = blk + 48;

  const int j = l >> 6;
  const int r = l & 63;
  const int low = (r < 32) ? 1 : 0;
  const int l2 = r & 31;
  const uint8_t qlb = ql[j * 32 + l2];
  uint8_t sc, mi;
  get_scale_min_k4((j << 1) + (low ? 0 : 1), scales, sc, mi);
  const int qhbit = (j << 1) + (low ? 0 : 1);  // u1/u2 = 1<<(2j) / 1<<(2j+1)
  const int v = (low ? (qlb & 0xF) : (qlb >> 4)) +
                ((qh[l2] & (1 << qhbit)) ? 16 : 0);
  out[i] = d * float(sc) * float(v) - dmin * float(mi);
}

__global__ void gather_q6_K_row(const uint8_t* __restrict__ w, int64_t row_host,
                                const float* __restrict__ row_dev,
                                float* __restrict__ out, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks_per_row * 256) return;
  const int64_t row = row_dev ? (int64_t)row_dev[0] : row_host;
  const int b = i >> 8;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 210;
  out[i] = dequant_q6_K_value(blk, i & 255);
}

__global__ void gather_q5_0_row(const uint8_t* __restrict__ w, int64_t row_host,
                                const float* __restrict__ row_dev,
                                float* __restrict__ out, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks_per_row * 32) return;
  const int64_t row = row_dev ? (int64_t)row_dev[0] : row_host;
  const int b = i >> 5;
  out[i] = dequant_q5_0_value(w + ((size_t)row * blocks_per_row + b) * 22, i & 31);
}

__global__ void gather_q8_0_row(const uint8_t* __restrict__ w, int64_t row_host,
                                const float* __restrict__ row_dev,
                                float* __restrict__ out, int blocks_per_row) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= blocks_per_row * 32) return;
  const int64_t row = row_dev ? (int64_t)row_dev[0] : row_host;
  const int b = i >> 5;
  out[i] = dequant_q8_0_value(w + ((size_t)row * blocks_per_row + b) * 34, i & 31);
}

// Launch the single-row gather kernel for `row` of a device-resident packed
// tensor. `w` points at the packed tensor start on the device. The row is either
// `row_host` (a host value baked into the launch) or, when `row_dev` is non-null,
// read from a device-resident token buffer at kernel time (Phase 3.2 B4: the
// next token id stays on the device). Shared by gather_f32_dev (row 0 of a
// just-uploaded single row) and gather_f32_resident_dev / _mapped_dev.
bool launch_gather_row(const uint8_t* w, int64_t row_host, const float* row_dev,
                       QuantType type, int width, int blocks_per_row, float* out) {
  const int grid = (width + 255) / 256;
  switch (type) {
    case QuantType::Q3_K:
      SB_NOTE_LAUNCH(gather_q3_K_row, grid, 256, 0, cur_stream());
      gather_q3_K_row<<<grid, 256, 0, cur_stream()>>>(w, row_host, row_dev, out, blocks_per_row);
      break;
    case QuantType::Q4_K:
      SB_NOTE_LAUNCH(gather_q4_K_row, grid, 256, 0, cur_stream());
      gather_q4_K_row<<<grid, 256, 0, cur_stream()>>>(w, row_host, row_dev, out, blocks_per_row);
      break;
    case QuantType::Q5_K:
      SB_NOTE_LAUNCH(gather_q5_K_row, grid, 256, 0, cur_stream());
      gather_q5_K_row<<<grid, 256, 0, cur_stream()>>>(w, row_host, row_dev, out, blocks_per_row);
      break;
    case QuantType::Q6_K:
      SB_NOTE_LAUNCH(gather_q6_K_row, grid, 256, 0, cur_stream());
      gather_q6_K_row<<<grid, 256, 0, cur_stream()>>>(w, row_host, row_dev, out, blocks_per_row);
      break;
    case QuantType::Q5_0:
      SB_NOTE_LAUNCH(gather_q5_0_row, grid, 256, 0, cur_stream());
      gather_q5_0_row<<<grid, 256, 0, cur_stream()>>>(w, row_host, row_dev, out, blocks_per_row);
      break;
    case QuantType::Q8_0:
      SB_NOTE_LAUNCH(gather_q8_0_row, grid, 256, 0, cur_stream());
      gather_q8_0_row<<<grid, 256, 0, cur_stream()>>>(w, row_host, row_dev, out, blocks_per_row);
      break;
    default:
      return false;
  }
  return cudaGetLastError() == cudaSuccess;
}

// --- batched embedding gather (prefill) --------------------------------------
//
// The n-token prefill preamble dequantizes token_embd[tokens[c]] and
// per_layer_token_embd[tokens[c]] for c in [0, n) in one launch per weight.
// One thread per (k, c) element: blockIdx.y selects the column c (and thus the
// token), threads across x cover the width. The dequant arithmetic is identical
// to gather_*_K_row, so each column is bit-identical to the n == 1 gather.

__global__ void gather_q3_K_batched(const uint8_t* __restrict__ w,
                                    const uint64_t* __restrict__ rows,
                                    float* __restrict__ out, int blocks_per_row,
                                    int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (k >= width) return;
  const int c = blockIdx.y;
  if (c >= n) return;
  const int64_t row = (int64_t)rows[c];
  const int b = k >> 8;
  const int l = k & 255;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 110;
  const uint8_t* hm = blk;
  const uint8_t* q = blk + 32;
  const float d = half_to_float(ld_u16(blk + 108));
  int8_t scales[16];
  unpack_q3_K_scales(blk + 96, scales);

  const int m = l >> 7;
  const int r = l & 127;
  const int j = r >> 5;
  const int l2 = r & 31;
  const int qv = int((q[m * 32 + l2] >> (j << 1)) & 3);
  const int hmset = (hm[l2] & (1 << (j + (m << 2)))) ? 0 : 4;
  const int sidx = (m << 3) + (j << 1) + (l2 >> 4);
  out[(size_t)c * width + k] = d * float(scales[sidx] - 32) * float(qv - hmset);
}

__global__ void gather_q4_K_batched(const uint8_t* __restrict__ w,
                                    const uint64_t* __restrict__ rows,
                                    float* __restrict__ out, int blocks_per_row,
                                    int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (k >= width) return;
  const int c = blockIdx.y;
  if (c >= n) return;
  const int64_t row = (int64_t)rows[c];
  const int b = k >> 8;
  const int l = k & 255;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 144;
  const float d = half_to_float(ld_u16(blk + 0));
  const float dmin = half_to_float(ld_u16(blk + 2));
  const uint8_t* scales = blk + 4;
  const uint8_t* q = blk + 16;

  const int j = l >> 6;
  const int r = l & 63;
  const int low = (r < 32) ? 1 : 0;
  const int l2 = r & 31;
  const uint8_t qb = q[j * 32 + l2];
  uint8_t sc, mi;
  get_scale_min_k4((j << 1) + (low ? 0 : 1), scales, sc, mi);
  const int v = low ? (qb & 0xF) : (qb >> 4);
  out[(size_t)c * width + k] = d * float(sc) * float(v) - dmin * float(mi);
}

__global__ void gather_q5_K_batched(const uint8_t* __restrict__ w,
                                    const uint64_t* __restrict__ rows,
                                    float* __restrict__ out, int blocks_per_row,
                                    int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (k >= width) return;
  const int c = blockIdx.y;
  if (c >= n) return;
  const int64_t row = (int64_t)rows[c];
  const int b = k >> 8;
  const int l = k & 255;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 176;
  const float d = half_to_float(ld_u16(blk + 0));
  const float dmin = half_to_float(ld_u16(blk + 2));
  const uint8_t* scales = blk + 4;
  const uint8_t* qh = blk + 16;
  const uint8_t* ql = blk + 48;

  const int j = l >> 6;
  const int r = l & 63;
  const int low = (r < 32) ? 1 : 0;
  const int l2 = r & 31;
  const uint8_t qlb = ql[j * 32 + l2];
  uint8_t sc, mi;
  get_scale_min_k4((j << 1) + (low ? 0 : 1), scales, sc, mi);
  const int qhbit = (j << 1) + (low ? 0 : 1);
  const int v = (low ? (qlb & 0xF) : (qlb >> 4)) +
                ((qh[l2] & (1 << qhbit)) ? 16 : 0);
  out[(size_t)c * width + k] = d * float(sc) * float(v) - dmin * float(mi);
}

__global__ void gather_q6_K_batched(const uint8_t* __restrict__ w,
                                    const uint64_t* __restrict__ rows,
                                    float* __restrict__ out, int blocks_per_row,
                                    int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 256;
  if (k >= width) return;
  const int c = blockIdx.y;
  if (c >= n) return;
  const int64_t row = (int64_t)rows[c];
  const int b = k >> 8;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 210;
  out[(size_t)c * width + k] = dequant_q6_K_value(blk, k & 255);
}

__global__ void gather_q5_0_batched(const uint8_t* __restrict__ w,
                                    const uint64_t* __restrict__ rows,
                                    float* __restrict__ out, int blocks_per_row,
                                    int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 32;
  if (k >= width) return;
  const int c = blockIdx.y;
  if (c >= n) return;
  const int64_t row = (int64_t)rows[c];
  const int b = k >> 5;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 22;
  out[(size_t)c * width + k] = dequant_q5_0_value(blk, k & 31);
}

__global__ void gather_q8_0_batched(const uint8_t* __restrict__ w,
                                    const uint64_t* __restrict__ rows,
                                    float* __restrict__ out, int blocks_per_row,
                                    int n) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int width = blocks_per_row * 32;
  if (k >= width) return;
  const int c = blockIdx.y;
  if (c >= n) return;
  const int64_t row = (int64_t)rows[c];
  const int b = k >> 5;
  const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * 34;
  out[(size_t)c * width + k] = dequant_q8_0_value(blk, k & 31);
}

// Launch the batched gather for n token ids (`rows`, device uint64_t[n]) of a
// device-resident packed tensor `w`. grid.y == n (one column per token),
// grid.x covers the width. Shared by the resident and mapped variants.
bool launch_gather_batched(const uint8_t* w, const uint64_t* rows, QuantType type,
                           int width, int blocks_per_row, int n, float* out) {
  const dim3 grid((width + 255) / 256, n);
  switch (type) {
    case QuantType::Q3_K:
      SB_NOTE_LAUNCH(gather_q3_K_batched, grid, 256, 0, cur_stream());
      gather_q3_K_batched<<<grid, 256, 0, cur_stream()>>>(w, rows, out, blocks_per_row, n);
      break;
    case QuantType::Q4_K:
      SB_NOTE_LAUNCH(gather_q4_K_batched, grid, 256, 0, cur_stream());
      gather_q4_K_batched<<<grid, 256, 0, cur_stream()>>>(w, rows, out, blocks_per_row, n);
      break;
    case QuantType::Q5_K:
      SB_NOTE_LAUNCH(gather_q5_K_batched, grid, 256, 0, cur_stream());
      gather_q5_K_batched<<<grid, 256, 0, cur_stream()>>>(w, rows, out, blocks_per_row, n);
      break;
    case QuantType::Q6_K:
      SB_NOTE_LAUNCH(gather_q6_K_batched, grid, 256, 0, cur_stream());
      gather_q6_K_batched<<<grid, 256, 0, cur_stream()>>>(w, rows, out, blocks_per_row, n);
      break;
    case QuantType::Q5_0:
      SB_NOTE_LAUNCH(gather_q5_0_batched, grid, 256, 0, cur_stream());
      gather_q5_0_batched<<<grid, 256, 0, cur_stream()>>>(w, rows, out, blocks_per_row, n);
      break;
    case QuantType::Q8_0:
      SB_NOTE_LAUNCH(gather_q8_0_batched, grid, 256, 0, cur_stream());
      gather_q8_0_batched<<<grid, 256, 0, cur_stream()>>>(w, rows, out, blocks_per_row, n);
      break;
    default:
      return false;
  }
  return cudaGetLastError() == cudaSuccess;
}

// --- fp16 dequantization for the cuBLAS batched matmul (C.1) ----------------
//
// Full-tensor dequantize to fp16, one thread per output element. The flat
// element index idx == b * 256 + l (b = super-block, l = element within it)
// maps exactly to the row-major [rows, cols] layout cuBLAS consumes: the packed
// weight is [rows][blocks_per_row] blocks and each block covers 256 contiguous
// columns, so out[r*cols + c] == dequant(block (r*blocks_per_row + c/256),
// element c%256). The per-element arithmetic is identical to gather_*_K_batched
// (and the scalar reference), only the output is rounded to fp16.

__global__ void dequant_q3_K_f16(const uint8_t* __restrict__ w,
                                 __half* __restrict__ out, int64_t numel) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const int64_t b = idx >> 8;
  const int l = int(idx & 255);
  const uint8_t* blk = w + b * 110;
  const uint8_t* hm = blk;
  const uint8_t* q = blk + 32;
  const float d = half_to_float(ld_u16(blk + 108));
  int8_t scales[16];
  unpack_q3_K_scales(blk + 96, scales);

  const int m = l >> 7;
  const int r = l & 127;
  const int j = r >> 5;
  const int l2 = r & 31;
  const int qv = int((q[m * 32 + l2] >> (j << 1)) & 3);
  const int hmset = (hm[l2] & (1 << (j + (m << 2)))) ? 0 : 4;
  const int sidx = (m << 3) + (j << 1) + (l2 >> 4);
  out[idx] = __float2half_rn(d * float(scales[sidx] - 32) * float(qv - hmset));
}

__global__ void dequant_q4_K_f16(const uint8_t* __restrict__ w,
                                 __half* __restrict__ out, int64_t numel) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const int64_t b = idx >> 8;
  const int l = int(idx & 255);
  const uint8_t* blk = w + b * 144;
  const float d = half_to_float(ld_u16(blk + 0));
  const float dmin = half_to_float(ld_u16(blk + 2));
  const uint8_t* scales = blk + 4;
  const uint8_t* q = blk + 16;

  const int j = l >> 6;
  const int r = l & 63;
  const int low = (r < 32) ? 1 : 0;
  const int l2 = r & 31;
  const uint8_t qb = q[j * 32 + l2];
  uint8_t sc, mi;
  get_scale_min_k4((j << 1) + (low ? 0 : 1), scales, sc, mi);
  const int v = low ? (qb & 0xF) : (qb >> 4);
  out[idx] = __float2half_rn(d * float(sc) * float(v) - dmin * float(mi));
}

__global__ void dequant_q5_K_f16(const uint8_t* __restrict__ w,
                                 __half* __restrict__ out, int64_t numel) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const int64_t b = idx >> 8;
  const int l = int(idx & 255);
  const uint8_t* blk = w + b * 176;
  const float d = half_to_float(ld_u16(blk + 0));
  const float dmin = half_to_float(ld_u16(blk + 2));
  const uint8_t* scales = blk + 4;
  const uint8_t* qh = blk + 16;
  const uint8_t* ql = blk + 48;

  const int j = l >> 6;
  const int r = l & 63;
  const int low = (r < 32) ? 1 : 0;
  const int l2 = r & 31;
  const uint8_t qlb = ql[j * 32 + l2];
  uint8_t sc, mi;
  get_scale_min_k4((j << 1) + (low ? 0 : 1), scales, sc, mi);
  const int qhbit = (j << 1) + (low ? 0 : 1);
  const int v = (low ? (qlb & 0xF) : (qlb >> 4)) +
                ((qh[l2] & (1 << qhbit)) ? 16 : 0);
  out[idx] = __float2half_rn(d * float(sc) * float(v) - dmin * float(mi));
}

__global__ void dequant_q6_K_f16(const uint8_t* __restrict__ w,
                                 __half* __restrict__ out, int64_t numel) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const int64_t b = idx >> 8;
  out[idx] = __float2half_rn(dequant_q6_K_value(w + b * 210, int(idx & 255)));
}

__global__ void dequant_q5_0_f16(const uint8_t* __restrict__ w,
                                 __half* __restrict__ out, int64_t numel) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const int64_t b = idx >> 5;
  out[idx] = __float2half_rn(dequant_q5_0_value(w + b * 22, int(idx & 31)));
}

__global__ void dequant_q8_0_f16(const uint8_t* __restrict__ w,
                                 __half* __restrict__ out, int64_t numel) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;
  const int64_t b = idx >> 5;
  out[idx] = __float2half_rn(dequant_q8_0_value(w + b * 34, int(idx & 31)));
}

// f32 -> fp16 (round-to-nearest), elementwise, for the activation operand.
__global__ void cast_f32_f16(const float* __restrict__ x, __half* __restrict__ y,
                             int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = __float2half_rn(x[i]);
}

// Recover the Q6 fallback activation from D4 MMQ blocks. The Q6 prefill plan
// selects D4_Q8_1, so d4 holds one float scale per 32 quantized values.
__global__ void dequant_mmq_q8_1_d4(const mmq::block_q8_1_mmq* __restrict__ q,
                                    float* __restrict__ x, int cols, int n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = (int64_t)cols * n;
  if (i >= total) return;
  const int token = int(i / cols);
  const int k = int(i % cols);
  const int block = k / 128;
  const int offset = k % 128;
  const mmq::block_q8_1_mmq& qb = q[(int64_t)block * n + token];
  x[i] = float(qb.qs[offset]) * qb.d4[offset / 32];
}

} // namespace

const void* upload_bytes_cached(const void* host, std::size_t bytes) {
  return upload_bytes(host, bytes);
}

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
  for (auto& kv : g_tc_cache) cudaFree(kv.second.q);
  g_tc_cache.clear();
  if (g_ple_mapped.host) cudaHostUnregister(const_cast<void*>(g_ple_mapped.host));
  g_ple_mapped = {};
  // cuBLAS handle + workspace (C.1): destroy so a later lazy create starts fresh.
  {
    std::lock_guard<std::mutex> blk(g_cublas_mutex);
    if (g_cublas_handle) {
      cublasDestroy(g_cublas_handle);
      g_cublas_handle = nullptr;
    }
    if (g_cublas_workspace) {
      cudaFree(g_cublas_workspace);
      g_cublas_workspace = nullptr;
    }
  }
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

// Legacy 32-element gemv (Stage 2). `_base` is one thread per (row, block) with
// atomic accumulation; `_coop` gives each lane whole blocks of one row so every
// row is written exactly once. These are deliberately simple scalar loops rather
// than the dp4a bit-twiddling of the K-quant kernels: the block is only 32
// elements, so there is nothing to amortize, and the arithmetic then matches
// dequantize + multiply element for element.
#define SONICBOOM_DEFINE_LEGACY_GEMV(suffix, blkbytes, dequant)                    \
  __global__ void gemv_##suffix##_base(const uint8_t* __restrict__ w,              \
                                       const float* __restrict__ x,                \
                                       float* __restrict__ y, int blocks_per_row,  \
                                       int num_blocks) {                           \
    const int i = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (i >= num_blocks) return;                                                   \
    const int row = i / blocks_per_row;                                            \
    const int b = i % blocks_per_row;                                              \
    const uint8_t* blk = w + (size_t)i * blkbytes;                                 \
    const float* xb = x + (size_t)b * 32;                                          \
    float sum = 0.0f;                                                              \
    _Pragma("unroll") for (int l = 0; l < 32; ++l) sum += dequant(blk, l) * xb[l]; \
    atomicAdd(&y[row], sum);                                                       \
  }                                                                                \
  __global__ void gemv_##suffix##_coop(const uint8_t* __restrict__ w,              \
                                       const float* __restrict__ x,                \
                                       float* __restrict__ y, int blocks_per_row,  \
                                       int num_rows) {                             \
    const int lane = threadIdx.x;                                                  \
    const int row = blockIdx.y * blockDim.y + threadIdx.y;                         \
    if (row >= num_rows) return;                                                   \
    float sum = 0.0f;                                                              \
    for (int b = lane; b < blocks_per_row; b += 32) {                              \
      const uint8_t* blk = w + ((size_t)row * blocks_per_row + b) * blkbytes;      \
      const float* xb = x + (size_t)b * 32;                                        \
      _Pragma("unroll") for (int l = 0; l < 32; ++l) sum += dequant(blk, l) * xb[l]; \
    }                                                                              \
    _Pragma("unroll") for (int o = 16; o > 0; o >>= 1)                             \
        sum += __shfl_down_sync(0xffffffffu, sum, o);                              \
    if (lane == 0) y[row] = sum;                                                   \
  }

SONICBOOM_DEFINE_LEGACY_GEMV(q5_0, 22, dequant_q5_0_value)
SONICBOOM_DEFINE_LEGACY_GEMV(q8_0, 34, dequant_q8_0_value)
#undef SONICBOOM_DEFINE_LEGACY_GEMV

// Launch the K-quant gemv against device-resident x/y, dispatching on
// blocks-per-row (see above). `_coop` writes each row exactly once (no zero-fill
// needed); `_base` atomically accumulates, so y is zero-filled first.
void launch_gemv(QuantType qt, const uint8_t* wp, const float* xp, float* yp,
                 int blocks_per_row, int num_rows) {
  // Q1: the tuned tile config for this (gemv, shape class, SM). The winner is
  // resolved once per process and memoized. `tile_m` selects the launch width
  // of the one-thread-per-super-block kernel below — the only tile knob the
  // current hand-written kernel exposes; `tile_k` is carried for the generated
  // gemv and does not change this launch yet. With SONICBOOM_AUTOTUNE unset or
  // 0 (the default) this is the static default (tile_m 256), which is exactly
  // the pre-Q1 launch, so the gate-off path is unchanged.
  const mlir_tune::TileConfig gemv_cfg =
      mlir_tune::gemv_config(blocks_per_row, num_rows);
  const int base_block =
      (gemv_cfg.tile_m == 128 || gemv_cfg.tile_m == 512) ? gemv_cfg.tile_m
                                                         : 256;
  if (blocks_per_row >= coop_min_blocks_per_row()) {
    const dim3 block(32, 8);
    const dim3 grid(1, (num_rows + 7) / 8);
    switch (qt) {
      case QuantType::Q3_K:
        SB_NOTE_LAUNCH(gemv_q3_K_coop, grid, block, 0, cur_stream());
        gemv_q3_K_coop<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q4_K:
        SB_NOTE_LAUNCH(gemv_q4_K_coop, grid, block, 0, cur_stream());
        gemv_q4_K_coop<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q5_K:
        SB_NOTE_LAUNCH(gemv_q5_K_coop, grid, block, 0, cur_stream());
        gemv_q5_K_coop<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q6_K:
        SB_NOTE_LAUNCH(gemv_q6_K_coop, grid, block, 0, cur_stream());
        gemv_q6_K_coop<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q5_0:
        SB_NOTE_LAUNCH(gemv_q5_0_coop, grid, block, 0, cur_stream());
        gemv_q5_0_coop<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
      case QuantType::Q8_0:
        SB_NOTE_LAUNCH(gemv_q8_0_coop, grid, block, 0, cur_stream());
        gemv_q8_0_coop<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_rows);
        break;
    }
  } else {
    // Zero-fill on the current stream so the base (atomic-accumulate) gemv is
    // capturable into a CUDA graph; the synchronous cudaMemset would otherwise
    // break capture. Ordered before the gemv kernels on the same stream.
    cudaMemsetAsync(yp, 0, (size_t)num_rows * sizeof(float), cur_stream());
    const int num_blocks = num_rows * blocks_per_row;
    const dim3 block(base_block);
    const dim3 grid((num_blocks + base_block - 1) / base_block);
    switch (qt) {
      case QuantType::Q3_K:
        SB_NOTE_LAUNCH(gemv_q3_K_base, grid, block, 0, cur_stream());
        gemv_q3_K_base<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q4_K:
        SB_NOTE_LAUNCH(gemv_q4_K_base, grid, block, 0, cur_stream());
        gemv_q4_K_base<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q5_K:
        SB_NOTE_LAUNCH(gemv_q5_K_base, grid, block, 0, cur_stream());
        gemv_q5_K_base<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q6_K:
        SB_NOTE_LAUNCH(gemv_q6_K_base, grid, block, 0, cur_stream());
        gemv_q6_K_base<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q5_0:
        SB_NOTE_LAUNCH(gemv_q5_0_base, grid, block, 0, cur_stream());
        gemv_q5_0_base<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
      case QuantType::Q8_0:
        SB_NOTE_LAUNCH(gemv_q8_0_base, grid, block, 0, cur_stream());
        gemv_q8_0_base<<<grid, block, 0, cur_stream()>>>(wp, xp, yp, blocks_per_row, num_blocks);
        break;
    }
  }
}

// q8_1 mmvq selection. On by default: matvec_f32 / matvec_f32_dev / matmul_f32
// quantize x to q8_1 and run the dp4a warp-per-row kernels. SONICBOOM_GEMV_F32=1
// restores the fp32 base/coop hybrid for differential testing against the CPU
// reference (the fp32 path is the exact dequant arithmetic).
//
// Feature q8_1_dispatch; SONICBOOM_GEMV_F32 remains its live alias, so a
// differential test can set/unset it mid-process to cover both the fp32 and
// q8_1 paths in one binary. It is called once per matvec/matmul (not per
// element), so the lookup cost is negligible.
bool q8_1_enabled() noexcept {
  return runtime::feature_enabled(runtime::Feature::Q8_1Dispatch);
}

// cuBLAS fp16 path for the batched prefill matmul (C.1). On by default; the
// threshold is n >= kCublasMinN. SONICBOOM_CUBLAS=0 disables it and restores the
// dp4a batched path for A/B measurement. Also requires q8_1_enabled() (the fp32
// fallback is the CPU-matching differential path and must not be routed through
// fp16 tensor cores).
bool cublas_enabled() noexcept {
  if (std::getenv("SONICBOOM_CUBLAS") == nullptr) return true;
  return std::string_view(std::getenv("SONICBOOM_CUBLAS")) != "0";
}

// n (batch) threshold above which the cuBLAS fp16 path is used. Below it the
// dp4a batched path wins (the fp16 dequant + upload overhead is not amortized
// over enough columns). Tunable via SONICBOOM_CUBLAS_MIN_N.
constexpr int kCublasMinN = 32;
int cublas_min_n() {
  static int v = [] {
    if (const char* e = std::getenv("SONICBOOM_CUBLAS_MIN_N")) {
      const int x = std::atoi(e);
      if (x > 0) return x;
    }
    return kCublasMinN;
  }();
  return v;
}

// MMQ int8 tensor-core path (C.2). On by default; SONICBOOM_MMQ=0 disables it
// and restores the cuBLAS fp16 path at n >= cublas_min_n(). Requires
// q8_1_enabled() (the fp32 fallback is the CPU-matching differential path and
// must not be routed through int8 tensor cores).
bool mmq_enabled() noexcept {
  if (std::getenv("SONICBOOM_MMQ") == nullptr) return true;
  return std::string_view(std::getenv("SONICBOOM_MMQ")) != "0";
}

// n (batch) threshold above which the mmq path is used. Tunable via
// SONICBOOM_MMQ_MIN_N for the C.1-vs-C.2 per-n comparison.
//
// Set to 2 (not 32) so the algorithm a prefill position uses does NOT depend on
// the segment size it lands in: mmq's per-column numerics are batch-independent,
// while the dp4a batched gemv (n < threshold) and mmq (n >= threshold) disagree
// numerically. Prefix-cache reuse splits a prompt at an arbitrary non-512-aligned
// position, so the seed prefix's final partial chunk and the full prefill's
// segment covering the same positions can have different sizes; if the small
// chunk fell below the mmq threshold it would take the gemv path and the reused
// KV cache would diverge from a full prefill (n == 1 stays on the single-column
// matvec path, which the decode spine also uses).
constexpr int kMmqMinN = 2;
int mmq_min_n() {
  static int v = [] {
    if (const char* e = std::getenv("SONICBOOM_MMQ_MIN_N")) {
      const int x = std::atoi(e);
      if (x > 0) return x;
    }
    return kMmqMinN;
  }();
  return v;
}

// MMQ J-tile and decomposition autotune (Stage 6, extended by Stage 1 and
// step 9). `select_J` in the ggml-derived mmq.cuh reproduces llama.cpp's
// mmq_x rule (smallest J minimising the column-tile count ceil(n/J)); that
// ignores M, so it can leave a matmul with few row tiles spread over too few
// blocks, and it never measures which tile shape actually keeps the tensor
// cores fed on this GPU. The tuner times the candidate set (mmq_x x
// decomposition, both from upstream's Ampere table) and caches the fastest.
//
// Cache key: the weight shape class (quant type, M, blocks_per_row), never n.
// The decomposition changes the floating-point summation order (split-K pieces
// are combined in a fixed order), so batch invariance requires that the same
// rows be reduced the same way whether they are prefilled in one segment or
// many (test_prefix_cache compares 14 cases bitwise). Keying the decomposition
// by class is what guarantees that; the J that rides along in the same entry is
// numerics-neutral (it selects the token-tile width only; the fixed split-K
// pieces and the tiled K-reduction order are independent of J).
//
// The search runs on the first call for a class outside graph capture (host
// timing is illegal while capturing) and only at a batch large enough to fill
// the device (kMmqDecompProbeN): below that the class keeps the heuristic and
// its policy decomposition without storing anything, so a decode-first session
// cannot pin a decode-scale tile onto a later prefill. Inside a capture the
// cached class entry, or the heuristic + class policy when none exists, is
// used.
//
// Each class entry carries two tiles, both measured at fixed probe shapes
// (kMmqDecompProbeN = the call that triggers the search, kMmqSmallProbeN = 64
// columns of the same launch): a prefill tile and a continuation tile. The
// split exists because a tile wide enough for n=512 wastes most of its columns
// at continuation scale (n=64: 1769 tok/s with the class prefill tile vs 2712
// with upstream's rule; reuse 512+64 78.1 vs 66.1 ms), and a static
// small-batch rule still left 1.8% on that metric against the step-7
// per-call tuning. Neither the key nor the stored value depends on the call's
// n: the probes are fixed constants of the tuner. J never changes the
// summation order, so this does not weaken batch invariance; the
// decomposition, which does, is class-wide for every n.
//
// A tile or decomposition other than the default replaces it only when it
// wins by more than the switch margin (kMmqTileSwitchMargin /
// kMmqDecompSwitchMargin). The decomposition default keeps the measured 3050
// behaviour (tiled) on this GPU.
// SONICBOOM_MMQ_J=<J> pins a tile width (A/B and the unit test);
// SONICBOOM_MMQ_DECOMP=tiled|stream_k pins the decomposition (the `mmq-decomp`
// ablation group) and limits the search to that decomposition;
// SONICBOOM_FEATURE_DISABLE=autotune turns the search off (ablation group
// `autotune`).
constexpr int kMmqTuneJs[] = {8,  16,  24,  32,  40,  48,  56,  64,
                              72, 80,  88,  96, 104, 112, 120, 128};
constexpr std::size_t kMmqTuneKeyCap = 64;
constexpr int kMmqDecompProbeN = 128;
// The tuner also measures every tile at this fixed continuation width, so the
// class entry carries a small-batch tile derived from a fixed probe shape
// rather than from whichever short batch happens to arrive first.
constexpr int kMmqSmallProbeN = 64;
constexpr double kMmqDecompSwitchMargin = 0.03;
constexpr double kMmqTileSwitchMargin = 0.03;

int mmq_j_override() {
  const char* e = std::getenv("SONICBOOM_MMQ_J");
  if (e == nullptr) return 0;
  const int x = std::atoi(e);
  for (int J : kMmqTuneJs)
    if (x == J) return x;
  return 0;
}

// SONICBOOM_MMQ_DECOMP: pin the decomposition for A/B ablation. Anything but
// the two exact strings leaves the plan's choice in force.
MmqDecomp mmq_decomp_override() {
  const char* e = std::getenv("SONICBOOM_MMQ_DECOMP");
  if (e == nullptr) return MmqDecomp::Auto;
  if (std::strcmp(e, "tiled") == 0) return MmqDecomp::Tiled;
  if (std::strcmp(e, "stream_k") == 0) return MmqDecomp::StreamK;
  return MmqDecomp::Auto;
}

// Cached SM count. 1 when the query fails, which makes every geometry a pure
// tiling one (ntiles blocks, no fixup): the pre-stream-k behaviour.
int mmq_sm_count() noexcept {
  static const int n = [] {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    int v = 0;
    if (cudaDeviceGetAttribute(&v, cudaDevAttrMultiProcessorCount, dev) !=
        cudaSuccess)
      return 1;
    return v > 0 ? v : 1;
  }();
  return n;
}

// The class policy is tiled. The fixed split-K schedule is correct and
// batch-invariant, but on the target GPU (RTX 3050, sm_86, 20 SMs) it is slower
// at every measured batch size with this runtime's already-wide tiled launch:
// Gemma pp512 1741 vs 1765 tok/s (-1.4%), Qwen pp512 12750 vs 13730 (-7.3%),
// Qwen n=64 10896 vs 10932 (tie), n=128/256 -5.7%/-6.1% (3 runs each, 2026-10-
// 18, measurement script /tmp/d3-s1-decomp-ab.md). The decomposition must be
// chosen from the weight shape only (batch invariance), so there is no
// shape/batch class in which selecting split-K would help a canonical metric.
// This function is the plan's default annotation; the per-GPU tuner
// (mmq_j_select) times stream-k as the alternative candidate for each class
// and may replace this default only on a > 3% win, so on this GPU it keeps
// tiled. Stream-k stays reachable through SONICBOOM_MMQ_DECOMP=stream_k for A/B
// and is exercised by test_cuda_quant.
MmqDecomp mmq_class_decomp(uint64_t rows, uint64_t cols,
                           QuantType type) noexcept {
  (void)rows;
  (void)cols;
  (void)type;
  return MmqDecomp::Tiled;
}

namespace {

// Auto resolves to the class policy (tiled). Batch size deliberately does not
// enter: the decomposition changes the floating-point summation order, so a
// choice that moved with n would make prefix-cache reuse diverge from a full
// prefill.
template <int Type>
MmqDecomp mmq_resolve_decomp(MmqDecomp requested, int M, int bpr) {
  (void)M;
  (void)bpr;
  if (requested != MmqDecomp::Auto) return requested;
  return MmqDecomp::Tiled;
}

// The shape class: weight shape + quant type. Deliberately no n and no
// requested decomposition -- the class is what makes the stored choice, in
// particular the numerics-bearing decomposition, independent of the batch
// (see the comment block at the top of this tuner).
struct MmqClassKey {
  int type;
  int rows;
  int bpr;
  bool operator==(const MmqClassKey&) const = default;
};

struct MmqClassKeyHash {
  std::size_t operator()(const MmqClassKey& k) const noexcept {
    std::uint64_t h = 0x9E3779B97F4A7C15ull;
    h ^= std::uint64_t(std::uint32_t(k.type)) * 0xC2B2AE3D27D4EB4Full;
    h ^= std::uint64_t(std::uint32_t(k.rows)) * 0x165667B19E3779F9ull;
    h ^= std::uint64_t(std::uint32_t(k.bpr)) * 0x9E3779B185EBCA87ull;
    return std::size_t(h ^ (h >> 29));
  }
};

// The tuner's result for a class: the prefill J, the continuation J measured
// at the fixed small probe width (0 when unknown), and the decomposition.
struct MmqChoice {
  int j;
  MmqDecomp decomp;
  int j_small = 0;
};

std::mutex g_mmq_j_mutex;
std::unordered_map<MmqClassKey, MmqChoice, MmqClassKeyHash> g_mmq_j_choice;

// Scratch for the fixed split-K partials (one J*I block per work unit; see
// split_k_scratch_floats). Grown, never shrunk.
void* g_mmq_fixup = nullptr;
size_t g_mmq_fixup_bytes = 0;

template <int Type, int J>
void mmq_launch_one(MmqDecomp decomp, const char* w, const int* y, float* dst,
                    int M, int n, int bpr, cudaStream_t stream) {
  if (decomp == MmqDecomp::StreamK) {
    const int nsm = mmq_sm_count();
    const int parts = mmq::split_k_parts(M, bpr, mmq::qk_of<Type>(), nsm);
    if (parts >= 2) {
      const size_t bytes =
          size_t(mmq::split_k_scratch_floats(M, n, J, parts)) * sizeof(float);
      float* tmp =
          static_cast<float*>(grow_scratch(g_mmq_fixup, g_mmq_fixup_bytes, bytes));
      if (tmp) {
        mmq::launch_mul_mat_q_split_k<Type, J>(w, y, dst, tmp, M, n, bpr, nsm,
                                               parts, stream);
        return;
      }
    }
  }
  mmq::launch_mul_mat_q<Type, J>(w, y, dst, M, n, bpr, stream);
}

template <int Type>
void mmq_launch(int J, MmqDecomp decomp, const char* w, const int* y, float* dst,
                int M, int n, int bpr, cudaStream_t stream) {
  switch (J) {
    case 8:   mmq_launch_one<Type, 8>  (decomp, w, y, dst, M, n, bpr, stream); break;
    case 16:  mmq_launch_one<Type, 16> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 24:  mmq_launch_one<Type, 24> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 32:  mmq_launch_one<Type, 32> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 40:  mmq_launch_one<Type, 40> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 48:  mmq_launch_one<Type, 48> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 56:  mmq_launch_one<Type, 56> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 64:  mmq_launch_one<Type, 64> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 72:  mmq_launch_one<Type, 72> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 80:  mmq_launch_one<Type, 80> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 88:  mmq_launch_one<Type, 88> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 96:  mmq_launch_one<Type, 96> (decomp, w, y, dst, M, n, bpr, stream); break;
    case 104: mmq_launch_one<Type, 104>(decomp, w, y, dst, M, n, bpr, stream); break;
    case 112: mmq_launch_one<Type, 112>(decomp, w, y, dst, M, n, bpr, stream); break;
    case 120: mmq_launch_one<Type, 120>(decomp, w, y, dst, M, n, bpr, stream); break;
    case 128: mmq_launch_one<Type, 128>(decomp, w, y, dst, M, n, bpr, stream); break;
    default:  mmq_launch_one<Type, 128>(decomp, w, y, dst, M, n, bpr, stream); break;
  }
}

template <int Type>
MmqChoice mmq_j_select(const char* w, const int* y, float* dst, int M, int n,
                       int bpr, MmqDecomp requested, cudaStream_t stream) {
  // The env switch is an ablation override of the plan's class decision; the
  // plan annotation itself is only the default candidate the tuner may
  // replace (by more than kMmqDecompSwitchMargin).
  const MmqDecomp env = mmq_decomp_override();
  const bool decomp_pinned = env != MmqDecomp::Auto;
  const MmqDecomp want = decomp_pinned ? env : requested;

  const int heuristic = mmq::select_J(n);
  const MmqDecomp policy = mmq_resolve_decomp<Type>(want, M, bpr);
  const MmqChoice fallback{heuristic, policy};

  const int forced = mmq_j_override();
  if (forced != 0) return MmqChoice{forced, policy};
  if (runtime::feature_disabled(runtime::Feature::Autotune)) return fallback;

  // The class entry's J is a prefill tile: it is applied only at the probe
  // scale and above. Below it, upstream's per-call rule (select_J(n)) applies
  // -- a tile tuned at n=512 computes up to twice the requested columns at
  // continuation scale (measured: reuse 512+64 78.1 ms with the class tile vs
  // 66.1 ms with upstream's, n=64 prefill 1769 vs 2712 tok/s). J is
  // numerics-neutral either way (test_cuda_quant: "mmq J invariant ... tiled
  // identical"), so this does not weaken batch invariance: the decomposition,
  // which does change the summation order, stays class-wide for every n.
  const auto apply_regime = [&](MmqChoice c) -> MmqChoice {
    if (decomp_pinned) c.decomp = policy;
    if (n < kMmqDecompProbeN) c.j = c.j_small > 0 ? c.j_small : heuristic;
    return c;
  };

  const MmqClassKey key{Type, M, bpr};
  {
    std::lock_guard<std::mutex> lk(g_mmq_j_mutex);
    const auto it = g_mmq_j_choice.find(key);
    if (it != g_mmq_j_choice.end()) return apply_regime(it->second);
    if (g_mmq_j_choice.size() >= kMmqTuneKeyCap) return fallback;
  }

  // Step 7 persistence, step 9 keys: one entry per class, holding both
  // numerics-neutral tiles and the class decomposition. The key is versioned
  // because changes to what a stored J means or to the cost it was measured
  // under invalidate the entry: c4 = step-9 candidate set, regime rule and
  // continuation-tile selection; c5 = the tiled launch's block order (token
  // tile fastest, for L2 weight reuse), which changes the tiled cost per J.
  const std::string disk_key =
      std::string("c5:t") + std::to_string(Type) + "-M" + std::to_string(M) +
      "-bpr" + std::to_string(bpr);
  {
    const std::map<std::string, int> cached =
        ::sonicboom::tune::lookup("mmq-j", disk_key);
    const auto jt = cached.find("j");
    const auto dt = cached.find("decomp");
    const auto st = cached.find("j_small");
    if (jt != cached.end() && dt != cached.end()) {
      bool valid_j = false;
      for (int J : kMmqTuneJs) valid_j = valid_j || (J == jt->second);
      const int j_small = st != cached.end() ? st->second : 0;
      if (valid_j && dt->second >= int(MmqDecomp::Tiled) &&
          dt->second <= int(MmqDecomp::StreamK)) {
        const MmqChoice c{jt->second,
                          static_cast<MmqDecomp>(dt->second), j_small};
        std::lock_guard<std::mutex> lk(g_mmq_j_mutex);
        g_mmq_j_choice.emplace(key, c);
        return apply_regime(c);
      }
    }
  }

  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
      capture != cudaStreamCaptureStatusNone)
    return fallback;

  // Do not pin a class from a batch too small to fill the device: the entry is
  // stored for every later n of that class, so a decode-scale first call would
  // otherwise decide the tile width and the decomposition of the prefill.
  if (n < kMmqDecompProbeN) return fallback;

  // Candidate set: upstream's mmq_x values (every multiple of 8, the whole
  // non-fallback Ampere J set) x the two decompositions. A pinned
  // decomposition times only itself. Our stream-k is a fixed split over K;
  // when it degenerates to one part it is the tiled kernel bit for bit, so the
  // duplicate candidate is skipped.
  MmqDecomp candidates[2];
  int ncandidates = 0;
  candidates[ncandidates++] = policy;
  if (!decomp_pinned) {
    const MmqDecomp other =
        policy == MmqDecomp::StreamK ? MmqDecomp::Tiled : MmqDecomp::StreamK;
    const bool other_is_kernel =
        other != MmqDecomp::StreamK ||
        mmq::split_k_parts(M, bpr, mmq::qk_of<Type>(), mmq_sm_count()) >= 2;
    if (other_is_kernel) candidates[ncandidates++] = other;
  }

  const bool verbose = std::getenv("SONICBOOM_AUTOTUNE_VERBOSE") != nullptr;
  cudaEvent_t begin{}, end{};
  if (cudaEventCreate(&begin) != cudaSuccess ||
      cudaEventCreate(&end) != cudaSuccess) {
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    return fallback;
  }
  auto time_one = [&](int J, MmqDecomp decomp, int nn) -> float {
    mmq_launch<Type>(J, decomp, w, y, dst, M, nn, bpr, stream);  // warm-up
    if (cudaEventRecord(begin, stream) != cudaSuccess) return -1.0f;
    constexpr int kReps = 2;
    for (int r = 0; r < kReps; ++r)
      mmq_launch<Type>(J, decomp, w, y, dst, M, nn, bpr, stream);
    if (cudaEventRecord(end, stream) != cudaSuccess) return -1.0f;
    if (cudaEventSynchronize(end) != cudaSuccess) {
      (void)cudaGetLastError();
      return -1.0f;
    }
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess ||
        cudaGetLastError() != cudaSuccess) {
      (void)cudaGetLastError();
      return -1.0f;
    }
    return ms;
  };

  const int heuristic_small = mmq::select_J(kMmqSmallProbeN);
  int cand_j[2] = {heuristic, heuristic};
  int cand_j_small[2] = {heuristic_small, heuristic_small};
  float cand_ms[2] = {0.0f, 0.0f};
  float cand_ms_small[2] = {0.0f, 0.0f};
  float heur_ms[2] = {0.0f, 0.0f};
  float heur_ms_small[2] = {0.0f, 0.0f};
  bool cand_have[2] = {false, false};
  bool cand_have_small[2] = {false, false};
  bool heur_have[2] = {false, false};
  bool heur_have_small[2] = {false, false};
  for (int c = 0; c < ncandidates; ++c) {
    const MmqDecomp decomp = candidates[c];
    for (int J : kMmqTuneJs) {
      const float ms = time_one(J, decomp, n);
      // The same tile at the fixed continuation width: dst has at least
      // kMmqDecompProbeN columns at this point, so writing
      // kMmqSmallProbeN columns is in bounds and touches only the columns the
      // following real launch would rewrite anyway.
      const float ms_small = time_one(J, decomp, kMmqSmallProbeN);
      if (ms >= 0.0f) {
        if (J == heuristic) {
          heur_ms[c] = ms;
          heur_have[c] = true;
        }
        if (!cand_have[c] || ms < cand_ms[c]) {
          cand_ms[c] = ms;
          cand_j[c] = J;
          cand_have[c] = true;
        }
      }
      if (ms_small >= 0.0f) {
        if (J == heuristic_small) {
          heur_ms_small[c] = ms_small;
          heur_have_small[c] = true;
        }
        if (!cand_have_small[c] || ms_small < cand_ms_small[c]) {
          cand_ms_small[c] = ms_small;
          cand_j_small[c] = J;
          cand_have_small[c] = true;
        }
      }
    }
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  if (!cand_have[0]) return fallback;

  // Per decomposition, the upstream rule at the probe shape is the default and
  // a different tile replaces it only on a clear win. Two reps can resolve a
  // few percent at best, and a narrow, noisy "win" at one n is exactly what
  // would be carried to every other n of the class.
  int chosen_j[2] = {heuristic, heuristic};
  int chosen_j_small[2] = {heuristic_small, heuristic_small};
  float chosen_ms[2] = {0.0f, 0.0f};
  for (int c = 0; c < ncandidates; ++c) {
    if (!cand_have[c]) continue;
    if (heur_have[c] &&
        cand_ms[c] < heur_ms[c] * (1.0f - float(kMmqTileSwitchMargin))) {
      chosen_j[c] = cand_j[c];
      chosen_ms[c] = cand_ms[c];
    } else {
      chosen_j[c] = heuristic;
      chosen_ms[c] = heur_have[c] ? heur_ms[c] : cand_ms[c];
    }
    // No switch margin for the continuation tile: it is measured at exactly
    // the width it is applied to (unlike the prefill tile, whose win has to
    // survive transfer to other n), and the step-7 per-call tuner's margin-free
    // pick is what this reproduces. The 3% margin here cost 0.7% on reuse.
    if (cand_have_small[c]) chosen_j_small[c] = cand_j_small[c];
  }

  int best_c = 0;
  float best_ms = chosen_ms[0];
  for (int c = 1; c < ncandidates; ++c) {
    // The policy decomposition keeps its place unless the alternative wins by
    // more than the switch margin; the 3050 measures tiled faster everywhere,
    // so in practice this only guards against noise flipping the choice (and
    // with it the summation order) between runs.
    if (cand_have[c] &&
        chosen_ms[c] < best_ms * (1.0f - float(kMmqDecompSwitchMargin))) {
      best_c = c;
      best_ms = chosen_ms[c];
    }
  }
  const MmqChoice best{chosen_j[best_c], candidates[best_c],
                       chosen_j_small[best_c]};
  {
    std::lock_guard<std::mutex> lk(g_mmq_j_mutex);
    g_mmq_j_choice.emplace(key, best);
  }
  ::sonicboom::tune::store(
      "mmq-j", disk_key,
      {{"j", best.j}, {"decomp", int(best.decomp)}, {"j_small", best.j_small}});
  if (verbose) {
    std::fprintf(stderr,
                 "[autotune] mmq type=%d M=%d bpr=%d (probe n=%d) "
                 "decomp=%s J=%d %.3f ms, small J=%d (heuristic %d/%d); "
                 "%s J=%d %.3f ms, small J=%d\n",
                 Type, M, bpr, n,
                 best.decomp == MmqDecomp::StreamK ? "stream_k" : "tiled",
                 best.j, best_ms, best.j_small, heuristic, heuristic_small,
                 ncandidates > 1
                     ? (candidates[1] == MmqDecomp::StreamK ? "stream_k"
                                                            : "tiled")
                     : "none",
                 chosen_j[1], chosen_ms[1], chosen_j_small[1]);
  }
  return best;
}

}  // namespace

template <int Type>
bool mul_mat_q_tuned(const char* w, const int* y, float* dst, int M, int n,
                     int bpr, MmqDecomp decomp, cudaStream_t stream) {
  const MmqChoice c = mmq_j_select<Type>(w, y, dst, M, n, bpr, decomp, stream);
  mmq_launch<Type>(c.j, c.decomp, w, y, dst, M, n, bpr, stream);
  return true;
}

// Lazily create the process-wide cuBLAS handle and its stream-capture workspace.
// Returns nullptr on failure (the caller falls back to the dp4a batched path).
cublasHandle_t cublas_handle() {
  std::lock_guard<std::mutex> lk(g_cublas_mutex);
  if (!g_cublas_handle) {
    if (cublasCreate(&g_cublas_handle) != CUBLAS_STATUS_SUCCESS) {
      g_cublas_handle = nullptr;
      return nullptr;
    }
    if (cudaMalloc(&g_cublas_workspace, kCublasWorkspaceBytes) == cudaSuccess)
      cublasSetWorkspace(g_cublas_handle, g_cublas_workspace,
                         kCublasWorkspaceBytes);
  }
  return g_cublas_handle;
}

// Which quant types' gemv kernels carry the optional epilogue operands (`bias`
// and `add`, the latter first, of the warp-per-row mmvq kernels). Types outside
// this set run without an epilogue, so a caller that must add a bias or a
// residual either falls back to a separate elementwise kernel or declines the
// fusion.
// Ablation for the Q6_K epilogue (G1): SONICBOOM_GEMV_Q6K_EPILOGUE=0 reports
// Q6_K unsupported again, so the projection runs the plain gemv followed by
// the separate add_bias/add kernels — the pre-G1 launch sequence. Read once,
// like the other backend switches.
static bool gemv_q6k_epilogue_enabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("SONICBOOM_GEMV_Q6K_EPILOGUE");
    return !(e && std::string_view(e) == "0");
  }();
  return enabled;
}

bool gemv_bias_epilogue_supported(QuantType qt) {
  switch (qt) {
    case QuantType::Q3_K:
    case QuantType::Q4_K:
    case QuantType::Q5_K:
    case QuantType::Q5_0:
    case QuantType::Q8_0:
      return true;
    case QuantType::Q6_K:
      return gemv_q6k_epilogue_enabled();
    default:
      return false;
  }
}

// The warp-per-row dp4a gemv against a *pre-quantized* q8_1 activation. Shared
// by launch_gemv_q8_1 (quantize + gemv in one call) and gemv_q8_1_dev (the
// quantize-once path, where the activation is already a q8_1 tensor).
// `bias` (a bound weight) and `add` (a device activation: the residual of a
// folded `Add(proj, residual)`) are the optional [num_rows] epilogue operands,
// applied in that order; the caller must pass null for a type where
// gemv_bias_epilogue_supported() is false.
// One q8_1 warp-per-row launch with `warps` rows per block. Each row is
// computed entirely by its own warp (row = blockIdx.y * blockDim.y +
// threadIdx.y, reduction by __shfl_down within that warp), so blockDim.y
// selects only how many rows share a block: the per-row arithmetic, and
// therefore every output byte, is independent of it.
static void launch_gemv_q8_1_rows(QuantType qt, const uint8_t* wp,
                                  const q8_1::block_q8_1* xq, float* yp,
                                  int blocks_per_row, int num_rows,
                                  const float* bias, const float* add,
                                  int warps) {
  const dim3 block(32, warps);
  const dim3 grid(1, (num_rows + warps - 1) / warps);
  switch (qt) {
    case QuantType::Q3_K:
      SB_NOTE_LAUNCH(q8_1::gemv_q3_K_q8_1, grid, block, 0, cur_stream());
      q8_1::gemv_q3_K_q8_1<<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, bias, add);
      break;
    case QuantType::Q4_K:
      SB_NOTE_LAUNCH(q8_1::gemv_q4_K_q8_1, grid, block, 0, cur_stream());
      q8_1::gemv_q4_K_q8_1<<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, bias, add);
      break;
    case QuantType::Q5_K:
      SB_NOTE_LAUNCH(q8_1::gemv_q5_K_q8_1, grid, block, 0, cur_stream());
      q8_1::gemv_q5_K_q8_1<<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, bias, add);
      break;
    case QuantType::Q6_K:
      SB_NOTE_LAUNCH(q8_1::gemv_q6_K_q8_1, grid, block, 0, cur_stream());
      q8_1::gemv_q6_K_q8_1<<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, bias, add);
      break;
    case QuantType::Q5_0:
      SB_NOTE_LAUNCH(q8_1::gemv_q5_0_q8_1, grid, block, 0, cur_stream());
      q8_1::gemv_q5_0_q8_1<<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, bias, add);
      break;
    case QuantType::Q8_0:
      SB_NOTE_LAUNCH(q8_1::gemv_q8_0_q8_1, grid, block, 0, cur_stream());
      q8_1::gemv_q8_0_q8_1<<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, bias, add);
      break;
  }
}

// tile_m is the block width in threads — Q1's meaning for the fp32 gemv launch
// — so 128 and 512 threads are 4 and 16 rows per block; anything else is the
// pre-Q1 (32, 8).
static int gemv_warps_for(const mlir_tune::TileConfig& cfg) {
  return cfg.tile_m == 128 ? 4 : cfg.tile_m == 512 ? 16 : 8;
}

// gridDim.y caps a launch at 65535 blocks, so a stored narrow block (fewer rows
// per block, more blocks) is unsupported for a very tall output. The wide rows
// buckets only keep the 262144-row logits projection and the 10240-row FFN in
// different classes; they do not remove the cap, so the choice is still
// widened to the narrowest supported block rather than failing the launch.
static int gemv_warps_for_launch(const mlir_tune::TileConfig& cfg,
                                 int num_rows) {
  int warps = gemv_warps_for(cfg);
  while (int64_t(warps) * 65535 < num_rows) warps *= 2;
  return warps;
}

// Q2: resolve this shape class's launch width. The first call for a class under
// SONICBOOM_AUTOTUNE=1 times the three candidate widths on the live buffers
// through the Q1 runner, which accepts a width only when its output matches the
// 256-thread launch bit for bit, and persists the winner. Skipped inside a CUDA
// graph capture, where the host must not synchronize or launch probes; the
// class stays unresolved and is tuned by the first call outside a capture.
static int resolve_gemv_warps(QuantType qt, const uint8_t* wp,
                              const q8_1::block_q8_1* xq, float* yp,
                              int blocks_per_row, int num_rows,
                              const float* bias, const float* add) {
  const mlir_tune::GemvChoice choice =
      mlir_tune::gemv_choice(blocks_per_row, num_rows);
  if (choice.resolved())
    return gemv_warps_for_launch(choice.cfg, num_rows);

  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(cur_stream(), &capture) != cudaSuccess ||
      capture != cudaStreamCaptureStatusNone)
    return gemv_warps_for_launch(choice.cfg, num_rows);

  static const mlir_tune::TileConfig kWidths[] = {
      mlir_tune::default_config("gemv"),
      mlir_tune::TileConfig{128, 1, 32, 1, 1},
      mlir_tune::TileConfig{512, 1, 32, 1, 1},
  };
  mlir_tune::AutotuneRunner::Problem p;
  p.candidates.assign(std::begin(kWidths), std::end(kWidths));
  p.compile = [&](const mlir_tune::TileConfig& cfg)
      -> std::optional<mlir_tune::AutotuneRunner::Launch> {
    const int warps = gemv_warps_for(cfg);
    // gridDim.y caps the launch at 65535 blocks: a narrow block (more rows per
    // block) is unsupported for a very tall output, e.g. the 262144-row logits
    // projection. Such a candidate is skipped, not launched and timed as a
    // fast failure.
    if (int64_t(warps) * 65535 < num_rows) return std::nullopt;
    // Each "launch" is a short burst of the same gemv. The runner times one
    // launch per host wait, and a wait costs about as much as the smaller
    // projections; a burst amortizes that floor so the 1-3% width differences
    // are measurable. Every launch in the burst writes the same bytes.
    constexpr int kProbeReps = 16;
    return mlir_tune::AutotuneRunner::Launch([=] {
      for (int i = 0; i < kProbeReps; ++i)
        launch_gemv_q8_1_rows(qt, wp, xq, yp, blocks_per_row, num_rows, bias,
                              add, warps);
    });
  };
  // The runner compares on the host, so the window is a device-to-host copy of
  // this call's own output (every width writes each row exactly once).
  std::vector<float> host_out(static_cast<std::size_t>(num_rows));
  p.output = host_out.data();
  p.output_bytes = host_out.size() * sizeof(float);
  p.sync = [&] {
    // The copy is enqueued behind the candidate on the same stream, so an
    // iteration pays one host wait, not a wait plus a synchronous copy. The
    // copy time is identical for every candidate and cancels in the ranking.
    cudaError_t e = cudaMemcpyAsync(host_out.data(), yp, p.output_bytes,
                                    cudaMemcpyDeviceToHost, cur_stream());
    if (e == cudaSuccess) e = cudaStreamSynchronize(cur_stream());
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[autotune] gemv probe launch: %s\n",
                   cudaGetErrorString(e));
      // Poison the window so the runner's bit-identity check rejects a failed
      // candidate instead of seeing the previous candidate's output, and clear
      // the error so it cannot fail the caller's node.
      std::memset(host_out.data(), 0xFF, p.output_bytes);
      cudaGetLastError();
    }
  };
  // The widths differ by at most a few percent on a class where they differ at
  // all, and the choice is persisted, so the measurement must be stable: 12
  // timed bursts of 16 launches, min. Timing waits on the stream only; the
  // bit-identity copy is the check runs' job, not the timed ones'.
  p.timing_sync = [&] { (void)cudaStreamSynchronize(cur_stream()); };
  p.iters = 12;
  const auto best = mlir_tune::AutotuneRunner::run(p);
  if (!best) return gemv_warps_for_launch(choice.cfg, num_rows);

  // The production width is the reference to beat. A candidate replaces it
  // only when it is faster by more than the probe's resolution: on the
  // memory-bound classes (the 151936-row logits projection, the 4864-row FFN)
  // the three widths are indistinguishable per kernel, and a pick chosen from
  // that noise flips between processes and is persisted that way.
  const mlir_tune::TileConfig production = mlir_tune::default_config("gemv");
  mlir_tune::TileConfig winner = *best;
  if (!(winner == production)) {
    // 100 launches per burst: 16 leaves a few percent of burst-to-burst jitter
    // on the short kernels, which reads as noise and rejects a real win.
    constexpr int kProbeReps = 100;
    auto timed_burst = [&](const mlir_tune::TileConfig &cfg,
                           double &ms) -> bool {
      const int warps = gemv_warps_for(cfg);
      if (int64_t(warps) * 65535 < num_rows) return false;
      if (cudaStreamSynchronize(cur_stream()) != cudaSuccess) return false;
      const auto t0 = std::chrono::steady_clock::now();
      for (int r = 0; r < kProbeReps; ++r)
        launch_gemv_q8_1_rows(qt, wp, xq, yp, blocks_per_row, num_rows, bias,
                              add, warps);
      if (cudaStreamSynchronize(cur_stream()) != cudaSuccess) return false;
      ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
               .count();
      return cudaGetLastError() == cudaSuccess;
    };
    // The first calls of a process run while the memory clock is still ramping
    // (idle 810 MHz to 6801 MHz on this part), so three untimed bursts bring it
    // to steady state before anything is compared. The five timed pairs then
    // alternate order so a residual drift does not favour whichever side runs
    // second. The min of the five references is the target; their spread is
    // diagnostic.
    constexpr int kRounds = 5;
    double refs[kRounds] = {}, wins[kRounds] = {}, scratch = 0;
    bool ok = true;
    for (int i = 0; i < 3 && ok; ++i) ok = timed_burst(production, scratch);
    for (int i = 0; ok && i < kRounds; ++i) {
      ok = i % 2 == 0
               ? timed_burst(production, refs[i]) && timed_burst(winner, wins[i])
               : timed_burst(winner, wins[i]) && timed_burst(production, refs[i]);
    }
    if (ok) {
      double ref = refs[0], win = wins[0];
      double spread = 0;
      for (int i = 1; i < kRounds; ++i) {
        spread = std::max(spread, std::fabs(refs[i] - refs[0]));
        ref = std::min(ref, refs[i]);
        win = std::min(win, wins[i]);
      }
      // The min of five bursts reproduces to a few tenths of a percent per
      // side across processes on this part, so a pick closer than 0.5% to
      // production is not reproducible and production is kept. The spread is only
      // diagnostic: a large one means a burst was disturbed (a transient
      // slows a burst, it does not speed one up), not that the two sides
      // differ by that much — which is why the min, not the spread, is
      // compared.
      constexpr double kResolution = 0.005;
      if (!(win < ref * (1.0 - kResolution))) winner = production;
      if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE"))
        std::fprintf(stderr,
                     "[autotune] gemv probe: prod=%.4f ms best=%.4f ms "
                     "spread=%.2f%% -> %s\n",
                     ref * 1e3, win * 1e3, spread / refs[0] * 100.0,
                     winner == production ? "keep production" : "take");
    }
  }

  const bool stored = mlir_tune::gemv_choice_store(blocks_per_row, num_rows,
                                                   winner);
  std::fprintf(stderr, "[autotune] gemv %s: tile_m=%d (%d warps/block)%s\n",
               mlir_tune::gemv_shape_class(blocks_per_row, num_rows).c_str(),
               winner.tile_m, gemv_warps_for_launch(winner, num_rows),
               stored ? "" : " [not persisted]");
  return gemv_warps_for_launch(winner, num_rows);
}

void launch_gemv_q8_1_from_quantized(QuantType qt, const uint8_t* wp,
                                     const q8_1::block_q8_1* xq, float* yp,
                                     int blocks_per_row, int num_rows,
                                     const float* bias = nullptr,
                                     const float* add = nullptr) {
  const int warps = resolve_gemv_warps(qt, wp, xq, yp, blocks_per_row, num_rows,
                                       bias, add);
  launch_gemv_q8_1_rows(qt, wp, xq, yp, blocks_per_row, num_rows, bias, add,
                        warps);
}

// Batched q8_1 gemv against a pre-quantized [n, blocks_per_row] activation (each
// position's blocks contiguous). One warp per output row; each weight block is
// reused across a tile of T positions. dst is [n, num_rows] (position-major).
void launch_gemv_q8_1_batched(QuantType qt, const uint8_t* wp,
                              const q8_1::block_q8_1* xq, float* yp,
                              int blocks_per_row, int num_rows, int n) {
  const dim3 block(32, 8);
  const dim3 grid(1, (num_rows + 7) / 8);
  constexpr int T = q8_1::GEMV_BATCH_T;
  switch (qt) {
    case QuantType::Q3_K:
      SB_NOTE_LAUNCH_TP("q8_1::gemv_q3_K_q8_1_batched",
                        ::sonicboom::planner::plan_detail::tp_label({{"T", T}}),
                        grid, block, 0, cur_stream());
      q8_1::gemv_q3_K_q8_1_batched<T><<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, n);
      break;
    case QuantType::Q4_K:
      SB_NOTE_LAUNCH_TP("q8_1::gemv_q4_K_q8_1_batched",
                        ::sonicboom::planner::plan_detail::tp_label({{"T", T}}),
                        grid, block, 0, cur_stream());
      q8_1::gemv_q4_K_q8_1_batched<T><<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, n);
      break;
    case QuantType::Q5_K:
      SB_NOTE_LAUNCH_TP("q8_1::gemv_q5_K_q8_1_batched",
                        ::sonicboom::planner::plan_detail::tp_label({{"T", T}}),
                        grid, block, 0, cur_stream());
      q8_1::gemv_q5_K_q8_1_batched<T><<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, n);
      break;
    case QuantType::Q6_K:
      SB_NOTE_LAUNCH_TP("q8_1::gemv_q6_K_q8_1_batched",
                        ::sonicboom::planner::plan_detail::tp_label({{"T", T}}),
                        grid, block, 0, cur_stream());
      q8_1::gemv_q6_K_q8_1_batched<T><<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, n);
      break;
    case QuantType::Q5_0:
      SB_NOTE_LAUNCH_TP("q8_1::gemv_q5_0_q8_1_batched",
                        ::sonicboom::planner::plan_detail::tp_label({{"T", T}}),
                        grid, block, 0, cur_stream());
      q8_1::gemv_q5_0_q8_1_batched<T><<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, n);
      break;
    case QuantType::Q8_0:
      SB_NOTE_LAUNCH_TP("q8_1::gemv_q8_0_q8_1_batched",
                        ::sonicboom::planner::plan_detail::tp_label({{"T", T}}),
                        grid, block, 0, cur_stream());
      q8_1::gemv_q8_0_q8_1_batched<T><<<grid, block, 0, cur_stream()>>>(
          wp, xq, yp, blocks_per_row, num_rows, n);
      break;
  }
}

// Quantize x -> q8_1 into `yq` (nblocks blocks, nblocks == n / 32). The
// quantize-once seam: exposed so the CUDA backend can run a planner-inserted
// explicit Quantize node whose output is shared by several gemvs.
bool quantize_q8_1_dev(const float* x, BlockQ8_1* yq, int nblocks) {
  if (!available() || !x || !yq || nblocks <= 0) return false;
  SB_NOTE_LAUNCH(q8_1::quantize_q8_1, q8_1::quantize_q8_1_grid(nblocks), q8_1::kQuantizeBlockThreads, 0, cur_stream());
  q8_1::quantize_q8_1<<<q8_1::quantize_q8_1_grid(nblocks), q8_1::kQuantizeBlockThreads, 0, cur_stream()>>>(
      x, reinterpret_cast<q8_1::block_q8_1*>(yq), nblocks);
  return cudaGetLastError() == cudaSuccess;
}

// rms_norm + q8_1 quantize fused (T3): one kernel computes the rms_norm
// sum-of-squares (double, warp-shuffle + shared reduction, matching
// nn::cuda::rms_norm_dev's rms_sum_sq_kernel), derives
// inv = 1/sqrtf(float(sum/n) + eps), then normalizes and quantizes to q8_1 in
// the same pass — the f32 normalized activation is never materialized. The
// normalization + amax/127 quantization is bit-identical to rms_norm_dev
// followed by quantize_q8_1_dev.
//
// The block is fixed at 1024 threads so the sum-of-squares accumulation order
// (thread t reads x[t], x[t+1024], …) is identical to rms_norm_fused_kernel,
// keeping `inv` bit-identical to the unfused path.
//
// Quantization is one warp per 32-element q8_1 block, one lane per element —
// the same mapping as quantize_q8_1. Each lane computes its normalized value
// once (kept in a register and reused by the amax butterfly), so the block
// costs two coalesced 128 B loads instead of 32 strided lines per warp, and no
// thread runs a 32-iteration serial chain.
template <typename R>
__global__ void __launch_bounds__(1024) rms_norm_quantize_q8_1_kernel(
    const float* __restrict__ x, const float* __restrict__ weight, int n, float eps,
    q8_1::block_q8_1* __restrict__ y) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ typename R::part_t warp_part[32];
  __shared__ float inv;
  const typename R::acc_t t =
      sonicboom::nn::cuda::block_sum_sq<R>(x, n, warp_part);
  if (tid == 0) inv = 1.0f / sqrtf(R::mean_to_float(t, n) + eps);
  __syncthreads();
  const float iv = inv;
  const int nblocks = n / 32;
  for (int ib = warp; ib < nblocks; ib += blockDim.x >> 5) {
    const size_t base = (size_t)ib * 32;
    const float xv = x[base + lane];
    const float nv = weight ? xv * iv * weight[base + lane] : xv * iv;
    // amax: same fmaxf(fabsf(nv)) fold as the per-thread serial scan, just
    // butterfly-reduced across the lanes of the block's warp.
    float amax = fmaxf(0.0f, fabsf(nv));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float d = amax / 127.0f;
    if (lane == 0) y[ib].d = d;
    y[ib].qs[lane] = (d == 0.0f) ? int8_t(0) : int8_t(roundf(nv / d));
  }
}

bool rms_norm_quantize_q8_1_dev(std::span<const float> weight, const float* x,
                                float eps, BlockQ8_1* yq, int n) {
  if (!available() || !x || !yq || n <= 0 || (n & 31) != 0) return false;
  if (!weight.empty() && weight.size() < (size_t)n) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(upload_bytes(weight.data(), n * sizeof(float)));
    if (!d_w) return false;
  }
  SB_NOTE_LAUNCH(rms_norm_quantize_q8_1_kernel, 1, 1024, 0, cur_stream());
  auto* y = reinterpret_cast<q8_1::block_q8_1*>(yq);
  cudaStream_t st = cur_stream();
  switch (sonicboom::planner::device_reduce_mode()) {
    case sonicboom::compiler::ReduceMode::Df32:
      rms_norm_quantize_q8_1_kernel<sonicboom::nn::cuda::SumDf32>
          <<<1, 1024, 0, st>>>(x, d_w, n, eps, y);
      break;
    case sonicboom::compiler::ReduceMode::F32:
      rms_norm_quantize_q8_1_kernel<sonicboom::nn::cuda::SumF32>
          <<<1, 1024, 0, st>>>(x, d_w, n, eps, y);
      break;
    default:
      rms_norm_quantize_q8_1_kernel<sonicboom::nn::cuda::SumF64>
          <<<1, 1024, 0, st>>>(x, d_w, n, eps, y);
      break;
  }
  return cudaGetLastError() == cudaSuccess;
}

// gelu×up -> q8_1 fused (T5): one thread per 32-element q8_1 block (matching
// quantize_q8_1), computing gelu_fp16(gate) * up and amax/127-quantizing the
// product in one pass — the f32 gelu and fh intermediates are never
// materialized. Bit-identical to gelu_fp16_dev + mul_dev + quantize_q8_1_dev:
// the gelu/mul are pure functions of gate/up, recomputed identically in the
// amax and quantize passes.
__global__ void gelu_mul_quantize_q8_1_kernel(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              int n, q8_1::block_q8_1* __restrict__ y) {
  const int ib = blockIdx.x * blockDim.x + threadIdx.x;
  const int nblocks = n / 32;
  if (ib >= nblocks) return;
  const float* gb = gate + (size_t)ib * 32;
  const float* ub = up + (size_t)ib * 32;

  float amax = 0.0f;
  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    const float fh = gelu_fp16(gb[k]) * ub[k];
    amax = fmaxf(amax, fabsf(fh));
  }
  const float d = amax / 127.0f;
  y[ib].d = d;
  #pragma unroll
  for (int k = 0; k < 32; ++k) {
    const float fh = gelu_fp16(gb[k]) * ub[k];
    y[ib].qs[k] = (d == 0.0f) ? int8_t(0) : int8_t(roundf(fh / d));
  }
}

bool gelu_mul_quantize_q8_1_dev(const float* gate, const float* up, BlockQ8_1* yq,
                                int n) {
  if (!available() || !gate || !up || !yq || n <= 0 || (n & 31) != 0) return false;
  SB_NOTE_LAUNCH(gelu_mul_quantize_q8_1_kernel, (n / 32 + 255) / 256, 256, 0, cur_stream());
  gelu_mul_quantize_q8_1_kernel<<<(n / 32 + 255) / 256, 256, 0, cur_stream()>>>(
      gate, up, n, reinterpret_cast<q8_1::block_q8_1*>(yq));
  return cudaGetLastError() == cudaSuccess;
}

// silu×up -> q8_1 fused (T5, SwiGLU): the Silu counterpart of
// gelu_mul_quantize_q8_1 — same fusion, same arithmetic, only the activation
// differs. One warp per 32-element q8_1 block, lane per element (the mapping of
// quantize_q8_1), so a row (ff 4864 = 152 blocks) fills the GPU instead of being
// one thread's 32-iteration serial chain. Bit-identical to silu_dev + mul_dev +
// quantize_q8_1_dev: silu and the product are pure functions of gate/up
// recomputed identically, and amax is a max (order-independent), so the same
// `d` and the same quants come out.
__global__ void silu_mul_quantize_q8_1_kernel(const float* __restrict__ gate,
                                              const float* __restrict__ up,
                                              int nblocks,
                                              q8_1::block_q8_1* __restrict__ y) {
  const int ib = blockIdx.x * q8_1::kQuantizeWarpsPerBlock + (threadIdx.x >> 5);
  if (ib >= nblocks) return;
  const int lane = threadIdx.x & 31;
  const size_t i = (size_t)ib * 32 + lane;
  const float g = gate[i];
  const float v = (g / (1.0f + expf(-g))) * up[i];  // == nn::silu(g) * up
  float amax = fabsf(v);
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float d = amax / 127.0f;
  if (lane == 0) y[ib].d = d;
  y[ib].qs[lane] = (d == 0.0f) ? int8_t(0) : int8_t(roundf(v / d));
}

bool silu_mul_quantize_q8_1_dev(const float* gate, const float* up, BlockQ8_1* yq,
                                int n) {
  if (!available() || !gate || !up || !yq || n <= 0 || (n & 31) != 0) return false;
  const int nblocks = n / 32;
  SB_NOTE_LAUNCH(silu_mul_quantize_q8_1_kernel, q8_1::quantize_q8_1_grid(nblocks), q8_1::kQuantizeBlockThreads, 0, cur_stream());
  silu_mul_quantize_q8_1_kernel<<<q8_1::quantize_q8_1_grid(nblocks),
                                  q8_1::kQuantizeBlockThreads, 0, cur_stream()>>>(
      gate, up, nblocks, reinterpret_cast<q8_1::block_q8_1*>(yq));
  return cudaGetLastError() == cudaSuccess;
}

// The K-quant gemv against a pre-quantized q8_1 activation — no internal
// quantize. W is uploaded+cached exactly as matvec_f32_dev. `bias` is the
// optional [rows] projection bias and `add` the optional [rows] residual
// activation, both folded into the gemv epilogue (see
// gemv_bias_epilogue_supported: an unsupported type with a non-empty bias fails
// rather than silently dropping it). `add` is a device activation (the residual
// of `Add(proj, residual)`) already resident in the device pool; the caller
// guarantees the epilogue supports the weight's type before passing it.
bool gemv_q8_1_dev(const QuantizedTensor& W, const BlockQ8_1* xq, float* y,
                   std::span<const float> bias, const float* add) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (!xq || !y) return false;
  const uint32_t blocks_per_row = uint32_t(cols / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const float* d_bias = nullptr;
  if (!bias.empty()) {
    if (bias.size() != rows || !gemv_bias_epilogue_supported(W.type())) return false;
    d_bias = static_cast<const float*>(upload_bytes(bias.data(), rows * sizeof(float)));
    if (!d_bias) return false;
  }
  if (add && !gemv_bias_epilogue_supported(W.type())) return false;

  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  launch_gemv_q8_1_from_quantized(
      W.type(), wp, reinterpret_cast<const q8_1::block_q8_1*>(xq), y,
      int(blocks_per_row), int(rows), d_bias, add);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess)
    std::fprintf(stderr, "[gemv] rows=%llu bpr=%u: %s\n",
                 (unsigned long long)rows, blocks_per_row,
                 cudaGetErrorString(e));
  return e == cudaSuccess;
}

// The quant types the merged sibling launch dispatches per row (G2). The
// legacy formats were D1's only members; the K-quants read the same q8_1
// activation and differ only in their per-row lane-to-block walk, so a merged
// launch runs each row exactly as that type's standalone kernel does.
static int sibling_slot_type(QuantType t) {
  switch (t) {
    case QuantType::Q5_0: return q8_1::kSlotQ5_0;
    case QuantType::Q8_0: return q8_1::kSlotQ8_0;
    case QuantType::Q3_K: return q8_1::kSlotQ3_K;
    case QuantType::Q4_K: return q8_1::kSlotQ4_K;
    case QuantType::Q5_K: return q8_1::kSlotQ5_K;
    case QuantType::Q6_K: return q8_1::kSlotQ6_K;
  }
  return -1;
}

// Ablation for the K-quant extension: SONICBOOM_GEMV_SIBLING_KQUANT=0 keeps
// the merged launch to the legacy formats, so a mixed group falls back to its
// members' separate launches — the pre-G2 execution of the same plan. Read
// once, like the other backend switches.
static bool gemv_sibling_kquant_enabled() {
  static const bool enabled = [] {
    const char* e = std::getenv("SONICBOOM_GEMV_SIBLING_KQUANT");
    return !(e && std::string_view(e) == "0");
  }();
  return enabled;
}

// Two or three projections over one q8_1 activation, one launch. The decode
// graph's siblings share their activation row for row, so the merged kernel
// lays the members' rows end to end and every row keeps its own type's
// lane-to-block walk and reduction: bit-identical per row, one ramp/tail for
// the group instead of one each. Members may be any of the supported quant
// types (D1 required all-legacy; G2 adds the K-quants, which is what Gemma's
// q3_K/q3_K/q5_K attention trio needs).
bool gemv_q8_1_siblings_dev(std::span<const GemvSibling> siblings,
                            const BlockQ8_1* xq) {
  if (!available() || !xq) return false;
  if (siblings.size() < 2 || siblings.size() > 3) return false;

  q8_1::legacy_slot slots[3] = {};
  uint32_t blocks_per_row = 0;
  for (std::size_t i = 0; i < siblings.size(); ++i) {
    const GemvSibling& s = siblings[i];
    if (!s.weight || !s.y) return false;
    const QuantizedTensor& W = *s.weight;
    if (!W.valid() || W.dims().size() != 2) return false;
    const int slot_type = sibling_slot_type(W.type());
    if (slot_type < 0) return false;
    if (!is_legacy_quant(W.type()) && !gemv_sibling_kquant_enabled())
      return false;
    const uint64_t cols = W.dims()[0];
    const uint64_t rows = W.dims()[1];
    if (rows == 0 || cols % q8_1::QK8_1 != 0) return false;
    const uint32_t bpr = uint32_t(cols / q8_1::QK8_1);
    if (bpr == 0) return false;
    if (blocks_per_row == 0)
      blocks_per_row = bpr;
    else if (blocks_per_row != bpr)
      return false;  // not the same activation
    const float* d_bias = nullptr;
    if (!s.bias.empty()) {
      if (s.bias.size() != rows) return false;
      d_bias = static_cast<const float*>(
          upload_bytes(s.bias.data(), rows * sizeof(float)));
      if (!d_bias) return false;
    }
    const void* d_w = upload_weight(W);
    if (!d_w) return false;
    slots[i] = q8_1::legacy_slot{static_cast<const uint8_t*>(d_w), d_bias, s.y,
                                 int(rows), slot_type};
  }

  const int total_rows = slots[0].rows + slots[1].rows + slots[2].rows;
  const dim3 block(32, 8);
  const dim3 grid(1, (total_rows + 7) / 8);
  SB_NOTE_LAUNCH(q8_1::gemv_legacy_siblings_q8_1, grid, block, 0, cur_stream());
  q8_1::gemv_legacy_siblings_q8_1<<<grid, block, 0, cur_stream()>>>(
      slots[0], slots[1], slots[2],
      reinterpret_cast<const q8_1::block_q8_1*>(xq), int(blocks_per_row));
  return cudaGetLastError() == cudaSuccess;
}

bool gemv_q8_1_batched_dev(const QuantizedTensor& W, const BlockQ8_1* xq,
                           float* y, int n) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (!xq || !y || n <= 0) return false;
  const uint32_t blocks_per_row = uint32_t(cols / W.block_size());  // exact
  if (blocks_per_row == 0) return false;
  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  launch_gemv_q8_1_batched(
      W.type(), static_cast<const uint8_t*>(d_w),
      reinterpret_cast<const q8_1::block_q8_1*>(xq), y, int(blocks_per_row),
      int(rows), n);
  return cudaGetLastError() == cudaSuccess;
}

// q8_1 activation blocks consumed per *weight* block. A K-quant weight block
// spans 256 elements, i.e. 8 of the 32-element q8_1 blocks; a legacy Q5_0/Q8_0
// weight block spans exactly 32 elements, i.e. one. The flat activation layout
// is the same either way (cols/32 blocks per position), but the weight-strided
// indexing differs, so every site that has to know must ask this.
inline int q8_1_blocks_per_weight_block(QuantType qt) {
  return is_legacy_quant(qt) ? 1 : (q8_1::QK_K / q8_1::QK8_1);
}

// Quantize x -> q8_1 (int8 + per-32-block scale) into the device scratch, then
// run the warp-per-row dp4a kernels (each row is written once by lane 0, so no
// zero-fill). This mirrors llama.cpp's mul_mat_vec_q path.
bool launch_gemv_q8_1(QuantType qt, const uint8_t* wp, const float* xp, float* yp,
                      int blocks_per_row, int num_rows) {
  const size_t nblocks = (size_t)blocks_per_row * q8_1_blocks_per_weight_block(qt);
  q8_1::block_q8_1* d_q8 = static_cast<q8_1::block_q8_1*>(
      grow_scratch(g_q8, g_q8_bytes, nblocks * sizeof(q8_1::block_q8_1)));
  if (!d_q8) return false;

  SB_NOTE_LAUNCH(q8_1::quantize_q8_1, q8_1::quantize_q8_1_grid(nblocks), q8_1::kQuantizeBlockThreads, 0, cur_stream());
  q8_1::quantize_q8_1<<<q8_1::quantize_q8_1_grid(nblocks), q8_1::kQuantizeBlockThreads, 0, cur_stream()>>>(xp, d_q8, int(nblocks));

  launch_gemv_q8_1_from_quantized(qt, wp, d_q8, yp, blocks_per_row, num_rows);
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
  if (q8_1_enabled())
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
  if (q8_1_enabled())
    return launch_gemv_q8_1(W.type(), wp, x, y, int(blocks_per_row), num_rows);
  launch_gemv(W.type(), wp, x, y, int(blocks_per_row), num_rows);
  return cudaGetLastError() == cudaSuccess;
}

// Batched (prefill) matmul: Y = W @ X for X [cols, n] (n innermost) and Y
// [rows, n] (n innermost). The q8_1 path transposes X to [n, cols], quantizes
// each position's activation to q8_1 (contiguous blocks), then runs one batched
// warp-per-row kernel that reuses each weight block across a tile of positions —
// so the weight is read once per batch instead of once per column. The per-column
// arithmetic is bit-identical to matvec_f32_dev (same vec_dot + same
// quantize_q8_1), so this matches the CPU matmul_f32 to the same fp32 tolerance
// and keeps prefill/decode quantization consistent. The fp32 fallback
// (SONICBOOM_GEMV_F32=1) keeps the n× gather/gemv/scatter loop. n == 1 falls
// back to the single-column path.
// cuBLAS fp16 batched matmul (C.1): Y = W @ X for X [cols, n] (n innermost),
// Y [rows, n]. Dequantize W's packed blocks to fp16 (row-major [rows, cols])
// into a reusable scratch buffer, convert X to fp16, then one cublasGemmEx
// (fp16 in, fp32 out, CUBLAS_COMPUTE_32F → tensor cores on SM_86). The
// dequantize/convert run once and are amortized over all n columns, and the
// tensor-core GEMM is far cheaper than the dp4a per-column gemv it replaces.
//
// cuBLAS is column-major. With position-major X == row-major [n, cols] (==
// column-major [cols, n]) and W fp16 == row-major [rows, cols] (== column-major
// [cols, rows]), the GEMM writes Y [n, rows] directly with A = W^T (op=T,
// lda=cols), B = X (op=N, ldb=cols), C = Y (m=rows, n=n, ldc=rows):
//   Y[c, r] = sum_k W[r,k] X[c,k].
bool matmul_f16_cublas_dev(const QuantizedTensor& W, const float* X, float* Y,
                           int n) {
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];

  // 1. Dequantize W -> fp16 into reusable scratch (row-major [rows, cols]).
  __half* wf = static_cast<__half*>(
      grow_scratch(g_wf16, g_wf16_bytes, rows * cols * sizeof(__half)));
  if (!wf) return false;
  const uint8_t* wp = static_cast<const uint8_t*>(upload_weight(W));
  if (!wp) return false;
  const int64_t numel = int64_t(rows * cols);
  const dim3 dgrid((numel + 255) / 256);
  switch (W.type()) {
    case QuantType::Q3_K:
      SB_NOTE_LAUNCH(dequant_q3_K_f16, dgrid, 256, 0, cur_stream());
      dequant_q3_K_f16<<<dgrid, 256, 0, cur_stream()>>>(wp, wf, numel);
      break;
    case QuantType::Q4_K:
      SB_NOTE_LAUNCH(dequant_q4_K_f16, dgrid, 256, 0, cur_stream());
      dequant_q4_K_f16<<<dgrid, 256, 0, cur_stream()>>>(wp, wf, numel);
      break;
    case QuantType::Q5_K:
      SB_NOTE_LAUNCH(dequant_q5_K_f16, dgrid, 256, 0, cur_stream());
      dequant_q5_K_f16<<<dgrid, 256, 0, cur_stream()>>>(wp, wf, numel);
      break;
    case QuantType::Q6_K:
      SB_NOTE_LAUNCH(dequant_q6_K_f16, dgrid, 256, 0, cur_stream());
      dequant_q6_K_f16<<<dgrid, 256, 0, cur_stream()>>>(wp, wf, numel);
      break;
    case QuantType::Q5_0:
      SB_NOTE_LAUNCH(dequant_q5_0_f16, dgrid, 256, 0, cur_stream());
      dequant_q5_0_f16<<<dgrid, 256, 0, cur_stream()>>>(wp, wf, numel);
      break;
    case QuantType::Q8_0:
      SB_NOTE_LAUNCH(dequant_q8_0_f16, dgrid, 256, 0, cur_stream());
      dequant_q8_0_f16<<<dgrid, 256, 0, cur_stream()>>>(wp, wf, numel);
      break;
    default:
      return false;
  }

  // 2. Convert X f32 -> fp16 (position-major [n, cols], flat element count).
  __half* xf = static_cast<__half*>(
      grow_scratch(g_xf16, g_xf16_bytes, cols * (size_t)n * sizeof(__half)));
  if (!xf) return false;
  const int64_t xn = int64_t(cols) * n;
  SB_NOTE_LAUNCH(cast_f32_f16, (xn + 255) / 256, 256, 0, cur_stream());
  cast_f32_f16<<<(xn + 255) / 256, 256, 0, cur_stream()>>>(X, xf, xn);

  // 3. cuBLAS tensor-core GEMM on the current stream: C = W^T @ X.
  cublasHandle_t h = cublas_handle();
  if (!h) return false;
  cublasSetStream(h, cur_stream());
  const float alpha = 1.0f, beta = 0.0f;
  const cublasStatus_t st = cublasGemmEx(
      h, CUBLAS_OP_T, CUBLAS_OP_N, int(rows), n, int(cols), &alpha, wf,
      CUDA_R_16F, int(cols), xf, CUDA_R_16F, int(cols), &beta, Y, CUDA_R_32F,
      int(rows), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  return st == CUBLAS_STATUS_SUCCESS && cudaGetLastError() == cudaSuccess;
}

// MMQ int8 tensor-core batched matmul (C.2). Dequantizes the weight tile into
// shared memory as int8 and multiplies against a block_q8_1_mmq activation via
// m16n8k16/m16n8k32 tensor cores. X is [n, cols] position-major, which is
// already the layout quantize_activation needs (each position's 128-element
// q8_1 blocks contiguous), and mul_mat_q natively writes dst [n, rows] — so no
// transpose on either side of the kernel (C.3a). Q3_K uses a float d4 activation
// scale layout (D4); Q4_K/Q5_K use half2 ds4 (DS4), matching the weight's
// vec_dot path. Q6_K uses D4 (float activation scales); its signed per-16-value
// weight scales remain separate from the block's fp16 d, matching ggml's
// Q6_K MMQ tile layout. The legacy 32-element formats Q5_0/Q8_0 also use the
// D4 layout (ggml's MMQ_Q8_1_DS_LAYOUT_D4) with their own tile loaders.
bool matmul_mmq_dev(const QuantizedTensor& W, const float* X, float* Y, int n) {
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (cols % mmq::QK8_1_MMQ != 0) return false;  // kernel precondition
  const int ic = int(cols), ir = int(rows);
  const int blocks_per_row = int(cols / W.block_size());  // exact (valid())

  // TODO(q6k-mmq): incoherent output on Gemma Q4_K_M, disabled until fixed.
  if (W.type() == QuantType::Q6_K)
    return matmul_f16_cublas_dev(W, X, Y, n);

  const uint8_t* wp = static_cast<const uint8_t*>(upload_weight(W));
  if (!wp) return false;

  // 1. Quantize X [n, cols] -> block_q8_1_mmq [K/128][n] (+ J_max tail guard).
  mmq::block_q8_1_mmq* y_q8 = static_cast<mmq::block_q8_1_mmq*>(
      grow_scratch(g_mmq_q8, g_mmq_q8_bytes,
                   mmq::mmq_q8_blocks(ic, n) * sizeof(mmq::block_q8_1_mmq)));
  if (!y_q8) return false;

  // 2. Zero the pad plane(s) a K that is not a multiple of MMQ_ITER_K (256)
  // leaves: the last K-iteration reads a plane the quantizer never wrote. Runs
  // before the quantize but touches none of its blocks. No-op when K % 256 == 0.
  mmq::zero_pad_planes(y_q8, ic, n, cur_stream());

  bool ok = false;
  switch (W.type()) {
    case QuantType::Q3_K:
      mmq::quantize_activation<mmq::D4>(X, y_q8, ic, n, cur_stream());
      ok = mul_mat_q_tuned<mmq::MMQ_Q3_K>(
          reinterpret_cast<const char*>(wp), reinterpret_cast<const int*>(y_q8),
          Y, ir, n, blocks_per_row, MmqDecomp::Auto, cur_stream());
      break;
    case QuantType::Q4_K:
      mmq::quantize_activation<mmq::DS4>(X, y_q8, ic, n, cur_stream());
      ok = mul_mat_q_tuned<mmq::MMQ_Q4_K>(
          reinterpret_cast<const char*>(wp), reinterpret_cast<const int*>(y_q8),
          Y, ir, n, blocks_per_row, MmqDecomp::Auto, cur_stream());
      break;
    case QuantType::Q5_K:
      mmq::quantize_activation<mmq::DS4>(X, y_q8, ic, n, cur_stream());
      ok = mul_mat_q_tuned<mmq::MMQ_Q5_K>(
          reinterpret_cast<const char*>(wp), reinterpret_cast<const int*>(y_q8),
          Y, ir, n, blocks_per_row, MmqDecomp::Auto, cur_stream());
      break;
    // The legacy 32-element formats use the D4 float activation scale layout
    // (ggml's MMQ_Q8_1_DS_LAYOUT_D4), same as Q3_K.
    case QuantType::Q5_0:
      mmq::quantize_activation<mmq::D4>(X, y_q8, ic, n, cur_stream());
      ok = mul_mat_q_tuned<mmq::MMQ_Q5_0>(
          reinterpret_cast<const char*>(wp), reinterpret_cast<const int*>(y_q8),
          Y, ir, n, blocks_per_row, MmqDecomp::Auto, cur_stream());
      break;
    case QuantType::Q8_0:
      mmq::quantize_activation<mmq::D4>(X, y_q8, ic, n, cur_stream());
      ok = mul_mat_q_tuned<mmq::MMQ_Q8_0>(
          reinterpret_cast<const char*>(wp), reinterpret_cast<const int*>(y_q8),
          Y, ir, n, blocks_per_row, MmqDecomp::Auto, cur_stream());
      break;
    default:
      return false;
  }
  if (!ok) return false;
  return cudaGetLastError() == cudaSuccess;
}

// MMQ matmul consuming a *pre-quantized* block_q8_1_mmq activation (C.4): no
// internal quantize. W is the Q*_K weight [cols, rows]; y_q8 is [cols/128][n]
// (index blk*n + token); Y is [n, rows] position-major (the tensor-core matmul's
// native output — no transpose, matching matmul_mmq_dev's C.3a path).
bool matmul_mmq_q8_dev(const QuantizedTensor& W, const BlockQ8_1Mmq* y_q8,
                       float* Y, int n, MmqDecomp decomp) {
  if (!available() || !W.valid() || W.dims().size() != 2 || !y_q8 || !Y || n <= 0)
    return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (cols % mmq::QK8_1_MMQ != 0) return false;  // kernel precondition
  const int ir = int(rows);
  const int blocks_per_row = int(cols / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  if (W.type() == QuantType::Q6_K) {
    // Keep the Q6_K MMQ weight kernel disabled; use D4 activation + fp16 cuBLAS.
    float* X = static_cast<float*>(grow_scratch(
        g_mmq_q6_x_f32, g_mmq_q6_x_f32_bytes, cols * (size_t)n * sizeof(float)));
    if (!X) return false;
    const int64_t nx = int64_t(cols) * n;
    SB_NOTE_LAUNCH(dequant_mmq_q8_1_d4, (nx + 255) / 256, 256, 0, cur_stream());
    dequant_mmq_q8_1_d4<<<(nx + 255) / 256, 256, 0, cur_stream()>>>(
        reinterpret_cast<const mmq::block_q8_1_mmq*>(y_q8), X, int(cols), n);
    if (cudaGetLastError() != cudaSuccess) return false;
    return matmul_f16_cublas_dev(W, X, Y, n);
  }

  const uint8_t* wp = static_cast<const uint8_t*>(upload_weight(W));
  if (!wp) return false;

  // A K that is not a multiple of MMQ_ITER_K (256) leaves an odd plane count; the
  // last K-iteration reads one plane past what the fused emitter wrote, so zero
  // it (no-op for K % 256 == 0, i.e. every K-quant weight).
  mmq::zero_pad_planes(reinterpret_cast<const mmq::block_q8_1_mmq*>(y_q8),
                       int(cols), n, cur_stream());

  const int* yq = reinterpret_cast<const int*>(y_q8);
  bool ok = false;
  switch (W.type()) {
    case QuantType::Q3_K:
      ok = mul_mat_q_tuned<mmq::MMQ_Q3_K>(
          reinterpret_cast<const char*>(wp), yq, Y, ir, n, blocks_per_row, decomp, cur_stream());
      break;
    case QuantType::Q4_K:
      ok = mul_mat_q_tuned<mmq::MMQ_Q4_K>(
          reinterpret_cast<const char*>(wp), yq, Y, ir, n, blocks_per_row, decomp, cur_stream());
      break;
    case QuantType::Q5_K:
      ok = mul_mat_q_tuned<mmq::MMQ_Q5_K>(
          reinterpret_cast<const char*>(wp), yq, Y, ir, n, blocks_per_row, decomp, cur_stream());
      break;
    case QuantType::Q6_K:
      ok = mul_mat_q_tuned<mmq::MMQ_Q6_K>(
          reinterpret_cast<const char*>(wp), yq, Y, ir, n, blocks_per_row, decomp, cur_stream());
      break;
    case QuantType::Q5_0:
      ok = mul_mat_q_tuned<mmq::MMQ_Q5_0>(
          reinterpret_cast<const char*>(wp), yq, Y, ir, n, blocks_per_row, decomp, cur_stream());
      break;
    case QuantType::Q8_0:
      ok = mul_mat_q_tuned<mmq::MMQ_Q8_0>(
          reinterpret_cast<const char*>(wp), yq, Y, ir, n, blocks_per_row, decomp, cur_stream());
      break;
    default:
      return false;
  }
  if (!ok) return false;
  return cudaGetLastError() == cudaSuccess;
}

// MMQ2: the generated kernel's (tile_m, tile_n) for one shape class. The first
// call for a class under SONICBOOM_AUTOTUNE=1 times the candidate grid on the
// live buffers; a candidate is accepted only when its output matches the
// smallest config (16x16) bit for bit, and the winner is persisted per SM.
// Skipped inside CUDA-graph capture, where the host must not synchronize and a
// JIT module load allocates: the class is tuned by the first uncaptured call.
static compiler::MmqTcShape mmq_tc_shape_for(const mlir_tune::TileConfig& cfg,
                                             int ds) {
  compiler::MmqTcShape s;
  s.bm = cfg.tile_m;
  s.bn = cfg.tile_n;
  s.ds = ds;
  return s;
}

static compiler::MmqTcShape resolve_mmq_tc_shape(
    const compiler::MmqTcShape& base, const void* xq, const void* wq,
    const void* ws, float* y, int m_rows, int k, int n_tokens) {
  const mlir_tune::MmqTcChoice choice =
      mlir_tune::mmq_tc_choice(n_tokens, m_rows, k);
  if (choice.resolved()) {
    const compiler::MmqTcShape shape = mmq_tc_shape_for(choice.cfg, base.ds);
    return compiler::mmq_tc_shape_supported(shape) ? shape : base;
  }

  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(cur_stream(), &capture) != cudaSuccess ||
      capture != cudaStreamCaptureStatusNone)
    return base;

  // The comparison window is a prefix of the output, capped at 64 MiB so the
  // 151936-row logits projection does not turn every timed burst into a 300 MB
  // device-to-host copy. The tiles partition both dimensions, so the prefix
  // sees the same per-block code on every candidate; the token-tail guards are
  // covered by the small-M classes, where the whole output fits.
  constexpr size_t kMaxCompareBytes = size_t(64) << 20;
  const size_t out_bytes = size_t(m_rows) * size_t(n_tokens) * sizeof(float);
  const size_t cmp_bytes = std::min(out_bytes, kMaxCompareBytes);
  std::vector<float> host_out(cmp_bytes / sizeof(float));

  bool launch_ok = false;
  // Each "launch" is a short burst of the same matmul: the runner times one
  // launch per host wait, and a wait costs about as much as the smaller
  // projections, so the burst amortizes that floor (the gemv width probe uses
  // the same shape).
  constexpr int kProbeReps = 16;
  auto make_launch = [&](const compiler::MmqTcShape& shape) {
    return mlir_tune::AutotuneRunner::Launch([&, shape] {
      bool ok = true;
      for (int i = 0; i < kProbeReps && ok; ++i)
        ok = planner::glue::launch_mmq_tc(shape, xq, wq, ws, y, m_rows, k,
                                          n_tokens);
      launch_ok = ok;
    });
  };

  const std::span<const mlir_tune::TileConfig> grid =
      mlir_tune::mmq_tc_candidates();
  std::vector<mlir_tune::TileConfig> candidates;
  candidates.push_back(grid.front());  // the reference is never pruned
  for (size_t i = 1; i < grid.size(); ++i) {
    // Prune a token tile more than 4x the token count: its tail waste cannot
    // win, and launching it only costs probe time.
    if (grid[i].tile_n <= 4 * n_tokens) candidates.push_back(grid[i]);
  }

  mlir_tune::AutotuneRunner::Problem p;
  p.candidates = std::move(candidates);
  p.compile = [&](const mlir_tune::TileConfig& cfg)
      -> std::optional<mlir_tune::AutotuneRunner::Launch> {
    // Skip a config the generator does not cover (the runner then times the
    // rest; a covered grid always leaves the reference in place).
    const compiler::MmqTcShape shape = mmq_tc_shape_for(cfg, base.ds);
    if (!compiler::mmq_tc_shape_supported(shape)) return std::nullopt;
    return make_launch(shape);
  };
  p.output = host_out.data();
  p.output_bytes = cmp_bytes;
  p.sync = [&] {
    if (!launch_ok) {
      // A failed candidate must not look like a bit-identical fast one: poison
      // the window and clear the sticky error so the caller's node still runs.
      std::memset(host_out.data(), 0xFF, cmp_bytes);
      cudaGetLastError();
      return;
    }
    // The copy is enqueued behind the candidate on the same stream, so a burst
    // pays one host wait; the copy is identical for every candidate and
    // cancels in the ranking.
    cudaError_t e = cudaMemcpyAsync(host_out.data(), y, cmp_bytes,
                                    cudaMemcpyDeviceToHost, cur_stream());
    if (e == cudaSuccess) e = cudaStreamSynchronize(cur_stream());
    if (e != cudaSuccess) {
      std::fprintf(stderr, "[autotune] matmul_q8_tc probe: %s\n",
                   cudaGetErrorString(e));
      std::memset(host_out.data(), 0xFF, cmp_bytes);
      cudaGetLastError();
    }
  };
  // Timing waits on the stream only; the bit-identity copy belongs to the
  // check runs.
  p.timing_sync = [&] { (void)cudaStreamSynchronize(cur_stream()); };
  // 8 timed bursts of 16 launches, min: the tile differences are small and the
  // choice is persisted, so the measurement must be stable.
  p.iters = 8;
  const auto best = mlir_tune::AutotuneRunner::run(p);
  if (!best) return base;

  const bool stored =
      mlir_tune::mmq_tc_choice_store(n_tokens, m_rows, k, *best);
  std::fprintf(stderr, "[autotune] matmul_q8_tc %s: tile_m=%d tile_n=%d%s\n",
               mlir_tune::mmq_tc_shape_class(n_tokens, m_rows, k).c_str(),
               best->tile_m, best->tile_n, stored ? "" : " [not persisted]");
  const compiler::MmqTcShape shape = mmq_tc_shape_for(*best, base.ds);
  return compiler::mmq_tc_shape_supported(shape) ? shape : base;
}

// MMQ1: the same operands as matmul_mmq_q8_dev, multiplied by the generated
// int8 tensor-core kernel over the per-weight q8_1 copy (upload_tc_weight).
// The activation is the fused quantizers' block_q8_1_mmq — the kernel reads the
// same layout the ggml mmq kernel does — so the two paths differ only in the
// weight representation. The tile class comes from the MMQ2 autotuner.
bool matmul_mmq_tc_dev(const QuantizedTensor& W, const BlockQ8_1Mmq* yq,
                       float* Y, int n, int ds) {
  if (!available() || !W.valid() || W.dims().size() != 2 || !yq || !Y || n <= 0)
    return false;
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  if (cols == 0 || rows == 0 || cols % 128 != 0) return false;
  if (cols > INT32_MAX || rows > INT32_MAX) return false;
  if (ds < 0 || ds > 2) return false;

  const TcWeight tw = upload_tc_weight(W, int(rows), int(cols));
  if (!tw.q) return false;

  compiler::MmqTcShape shape;
  shape.ds = ds;
  if (!compiler::mmq_tc_shape_supported(shape)) return false;
  shape = resolve_mmq_tc_shape(shape, yq, tw.q, tw.s, Y, int(rows), int(cols),
                               n);
  return planner::glue::launch_mmq_tc(shape, yq, tw.q, tw.s, Y, int(rows),
                                      int(cols), n);
}

// --- MMQ batched/fused prefill quantize (C.4, position-major) ---------------

// Quantize one 128-element block (32 lanes) to block_q8_1_mmq, bit-identical to
// ggml_derived/mmq.cuh's quantize_mmq_q8_1: each lane owns a 4-float (float4)
// slot, amax is an fmaxf butterfly and the DS4 `sum` is the exact
// (s0+s4)+(s2+s6)+((s1+s5)+(s3+s7)) shuffle-xor tree. The fused kernels below
// route one 128-element block per warp so the lane mapping matches the unfused
// quantize, making the fused and unfused mmq paths write identical bytes (the
// prefix-cache full-vs-reuse test depends on this). The caller must have all 32
// lanes of the warp active (full shuffle mask).
template <int ds>  // mmq::D4 (Q3_K/Q6_K) or mmq::DS4 (Q4_K/Q5_K)
__device__ __forceinline__ void quantize_mmq_q8_1_block(
    const float (&f)[4], int blk, int token, int n, int lane,
    mmq::block_q8_1_mmq* __restrict__ y) {
  float amax = fabsf(f[0]);
  amax = fmaxf(amax, fabsf(f[1]));
  amax = fmaxf(amax, fabsf(f[2]));
  amax = fmaxf(amax, fabsf(f[3]));
#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, offset, 32));

  float sum = 0.0f;
  if constexpr (ds == mmq::DS4) {
    sum = f[0] + f[1] + f[2] + f[3];
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
      sum += __shfl_xor_sync(0xffffffffu, sum, offset, 32);
  }

  char4 q;
  float d;
  if constexpr (ds == mmq::D4_Q8_1) {
    d = amax / 127.0f;
    q.x = roundf(f[0] / d);
    q.y = roundf(f[1] / d);
    q.z = roundf(f[2] / d);
    q.w = roundf(f[3] / d);
  } else {
    const float d_inv = 127.0f / amax;
    d = 1.0f / d_inv;
    q.x = roundf(f[0] * d_inv);
    q.y = roundf(f[1] * d_inv);
    q.z = roundf(f[2] * d_inv);
    q.w = roundf(f[3] * d_inv);
  }

  mmq::block_q8_1_mmq& yb = y[(size_t)blk * n + token];
  reinterpret_cast<char4*>(yb.qs)[lane] = q;

  if (lane % 8 == 0) {
    const int sub = lane >> 3;
    if constexpr (ds == mmq::DS4) yb.ds4[sub] = make_half2(d, sum);
    else                          yb.d4[sub]  = d;
  }
}

// rms_norm -> block_q8_1_mmq fused (C.4, position-major), split into two
// kernels so both phases run in the geometry their standalone counterparts use,
// at any `dim`. The single-kernel form used one 1024-thread block per token and
// one 128-element block per warp for the quantize, which left only dim/128 of
// its 32 warps with quantize work (7 of 32 at Qwen's d_model 896) and measured
// 45 GB/s against the standalone quantize_mmq_q8_1's 226 GB/s; its 1024-thread
// double-precision reduction also cost ~4x the 256-thread one below (fp64
// shuffles/shared writes per row), which is what made the whole op slow at
// small dim.
//
// Phase A computes the per-row `inv` with nn::cuda::rms_norm_heads_dev's
// reduction verbatim — that is the norm the unfused prefill path runs for a
// batched RmsNorm node (the plan's RmsNorm+QuantizedMatmul pair), so the fused
// and unfused emitters write the same bytes; phase B carries the multiply and
// the quantize in the standalone quantizer's geometry. Both are then
// bandwidth-bound, and the byte-identity the prefix-cache reuse test needs is
// between these two emitters, not between either of them and rms_norm_dev.

// Phase A: the per-row `inv`. One 256-thread block per row, strided double
// partials + warp shuffle + a warp-0 shared reduce — same tree and same fp32
// rounding point as rms_heads_fused_kernel, so `inv` matches that kernel's.
template <typename R>
__global__ void __launch_bounds__(256) rms_norm_inv_kernel(
    const float* __restrict__ x, int dim, float eps,
    float* __restrict__ inv_out) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const float* xrow = x + (size_t)token * dim;

  __shared__ typename R::part_t warp_part[32];
  const typename R::acc_t t =
      sonicboom::nn::cuda::block_sum_sq<R>(xrow, dim, warp_part);
  if (tid == 0)
    inv_out[token] = 1.0f / sqrtf(R::mean_to_float(t, dim) + eps);
}

// Phase B: apply `inv` (and the optional norm weight) and quantize. 128 threads
// per block and one 128-element block per warp — mmq::quantize_activation's
// geometry (grid (n, dim/512)), with the same lane-owned float4, so the blocks
// written are the same bytes quantize_mmq_q8_1_block documents as bit-identical
// to quantize_mmq_q8_1. Warps with no block left in the tail plane exit whole
// (never a lane), keeping the amax/sum butterfly's shuffle masks full.
template <int ds>  // mmq::D4 (Q3_K/Q5_0/Q8_0) or mmq::DS4 (Q4_K/Q5_K)
__global__ void __launch_bounds__(128) scale_quantize_mmq_q8_1_kernel(
    const float* __restrict__ x, const float* __restrict__ inv,
    const float* __restrict__ weight, int dim, int n,
    mmq::block_q8_1_mmq* __restrict__ y) {
  const int lane = threadIdx.x & 31;
  const int blk = (int)blockIdx.y * 4 + (threadIdx.x >> 5);
  if (blk >= dim / 128) return;
  const int token = blockIdx.x;
  const int base = blk * 128 + lane * 4;
  const float* xrow = x + (size_t)token * dim;
  const float iv = inv[token];
  float f[4];
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const float v = xrow[base + k];
    f[k] = weight ? v * iv * weight[base + k] : v * iv;
  }
  quantize_mmq_q8_1_block<ds>(f, blk, token, n, lane, y);
}

bool rms_norm_quantize_mmq_q8_1_dev(std::span<const float> weight, const float* x,
                                    float eps, BlockQ8_1Mmq* yq, int dim, int n,
                                    int ds) {
  if (!available() || !x || !yq || dim <= 0 || n <= 0 || (dim & 127) != 0) return false;
  if (!weight.empty() && weight.size() < (size_t)dim) return false;
  const float* d_w = nullptr;
  if (!weight.empty()) {
    d_w = static_cast<const float*>(upload_bytes(weight.data(), dim * sizeof(float)));
    if (!d_w) return false;
  }
  float* inv = static_cast<float*>(
      grow_scratch(g_rms_inv, g_rms_inv_bytes, (size_t)n * sizeof(float)));
  if (!inv) return false;
  auto* y = reinterpret_cast<mmq::block_q8_1_mmq*>(yq);
  SB_NOTE_LAUNCH(rms_norm_inv_kernel, n, 256, 0, cur_stream());
  cudaStream_t st = cur_stream();
  switch (sonicboom::planner::device_reduce_mode()) {
    case sonicboom::compiler::ReduceMode::Df32:
      rms_norm_inv_kernel<sonicboom::nn::cuda::SumDf32>
          <<<n, 256, 0, st>>>(x, dim, eps, inv);
      break;
    case sonicboom::compiler::ReduceMode::F32:
      rms_norm_inv_kernel<sonicboom::nn::cuda::SumF32>
          <<<n, 256, 0, st>>>(x, dim, eps, inv);
      break;
    default:
      rms_norm_inv_kernel<sonicboom::nn::cuda::SumF64>
          <<<n, 256, 0, st>>>(x, dim, eps, inv);
      break;
  }
  const dim3 grid(n, (dim / 128 + 3) / 4);  // one 128-element block per warp
  if (ds == mmq::DS4) {
    SB_NOTE_LAUNCH_TP("scale_quantize_mmq_q8_1_kernel", "ds=ds4", grid, 128, 0, cur_stream());
    scale_quantize_mmq_q8_1_kernel<mmq::DS4><<<grid, 128, 0, cur_stream()>>>(
        x, inv, d_w, dim, n, y);
  } else if (ds == mmq::D4_Q8_1) {
    SB_NOTE_LAUNCH_TP("scale_quantize_mmq_q8_1_kernel", "ds=d4_q8_1", grid, 128, 0, cur_stream());
    scale_quantize_mmq_q8_1_kernel<mmq::D4_Q8_1><<<grid, 128, 0, cur_stream()>>>(
        x, inv, d_w, dim, n, y);
  } else {
    SB_NOTE_LAUNCH_TP("scale_quantize_mmq_q8_1_kernel", "ds=d4", grid, 128, 0, cur_stream());
    scale_quantize_mmq_q8_1_kernel<mmq::D4><<<grid, 128, 0, cur_stream()>>>(
        x, inv, d_w, dim, n, y);
  }
  return cudaGetLastError() == cudaSuccess;
}

// gelu×up -> block_q8_1_mmq fused (C.4, position-major): one 128-element block per
// warp, each lane computing gelu_fp16(gate) * up over its float4 and quantizing via
// quantize_mmq_q8_1_block (bit-identical to mmq::quantize_activation) — the f32 gelu
// and product intermediates are never materialized.
template <int ds>
__global__ void gelu_mul_quantize_mmq_q8_1_kernel(
    const float* __restrict__ gate, const float* __restrict__ up, int dim, int n,
    mmq::block_q8_1_mmq* __restrict__ y) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  const float* grow = gate + (size_t)token * dim;
  const float* urow = up + (size_t)token * dim;
  const int nblocks = dim / 128;
  for (int blk = warp; blk < nblocks; blk += nwarps) {
    const int base = blk * 128 + lane * 4;
    float f[4];
#pragma unroll
    for (int k = 0; k < 4; ++k)
      f[k] = gelu_fp16(grow[base + k]) * urow[base + k];
    quantize_mmq_q8_1_block<ds>(f, blk, token, n, lane, y);
  }
}

bool gelu_mul_quantize_mmq_q8_1_dev(const float* gate, const float* up,
                                    BlockQ8_1Mmq* yq, int dim, int n, int ds) {
  if (!available() || !gate || !up || !yq || dim <= 0 || n <= 0 || (dim & 127) != 0)
    return false;
  auto* y = reinterpret_cast<mmq::block_q8_1_mmq*>(yq);
  if (ds == mmq::DS4) {
    SB_NOTE_LAUNCH_TP("gelu_mul_quantize_mmq_q8_1_kernel", "ds=ds4", n, 256, 0, cur_stream());
    gelu_mul_quantize_mmq_q8_1_kernel<mmq::DS4><<<n, 256, 0, cur_stream()>>>(
        gate, up, dim, n, y);
  } else if (ds == mmq::D4_Q8_1) {
    SB_NOTE_LAUNCH_TP("gelu_mul_quantize_mmq_q8_1_kernel", "ds=d4_q8_1", n, 256, 0, cur_stream());
    gelu_mul_quantize_mmq_q8_1_kernel<mmq::D4_Q8_1><<<n, 256, 0, cur_stream()>>>(
        gate, up, dim, n, y);
  } else {
    SB_NOTE_LAUNCH_TP("gelu_mul_quantize_mmq_q8_1_kernel", "ds=d4", n, 256, 0, cur_stream());
    gelu_mul_quantize_mmq_q8_1_kernel<mmq::D4><<<n, 256, 0, cur_stream()>>>(
        gate, up, dim, n, y);
  }
  return cudaGetLastError() == cudaSuccess;
}

// silu×up -> block_q8_1_mmq fused (C.4, position-major, SwiGLU): the Silu
// counterpart of gelu_mul_quantize_mmq_q8_1 — same fusion, same block mapping,
// same quantizer, only the activation differs (silu(g) = g/(1+expf(-g)), the
// expression nn::cuda::silu uses, so the fused blocks are bit-identical to the
// unfused silu + mul + the matmul's internal quantize).
template <int ds>
__global__ void silu_mul_quantize_mmq_q8_1_kernel(
    const float* __restrict__ gate, const float* __restrict__ up, int dim, int n,
    mmq::block_q8_1_mmq* __restrict__ y) {
  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int nwarps = (blockDim.x + 31) >> 5;
  const float* grow = gate + (size_t)token * dim;
  const float* urow = up + (size_t)token * dim;
  const int nblocks = dim / 128;
  for (int blk = warp; blk < nblocks; blk += nwarps) {
    const int base = blk * 128 + lane * 4;
    float f[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const float g = grow[base + k];
      f[k] = (g / (1.0f + expf(-g))) * urow[base + k];
    }
    quantize_mmq_q8_1_block<ds>(f, blk, token, n, lane, y);
  }
}

bool silu_mul_quantize_mmq_q8_1_dev(const float* gate, const float* up,
                                    BlockQ8_1Mmq* yq, int dim, int n, int ds) {
  if (!available() || !gate || !up || !yq || dim <= 0 || n <= 0 || (dim & 127) != 0)
    return false;
  auto* y = reinterpret_cast<mmq::block_q8_1_mmq*>(yq);
  if (ds == mmq::DS4) {
    SB_NOTE_LAUNCH_TP("silu_mul_quantize_mmq_q8_1_kernel", "ds=ds4", n, 256, 0, cur_stream());
    silu_mul_quantize_mmq_q8_1_kernel<mmq::DS4><<<n, 256, 0, cur_stream()>>>(
        gate, up, dim, n, y);
  } else if (ds == mmq::D4_Q8_1) {
    SB_NOTE_LAUNCH_TP("silu_mul_quantize_mmq_q8_1_kernel", "ds=d4_q8_1", n, 256, 0, cur_stream());
    silu_mul_quantize_mmq_q8_1_kernel<mmq::D4_Q8_1><<<n, 256, 0, cur_stream()>>>(
        gate, up, dim, n, y);
  } else {
    SB_NOTE_LAUNCH_TP("silu_mul_quantize_mmq_q8_1_kernel", "ds=d4", n, 256, 0, cur_stream());
    silu_mul_quantize_mmq_q8_1_kernel<mmq::D4><<<n, 256, 0, cur_stream()>>>(
        gate, up, dim, n, y);
  }
  return cudaGetLastError() == cudaSuccess;
}

bool matmul_f32_dev(const QuantizedTensor& W, const float* X, float* Y, int n) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  if (n <= 0 || !X || !Y) return false;
  if (n == 1) return matvec_f32_dev(W, X, Y);
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  const uint32_t blocks_per_row = uint32_t(cols / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  // MMQ int8 tensor-core path (C.2) — measured faster than cuBLAS fp16 (C.1)
  // at every n tested (wall tok/s, mmq vs cuBLAS: 64→193/127, 128→326/224,
  // 256→434/336, 512→458/396, 2048→324/286), so it is selected for all
  // n >= mmq_min_n(); cuBLAS stays reachable via SONICBOOM_MMQ=0. Only when
  // q8_1 is active — the fp32 fallback is the CPU-matching differential path.
  // The MMQ kernel tiles for the 256-element K-quant super-block (Q3_K/Q4_K/Q5_K)
  // and for the legacy 32-element formats (Q5_0/Q8_0, 8 blocks per 256-value
  // iteration). Q6_K (TODO(q6k-mmq)) skips it and takes the cuBLAS fp16 or dp4a
  // batched path below — matmul_mmq_dev would reject it anyway, and returning its
  // false here would fail the whole call instead of falling through.
  const bool mmq_type = W.type() != QuantType::Q6_K;
  if (q8_1_enabled() && mmq_enabled() && n >= mmq_min_n() && mmq_type)
    return matmul_mmq_dev(W, X, Y, n);

  // TODO(q6k-mmq): incoherent output on Gemma Q4_K_M, disabled until fixed.
  if (W.type() == QuantType::Q6_K && q8_1_enabled() && mmq_enabled() &&
      n >= mmq_min_n() && cublas_enabled() && n >= cublas_min_n())
    return matmul_f16_cublas_dev(W, X, Y, n);

  // cuBLAS fp16 path for large batches (C.1). Only when q8_1 is active — the
  // fp32 fallback is the CPU-matching differential path and must stay fp32.
  if (q8_1_enabled() && cublas_enabled() && n >= cublas_min_n())
    return matmul_f16_cublas_dev(W, X, Y, n);

  const void* d_w = upload_weight(W);
  if (!d_w) return false;

  const uint8_t* wp = static_cast<const uint8_t*>(d_w);
  const int ic = int(cols), ir = int(rows);

  if (q8_1_enabled()) {
    // X is [n, cols] position-major, so each position's 32-element q8_1 blocks
    // are already contiguous — no transpose before quantization (C.3a).
    const size_t nblocks =
        (size_t)blocks_per_row * q8_1_blocks_per_weight_block(W.type());
    q8_1::block_q8_1* xq = static_cast<q8_1::block_q8_1*>(grow_scratch(
        g_q8, g_q8_bytes, nblocks * (size_t)n * sizeof(q8_1::block_q8_1)));
    if (!xq) return false;
    SB_NOTE_LAUNCH(q8_1::quantize_q8_1, q8_1::quantize_q8_1_grid(int(nblocks * n)), q8_1::kQuantizeBlockThreads, 0, cur_stream());
    q8_1::quantize_q8_1<<<q8_1::quantize_q8_1_grid(int(nblocks * n)), q8_1::kQuantizeBlockThreads, 0, cur_stream()>>>(
        X, xq, int(nblocks * n));

    // Batched warp-per-row gemv -> Y[n, rows] (position-major).
    launch_gemv_q8_1_batched(W.type(), wp, xq, Y, int(blocks_per_row), ir, n);
    return cudaGetLastError() == cudaSuccess;
  }

  // fp32 fallback (differential testing against the CPU reference): each
  // position is contiguous in X [n, cols], so the gemv reads the position's
  // row in place and the result is copied back contiguously to Y [n, rows].
  float* row = static_cast<float*>(
      grow_scratch(g_row, g_row_bytes, rows * sizeof(float)));
  if (!row) return false;
  for (int c = 0; c < n; ++c) {
    launch_gemv(W.type(), wp, X + (size_t)c * ic, row, int(blocks_per_row), ir);
    if (cudaMemcpyAsync(Y + (size_t)c * ir, row, rows * sizeof(float),
                        cudaMemcpyDeviceToDevice, cur_stream()) != cudaSuccess)
      return false;
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

// Device-resident single-row embedding gather (Phase 3.2): dequantize one row
// of a K-quant embedding to device f32, so the decode preamble's token_embd /
// per_layer_token_embd lookups run on the GPU and the two per-token H2D uploads
// (embd 10240 B, ple_row 43008 B) disappear. Only the token's packed row is
// uploaded (the full tensor may be ~1.5 GiB and is not kept resident); no host
// copy of the dequantized row happens.
bool gather_f32_dev(const QuantizedTensor& W, int64_t token, float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint64_t vocab = W.dims()[1];
  if (token < 0 || uint64_t(token) >= vocab || !out) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  // Upload only this token's packed row. The full tensor is ~1.5 GiB for
  // per_layer_token_embd ([total_per, vocab] Q4_K) and does not fit resident
  // next to the block weights on an 8 GiB part, so a whole-tensor cache is the
  // wrong tool here — the gather needs just `blocks_per_row` blocks.
  const uint32_t bpb = W.bytes_per_block();
  const size_t row_bytes = size_t(blocks_per_row) * bpb;
  const size_t row_off = size_t(token) * row_bytes;
  void* d_row = grow_scratch(g_gather, g_gather_bytes, row_bytes);
  if (!d_row) return false;
  if (cudaMemcpy(d_row, reinterpret_cast<const uint8_t*>(W.data().data()) + row_off,
                 row_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  const uint8_t* wp = static_cast<const uint8_t*>(d_row);
  return launch_gather_row(wp, /*row_host=*/0, /*row_dev=*/nullptr, W.type(),
                           int(width), int(blocks_per_row), out);
}

// Public host-in/out single-row gather (Phase 3.2 B5): allocates a device output
// scratch, runs gather_f32_dev into it, and copies the dequantized row back to
// host. Exists so the differential test (vs quant::dequantize_row_f32) can stay a
// clean public-header suite without touching raw CUDA.
bool gather_f32(const QuantizedTensor& W, int64_t token, std::span<float> out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  if (token < 0 || uint64_t(token) >= W.dims()[1]) return false;
  if (out.size() < width) return false;

  float* d_out = static_cast<float*>(
      grow_scratch(g_gather_out, g_gather_out_bytes, width * sizeof(float)));
  if (!d_out) return false;
  if (!gather_f32_dev(W, token, d_out)) return false;
  return cudaMemcpy(out.data(), d_out, width * sizeof(float),
                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

// Device-resident single-row gather (Phase 3.2 B2): like gather_f32_dev, but
// reads the token's row straight out of the cached resident weight — no
// per-token H2D upload. Only for token_embd (the tied lm_head), whose full
// tensor is already resident on the device (uploaded once by the logits matvec
// via upload_weight). The packed layout is [vocab][blocks_per_row] blocks, so
// the gather kernel's `row` offset walks straight to the token's packed row.
bool gather_f32_resident_dev(const QuantizedTensor& W, int64_t token, float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint64_t vocab = W.dims()[1];
  if (token < 0 || uint64_t(token) >= vocab || !out) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  // The resident copy is cached by (data pointer, byte size); the logits matvec
  // already uploaded it, so this is a lookup, not a copy.
  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  return launch_gather_row(static_cast<const uint8_t*>(d_w), int64_t(token),
                           /*row_dev=*/nullptr, W.type(), int(width),
                           int(blocks_per_row), out);
}

// Device-resident single-row gather, device-row variant (Phase 3.2 B4): like
// gather_f32_resident_dev but reads the token id from a device buffer
// (`token_dev`, a float index) at kernel time instead of a host value, so the
// next token id stays on the device between decode steps (argmax -> token_dev ->
// gather, no host round-trip). `token_dev` must point at one device float.
bool gather_f32_resident_dev_row(const QuantizedTensor& W, const float* token_dev,
                                 float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0 || !token_dev || !out) return false;
  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  return launch_gather_row(static_cast<const uint8_t*>(d_w), /*row_host=*/0,
                           token_dev, W.type(), int(width), int(blocks_per_row),
                           out);
}

// Zero-copy single-row gather (Phase 3.2 B3): pin+map W's packed bytes in place
// (cudaHostRegisterMapped) once, then let the gather kernel read the token's row
// straight out of host memory via the mapped device pointer. No per-token copy,
// synchronous or otherwise. Only for weights too big to keep resident (the
// ~1.5 GiB per_layer_token_embd); token_embd uses gather_f32_resident_dev.
bool gather_f32_mapped_dev(const QuantizedTensor& W, int64_t token, float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint64_t vocab = W.dims()[1];
  if (token < 0 || uint64_t(token) >= vocab || !out) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  // Register once, keyed by the source address so a model reload (new address)
  // re-pins instead of dereferencing a stale mapping.
  const void* d_w = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    const void* src = W.data().data();
    const size_t bytes = W.byte_size();
    if (g_ple_mapped.host != src || g_ple_mapped.bytes != bytes) {
      if (g_ple_mapped.host)
        cudaHostUnregister(const_cast<void*>(g_ple_mapped.host));
      g_ple_mapped = {};
      void* h = const_cast<void*>(src);
      if (cudaHostRegister(h, bytes, cudaHostRegisterMapped) != cudaSuccess)
        return false;
      void* dev = nullptr;
      if (cudaHostGetDevicePointer(&dev, h, 0) != cudaSuccess) {
        cudaHostUnregister(h);
        return false;
      }
      g_ple_mapped = {dev, src, bytes};
    }
    d_w = g_ple_mapped.dev;
  }
  return launch_gather_row(static_cast<const uint8_t*>(d_w), int64_t(token),
                           /*row_dev=*/nullptr, W.type(), int(width),
                           int(blocks_per_row), out);
}

// Zero-copy single-row gather, device-row variant (Phase 3.2 B4): like
// gather_f32_mapped_dev but reads the token id from a device buffer at kernel
// time (see gather_f32_resident_dev_row).
bool gather_f32_mapped_dev_row(const QuantizedTensor& W, const float* token_dev,
                               float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0 || !token_dev || !out) return false;

  const void* d_w = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    const void* src = W.data().data();
    const size_t bytes = W.byte_size();
    if (g_ple_mapped.host != src || g_ple_mapped.bytes != bytes) {
      if (g_ple_mapped.host)
        cudaHostUnregister(const_cast<void*>(g_ple_mapped.host));
      g_ple_mapped = {};
      void* h = const_cast<void*>(src);
      if (cudaHostRegister(h, bytes, cudaHostRegisterMapped) != cudaSuccess)
        return false;
      void* dev = nullptr;
      if (cudaHostGetDevicePointer(&dev, h, 0) != cudaSuccess) {
        cudaHostUnregister(h);
        return false;
      }
      g_ple_mapped = {dev, src, bytes};
    }
    d_w = g_ple_mapped.dev;
  }
  return launch_gather_row(static_cast<const uint8_t*>(d_w), /*row_host=*/0,
                           token_dev, W.type(), int(width), int(blocks_per_row),
                           out);
}

// Batched (prefill) embedding gather, resident variant: dequantize the n token
// rows of a device-resident packed tensor (token_embd, already resident via the
// logits matvec) to device f32 in one launch. `tokens` is a host span of n token
// ids, uploaded once to a device scratch; `out` is [width, n] column-major
// (n innermost). Mirrors gather_f32_resident_dev (the n == 1 case).
bool gather_f32_resident_dev_batched(const QuantizedTensor& W,
                                     std::span<const uint64_t> tokens, float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint64_t vocab = W.dims()[1];
  const int n = int(tokens.size());
  if (n <= 0 || !out) return false;
  for (uint64_t t : tokens)
    if (t >= vocab) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  void* d_rows = grow_scratch(g_tokens, g_tokens_bytes, size_t(n) * sizeof(uint64_t));
  if (!d_rows) return false;
  if (cudaMemcpy(d_rows, tokens.data(), size_t(n) * sizeof(uint64_t),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  return launch_gather_batched(static_cast<const uint8_t*>(d_w),
                               static_cast<const uint64_t*>(d_rows), W.type(),
                               int(width), int(blocks_per_row), n, out);
}

// Batched (prefill) embedding gather, mapped variant: like the resident variant
// but reads rows straight out of the pin+mapped host tensor (per_layer_token_embd,
// ~1.5 GiB and not kept resident). Mirrors gather_f32_mapped_dev (n == 1).
bool gather_f32_mapped_dev_batched(const QuantizedTensor& W,
                                   std::span<const uint64_t> tokens, float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  const uint64_t vocab = W.dims()[1];
  const int n = int(tokens.size());
  if (n <= 0 || !out) return false;
  for (uint64_t t : tokens)
    if (t >= vocab) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    const void* src = W.data().data();
    const size_t bytes = W.byte_size();
    if (g_ple_mapped.host != src || g_ple_mapped.bytes != bytes) {
      if (g_ple_mapped.host)
        cudaHostUnregister(const_cast<void*>(g_ple_mapped.host));
      g_ple_mapped = {};
      void* h = const_cast<void*>(src);
      if (cudaHostRegister(h, bytes, cudaHostRegisterMapped) != cudaSuccess)
        return false;
      void* dev = nullptr;
      if (cudaHostGetDevicePointer(&dev, h, 0) != cudaSuccess) {
        cudaHostUnregister(h);
        return false;
      }
      g_ple_mapped = {dev, src, bytes};
    }
    d_w = g_ple_mapped.dev;
  }
  void* d_rows = grow_scratch(g_tokens, g_tokens_bytes, size_t(n) * sizeof(uint64_t));
  if (!d_rows) return false;
  if (cudaMemcpy(d_rows, tokens.data(), size_t(n) * sizeof(uint64_t),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  return launch_gather_batched(static_cast<const uint8_t*>(d_w),
                               static_cast<const uint64_t*>(d_rows), W.type(),
                               int(width), int(blocks_per_row), n, out);
}

// Device-token variants (CUDA Graphs for chunked prefill): read the n token ids
// straight from a persistent device buffer, skipping the synchronous cudaMemcpy
// H2D that the host-span variants perform (that copy cannot be recorded into a
// graph). The caller stages the ids into the device buffer between replays.
bool gather_f32_resident_dev_batched_ids(const QuantizedTensor& W,
                                         const uint64_t* token_ids_dev, int n,
                                         float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  if (n <= 0 || !token_ids_dev || !out) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = upload_weight(W);
  if (!d_w) return false;
  return launch_gather_batched(static_cast<const uint8_t*>(d_w), token_ids_dev,
                               W.type(), int(width), int(blocks_per_row), n, out);
}

bool gather_f32_mapped_dev_batched_ids(const QuantizedTensor& W,
                                       const uint64_t* token_ids_dev, int n,
                                       float* out) {
  if (!available()) return false;
  if (!W.valid() || W.dims().size() != 2) return false;
  const uint64_t width = W.dims()[0];
  if (n <= 0 || !token_ids_dev || !out) return false;
  const uint32_t blocks_per_row = uint32_t(width / W.block_size());  // exact (valid())
  if (blocks_per_row == 0) return false;

  const void* d_w = nullptr;
  {
    std::lock_guard<std::mutex> lk(g_cache_mutex);
    const void* src = W.data().data();
    const size_t bytes = W.byte_size();
    if (g_ple_mapped.host != src || g_ple_mapped.bytes != bytes) {
      if (g_ple_mapped.host)
        cudaHostUnregister(const_cast<void*>(g_ple_mapped.host));
      g_ple_mapped = {};
      void* h = const_cast<void*>(src);
      if (cudaHostRegister(h, bytes, cudaHostRegisterMapped) != cudaSuccess)
        return false;
      void* dev = nullptr;
      if (cudaHostGetDevicePointer(&dev, h, 0) != cudaSuccess) {
        cudaHostUnregister(h);
        return false;
      }
      g_ple_mapped = {dev, src, bytes};
    }
    d_w = g_ple_mapped.dev;
  }
  return launch_gather_batched(static_cast<const uint8_t*>(d_w), token_ids_dev,
                               W.type(), int(width), int(blocks_per_row), n, out);
}

} // namespace sonicboom::quant::cuda
