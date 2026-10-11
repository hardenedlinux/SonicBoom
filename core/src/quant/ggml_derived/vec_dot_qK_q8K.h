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
// Derived from ggml (llama.cpp), MIT-licensed. The q8_K scalar reference
// below (nearest_int, the Q8KBlock/block_q8_K layout, quantize_q8_K_blocks,
// and the scalar dot_q3/q4/q5_K_block inner products) reproduces the
// following upstream sources:
//
//   ggml/src/ggml-quants.c  — nearest_int, quantize_row_q8_K_ref,
//                             ggml_vec_dot_q3/q4/q5_K_q8_K_generic
//   ggml/src/ggml-quants.h  — block_q8_K layout
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
// SonicBoom's own additions (the half_to_float / load_u16_le helpers below and
// the AVX2 kernels that stay in quantized_matmul.cpp) are GPL-3.0-or-later as
// above.
// ---------------------------------------------------------------------------

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace sonicboom::quant::ggml_derived {

// --- self-contained helpers (SonicBoom's own, not ggml-derived) ------------

// IEEE 754 binary16 -> binary32. Bit-identical to ggml's GGML_FP16_TO_FP32 for
// all finite values; mirrored from core/src/quant/dequant.cpp so this header
// stays self-contained.
inline float half_to_float(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exp = uint32_t(h >> 10) & 0x1Fu;
  uint32_t mant = uint32_t(h) & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;  // +0 / -0
    } else {
      // Subnormal: normalize the significand.
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) {
        mant <<= 1;
        --exp;
      }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);  // Inf / NaN
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(bits);
}

// Load a little-endian uint16 from a byte span (GGUF is little-endian).
inline uint16_t load_u16_le(const std::byte* p) {
  const uint16_t lo = uint16_t(std::to_integer<uint8_t>(p[0]));
  const uint16_t hi = uint16_t(std::to_integer<uint8_t>(p[1]));
  return uint16_t(lo | (hi << 8));
}

// --- derived from ggml/src/ggml-quants.c -----------------------------------

// ggml's nearest_int: round-half-to-even via the 1.5*2^23 magic constant.
// Reproduced verbatim so the q8_K activation quantization below agrees
// bit-for-bit with llama.cpp's quantize_row_q8_K.
inline int nearest_int(float fval) {
  float val = fval + 12582912.0f;
  int i;
  std::memcpy(&i, &val, sizeof(int));
  return (i & 0x007fffff) - 0x00400000;
}

// A q8_K activation block in raw form, laid out exactly like ggml's block_q8_K
// (d, then qs[256], then bsums[16] — the per-16-element signed sums). Keeping the
// activation in this raw form is what lets the dot below stay in fixed point
// instead of dequantizing to f32 first.
struct Q8KBlock {
  float d;
  int8_t qs[256];
  int16_t bsums[16];
};

// Quantize a contiguous f32 activation vector (length `k`, a multiple of 256)
// to q8_K blocks, matching ggml's quantize_row_q8_K_ref: per 256-block, `m` is
// the element of largest magnitude (signed); iscale = -127/m; qs[j] =
// min(127, nearest_int(iscale*x[j])); d = 1/iscale; bsums[j] are the running
// sums of |qs| over each 16-group. An all-zero block yields d = 0, qs = 0,
// bsums = 0. `out` holds `k/256` blocks.
inline void quantize_q8_K_blocks(const float* x, uint64_t nb, Q8KBlock* out) {
  constexpr uint64_t QK = 256;
  for (uint64_t b = 0; b < nb; ++b) {
    const float* xb = x + b * QK;
    float amax = 0.0f, m = 0.0f;
    for (uint64_t j = 0; j < QK; ++j) {
      const float ax = std::fabs(xb[j]);
      if (ax > amax) {
        amax = ax;
        m = xb[j];
      }
    }
    if (amax == 0.0f) {
      out[b].d = 0.0f;
      std::memset(out[b].qs, 0, QK);
      std::memset(out[b].bsums, 0, sizeof(out[b].bsums));
      continue;
    }
    const float iscale = -127.0f / m;
    for (uint64_t j = 0; j < QK; ++j) {
      const int v = nearest_int(iscale * xb[j]);
      out[b].qs[j] = int8_t((127 < v) ? 127 : v);  // ggml's MIN(127, v)
    }
    for (uint64_t j = 0; j < QK / 16; ++j) {
      int sum = 0;
      for (uint64_t ii = 0; ii < 16; ++ii) sum += out[b].qs[j * 16 + ii];
      out[b].bsums[j] = int16_t(sum);
    }
    out[b].d = 1.0f / iscale;  // == -m/127
  }
}

// Fused q*_K·q8_K per-block inner products, reproducing ggml's
// ggml_vec_dot_q*_K_q8_K_generic exactly (same integer widths, same
// accumulation order). Each accumulates into the caller's per-lane `sums[8]`
// and the dmin-correction accumulator `sumf`, so the caller can reproduce
// ggml's cross-block float accumulation order (sums[l] += d*aux32[l] per block,
// sumf -= dmin*sumi per block, then sumf += Σ sums[l] at the end).

inline void dot_q3_K_block(const std::byte* wp, const Q8KBlock& q8, float sums[8],
                           float& sumf) {
  (void)sumf;  // q3_K has no dmin correction.
  // block_q3_K (110 bytes): hmask[32] qs[64] scales[12] d(fp16)[2]
  const uint8_t* hm = reinterpret_cast<const uint8_t*>(wp + 0);
  const uint8_t* q3 = reinterpret_cast<const uint8_t*>(wp + 32);
  const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(wp + 96);
  const float d = half_to_float(load_u16_le(wp + 108));

  int8_t aux8[256];
  {
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < 256; j += 128) {
      for (int l = 0; l < 32; ++l) a[l] = int8_t(q3[l] & 3);
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = int8_t((q3[l] >> 2) & 3);
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = int8_t((q3[l] >> 4) & 3);
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = int8_t((q3[l] >> 6) & 3);
      for (int l = 0; l < 32; ++l) a[l] -= (hm[l] & m) ? 0 : 4;
      a += 32; m <<= 1;
      q3 += 32;
    }
  }

  // Unpack the 12-byte scales block into 16 int8 scale values (6-bit each).
  uint32_t auxs[4];
  std::memcpy(auxs, scales_src, 12);
  const uint32_t kmask1 = 0x03030303u;
  const uint32_t kmask2 = 0x0f0f0f0fu;
  const uint32_t tmp = auxs[2];
  auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
  auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
  const int8_t* scales = reinterpret_cast<const int8_t*>(&auxs[0]);

  const int8_t* q8s = q8.qs;
  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int16_t aux16[8];
  const int8_t* a = aux8;
  for (int j = 0; j < 16; ++j) {
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += (scales[j] - 32) * aux16[l];
    q8s += 8; a += 8;
  }

  const float dq = d * q8.d;
  for (int l = 0; l < 8; ++l) sums[l] += dq * float(aux32[l]);
}

inline void dot_q4_K_block(const std::byte* wp, const Q8KBlock& q8, float sums[8],
                           float& sumf) {
  // block_q4_K (144 bytes): d(fp16)[2] dmin(fp16)[2] scales[12] qs[128]
  const float d = half_to_float(load_u16_le(wp + 0));
  const float dmin = half_to_float(load_u16_le(wp + 2));
  const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(wp + 4);
  const uint8_t* q4 = reinterpret_cast<const uint8_t*>(wp + 16);

  int8_t aux8[256];
  {
    int8_t* a = aux8;
    for (int j = 0; j < 4; ++j) {  // QK_K/64
      for (int l = 0; l < 32; ++l) a[l] = int8_t(q4[l] & 0xF);
      a += 32;
      for (int l = 0; l < 32; ++l) a[l] = int8_t(q4[l] >> 4);
      a += 32;
      q4 += 32;
    }
  }

  // Unpack the 12-byte scales block into 8 scale bytes + 8 min bytes.
  uint32_t utmp[4];
  std::memcpy(utmp, scales_src, 12);
  const uint32_t kmask1 = 0x3f3f3f3fu;
  const uint32_t kmask2 = 0x0f0f0f0fu;
  const uint32_t kmask3 = 0x03030303u;
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);

  const int8_t* q8s = q8.qs;

  int sumi = 0;
  for (int j = 0; j < 16; ++j) sumi += q8.bsums[j] * mins[j / 2];

  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int16_t aux16[8];
  const int8_t* a = aux8;
  int is = 0;
  for (int j = 0; j < 8; ++j) {  // QK_K/32
    const int32_t scale = scales[is++];
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
  }

  const float dq = d * q8.d;
  for (int l = 0; l < 8; ++l) sums[l] += dq * float(aux32[l]);
  const float dminq = dmin * q8.d;
  sumf -= dminq * float(sumi);
}

inline void dot_q5_K_block(const std::byte* wp, const Q8KBlock& q8, float sums[8],
                           float& sumf) {
  // block_q5_K (176 bytes): d(fp16)[2] dmin(fp16)[2] scales[12] qh[32] qs[128]
  const float d = half_to_float(load_u16_le(wp + 0));
  const float dmin = half_to_float(load_u16_le(wp + 2));
  const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(wp + 4);
  const uint8_t* hm = reinterpret_cast<const uint8_t*>(wp + 16);
  const uint8_t* q4 = reinterpret_cast<const uint8_t*>(wp + 48);

  int8_t aux8[256];
  {
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < 4; ++j) {  // QK_K/64
      for (int l = 0; l < 32; ++l) a[l] = int8_t(q4[l] & 0xF);
      for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m) ? 16 : 0;
      a += 32; m <<= 1;
      for (int l = 0; l < 32; ++l) a[l] = int8_t(q4[l] >> 4);
      for (int l = 0; l < 32; ++l) a[l] += (hm[l] & m) ? 16 : 0;
      a += 32; m <<= 1;
      q4 += 32;
    }
  }

  uint32_t utmp[4];
  std::memcpy(utmp, scales_src, 12);
  const uint32_t kmask1 = 0x3f3f3f3fu;
  const uint32_t kmask2 = 0x0f0f0f0fu;
  const uint32_t kmask3 = 0x03030303u;
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;
  const uint8_t* scales = reinterpret_cast<const uint8_t*>(&utmp[0]);
  const uint8_t* mins = reinterpret_cast<const uint8_t*>(&utmp[2]);

  const int8_t* q8s = q8.qs;

  int sumi = 0;
  for (int j = 0; j < 16; ++j) sumi += q8.bsums[j] * mins[j / 2];

  int32_t aux32[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int16_t aux16[8];
  const int8_t* a = aux8;
  int is = 0;
  for (int j = 0; j < 8; ++j) {  // QK_K/32
    const int32_t scale = scales[is++];
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = int16_t(q8s[l] * a[l]);
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8s += 8; a += 8;
  }

  const float dq = d * q8.d;
  for (int l = 0; l < 8; ++l) sums[l] += dq * float(aux32[l]);
  const float dminq = dmin * q8.d;
  sumf -= dminq * float(sumi);
}

} // namespace sonicboom::quant::ggml_derived
