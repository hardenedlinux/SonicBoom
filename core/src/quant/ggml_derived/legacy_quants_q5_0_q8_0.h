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
// Derived from ggml (llama.cpp), MIT-licensed. The "legacy" 32-element block
// formats Q5_0 and Q8_0 below (block layouts, dequantization, q8_0 activation
// quantization, and the fused q5_0/q8_0 x q8_0 inner products) reproduce:
//
//   ggml/src/ggml-common.h   — block_q5_0, block_q8_0
//   ggml/src/ggml-quants.c   — dequantize_row_q5_0, dequantize_row_q8_0,
//                              quantize_row_q8_0_ref
//   ggml/src/ggml-cpu/quants.c — ggml_vec_dot_q5_0_q8_0_generic,
//                              ggml_vec_dot_q8_0_q8_0_generic
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
// SonicBoom's own additions (float_to_half, the byte-span entry points, and the
// per-block dot shape used by quantized_matmul.cpp) are GPL-3.0-or-later as
// above.
// ---------------------------------------------------------------------------
//
// Difference from the K-quants: Q5_0/Q8_0 blocks hold QK=32 elements and carry
// a single fp16 scale `d` (Q5_0 also a 5th-bit plane `qh`), where the K-quants
// use 256-element super-blocks with 6-bit sub-scales and a min term. ggml's
// activation type for both is q8_0 (32-element int8 blocks, one fp16 scale),
// not q8_K — so the legacy matmul path quantizes activations to q8_0 while the
// K-quant path quantizes to q8_K.

#pragma once

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

// Reuses the shared ggml_derived half_to_float (same value, one definition for
// the whole namespace).
#include "vec_dot_qK_q8K.h"

namespace sonicboom::quant::ggml_derived {

// --- block layouts (ggml-common.h) -----------------------------------------

constexpr int QK5_0 = 32;
constexpr int QK8_0 = 32;

struct block_q5_0 {           // 22 bytes
  uint16_t d;                 // fp16 delta
  uint8_t qh[4];              // 5th bit of the quants
  uint8_t qs[QK5_0 / 2];      // nibbles
};
static_assert(sizeof(block_q5_0) == 2 + 4 + QK5_0 / 2, "q5_0 block size");

struct block_q8_0 {           // 34 bytes
  uint16_t d;                 // fp16 delta
  int8_t qs[QK8_0];           // quants
};
static_assert(sizeof(block_q8_0) == 2 + QK8_0, "q8_0 block size");

// --- fp16 helpers ----------------------------------------------------------
//
// half_to_float comes from vec_dot_qK_q8K.h (same namespace, one definition).

// binary32 -> binary16 with round-to-nearest-even, matching GGML_FP32_TO_FP16
// (_cvtss_sh with the default rounding mode) and ggml's portable fallback.
inline uint16_t float_to_half(float value) {
  const uint32_t x = std::bit_cast<uint32_t>(value);
  const uint32_t sign = (x >> 16) & 0x8000u;
  int exp = int((x >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) {
    if (exp < -10) return uint16_t(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t result = mant >> shift;
    const uint32_t rem = mant & ((uint32_t(1) << shift) - 1);
    const uint32_t halfway = uint32_t(1) << (shift - 1);
    if (rem > halfway || (rem == halfway && (result & 1))) ++result;
    return uint16_t(sign | result);
  }
  if (exp >= 31) return uint16_t(sign | 0x7c00u | (mant ? 0x0200u : 0));
  uint32_t result = (uint32_t(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (result & 1))) ++result;
  return uint16_t(sign | result);
}

// --- dequantization (ggml-quants.c dequantize_row_q5_0 / _q8_0) -------------

// Q5_0: 5-bit quants (low nibble in qs, 5th bit in qh), offset by -16, scaled by
// the fp16 delta. `out` receives n_blocks * 32 floats.
inline void dequantize_q5_0(std::span<const std::byte> blocks, uint64_t n_blocks,
                            float* out) {
  const auto* x = reinterpret_cast<const block_q5_0*>(blocks.data());
  for (uint64_t i = 0; i < n_blocks; ++i) {
    const float d = half_to_float(x[i].d);
    uint32_t qh;
    std::memcpy(&qh, x[i].qh, sizeof(qh));
    for (int j = 0; j < QK5_0 / 2; ++j) {
      const uint8_t xh_0 = uint8_t(((qh >> (j + 0)) << 4) & 0x10u);
      const uint8_t xh_1 = uint8_t(((qh >> (j + 12))) & 0x10u);
      const int32_t v0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t v1 = ((x[i].qs[j] >> 4) | xh_1) - 16;
      out[i * QK5_0 + j + 0] = float(v0) * d;
      out[i * QK5_0 + j + QK5_0 / 2] = float(v1) * d;
    }
  }
}

// Q8_0: plain int8 quants scaled by the fp16 delta.
inline void dequantize_q8_0(std::span<const std::byte> blocks, uint64_t n_blocks,
                            float* out) {
  const auto* x = reinterpret_cast<const block_q8_0*>(blocks.data());
  for (uint64_t i = 0; i < n_blocks; ++i) {
    const float d = half_to_float(x[i].d);
    for (int j = 0; j < QK8_0; ++j) out[i * QK8_0 + j] = float(x[i].qs[j]) * d;
  }
}

// --- activation quantization (ggml-quants.c quantize_row_q8_0_ref) ----------

// Quantize a contiguous f32 activation (length `n`, a multiple of 32) to q8_0:
// per block, d = amax / 127 stored as fp16, qs[j] = round(x[j] / d).
inline void quantize_q8_0(const float* x, uint64_t n, block_q8_0* out) {
  const uint64_t nb = n / QK8_0;
  for (uint64_t i = 0; i < nb; ++i) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
      const float v = x[i * QK8_0 + j];
      amax = amax > std::fabs(v) ? amax : std::fabs(v);
    }
    const float d = amax / 127.0f;
    const float id = d ? 1.0f / d : 0.0f;
    out[i].d = float_to_half(d);
    for (int j = 0; j < QK8_0; ++j) {
      out[i].qs[j] = int8_t(std::lround(x[i * QK8_0 + j] * id));
    }
  }
}

// --- fused inner products (ggml-cpu/quants.c *) ----------------------------
//
// Both return the sum of the per-block contributions in block order, which is
// the accumulation order ggml uses, so a row dot is bit-exact against the
// reference. `w` points at `n_blocks` packed weight blocks; `y` at `n_blocks`
// q8_0 activation blocks.

inline float vec_dot_q5_0_q8_0(const std::byte* w, const block_q8_0* y,
                              uint64_t n_blocks) {
  const auto* x = reinterpret_cast<const block_q5_0*>(w);
  float sumf = 0.0f;
  for (uint64_t i = 0; i < n_blocks; ++i) {
    uint32_t qh;
    std::memcpy(&qh, x[i].qh, sizeof(qh));
    int sumi0 = 0;
    int sumi1 = 0;
    for (int j = 0; j < QK8_0 / 2; ++j) {
      const uint8_t xh_0 = uint8_t(((qh & (1u << (j + 0))) >> (j + 0)) << 4);
      const uint8_t xh_1 = uint8_t((qh & (1u << (j + 16))) >> (j + 12));
      const int32_t v0 = ((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t v1 = ((x[i].qs[j] >> 4) | xh_1) - 16;
      sumi0 += v0 * int32_t(y[i].qs[j]);
      sumi1 += v1 * int32_t(y[i].qs[j + QK8_0 / 2]);
    }
    const int sumi = sumi0 + sumi1;
    // Left-to-right association, matching ggml's `sumf += sumi*d5*d8` as
    // ((sumi*d5)*d8): bit-identical to the reference, not merely equivalent.
    sumf += float(sumi) * half_to_float(x[i].d) * half_to_float(y[i].d);
  }
  return sumf;
}

inline float vec_dot_q8_0_q8_0(const std::byte* w, const block_q8_0* y,
                              uint64_t n_blocks) {
  const auto* x = reinterpret_cast<const block_q8_0*>(w);
  float sumf = 0.0f;
  for (uint64_t i = 0; i < n_blocks; ++i) {
    int sumi = 0;
    for (int j = 0; j < QK8_0; ++j) sumi += int32_t(x[i].qs[j]) * int32_t(y[i].qs[j]);
    sumf += float(sumi) * half_to_float(x[i].d) * half_to_float(y[i].d);
  }
  return sumf;
}

} // namespace sonicboom::quant::ggml_derived
