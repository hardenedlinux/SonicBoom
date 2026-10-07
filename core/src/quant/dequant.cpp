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

#include <sonicboom/quant/dequant.h>

#include <bit>
#include <cstdint>
#include <cstring>

// Reference dequantization of the ggml block-quantized formats Q3_K, Q4_K,
// Q5_K. The reconstruction formulas and block layouts below are confirmed
// against ggml's reference implementation:
//
//   ggml/src/ggml-quants.c  (dequantize_row_q3_K / _q4_K / _q5_K)
//   ggml/src/ggml-common.h  (block_q3_K / block_q4_K / block_q5_K)
//
// (see https://github.com/hardenedlinux/velum/third_party/ggml). This file re-implements
// the arithmetic independently in clean C++23 — it does not include or link
// ggml. ggml is MIT-licensed; the differential tests in tests/core/
// test_dequant.cpp run this code against a verbatim ggml baseline for bit-exact
// agreement.

namespace sonicboom::quant {

namespace {

// IEEE 754 binary16 -> binary32. Identical in result to ggml's
// GGML_FP16_TO_FP32 (_cvtsh_ss / table) for all finite values; NaNs/Inf are
// preserved bit-for-bit but do not appear in valid weight scales.
float half_to_float(uint16_t h) {
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
uint16_t load_u16_le(const std::byte* p) {
  const uint16_t lo = uint16_t(std::to_integer<uint8_t>(p[0]));
  const uint16_t hi = uint16_t(std::to_integer<uint8_t>(p[1]));
  return uint16_t(lo | (hi << 8));
}

// Unpack the j-th 6-bit (scale, min) pair from a q4_K/q5_K 12-byte scales
// block. Identical to ggml's get_scale_min_k4.
void get_scale_min_k4(int j, const uint8_t* scales, uint8_t& d, uint8_t& m) {
  if (j < 4) {
    d = scales[j] & 63;
    m = scales[j + 4] & 63;
  } else {
    d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
    m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4);
  }
}

// Unpack the q3_K 12-byte scales block into 16 int8 values (6-bit each, so the
// 16 values fit in 96 bits). Identical to the aux/tmp sequence in ggml's
// dequantize_row_q3_K. The consumer subtracts 32 to recover the signed scale.
void unpack_q3_K_scales(const uint8_t* src, int8_t* scales) {
  uint32_t a[4] = {0, 0, 0, 0};
  std::memcpy(a, src, 12);  // a[0..2] = scales bytes 0..11 (LE)
  const uint32_t a0 = a[0], a1 = a[1], tmp = a[2];
  const uint32_t kmask1 = 0x03030303u;
  const uint32_t kmask2 = 0x0f0f0f0fu;
  uint32_t b[4];
  b[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
  b[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
  b[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
  b[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
  std::memcpy(scales, b, 16);
}

} // namespace

void dequantize_q3_K(std::span<const std::byte> blocks, uint64_t n_blocks,
                     float* out) {
  const std::byte* p = blocks.data();
  for (uint64_t b = 0; b < n_blocks; ++b) {
    // Serialized layout (block_q3_K, 110 bytes):
    //   hmask[32] qs[64] scales[12] d(bf16/fp16)[2]
    const uint8_t* hm = reinterpret_cast<const uint8_t*>(p + 0);
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p + 32);
    const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(p + 96);
    const float d_all = half_to_float(load_u16_le(p + 108));

    int8_t scales[16];
    unpack_q3_K_scales(scales_src, scales);

    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        float dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int8_t v = int8_t((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4);
          *out++ = dl * float(v);
        }
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          const int8_t v = int8_t((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4);
          *out++ = dl * float(v);
        }
        shift += 2;
        m <<= 1;
      }
      q += 32;
    }
    p += 110;
  }
}

void dequantize_q4_K(std::span<const std::byte> blocks, uint64_t n_blocks,
                     float* out) {
  const std::byte* p = blocks.data();
  for (uint64_t b = 0; b < n_blocks; ++b) {
    // Serialized layout (block_q4_K, 144 bytes):
    //   d(fp16)[2] dmin(fp16)[2] scales[12] qs[128]
    const float d = half_to_float(load_u16_le(p + 0));
    const float dmin = half_to_float(load_u16_le(p + 2));
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(p + 4);
    const uint8_t* q = reinterpret_cast<const uint8_t*>(p + 16);

    int is = 0;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, mi;
      get_scale_min_k4(is, scales, sc, mi);
      const float d1 = d * sc;
      const float m1 = dmin * mi;
      get_scale_min_k4(is + 1, scales, sc, mi);
      const float d2 = d * sc;
      const float m2 = dmin * mi;
      for (int l = 0; l < 32; ++l) *out++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *out++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
    p += 144;
  }
}

void dequantize_q5_K(std::span<const std::byte> blocks, uint64_t n_blocks,
                     float* out) {
  const std::byte* p = blocks.data();
  for (uint64_t b = 0; b < n_blocks; ++b) {
    // Serialized layout (block_q5_K, 176 bytes):
    //   d(fp16)[2] dmin(fp16)[2] scales[12] qh[32] qs[128]
    const float d = half_to_float(load_u16_le(p + 0));
    const float dmin = half_to_float(load_u16_le(p + 2));
    const uint8_t* scales = reinterpret_cast<const uint8_t*>(p + 4);
    const uint8_t* qh = reinterpret_cast<const uint8_t*>(p + 16);
    const uint8_t* ql = reinterpret_cast<const uint8_t*>(p + 48);

    int is = 0;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < 256; j += 64) {
      uint8_t sc, mi;
      get_scale_min_k4(is, scales, sc, mi);
      const float d1 = d * sc;
      const float m1 = dmin * mi;
      get_scale_min_k4(is + 1, scales, sc, mi);
      const float d2 = d * sc;
      const float m2 = dmin * mi;
      for (int l = 0; l < 32; ++l) *out++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l) *out++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
    p += 176;
  }
}

bool dequantize_f32(const QuantizedTensor& t, float* out, uint64_t out_capacity) {
  if (!t.valid()) return false;
  if (out_capacity < t.numel()) return false;
  switch (t.type()) {
    case QuantType::Q3_K:
      dequantize_q3_K(t.data(), t.num_blocks(), out);
      return true;
    case QuantType::Q4_K:
      dequantize_q4_K(t.data(), t.num_blocks(), out);
      return true;
    case QuantType::Q5_K:
      dequantize_q5_K(t.data(), t.num_blocks(), out);
      return true;
  }
  return false;
}

bool dequantize_row_f32(const QuantizedTensor& t, uint64_t row, float* out,
                        uint64_t out_capacity) {
  if (!t.valid()) return false;
  if (t.dims().size() != 2) return false;
  if (row >= t.dims()[1]) return false;
  const uint64_t width = t.dims()[0];  // contiguous dim
  if (out_capacity < width) return false;

  const uint64_t blocks_per_row = width / t.block_size();  // exact (valid())
  const uint64_t row_bytes = blocks_per_row * t.bytes_per_block();
  const std::span<const std::byte> blocks =
      t.data().subspan(row * row_bytes, row_bytes);

  switch (t.type()) {
    case QuantType::Q3_K:
      dequantize_q3_K(blocks, blocks_per_row, out);
      return true;
    case QuantType::Q4_K:
      dequantize_q4_K(blocks, blocks_per_row, out);
      return true;
    case QuantType::Q5_K:
      dequantize_q5_K(blocks, blocks_per_row, out);
      return true;
  }
  return false;
}

} // namespace sonicboom::quant
