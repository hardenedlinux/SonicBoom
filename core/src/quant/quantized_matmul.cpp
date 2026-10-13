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

#include <sonicboom/quant/quantized_matmul.h>

#include <sonicboom/thread_pool.h>

#include <sonicboom/quant/dequant.h>

#include "ggml_derived/vec_dot_qK_q8K.h"

#ifdef SONICBOOM_USE_CUDA
#include <sonicboom/quant/quantized_matmul_cuda.h>
#endif

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <vector>

// Scalar reference quantized matvec/matmul. Two activation modes:
//
//   matmul_f32      — plain f32 activation. For each output row it walks the
//                     row's blocks, dequantizes each 256-element block into a
//                     small scratch buffer, and accumulates a f32 dot against
//                     the matching activation slice. The mathematically closest
//                     reference.
//   matmul_f32_q8_K — llama.cpp-equivalent. The f32 activation is quantized to
//                     q8_K (int8 + per-256-block scale), then dotted against the
//                     weight using ggml's fused fixed-point inner products
//                     (ggml_vec_dot_q*_K_q8_K_generic): the q8*q_nibble product
//                     stays in int16, the scale*q8*q_nibble in int32, and only
//                     the outer d scale and the final sums are float. This
//                     reproduces llama.cpp's arithmetic bit-for-bit (modulo fp32
//                     accumulation order, which is kept identical), for direct
//                     differential comparison against the baseline.

namespace sonicboom::quant {

namespace {

// ggml-derived scalar q8_K reference (moved to ggml_derived/vec_dot_qK_q8K.h,
// which carries the ggml MIT copyright + permission notice).
using ggml_derived::Q8KBlock;
using ggml_derived::dot_q3_K_block;
using ggml_derived::dot_q4_K_block;
using ggml_derived::dot_q5_K_block;
using ggml_derived::quantize_q8_K_blocks;

// IEEE 754 binary16 -> binary32. Bit-identical to ggml's GGML_FP16_TO_FP32 for
// all finite values; mirrored from core/src/quant/dequant.cpp so this file stays
// self-contained without exposing an fp16 helper on a public header.
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


void dequantize_block(QuantType type, std::span<const std::byte> block,
                      float* out) {
  switch (type) {
    case QuantType::Q3_K: dequantize_q3_K(block, 1, out); break;
    case QuantType::Q4_K: dequantize_q4_K(block, 1, out); break;
    case QuantType::Q5_K: dequantize_q5_K(block, 1, out); break;
  }
}



// --- AVX2 fast path -------------------------------------------------------
//
// Same fixed-point arithmetic and 8-lane int32 accumulation order as the scalar
// reference above, so these are bit-exact. Only the innermost int8→int16→int32
// accumulate is vectorized (8 lanes = one 256-bit op); the nibble unpack and
// scale/mins unpack stay scalar for now. The dequant output (aux8) is exact
// integers and the accumulation is exact 32-bit integer math, so the only
// float operations are the final dq/dminq scales — identical to the scalar.
//
// The functions carry target("avx2") so the TU compiles without -mavx2; the
// runtime dispatch in block_dot_for() gates them behind
// __builtin_cpu_supports("avx2").

// 8 int8 → 8 int16 (sign-extended) as 256-bit lanes. `p` holds 8 int8.
__attribute__((target("avx2")))
inline __m256i cvt8_epi16(const int8_t* p) {
  return _mm256_cvtepi8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

// 8 int16 products (q8s[l] * a[l]) widened to int32 and accumulated with a
// broadcast scale, exactly mirroring the scalar `aux32[l] += scale * aux16[l]`
// for one 8-element group. Returns acc + scale*Σ over the group.
__attribute__((target("avx2")))
inline __m256i madd_q8_aux(const int8_t* q8s, const int8_t* a, int32_t scale,
                           __m256i acc) {
  const __m256i prod16 = _mm256_mullo_epi16(cvt8_epi16(q8s), cvt8_epi16(a));
  const __m256i prod32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod16));
  return _mm256_add_epi32(acc, _mm256_mullo_epi32(prod32, _mm256_set1_epi32(scale)));
}

// --- Vectorized nibble unpack ---------------------------------------------
//
// The scalar reference builds a 256-entry `aux8` buffer by walking the packed
// 2/4-bit weight nibbles with per-byte shift/mask. These AVX2 helpers produce
// the identical byte sequence (same values, same order) using 16-bit-lane shift
// + mask, so the dot that follows stays bit-exact. srli_epi16 shifts within
// 16-bit lanes; masking each byte down to its low 2/4 bits removes the bleed
// from the neighbouring byte, which is what makes the per-byte extract correct.

// High nibble of every byte (mask off the neighbour-byte bleed).
__attribute__((target("avx2")))
inline __m256i nibbles_hi(__m256i v) {
  return _mm256_and_si256(_mm256_srli_epi16(v, 4), _mm256_set1_epi8(0x0F));
}

// The `shift`-bit field of every byte, masked to `mask` bits (2 for q3, 4 for
// q4/q5). `mask` must fit in a signed char (0x03 / 0x0F here).
__attribute__((target("avx2")))
inline __m256i field_lo(__m256i v, int shift, int mask) {
  return _mm256_and_si256(_mm256_srli_epi16(v, shift), _mm256_set1_epi8((char)mask));
}

// 0x10 where (byte & m) != 0, else 0. Bitwise throughout, so the sign of `m`
// (e.g. m == 0x80) is irrelevant.
__attribute__((target("avx2")))
inline __m256i add16_if_mask(__m256i bytes, uint8_t m) {
  const __m256i bit = _mm256_and_si256(bytes, _mm256_set1_epi8((char)m));
  return _mm256_andnot_si256(_mm256_cmpeq_epi8(bit, _mm256_setzero_si256()),
                             _mm256_set1_epi8(16));
}

// -4 where (byte & m) == 0, else 0 (q3_K's signed-offset decode).
__attribute__((target("avx2")))
inline __m256i sub4_if_unset(__m256i bytes, uint8_t m) {
  const __m256i bit = _mm256_and_si256(bytes, _mm256_set1_epi8((char)m));
  return _mm256_and_si256(_mm256_cmpeq_epi8(bit, _mm256_setzero_si256()),
                          _mm256_set1_epi8(-4));
}

__attribute__((target("avx2")))
void dot_q3_K_block_avx2(const std::byte* wp, const Q8KBlock& q8, float sums[8],
                         float& sumf) {
  (void)sumf;  // q3_K has no dmin correction.
  const uint8_t* hm = reinterpret_cast<const uint8_t*>(wp + 0);
  const uint8_t* q3 = reinterpret_cast<const uint8_t*>(wp + 32);
  const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(wp + 96);
  const float d = half_to_float(load_u16_le(wp + 108));

  int8_t aux8[256];
  {
    const __m256i hmv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hm));
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int chunk = 0; chunk < 2; ++chunk) {
      const __m256i v =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q3 + chunk * 32));
      for (int k = 0; k < 4; ++k) {
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(a),
                            _mm256_add_epi8(field_lo(v, 2 * k, 0x03),
                                            sub4_if_unset(hmv, m)));
        a += 32;
        m <<= 1;
      }
    }
  }

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
  const int8_t* a = aux8;
  __m256i acc = _mm256_setzero_si256();
  for (int j = 0; j < 16; ++j) {
    const int32_t sc = scales[j] - 32;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
  }
  int32_t aux32[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(aux32), acc);

  const float dq = d * q8.d;
  for (int l = 0; l < 8; ++l) sums[l] += dq * float(aux32[l]);
}

__attribute__((target("avx2")))
void dot_q4_K_block_avx2(const std::byte* wp, const Q8KBlock& q8, float sums[8],
                         float& sumf) {
  const float d = half_to_float(load_u16_le(wp + 0));
  const float dmin = half_to_float(load_u16_le(wp + 2));
  const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(wp + 4);
  const uint8_t* q4 = reinterpret_cast<const uint8_t*>(wp + 16);

  int8_t aux8[256];
  {
    int8_t* a = aux8;
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    for (int j = 0; j < 4; ++j) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a),
                          _mm256_and_si256(v, mask0F));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a + 32), nibbles_hi(v));
      a += 64;
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

  const int8_t* a = aux8;
  __m256i acc = _mm256_setzero_si256();
  int is = 0;
  for (int j = 0; j < 8; ++j) {
    const int32_t sc = scales[is++];
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
  }
  int32_t aux32[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(aux32), acc);

  const float dq = d * q8.d;
  for (int l = 0; l < 8; ++l) sums[l] += dq * float(aux32[l]);
  const float dminq = dmin * q8.d;
  sumf -= dminq * float(sumi);
}

__attribute__((target("avx2")))
void dot_q5_K_block_avx2(const std::byte* wp, const Q8KBlock& q8, float sums[8],
                         float& sumf) {
  const float d = half_to_float(load_u16_le(wp + 0));
  const float dmin = half_to_float(load_u16_le(wp + 2));
  const uint8_t* scales_src = reinterpret_cast<const uint8_t*>(wp + 4);
  const uint8_t* hm = reinterpret_cast<const uint8_t*>(wp + 16);
  const uint8_t* q4 = reinterpret_cast<const uint8_t*>(wp + 48);

  int8_t aux8[256];
  {
    const __m256i hmv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(hm));
    const __m256i mask0F = _mm256_set1_epi8(0x0F);
    int8_t* a = aux8;
    uint8_t m = 1;
    for (int j = 0; j < 4; ++j) {
      const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q4));
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a),
                          _mm256_add_epi8(_mm256_and_si256(v, mask0F),
                                          add16_if_mask(hmv, m)));
      a += 32; m <<= 1;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(a),
                          _mm256_add_epi8(nibbles_hi(v),
                                          add16_if_mask(hmv, m)));
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

  const int8_t* a = aux8;
  __m256i acc = _mm256_setzero_si256();
  int is = 0;
  for (int j = 0; j < 8; ++j) {
    const int32_t sc = scales[is++];
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
    acc = madd_q8_aux(q8s, a, sc, acc); q8s += 8; a += 8;
  }
  int32_t aux32[8];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(aux32), acc);

  const float dq = d * q8.d;
  for (int l = 0; l < 8; ++l) sums[l] += dq * float(aux32[l]);
  const float dminq = dmin * q8.d;
  sumf -= dminq * float(sumi);
}

// Shared validation for the matmul variants: shape, non-zero batch, and
// X/Y sizes. Fills *cols/*rows (W's GGUF-order dims).
bool validate_matmul(const QuantizedTensor& W, uint64_t n, uint64_t x_elems,
                     uint64_t y_elems, uint64_t* cols, uint64_t* rows) {
  if (n == 0) return false;
  if (!W.valid()) return false;
  if (W.dims().size() != 2) return false;
  *cols = W.dims()[0];  // contiguous / inner dim
  *rows = W.dims()[1];  // outer dim
  if (x_elems < *cols * n || y_elems < *rows * n) return false;
  return true;
}

// Streaming f32 dot: Y = W @ Xa where Xa is [n, cols] position-major and Y is
// [n, rows] position-major. Used only by the plain-f32 mode (matmul_f32); the
// q8_K mode uses the fused fixed-point dot instead.
void dot_stream(const QuantizedTensor& W, const float* Xa, uint64_t cols,
                uint64_t rows, uint64_t n, float* Y) {
  const uint32_t block_size = W.block_size();
  const uint32_t bytes_per_block = W.bytes_per_block();
  const uint64_t blocks_per_row = cols / block_size;  // exact (valid() guarantees)

  const std::byte* data = W.data().data();
  std::vector<float> scratch(block_size);

  for (uint64_t i = 0; i < rows * n; ++i) Y[i] = 0.0f;

  for (uint64_t r = 0; r < rows; ++r) {
    for (uint64_t b = 0; b < blocks_per_row; ++b) {
      const uint64_t block_idx = r * blocks_per_row + b;
      const std::byte* block_ptr = data + block_idx * bytes_per_block;
      dequantize_block(W.type(), {block_ptr, bytes_per_block}, scratch.data());

      const uint64_t k0 = b * block_size;
      for (uint64_t c = 0; c < n; ++c) {
        const float* xc = Xa + c * cols + k0;
        float* yc = Y + c * rows;
        for (uint64_t j = 0; j < block_size; ++j) yc[r] += scratch[j] * xc[j];
      }
    }
  }
}

// Select the fused block-dot for the weight's quant type.
using BlockDotFn = void (*)(const std::byte*, const Q8KBlock&, float[8], float&);

BlockDotFn block_dot_for(QuantType type) {
  // Runtime dispatch: AVX2 kernels are bit-exact with the scalar reference, so
  // the only behavioural difference is speed. Guard with __builtin_cpu_supports
  // so the same libsonicboom.so runs on pre-AVX2 CPUs via the scalar path.
  if (__builtin_cpu_supports("avx2")) {
    switch (type) {
      case QuantType::Q3_K: return dot_q3_K_block_avx2;
      case QuantType::Q4_K: return dot_q4_K_block_avx2;
      case QuantType::Q5_K: return dot_q5_K_block_avx2;
    }
    return nullptr;
  }
  switch (type) {
    case QuantType::Q3_K: return dot_q3_K_block;
    case QuantType::Q4_K: return dot_q4_K_block;
    case QuantType::Q5_K: return dot_q5_K_block;
  }
  return nullptr;
}

} // namespace

bool matvec_f32(const QuantizedTensor& W, std::span<const float> x,
                std::span<float> y) {
  return matmul_f32(W, x, y, 1);
}

bool matvec_f32_q8_K(const QuantizedTensor& W, std::span<const float> x,
                     std::span<float> y) {
  return matmul_f32_q8_K(W, x, y, 1);
}

bool matmul_f32(const QuantizedTensor& W, std::span<const float> X,
                std::span<float> Y, uint64_t n) {
  uint64_t cols, rows;
  if (!validate_matmul(W, n, X.size(), Y.size(), &cols, &rows)) return false;
  dot_stream(W, X.data(), cols, rows, n, Y.data());
  return true;
}

bool matmul_f32_q8_K(const QuantizedTensor& W, std::span<const float> X,
                     std::span<float> Y, uint64_t n) {
  uint64_t cols, rows;
  if (!validate_matmul(W, n, X.size(), Y.size(), &cols, &rows)) return false;

  const BlockDotFn block_dot = block_dot_for(W.type());
  if (block_dot == nullptr) return false;

  const uint32_t bytes_per_block = W.bytes_per_block();
  const uint64_t blocks_per_row = cols / W.block_size();  // exact (valid())
  const std::byte* data = W.data().data();

  // Quantize the activation to q8_K per position (llama.cpp quantizes the
  // activation per-column, so the batch dimension is independent here too), then
  // dot each weight row against it with the fused fixed-point inner product.
  std::vector<Q8KBlock> act_blocks(blocks_per_row);
  std::vector<float> col(cols);

  // Rows are mutually independent (each writes only Y[c*rows+r]), so the row
  // loop splits across the thread pool. The per-row arithmetic is unchanged —
  // only which worker runs a row differs — so this stays bit-exact with the
  // single-threaded path and deterministic for any worker count.
  ThreadPool& pool = ThreadPool::instance();

  for (uint64_t c = 0; c < n; ++c) {
    for (uint64_t k = 0; k < cols; ++k) col[k] = X[c * cols + k];
    quantize_q8_K_blocks(col.data(), blocks_per_row, act_blocks.data());

    pool.parallel_for(0, rows, [&](uint64_t r_begin, uint64_t r_end) {
      for (uint64_t r = r_begin; r < r_end; ++r) {
        float sums[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float sumf = 0.0f;
        const uint64_t row_block_base = r * blocks_per_row;
        for (uint64_t b = 0; b < blocks_per_row; ++b) {
          const std::byte* block_ptr =
              data + (row_block_base + b) * bytes_per_block;
          block_dot(block_ptr, act_blocks[b], sums, sumf);
        }
        for (int l = 0; l < 8; ++l) sumf += sums[l];
        Y[c * rows + r] = sumf;
      }
    });
  }
  return true;
}

bool cuda_available() {
#ifdef SONICBOOM_USE_CUDA
  return cuda::available();
#else
  return false;
#endif
}

bool matvec(const QuantizedTensor& W, std::span<const float> x,
            std::span<float> y, MatmulBackend backend) {
  switch (backend) {
    case MatmulBackend::CpuQ8K:
      return matvec_f32_q8_K(W, x, y);
    case MatmulBackend::CpuF32:
      return matvec_f32(W, x, y);
    case MatmulBackend::Cuda:
#ifdef SONICBOOM_USE_CUDA
      if (cuda::available()) return cuda::matvec_f32(W, x, y);
#endif
      return matvec_f32(W, x, y);
  }
  return false;
}

bool matmul(const QuantizedTensor& W, std::span<const float> X,
            std::span<float> Y, uint64_t n, MatmulBackend backend) {
  switch (backend) {
    case MatmulBackend::CpuQ8K:
      return matmul_f32_q8_K(W, X, Y, n);
    case MatmulBackend::CpuF32:
      return matmul_f32(W, X, Y, n);
    case MatmulBackend::Cuda:
#ifdef SONICBOOM_USE_CUDA
      if (cuda::available()) return cuda::matmul_f32(W, X, Y, n);
#endif
      return matmul_f32(W, X, Y, n);
  }
  return false;
}

} // namespace sonicboom::quant
