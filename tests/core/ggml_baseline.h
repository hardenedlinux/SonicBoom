// Reference dequantization baseline for differential testing.
//
// This file contains a verbatim (minimally adapted) transcription of ggml's
// block-quantized dequantization for Q3_K, Q4_K and Q5_K, used ONLY to verify
// SonicBoom's independent reimplementation (core/src/quant/dequant.cpp) in
// tests/core/test_dequant.cpp. It is not linked into libsonicboom.so and is not
// part of the runtime.
//
// Upstream source (MIT license, Copyright (c) 2023-2024 The ggml authors):
//   ggml/src/ggml-quants.c  — dequantize_row_q3_K / _q4_K / _q5_K, get_scale_min_k4
//   ggml/src/ggml-common.h  — block_q3_K / block_q4_K / block_q5_K, QK_K, K_SCALE_SIZE
// (upstream ggml checkout)
//
// The only adaptation is substituting `GGML_FP16_TO_FP32` with the local
// `baseline_half_to_float` (an independent half->float formulation) and dropping
// ggml's SIMD/restrict annotations. The block layout, scale unpacking and
// reconstruction arithmetic are unchanged.

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace ggml_baseline {

typedef uint16_t ggml_half;

#define GGML_BASELINE_QK_K 256
#define GGML_BASELINE_K_SCALE_SIZE 12

// Independent half->float (IEEE 754 binary16 -> binary32). Written differently
// from SonicBoom's half_to_float so the two are textually independent, but the
// result is bit-identical for every finite input.
static inline float baseline_half_to_float(uint16_t h) {
  const uint32_t s = uint32_t(h & 0x8000u) << 16;
  const uint32_t e = uint32_t(h >> 10) & 0x1Fu;
  const uint32_t m = uint32_t(h) & 0x3FFu;
  if (e == 0) {
    if (m == 0) return std::bit_cast<float>(s);
    // subnormal: ±(m * 2^-24), exactly representable in float. The sign bit must
    // be preserved (0x33800000 | s is +2^-24 for sign=0, -2^-24 for sign=1).
    return std::bit_cast<float>(0x33800000u | s) * float(m);
  }
  if (e == 0x1Fu) return std::bit_cast<float>(s | 0x7F800000u | (m << 13));
  return std::bit_cast<float>(s | ((e + 112) << 23) | (m << 13));
}

typedef struct {
  uint8_t hmask[GGML_BASELINE_QK_K / 8];
  uint8_t qs[GGML_BASELINE_QK_K / 4];
  uint8_t scales[12];
  ggml_half d;
} block_q3_K;

typedef struct {
  ggml_half d;
  ggml_half dmin;
  uint8_t scales[GGML_BASELINE_K_SCALE_SIZE];
  uint8_t qs[GGML_BASELINE_QK_K / 2];
} block_q4_K;

typedef struct {
  ggml_half d;
  ggml_half dmin;
  uint8_t scales[GGML_BASELINE_K_SCALE_SIZE];
  uint8_t qh[GGML_BASELINE_QK_K / 8];
  uint8_t qs[GGML_BASELINE_QK_K / 2];
} block_q5_K;

typedef struct {
  uint8_t ql[128];
  uint8_t qh[64];
  int8_t scales[16];
  ggml_half d;
} block_q6_K;

static_assert(sizeof(block_q3_K) == 110, "q3_K block size");
static_assert(sizeof(block_q4_K) == 144, "q4_K block size");
static_assert(sizeof(block_q5_K) == 176, "q5_K block size");
static_assert(sizeof(block_q6_K) == 210, "q6_K block size");

typedef struct {
  float d;
  int8_t qs[GGML_BASELINE_QK_K];
  int16_t bsums[GGML_BASELINE_QK_K / 16];
} block_q8_K;

static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

static inline void dequantize_row_q3_K(const block_q3_K* x, float* y, int64_t k) {
  const int nb = k / GGML_BASELINE_QK_K;
  const uint32_t kmask1 = 0x03030303;
  const uint32_t kmask2 = 0x0f0f0f0f;
  uint32_t aux[4];
  const int8_t* scales = (const int8_t*)aux;

  for (int i = 0; i < nb; i++) {
    const float d_all = baseline_half_to_float(x[i].d);
    const uint8_t* q = x[i].qs;
    const uint8_t* hm = x[i].hmask;
    uint8_t m = 1;

    // memcpy-style copy of the 12 scale bytes (verbatim ggml bit-twiddling).
    aux[0] = aux[1] = aux[2] = aux[3] = 0;
    const uint8_t* sc = x[i].scales;
    for (int b = 0; b < 12; ++b) ((uint8_t*)aux)[b] = sc[b];
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int is = 0;
    float dl;
    for (int n = 0; n < GGML_BASELINE_QK_K; n += 128) {
      int shift = 0;
      for (int j = 0; j < 4; ++j) {
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * ((int8_t)((q[l + 0] >> shift) & 3) - ((hm[l + 0] & m) ? 0 : 4));
        }
        dl = d_all * (scales[is++] - 32);
        for (int l = 0; l < 16; ++l) {
          *y++ = dl * ((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
        }
        shift += 2;
        m <<= 1;
      }
      q += 32;
    }
  }
}

static inline void dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k) {
  const int nb = k / GGML_BASELINE_QK_K;
  for (int i = 0; i < nb; i++) {
    const uint8_t* q = x[i].qs;
    const float d = baseline_half_to_float(x[i].d);
    const float min = baseline_half_to_float(x[i].dmin);
    int is = 0;
    uint8_t sc, m;
    for (int j = 0; j < GGML_BASELINE_QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
      q += 32;
      is += 2;
    }
  }
}

static inline void dequantize_row_q5_K(const block_q5_K* x, float* y, int64_t k) {
  const int nb = k / GGML_BASELINE_QK_K;
  for (int i = 0; i < nb; i++) {
    const uint8_t* ql = x[i].qs;
    const uint8_t* qh = x[i].qh;
    const float d = baseline_half_to_float(x[i].d);
    const float min = baseline_half_to_float(x[i].dmin);
    int is = 0;
    uint8_t sc, m;
    uint8_t u1 = 1, u2 = 2;
    for (int j = 0; j < GGML_BASELINE_QK_K; j += 64) {
      get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
      const float d1 = d * sc;
      const float m1 = min * m;
      get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
      const float d2 = d * sc;
      const float m2 = min * m;
      for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
      for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
      ql += 32;
      is += 2;
      u1 <<= 2;
      u2 <<= 2;
    }
  }
}

static inline void dequantize_row_q6_K(const block_q6_K* x, float* y, int64_t k) {
  const int nb = k / GGML_BASELINE_QK_K;
  for (int i = 0; i < nb; ++i) {
    const float d = baseline_half_to_float(x[i].d);
    const uint8_t* ql = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* sc = x[i].scales;
    for (int n = 0; n < GGML_BASELINE_QK_K; n += 128) {
      for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int8_t q1 = int8_t((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        const int8_t q2 = int8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        const int8_t q3 = int8_t((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        const int8_t q4 = int8_t((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
        y[l] = d * sc[is + 0] * q1;
        y[l + 32] = d * sc[is + 2] * q2;
        y[l + 64] = d * sc[is + 4] * q3;
        y[l + 96] = d * sc[is + 6] * q4;
      }
      y += 128;
      ql += 64;
      qh += 32;
      sc += 8;
    }
  }
}

// Legacy 32-element formats (Q5_0 / Q8_0), ported from the same sources:
//   ggml/src/ggml-common.h  — block_q5_0 / block_q8_0, QK5_0, QK8_0
//   ggml/src/ggml-quants.c  — dequantize_row_q5_0 / _q8_0
#define GGML_BASELINE_QK5_0 32
#define GGML_BASELINE_QK8_0 32

typedef struct {
  uint16_t d;
  uint8_t qh[4];
  uint8_t qs[GGML_BASELINE_QK5_0 / 2];
} block_q5_0;
static_assert(sizeof(block_q5_0) == 2 + 4 + GGML_BASELINE_QK5_0 / 2, "q5_0 block size");

typedef struct {
  uint16_t d;
  int8_t qs[GGML_BASELINE_QK8_0];
} block_q8_0;
static_assert(sizeof(block_q8_0) == 2 + GGML_BASELINE_QK8_0, "q8_0 block size");

static inline void dequantize_row_q5_0(const block_q5_0* x, float* y, int64_t k) {
  const int nb = int(k / GGML_BASELINE_QK5_0);
  for (int i = 0; i < nb; ++i) {
    const float d = baseline_half_to_float(x[i].d);
    uint32_t qh;
    memcpy(&qh, x[i].qh, sizeof(qh));
    for (int j = 0; j < GGML_BASELINE_QK5_0 / 2; ++j) {
      const uint8_t xh_0 = uint8_t(((qh >> j) << 4) & 0x10);
      const uint8_t xh_1 = uint8_t((qh >> (j + 12)) & 0x10);
      const int32_t x0 = int32_t((x[i].qs[j] & 0x0F) | xh_0) - 16;
      const int32_t x1 = int32_t((x[i].qs[j] >> 4) | xh_1) - 16;
      y[j + 0] = x0 * d;
      y[j + GGML_BASELINE_QK5_0 / 2] = x1 * d;
    }
    y += GGML_BASELINE_QK5_0;
  }
}

static inline void dequantize_row_q8_0(const block_q8_0* x, float* y, int64_t k) {
  const int nb = int(k / GGML_BASELINE_QK8_0);
  for (int i = 0; i < nb; ++i) {
    const float d = baseline_half_to_float(x[i].d);
    for (int j = 0; j < GGML_BASELINE_QK8_0; ++j) y[j] = x[i].qs[j] * d;
    y += GGML_BASELINE_QK8_0;
  }
}

static inline void quantize_row_q8_K_ref(const float* x, block_q8_K* y, int64_t k) {
  const int nb = int(k / GGML_BASELINE_QK_K);
  for (int b = 0; b < nb; ++b) {
    float amax = 0.0f, signed_max = 0.0f;
    for (int j = 0; j < GGML_BASELINE_QK_K; ++j) {
      const float ax = std::fabs(x[b * GGML_BASELINE_QK_K + j]);
      if (ax > amax) { amax = ax; signed_max = x[b * GGML_BASELINE_QK_K + j]; }
    }
    const float iscale = amax == 0.0f ? 0.0f : -127.0f / signed_max;
    y[b].d = iscale == 0.0f ? 0.0f : 1.0f / iscale;
    for (int j = 0; j < GGML_BASELINE_QK_K; ++j) {
      const float v = iscale * x[b * GGML_BASELINE_QK_K + j] + 12582912.0f;
      const int q = iscale == 0.0f ? 0 : (int(std::bit_cast<uint32_t>(v) & 0x007fffffu) - 0x00400000);
      y[b].qs[j] = int8_t(q);
    }
    for (int j = 0; j < GGML_BASELINE_QK_K / 16; ++j) {
      int sum = 0;
      for (int l = 0; l < 16; ++l) sum += y[b].qs[j * 16 + l];
      y[b].bsums[j] = int16_t(sum);
    }
  }
}

static inline float vec_dot_q6_K_q8_K_generic(const block_q6_K* x,
                                               const block_q8_K* y,
                                               int64_t n) {
  const int nb = int(n / GGML_BASELINE_QK_K);
  float sums[8] = {};
  float sumf = 0.0f;
  for (int i = 0; i < nb; ++i) {
    int8_t aux8[GGML_BASELINE_QK_K];
    int32_t aux32[8] = {};
    const uint8_t* ql = x[i].ql;
    const uint8_t* qh = x[i].qh;
    const int8_t* q8 = y[i].qs;
    int8_t* a = aux8;
    for (int j = 0; j < GGML_BASELINE_QK_K; j += 128) {
      for (int l = 0; l < 32; ++l) {
        a[l] = int8_t((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
        a[l + 32] = int8_t((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
        a[l + 64] = int8_t((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
        a[l + 96] = int8_t((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
      }
      a += 128; ql += 64; qh += 32;
    }
    a = aux8;
    int is = 0;
    for (int j = 0; j < GGML_BASELINE_QK_K / 16; ++j) {
      const int scale = x[i].scales[is++];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * int16_t(q8[l] * a[l]);
      q8 += 8; a += 8;
      for (int l = 0; l < 8; ++l) aux32[l] += scale * int16_t(q8[l] * a[l]);
      q8 += 8; a += 8;
    }
    const float d = baseline_half_to_float(x[i].d) * y[i].d;
    for (int l = 0; l < 8; ++l) sums[l] += d * float(aux32[l]);
  }
  for (float v : sums) sumf += v;
  return sumf;
}

} // namespace ggml_baseline
