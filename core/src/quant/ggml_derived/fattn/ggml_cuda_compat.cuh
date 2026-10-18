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
// Minimal stand-ins for the ggml-cuda "common.cuh" surface used by the
// imported flash-attention kernel (mma.cuh, cp-async.cuh, fattn-mma-f16.cuh).
// The definitions are adapted from ggml (llama.cpp), MIT-licensed, Copyright
// (c) 2023-2026 The ggml authors — specifically the CUDA arch/availability
// macros, NO_DEVICE_CODE, ggml_cuda_memcpy_1, get_alibi_slope,
// ggml_cuda_mad, ggml_cuda_unroll and the fastdiv helpers. ggml's full
// common.cuh is a backend-wide header (streams, pools, ggml tensor machinery)
// that SonicBoom does not link; this file keeps exactly the subset the
// attention kernel needs so the imported code compiles against Layer 2's
// private CUDA sources without pulling in a ggml runtime.
//
// CUDA-only and private to core/src: nothing here is exposed through Layer 2
// or the C API.
#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <algorithm>
#include <limits>

#define WARP_SIZE 32
#define GGML_CUDA_CC_PASCAL      600
#define GGML_CUDA_CC_VOLTA       700
#define GGML_CUDA_CC_TURING      750
#define GGML_CUDA_CC_AMPERE      800
#define GGML_CUDA_CC_ADALOVELACE 890
#define GGML_CUDA_CC_HOPPER      900
#define GGML_CUDA_CC_BLACKWELL  1200
#define GGML_CUDA_CC_RUBIN      1300

#define STRINGIZE_IMPL(...) #__VA_ARGS__
#define STRINGIZE(...) STRINGIZE_IMPL(__VA_ARGS__)

#define GGML_UNUSED(x) (void)(x)
#define GGML_UNUSED_VARS(...) (void)(0 __VA_OPT__(,) __VA_ARGS__)
#define GGML_ASSERT(x)                                                         \
  do {                                                                         \
    if (!(x)) {                                                                \
      fprintf(stderr, "GGML_ASSERT: %s:%d: %s\n", __FILE__, __LINE__, #x);     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define GGML_CUDA_RESTRICT __restrict__

// Arch-derived availability, matching ggml's definitions (there the macros are
// defined per device compilation pass exactly like this).
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
#define FP16_AVAILABLE
#endif
#if defined(FP16_AVAILABLE) && __CUDA_ARCH__ != 610
#define FAST_FP16_AVAILABLE
#endif
#if __CUDA_ARCH__ == GGML_CUDA_CC_VOLTA
#define VOLTA_MMA_AVAILABLE
#endif
#if __CUDA_ARCH__ >= GGML_CUDA_CC_TURING
#define TURING_MMA_AVAILABLE
#endif
#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#define CP_ASYNC_AVAILABLE
#endif
#define FLASH_ATTN_AVAILABLE

static constexpr __device__ int ggml_cuda_get_physical_warp_size() {
  return 32;
}

static constexpr __device__ int ggml_cuda_get_max_cpy_bytes() {
  return 16;
}

[[noreturn]] static __device__ void no_device_code(
    const char* file_name, const int line, const char* function_name,
    const int arch, const char* arch_list) {
  (void)arch_list;
  printf("ERROR: no device code: %s:%d %s arch %d\n", file_name, line,
         function_name, arch);
  __trap();
}

#ifdef __CUDA_ARCH__
#define NO_DEVICE_CODE \
  no_device_code(__FILE__, __LINE__, __FUNCTION__, __CUDA_ARCH__, "")
#else
#define NO_DEVICE_CODE
#endif

static __device__ __forceinline__ void ggml_cuda_pdl_sync() {}

template <int nbytes, int alignment = 0>
static __device__ __forceinline__ void ggml_cuda_memcpy_1(
    void* __restrict__ dst, const void* __restrict__ src) {
  constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;
  static_assert(nbytes % nb_per_cpy == 0, "bad alignment");
#pragma unroll
  for (int i = 0; i < nbytes / nb_per_cpy; ++i) {
    if constexpr (nb_per_cpy == 1) {
      ((char*)dst)[i] = ((const char*)src)[i];
    } else if constexpr (nb_per_cpy == 2) {
      ((short*)dst)[i] = ((const short*)src)[i];
    } else if constexpr (nb_per_cpy == 4) {
      ((int*)dst)[i] = ((const int*)src)[i];
    } else if constexpr (nb_per_cpy == 8) {
      ((int2*)dst)[i] = ((const int2*)src)[i];
    } else if constexpr (nb_per_cpy == 16) {
      ((int4*)dst)[i] = ((const int4*)src)[i];
    } else {
      static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
    }
  }
}

static __device__ __forceinline__ float get_alibi_slope(
    const float max_bias, const uint32_t h, const uint32_t n_head_log2,
    const float m0, const float m1) {
  if (max_bias <= 0.0f) {
    return 1.0f;
  }
  const float base = h < n_head_log2 ? m0 : m1;
  const int exph = h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1;
  return powf(base, exph);
}

static __device__ __forceinline__ void ggml_cuda_mad(half2& acc,
                                                     const half2 v,
                                                     const half2 u) {
#ifdef FAST_FP16_AVAILABLE
  acc += v * u;
#else
  const float2 tmpv = __half22float2(v);
  const float2 tmpu = __half22float2(u);
  float2 tmpacc = __half22float2(acc);
  tmpacc.x += tmpv.x * tmpu.x;
  tmpacc.y += tmpv.y * tmpu.y;
  acc = make_half2(tmpacc.x, tmpacc.y);
#endif
}

static const uint3 init_fastdiv_values(uint64_t d_64) {
  uint32_t d = (uint32_t)d_64;
  uint32_t L = 0;
  while (L < 32 && (uint32_t{1} << L) < d) {
    L++;
  }
  uint32_t mp =
      (uint32_t)((uint64_t{1} << 32) * ((uint64_t{1} << L) - d) / d + 1);
  return make_uint3(mp, L, d);
}

static __device__ __forceinline__ uint32_t fastdiv(uint32_t n,
                                                   const uint3 fd) {
  const uint32_t hi = __umulhi(n, fd.x);
  return (hi + n) >> fd.y;
}

static __device__ __forceinline__ uint32_t fastmodulo(uint32_t n,
                                                      const uint3 fd) {
  return n - fastdiv(n, fd) * fd.z;
}

// Host-side availability queries used by the config tables. The GPU the build
// targets is the one the library runs on (single-device runtime); the query is
// a pure compute-capability comparison.
enum ggml_type : int {
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_MXFP4 = 39,
  GGML_TYPE_NVFP4 = 40,
};

static bool fp16_available(const int cc) { return cc >= GGML_CUDA_CC_PASCAL; }
static bool fast_fp16_available(const int cc) {
  return cc >= GGML_CUDA_CC_PASCAL && cc != 610;
}
static bool fp16_mma_hardware_available(const int cc) {
  return cc >= GGML_CUDA_CC_VOLTA;
}
static bool bf16_mma_hardware_available(const int cc) {
  return cc >= GGML_CUDA_CC_AMPERE;
}
static bool amd_mfma_available(const int) { return false; }
static bool amd_wmma_available(const int) { return false; }
static bool volta_mma_available(const int cc) {
  return cc == GGML_CUDA_CC_VOLTA;
}
static bool turing_mma_available(const int cc) {
  return cc >= GGML_CUDA_CC_TURING;
}
static bool ampere_mma_available(const int cc) {
  return cc >= GGML_CUDA_CC_AMPERE;
}
static bool cp_async_available(const int cc) {
  return cc >= GGML_CUDA_CC_AMPERE;
}
static bool blackwell_mma_available(const int cc) {
  return cc >= GGML_CUDA_CC_BLACKWELL;
}

template <int n>
struct ggml_cuda_unroll {
  template <typename Func, typename... Args>
  __device__ void operator()(const Func& f, Args... args) const {
    f(n - 1, args...);
    ggml_cuda_unroll<n - 1>{}(f, args...);
  }
};

template <>
struct ggml_cuda_unroll<1> {
  template <typename Func, typename... Args>
  __device__ void operator()(const Func& f, Args... args) const {
    f(0, args...);
  }
};

// Constants from ggml's fattn-common.cuh that the imported kernel uses.
#define FATTN_KQ_STRIDE 256
#define SOFTMAX_FTZ_THRESHOLD -20.0f
#define FATTN_KQ_MAX_OFFSET (3.0f * 0.6931f)
