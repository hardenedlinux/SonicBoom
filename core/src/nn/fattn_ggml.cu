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

// Adapter for the imported ggml flash-attention kernel.
//
// The compute kernel is ggml's flash_attn_ext_f16 (fattn-mma-f16.cuh, MIT,
// imported under quant/ggml_derived/fattn/). It is selected explicitly with
// SONICBOOM_ATTN_IMPL=ggml and is never chosen by "auto"; the native
// (own/codegen) path is the default and the fallback for every shape this
// adapter does not cover. Removing the backend means deleting this file, the
// four headers under quant/ggml_derived/fattn/ and the three references in
// core/CMakeLists.txt, cuda_elementwise.cu and cuda_resident.h.
//
// It expects:
//   - Q as f32, [query][head][head_dim] with query stride nb01 and head stride
//     nb02 (its own dst layout is the same [query][head][head_dim]),
//   - K and V as f16, [head][key][head_dim] with key stride nb11 and head
//     stride nb12 (ne11_pad keys per head so whole key tiles stay in bounds),
//   - an additive f16 mask (0 or -inf) [query][ne11_pad], stride nb31,
//   - an optional per-query-tile key count (KV_max) that bounds the K scan.
//
// SonicBoom's cache is f32 (strict profile) or f16 (standard profile), a
// chunk's K/V live in a separate f32 batch buffer from the prior-context cache,
// and the causal + sliding-window mask is implicit. This file materializes the
// three inputs
// the imported kernel needs with fixed-grid layout kernels (gather/cast,
// additive mask, per-tile key count) and then launches the imported kernel
// directly. Those layout kernels are data movement, not attention math: the
// scores/softmax/PV arithmetic all happens in the imported kernel.
//
// Everything is sized from (n, n_heads, head_dim, window, n_slots), never from
// the start position: the start pointer stays a device value and the kernels
// derive the actual key range in-kernel, so the ggml path remains capturable
// into a CUDA Graph exactly like the native tiled kernel.
//
// Performance note: the K/V materialization (and, under the strict profile,
// the f32->f16 cast) is the price of reusing a kernel whose contract is f16
// KV. It is accounted for in the benchmark; the native path keeps its
// zero-copy cache. The q8 KV storage has no f16 analogue here, so that arm
// declines and the native path runs.

#include "cuda_resident.h"

#include "cuda_stream.h"

#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)
#include "../quant/ggml_derived/fattn/fattn-mma-f16.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace sonicboom::nn::cuda {
namespace {

// Scratch buffers, reused across layers. Reallocation zeroes them once: rows
// beyond the actual key span are masked out (-inf) by the mask, but their K/V
// bytes must still be finite (a stale NaN would turn 0 * NaN into NaN).
void* g_kv_half = nullptr;
size_t g_kv_half_bytes = 0;
void* g_mask = nullptr;
size_t g_mask_bytes = 0;
void* g_kvmax = nullptr;
size_t g_kvmax_bytes = 0;

void* grow_zeroed(void*& buf, size_t& have, size_t need) {
  if (have >= need) return buf;
  if (buf) cudaFree(buf);
  buf = nullptr;
  have = 0;
  if (cudaMalloc(&buf, need) != cudaSuccess) return nullptr;
  if (cudaMemsetAsync(buf, 0, need, sonicboom::cuda::sb_current_stream()) !=
      cudaSuccess) {
    cudaFree(buf);
    buf = nullptr;
    return nullptr;
  }
  have = need;
  return buf;
}

int device_cc() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0;
  // cudaGetDeviceProperties is not free (it is a driver query, ~1.5 ms on
  // driver 595 even with a live context), and this runs once per attention
  // call -- several thousand times in a long prefill. The compute capability
  // of a device never changes, so query it once per device index.
  static int cache[16] = {};  // 0 = not queried; a valid cc is > 0
  if (dev >= 0 && dev < 16 && cache[dev] > 0) return cache[dev];
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 0;
  const int cc = prop.major * 100 + prop.minor * 10;
  if (dev >= 0 && dev < 16) cache[dev] = cc;
  return cc;
}

// SONICBOOM_ATTN_DEBUG=1 reports the first failing stage on stderr; silent
// otherwise (the caller falls back to the native path either way).
void debug_check(const char* stage, cudaStream_t stream) {
  static const bool on = std::getenv("SONICBOOM_ATTN_DEBUG") != nullptr;
  if (!on) return;
  // The kernels run on the runtime's non-default stream, so a launch error is
  // only observable asynchronously; synchronize to attribute it to a stage.
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess)
    std::fprintf(stderr, "[fattn-ggml] %s: %s\n", stage,
                 cudaGetErrorString(err));
}

// Base key (absolute) of relative index 0: the earliest key any query in the
// segment can attend, aligned down to the kernel's key-block size nbatch_fa.
// The alignment is what makes the split batch-invariant: a key's (block, slot
// within block) position must depend only on its absolute index, never on
// where the segment starts. With base aligned, a reused prefix and a full
// prefill group every key into the same online-softmax block, so the
// accumulation order (and therefore the result) is bit-identical. Unaligned,
// base = start-(window-1) moves the block boundaries with `start` and the
// rounding differs run to run.
__device__ __forceinline__ uint64_t span_base(uint64_t start, uint64_t window,
                                              int nbatch_fa) {
  uint64_t base =
      (window > 0 && start >= window - 1) ? start - (window - 1) : 0;
  if (nbatch_fa > 1) base -= base % (uint64_t)nbatch_fa;
  return base;
}

// A cache element as f32. Under the standard profile the stored f16 is the
// same value the strict profile would hold (K/V are fp16-rounded before
// either storage exists), so widening is exact.
__device__ __forceinline__ float ggml_cache_load(const float* p) { return *p; }
__device__ __forceinline__ float ggml_cache_load(const __half* p) {
  return __half2float(*p);
}

// Materialize K/V for relative keys [0, ne11) as f16, laid out
// [head][ne11_pad][head_dim]. Keys before `start` come from the ring cache
// (slot = key % n_slots), keys in [start, start+n) from the chunk's batch
// buffer; keys outside the chunk are left untouched (their mask is -inf and
// KV_max stops the scan before them). CACHE is the cache element type: float
// under the strict profile, __half under the standard profile. Batch rows are
// f32 in both and are rounded to f16 exactly as the KV append rounds them.
template <typename CACHE>
__global__ void fattn_ggml_gather_kernel(
    const float* __restrict__ k_batch, const float* __restrict__ v_batch,
    const CACHE* __restrict__ k_cache, const CACHE* __restrict__ v_cache,
    const uint64_t* __restrict__ start_ptr, uint64_t n_slots, int n,
    int n_heads_kv, int head_dim, int ne11, int ne11_pad, uint64_t window,
    int nbatch_fa, __half* __restrict__ k_half,
    __half* __restrict__ v_half) {
  const uint64_t start = *start_ptr;
  const uint64_t key_end = start + (uint64_t)n;
  const uint64_t base = span_base(start, window, nbatch_fa);
  const int64_t units = (int64_t)ne11 * n_heads_kv;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;
  for (int64_t u = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; u < units;
       u += stride) {
    const int rel = int(u / n_heads_kv);
    const int h = int(u % n_heads_kv);
    const uint64_t abs = base + (uint64_t)rel;
    if (abs >= key_end) continue;
    const int64_t dst_off = ((int64_t)h * ne11_pad + rel) * head_dim;
    const bool cached = abs < start;
    const uint64_t slot = cached ? abs % n_slots : 0;
    const uint64_t bi = cached ? 0 : abs - start;
    const uint64_t row = (cached ? slot : bi) * (uint64_t)n_heads_kv + h;

    __half* dst = k_half + dst_off;
    if (cached) {
      const CACHE* src = k_cache + row * head_dim;
      for (int d = 0; d < head_dim; ++d)
        dst[d] = __float2half_rn(ggml_cache_load(src + d));
    } else {
      const float* src = k_batch + row * head_dim;
      for (int d = 0; d < head_dim; ++d) dst[d] = __float2half_rn(src[d]);
    }

    dst = v_half + dst_off;
    if (cached) {
      const CACHE* src = v_cache + row * head_dim;
      for (int d = 0; d < head_dim; ++d)
        dst[d] = __float2half_rn(ggml_cache_load(src + d));
    } else {
      const float* src = v_batch + row * head_dim;
      for (int d = 0; d < head_dim; ++d) dst[d] = __float2half_rn(src[d]);
    }
  }
}

// Additive causal + sliding-window mask, [n_q, ne11_pad] f16, 0 where the key
// is visible and -inf otherwise. Padding columns are -inf by construction.
__global__ void fattn_ggml_mask_kernel(const uint64_t* __restrict__ start_ptr,
                                       int n_q, int ne11_pad,
                                       uint64_t window, int nbatch_fa,
                                       __half* __restrict__ mask) {
  const uint64_t start = *start_ptr;
  const uint64_t base = span_base(start, window, nbatch_fa);
  const int64_t total = (int64_t)n_q * ne11_pad;
  const int64_t stride = (int64_t)blockDim.x * gridDim.x;
  const __half zero = __float2half_rn(0.0f);
  const __half neg_inf = __float2half_rn(-INFINITY);
  for (int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; i < total;
       i += stride) {
    const int row = int(i / ne11_pad);
    const int rel = int(i % ne11_pad);
    const uint64_t p = start + (uint64_t)row;
    const uint64_t abs = base + (uint64_t)rel;
    const bool visible = abs <= p && (window == 0 || p - abs < window);
    mask[i] = visible ? zero : neg_inf;
  }
}

// Per-query-tile key count, rounded up to the kernel's key-block size. The
// kernel divides it by nbatch_fa to clamp its K loop; it is the one host-sized
// input whose value depends on `start`, so the kernel fills it on device.
__global__ void fattn_ggml_kvmax_kernel(int* __restrict__ kvmax,
                                        const uint64_t* __restrict__ start_ptr,
                                        int n_q, int ncols1, int ne11,
                                        int nbatch_fa, uint64_t window) {
  const int jt = blockIdx.x * blockDim.x + threadIdx.x;
  const int iter_j = (n_q + ncols1 - 1) / ncols1;
  if (jt >= iter_j) return;
  const uint64_t start = *start_ptr;
  const uint64_t base = span_base(start, window, nbatch_fa);
  const int rows = n_q - jt * ncols1;
  const int last = rows < ncols1 ? rows - 1 : ncols1 - 1;
  const uint64_t p_last = start + (uint64_t)(jt * ncols1 + last);
  const uint64_t need = p_last >= base ? p_last - base + 1 : 0;
  int blocks = (int)((need + (uint64_t)nbatch_fa - 1) / (uint64_t)nbatch_fa);
  const int total = (ne11 + nbatch_fa - 1) / nbatch_fa;
  if (blocks > total) blocks = total;
  if (blocks < 1) blocks = 1;  // kb0_start < kb0_stop is a kernel invariant
  kvmax[jt] = blocks * nbatch_fa;
}

template <int DKQ, int DV, int ncols1, int ncols2>
bool launch_ggml_attn(const float* q, const __half* k_half,
                      const __half* v_half, const __half* mask,
                      const int* kvmax, int n_q, int n_heads_q,
                      int n_heads_kv, int ne11, int ne11_pad, int cc,
                      float scale, float* out) {
  constexpr int ncols = ncols1 * ncols2;
  constexpr bool V_is_K_view = false;

  const fattn_mma_config cfg =
      ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc);
  if (cfg.nthreads <= 0 || cfg.nbatch_fa <= 0) return false;
  const int nthreads = cfg.nthreads;
  // The kernel indexes threads as (lane, warp): threadIdx.x is the lane and
  // threadIdx.y the warp, so the block must be 2D. A flat <nthreads> launch
  // leaves threadIdx.y == 0 and every warp's tile work lands in warp 0's
  // slots (deterministically wrong output, not a crash).
  const dim3 block_dim(WARP_SIZE, nthreads / WARP_SIZE, 1);
  const int nwarps = nthreads / WARP_SIZE;
  const int nstages =
      ggml_cuda_fattn_mma_get_nstages(DKQ, DV, ncols1, ncols2, cc);
  const bool swz = ggml_cuda_fattn_mma_get_swizzled(DKQ, DV, ncols, cc);
  const int stride_tile_K = swz ? cfg.nbatch_K2 : cfg.nbatch_K2 + 4;
  const int stride_tile_V = V_is_K_view
                                ? stride_tile_K
                                : (swz ? cfg.nbatch_V2 : cfg.nbatch_V2 + 4);
  const int cols_per_warp = std::min(ncols, get_cols_per_warp(cc));

  // Same shared-memory accounting as ggml's mma-f16 case launcher.
  const size_t nbytes_shared_KV_1stage =
      (size_t)cfg.nbatch_fa * std::max(stride_tile_K, stride_tile_V) *
      sizeof(half2);
  const size_t nbytes_shared_KV_2stage =
      (size_t)cfg.nbatch_fa * (stride_tile_K + stride_tile_V) * sizeof(half2);
  const size_t nbytes_shared_Q = (size_t)ncols * (DKQ / 2 + 4) * sizeof(half2);
  const size_t nbytes_shared_mask =
      (size_t)ncols1 * (cfg.nbatch_fa / 2 + 4) * sizeof(half2);
  const size_t nbytes_shared_combine =
      (size_t)nwarps * cols_per_warp * (cfg.nbatch_combine + 4) *
      sizeof(half2);
  const size_t nbytes_shared_KV =
      nstages <= 1 ? nbytes_shared_KV_1stage : nbytes_shared_KV_2stage;
  const size_t nbytes_shared_total = std::max(
      nbytes_shared_combine,
      cfg.Q_in_reg ? std::max(nbytes_shared_Q,
                              nbytes_shared_KV + nbytes_shared_mask)
                   : nbytes_shared_Q + nbytes_shared_KV + nbytes_shared_mask);

  auto kfn = flash_attn_ext_f16<DKQ, DV, ncols1, ncols2, false, V_is_K_view,
                                false>;
  static bool smem_configured = false;
  if (!smem_configured) {
    if (cudaFuncSetAttribute((const void*)kfn,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)nbytes_shared_total) != cudaSuccess)
      return false;
    smem_configured = true;
  }

  const int gqa_ratio = n_heads_q / n_heads_kv;
  const int iter_j = (n_q + ncols1 - 1) / ncols1;
  const int iter_z_gqa = (gqa_ratio + ncols2 - 1) / ncols2;
  const int tiles = iter_j * iter_z_gqa * n_heads_kv;

  const uint3 ne01 = init_fastdiv_values((uint64_t)n_q);
  const int nb01 = n_heads_q * DKQ * (int)sizeof(float);
  const int nb02 = DKQ * (int)sizeof(float);
  const int nb03 = n_q * nb01;
  // K/V are [head][ne11_pad][D]: key stride D, head stride ne11_pad*D.
  const int nb11 = DKQ * (int)sizeof(__half);
  const int nb12 = ne11_pad * nb11;
  const int64_t nb13 = (int64_t)n_heads_kv * nb12;

  const float max_bias = 0.0f;
  const float m0 = 1.0f, m1 = 1.0f;
  const uint32_t n_head_log2 = 1u;
  cudaStream_t stream = sonicboom::cuda::sb_current_stream();

  if (std::getenv("SONICBOOM_ATTN_DEBUG")) {
    std::fprintf(stderr,
                 "[fattn-ggml] launch DKQ=%d DV=%d ncols1=%d ncols2=%d n_q=%d "
                 "hq=%d hkv=%d ne11=%d pad=%d cc=%d tiles=%d threads=%d "
                 "smem=%zu nstages=%d Q_in_reg=%d\n",
                 DKQ, DV, ncols1, ncols2, n_q, n_heads_q, n_heads_kv, ne11,
                 ne11_pad, cc, tiles, nthreads, nbytes_shared_total, nstages,
                 (int)cfg.Q_in_reg);
  }

  SB_NOTE_LAUNCH_TP("flash_attention_ggml_kernel",
                    ::sonicboom::planner::plan_detail::tp_label(
                        {{"DKQ", DKQ}, {"DV", DV},
                         {"ncols1", ncols1}, {"ncols2", ncols2}}),
                    tiles, nthreads, nbytes_shared_total, stream);

  kfn<<<tiles, block_dim, nbytes_shared_total, stream>>>(
      (const char*)q, (const char*)k_half, (const char*)v_half,
      (const char*)mask, nullptr, kvmax, out, nullptr, scale, max_bias, m0, m1,
      n_head_log2, 0.0f,
      DKQ, ne01, n_heads_q, 1, nb01, nb02, nb03,
      DKQ, ne11, n_heads_kv, 1, nb11, nb12, nb13,
      nb11, nb12, nb13,
      ne11, 1, 1, ne11_pad * (int)sizeof(__half),
      ne11_pad * (int)sizeof(__half), (int64_t)n_q * ne11_pad * 2);
  debug_check("flash_attn_ext_f16", stream);
  return cudaGetLastError() == cudaSuccess;
}

template <int DKQ, int DV, int ncols1, int ncols2>
bool run_ggml_chunk(const float* q, const float* k_batch,
                    const float* v_batch, const float* k_cache,
                    const float* v_cache, const uint64_t* start,
                    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv,
                    uint64_t sliding_window, float scale, float* out,
                    bool kv_f16) {
  constexpr int ncols = ncols1 * ncols2;
  const int cc = device_cc();
  if (cc <= 0) return false;
  const fattn_mma_config cfg =
      ggml_cuda_fattn_mma_get_config(DKQ, DV, ncols, cc);
  if (cfg.nbatch_fa <= 0) return false;

  const int nbatch_fa = cfg.nbatch_fa;
  // Relative-key span. window > 0: window-1 + n, plus up to nbatch_fa-1 keys
  // of alignment slop below the unaligned span base (see span_base); global:
  // n_slots + n (the largest start is n_slots-1, queries see [0, start+n)).
  const int ne11 =
      (int)((sliding_window > 0 ? sliding_window - 1 + (uint64_t)(nbatch_fa - 1)
                                : n_slots) + (uint64_t)n);
  const int ne11_pad = (ne11 + nbatch_fa - 1) / nbatch_fa * nbatch_fa;
  const int iter_j = (n + ncols1 - 1) / ncols1;

  const size_t kv_elems = (size_t)ne11_pad * n_heads_kv * DKQ;
  const size_t half_bytes = kv_elems * sizeof(__half);
  void* kv = grow_zeroed(g_kv_half, g_kv_half_bytes, half_bytes * 2);
  if (!kv) return false;
  __half* k_half = static_cast<__half*>(kv);
  __half* v_half = k_half + kv_elems;

  const size_t mask_bytes = (size_t)n * ne11_pad * sizeof(__half);
  void* mask = grow_zeroed(g_mask, g_mask_bytes, mask_bytes);
  if (!mask) return false;

  const size_t kvmax_bytes = (size_t)iter_j * sizeof(int);
  void* kvmax = grow_zeroed(g_kvmax, g_kvmax_bytes, kvmax_bytes);
  if (!kvmax) return false;

  cudaStream_t stream = sonicboom::cuda::sb_current_stream();

  const int gather_units = ne11 * n_heads_kv;
  const int gather_blocks = gather_units < 1024 ? gather_units : 1024;
  if (kv_f16) {
    SB_NOTE_LAUNCH_TP("fattn_ggml_gather_kernel", "cache=f16", gather_blocks,
                      256, 0, stream);
    fattn_ggml_gather_kernel<__half><<<gather_blocks, 256, 0, stream>>>(
        k_batch, v_batch, reinterpret_cast<const __half*>(k_cache),
        reinterpret_cast<const __half*>(v_cache), start, n_slots, n, n_heads_kv,
        DKQ, ne11, ne11_pad, sliding_window, nbatch_fa, k_half, v_half);
  } else {
    SB_NOTE_LAUNCH_TP("fattn_ggml_gather_kernel", "cache=f32", gather_blocks,
                      256, 0, stream);
    fattn_ggml_gather_kernel<float><<<gather_blocks, 256, 0, stream>>>(
        k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_kv, DKQ,
        ne11, ne11_pad, sliding_window, nbatch_fa, k_half, v_half);
  }
  debug_check("gather", stream);

  const int mask_total = n * ne11_pad;
  const int mask_blocks =
      (mask_total + 255) / 256 < 2048 ? (mask_total + 255) / 256 : 2048;
  SB_NOTE_LAUNCH(fattn_ggml_mask_kernel, mask_blocks, 256, 0, stream);
  fattn_ggml_mask_kernel<<<mask_blocks, 256, 0, stream>>>(
      start, n, ne11_pad, sliding_window, nbatch_fa,
      static_cast<__half*>(mask));
  debug_check("mask", stream);

  const int kvmax_blocks = (iter_j + 31) / 32;
  SB_NOTE_LAUNCH(fattn_ggml_kvmax_kernel, kvmax_blocks, 32, 0, stream);
  fattn_ggml_kvmax_kernel<<<kvmax_blocks, 32, 0, stream>>>(
      static_cast<int*>(kvmax), start, n, ncols1, ne11, nbatch_fa,
      sliding_window);
  debug_check("kvmax", stream);
  if (cudaGetLastError() != cudaSuccess) return false;

  return launch_ggml_attn<DKQ, DV, ncols1, ncols2>(
      q, k_half, v_half, static_cast<const __half*>(mask),
      static_cast<const int*>(kvmax), n, n_heads_q, n_heads_kv, ne11, ne11_pad,
      cc, scale, out);
}

}  // namespace

bool flash_attention_ggml_chunk_dev_pos(
    const float* q, const float* k_batch, const float* v_batch,
    const float* k_cache, const float* v_cache, const uint64_t* start,
    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* out, bool kv_f16,
    bool kv_q8) {
  if (n <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 || head_dim <= 0 ||
      n_slots == 0)
    return false;
  if (!q || !k_batch || !v_batch || !k_cache || !v_cache || !start || !out)
    return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  // The imported kernel's K/V contract is f16. A q8 cache would have to be
  // dequantized into the same staging buffers; that arm is not ported, so the
  // native kernel serves q8 (the caller falls back).
  if (kv_q8) return false;
  const int gqa = n_heads_q / n_heads_kv;

  // Tile configurations are the pinned llama.cpp commit's dispatch for these
  // shape classes on a Turing+/Ampere device at prefill batch sizes (it picks
  // the ncols2 = 8 GQA group for a ratio > 4, then the widest legal ncols1).
  // A partial last query tile is fine: the imported kernel bounds Q reads and
  // output writes by ne01.z and wraps mask row lookups back into the valid
  // query range, so n need not be a multiple of the tile width.
  //
  // d128/gqa6 is DeepSeek-R1-Distill-Qwen-1.5B; d128/gqa7 is Qwen2.5-7B.
  // d256/gqa4 is Gemma 4 E4B's sliding-window head, d512/gqa4 its global head,
  // d64/gqa7 Qwen2.5-0.5B.
  bool ok = false;
  if (head_dim == 256 && gqa == 4)
    ok = run_ggml_chunk<256, 256, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  else if (head_dim == 512 && gqa == 4)
    ok = run_ggml_chunk<512, 512, 16, 4>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  else if (head_dim == 64 && gqa == 7)
    ok = run_ggml_chunk<64, 64, 8, 8>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  else if (head_dim == 128 && (gqa == 6 || gqa == 7))
    ok = run_ggml_chunk<128, 128, 8, 8>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  return ok;
}

}  // namespace sonicboom::nn::cuda
