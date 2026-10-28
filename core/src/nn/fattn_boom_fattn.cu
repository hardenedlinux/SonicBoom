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

// SonicBoom "GGML-inspired" chunked-prefill flash attention (`boom_fattn`).
//
// This is SonicBoom's own kernel: the algorithm was informed by the prefill
// path of the pinned llama.cpp commit (per-key f16 KV materialization, tiled
// MMA attention with an online softmax, GQA query grouping, cp.async staging,
// fp16 VKQ accumulation), but no ggml source is copied and the kernel does not
// call any ggml/llama.cpp code. It is a third, independent backend next to the
// native tiled/scalar kernels and the imported ggml kernel:
//
//   SONICBOOM_ATTN_IMPL=boom_fattn   (this file; "auto" never selects it)
//   SONICBOOM_ATTN_PREFILL=boom_fattn (prefill leg only)
//
// boom_fattn is SonicBoom's own implementation and its only name: selection
// values, configuration switches, kernel symbols and trace strings all use it.
//
// q8 KV storage and shapes without an instantiation decline (return false) and
// the caller falls through to the native path. Removing the backend is
// deleting this file, its declaration in cuda_resident.h, the three-line
// dispatch block in cuda_elementwise.cu and this file's line in
// core/CMakeLists.txt.
//
// Structure, and where it differs from the imported ggml kernel:
//
//   1. A gather kernel materializes the KV span this call needs once, as
//      [kv_head][key][dim] f16 for K and [kv_head][dim][key] f16 for V
//      (transposed, so the PV B-fragments are contiguous key runs). Reads of
//      the f32 batch buffer and of the f32/f16 cache are vectorized, and the
//      transpose is done through one 32x64 shared-memory tile per block. This
//      replaces ggml's scalar per-element cast and its large f16 additive mask.
//   2. The attention kernel derives the causal + sliding-window mask in
//      registers from the device start position, so there is no mask buffer
//      and no KV_max buffer, and the launch shape never depends on `start`
//      (CUDA-Graph capturable exactly like the native kernel).
//   3. Query tiles are 32 rows with all GQA heads of the KV head in one block
//      (ggml's prefill tiles are 16 queries), so the K/V traffic per key tile
//      is amortized over 32 query rows. A block is UNITS*NW = 32 warps (1024
//      threads); each unit is 16 query rows of one GQA group, split across NW
//      dim slices for the PV phase. Phase 1 (S and online softmax) runs on the
//      slice-0 warps and hands P (f16) and the per-sub-tile rescale factors
//      (f32) to the other slices through shared memory, as the native tiled
//      kernel does. This is the kernel selected by SONICBOOM_BOOM_FATTN_QREG=0;
//      its phase 1 is the serial part of the block and is why the imported
//      ggml kernel wins at long context. The d512 default instead uses the
//      query-resident variant below (mode 1), which splits phase 1 over the
//      unit's dim slices. The d256 default (the Gemma sliding-window head)
//      uses that variant too (SONICBOOM_BOOM_FATTN_D256=1): 16-key tiles
//      re-read Q twice as often as 32-key tiles, and phase 1 on 8 of 32 warps
//      was the prefill bottleneck. The variant is window-agnostic; the window
//      itself comes only from the per-layer metadata and may be 0 here.
//   4. K and V staging is cp.async double-buffered per key tile; the V tile is
//      padded to a 16-byte-aligned stride (BK+8 halves) so every cp.async is a
//      16-byte transfer.
//   5. Q is materialized as f16 once per call (a linear cast kernel) and read
//      from there by phase 1: the same __float2half_rn rounding as before, half
//      the bytes, no per-tile conversion. Measured +7.6% at pp4096 and +12% at
//      pp8192 on top of the f16-KV kernel.
//   6. The PV accumulation is fp16 (half2) by default, like ggml's prefill
//      kernel, and can be switched to fp32 with SONICBOOM_BOOM_FATTN_ACC=f32. The
//      fp32 variant halves the tile or widens the dim split to stay within the
//      64-register budget of a 1024-thread block.
//   7. A block's key loop stops at its own last query row, not at the end of
//      the chunk: causal prefill therefore does n^2/2 per chunk instead of n^2,
//      and a sliding-window block scans its window instead of the whole chunk.
//      Measured +5..8% end to end; the dropped tiles are fully masked.
//
// Batch invariance: key tiles are aligned to absolute multiples of BK, never
// to the chunk boundary, so a query's tile sequence (and therefore its
// softmax rescale order and bit pattern) is the same for any chunking of the
// same absolute positions.

#include "cuda_resident.h"

#include "cuda_stream.h"
#include "fattn_tune.h"

#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include "fattn_boom_device.cuh"

namespace sonicboom::nn::cuda {
namespace {

// ---------------------------------------------------------------------------
// Host side: workspace, launch.
// ---------------------------------------------------------------------------

// KV workspace, reused across layers and grown on demand. The gather kernel
// writes every element of the current span, so no zeroing is needed on growth.
// Under CUDA-Graph capture a growth would need an illegal cudaMalloc: decline
// and let the native path run (capture happens after warm-up, so this is a
// fallback, not the normal route).
void* g_q16 = nullptr;
size_t g_q16_bytes = 0;
void* g_kv_k = nullptr;
size_t g_kv_k_bytes = 0;
void* g_kv_v = nullptr;
size_t g_kv_v_bytes = 0;

void* grow(void*& buf, size_t& have, size_t need) {
  if (have >= need) return buf;
  if (sonicboom::cuda::sb_capturing()) return nullptr;
  if (buf) cudaFree(buf);
  buf = nullptr;
  have = 0;
  if (cudaMalloc(&buf, need) != cudaSuccess) return nullptr;
  have = need;
  return buf;
}

void debug_check(const char* stage, cudaStream_t stream) {
  static const bool on = std::getenv("SONICBOOM_ATTN_DEBUG") != nullptr;
  if (!on) return;
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    std::fprintf(stderr, "[boom_fattn] %s: %s\n", stage, cudaGetErrorString(e));
  }
}

// SONICBOOM_BOOM_FATTN_QREG selects the phase-1 variant (d512, f16
// accumulator):
//   1 (default) split QK^T across the NW dim slices, Q resident in registers
//   0           the previous kernel: BR=32 / 1024-thread block, phase 1 on the
//               slice-0 warps, Q re-read from global per key tile
//               (explicit fallback; the only mode that changes the query tile
//               and block shape, so it is kept as the revert path)
//   2           split QK^T only, Q still re-read from global per key tile
//               (control: isolates the register residency from the split)
//   3           the baseline kernel at BR=16 / 512 threads, no split
//               (control: isolates the split from the smaller query tile and
//               block, which modes 1 and 2 also change)
// Modes 0-3 apply to head_dim 512 with the default f16 accumulator only; the
// d256 configuration has its own variant and switch below, and the
// f32-accumulator configuration keeps its previous kernel regardless of this
// setting.
// Read per call so the variants can be benchmarked in one process.
int qreg_mode() {
  const char* e = std::getenv("SONICBOOM_BOOM_FATTN_QREG");
  if (e == nullptr) return 1;
  if (std::string_view(e) == "0") return 0;
  if (std::string_view(e) == "1") return 1;
  if (std::string_view(e) == "2") return 2;
  if (std::string_view(e) == "3") return 3;
  return 1;
}

// SONICBOOM_BOOM_FATTN_D256 selects the d256 head-width kernel (window-agnostic;
// the windowed traversal is driven by the per-layer window argument):
//   1 (default) query-resident, QK^T split over 4 dim slices, BR16/BK16,
//               512-thread block (the d512 mode-1 schedule at this head width)
//   0           the previous kernel: BR32/BK32, phase 1 on the slice-0 warps,
//               Q re-read from global per key tile (explicit fallback)
//   2           split QK^T only, Q still re-read per key tile (control)
//   3           BR16/BK16 without the split (control: isolates the geometry)
//   4           as 1 but split over 8 dim slices (1024-thread block; control)
// The d256 f32-accumulator configuration keeps its previous kernel regardless
// of this setting.
int d256_mode() {
  const char* e = std::getenv("SONICBOOM_BOOM_FATTN_D256");
  if (e == nullptr) return 1;
  if (std::string_view(e) == "0") return 0;
  if (std::string_view(e) == "1") return 1;
  if (std::string_view(e) == "2") return 2;
  if (std::string_view(e) == "3") return 3;
  if (std::string_view(e) == "4") return 4;
  return 1;
}

// Stage this call's inputs: materialize the KV span the attention kernel scans
// as [kv_head][key][dim] f16 (K) and [kv_head][dim][key] f16 (V), and cast Q to
// f16 once. Shared by the tuned path and the fixed legacy variants.
//
// Relative span bound. Windowed: the span starts at most window-1 keys below
// the first query, and the span base is aligned down by up to 31 keys; the last
// key tile stages up to BK-1 keys past the final key. Global: base is 0 and the
// largest span is n_slots-1+n. Rounding to 32 keeps the gather grid and the V
// rows 16-byte aligned.
template <int HEAD_DIM, int BK>
bool stage_kv(const float* q, const float* k_batch, const float* v_batch,
              const float* k_cache, const float* v_cache,
              const uint64_t* start, uint64_t n_slots, int n, int n_heads_q,
              int n_heads_kv, uint64_t window, bool kv_f16, __half** q16_out,
              __half** k_ws_out, __half** v_ws_out, int* ne_pad_out) {
  const uint64_t span =
      (window > 0 ? window - 1 + (uint64_t)n + 31 + (uint64_t)BK
                  : n_slots + (uint64_t)n + (uint64_t)BK);
  const int ne_pad = (int)((span + 31) / 32 * 32);
  const size_t need =
      (size_t)ne_pad * (size_t)n_heads_kv * (size_t)HEAD_DIM * sizeof(__half);
  __half* k_ws = static_cast<__half*>(grow(g_kv_k, g_kv_k_bytes, need));
  if (!k_ws) return false;
  __half* v_ws = static_cast<__half*>(grow(g_kv_v, g_kv_v_bytes, need));
  if (!v_ws) return false;

  cudaStream_t stream = sonicboom::cuda::sb_current_stream();

  // Both kernels derive the span base from the device start position.
  const dim3 gather_grid(HEAD_DIM / 64, (unsigned)(ne_pad / 32),
                         (unsigned)n_heads_kv);
  SB_NOTE_LAUNCH_TP("boom_fattn_gather_kernel", kv_f16 ? "cache=f16" : "cache=f32",
                    gather_grid.x * gather_grid.y * gather_grid.z, 256, 0,
                    stream);
  if (kv_f16) {
    boom_fattn_gather_kernel<__half><<<gather_grid, 256, 0, stream>>>(
        k_batch, v_batch, reinterpret_cast<const __half*>(k_cache),
        reinterpret_cast<const __half*>(v_cache), start, n_slots, n, n_heads_kv,
        HEAD_DIM, ne_pad, window, k_ws, v_ws);
  } else {
    boom_fattn_gather_kernel<float><<<gather_grid, 256, 0, stream>>>(
        k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_kv,
        HEAD_DIM, ne_pad, window, k_ws, v_ws);
  }
  debug_check("gather", stream);

  const int q_elems = n * n_heads_q * HEAD_DIM;
  __half* q16 = static_cast<__half*>(
      grow(g_q16, g_q16_bytes, (size_t)q_elems * sizeof(__half)));
  if (!q16) return false;
  const int qcast_blocks = (q_elems / 8 + 255) / 256;
  SB_NOTE_LAUNCH(boom_fattn_qcast_kernel, qcast_blocks, 256, 0, stream);
  boom_fattn_qcast_kernel<<<qcast_blocks, 256, 0, stream>>>(q, q_elems, q16);
  debug_check("qcast", stream);

  *q16_out = q16;
  *k_ws_out = k_ws;
  *v_ws_out = v_ws;
  *ne_pad_out = ne_pad;
  return true;
}

// The fixed-geometry launch used by the legacy/experimental variants (the
// SONICBOOM_BOOM_FATTN_QREG / _D256 modes and the f32 accumulator). The default
// configuration goes through the candidate tuner instead (run_tuned below).
template <int HEAD_DIM, int GQA, int BR, int BK, int NW, bool F16ACC,
          bool QREG = false, bool QRES = true>
bool run_chunk(const float* q, const float* k_batch, const float* v_batch,
               const float* k_cache, const float* v_cache,
               const uint64_t* start, uint64_t n_slots, int n, int n_heads_q,
               int n_heads_kv, uint64_t window, float scale, float* out,
               bool kv_f16) {
  constexpr int UNITS = (BR / 16) * GQA;
  constexpr int WARPS = UNITS * NW;
  constexpr int BKP = QREG ? BK : BK + 8;
  // The QREG kernel pads its K tile rows (see its KSTRIDE); the baseline
  // kernel's layout is unchanged.
  constexpr int KSTRIDE = QREG ? HEAD_DIM + 8 : HEAD_DIM;
  constexpr int SUBS = BK / 16;
  constexpr size_t smem =
      (size_t)(2 * BK * KSTRIDE + (QREG ? 3 : 2) * HEAD_DIM * BKP +
                UNITS * 16 * BK) *
          sizeof(__half) +
      (size_t)(UNITS * SUBS * 16 + UNITS * 16) * sizeof(float) +
      (QREG ? (size_t)UNITS * NW * SUBS * 256 * sizeof(float) : 0);
  static_assert(WARPS == 16 || WARPS == 32,
                "boom_fattn launch geometry is a 512- or 1024-thread block");

  using KFn = void (*)(const __half*, const __half*, const __half*,
                       const uint64_t*, uint64_t, uint64_t, int, int, int,
                       int, float, float*);
  KFn kernel;
  if constexpr (QREG)
    kernel =
        boom_fattn_attn_qreg_kernel<HEAD_DIM, GQA, BR, BK, NW, F16ACC, QRES>;
  else
    kernel = boom_fattn_attn_kernel<HEAD_DIM, GQA, BR, BK, NW, F16ACC>;
  static bool attr_ok = false;
  if (!attr_ok) {
    if (cudaFuncSetAttribute(kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem) != cudaSuccess)
      return false;
    attr_ok = true;
  }

  __half* q16 = nullptr;
  __half* k_ws = nullptr;
  __half* v_ws = nullptr;
  int ne_pad = 0;
  if (!stage_kv<HEAD_DIM, BK>(q, k_batch, v_batch, k_cache, v_cache, start,
                              n_slots, n, n_heads_q, n_heads_kv, window,
                              kv_f16, &q16, &k_ws, &v_ws, &ne_pad))
    return false;

  cudaStream_t stream = sonicboom::cuda::sb_current_stream();
  const dim3 grid((unsigned)((n + BR - 1) / BR), (unsigned)n_heads_kv);
  if constexpr (QREG)
    SB_NOTE_LAUNCH(boom_fattn_attn_qreg_kernel, grid.x * grid.y, WARPS * 32, smem,
                   stream);
  else
    SB_NOTE_LAUNCH(boom_fattn_attn_kernel, grid.x * grid.y, WARPS * 32, smem, stream);
  kernel<<<grid, WARPS * 32, smem, stream>>>(
      q16, k_ws, v_ws, start, n_slots, window, n, n_heads_q, n_heads_kv, ne_pad,
      scale, out);
  debug_check("attention", stream);
  return cudaGetLastError() == cudaSuccess;
}

// The tuned path: stage, then let the candidate tuner select and launch (the
// selection is a map lookup after the first call for this shape class).
template <int HEAD_DIM, int GQA>
bool run_tuned(const float* q, const float* k_batch, const float* v_batch,
               const float* k_cache, const float* v_cache,
               const uint64_t* start, uint64_t n_slots, int n, int n_heads_q,
               int n_heads_kv, uint64_t window, float scale, float* out,
               bool kv_f16) {
  __half* q16 = nullptr;
  __half* k_ws = nullptr;
  __half* v_ws = nullptr;
  int ne_pad = 0;
  if (!stage_kv<HEAD_DIM, 16>(q, k_batch, v_batch, k_cache, v_cache, start,
                              n_slots, n, n_heads_q, n_heads_kv, window,
                              kv_f16, &q16, &k_ws, &v_ws, &ne_pad))
    return false;

  fattn_tune::Call call;
  call.q16 = q16;
  call.k_ws = k_ws;
  call.v_ws = v_ws;
  call.start = start;
  call.n_slots = n_slots;
  call.window = window;
  call.n = n;
  call.n_heads_q = n_heads_q;
  call.n_heads_kv = n_heads_kv;
  call.ne_pad = ne_pad;
  call.scale = scale;
  call.out = out;
  call.head_dim = HEAD_DIM;
  call.gqa = GQA;
  if (fattn_tune::run(call)) return true;
  // The tuner already emitted its report; this line is the caller-visible
  // reason the call declines and the native path takes over. Once per message.
  static std::string last_reason;
  const std::string reason = fattn_tune::last_error();
  if (!reason.empty() && reason != last_reason) {
    std::fprintf(stderr, "[boom_fattn] candidate launch declined: %s\n",
                 reason.c_str());
    last_reason = reason;
  }
  return false;
}

// Read per call (not cached) so the mode can be switched between launches in
// one process; getenv is negligible next to an attention launch.
bool acc_f16() {
  const char* e = std::getenv("SONICBOOM_BOOM_FATTN_ACC");
  return !(e != nullptr && std::string_view(e) == "f32");
}

template <int HEAD_DIM, int GQA, int BR, int BK, int NW, bool F16ACC,
          bool QREG = false, bool QRES = true>
void describe(int* br, int* bk, int* block, size_t* smem) {
  constexpr int UNITS = (BR / 16) * GQA;
  constexpr int WARPS = UNITS * NW;
  constexpr int BKP = QREG ? BK : BK + 8;
  constexpr int KSTRIDE = QREG ? HEAD_DIM + 8 : HEAD_DIM;
  constexpr int SUBS = BK / 16;
  if (br) *br = BR;
  if (bk) *bk = BK;
  if (block) *block = WARPS * 32;
  if (smem)
    *smem = (size_t)(2 * BK * KSTRIDE + (QREG ? 3 : 2) * HEAD_DIM * BKP +
                     UNITS * 16 * BK) *
                sizeof(__half) +
            (size_t)(UNITS * SUBS * 16 + UNITS * 16) * sizeof(float) +
            (QREG ? (size_t)UNITS * NW * SUBS * 256 * sizeof(float) : 0);
}

}  // namespace

bool flash_attention_boom_fattn_chunk_dev_pos(
    const float* q, const float* k_batch, const float* v_batch,
    const float* k_cache, const float* v_cache, const uint64_t* start,
    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* out, bool kv_f16,
    bool kv_q8) {
  if (kv_q8 || !q || !k_batch || !v_batch || !k_cache || !v_cache || !start ||
      !out)
    return false;
  if (n <= 0 || n_heads_q <= 0 || n_heads_kv <= 0 || n_slots == 0) return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  const int gqa = n_heads_q / n_heads_kv;
  if (gqa != 4) return false;  // instantiated classes only
  // A 1024-thread block caps ptxas at 64 registers/thread, so the per-warp PV
  // output slice is kept at <= 16 mma tiles (32 accumulator registers). The
  // fp32 accumulator doubles that per tile, which halves the query tile for
  // d512: BR16 x 8 dim slices instead of BR32 x 4.
  if (head_dim == 512) {
    if (acc_f16()) {
      const int mode = qreg_mode();
      // Mode 1 is the default and the candidate the tuner keeps as its
      // reference; the tuner selects among the layout/schedule variants of
      // this schedule and launches it.
      if (mode == 1)
        return run_tuned<512, 4>(q, k_batch, v_batch, k_cache, v_cache, start,
                                 n_slots, n, n_heads_q, n_heads_kv,
                                 sliding_window, scale, out, kv_f16);
      if (mode == 2)
        return run_chunk<512, 4, 16, 16, 4, true, true, false>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 3)
        return run_chunk<512, 4, 16, 16, 4, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      return run_chunk<512, 4, 32, 16, 4, true>(
          q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
          n_heads_kv, sliding_window, scale, out, kv_f16);
    }
    return run_chunk<512, 4, 16, 16, 8, false>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  }
  if (head_dim == 256) {
    if (acc_f16()) {
      const int mode = d256_mode();
      if (mode == 0)
        return run_chunk<256, 4, 32, 32, 4, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 2)
        return run_chunk<256, 4, 16, 16, 4, true, true, false>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 3)
        return run_chunk<256, 4, 16, 16, 4, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      if (mode == 4)
        return run_chunk<256, 4, 16, 16, 8, true, true, true>(
            q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n,
            n_heads_q, n_heads_kv, sliding_window, scale, out, kv_f16);
      // Mode 1 (the default) is the tuner's reference at this head width.
      return run_tuned<256, 4>(q, k_batch, v_batch, k_cache, v_cache, start,
                               n_slots, n, n_heads_q, n_heads_kv,
                               sliding_window, scale, out, kv_f16);
    }
    return run_chunk<256, 4, 32, 32, 4, false>(
        q, k_batch, v_batch, k_cache, v_cache, start, n_slots, n, n_heads_q,
        n_heads_kv, sliding_window, scale, out, kv_f16);
  }
  return false;
}

namespace {

// Fill the trace's launch geometry from the tuner's current selection.
bool describe_tuned(int head_dim, int gqa, int* br, int* bk, int* block,
                    size_t* smem) {
  fattn_tune::Geometry g;
  if (!fattn_tune::selected(head_dim, gqa, &g)) return false;
  if (br) *br = g.br;
  if (bk) *bk = g.bk;
  if (block) *block = g.block;
  if (smem) *smem = g.smem;
  return true;
}

}  // namespace

bool flash_attention_boom_fattn_config(int head_dim, int gqa, int* br, int* bk,
                                   int* block, size_t* smem) {
  if (gqa != 4) return false;
  if (head_dim == 512) {
    if (acc_f16()) {
      const int mode = qreg_mode();
      if (mode == 1)
        return describe_tuned(512, 4, br, bk, block, smem);
      else if (mode == 2)
        describe<512, 4, 16, 16, 4, true, true, false>(br, bk, block, smem);
      else if (mode == 3)
        describe<512, 4, 16, 16, 4, true>(br, bk, block, smem);
      else
        describe<512, 4, 32, 16, 4, true>(br, bk, block, smem);
    } else {
      describe<512, 4, 16, 16, 8, false>(br, bk, block, smem);
    }
    return true;
  }
  if (head_dim == 256) {
    if (acc_f16()) {
      const int mode = d256_mode();
      if (mode == 0)
        describe<256, 4, 32, 32, 4, true>(br, bk, block, smem);
      else if (mode == 2)
        describe<256, 4, 16, 16, 4, true, true, false>(br, bk, block, smem);
      else if (mode == 3)
        describe<256, 4, 16, 16, 4, true>(br, bk, block, smem);
      else if (mode == 4)
        describe<256, 4, 16, 16, 8, true, true, true>(br, bk, block, smem);
      else
        return describe_tuned(256, 4, br, bk, block, smem);
    } else {
      describe<256, 4, 32, 32, 4, false>(br, bk, block, smem);
    }
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// The candidate table's AOT half (fattn_tune.h's registration hook). Each row
// is the QREG kernel at the fixed geometry family (BR16/BK16/NW4, f16
// accumulator, Q resident) with the row's layout and schedule. A build only
// has an image for the architectures CMake compiled (CMake default: the build
// machine's); the tuner checks that at runtime and JITs the rest.
// ---------------------------------------------------------------------------
namespace {

template <int HEAD_DIM, int GQA, int KPAD, int VPAD, int SPW, bool PIPE>
void* qreg_aot() {
  return reinterpret_cast<void*>(
      &boom_fattn_attn_qreg_kernel<HEAD_DIM, GQA, 16, 16, 4, true, true, KPAD,
                                   VPAD, SPW, PIPE>);
}

template <int HEAD_DIM>
void* qreg_aot_candidate(int candidate) {
  switch (candidate) {
    case 1: return qreg_aot<HEAD_DIM, 4, 8, 0, 2, true>();
    case 2: return qreg_aot<HEAD_DIM, 4, 8, 0, 2, false>();
    case 3: return qreg_aot<HEAD_DIM, 4, 8, 0, 1, true>();
    case 4: return qreg_aot<HEAD_DIM, 4, 0, 0, 2, true>();
    case 5: return qreg_aot<HEAD_DIM, 4, 8, 8, 2, false>();
    case 6: return qreg_aot<HEAD_DIM, 4, 8, 8, 2, true>();
    case 7: return qreg_aot<HEAD_DIM, 4, 8, 4, 2, false>();
    default: return nullptr;
  }
}

}  // namespace

void* fattn_tune::aot_kernel(int candidate, int head_dim) {
  if (head_dim == 512) return qreg_aot_candidate<512>(candidate);
  if (head_dim == 256) return qreg_aot_candidate<256>(candidate);
  return nullptr;
}

}  // namespace sonicboom::nn::cuda
