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

#pragma once

// Candidate selection for boom_fattn: a one-time JIT for candidates the build
// has no image for, a startup autotune that picks one candidate per shape
// class, and a persistent per-device selection cache.
//
// The two stages are independent:
//
//   1. fattn_jit (fattn_jit.h) answers "can this candidate run on this device".
//      The AOT cubin in libsonicboom.so only covers the architectures the
//      library was built for (CMake default: the build machine's), so on any
//      other device a candidate is compiled once from the same device header
//      with NVRTC and the PTX is cached under $XDG_CACHE_HOME/sonicboom/kernels.
//   2. This file answers "which candidate". It resolves a shape class once per
//      device, persists the winner, and every later launch is a map lookup.
//
// Candidates are the existing boom_fattn variants (K row padding, V row
// padding, phase-1b warp assignment, sequential vs QK/PV pipeline schedule),
// registered as descriptors in fattn_tune.cpp. The reference candidate is the
// configuration measured best on an RTX 3050 (KSTRIDE padding, 8-warp phase-1b
// row-half split, unpadded V, QK/PV pipeline), and it is the incumbent: a
// candidate only displaces it by beating it by more than kMarginPct, so a
// measurement that is merely noise cannot select a slower kernel.
//
// Gates (all read per call):
//   SONICBOOM_FATTN_AUTOTUNE=0|1|force  off / on (default) / retime now
//   SONICBOOM_FATTN_CANDIDATE=<id>      force one candidate (by id or name)
//   SONICBOOM_FATTN_CANDIDATES=<a,b,..> restrict the tuning set
//   SONICBOOM_FATTN_FA2=0|1             row-owning schedule for d128/GQA-7
//                                       global attention (default 1);
//                                       unset or 1 selects it, 0 keeps the
//                                       query-resident family
//   SONICBOOM_FATTN_JIT=off|auto|force  AOT only / AOT-or-JIT / JIT only
//   SONICBOOM_FATTN_REPORT=0|1          startup report (default 1)
//   SONICBOOM_FATTN_REPORT_FILE=<path>  also write the full report there
//   SONICBOOM_FATTN_SHOW=1              print the cached selection, no tuning
//   SONICBOOM_FATTN_AUTOTUNE_BUDGET_MS  cold-tuning budget (default 4000)
//   SONICBOOM_FATTN_FP_SIMULATE=...     override fingerprint components in the
//                                       key and the report only (cache tests)
// SONICBOOM_AUTOTUNE=0 also turns this off (the global tuning gate).
//
// Internal machinery (core/src/nn), not part of Layer 2.

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace sonicboom::nn::cuda::fattn_tune {

// Candidate registry version. Part of the persistent cache key: bump it when a
// candidate is added, removed or its code changes, and every stored selection
// for this backend is invalidated.
constexpr int kRegistryVersion = 5;

// Fingerprint schema version. A selection key is
//   f<version>/<capability>/<software>/<workload>
// where each component is a compact string built only from fields that can
// change candidate legality, binary compatibility or the performance ordering
// (see docs/fattn-autotune.md for the field-to-role table). Bump this version
// when the *meaning* of a component changes (a field is added or dropped, or a
// bucketing rule changes), so a stored record made under the old meaning can
// never be read as the new one even if the strings happen to match.
//
// The capability component is probed from the device (compute capability, SM
// count, shared-memory and register/thread limits), never inferred from the
// product name. The software component is the driver/runtime CUDA version, the
// compile path (AOT cubin / NVRTC), the NVRTC version when the JIT is used, the
// device-source hash and the candidate registry version. The workload
// component is the head dim, GQA, KV dtype, mask kind (sliding window or
// global) and a power-of-two bucket of the KV span a call can read. The query
// count is not in any component: one kernel per shape class (run rule R1).
//
// SONICBOOM_FATTN_FP_SIMULATE="capability=..;software=..;workload=.." replaces a
// component in the key and the report only. Legality checks, compilation and
// timing always use the real device and the real build, so the switch can
// prove cache isolation but can never run a kernel the device cannot run.
constexpr int kFingerprintVersion = 1;

// True when this head width / GQA has a candidate set.
bool supported(int head_dim, int gqa);

// Registration hook, implemented by the AOT translation unit
// (boom_fattn.cu): the kernel for a candidate this build instantiates, or
// nullptr when the cubin has no instantiation for it. A non-null pointer does
// not mean it can run on every device: the caller checks
// cudaFuncGetAttributes() and falls back to the JIT when there is no image for
// the current architecture.
void* aot_kernel(int candidate, int head_dim, int gqa);

// The launch geometry a candidate resolves to, for the plan trace and for
// flash_attention_boom_fattn_config().
struct Geometry {
  int br = 16;
  int bk = 16;
  int block = 512;
  int kstride = 520;
  int bkp = 16;
  int softmax_warps = 8;
  bool pipeline = true;
  size_t smem = 0;
};

// One attention call. The buffers are staged by the caller (gather + Q cast)
// and are what the candidates run on, so timing uses the real shape.
struct Call {
  const __half* q16 = nullptr;
  const __half* k_ws = nullptr;
  const __half* v_ws = nullptr;
  // Direct-load sources for the row-owning family: the f32 Q activation, the
  // current chunk's f32 K/V batch and the layer's K/V cache. The row-owning
  // kernel reads these itself (cp.async from the cache, rounding the batch and
  // Q in-kernel), so it needs no staged workspace. The query-resident family
  // keeps q16/k_ws/v_ws.
  const float* q_f32 = nullptr;
  const float* k_batch = nullptr;
  const float* v_batch = nullptr;
  const void* k_cache = nullptr;
  const void* v_cache = nullptr;
  int kv_dtype = 0;  // 0 = f32, 1 = f16, 2 = q8 cache
  const uint64_t* start = nullptr;
  uint64_t n_slots = 0;
  uint64_t window = 0;
  int n = 0;
  int n_heads_q = 0;
  int n_heads_kv = 0;
  int ne_pad = 0;
  float scale = 0.0f;
  float* out = nullptr;
  int head_dim = 0;
  int gqa = 0;
  // The caller staged the wider span padding for the row-owning family. The
  // selected candidate must belong to the family the workspace was staged
  // for, so this gates family 1 in call_family() before the shape and env
  // checks: a verify call stages the query-resident family whatever the
  // prefill default is.
  bool row_owning = false;
};

// Select (once per shape class per device) and launch. False means no
// candidate could run at all and the caller must use its own fallback path;
// the reason is available through last_error().
bool run(const Call& call);

// The candidate family a call resolves to (1 = row-owning, 0 = query-resident),
// using the same gates run() applies. The caller uses it to decide whether the
// call needs the direct-load sources (family 1) or the staged gather + qcast
// workspace (family 0, and any row-owning call whose gate fails).
int family_for(const Call& call);

// The geometry of the current selection for (head_dim, gqa), or the reference
// geometry when the class has not been resolved. False when unsupported.
bool selected(int head_dim, int gqa, Geometry* geo);

// Why the last run() returned false (a launch failure, no legal candidate).
const char* last_error();

// Test hook: drop the process memo (the disk cache is untouched), so the next
// call resolves the class again.
void reset_for_test();

}  // namespace sonicboom::nn::cuda::fattn_tune
