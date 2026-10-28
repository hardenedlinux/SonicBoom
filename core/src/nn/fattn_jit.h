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

// Runtime half of the boom_fattn candidate JIT.
//
// The AOT instantiations in fattn_tune.cu are cubins for the architectures the
// library was built for (CMake default: the build machine's). A BOOM_FATTN
// candidate cannot run on another architecture, so the tuner compiles the
// candidate for the device it actually finds:
//
//   boom_fattn_device.cuh (embedded at build time, same file the AOT TU uses)
//        + a generated extern "C" wrapper per candidate
//        -> NVRTC -> PTX (computed for the device's compute capability)
//        -> $XDG_CACHE_HOME/sonicboom/kernels/fattn-<hash>-sm<NN>.ptx
//        -> cuModuleLoadData -> cuLaunchKernel
//
// Compilation happens once per (candidate, device, source) and the PTX text is
// reused by later processes; a failed compile is reported and the candidate is
// dropped, never retried in the same process. NVRTC is loaded with dlopen so a
// machine without libnvrtc still runs the AOT path (available() is false and
// the tuner keeps only AOT candidates).
//
// Internal machinery (core/src/nn), not part of Layer 2.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <string>

namespace sonicboom::nn::cuda::fattn_jit {

// The candidate's launch geometry and layout knobs. `id` only names the
// generated entry point (sb_fattn_jit_<id>).
struct Spec {
  int id = 0;
  int head_dim = 512;
  int gqa = 4;
  int br = 16;
  int bk = 16;
  int nw = 4;
  int kpad = 8;   // extra halves per K row
  int vpad = 0;   // extra halves per V row
  int spw = 2;    // phase-1b warps (1 = 4 warps, 2 = 8 warps)
  int pipe = 1;   // 1 = QK/PV pipeline, 0 = sequential
};

// True when libnvrtc and the CUDA driver are both usable.
bool available();

// Why JIT is unavailable, for the report ("nvrtc: ..." / empty when available).
std::string unavailable_reason();

// The embedded device source (diagnostics and tests; the same text the AOT TU
// includes). Not affected by availability.
std::string device_source();

// FNV-1a of that source text: the kernel implementation identity a selection
// or artifact key can record. Not affected by availability.
uint64_t source_hash();

// Version of the NVRTC library this process loaded ("12.4"), or
// "unavailable" when there is none. NVRTC emits the PTX, so its version is
// part of both the artifact key (binary compatibility) and the kernel
// implementation identity of a JIT-compiled candidate.
std::string nvrtc_version_string();

// Compile (or load from the PTX cache) the candidate for the current device.
// False leaves a one-line reason in *err and never throws. Thread-safe.
bool prepare(const Spec& spec, std::string* err);

// True once prepare() succeeded for exactly this spec in this process.
bool ready(const Spec& spec);

// Launch a prepared candidate with the same grid as the AOT entry point.
// `args` are pointers to the kernel parameters in signature order (q, k, v,
// start, n_slots, window, n, n_heads_q, n_heads_kv, ne_pad, scale, out);
// `smem` is the dynamic shared-memory size, `block` the block size. False when
// the candidate was not prepared or the launch failed.
bool launch(const Spec& spec, dim3 grid, void* const args[12],
            cudaStream_t stream, size_t smem, int block);

// Registers per thread, dynamic smem this launch would need (kernel static
// smem is 0) and local (spill) bytes of a prepared candidate. Values come from
// cuFuncGetAttribute; false when not prepared.
bool attributes(const Spec& spec, int* regs, int* local_bytes, int* max_threads);

// Cache file this candidate uses (diagnostics; "" when unavailable).
std::string cache_path(const Spec& spec);

// Number of JIT compiles this process performed (cache hits excluded).
int compile_count();

}  // namespace sonicboom::nn::cuda::fattn_jit
