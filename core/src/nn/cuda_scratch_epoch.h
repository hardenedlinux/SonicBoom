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

// Process-global CUDA scratch identity (B0.2, CUDA Graph resource identity).
//
// Several kernels use file-static device scratch (core/src/quant/
// cuda_quant_matmul.cu and core/src/nn/cuda_elementwise.cu grow_scratch). A
// recorded CUDA graph bakes those pointers in. Growing a buffer used to
// cudaFree the old one, which left every earlier capture pointing at freed
// memory. Two mechanisms live here, shared through inline variables so the .cu
// kernels and the C++ backend see one instance:
//
//   * live_captures(): the number of instantiated StepPrograms. While it is
//     non-zero, grow_scratch/release retain the old allocation instead of
//     freeing it (the capture keeps reading/writing valid memory at its own,
//     already-baked address). Retention is on by default;
//     SONICBOOM_CAPTURE_SCRATCH_RETAIN=0 disables it for an ablation.
//   * scratch_epoch(): a counter bumped on every scratch reallocation or
//     release. A captured step records it; a replay with retention disabled
//     and a changed epoch is refused (the graph's pointers may be dangling).
//
// The backend's capture-resource check (SonicCudaBackend::
// capture_resources_ok) covers the per-session buffers and the KV cache; this
// header covers the process-global workspace those two .cu files own.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <string_view>

namespace sonicboom::cuda {

inline std::atomic<uint64_t>& scratch_epoch_counter() {
  static std::atomic<uint64_t> e{0};
  return e;
}

inline uint64_t scratch_epoch() {
  return scratch_epoch_counter().load(std::memory_order_relaxed);
}

inline void scratch_epoch_bump() {
  scratch_epoch_counter().fetch_add(1, std::memory_order_relaxed);
}

inline std::atomic<uint64_t>& live_capture_counter() {
  static std::atomic<uint64_t> n{0};
  return n;
}

// A StepProgram owns a live capture for its whole lifetime.
inline void capture_register() {
  live_capture_counter().fetch_add(1, std::memory_order_relaxed);
}

inline void capture_unregister() {
  live_capture_counter().fetch_sub(1, std::memory_order_relaxed);
}

inline bool captures_live() {
  return live_capture_counter().load(std::memory_order_relaxed) != 0;
}

inline bool scratch_retention_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("SONICBOOM_CAPTURE_SCRATCH_RETAIN");
    return !(e && std::string_view(e) == "0");
  }();
  return v;
}

// True when a scratch reallocation may free the old buffer: nobody holds a
// capture, or the retention ablation is on. Otherwise the caller must retain.
inline bool scratch_may_free() {
  return !captures_live() || !scratch_retention_enabled();
}

} // namespace sonicboom::cuda
