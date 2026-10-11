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

#include <cuda_runtime.h>

// Internal per-thread "current CUDA stream" for the device-resident kernels.
//
// Every *_dev entry point in cuda_elementwise.cu / cuda_quant_matmul.cu and the
// planner's device_pool copy/memset helpers launch on this stream instead of the
// implicit default stream, so the CUDA backend can run — and capture — the whole
// decode step on a single non-default stream. Defaults to the legacy default
// stream (cudaStreamLegacy): without a StreamScope the kernels behave exactly as
// before (all work on stream 0, the prior behaviour).
//
// NOT part of the public Layer 2 surface: lives under core/src/.

namespace sonicboom::cuda {

inline cudaStream_t& sb_current_stream() {
  static thread_local cudaStream_t s = cudaStreamLegacy;
  return s;
}

// RAII: push `stream` as the current stream for the scope, restore on exit.
class StreamScope {
 public:
  explicit StreamScope(cudaStream_t stream)
      : prev_(sb_current_stream()) {
    sb_current_stream() = stream;
  }
  ~StreamScope() { sb_current_stream() = prev_; }
  StreamScope(const StreamScope&) = delete;
  StreamScope& operator=(const StreamScope&) = delete;

 private:
  cudaStream_t prev_;
};

}  // namespace sonicboom::cuda
