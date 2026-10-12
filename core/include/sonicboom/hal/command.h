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

// HAL command-recording seam (Step 1). A backend that can pre-record one step
// and replay it returns a StepProgram from Backend::record_step(). The planner
// and executor only ever see this opaque interface — no CUDA stream / CUDA
// Graph / Vulkan command-buffer type appears here. The concrete mapping is a
// backend-private detail:
//   - CUDA: an instantiated cudaGraphExec, replayed with cudaGraphLaunch;
//   - CPU:  a recorded task list re-executed in order per replay;
//   - Vulkan (community): a recorded vk::CommandBuffer re-submitted per replay.

#include <sonicboom/hal/buffer.h>

#include <cstdint>

namespace sonicboom::hal {

// A pre-recorded, re-playable step owned by the backend.
class StepProgram {
 public:
  virtual ~StepProgram() = default;

  // Launch / re-run one step (async where the backend supports it).
  virtual bool replay() = 0;

  // Block until the last replay's work completes (a fence).
  virtual bool sync() = 0;

  // True when the program is valid and replay-able.
  virtual bool valid() const = 0;

  // Stage host data into a device buffer on the program's own stream (async
  // where the backend supports it), ordered before the next replay(). This is
  // the async complement to Allocator::copy_to_device: it uploads a replayed
  // step's per-replay inputs (e.g. a prefill chunk's token ids / start position)
  // without a host-blocking synchronous H2D, so the host can enqueue staging +
  // replay back-to-back and never wait on the device. The CUDA backend maps it
  // to cudaMemcpyAsync on the graph stream; the CPU backend (whose buffers are
  // host memory) copies directly. Returns false on failure.
  virtual bool stage_upload(const Buffer& dst, const void* host,
                            uint64_t bytes) = 0;
};

}  // namespace sonicboom::hal
