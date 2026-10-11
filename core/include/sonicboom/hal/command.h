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
};

}  // namespace sonicboom::hal
