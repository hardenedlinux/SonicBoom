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

// Vulkan backend — community-backend stub (Step 3). This is the scaffold a
// third-party backend author copies: it implements the `Backend` interface
// (core/include/sonicboom/planner/backend.h) but registers *zero* kernels, so
// every task routed to it falls back to the CPU reference. It performs no real
// Vulkan compute and links no Vulkan loader — it exists to prove the authoring
// path (compile, instantiate, register in a backend map, advertise an empty op
// set) and is OFF by default (SONICBOOM_USE_VULKAN).
//
// It deliberately lives under core/src/ (internal), not the public Layer 2 tree:
// a community backend is a plugin (libsonicboom-backend-vulkan.so in the Step 4
// layout), and its declaration is not part of the stable runtime interface. See
// docs/backend-author-guide.md.

#include <sonicboom/planner/backend.h>

namespace sonicboom::planner {

class VulkanBackend : public Backend {
public:
  VulkanBackend() = default;

  // Vulkan is a GPU-class accelerator, distinct from the CPU and the CUDA GPU.
  DeviceKind kind() const noexcept override { return DeviceKind::Accelerator; }

  // Zero ops: every task fails with BackendFailure, which the caller treats as
  // "not supported here" and falls back to the CPU reference (SonicBackend).
  // A real implementation would dispatch to the device kernels registered for
  // each OpKind; the stub registers none.
  std::expected<std::vector<TensorValue>, RuntimeError> execute(
      const TaskDesc& task, const std::vector<TensorValue>& inputs) override;
};

} // namespace sonicboom::planner
