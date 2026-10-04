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

#include <sonicboom/planner/execution_plan.h>

namespace sonicboom::planner {

const Device* ExecutionPlan::find_device(DeviceId id) const noexcept {
  for (const auto& d : devices)
    if (d.id == id)
      return &d;
  return nullptr;
}

const MemorySpace* ExecutionPlan::find_memory_space(MemorySpaceId id) const noexcept {
  for (const auto& m : memory_spaces)
    if (m.id == id)
      return &m;
  return nullptr;
}

const TensorDesc* ExecutionPlan::find_tensor(TensorId id) const noexcept {
  for (const auto& t : tensors)
    if (t.id == id)
      return &t;
  return nullptr;
}

const TaskDesc* ExecutionPlan::find_task(TaskId id) const noexcept {
  for (const auto& t : tasks)
    if (t.id == id)
      return &t;
  return nullptr;
}

const BufferAllocation* ExecutionPlan::find_buffer(BufferId id) const noexcept {
  for (const auto& b : allocations)
    if (b.buffer == id)
      return &b;
  return nullptr;
}

} // namespace sonicboom::planner
