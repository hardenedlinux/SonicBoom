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

#include <sonicboom/planner/transfer_scheduler.h>

#include <cstdint>
#include <string>
#include <unordered_map>

namespace sonicboom::planner {

std::expected<void, PlannerError> TransferScheduler::schedule(
    ExecutionPlan& plan) {
  // Map each tensor to the memory space of its buffer, from the allocations.
  std::unordered_map<uint32_t, MemorySpaceId> tensor_space;
  for (const auto& a : plan.allocations)
    for (TensorId t : a.assigned_tensors)
      tensor_space[t.value] = a.memory_space;

  // Every tensor touched by a task (input or output) must already live in the
  // task's memory space; otherwise a transfer would be required, which v0 does
  // not implement.
  for (const auto& t : plan.tasks) {
    if (!t.memory_space)
      continue;
    const MemorySpaceId task_space = *t.memory_space;
    auto check = [&](TensorId tid) -> std::expected<void, PlannerError> {
      auto it = tensor_space.find(tid.value);
      if (it == tensor_space.end())
        return {};  // tensor has no buffer yet; nothing to move
      if (it->second != task_space) {
        PlannerError e(PlannerErrorCode::UnsupportedCapability,
                       "tensor requires a transfer between memory spaces, which "
                       "v0 does not implement",
                       Phase::Transfer);
        e.context.task = t.id;
        e.context.tensor = tid;
        return std::unexpected(std::move(e));
      }
      return {};
    };
    for (TensorId in : t.inputs)
      if (auto r = check(in); !r)
        return std::unexpected(r.error());
    for (TensorId out : t.outputs)
      if (auto r = check(out); !r)
        return std::unexpected(r.error());
  }

  // v0: no transfers. Record zero transfer cost and re-summarize.
  plan.estimated_cost.estimated_transfer_us = 0.0;
  if (!plan.estimated_cost.summarize())
    return std::unexpected(PlannerError(PlannerErrorCode::CostModelError,
                                        "transfer cost summary overflow",
                                        Phase::Transfer));

  return {};
}

} // namespace sonicboom::planner
