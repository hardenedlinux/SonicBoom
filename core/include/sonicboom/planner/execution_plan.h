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

// ExecutionPlan: the immutable, independently-verifiable contract between the
// static planner and the runtime executor. It carries everything needed to
// validate and execute the plan — devices, memory spaces, tensors, tasks, tensor
// lifetimes, buffer allocations, the deterministic execution order, and the
// estimated cost/memory summaries — with no dependency on MLIR/LLVM/ATen/c10
// objects. The plan is immutable after validation and before execution.

#include <sonicboom/planner/cost.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/ids.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/task.h>

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace sonicboom::planner {

// A tensor's lifetime in task-space: who produces it, who consumes it, and the
// first/last task in the execution order that requires it to be alive.
struct TensorLifetime {
  TensorId tensor;
  std::optional<TaskId> producer;  // nullopt for graph inputs / constants
  std::vector<TaskId> consumers;
  TaskId first_required;
  TaskId last_required;
};

// A logical buffer assignment. A physical buffer may host multiple tensors only
// when the memory planner proved their lifetimes cannot overlap.
struct BufferAllocation {
  BufferId buffer;
  MemorySpaceId memory_space;
  uint64_t offset_bytes = 0;
  uint64_t size_bytes = 0;
  uint64_t alignment_bytes = 1;
  std::vector<TensorId> assigned_tensors;
};

struct PlanMemorySummary {
  std::map<MemorySpaceId, uint64_t> peak_bytes;
  std::map<MemorySpaceId, uint64_t> effective_budget_bytes;
};

struct ExecutionPlan {
  static constexpr uint32_t kSchemaVersion = 1;
  uint32_t schema_version = kSchemaVersion;

  uint64_t plan_id = 0;
  uint64_t model_fingerprint = 0;
  uint64_t resource_fingerprint = 0;
  uint32_t resource_version = 0;  // snapshot version the fingerprint was over

  std::vector<Device> devices;
  std::vector<MemorySpace> memory_spaces;
  std::vector<TensorDesc> tensors;
  std::vector<TaskDesc> tasks;
  std::vector<TensorLifetime> lifetimes;
  std::vector<BufferAllocation> allocations;
  std::vector<TaskId> execution_order;

  PlanCostSummary estimated_cost;
  PlanMemorySummary memory_summary;

  const Device* find_device(DeviceId id) const noexcept;
  const MemorySpace* find_memory_space(MemorySpaceId id) const noexcept;
  const TensorDesc* find_tensor(TensorId id) const noexcept;
  const TaskDesc* find_task(TaskId id) const noexcept;
  const BufferAllocation* find_buffer(BufferId id) const noexcept;
};

} // namespace sonicboom::planner
