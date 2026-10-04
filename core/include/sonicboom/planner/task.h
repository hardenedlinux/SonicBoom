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

// Task model and DAG. A task carries a unique TaskId, a kind, its tensor
// inputs/outputs, explicit dependency TaskIds, an optional device/memory-space
// assignment, an optional estimated cost, and a kind-specific payload. The
// payload is a std::variant so a transfer task cannot accidentally carry a
// compute payload and vice-versa. The DAG invariants (resolvable references,
// acyclicity) are enforced by the planner and re-checked by the validator.

#include <sonicboom/planner/cost.h>
#include <sonicboom/planner/ids.h>
#include <sonicboom/planner/op_kind.h>

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace sonicboom::planner {

enum class TaskKind : uint8_t {
  Compute,
  Transfer,
  Allocate,
  Release,
  Synchronize,
};
const char* task_kind_name(TaskKind k) noexcept;

// Compute payload: which operator/region this task represents and which JIT
// entry implements it. In v0 the compiler emits one whole-graph entry, so a
// compute task covers the whole graph (graph_nodes = every node, jit_entry = 0).
struct ComputeTaskDesc {
  OpKind op = OpKind::WholeGraph;
  std::vector<GraphNodeId> graph_nodes;
  JitEntryId jit_entry;
};

// Transfer payload: an explicit move of `bytes` of one tensor between two
// memory spaces. A task without a valid source/destination is invalid.
struct TransferTaskDesc {
  TensorId tensor;
  MemorySpaceId source;
  MemorySpaceId destination;
  uint64_t bytes = 0;
};

struct AllocateTaskDesc {
  BufferId buffer;
  MemorySpaceId memory_space;
  uint64_t size_bytes = 0;
  uint64_t alignment_bytes = 1;
};

struct ReleaseTaskDesc {
  BufferId buffer;
};

struct TaskDesc {
  TaskId id;
  TaskKind kind;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::vector<TaskId> dependencies;
  std::optional<DeviceId> device;
  std::optional<MemorySpaceId> memory_space;
  std::optional<ComputeCost> cost;  // present for compute tasks only
  std::variant<std::monostate, ComputeTaskDesc, TransferTaskDesc,
               AllocateTaskDesc, ReleaseTaskDesc>
      payload;

  // Payload accessors returning nullopt when the payload does not match `kind`.
  const ComputeTaskDesc* as_compute() const noexcept;
  const TransferTaskDesc* as_transfer() const noexcept;
  const AllocateTaskDesc* as_allocate() const noexcept;
  const ReleaseTaskDesc* as_release() const noexcept;
};

} // namespace sonicboom::planner
