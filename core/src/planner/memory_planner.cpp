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

#include <sonicboom/planner/memory_planner.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace sonicboom::planner {

MemoryPlanner::MemoryPlanner(const CostModel& cost_model)
    : cost_model_(cost_model) {}

namespace {

// Align `size` up to a multiple of `align` (which must be >= 1), without
// wrapping. Returns nullopt on overflow.
std::optional<uint64_t> align_up(uint64_t size, uint64_t align) noexcept {
  if (align <= 1)
    return size;
  uint64_t rem = size % align;
  if (rem == 0)
    return size;
  uint64_t add = align - rem;
  if (size > UINT64_MAX - add)
    return std::nullopt;
  return size + add;
}

} // namespace

std::expected<void, PlannerError> MemoryPlanner::plan_memory(
    ExecutionPlan& plan) const {
  // Locate the compute task and the memory space it executes against. v0 has a
  // single whole-graph compute task on the host space.
  const TaskDesc* compute = nullptr;
  for (const auto& t : plan.tasks)
    if (t.kind == TaskKind::Compute) {
      compute = &t;
      break;
    }
  if (!compute || !compute->memory_space)
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidPlan,
        "memory planner requires a compute task with a memory space",
        Phase::Memory));

  // M3: buffer bookkeeping is host-only. Device-local activations are pooled and
  // managed by the device buffer pool (device_pool.h) inside the executor and
  // the CUDA backend, not by static buffer allocations, so every tensor's
  // logical buffer lives in the host space regardless of which device the
  // producing task executes on. This also keeps the budget check against the
  // host budget — weights are not graph tensors (see model/plan.h), so the host
  // working set is only the float32 activations.
  const MemorySpace* space = nullptr;
  for (const auto& m : plan.memory_spaces)
    if (m.kind == MemoryKind::Host) {
      space = &m;
      break;
    }
  if (!space)
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidPlan,
        "no host memory space in plan for buffer allocation", Phase::Memory));
  const MemorySpaceId space_id = space->id;
  const uint64_t align = space->alignment_bytes == 0 ? 1 : space->alignment_bytes;

  // 1. Lifetimes. The whole-graph model (a single whole-graph compute task)
  //    keeps every intermediate internal to the compiled unit, so a
  //    non-input/non-constant tensor is produced by that task and a non-output
  //    tensor is consumed by it. The per-node model instead derives producer
  //    and consumers from each task's explicit inputs/outputs.
  bool whole_graph = false;
  for (const auto& t : plan.tasks)
    if (t.kind == TaskKind::Compute && t.as_compute() &&
        t.as_compute()->op == OpKind::WholeGraph)
      whole_graph = true;

  plan.lifetimes.clear();
  plan.lifetimes.reserve(plan.tensors.size());
  if (whole_graph) {
    const TaskId wg = compute->id;
    for (const auto& t : plan.tensors) {
      TensorLifetime lt;
      lt.tensor = t.id;
      const bool produced_outside = t.is_graph_input || t.is_constant;
      if (!produced_outside)
        lt.producer = wg;
      if (!t.is_graph_output)
        lt.consumers.push_back(wg);
      lt.first_required = wg;
      lt.last_required = wg;
      plan.lifetimes.push_back(std::move(lt));
    }
  } else {
    std::unordered_map<uint32_t, uint32_t> position;
    for (uint32_t p = 0; p < plan.execution_order.size(); ++p)
      position[plan.execution_order[p].value] = p;

    // Producer/consumer edges in one pass over the tasks (O(tasks + edges))
    // instead of rescanning every task for every tensor. Insertion order matches
    // the old per-tensor scan: tasks in plan order, inputs in task order, and
    // the producer is the last task that lists the tensor as an output.
    std::unordered_map<uint32_t, TaskId> producers;
    std::unordered_map<uint32_t, std::vector<TaskId>> consumers;
    for (const auto& task : plan.tasks) {
      if (task.kind != TaskKind::Compute)
        continue;
      for (TensorId o : task.outputs)
        producers[o.value] = task.id;
      for (TensorId i : task.inputs)
        consumers[i.value].push_back(task.id);
    }

    for (const auto& t : plan.tensors) {
      TensorLifetime lt;
      lt.tensor = t.id;
      if (auto p = producers.find(t.id.value); p != producers.end())
        lt.producer = p->second;
      if (auto c = consumers.find(t.id.value); c != consumers.end())
        lt.consumers = std::move(c->second);

      uint32_t first = UINT32_MAX, last = 0;
      auto consider = [&](TaskId id) {
        auto it = position.find(id.value);
        if (it == position.end())
          return;
        first = std::min(first, it->second);
        last = std::max(last, it->second);
      };
      if (lt.producer)
        consider(*lt.producer);
      for (TaskId c : lt.consumers)
        consider(c);
      if (first == UINT32_MAX) {  // no producer or consumer: pin to first task
        first = 0;
        last = 0;
      }
      lt.first_required = plan.execution_order[first];
      lt.last_required = plan.execution_order[last];
      plan.lifetimes.push_back(std::move(lt));
    }
  }

  // 2. Buffer allocation. A node's IR `buf` attribute (BufferReusePass) names
  //    the buffer class its result was assigned to by liveness: every tensor in
  //    one class has a lifetime disjoint from the others, so they share one
  //    buffer of the class's largest size. A tensor the pass did not name — a
  //    graph input, a constant, a multi-result node's results, or every tensor
  //    of a whole-graph plan — keeps a buffer of its own. (A whole-graph plan's
  //    lifetimes fully overlap, so no reuse is possible there: the honest
  //    static result is N buffers.)
  std::unordered_map<uint32_t, int64_t> tensor_class;
  if (!whole_graph) {
    for (const auto& task : plan.tasks) {
      if (task.kind != TaskKind::Compute)
        continue;
      const auto* c = task.as_compute();
      // The attribute is stated once per node, so only a one-result node can
      // name its result's class.
      if (!c || task.outputs.size() != 1)
        continue;
      for (const auto& a : c->attributes) {
        if (a.name != "buf" || a.kind() != sx::AttrValueKind::Int)
          continue;
        tensor_class[task.outputs[0].value] = std::get<int64_t>(a.value);
        break;
      }
    }
  }

  // Aligned sizes and, per class, the largest member's size.
  std::unordered_map<uint32_t, uint64_t> aligned_size;
  aligned_size.reserve(plan.tensors.size());
  std::unordered_map<int64_t, uint64_t> class_size;
  for (const auto& t : plan.tensors) {
    auto aligned = align_up(t.size_bytes, align);
    if (!aligned)
      return std::unexpected(PlannerError(
          PlannerErrorCode::ArithmeticOverflow,
          "buffer size alignment overflow for tensor '" + t.name + "'",
          Phase::Memory));
    aligned_size[t.id.value] = *aligned;
    if (auto it = tensor_class.find(t.id.value); it != tensor_class.end()) {
      uint64_t& size = class_size[it->second];
      size = std::max(size, *aligned);
    }
  }

  plan.allocations.clear();
  plan.allocations.reserve(plan.tensors.size());
  uint64_t peak = 0;
  uint32_t next_buffer = 0;
  double memory_latency_us = 0.0;
  std::unordered_map<int64_t, std::size_t> class_allocation;

  auto add_allocation =
      [&](uint64_t size, TensorId tensor) -> std::expected<std::size_t, PlannerError> {
    if (size > UINT64_MAX - peak)
      return std::unexpected(PlannerError(PlannerErrorCode::ArithmeticOverflow,
                                          "peak memory overflow", Phase::Memory));
    BufferAllocation a;
    a.buffer = BufferId{next_buffer};
    a.memory_space = space_id;
    a.offset_bytes = 0;
    a.size_bytes = size;
    a.alignment_bytes = align;
    a.assigned_tensors.push_back(tensor);

    MemoryRequest req;
    req.size_bytes = size;
    req.alignment_bytes = align;
    auto mc = cost_model_.estimate_memory(req, space_id);
    if (!mc)
      return std::unexpected(mc.error());

    plan.allocations.push_back(std::move(a));
    peak += size;
    if (!next_id(next_buffer))
      return std::unexpected(PlannerError(PlannerErrorCode::InternalError,
                                          "buffer id counter overflow",
                                          Phase::Memory));
    memory_latency_us += mc->allocation_latency_us;
    return plan.allocations.size() - 1;
  };

  for (const auto& t : plan.tensors) {
    const uint64_t size = aligned_size[t.id.value];
    auto cls = tensor_class.find(t.id.value);
    if (cls == tensor_class.end()) {
      auto added = add_allocation(size, t.id);
      if (!added)
        return std::unexpected(std::move(added.error()));
      continue;
    }
    auto known = class_allocation.find(cls->second);
    if (known != class_allocation.end()) {
      plan.allocations[known->second].assigned_tensors.push_back(t.id);
      continue;
    }
    auto added = add_allocation(class_size[cls->second], t.id);
    if (!added)
      return std::unexpected(std::move(added.error()));
    class_allocation[cls->second] = *added;
  }

  // 3. Memory summary and budget check.
  plan.memory_summary.peak_bytes[space_id] = peak;
  if (auto budget = space->effective_budget_bytes()) {
    plan.memory_summary.effective_budget_bytes[space_id] = *budget;
    if (peak > *budget) {
      PlannerError e(PlannerErrorCode::InsufficientMemory,
                     "peak working set " + std::to_string(peak) +
                         " bytes exceeds budget " + std::to_string(*budget) +
                         " bytes",
                     Phase::Memory);
      e.context.memory_space = space_id;
      e.context.required_bytes = peak;
      e.context.available_bytes = *budget;
      return std::unexpected(std::move(e));
    }
  }

  // 4. Memory cost component.
  plan.estimated_cost.estimated_memory_us = memory_latency_us;
  if (!plan.estimated_cost.summarize())
    return std::unexpected(PlannerError(PlannerErrorCode::CostModelError,
                                        "memory cost summary overflow",
                                        Phase::Memory));

  return {};
}

} // namespace sonicboom::planner
