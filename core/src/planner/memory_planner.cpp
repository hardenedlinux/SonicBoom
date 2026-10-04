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

#include <cstdint>
#include <optional>
#include <string>
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

  const MemorySpaceId space_id = *compute->memory_space;
  const MemorySpace* space = nullptr;
  for (const auto& m : plan.memory_spaces)
    if (m.id == space_id) {
      space = &m;
      break;
    }
  if (!space)
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidPlan,
        "compute task memory space is not present in the plan", Phase::Memory));
  const uint64_t align = space->alignment_bytes == 0 ? 1 : space->alignment_bytes;

  const TaskId compute_task = compute->id;

  // 1. Lifetimes. In the whole-graph model every tensor is live for the single
  //    compute task: inputs/constants are produced before it (no producer) and
  //    read by it; node outputs (intermediate and final) are produced by it.
  plan.lifetimes.clear();
  plan.lifetimes.reserve(plan.tensors.size());
  for (const auto& t : plan.tensors) {
    TensorLifetime lt;
    lt.tensor = t.id;
    const bool produced_outside = t.is_graph_input || t.is_constant;
    if (!produced_outside)
      lt.producer = compute_task;
    if (!t.is_graph_output)
      lt.consumers.push_back(compute_task);
    lt.first_required = compute_task;
    lt.last_required = compute_task;
    plan.lifetimes.push_back(std::move(lt));
  }

  // 2. Buffer allocation: one aligned buffer per tensor. Because the whole-graph
  //    task keeps every tensor live simultaneously, lifetimes fully overlap and
  //    no buffer reuse is possible — the honest static result is N buffers.
  plan.allocations.clear();
  plan.allocations.reserve(plan.tensors.size());
  uint64_t peak = 0;
  uint32_t next_buffer = 0;
  double memory_latency_us = 0.0;

  for (const auto& t : plan.tensors) {
    auto aligned = align_up(t.size_bytes, align);
    if (!aligned)
      return std::unexpected(PlannerError(
          PlannerErrorCode::ArithmeticOverflow,
          "buffer size alignment overflow for tensor '" + t.name + "'",
          Phase::Memory));
    const uint64_t size = *aligned;
    if (size > UINT64_MAX - peak)
      return std::unexpected(PlannerError(PlannerErrorCode::ArithmeticOverflow,
                                          "peak memory overflow", Phase::Memory));

    BufferAllocation a;
    a.buffer = BufferId{next_buffer};
    a.memory_space = space_id;
    a.offset_bytes = 0;
    a.size_bytes = size;
    a.alignment_bytes = align;
    a.assigned_tensors.push_back(t.id);
    plan.allocations.push_back(std::move(a));
    peak += size;
    if (!next_id(next_buffer))
      return std::unexpected(PlannerError(PlannerErrorCode::InternalError,
                                          "buffer id counter overflow",
                                          Phase::Memory));

    MemoryRequest req;
    req.size_bytes = size;
    req.alignment_bytes = align;
    auto mc = cost_model_.estimate_memory(req, space_id);
    if (!mc)
      return std::unexpected(mc.error());
    memory_latency_us += mc->allocation_latency_us;
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
