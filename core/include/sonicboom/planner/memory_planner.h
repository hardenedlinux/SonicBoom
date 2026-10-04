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

// v0 memory planner. It fills the memory-planning portion of an ExecutionPlan
// produced by StaticPlanner:
//   - tensor lifetimes (producer/consumers/first-last required),
//   - buffer allocations (one aligned buffer per tensor; v0's whole-graph task
//     keeps every tensor live simultaneously, so no reuse is possible),
//   - the peak working set per memory space (memory_summary),
//   - an InsufficientMemory error when the peak exceeds the effective budget,
//   - the memory cost component of the plan cost summary.
//
// The whole-graph compiled unit keeps all inputs, weights, activations, and
// outputs materialized at once, so the honest static peak is the sum of every
// tensor's aligned size — this is a conservative upper bound, not a simulation.

#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/execution_plan.h>

#include <expected>

namespace sonicboom::planner {

class MemoryPlanner {
public:
  // `cost_model` must outlive the planner; it provides per-allocation cost and
  // validates memory-space ids against its snapshot.
  explicit MemoryPlanner(const CostModel& cost_model);

  // Fill lifetimes, allocations, memory_summary, and memory cost on `plan`.
  std::expected<void, PlannerError> plan_memory(ExecutionPlan& plan) const;

private:
  const CostModel& cost_model_;
};

} // namespace sonicboom::planner
