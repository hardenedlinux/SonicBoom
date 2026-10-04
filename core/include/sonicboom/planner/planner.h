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

// Static, deterministic planner (v0). Given an adapted Graph, a resource
// snapshot, and a cost model, it builds the task DAG, assigns devices and
// memory spaces, verifies every operator/dtype is executable on the assigned
// device, computes the deterministic execution order, and estimates cost.
//
// Memory planning (tensor lifetimes + buffer allocation), transfer scheduling,
// and validation are separate stages composed by the pipeline; this stage
// leaves `lifetimes`, `allocations`, and `memory_summary` empty for those later
// stages to fill.

#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/execution_plan.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/resource.h>

#include <expected>

namespace sonicboom::planner {

class StaticPlanner {
public:
  // `snapshot` is copied (the plan records its devices/memory spaces);
  // `cost_model` must outlive the planner.
  StaticPlanner(const ResourceSnapshot& snapshot, const CostModel& cost_model);

  // Build the (memory/transfer-not-yet-planned) ExecutionPlan for `graph`.
  // Deterministic: identical inputs produce an identical plan (same ids,
  // same fingerprints, same order, same cost).
  std::expected<ExecutionPlan, PlannerError> plan(const Graph& graph) const;

private:
  ResourceSnapshot snapshot_;
  const CostModel& cost_model_;
};

} // namespace sonicboom::planner
