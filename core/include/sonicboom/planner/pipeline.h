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

// The full v0 static planning pipeline: build the task DAG and execution order
// (StaticPlanner), assign tensor lifetimes and buffers (MemoryPlanner), verify
// no transfers are required (TransferScheduler), and re-validate the result
// (PlanValidator). Returns a fully populated, validated ExecutionPlan.

#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/execution_plan.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/resource.h>

#include <expected>

namespace sonicboom::planner {

std::expected<ExecutionPlan, PlannerError> plan_execution(
    const Graph& graph, const ResourceSnapshot& snapshot,
    const CostModel& cost_model);

} // namespace sonicboom::planner
