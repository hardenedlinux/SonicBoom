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

#include <sonicboom/planner/pipeline.h>

#include <sonicboom/planner/memory_planner.h>
#include <sonicboom/planner/plan_validator.h>
#include <sonicboom/planner/planner.h>
#include <sonicboom/planner/transfer_scheduler.h>

#include <utility>

namespace sonicboom::planner {

std::expected<ExecutionPlan, PlannerError> plan_execution(
    const Graph& graph, const ResourceSnapshot& snapshot,
    const CostModel& cost_model) {
  StaticPlanner planner(snapshot, cost_model);
  auto plan = planner.plan(graph);
  if (!plan)
    return std::unexpected(plan.error());

  MemoryPlanner mem(cost_model);
  if (auto r = mem.plan_memory(*plan); !r)
    return std::unexpected(r.error());

  TransferScheduler transfer(cost_model);
  if (auto r = transfer.schedule(*plan); !r)
    return std::unexpected(r.error());

  if (auto r = PlanValidator::validate(*plan); !r)
    return std::unexpected(r.error());

  return std::move(*plan);
}

} // namespace sonicboom::planner
