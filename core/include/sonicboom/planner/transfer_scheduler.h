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

// Transfer scheduler (M3). Walks the plan's execution order and emits explicit
// Transfer tasks wherever a float32 activation is produced in one memory space
// but consumed by a compute task in another (host↔device and device↔device).
// Graph inputs and constants start in the host space; Int64 scalar graph inputs
// (token id / position) are host metadata that backends read directly, so they
// are never transferred. A device-resident graph output is downloaded back to
// the host so the executor returns host bytes. Each transfer is placed in the
// execution order after its producer and before its consumer, with the consumer
// gaining a dependency on the transfer; the transfer cost is estimated through
// the cost model and accumulated into the plan.

#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/execution_plan.h>

#include <expected>

namespace sonicboom::planner {

class TransferScheduler {
public:
  explicit TransferScheduler(const CostModel& cost_model);

  std::expected<void, PlannerError> schedule(ExecutionPlan& plan);

private:
  const CostModel& cost_model_;
};

} // namespace sonicboom::planner
