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

// v0 transfer scheduler. With a single host memory space and CPU-only compute,
// no tensor movement between memory spaces is ever required. This stage
// verifies that invariant: a tensor whose buffer lives in a memory space
// different from the task that produces/consumes it would require a transfer,
// which v0 does not implement — such a plan is rejected (UnsupportedCapability)
// rather than silently scheduled. Otherwise it records zero transfer cost.

#include <sonicboom/planner/execution_plan.h>

#include <expected>

namespace sonicboom::planner {

class TransferScheduler {
public:
  static std::expected<void, PlannerError> schedule(ExecutionPlan& plan);
};

} // namespace sonicboom::planner
