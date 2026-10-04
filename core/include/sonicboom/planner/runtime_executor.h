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

// Runtime executor. Given a validated ExecutionPlan and a Backend, it verifies
// the live resource fingerprint still matches the plan, re-validates the plan
// defensively, then walks execution_order and dispatches each task. It never
// replans, migrates, or alters the plan — on any mismatch it fails with a
// RuntimeError instead of silently doing something different.

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/execution_plan.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/sx/exec.h>

#include <expected>
#include <vector>

namespace sonicboom::planner {

// Output of one plan execution: one buffer per graph output tensor, in plan
// tensor order.
struct ExecutionResult {
  std::vector<sx::Bytes> outputs;
};

class RuntimeExecutor {
public:
  // `backend` must outlive the executor.
  explicit RuntimeExecutor(Backend& backend);

  // Execute `plan`. `inputs` holds one raw buffer per graph-input tensor (in
  // plan order); constants/weights are baked into the backend's compiled unit.
  std::expected<ExecutionResult, RuntimeError> execute(
      const ExecutionPlan& plan, const ResourceSnapshot& live,
      const std::vector<sx::Bytes>& inputs) const;

private:
  Backend& backend_;
};

} // namespace sonicboom::planner
