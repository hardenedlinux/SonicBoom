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

#include <sonicboom/planner/runtime_executor.h>

#include <sonicboom/planner/plan_validator.h>

#include <cstddef>
#include <string>
#include <utility>

namespace sonicboom::planner {

RuntimeExecutor::RuntimeExecutor(Backend& backend) : backend_(backend) {}

std::expected<ExecutionResult, RuntimeError> RuntimeExecutor::execute(
    const ExecutionPlan& plan, const ResourceSnapshot& live,
    const std::vector<sx::Bytes>& inputs) const {
  // 1. The live resources must still match what was planned. The executor must
  //    never silently replan or migrate, so a mismatch is a hard error.
  if (resource_fingerprint(live) != plan.resource_fingerprint)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::ResourceChanged,
        "live resource fingerprint does not match the plan",
        Phase::Execution));

  // 2. Re-validate the plan defensively; never execute a corrupt plan.
  if (auto v = PlanValidator::validate(plan); !v)
    return std::unexpected(RuntimeError(RuntimeErrorCode::InvalidPlan,
                                        v.error().message, Phase::Execution));

  // 3. Bound the input/output buffers against the plan's graph I/O.
  std::size_t n_inputs = 0, n_outputs = 0;
  for (const auto& t : plan.tensors) {
    if (t.is_graph_input)
      ++n_inputs;
    if (t.is_graph_output)
      ++n_outputs;
  }
  if (inputs.size() != n_inputs)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::InvalidPlan,
        "input buffer count does not match the plan's graph inputs",
        Phase::Execution));

  // 4. Walk execution_order and dispatch. v0 has a single whole-graph compute
  //    task, so the graph-input buffers are passed straight through.
  std::vector<sx::Bytes> outputs;
  for (TaskId tid : plan.execution_order) {
    const TaskDesc* task = plan.find_task(tid);
    if (!task)
      return std::unexpected(RuntimeError(
          RuntimeErrorCode::InvalidPlan,
          "execution order references unknown task", Phase::Execution));

    switch (task->kind) {
      case TaskKind::Compute: {
        auto res = backend_.execute(*task, inputs);
        if (!res)
          return std::unexpected(res.error());
        outputs = std::move(*res);
        break;
      }
      case TaskKind::Transfer:
        return std::unexpected(RuntimeError(
            RuntimeErrorCode::TransferFailure,
            "v0 executor does not implement transfer tasks", Phase::Execution));
      case TaskKind::Allocate:
      case TaskKind::Release:
      case TaskKind::Synchronize:
        return std::unexpected(RuntimeError(
            RuntimeErrorCode::BackendFailure,
            "v0 executor does not implement task kind '" +
                std::string(task_kind_name(task->kind)) + "'",
            Phase::Execution));
    }
  }

  if (outputs.size() != n_outputs)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::InvalidPlan,
        "execution produced an unexpected number of outputs", Phase::Execution));

  return ExecutionResult{std::move(outputs)};
}

} // namespace sonicboom::planner
