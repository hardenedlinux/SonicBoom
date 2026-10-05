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
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace sonicboom::planner {

RuntimeExecutor::RuntimeExecutor(Backend& backend) {
  backends_[BackendTag::Mlir] = &backend;
}

RuntimeExecutor::RuntimeExecutor(std::map<BackendTag, Backend*> backends)
    : backends_(std::move(backends)) {}

void RuntimeExecutor::set_task_backend(TaskId task, Backend* backend) {
  task_backends_[task] = backend;
}

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

  // 4. Seed the value map with the graph-input buffers, in plan tensor order.
  std::map<uint32_t, sx::Bytes> values;  // TensorId -> raw bytes
  std::size_t in_i = 0;
  for (const auto& t : plan.tensors)
    if (t.is_graph_input)
      values[t.id.value] = inputs[in_i++];

  // 5. Walk execution_order: dispatch each compute task to its routed backend,
  //    handing intermediate tensors between tasks by TensorId.
  for (TaskId tid : plan.execution_order) {
    const TaskDesc* task = plan.find_task(tid);
    if (!task)
      return std::unexpected(RuntimeError(
          RuntimeErrorCode::InvalidPlan,
          "execution order references unknown task", Phase::Execution));

    switch (task->kind) {
      case TaskKind::Compute: {
        const auto* compute = task->as_compute();
        if (!compute)
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "compute task has no compute payload", Phase::Execution));
        Backend* backend = nullptr;
        if (auto t = task_backends_.find(tid); t != task_backends_.end())
          backend = t->second;
        else {
          auto it = backends_.find(compute->backend);
          if (it != backends_.end())
            backend = it->second;
        }
        if (!backend)
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::BackendFailure,
              "no backend registered for tag '" +
                  std::string(backend_tag_name(compute->backend)) + "'",
              Phase::Execution));

        std::vector<sx::Bytes> task_inputs;
        task_inputs.reserve(task->inputs.size());
        for (TensorId in : task->inputs) {
          // Constants (weights/axes/shapes) are baked into the backend's
          // compiled unit, not passed as runtime buffers.
          if (const auto* td = plan.find_tensor(in);
              td && td->is_constant)
            continue;
          auto v = values.find(in.value);
          if (v == values.end())
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::InvalidPlan,
                "missing value for input tensor", Phase::Execution));
          task_inputs.push_back(v->second);
        }

        auto res = backend->execute(*task, task_inputs);
        if (!res)
          return std::unexpected(res.error());
        if (res->size() != task->outputs.size())
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "backend produced an unexpected number of outputs",
              Phase::Execution));
        for (std::size_t k = 0; k < task->outputs.size(); ++k)
          values[task->outputs[k].value] = std::move((*res)[k]);
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

  // 6. Gather the graph-output buffers in plan tensor order.
  std::vector<sx::Bytes> outputs;
  for (const auto& t : plan.tensors)
    if (t.is_graph_output) {
      auto v = values.find(t.id.value);
      if (v == values.end())
        return std::unexpected(RuntimeError(
            RuntimeErrorCode::InvalidPlan,
            "graph output tensor has no computed value", Phase::Execution));
      outputs.push_back(std::move(v->second));
    }

  if (outputs.size() != n_outputs)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::InvalidPlan,
        "execution produced an unexpected number of outputs", Phase::Execution));

  return ExecutionResult{std::move(outputs)};
}

} // namespace sonicboom::planner
