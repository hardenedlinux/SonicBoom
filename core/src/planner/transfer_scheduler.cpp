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

#include <sonicboom/planner/transfer_scheduler.h>

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sonicboom::planner {

TransferScheduler::TransferScheduler(const CostModel& cost_model)
    : cost_model_(cost_model) {}

std::expected<void, PlannerError> TransferScheduler::schedule(
    ExecutionPlan& plan) {
  // The host space is the fixed anchor: graph inputs/constants start here and
  // graph outputs must return here (the executor emits host bytes).
  const MemorySpace* host = nullptr;
  for (const auto& m : plan.memory_spaces)
    if (m.kind == MemoryKind::Host) {
      host = &m;
      break;
    }
  if (!host)
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidPlan,
        "no host memory space in plan", Phase::Transfer));

  // Per-tensor "current home" space. Only float32 activations move; constants
  // are baked into backends and never appear in the executor's value map, and
  // Int64 scalar graph inputs (token id / position) are host metadata that the
  // backends read directly.
  std::unordered_map<uint32_t, MemorySpaceId> home;
  for (const auto& t : plan.tensors)
    if (t.is_graph_input)
      home[t.id.value] = host->id;

  // Producer of each tensor, for transfer dependency fix-up.
  std::unordered_map<uint32_t, TaskId> producer;
  for (const auto& t : plan.tasks)
    if (t.kind == TaskKind::Compute)
      for (TensorId o : t.outputs)
        producer[o.value] = t.id;

  std::vector<TaskDesc> transfers;
  std::unordered_map<uint32_t, std::vector<TaskId>> extra_deps;  // task -> transfers
  std::vector<TaskId> new_order;
  uint32_t next_id_val = static_cast<uint32_t>(plan.tasks.size());
  double transfer_us = 0.0;

  auto emit = [&](TensorId tensor, const TensorDesc& td, MemorySpaceId src,
                  MemorySpaceId dst, std::optional<TaskId> prod,
                  std::optional<TaskId> consumer)
      -> std::expected<void, PlannerError> {
    if (next_id_val == UINT32_MAX)
      return std::unexpected(PlannerError(
          PlannerErrorCode::InternalError, "transfer task id counter overflow",
          Phase::Transfer));
    const TaskId tid{next_id_val++};

    TaskDesc tr;
    tr.id = tid;
    tr.kind = TaskKind::Transfer;
    tr.payload = TransferTaskDesc{tensor, src, dst, td.size_bytes};
    if (prod)
      tr.dependencies.push_back(*prod);

    auto tc = cost_model_.estimate_transfer(src, dst);
    if (!tc)
      return std::unexpected(tc.error());
    transfer_us += tc->fixed_latency_us +
                   static_cast<double>(td.size_bytes) /
                       tc->effective_bandwidth_bytes_per_us;

    new_order.push_back(tid);
    if (consumer)
      extra_deps[consumer->value].push_back(tid);
    transfers.push_back(std::move(tr));
    return {};
  };

  // Walk the compute tasks in execution order; before each task, move any input
  // whose home space differs from the task's space into that space.
  for (TaskId tid : plan.execution_order) {
    const TaskDesc* task = plan.find_task(tid);
    if (!task || task->kind != TaskKind::Compute) {
      new_order.push_back(tid);
      continue;
    }
    const MemorySpaceId task_space = task->memory_space.value_or(host->id);

    for (TensorId in : task->inputs) {
      const TensorDesc* td = plan.find_tensor(in);
      if (!td || td->is_constant || td->dtype != sx::DType::Float32)
        continue;
      auto h = home.find(in.value);
      const MemorySpaceId src = (h != home.end()) ? h->second : host->id;
      if (src == task_space)
        continue;

      std::optional<TaskId> prod;
      if (auto p = producer.find(in.value); p != producer.end())
        prod = p->second;
      if (auto r = emit(in, *td, src, task_space, prod, tid); !r)
        return std::unexpected(r.error());
      home[in.value] = task_space;
    }

    new_order.push_back(tid);
    for (TensorId o : task->outputs)
      home[o.value] = task_space;
  }

  // A device-resident graph output must return to the host so the executor can
  // hand back host bytes. (In the Gemma 4 decode plan the softcapped logits are
  // already produced on the host by the CPU Softcap, so this is normally a
  // no-op; it guards any plan whose final output stays on the device.)
  for (const auto& t : plan.tensors) {
    if (!t.is_graph_output)
      continue;
    auto h = home.find(t.id.value);
    const MemorySpaceId src = (h != home.end()) ? h->second : host->id;
    if (src == host->id)
      continue;

    std::optional<TaskId> prod;
    if (auto p = producer.find(t.id.value); p != producer.end())
      prod = p->second;
    if (auto r = emit(t.id, t, src, host->id, prod, std::nullopt); !r)
      return std::unexpected(r.error());
    home[t.id.value] = host->id;
  }

  // Commit: append the transfer tasks, add the consumer dependencies, and
  // replace the execution order.
  for (auto& t : transfers)
    plan.tasks.push_back(std::move(t));
  for (const auto& [tid, deps] : extra_deps)
    for (auto& task : plan.tasks)
      if (task.id == TaskId{tid}) {
        for (TaskId d : deps)
          task.dependencies.push_back(d);
        break;
      }
  plan.execution_order = std::move(new_order);

  plan.estimated_cost.estimated_transfer_us = transfer_us;
  if (!plan.estimated_cost.summarize())
    return std::unexpected(PlannerError(PlannerErrorCode::CostModelError,
                                        "transfer cost summary overflow",
                                        Phase::Transfer));

  return {};
}

} // namespace sonicboom::planner
