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

#include <sonicboom/planner/planner.h>

#include "fingerprint.h"

#include <cstdint>
#include <utility>

namespace sonicboom::planner {

StaticPlanner::StaticPlanner(const ResourceSnapshot& snapshot,
                             const CostModel& cost_model)
    : snapshot_(snapshot), cost_model_(cost_model) {}

namespace {

const Device* find_cpu(const ResourceSnapshot& s) {
  const Device* best = nullptr;
  for (const auto& d : s.devices)
    if (d.kind == DeviceKind::CPU && (!best || d.id < best->id))
      best = &d;
  return best;
}

const MemorySpace* find_host(const ResourceSnapshot& s) {
  const MemorySpace* best = nullptr;
  for (const auto& m : s.memory_spaces)
    if (m.kind == MemoryKind::Host && (!best || m.id < best->id))
      best = &m;
  return best;
}

bool supports_dtype(const Device& d, sx::DType dt) {
  for (sx::DType x : d.capability.supported_dtypes)
    if (x == dt)
      return true;
  return false;
}

bool supports_op(const Device& d, OpKind op) {
  for (OpKind x : d.capability.supported_ops)
    if (x == op)
      return true;
  return false;
}

} // namespace

std::expected<ExecutionPlan, PlannerError> StaticPlanner::plan(
    const Graph& graph) const {
  // The snapshot must be internally consistent before anything is planned.
  if (auto v = validate_snapshot(snapshot_); !v)
    return std::unexpected(v.error());

  const Device* cpu = find_cpu(snapshot_);
  if (!cpu || !cpu->capability.can_execute)
    return std::unexpected(PlannerError(
        PlannerErrorCode::UnsupportedCapability,
        "no executable CPU device in snapshot", Phase::Planning));

  const MemorySpace* host = find_host(snapshot_);
  if (!host)
    return std::unexpected(PlannerError(
        PlannerErrorCode::UnsupportedCapability,
        "no host memory space in snapshot", Phase::Planning));

  // Capability check: every operator and dtype must be executable on the CPU.
  for (const auto& node : graph.nodes) {
    if (!supports_op(*cpu, node.op)) {
      PlannerError e(PlannerErrorCode::UnsupportedCapability,
                     "CPU device cannot execute operator '" + node.name + "'",
                     Phase::Planning);
      e.context.node = node.id;
      return std::unexpected(std::move(e));
    }
    for (TensorId tid : node.inputs) {
      const auto* t = graph.find_tensor(tid);
      // Constants (weights, axes, shapes) are baked into the lowering, not
      // computed, so their dtype does not constrain the device's compute
      // dtypes. Only non-constant inputs (graph inputs / other node outputs)
      // are checked.
      if (t && !t->is_constant && !supports_dtype(*cpu, t->dtype)) {
        PlannerError e(PlannerErrorCode::UnsupportedDType,
                       "CPU device does not support dtype of tensor '" + t->name +
                           "'",
                       Phase::Planning);
        e.context.node = node.id;
        e.context.tensor = tid;
        return std::unexpected(std::move(e));
      }
    }
    for (TensorId tid : node.outputs) {
      const auto* t = graph.find_tensor(tid);
      if (t && !supports_dtype(*cpu, t->dtype)) {
        PlannerError e(PlannerErrorCode::UnsupportedDType,
                       "CPU device does not support dtype of tensor '" + t->name +
                           "'",
                       Phase::Planning);
        e.context.node = node.id;
        e.context.tensor = tid;
        return std::unexpected(std::move(e));
      }
    }
  }

  ExecutionPlan plan;
  plan.schema_version = ExecutionPlan::kSchemaVersion;
  plan.model_fingerprint = model_fingerprint(graph);
  plan.resource_fingerprint = resource_fingerprint(snapshot_);
  plan.resource_version = snapshot_.version;
  plan.devices = snapshot_.devices;
  plan.memory_spaces = snapshot_.memory_spaces;
  plan.tensors = graph.tensors;

  // v0 task granularity matches the compiled unit: the compiler emits one
  // whole-graph JIT entry, so there is exactly one compute task covering every
  // node (jit_entry 0). Per-operator identity is preserved in graph_nodes.
  TaskDesc task;
  task.id = TaskId{0};
  task.kind = TaskKind::Compute;
  task.device = cpu->id;
  task.memory_space = host->id;

  ComputeTaskDesc compute;
  compute.op = OpKind::WholeGraph;
  compute.jit_entry = JitEntryId{0};
  for (const auto& node : graph.nodes)
    compute.graph_nodes.push_back(node.id);
  task.payload = compute;

  for (const auto& t : graph.tensors) {
    if (t.is_graph_input || t.is_constant)
      task.inputs.push_back(t.id);
    if (t.is_graph_output)
      task.outputs.push_back(t.id);
  }

  // Decompose the whole graph into per-node compute requests and sum, since the
  // cost model estimates single operators (it rejects WholeGraph).
  double total_latency_us = 0.0;
  double min_confidence = 1.0;
  for (const auto& node : graph.nodes) {
    uint64_t in_bytes = 0;
    for (TensorId tid : node.inputs)
      if (const auto* t = graph.find_tensor(tid))
        in_bytes += t->size_bytes;
    uint64_t out_bytes = 0;
    for (TensorId tid : node.outputs)
      if (const auto* t = graph.find_tensor(tid))
        out_bytes += t->size_bytes;

    sx::DType dtype = sx::DType::Float32;
    if (!node.outputs.empty())
      if (const auto* t = graph.find_tensor(node.outputs.front()))
        dtype = t->dtype;

    ComputeRequest req;
    req.op = node.op;
    req.input_bytes = in_bytes;
    req.output_bytes = out_bytes;
    req.dtype = dtype;

    auto est = cost_model_.estimate_compute(req, cpu->id);
    if (!est)
      return std::unexpected(est.error());
    total_latency_us += est->estimated_latency_us;
    if (est->confidence < min_confidence)
      min_confidence = est->confidence;
  }
  task.cost = ComputeCost{total_latency_us, min_confidence};

  plan.tasks.push_back(std::move(task));
  plan.execution_order.push_back(TaskId{0});

  plan.estimated_cost.estimated_compute_us = total_latency_us;
  plan.estimated_cost.estimated_transfer_us = 0.0;
  plan.estimated_cost.estimated_memory_us = 0.0;
  if (!plan.estimated_cost.summarize())
    return std::unexpected(PlannerError(
        PlannerErrorCode::CostModelError,
        "plan cost summary overflow", Phase::Planning));

  // Deterministic plan id: FNV-1a over the two fingerprints (v0).
  detail::Fnv1a h;
  h.u64(plan.model_fingerprint);
  h.u64(plan.resource_fingerprint);
  plan.plan_id = h.get();

  return plan;
}

} // namespace sonicboom::planner
