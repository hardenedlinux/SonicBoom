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

#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
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

const Device* find_gpu(const ResourceSnapshot& s) {
  const Device* best = nullptr;
  for (const auto& d : s.devices)
    if (d.kind == DeviceKind::GPU && (!best || d.id < best->id))
      best = &d;
  return best;
}

const MemorySpace* find_device_local(const ResourceSnapshot& s) {
  const MemorySpace* best = nullptr;
  for (const auto& m : s.memory_spaces)
    if (m.kind == MemoryKind::DeviceLocal && (!best || m.id < best->id))
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

  const Device* gpu = find_gpu(snapshot_);
  const MemorySpace* device_local = find_device_local(snapshot_);

  // The effective backend of a node: its explicit per-node override (the Gemma 4
  // plan emitter pins everything to Sonic), else the route_op default.
  auto effective_backend = [](const GraphNodeDesc& node) noexcept {
    return node.backend.value_or(route_op(node.op));
  };

  // M3 placement seam: route a node to the GPU (device-local) only when it
  // dispatches to the Sonic backend AND a usable GPU device that supports its op
  // is present; every other backend (MLIR JIT, native-torch dispatcher) is
  // CPU-only in v0, so those nodes stay on the CPU (host). Tying placement to
  // backend keeps it consistent with routing — the shared `Add` operator is MLIR
  // by default (→ CPU) but is pinned to Sonic by the Gemma 4 emitter (→ GPU).
  auto place = [&](const GraphNodeDesc& node)
      -> std::pair<const Device*, const MemorySpace*> {
    if (effective_backend(node) == BackendTag::Sonic && gpu &&
        gpu->capability.can_execute && device_local &&
        supports_op(*gpu, node.op))
      return {gpu, device_local};
    return {cpu, host};
  };

  // Capability check: every operator and dtype must be executable on its
  // assigned device.
  for (const auto& node : graph.nodes) {
    const Device* dev = place(node).first;
    if (!supports_op(*dev, node.op)) {
      PlannerError e(PlannerErrorCode::UnsupportedCapability,
                     "device cannot execute operator '" + node.name + "'",
                     Phase::Planning);
      e.context.node = node.id;
      return std::unexpected(std::move(e));
    }
    for (TensorId tid : node.inputs) {
      const auto* t = graph.find_tensor(tid);
      // Constants (weights, axes, shapes) are baked into the lowering, and graph
      // inputs are externally-provided data (not computed), so neither dtype
      // constrains the device's compute dtypes. Only computed tensors (other
      // nodes' outputs) are checked.
      if (t && !t->is_constant && !t->is_graph_input &&
          !supports_dtype(*dev, t->dtype)) {
        PlannerError e(PlannerErrorCode::UnsupportedDType,
                       "device does not support dtype of tensor '" + t->name +
                           "'",
                       Phase::Planning);
        e.context.node = node.id;
        e.context.tensor = tid;
        return std::unexpected(std::move(e));
      }
    }
    for (TensorId tid : node.outputs) {
      const auto* t = graph.find_tensor(tid);
      if (t && !supports_dtype(*dev, t->dtype)) {
        PlannerError e(PlannerErrorCode::UnsupportedDType,
                       "device does not support dtype of tensor '" + t->name +
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

  // Per-node compute cost, shared by both task models below (the cost model
  // estimates single operators and rejects WholeGraph).
  auto node_cost = [&](const GraphNodeDesc& node, DeviceId device)
      -> std::expected<ComputeCost, PlannerError> {
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
    return cost_model_.estimate_compute(req, device);
  };

  double total_latency_us = 0.0;
  double min_confidence = 1.0;

  // Does any node dispatch to a non-MLIR backend (native-torch or Sonic)?
  bool has_non_mlir = false;
  for (const auto& node : graph.nodes)
    if (effective_backend(node) != BackendTag::Mlir)
      has_non_mlir = true;

  if (!has_non_mlir) {
    // Homogeneous MLIR graph: one whole-graph compute task (jit_entry 0), as
    // before co-execution. Per-operator identity is preserved in graph_nodes.
    TaskDesc task;
    task.id = TaskId{0};
    task.kind = TaskKind::Compute;
    task.device = cpu->id;
    task.memory_space = host->id;

    ComputeTaskDesc compute;
    compute.op = OpKind::WholeGraph;
    compute.backend = BackendTag::Mlir;
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

    for (const auto& node : graph.nodes) {
      auto est = node_cost(node, cpu->id);
      if (!est)
        return std::unexpected(est.error());
      total_latency_us += est->estimated_latency_us;
      min_confidence = std::min(min_confidence, est->confidence);
    }
    task.cost = ComputeCost{total_latency_us, min_confidence};

    plan.tasks.push_back(std::move(task));
    plan.execution_order.push_back(TaskId{0});
  } else {
    // Mixed graph: group topologically-consecutive same-backend nodes into
    // execution regions. A NativeTorch node is always its own single-node
    // region (the dispatcher runs one node per task); a run of consecutive MLIR
    // nodes merges into one region compiled from a single slice, so region-
    // internal tensors never cross a task boundary. Node order is topological
    // (every node's inputs are inputs/parameters/earlier outputs), so
    // dependencies point strictly backward and region order is a valid
    // execution order.
    std::unordered_map<uint32_t, TaskId> producer;  // tensor -> producing task

    // 1. Partition nodes into regions (greedy scan over topological order).
    struct Region {
      BackendTag backend;
      std::vector<const GraphNodeDesc*> nodes;
    };
    std::vector<Region> regions;
    for (const auto& node : graph.nodes) {
      const BackendTag b = effective_backend(node);
      if (!regions.empty() && regions.back().backend == BackendTag::Mlir &&
          b == BackendTag::Mlir) {
        regions.back().nodes.push_back(&node);  // extend the MLIR run
      } else {
        Region r;
        r.backend = b;
        r.nodes.push_back(&node);
        regions.push_back(std::move(r));
      }
    }

    std::unordered_set<uint32_t> graph_outputs;
    for (TensorId o : graph.outputs)
      graph_outputs.insert(o.value);

    // 2. Emit one compute task per region.
    uint32_t next_task = 0;
    for (const Region& r : regions) {
      TaskDesc task;
      task.id = TaskId{next_task};
      task.kind = TaskKind::Compute;
      // Placement: a region is a single node (non-MLIR), so its device/memory
      // space follows the node's backend + op.
      auto placed = place(*r.nodes.front());
      task.device = placed.first->id;
      task.memory_space = placed.second->id;

      ComputeTaskDesc compute;
      compute.backend = r.backend;
      // Representative op (WholeGraph is reserved for the homogeneous fast
      // path); graph_nodes below carries the full region membership.
      compute.op = r.nodes.front()->op;
      compute.jit_entry = JitEntryId{0};  // each region compiled separately
      for (const auto* n : r.nodes)
        compute.graph_nodes.push_back(n->id);
      task.payload = compute;

      // Region-internal produced set: tensors produced by any region node.
      std::unordered_set<uint32_t> produced;
      std::unordered_set<uint32_t> region_nodes;
      for (const auto* n : r.nodes) {
        region_nodes.insert(n->id.value);
        for (TensorId o : n->outputs)
          produced.insert(o.value);
      }

      // External inputs: node inputs not produced within the region, first-
      // reference order, deduplicated.
      std::unordered_set<uint32_t> seen_in;
      for (const auto* n : r.nodes)
        for (TensorId in : n->inputs)
          if (!produced.count(in.value) && seen_in.insert(in.value).second)
            task.inputs.push_back(in);

      // External outputs: region-produced tensors that are graph outputs or are
      // consumed by a node outside the region, in node order then output order.
      std::unordered_set<uint32_t> outside_consumers;
      for (const auto& node : graph.nodes) {
        if (region_nodes.count(node.id.value))
          continue;
        for (TensorId in : node.inputs)
          outside_consumers.insert(in.value);
      }
      for (const auto* n : r.nodes)
        for (TensorId o : n->outputs)
          if (outside_consumers.count(o.value) || graph_outputs.count(o.value))
            task.outputs.push_back(o);

      // Dependencies: producers of the external inputs (strictly earlier).
      std::unordered_set<uint32_t> deps;
      for (TensorId in : task.inputs) {
        auto it = producer.find(in.value);
        if (it != producer.end())
          deps.insert(it->second.value);
      }
      for (uint32_t d : deps)
        task.dependencies.push_back(TaskId{d});

      // Region cost: sum per-node costs (the model estimates single operators).
      double region_latency = 0.0;
      double region_conf = 1.0;
      for (const auto* n : r.nodes) {
        auto est = node_cost(*n, place(*n).first->id);
        if (!est)
          return std::unexpected(est.error());
        region_latency += est->estimated_latency_us;
        region_conf = std::min(region_conf, est->confidence);
      }
      task.cost = ComputeCost{region_latency, region_conf};
      total_latency_us += region_latency;
      min_confidence = std::min(min_confidence, region_conf);

      plan.tasks.push_back(std::move(task));
      plan.execution_order.push_back(TaskId{next_task});
      for (const auto* n : r.nodes)
        for (TensorId o : n->outputs)
          producer[o.value] = TaskId{next_task};
      if (!next_id(next_task))
        return std::unexpected(PlannerError(
            PlannerErrorCode::InternalError,
            "task id counter overflow", Phase::Planning));
    }
  }

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
