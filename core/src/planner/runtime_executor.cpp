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

#include "device_pool.h"

#include <sonicboom/planner/plan_record.h>

#include <cstddef>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
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

  // 2. (Plan correctness is already validated once at build time by
  //     plan_execution in pipeline.cpp; the plan is immutable afterward.
  //     Re-running PlanValidator::validate here on every execute() — the hot
  //     per-token decode path — is O(nodes × edges) redundant work.)

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
  //    Graph inputs are host scalars (token id / position) even on the CUDA
  //    path: they are read directly by the backends, never transferred.
  std::map<uint32_t, TensorValue> values;  // TensorId -> runtime value
  std::map<BufferId, uint64_t> allocated;  // buffer -> device handle (Allocate)
  // tensor -> last task that requires it alive (from the memory planner's
  // lifetimes). The device pool reuses a buffer only after it is released, so a
  // device-resident activation is returned to the pool as soon as its last
  // consumer has read it — keeping the peak device working set bounded instead of
  // holding every intermediate until the end of the step (which overflows the
  // device for a batched prefill graph, whose activations are [d, n]).
  std::map<uint32_t, TaskId> last_use;
  for (const auto& lt : plan.lifetimes)
    last_use[lt.tensor.value] = lt.last_required;
  std::size_t in_i = 0;
  for (const auto& t : plan.tensors)
    if (t.is_graph_input)
      values[t.id.value] = TensorValue::from_host(inputs[in_i++]);

  // 4b. Optional execution record (plan_record.h). Off by default: with no
  //     recorder installed — recording_requested() and recorder() both cheap —
  //     nothing below runs and the loop is as it was. The record is assembled
  //     here because this is the one place where the plan's task descriptions,
  //     the plan's tensor/buffer assignments, and the kernels the backend
  //     launches (through plan_hook.h) are all in scope at once.
  PlanRecorder* rec = recording_requested() ? recorder() : nullptr;
  std::map<uint32_t, const BufferAllocation*> tensor_buffer;
  uint32_t record_step = 0;
  if (rec) {
    for (const auto& ba : plan.allocations)
      for (TensorId t : ba.assigned_tensors)
        tensor_buffer.emplace(t.value, &ba);
  }
  // Record one tensor the current step touches, with its buffer's logical id
  // and byte range. Constants are recorded too (role "const"): a quantized
  // weight is a constant input, and its id/dtype are what identify the step's
  // arithmetic.
  auto note_tensor = [&](std::string_view io, uint32_t index, TensorId id) {
    const TensorDesc* td = plan.find_tensor(id);
    if (!td) return;
    BufferRecord b;
    b.index = index;
    b.tensor = id.value;
    b.role = td->is_constant     ? "const"
             : td->is_graph_input ? "in_graph"
             : td->is_graph_output ? "out_graph"
                                   : std::string(io);
    b.shape = shape_text(td->shape);
    b.dtype = sx::dtype_name(td->dtype);
    b.size_bytes = td->size_bytes;
    if (auto it = tensor_buffer.find(id.value); it != tensor_buffer.end()) {
      b.buffer = it->second->buffer.value;
      b.offset = it->second->offset_bytes;
      b.size_bytes = it->second->size_bytes;
    }
    rec->add_buffer(std::move(b));
  };

  // 5. Walk execution_order: dispatch each task to its routed backend, handing
  //    intermediate tensors between tasks by TensorId. Transfer tasks move a
  //    value between host and device; the device buffers are pooled and returned
  //    at the end of the step (below).
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

        std::vector<TensorValue> task_inputs;
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

        if (rec) {
          const GraphNodeId head = compute->graph_nodes.empty()
                                       ? GraphNodeId{0}
                                       : compute->graph_nodes.front();
          rec->begin_step(record_step++, tid.value, head.value,
                          op_kind_name(compute->op),
                          backend_tag_name(compute->backend), std::string_view(),
                          compute->attributes, nullptr);
          for (std::size_t k = 0; k < task->inputs.size(); ++k)
            note_tensor("in", uint32_t(k), task->inputs[k]);
          for (std::size_t k = 0; k < task->outputs.size(); ++k)
            note_tensor("out", uint32_t(k), task->outputs[k]);
        }

        auto res = backend->execute(*task, task_inputs);
        if (rec) rec->end_step();  // before any early return, so the step closes
        if (!res)
          return std::unexpected(res.error());
        if (res->size() != task->outputs.size())
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "backend produced an unexpected number of outputs",
              Phase::Execution));
        for (std::size_t k = 0; k < task->outputs.size(); ++k)
          values[task->outputs[k].value] = std::move((*res)[k]);

        // Release an input's device buffer once this task is its last consumer,
        // so a dead activation's memory is reused by later tasks. Graph outputs
        // and constants are skipped (constants are baked, outputs are collected
        // at the end); a tensor whose last consumer is a Transfer task is left
        // for that task's own release.
        for (TensorId in : task->inputs) {
          const auto* td = plan.find_tensor(in);
          if (!td || td->is_constant || td->is_graph_output)
            continue;
          auto lu = last_use.find(in.value);
          if (lu == last_use.end() || lu->second != tid)
            continue;
          auto v = values.find(in.value);
          if (v == values.end() || !v->second.on_device)
            continue;
          device::release(v->second.device.handle, v->second.device.size_bytes);
          v->second.on_device = false;
          v->second.device.handle = 0;
          v->second.device.size_bytes = 0;
        }
        break;
      }
      case TaskKind::Transfer: {
        const auto* tr = task->as_transfer();
        if (!tr)
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "transfer task has no transfer payload", Phase::Execution));
        const MemorySpace* src = plan.find_memory_space(tr->source);
        const MemorySpace* dst = plan.find_memory_space(tr->destination);
        const bool src_dev = src && src->kind != MemoryKind::Host;
        const bool dst_dev = dst && dst->kind != MemoryKind::Host;
        auto v = values.find(tr->tensor.value);
        if (v == values.end())
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "transfer task references a missing value", Phase::Execution));

        if (!src_dev && dst_dev) {
          // Host -> device: upload, drop the host bytes.
          if (v->second.on_device)
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure,
                "H2D transfer found an already-device value", Phase::Execution));
          uint64_t handle = device::acquire(tr->bytes);
          if (!handle)
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure,
                "device buffer allocation failed", Phase::Execution));
          if (!device::upload(v->second.host.data(), handle, tr->bytes)) {
            device::release(handle, tr->bytes);
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure, "H2D copy failed",
                Phase::Execution));
          }
          values[tr->tensor.value] = TensorValue::from_device(handle, tr->bytes);
        } else if (src_dev && !dst_dev) {
          // Device -> host: download, release the device buffer.
          if (!v->second.on_device)
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure,
                "D2H transfer found a host value", Phase::Execution));
          sx::Bytes host(tr->bytes);
          if (!device::download(v->second.device.handle, host.data(), tr->bytes)) {
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure, "D2H copy failed",
                Phase::Execution));
          }
          device::release(v->second.device.handle, v->second.device.size_bytes);
          values[tr->tensor.value] = TensorValue::from_host(std::move(host));
        } else if (src_dev && dst_dev) {
          // Device -> device: acquire a destination buffer and copy.
          if (!v->second.on_device)
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure,
                "D2D transfer found a host value", Phase::Execution));
          uint64_t handle = device::acquire(tr->bytes);
          if (!handle)
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure,
                "device buffer allocation failed", Phase::Execution));
          if (!device::device_copy(v->second.device.handle, handle, tr->bytes)) {
            device::release(handle, tr->bytes);
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure, "D2D copy failed",
                Phase::Execution));
          }
          device::release(v->second.device.handle, v->second.device.size_bytes);
          values[tr->tensor.value] = TensorValue::from_device(handle, tr->bytes);
        } else {
          // Host -> host: a plain byte copy (defensive; the v0 planner never
          // emits this).
          if (v->second.on_device)
            return std::unexpected(RuntimeError(
                RuntimeErrorCode::TransferFailure,
                "host transfer found a device value", Phase::Execution));
          sx::Bytes host(tr->bytes);
          std::memcpy(host.data(), v->second.host.data(), tr->bytes);
          values[tr->tensor.value] = TensorValue::from_host(std::move(host));
        }
        break;
      }
      case TaskKind::Synchronize:
        if (!device::synchronize())
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::TransferFailure, "device synchronize failed",
              Phase::Execution));
        break;
      case TaskKind::Allocate: {
        const auto* al = task->as_allocate();
        if (!al)
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "allocate task has no payload", Phase::Execution));
        if (allocated.find(al->buffer) != allocated.end())
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "buffer already allocated", Phase::Execution));
        uint64_t handle = device::acquire(al->size_bytes);
        if (!handle)
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::TransferFailure, "device allocation failed",
              Phase::Execution));
        allocated[al->buffer] = handle;
        break;
      }
      case TaskKind::Release: {
        const auto* rl = task->as_release();
        if (!rl)
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "release task has no payload", Phase::Execution));
        auto it = allocated.find(rl->buffer);
        if (it == allocated.end())
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::InvalidPlan,
              "release task references an unallocated buffer", Phase::Execution));
        device::release(it->second, /*bytes=*/0);
        allocated.erase(it);
        break;
      }
    }
  }

  // 6. Gather the graph-output buffers in plan tensor order. A device-resident
  //    output is downloaded to host first (its device buffer is returned to the
  //    pool). The plan's transfer scheduler normally emits an explicit D2H for
  //    such outputs; this is a defensive fallback.
  std::vector<sx::Bytes> outputs;
  outputs.reserve(n_outputs);
  for (const auto& t : plan.tensors)
    if (t.is_graph_output) {
      auto v = values.find(t.id.value);
      if (v == values.end())
        return std::unexpected(RuntimeError(
            RuntimeErrorCode::InvalidPlan,
            "graph output tensor has no computed value", Phase::Execution));
      if (v->second.on_device) {
        sx::Bytes host(v->second.device.size_bytes);
        if (!device::download(v->second.device.handle, host.data(),
                              v->second.device.size_bytes))
          return std::unexpected(RuntimeError(
              RuntimeErrorCode::TransferFailure,
              "graph output D2H failed", Phase::Execution));
        device::release(v->second.device.handle, v->second.device.size_bytes);
        v->second = TensorValue::from_host(std::move(host));
      }
      outputs.push_back(std::move(v->second.host));
    }

  // 7. Return every still-live device buffer to the pool so the next step's
  //    working set reuses the same device memory instead of reallocating.
  for (auto& [id, val] : values)
    if (val.on_device)
      device::release(val.device.handle, val.device.size_bytes);

  if (outputs.size() != n_outputs)
    return std::unexpected(RuntimeError(
        RuntimeErrorCode::InvalidPlan,
        "execution produced an unexpected number of outputs", Phase::Execution));

  return ExecutionResult{std::move(outputs)};
}

} // namespace sonicboom::planner
