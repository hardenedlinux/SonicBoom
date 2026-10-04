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

#include <sonicboom/planner/plan_validator.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace sonicboom::planner {

namespace {

PlannerError invalid(const std::string& msg) {
  return PlannerError(PlannerErrorCode::InvalidPlan, msg, Phase::Validation);
}

} // namespace

std::expected<void, PlannerError> PlanValidator::validate(
    const ExecutionPlan& plan) {
  if (plan.schema_version != ExecutionPlan::kSchemaVersion)
    return std::unexpected(invalid("unsupported schema version " +
                                   std::to_string(plan.schema_version)));

  // --- devices -------------------------------------------------------------
  if (plan.devices.empty())
    return std::unexpected(invalid("plan has no devices"));
  {
    std::unordered_set<uint32_t> ids;
    for (const auto& d : plan.devices)
      if (!ids.insert(d.id.value).second)
        return std::unexpected(invalid("duplicate device id " +
                                       std::to_string(d.id.value)));
  }

  // --- memory spaces -------------------------------------------------------
  if (plan.memory_spaces.empty())
    return std::unexpected(invalid("plan has no memory spaces"));
  {
    std::unordered_set<uint32_t> ids;
    for (const auto& m : plan.memory_spaces) {
      if (!ids.insert(m.id.value).second)
        return std::unexpected(invalid("duplicate memory space id " +
                                       std::to_string(m.id.value)));
      if (!plan.find_device(m.owner))
        return std::unexpected(invalid("memory space " +
                                       std::to_string(m.id.value) +
                                       " references unknown device"));
    }
  }

  // --- tensors -------------------------------------------------------------
  {
    std::unordered_set<uint32_t> ids;
    for (const auto& t : plan.tensors)
      if (!ids.insert(t.id.value).second)
        return std::unexpected(invalid("duplicate tensor id " +
                                       std::to_string(t.id.value)));
  }

  // --- tasks ---------------------------------------------------------------
  if (plan.tasks.empty())
    return std::unexpected(invalid("plan has no tasks"));
  {
    std::unordered_set<uint32_t> ids;
    for (const auto& t : plan.tasks) {
      if (!ids.insert(t.id.value).second)
        return std::unexpected(invalid("duplicate task id " +
                                       std::to_string(t.id.value)));
    }
  }

  for (const auto& t : plan.tasks) {
    if (t.device && !plan.find_device(*t.device))
      return std::unexpected(invalid("task " + std::to_string(t.id.value) +
                                     " references unknown device"));
    if (t.memory_space && !plan.find_memory_space(*t.memory_space))
      return std::unexpected(invalid("task " + std::to_string(t.id.value) +
                                     " references unknown memory space"));

    for (TaskId dep : t.dependencies) {
      if (dep == t.id)
        return std::unexpected(invalid("task " + std::to_string(t.id.value) +
                                       " depends on itself"));
      if (!plan.find_task(dep))
        return std::unexpected(invalid("task " + std::to_string(t.id.value) +
                                       " references unknown dependency " +
                                       std::to_string(dep.value)));
    }

    for (TensorId in : t.inputs)
      if (!plan.find_tensor(in))
        return std::unexpected(invalid("task " + std::to_string(t.id.value) +
                                       " references unknown input tensor"));
    for (TensorId out : t.outputs)
      if (!plan.find_tensor(out))
        return std::unexpected(invalid("task " + std::to_string(t.id.value) +
                                       " references unknown output tensor"));

    switch (t.kind) {
      case TaskKind::Compute:
        if (!t.as_compute())
          return std::unexpected(invalid("compute task missing compute payload"));
        break;
      case TaskKind::Transfer: {
        const auto* tr = t.as_transfer();
        if (!tr)
          return std::unexpected(invalid("transfer task missing transfer payload"));
        if (!plan.find_memory_space(tr->source) ||
            !plan.find_memory_space(tr->destination))
          return std::unexpected(invalid("transfer task has invalid endpoint"));
        if (tr->source == tr->destination)
          return std::unexpected(invalid("transfer task has identical endpoints"));
        if (tr->bytes == 0)
          return std::unexpected(invalid("transfer task has zero bytes"));
        if (!plan.find_tensor(tr->tensor))
          return std::unexpected(invalid("transfer task references unknown tensor"));
        break;
      }
      case TaskKind::Allocate: {
        const auto* al = t.as_allocate();
        if (!al || !plan.find_buffer(al->buffer))
          return std::unexpected(invalid("allocate task references unknown buffer"));
        break;
      }
      case TaskKind::Release: {
        const auto* rl = t.as_release();
        if (!rl || !plan.find_buffer(rl->buffer))
          return std::unexpected(invalid("release task references unknown buffer"));
        break;
      }
      case TaskKind::Synchronize:
        break;
    }
  }

  // --- lifetimes ------------------------------------------------------------
  // Every recorded lifetime must resolve to tensors/tasks that exist in the
  // plan; a corrupted plan must not smuggle dangling references past the trust
  // boundary. (Presence of lifetimes is not required: a static-planner-only
  // plan legitimately has none.)
  for (const auto& lt : plan.lifetimes) {
    if (!plan.find_tensor(lt.tensor))
      return std::unexpected(invalid(
          "lifetime references unknown tensor " + std::to_string(lt.tensor.value)));
    if (lt.producer && !plan.find_task(*lt.producer))
      return std::unexpected(invalid(
          "lifetime references unknown producer task " +
          std::to_string(lt.producer->value)));
    for (TaskId c : lt.consumers)
      if (!plan.find_task(c))
        return std::unexpected(invalid(
            "lifetime references unknown consumer task " +
            std::to_string(c.value)));
    if (!plan.find_task(lt.first_required) || !plan.find_task(lt.last_required))
      return std::unexpected(invalid("lifetime references an unknown required task"));
  }

  // --- allocations ----------------------------------------------------------
  // Unique buffer ids, resolvable memory spaces, non-zero alignment, resolvable
  // assigned tensors, and no tensor hosted by two buffers at once.
  {
    std::unordered_set<uint32_t> buffer_ids;
    std::unordered_set<uint32_t> assigned_tensors;
    for (const auto& a : plan.allocations) {
      if (!buffer_ids.insert(a.buffer.value).second)
        return std::unexpected(invalid(
            "duplicate buffer id " + std::to_string(a.buffer.value)));
      if (!plan.find_memory_space(a.memory_space))
        return std::unexpected(invalid(
            "allocation references unknown memory space " +
            std::to_string(a.memory_space.value)));
      if (a.alignment_bytes == 0)
        return std::unexpected(invalid(
            "allocation has zero alignment"));
      for (TensorId t : a.assigned_tensors) {
        if (!plan.find_tensor(t))
          return std::unexpected(invalid(
              "allocation references unknown tensor " + std::to_string(t.value)));
        if (!assigned_tensors.insert(t.value).second)
          return std::unexpected(invalid(
              "tensor " + std::to_string(t.value) +
              " is hosted by multiple buffers"));
      }
    }
  }

  // --- memory summary -------------------------------------------------------
  // Keys must resolve to recorded memory spaces, and a recorded peak must not
  // exceed the corresponding recorded effective budget.
  for (const auto& [space, _] : plan.memory_summary.peak_bytes)
    if (!plan.find_memory_space(space))
      return std::unexpected(invalid(
          "memory summary references unknown memory space " +
          std::to_string(space.value)));
  for (const auto& [space, _] : plan.memory_summary.effective_budget_bytes)
    if (!plan.find_memory_space(space))
      return std::unexpected(invalid(
          "memory summary references unknown memory space " +
          std::to_string(space.value)));
  for (const auto& [space, peak] : plan.memory_summary.peak_bytes) {
    auto budget_it = plan.memory_summary.effective_budget_bytes.find(space);
    if (budget_it != plan.memory_summary.effective_budget_bytes.end() &&
        peak > budget_it->second)
      return std::unexpected(invalid(
          "memory summary peak exceeds effective budget for memory space " +
          std::to_string(space.value)));
  }

  // --- dependency DAG must be acyclic (Kahn's algorithm) -------------------
  {
    std::unordered_map<uint32_t, uint32_t> indegree;
    for (const auto& t : plan.tasks)
      indegree.emplace(t.id.value, static_cast<uint32_t>(t.dependencies.size()));

    uint32_t removed = 0;
    std::vector<uint32_t> zero;
    for (const auto& [id, deg] : indegree)
      if (deg == 0)
        zero.push_back(id);

    while (!zero.empty()) {
      const uint32_t id = zero.back();
      zero.pop_back();
      ++removed;
      // Removing `id` satisfies its dependents: every task `v` that lists `id`
      // as a dependency loses that incoming edge.
      for (const auto& v : plan.tasks)
        for (TaskId dep : v.dependencies)
          if (dep.value == id && --indegree[v.id.value] == 0)
            zero.push_back(v.id.value);
    }
    if (removed != plan.tasks.size())
      return std::unexpected(invalid("task dependency graph has a cycle"));
  }

  // --- execution order: permutation + topological --------------------------
  if (plan.execution_order.size() != plan.tasks.size())
    return std::unexpected(invalid("execution order is not a permutation of tasks"));

  std::unordered_map<uint32_t, uint32_t> position;
  {
    std::unordered_set<uint32_t> seen;
    uint32_t pos = 0;
    for (TaskId id : plan.execution_order) {
      if (!seen.insert(id.value).second)
        return std::unexpected(invalid("execution order repeats a task"));
      if (!plan.find_task(id))
        return std::unexpected(invalid("execution order references unknown task"));
      position.emplace(id.value, pos++);
    }
    if (seen.size() != plan.tasks.size())
      return std::unexpected(invalid("execution order omits a task"));
  }
  for (const auto& t : plan.tasks)
    for (TaskId dep : t.dependencies)
      if (position[dep.value] >= position[t.id.value])
        return std::unexpected(invalid("execution order violates task dependencies"));

  // --- resource fingerprint matches recorded resources ---------------------
  {
    ResourceSnapshot s;
    s.version = plan.resource_version;
    s.devices = plan.devices;
    s.memory_spaces = plan.memory_spaces;
    if (resource_fingerprint(s) != plan.resource_fingerprint)
      return std::unexpected(PlannerError(
          PlannerErrorCode::ResourceChanged,
          "resource fingerprint does not match recorded devices/memory spaces",
          Phase::Validation));
  }

  return {};
}

} // namespace sonicboom::planner
