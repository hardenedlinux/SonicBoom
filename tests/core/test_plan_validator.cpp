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

// PlanValidator tests (P5 §19.4): a valid plan passes; duplicate ids, missing
// dependencies, cycles, missing devices, invalid transfer endpoints, order
// violations, and fingerprint mismatches are each rejected.

#include <sonicboom/planner/plan_validator.h>

#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace pl = sonicboom::planner;
namespace sx = sonicboom::sx;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

// A minimal, valid two-task acyclic plan (task1 depends on task0).
pl::ExecutionPlan make_base() {
  pl::ExecutionPlan p;
  p.schema_version = pl::ExecutionPlan::kSchemaVersion;

  pl::Device d;
  d.id = pl::DeviceId{0};
  d.kind = pl::DeviceKind::CPU;
  d.name = "cpu";
  d.capability.can_execute = true;
  p.devices.push_back(d);

  pl::MemorySpace m;
  m.id = pl::MemorySpaceId{0};
  m.owner = pl::DeviceId{0};
  m.kind = pl::MemoryKind::Host;
  m.capacity_bytes = 1u << 20;
  m.alignment_bytes = 8;
  p.memory_spaces.push_back(m);

  pl::TensorDesc t;
  t.id = pl::TensorId{0};
  t.name = "x";
  t.shape = {1};
  t.dtype = sx::DType::Float32;
  t.size_bytes = 4;
  p.tensors.push_back(t);

  pl::TaskDesc t0;
  t0.id = pl::TaskId{0};
  t0.kind = pl::TaskKind::Compute;
  t0.payload = pl::ComputeTaskDesc{};
  p.tasks.push_back(t0);

  pl::TaskDesc t1;
  t1.id = pl::TaskId{1};
  t1.kind = pl::TaskKind::Compute;
  t1.payload = pl::ComputeTaskDesc{};
  t1.dependencies = {pl::TaskId{0}};
  p.tasks.push_back(t1);

  p.execution_order = {pl::TaskId{0}, pl::TaskId{1}};

  pl::ResourceSnapshot s;
  s.version = 0;
  s.devices = p.devices;
  s.memory_spaces = p.memory_spaces;
  p.resource_fingerprint = pl::resource_fingerprint(s);
  p.resource_version = 0;
  p.model_fingerprint = 0x1234;
  return p;
}
} // namespace

int main() {
  // --- valid plan passes ---------------------------------------------------
  {
    auto p = make_base();
    check(pl::PlanValidator::validate(p).has_value(), "valid plan passes");
  }

  // --- duplicate task id ---------------------------------------------------
  {
    auto p = make_base();
    p.tasks.push_back(p.tasks[0]);  // second task with id 0
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "duplicate task id rejected");
  }

  // --- missing dependency --------------------------------------------------
  {
    auto p = make_base();
    p.tasks[1].dependencies = {pl::TaskId{99}};
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "missing dependency rejected");
  }

  // --- dependency cycle ----------------------------------------------------
  {
    auto p = make_base();
    p.tasks[0].dependencies = {pl::TaskId{1}};
    p.tasks[1].dependencies = {pl::TaskId{0}};
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "dependency cycle rejected");
  }

  // --- missing device ------------------------------------------------------
  {
    auto p = make_base();
    p.tasks[0].device = pl::DeviceId{99};
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "missing device rejected");
  }

  // --- invalid transfer endpoints (identical) -----------------------------
  {
    auto p = make_base();
    pl::TaskDesc tr;
    tr.id = pl::TaskId{2};
    tr.kind = pl::TaskKind::Transfer;
    tr.payload = pl::TransferTaskDesc{pl::TensorId{0}, pl::MemorySpaceId{0},
                                      pl::MemorySpaceId{0}, 4};
    p.tasks.push_back(tr);
    p.execution_order.push_back(pl::TaskId{2});
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "identical transfer endpoints rejected");
  }

  // --- incorrect resource fingerprint -------------------------------------
  {
    auto p = make_base();
    p.resource_fingerprint = 0;  // tampered
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() &&
              r.error().code == pl::PlannerErrorCode::ResourceChanged,
          "resource fingerprint mismatch rejected");
  }

  // --- execution order violates dependencies ------------------------------
  {
    auto p = make_base();  // task1 depends on task0
    p.execution_order = {pl::TaskId{1}, pl::TaskId{0}};
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "order violation rejected");
  }

  // --- valid plan with lifetimes + allocations + memory summary passes -----
  {
    auto p = make_base();
    pl::TensorLifetime lt;
    lt.tensor = pl::TensorId{0};
    lt.first_required = pl::TaskId{0};
    lt.last_required = pl::TaskId{0};
    p.lifetimes.push_back(lt);
    pl::BufferAllocation a;
    a.buffer = pl::BufferId{0};
    a.memory_space = pl::MemorySpaceId{0};
    a.alignment_bytes = 8;
    a.size_bytes = 4;
    a.assigned_tensors.push_back(pl::TensorId{0});
    p.allocations.push_back(a);
    p.memory_summary.peak_bytes[pl::MemorySpaceId{0}] = 4;
    p.memory_summary.effective_budget_bytes[pl::MemorySpaceId{0}] = 1u << 20;
    check(pl::PlanValidator::validate(p).has_value(),
          "valid plan with lifetimes/allocations passes");
  }

  // --- lifetime references unknown tensor ----------------------------------
  {
    auto p = make_base();
    pl::TensorLifetime lt;
    lt.tensor = pl::TensorId{99};
    lt.first_required = pl::TaskId{0};
    lt.last_required = pl::TaskId{0};
    p.lifetimes.push_back(lt);
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "lifetime unknown tensor rejected");
  }

  // --- lifetime references unknown producer task ---------------------------
  {
    auto p = make_base();
    pl::TensorLifetime lt;
    lt.tensor = pl::TensorId{0};
    lt.producer = pl::TaskId{99};
    lt.first_required = pl::TaskId{0};
    lt.last_required = pl::TaskId{0};
    p.lifetimes.push_back(lt);
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "lifetime unknown producer rejected");
  }

  // --- duplicate buffer id --------------------------------------------------
  {
    auto p = make_base();
    pl::BufferAllocation a0;
    a0.buffer = pl::BufferId{0};
    a0.memory_space = pl::MemorySpaceId{0};
    a0.alignment_bytes = 8;
    a0.size_bytes = 4;
    p.allocations.push_back(a0);
    p.allocations.push_back(a0);  // same buffer id twice
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "duplicate buffer id rejected");
  }

  // --- allocation references unknown memory space --------------------------
  {
    auto p = make_base();
    pl::BufferAllocation a;
    a.buffer = pl::BufferId{0};
    a.memory_space = pl::MemorySpaceId{99};
    a.alignment_bytes = 8;
    a.size_bytes = 4;
    p.allocations.push_back(a);
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "allocation unknown memory space rejected");
  }

  // --- allocation has zero alignment ---------------------------------------
  {
    auto p = make_base();
    pl::BufferAllocation a;
    a.buffer = pl::BufferId{0};
    a.memory_space = pl::MemorySpaceId{0};
    a.alignment_bytes = 0;
    a.size_bytes = 4;
    p.allocations.push_back(a);
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "zero-alignment allocation rejected");
  }

  // --- tensor hosted by two buffers ----------------------------------------
  {
    auto p = make_base();
    pl::BufferAllocation a0;
    a0.buffer = pl::BufferId{0};
    a0.memory_space = pl::MemorySpaceId{0};
    a0.alignment_bytes = 8;
    a0.size_bytes = 4;
    a0.assigned_tensors.push_back(pl::TensorId{0});
    pl::BufferAllocation a1 = a0;
    a1.buffer = pl::BufferId{1};  // distinct buffer, same tensor
    p.allocations.push_back(a0);
    p.allocations.push_back(a1);
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "tensor in two buffers rejected");
  }

  // --- memory summary references unknown memory space ----------------------
  {
    auto p = make_base();
    p.memory_summary.peak_bytes[pl::MemorySpaceId{99}] = 100;
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "memory summary unknown space rejected");
  }

  // --- memory summary peak exceeds budget ----------------------------------
  {
    auto p = make_base();
    p.memory_summary.peak_bytes[pl::MemorySpaceId{0}] = 1u << 20;
    p.memory_summary.effective_budget_bytes[pl::MemorySpaceId{0}] = 100;
    auto r = pl::PlanValidator::validate(p);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::InvalidPlan,
          "memory summary peak over budget rejected");
  }

  if (g_failures == 0) {
    std::cout << "test_plan_validator OK\n";
    return 0;
  }
  std::cerr << "test_plan_validator FAILED: " << g_failures << " check(s)\n";
  return 1;
}
