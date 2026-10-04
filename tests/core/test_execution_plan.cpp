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

// ExecutionPlan + task payload tests (P2): the plan is a plain-data contract.
// This file covers the data structure itself — schema version, payload accessors,
// and the find_* helpers on a hand-assembled valid plan. Structural validation
// (duplicate ids, cycles, missing deps, fingerprint mismatch) lives in the
// PlanValidator tests (P5).

#include <sonicboom/planner/execution_plan.h>

#include <iostream>

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
} // namespace

int main() {
  // --- schema version defaults ---------------------------------------------
  pl::ExecutionPlan p;
  check(p.schema_version == pl::ExecutionPlan::kSchemaVersion,
        "schema version defaults to 1");
  check(p.plan_id == 0 && p.model_fingerprint == 0 &&
            p.resource_fingerprint == 0,
        "fingerprints default to 0");

  // --- task payload accessors ----------------------------------------------
  pl::TaskDesc compute;
  compute.id = pl::TaskId{0};
  compute.kind = pl::TaskKind::Compute;
  compute.payload = pl::ComputeTaskDesc{pl::OpKind::WholeGraph,
                                        {pl::GraphNodeId{0}}, pl::JitEntryId{0}};
  check(compute.as_compute() != nullptr, "compute payload accessible");
  check(compute.as_transfer() == nullptr && compute.as_allocate() == nullptr &&
            compute.as_release() == nullptr,
        "non-compute accessors null for compute task");
  check(compute.as_compute()->op == pl::OpKind::WholeGraph,
        "compute op = WholeGraph");
  check(compute.as_compute()->graph_nodes.size() == 1 &&
            compute.as_compute()->jit_entry == pl::JitEntryId{0},
        "compute payload fields correct");

  pl::TaskDesc transfer;
  transfer.id = pl::TaskId{1};
  transfer.kind = pl::TaskKind::Transfer;
  transfer.payload = pl::TransferTaskDesc{pl::TensorId{0}, pl::MemorySpaceId{0},
                                          pl::MemorySpaceId{1}, 128};
  check(transfer.as_transfer() != nullptr && transfer.as_compute() == nullptr,
        "transfer payload accessible, compute null");
  check(transfer.as_transfer()->bytes == 128, "transfer bytes correct");

  pl::TaskDesc alloc;
  alloc.kind = pl::TaskKind::Allocate;
  alloc.payload = pl::AllocateTaskDesc{pl::BufferId{0}, pl::MemorySpaceId{0},
                                       64, 16};
  check(alloc.as_allocate() != nullptr, "allocate payload accessible");
  check(alloc.as_allocate()->alignment_bytes == 16, "allocate alignment correct");

  pl::TaskDesc rel;
  rel.kind = pl::TaskKind::Release;
  rel.payload = pl::ReleaseTaskDesc{pl::BufferId{0}};
  check(rel.as_release() != nullptr, "release payload accessible");
  check(rel.as_release()->buffer == pl::BufferId{0}, "release buffer correct");

  pl::TaskDesc empty;
  empty.kind = pl::TaskKind::Synchronize;
  check(empty.as_compute() == nullptr && empty.as_transfer() == nullptr &&
            empty.as_allocate() == nullptr && empty.as_release() == nullptr,
        "synchronize task has no payload");

  // --- find_* helpers on a hand-assembled plan -----------------------------
  pl::ExecutionPlan plan;
  plan.plan_id = 42;
  plan.model_fingerprint = 0xDEADBEEF;
  plan.resource_fingerprint = 0xCAFEF00D;

  pl::Device cpu;
  cpu.id = pl::DeviceId{0};
  cpu.kind = pl::DeviceKind::CPU;
  plan.devices.push_back(cpu);

  pl::MemorySpace host;
  host.id = pl::MemorySpaceId{0};
  host.kind = pl::MemoryKind::Host;
  host.capacity_bytes = 1ull << 30;
  plan.memory_spaces.push_back(host);

  pl::TensorDesc x;
  x.id = pl::TensorId{0};
  x.name = "x";
  x.shape = {1, 3};
  x.dtype = sx::DType::Float32;
  x.size_bytes = 12;
  plan.tensors.push_back(x);

  pl::TaskDesc t0;
  t0.id = pl::TaskId{0};
  t0.kind = pl::TaskKind::Compute;
  t0.payload = pl::ComputeTaskDesc{};
  plan.tasks.push_back(t0);

  pl::BufferAllocation b0;
  b0.buffer = pl::BufferId{0};
  b0.memory_space = pl::MemorySpaceId{0};
  b0.size_bytes = 12;
  plan.allocations.push_back(b0);

  check(plan.find_tensor(pl::TensorId{0}) != nullptr, "find_tensor hit");
  check(plan.find_tensor(pl::TensorId{1}) == nullptr, "find_tensor miss");
  check(plan.find_task(pl::TaskId{0}) != nullptr, "find_task hit");
  check(plan.find_task(pl::TaskId{9}) == nullptr, "find_task miss");
  check(plan.find_buffer(pl::BufferId{0}) != nullptr, "find_buffer hit");
  check(plan.find_buffer(pl::BufferId{2}) == nullptr, "find_buffer miss");

  if (g_failures == 0) {
    std::cout << "test_execution_plan OK\n";
    return 0;
  }
  std::cerr << "test_execution_plan FAILED: " << g_failures << " check(s)\n";
  return 1;
}
