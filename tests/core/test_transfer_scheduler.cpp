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

// Transfer scheduler tests (P5): v0 records no transfers for a single memory
// space, and rejects a cross-space tensor requirement it cannot implement.

#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/planner.h>
#include <sonicboom/planner/transfer_scheduler.h>

#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

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

sx::TensorType ft(std::vector<int64_t> dims) {
  return {sx::DType::Float32, sx::Shape{std::move(dims)}};
}
} // namespace

int main() {
  // --- full pipeline: single host space → no transfers ---------------------
  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "snapshot builds");
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel cost(*snap);

  sx::Document d;
  d.version_major = 0;
  d.version_minor = 1;
  d.graph.name = "g";
  d.graph.opsets.push_back({"default", 20});
  d.graph.inputs.push_back({"x", ft({1, 3})});
  sx::Node n;
  n.op_name = "relu";
  n.inputs = {"x"};
  n.outputs = {{"y", ft({1, 3})}};
  d.graph.nodes.push_back(n);
  d.graph.outputs = {"y"};

  auto g = pl::adapt_graph(d);
  check(g.has_value(), "graph adapts");
  if (!g)
    return 1;
  auto plan = pl::plan_execution(*g, *snap, cost);
  check(plan.has_value(), "full pipeline succeeds");
  check(plan && plan->estimated_cost.estimated_transfer_us == 0.0,
        "no transfer cost in v0");
  check(plan && plan->tasks.size() == 1, "only the compute task (no transfers)");

  // --- cross-space requirement → rejected (not silently scheduled) ---------
  {
    pl::ExecutionPlan p;
    p.schema_version = pl::ExecutionPlan::kSchemaVersion;

    pl::Device d0;
    d0.id = pl::DeviceId{0};
    d0.kind = pl::DeviceKind::CPU;
    d0.name = "cpu";
    d0.capability.can_execute = true;
    p.devices.push_back(d0);

    pl::MemorySpace host;
    host.id = pl::MemorySpaceId{0};
    host.owner = pl::DeviceId{0};
    host.kind = pl::MemoryKind::Host;
    host.capacity_bytes = 1u << 20;
    host.alignment_bytes = 8;
    p.memory_spaces.push_back(host);

    pl::MemorySpace pinned;
    pinned.id = pl::MemorySpaceId{1};
    pinned.owner = pl::DeviceId{0};
    pinned.kind = pl::MemoryKind::PinnedHost;
    pinned.capacity_bytes = 1u << 20;
    pinned.alignment_bytes = 8;
    p.memory_spaces.push_back(pinned);

    pl::TensorDesc t;
    t.id = pl::TensorId{0};
    t.name = "x";
    t.shape = {1};
    t.dtype = sx::DType::Float32;
    t.size_bytes = 4;
    p.tensors.push_back(t);

    // Tensor 0 lives in the pinned space…
    pl::BufferAllocation b;
    b.buffer = pl::BufferId{0};
    b.memory_space = pl::MemorySpaceId{1};
    b.size_bytes = 4;
    b.assigned_tensors.push_back(pl::TensorId{0});
    p.allocations.push_back(b);

    // …but the compute task executes in the host space.
    pl::TaskDesc task;
    task.id = pl::TaskId{0};
    task.kind = pl::TaskKind::Compute;
    task.memory_space = pl::MemorySpaceId{0};
    task.inputs.push_back(pl::TensorId{0});
    task.payload = pl::ComputeTaskDesc{};
    p.tasks.push_back(task);
    p.execution_order.push_back(pl::TaskId{0});

    auto r = pl::TransferScheduler::schedule(p);
    check(!r.has_value() &&
              r.error().code == pl::PlannerErrorCode::UnsupportedCapability,
          "cross-space tensor requirement rejected");
  }

  if (g_failures == 0) {
    std::cout << "test_transfer_scheduler OK\n";
    return 0;
  }
  std::cerr << "test_transfer_scheduler FAILED: " << g_failures << " check(s)\n";
  return 1;
}
