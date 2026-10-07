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

// Transfer scheduler tests (M3): a single host space records no transfers, and a
// cross-space (host → device) tensor requirement emits an explicit Transfer task
// (with the producer as its dependency and the consumer depending on it).

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

  // --- cross-space (host → device) → an explicit Transfer task -------------
  {
    pl::ResourceSnapshot snap2;
    snap2.version = 1;

    pl::Device cpu;
    cpu.id = pl::DeviceId{0};
    cpu.kind = pl::DeviceKind::CPU;
    cpu.name = "cpu";
    cpu.capability.can_execute = true;
    cpu.capability.supported_ops = {pl::OpKind::Relu};
    cpu.capability.supported_dtypes = {sx::DType::Float32};
    snap2.devices.push_back(cpu);

    pl::Device gpu;
    gpu.id = pl::DeviceId{1};
    gpu.kind = pl::DeviceKind::GPU;
    gpu.name = "cuda";
    gpu.capability.can_execute = true;
    gpu.capability.supported_ops = {pl::OpKind::Relu};
    gpu.capability.supported_dtypes = {sx::DType::Float32};
    snap2.devices.push_back(gpu);

    pl::MemorySpace host;
    host.id = pl::MemorySpaceId{0};
    host.owner = pl::DeviceId{0};
    host.kind = pl::MemoryKind::Host;
    host.capacity_bytes = 1u << 20;
    host.alignment_bytes = 8;
    snap2.memory_spaces.push_back(host);

    pl::MemorySpace dev;
    dev.id = pl::MemorySpaceId{1};
    dev.owner = pl::DeviceId{1};
    dev.kind = pl::MemoryKind::DeviceLocal;
    dev.capacity_bytes = 1u << 20;
    dev.alignment_bytes = 8;
    snap2.memory_spaces.push_back(dev);

    pl::AnalyticalCpuCostModel cost2(snap2);

    pl::ExecutionPlan p;
    p.schema_version = pl::ExecutionPlan::kSchemaVersion;
    p.devices = snap2.devices;
    p.memory_spaces = snap2.memory_spaces;

    pl::TensorDesc t;
    t.id = pl::TensorId{0};
    t.name = "x";
    t.shape = {1};
    t.dtype = sx::DType::Float32;
    t.size_bytes = 4;
    p.tensors.push_back(t);

    // Task 0 produces tensor 0 on the host; task 1 consumes it on the device.
    pl::TaskDesc producer;
    producer.id = pl::TaskId{0};
    producer.kind = pl::TaskKind::Compute;
    producer.memory_space = pl::MemorySpaceId{0};
    producer.outputs.push_back(pl::TensorId{0});
    producer.payload = pl::ComputeTaskDesc{};
    p.tasks.push_back(producer);

    pl::TaskDesc consumer;
    consumer.id = pl::TaskId{1};
    consumer.kind = pl::TaskKind::Compute;
    consumer.memory_space = pl::MemorySpaceId{1};
    consumer.inputs.push_back(pl::TensorId{0});
    consumer.payload = pl::ComputeTaskDesc{};
    p.tasks.push_back(consumer);

    p.execution_order.push_back(pl::TaskId{0});
    p.execution_order.push_back(pl::TaskId{1});

    pl::TransferScheduler transfer(cost2);
    auto r = transfer.schedule(p);
    check(r.has_value(), "cross-space plan schedules");
    check(p.tasks.size() == 3, "one transfer task inserted");
    check(p.execution_order.size() == 3, "execution order covers the transfer");

    const pl::TaskDesc* tr = nullptr;
    const pl::TaskDesc* c = nullptr;
    for (const auto& task : p.tasks) {
      if (task.kind == pl::TaskKind::Transfer)
        tr = &task;
      if (task.id == pl::TaskId{1})
        c = &task;
    }
    check(tr != nullptr, "a transfer task exists");
    check(tr && tr->as_transfer() &&
              tr->as_transfer()->source == pl::MemorySpaceId{0} &&
              tr->as_transfer()->destination == pl::MemorySpaceId{1} &&
              tr->as_transfer()->tensor == pl::TensorId{0} &&
              tr->as_transfer()->bytes == 4,
          "transfer payload is host→device for tensor 0");
    check(tr && tr->dependencies.size() == 1 &&
              tr->dependencies[0] == pl::TaskId{0},
          "transfer depends on the producer");
    bool dep_ok = false;
    if (c)
      for (pl::TaskId d : c->dependencies)
        if (d == tr->id)
          dep_ok = true;
    check(dep_ok, "consumer depends on the transfer");
    check(p.execution_order[0] == pl::TaskId{0} &&
              p.execution_order[2] == pl::TaskId{1},
          "execution order interleaves the transfer before the consumer");
  }

  if (g_failures == 0) {
    std::cout << "test_transfer_scheduler OK\n";
    return 0;
  }
  std::cerr << "test_transfer_scheduler FAILED: " << g_failures << " check(s)\n";
  return 1;
}
