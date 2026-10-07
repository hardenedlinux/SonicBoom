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

// Static planner tests (P3): whole-graph task construction, device/memory-space
// assignment, deterministic fingerprints/ids/order, capability rejection, and
// deterministic cost estimation.

#include <sonicboom/planner/planner.h>

#include <sonicboom/sx/ir.h>

#include <cmath>
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

sx::Document doc(const std::string& name) {
  sx::Document d;
  d.version_major = 0;
  d.version_minor = 1;
  d.graph.name = name;
  d.graph.opsets.push_back({"default", 20});
  return d;
}

sx::Node relu(const std::string& in, const std::string& out,
              std::vector<int64_t> dims) {
  sx::Node n;
  n.op_name = "relu";
  n.inputs = {in};
  n.outputs = {{out, ft(std::move(dims))}};
  return n;
}

pl::Device make_cpu(uint32_t id = 0) {
  pl::Device d;
  d.id = pl::DeviceId{id};
  d.kind = pl::DeviceKind::CPU;
  d.name = "cpu";
  d.capability.can_execute = true;
  d.capability.supported_ops = {pl::OpKind::Conv,    pl::OpKind::Relu,
                                pl::OpKind::Add,     pl::OpKind::MaxPool,
                                pl::OpKind::ReduceMean, pl::OpKind::Reshape,
                                pl::OpKind::Gemm};
  d.capability.supported_dtypes = {sx::DType::Float32};
  return d;
}

pl::MemorySpace make_host(uint32_t id = 0, uint32_t owner = 0) {
  pl::MemorySpace m;
  m.id = pl::MemorySpaceId{id};
  m.owner = pl::DeviceId{owner};
  m.kind = pl::MemoryKind::Host;
  m.capacity_bytes = 1ull << 30;
  m.alignment_bytes = 64;
  return m;
}
} // namespace

int main() {
  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "CPU provider builds a snapshot");
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel cost(*snap);
  pl::StaticPlanner planner(*snap, cost);

  // --- simple chain: x -> Relu -> y ----------------------------------------
  auto d = doc("g");
  d.graph.inputs.push_back({"x", ft({1, 3})});
  d.graph.nodes.push_back(relu("x", "y", {1, 3}));
  d.graph.outputs = {"y"};

  auto g = pl::adapt_graph(d);
  check(g.has_value(), "graph adapts");
  if (!g)
    return 1;

  auto plan = planner.plan(*g);
  check(plan.has_value(), "plan succeeds");
  if (!plan) {
    std::cerr << "unexpected: " << plan.error().message << "\n";
    return 1;
  }

  check(plan->schema_version == pl::ExecutionPlan::kSchemaVersion,
        "schema version 1");
  check(plan->tasks.size() == 1, "single whole-graph compute task");
  check(plan->tasks[0].kind == pl::TaskKind::Compute, "task kind = compute");
  const auto* comp = plan->tasks[0].as_compute();
  check(comp != nullptr, "compute payload present");
  check(comp && comp->op == pl::OpKind::WholeGraph, "op = WholeGraph");
  check(comp && comp->jit_entry == pl::JitEntryId{0}, "jit entry 0");
  check(comp && comp->graph_nodes.size() == 1 &&
            comp->graph_nodes[0] == pl::GraphNodeId{0},
        "graph_nodes = [node0]");
  check(plan->tasks[0].device == pl::DeviceId{0}, "task device = cpu(0)");
  check(plan->tasks[0].memory_space == pl::MemorySpaceId{0},
        "task memory space = host(0)");
  check(plan->tasks[0].inputs.size() == 1 &&
            plan->tasks[0].inputs[0] == pl::TensorId{0},
        "task input = x");
  check(plan->tasks[0].outputs.size() == 1 &&
            plan->tasks[0].outputs[0] == pl::TensorId{1},
        "task output = y");
  check(plan->execution_order.size() == 1 &&
            plan->execution_order[0] == pl::TaskId{0},
        "execution order = [task0]");

  check(plan->model_fingerprint == pl::model_fingerprint(*g),
        "model fingerprint recorded");
  check(plan->resource_fingerprint == pl::resource_fingerprint(*snap),
        "resource fingerprint recorded");
  check(plan->devices.size() == snap->devices.size() &&
            plan->memory_spaces.size() == snap->memory_spaces.size() &&
            plan->tensors.size() == 2,
        "devices/memory/tensors copied into plan");

  check(plan->estimated_cost.estimated_compute_us > 0.0, "compute cost > 0");
  check(plan->estimated_cost.estimated_transfer_us == 0.0 &&
            plan->estimated_cost.estimated_memory_us == 0.0,
        "transfer/memory cost zero at this stage");
  check(std::fabs(plan->estimated_cost.estimated_total_us -
                  plan->estimated_cost.estimated_compute_us) < 1e-12,
        "total == compute at this stage");

  // --- determinism ---------------------------------------------------------
  auto plan2 = planner.plan(*g);
  check(plan2.has_value(), "second plan succeeds");
  check(plan2 && plan2->plan_id == plan->plan_id, "plan id deterministic");
  check(plan2 && plan2->estimated_cost.estimated_compute_us ==
                     plan->estimated_cost.estimated_compute_us,
        "compute cost deterministic");
  check(plan2 && plan2->tasks[0].as_compute()->graph_nodes ==
                     plan->tasks[0].as_compute()->graph_nodes,
        "task graph_nodes deterministic");

  // --- longer chain costs strictly more ------------------------------------
  auto d2 = doc("g2");
  d2.graph.inputs.push_back({"x", ft({1, 3})});
  d2.graph.nodes.push_back(relu("x", "h", {1, 3}));
  d2.graph.nodes.push_back(relu("h", "y", {1, 3}));
  d2.graph.outputs = {"y"};
  auto g2 = pl::adapt_graph(d2);
  auto plan3 = planner.plan(*g2);
  check(g2.has_value() && plan3.has_value(), "2-node chain plans");
  check(plan3 && plan3->tasks[0].as_compute()->graph_nodes.size() == 2,
        "2-node chain covers 2 nodes");
  check(plan3 && plan3->estimated_cost.estimated_compute_us >
                     plan->estimated_cost.estimated_compute_us,
        "longer chain costs more");

  // --- no CPU device -> UnsupportedCapability ------------------------------
  {
    pl::ResourceSnapshot s;
    pl::Device gpu;
    gpu.id = pl::DeviceId{0};
    gpu.kind = pl::DeviceKind::GPU;
    gpu.name = "gpu";
    gpu.capability.can_execute = true;
    s.devices.push_back(gpu);
    s.memory_spaces.push_back(make_host(0, 0));
    pl::AnalyticalCpuCostModel cm(s);
    pl::StaticPlanner p(s, cm);
    auto r = p.plan(*g);
    check(!r.has_value() &&
              r.error().code == pl::PlannerErrorCode::UnsupportedCapability,
          "no CPU device rejected");
  }

  // --- unsupported operator -> UnsupportedCapability -----------------------
  {
    pl::ResourceSnapshot s;
    pl::Device cpu = make_cpu();
    cpu.capability.supported_ops = {};  // no ops
    s.devices.push_back(cpu);
    s.memory_spaces.push_back(make_host());
    pl::AnalyticalCpuCostModel cm(s);
    pl::StaticPlanner p(s, cm);
    auto r = p.plan(*g);
    check(!r.has_value() &&
              r.error().code == pl::PlannerErrorCode::UnsupportedCapability,
          "unsupported operator rejected");
  }

  // --- unsupported dtype -> UnsupportedDType -------------------------------
  {
    pl::ResourceSnapshot s;
    pl::Device cpu = make_cpu();
    cpu.capability.supported_dtypes = {sx::DType::Int64};  // no float32
    s.devices.push_back(cpu);
    s.memory_spaces.push_back(make_host());
    pl::AnalyticalCpuCostModel cm(s);
    pl::StaticPlanner p(s, cm);
    auto r = p.plan(*g);
    check(!r.has_value() && r.error().code == pl::PlannerErrorCode::UnsupportedDType,
          "unsupported dtype rejected");
  }

  // --- invalid snapshot -> InvalidConfiguration ----------------------------
  {
    pl::ResourceSnapshot s;
    pl::Device c0 = make_cpu(0);
    pl::Device c1 = make_cpu(0);  // duplicate id
    s.devices.push_back(c0);
    s.devices.push_back(c1);
    s.memory_spaces.push_back(make_host());
    pl::AnalyticalCpuCostModel cm(s);
    pl::StaticPlanner p(s, cm);
    auto r = p.plan(*g);
    check(!r.has_value() &&
              r.error().code == pl::PlannerErrorCode::InvalidConfiguration,
          "invalid snapshot rejected");
  }

  if (g_failures == 0) {
    std::cout << "test_planner OK\n";
    return 0;
  }
  std::cerr << "test_planner FAILED: " << g_failures << " check(s)\n";
  return 1;
}
