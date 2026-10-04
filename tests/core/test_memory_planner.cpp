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

// Memory planner tests (P4): lifetime computation, per-tensor buffer allocation
// with alignment, peak working-set summary, budget enforcement, and memory cost.

#include <sonicboom/planner/memory_planner.h>
#include <sonicboom/planner/planner.h>

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
  d.capability.supported_ops = {pl::OpKind::Relu, pl::OpKind::Add};
  d.capability.supported_dtypes = {sx::DType::Float32};
  return d;
}

pl::MemorySpace make_host(uint64_t capacity = 1ull << 30,
                          uint64_t align = 64) {
  pl::MemorySpace m;
  m.id = pl::MemorySpaceId{0};
  m.owner = pl::DeviceId{0};
  m.kind = pl::MemoryKind::Host;
  m.capacity_bytes = capacity;
  m.alignment_bytes = align;
  return m;
}
} // namespace

int main() {
  // --- normal case: simple chain ------------------------------------------
  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "CPU snapshot builds");
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel cost(*snap);
  pl::StaticPlanner planner(*snap, cost);
  pl::MemoryPlanner mem(cost);

  auto d = doc("g");
  d.graph.inputs.push_back({"x", ft({1, 3})});
  d.graph.nodes.push_back(relu("x", "y", {1, 3}));
  d.graph.outputs = {"y"};

  auto g = pl::adapt_graph(d);
  check(g.has_value(), "graph adapts");
  auto plan = planner.plan(*g);
  check(plan.has_value(), "plan succeeds");
  if (!plan)
    return 1;

  auto r = mem.plan_memory(*plan);
  check(r.has_value(), "memory planning succeeds");
  if (!r) {
    std::cerr << "unexpected: " << r.error().message << "\n";
    return 1;
  }

  check(plan->lifetimes.size() == plan->tensors.size(), "one lifetime per tensor");
  const auto* lt_x = &plan->lifetimes[0];  // x: graph input
  const auto* lt_y = &plan->lifetimes[1];  // y: graph output
  check(!lt_x->producer.has_value(), "input has no producer");
  check(lt_x->consumers.size() == 1 &&
            lt_x->consumers[0] == pl::TaskId{0},
        "input consumed by compute task");
  check(lt_y->producer == pl::TaskId{0}, "output produced by compute task");
  check(lt_y->consumers.empty(), "output has no consumers");

  check(plan->allocations.size() == plan->tensors.size(),
        "one buffer per tensor (no reuse)");
  // Default CPU alignment = 64; 12-byte tensors round up to 64 each.
  check(plan->allocations[0].size_bytes == 64 &&
            plan->allocations[1].size_bytes == 64,
        "tensors aligned to 64");
  check(plan->allocations[0].assigned_tensors.size() == 1 &&
            plan->allocations[0].assigned_tensors[0] == pl::TensorId{0},
        "buffer 0 hosts tensor 0");

  auto peak_it = plan->memory_summary.peak_bytes.find(pl::MemorySpaceId{0});
  check(peak_it != plan->memory_summary.peak_bytes.end() && peak_it->second == 128,
        "peak = 128 (two 64-byte buffers)");
  check(plan->memory_summary.effective_budget_bytes.count(pl::MemorySpaceId{0}),
        "effective budget recorded");
  check(plan->estimated_cost.estimated_memory_us > 0.0, "memory cost > 0");

  // --- intermediate tensor lifetime (2-node chain) ------------------------
  auto d2 = doc("g2");
  d2.graph.inputs.push_back({"x", ft({1, 3})});
  d2.graph.nodes.push_back(relu("x", "h", {1, 3}));
  d2.graph.nodes.push_back(relu("h", "y", {1, 3}));
  d2.graph.outputs = {"y"};
  auto g2 = pl::adapt_graph(d2);
  auto plan2 = planner.plan(*g2);
  check(g2.has_value() && plan2.has_value(), "2-node chain plans");
  if (!plan2)
    return 1;
  auto r2 = mem.plan_memory(*plan2);
  check(r2.has_value(), "2-node memory planning succeeds");
  // tensors: x(0, input), h(1, intermediate), y(2, output)
  const auto& lt_h = plan2->lifetimes[1];
  check(lt_h.producer == pl::TaskId{0} && lt_h.consumers.size() == 1 &&
            lt_h.consumers[0] == pl::TaskId{0},
        "intermediate produced and consumed by compute task");

  // --- insufficient memory --------------------------------------------------
  {
    pl::ResourceSnapshot s;
    s.devices.push_back(make_cpu());
    s.memory_spaces.push_back(make_host(/*capacity=*/100));
    pl::AnalyticalCpuCostModel cm(s);
    pl::StaticPlanner p(s, cm);
    pl::MemoryPlanner mp(cm);
    auto pg = p.plan(*g);
    check(pg.has_value(), "plan succeeds before memory check");
    auto mr = mp.plan_memory(*pg);
    check(!mr.has_value() &&
              mr.error().code == pl::PlannerErrorCode::InsufficientMemory,
          "tiny budget rejected with InsufficientMemory");
  }

  if (g_failures == 0) {
    std::cout << "test_memory_planner OK\n";
    return 0;
  }
  std::cerr << "test_memory_planner FAILED: " << g_failures << " check(s)\n";
  return 1;
}
