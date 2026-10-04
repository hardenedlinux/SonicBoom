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

// Static planner benchmark (P7): plans a small graph repeatedly and reports the
// measured per-plan wall-clock time. It asserts determinism (every plan carries
// the same id) and success, but makes no wall-clock-timing assertion — the
// numbers are informational, since absolute timing is environment-dependent.

#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/telemetry.h>

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

sx::Node relu(const std::string& in, const std::string& out,
              std::vector<int64_t> dims) {
  sx::Node n;
  n.op_name = "relu";
  n.inputs = {in};
  n.outputs = {{out, ft(std::move(dims))}};
  return n;
}
} // namespace

int main() {
  constexpr int kIters = 1000;

  // A small 3-op chain.
  sx::Document d;
  d.version_major = 0;
  d.version_minor = 1;
  d.graph.name = "bench";
  d.graph.opsets.push_back({"default", 20});
  d.graph.inputs.push_back({"x", ft({1, 64})});
  d.graph.nodes.push_back(relu("x", "a", {1, 64}));
  d.graph.nodes.push_back(relu("a", "b", {1, 64}));
  d.graph.nodes.push_back(relu("b", "y", {1, 64}));
  d.graph.outputs = {"y"};

  auto g = pl::adapt_graph(d);
  check(g.has_value(), "graph adapts");
  if (!g)
    return 1;

  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "snapshot builds");
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel cost(*snap);

  uint64_t first_plan_id = 0;
  pl::Timer t;
  for (int i = 0; i < kIters; ++i) {
    auto plan = pl::plan_execution(*g, *snap, cost);
    if (!plan) {
      std::cerr << "  FAIL: plan " << i << " failed: " << plan.error().message
                << "\n";
      ++g_failures;
      return 1;
    }
    if (i == 0)
      first_plan_id = plan->plan_id;
    if (plan->plan_id != first_plan_id) {
      std::cerr << "  FAIL: plan id not deterministic\n";
      ++g_failures;
      return 1;
    }
  }
  double total_us = t.lap_us();

  check(g_failures == 0, "all plans succeeded deterministically");
  std::cout << "planned " << kIters << " plans in " << total_us << " us ("
            << (total_us / kIters) << " us/plan)\n";

  if (g_failures == 0) {
    std::cout << "test_planner_benchmark OK\n";
    return 0;
  }
  std::cerr << "test_planner_benchmark FAILED\n";
  return 1;
}
