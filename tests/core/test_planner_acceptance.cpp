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

// Static planner acceptance (P7): the planner correctly and deterministically
// plans the full frozen ResNet-18 S-Expr v0.1 fixture end-to-end — adapt →
// plan → memory → transfer → validate — producing a single whole-graph compute
// task, a coherent memory summary within budget, and a valid plan. This proves
// the planner scales to a real ~50-op model without invoking MLIR/JIT.

#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/plan_validator.h>

#include <sonicboom/sx/parser.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef SONICBOOM_SRC_DIR
#error "SONICBOOM_SRC_DIR must be defined to the repository root"
#endif

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
  const std::string fixture =
      std::string(SONICBOOM_SRC_DIR) + "/design/s-expr-v0-resnet18.example.sx";

  std::ifstream in(fixture);
  if (!in) {
    std::cerr << "cannot open " << fixture << "\n";
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  auto doc = sx::parse_document(ss.str());
  check(doc.has_value(), "resnet18 parses");
  if (!doc)
    return 1;

  auto g = pl::adapt_graph(*doc);
  check(g.has_value(), "resnet18 adapts to planner graph");
  if (!g) {
    std::cerr << "adapt failed: " << g.error().message << "\n";
    return 1;
  }
  check(g->nodes.size() > 10, "resnet18 has a real multi-op graph");

  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "CPU snapshot builds");
  if (!snap)
    return 1;

  pl::AnalyticalCpuCostModel cost(*snap);
  auto plan = pl::plan_execution(*g, *snap, cost);
  check(plan.has_value(), "resnet18 plans successfully");
  if (!plan) {
    std::cerr << "plan failed: " << plan.error().message << "\n";
    return 1;
  }

  check(plan->tasks.size() == 1, "single whole-graph compute task");
  const auto* comp = plan->tasks[0].as_compute();
  check(comp && comp->op == pl::OpKind::WholeGraph, "compute op = WholeGraph");
  check(comp && comp->graph_nodes.size() == g->nodes.size(),
        "compute task covers every node");

  check(plan->model_fingerprint != 0, "model fingerprint non-zero");
  check(plan->resource_fingerprint != 0, "resource fingerprint non-zero");

  auto peak_it = plan->memory_summary.peak_bytes.find(pl::MemorySpaceId{0});
  check(peak_it != plan->memory_summary.peak_bytes.end() && peak_it->second > 0,
        "peak working set recorded and positive");
  auto budget_it =
      plan->memory_summary.effective_budget_bytes.find(pl::MemorySpaceId{0});
  check(budget_it != plan->memory_summary.effective_budget_bytes.end() &&
            peak_it != plan->memory_summary.peak_bytes.end() &&
            peak_it->second <= budget_it->second,
        "peak fits within the effective budget");

  check(plan->estimated_cost.estimated_total_us > 0.0, "estimated total cost > 0");
  check(pl::PlanValidator::validate(*plan).has_value(), "plan re-validates");

  // Determinism: planning the same graph again yields the same plan id.
  auto plan2 = pl::plan_execution(*g, *snap, cost);
  check(plan2.has_value() && plan2->plan_id == plan->plan_id,
        "planning is deterministic");

  if (g_failures == 0) {
    std::cout << "test_planner_acceptance OK (ResNet-18 plans: " << g->nodes.size()
              << " nodes, peak " << peak_it->second << " bytes)\n";
    return 0;
  }
  std::cerr << "test_planner_acceptance FAILED: " << g_failures << " check(s)\n";
  return 1;
}
