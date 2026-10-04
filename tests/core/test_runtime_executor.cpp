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

// Runtime executor tests (P6): full parse → plan → compile → execute pipeline,
// verifying the executor runs the whole-graph CPU entry and fails hard on a
// resource mismatch or a corrupt plan (no silent replanning).

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/runtime_executor.h>

#include <sonicboom/sx/parser.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
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

sx::Bytes pack(const std::vector<float>& v) {
  sx::Bytes b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<float> unpack(const sx::Bytes& b) {
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

// z = relu(x + b), single f32 input/output, shape [8].
const char* ADD_RELU =
    "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 8))))"
    " (outputs (output \"z\"))"
    " (parameters (parameter \"b\" (tensor float32 (shape 8))"
    "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
    " (nodes"
    "   (node add (inputs \"x\" \"b\")"
    "     (outputs (\"y\" (tensor float32 (shape 8)))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 8))))))))";
} // namespace

int main() {
  // --- full pipeline: parse → plan → compile → execute ---------------------
  auto doc = sx::parse_document(ADD_RELU);
  check(doc.has_value(), "doc parses");
  if (!doc)
    return 1;

  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "snapshot builds");
  if (!snap)
    return 1;

  auto g = pl::adapt_graph(*doc);
  check(g.has_value(), "graph adapts");
  if (!g)
    return 1;

  pl::AnalyticalCpuCostModel cost(*snap);
  auto plan = pl::plan_execution(*g, *snap, cost);
  check(plan.has_value(), "plan_execution succeeds");
  if (!plan) {
    std::cerr << "unexpected: " << plan.error().message << "\n";
    return 1;
  }

  auto backend = pl::CpuBackend::compile(*doc, {});
  check(backend.has_value(), "backend compiles");
  if (!backend)
    return 1;

  pl::RuntimeExecutor exec(**backend);

  const std::vector<float> x = {1.0f, 2.0f, -3.0f, 4.0f,
                                -5.0f, 6.0f, -7.0f, 8.0f};
  auto result = exec.execute(*plan, *snap, {pack(x)});
  check(result.has_value(), "execute succeeds");
  if (!result) {
    std::cerr << "unexpected: " << result.error().message << "\n";
    return 1;
  }

  // x + b = [1.5, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
  // relu  = [1.5, 1.0,  0.0, 1.0,  0.0, 1.0,  0.0, 1.0]
  const std::vector<float> expected = {1.5f, 1.0f, 0.0f, 1.0f,
                                       0.0f, 1.0f, 0.0f, 1.0f};
  check(result->outputs.size() == 1, "one output buffer");
  std::vector<float> got = unpack(result->outputs[0]);
  check(got.size() == 8, "output has 8 elements");
  for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i)
    check(std::fabs(got[i] - expected[i]) <= 1e-5f, "output element correct");

  // --- resource mismatch → ResourceChanged (no silent replanning) ----------
  {
    pl::CpuProviderConfig cfg;
    cfg.host_memory_budget_bytes = 1ull << 30;  // different fingerprint
    auto other = pl::CpuResourceProvider::snapshot(cfg);
    check(other.has_value(), "alternate snapshot builds");
    auto r = exec.execute(*plan, *other, {pack(x)});
    check(!r.has_value() &&
              r.error().code == pl::RuntimeErrorCode::ResourceChanged,
          "resource mismatch rejected");
  }

  // --- corrupt plan → InvalidPlan ------------------------------------------
  {
    auto bad = *plan;              // copy
    bad.execution_order.clear();   // empty order ≠ one task
    auto r = exec.execute(bad, *snap, {pack(x)});
    check(!r.has_value() && r.error().code == pl::RuntimeErrorCode::InvalidPlan,
          "corrupt plan rejected");
  }

  if (g_failures == 0) {
    std::cout << "test_runtime_executor OK\n";
    return 0;
  }
  std::cerr << "test_runtime_executor FAILED: " << g_failures << " check(s)\n";
  return 1;
}
