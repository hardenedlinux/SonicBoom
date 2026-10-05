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

// CPU co-execution integration test (Stage 3): a two-node graph `softmax → relu`
// is planned into two compute tasks, softmax routed to NativeTorchBackend and
// relu routed to the MLIR CpuBackend on a sliced sub-document, then executed
// end-to-end with an intermediate tensor handed off by TensorId. Verifies the
// numeric result, the intermediate tensor's shape/dtype in the plan, and two
// error paths (dispatch exception → RuntimeError; byte-count mismatch → error).

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/native_torch_backend.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/runtime_executor.h>

#include <sonicboom/sx/parser.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
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

const pl::TensorDesc* find_tensor(const pl::ExecutionPlan& plan,
                                  const std::string& name) {
  for (const auto& t : plan.tensors)
    if (t.name == name)
      return &t;
  return nullptr;
}

// z = relu(softmax(x, axis=1)); x,y,z : float32 [2,3].
const char* SOFTMAX_RELU =
    "(sonicboom-s-expr (version 0 1) (graph (name \"softmax_relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"y\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// Softmax with an out-of-range axis: the dispatch throws a c10 exception, which
// NativeTorchBackend must map to a RuntimeError (never a crash/abort).
const char* BAD_AXIS =
    "(sonicboom-s-expr (version 0 1) (graph (name \"bad_axis\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"y\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 5)))))))";
} // namespace

int main() {
  // --- plan the two-node graph into two routed compute tasks ----------------
  auto doc = sx::parse_document(SOFTMAX_RELU);
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

  check(plan->tasks.size() == 2, "plan has two compute tasks");
  check(plan->execution_order.size() == 2, "two-task execution order");

  // softmax → NativeTorch, relu → Mlir, in dependency order.
  {
    const pl::TaskDesc* s0 = plan->find_task(plan->execution_order[0]);
    const pl::TaskDesc* s1 = plan->find_task(plan->execution_order[1]);
    check(s0 && s0->as_compute() &&
              s0->as_compute()->backend == pl::BackendTag::NativeTorch &&
              s0->as_compute()->op == pl::OpKind::Softmax,
          "task 0 is native-torch softmax");
    check(s1 && s1->as_compute() &&
              s1->as_compute()->backend == pl::BackendTag::Mlir &&
              s1->as_compute()->op == pl::OpKind::Relu,
          "task 1 is mlir relu");
  }

  // Intermediate tensor "y" must carry its shape/dtype in the plan (the bridge
  // reads them; it must not infer them from byte length).
  {
    const pl::TensorDesc* y = find_tensor(*plan, "y");
    check(y != nullptr, "plan records intermediate tensor 'y'");
    if (y) {
      check(y->shape == (std::vector<int64_t>{2, 3}), "intermediate shape [2,3]");
      check(y->dtype == sx::DType::Float32, "intermediate dtype float32");
      check(y->size_bytes == 6 * sizeof(float), "intermediate byte size");
    }
  }

  // --- slice the MLIR region (relu) and compile each backend ----------------
  auto relu_doc = pl::slice_document(*doc, {"z"});
  check(relu_doc.has_value(), "relu region slices");
  if (!relu_doc) {
    std::cerr << "unexpected: " << relu_doc.error().message << "\n";
    return 1;
  }

  auto cpu = pl::CpuBackend::compile(*relu_doc, {});
  check(cpu.has_value(), "mlir backend compiles relu region");
  if (!cpu)
    return 1;

  auto nt = pl::NativeTorchBackend::compile(*g);
  check(nt.has_value(), "native-torch backend compiles");
  if (!nt)
    return 1;

  pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
      {pl::BackendTag::Mlir, cpu->get()},
      {pl::BackendTag::NativeTorch, nt->get()}});

  // --- execute: softmax(axis=1) then relu --------------------------------
  // x rows are identical ([1,2,3] twice), so an axis=1 (per-row) softmax gives
  // the classic 3-element result; an axis=0 (per-column) softmax would give all
  // 0.5s — the check below therefore also proves the axis→dim mapping.
  const std::vector<float> x = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  auto result = exec.execute(*plan, *snap, {pack(x)});
  check(result.has_value(), "execute succeeds");
  if (!result) {
    std::cerr << "unexpected: " << result.error().message << "\n";
    return 1;
  }

  check(result->outputs.size() == 1, "one output buffer");
  std::vector<float> got = unpack(result->outputs[0]);
  check(got.size() == 6, "output has 6 elements");
  const float p0 = 0.090030573f, p1 = 0.244728471f, p2 = 0.665240956f;
  const std::vector<float> expected = {p0, p1, p2, p0, p1, p2};  // relu identity
  for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i)
    check(std::fabs(got[i] - expected[i]) <= 1e-5f, "softmax→relu value correct");

  // --- error path 1: input byte-count mismatch → BackendFailure (bridge) ----
  {
    sx::Bytes bad(10);  // 10 bytes ≠ 24 bytes for float32[2,3]
    auto r = exec.execute(*plan, *snap, {bad});
    check(!r.has_value() &&
              r.error().code == pl::RuntimeErrorCode::BackendFailure,
          "byte-count mismatch rejected by the bridge");
  }

  // --- error path 2: out-of-range axis → dispatch throws → BackendFailure ---
  {
    auto bad_doc = sx::parse_document(BAD_AXIS);
    check(bad_doc.has_value(), "bad-axis doc parses");
    if (!bad_doc)
      return 1;

    auto bad_g = pl::adapt_graph(*bad_doc);
    check(bad_g.has_value(), "bad-axis graph adapts");
    if (!bad_g)
      return 1;

    auto bad_plan = pl::plan_execution(*bad_g, *snap, cost);
    check(bad_plan.has_value(), "bad-axis plan_execution succeeds");
    if (!bad_plan)
      return 1;

    auto bad_nt = pl::NativeTorchBackend::compile(*bad_g);
    check(bad_nt.has_value(), "bad-axis native backend compiles");
    if (!bad_nt)
      return 1;

    pl::RuntimeExecutor bad_exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::NativeTorch, bad_nt->get()}});
    auto r = bad_exec.execute(*bad_plan, *snap, {pack(x)});
    check(!r.has_value() &&
              r.error().code == pl::RuntimeErrorCode::BackendFailure,
          "out-of-range axis maps to BackendFailure");
  }

  if (g_failures == 0) {
    std::cout << "test_coexecution OK\n";
    return 0;
  }
  std::cerr << "test_coexecution FAILED: " << g_failures << " check(s)\n";
  return 1;
}
