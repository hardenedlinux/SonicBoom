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

// CPU co-execution branch/merge test (Stage 5 region partitioning). The
// four-node graph
//
//                    ┌─ softmax(x) → s ─ relu → a ─┐
//   x ───────────────┤                             ├─ add(a, b) → z
//                    └─ relu(x) → b ───────────────┘
//
// is planned into TWO regions: softmax on NativeTorch, and the three
// consecutive MLIR nodes (relu(s), relu(x), add) merged into one MLIR region
// compiled from a single slice. It verifies:
//   - two tasks in topological order, the MLIR region depending on the softmax
//     task and consuming two *distinct* external inputs (s and x);
//   - the region's internal tensors a and b stay inside the compiled unit (no
//     task-level I/O), yet the plan still records their shape/dtype;
//   - the numeric result z = relu(softmax(x)) + relu(x) matches an independent
//     hand-computed reference (the merge happens inside the compiled region);
//   - a task with no registered backend fails with a RuntimeError (no crash).

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

// The compute task whose outputs include the tensor named `name`.
const pl::TaskDesc* task_producing(const pl::ExecutionPlan& plan,
                                   const std::string& name) {
  for (const auto& t : plan.tasks)
    for (pl::TensorId o : t.outputs)
      if (const auto* td = plan.find_tensor(o))
        if (td->name == name)
          return &t;
  return nullptr;
}

// Build a slice Document for a region task: its graph_nodes select the kept
// nodes (one output name per node), its outputs become the graph output list.
std::expected<sx::Document, pl::PlannerError> region_slice(
    const sx::Document& doc, const pl::Graph& g, const pl::TaskDesc& task) {
  const auto* compute = task.as_compute();
  std::vector<std::string> kept;
  for (pl::GraphNodeId nid : compute->graph_nodes) {
    const pl::GraphNodeDesc* nd = g.find_node(nid);
    if (!nd)
      return std::unexpected(pl::PlannerError(
          pl::PlannerErrorCode::InvalidGraph, "region node not found",
          pl::Phase::GraphAnalysis));
    for (pl::TensorId o : nd->outputs)
      kept.push_back(g.find_tensor(o)->name);
  }
  std::vector<std::string> outs;
  for (pl::TensorId o : task.outputs)
    outs.push_back(g.find_tensor(o)->name);
  return pl::slice_region(doc, kept, outs);
}

// Branch/merge graph: z = add(relu(softmax(x, axis=1)), relu(x)).
const char* MERGE =
    "(sonicboom-s-expr (version 0 1) (graph (name \"merge\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"a\" (tensor float32 (shape 2 3)))))"
    "   (node relu (inputs \"x\")"
    "     (outputs (\"b\" (tensor float32 (shape 2 3)))))"
    "   (node add (inputs \"a\" \"b\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";
} // namespace

int main() {
  // --- plan the four-node graph into two routed regions --------------------
  auto doc = sx::parse_document(MERGE);
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

  check(plan->tasks.size() == 2, "plan has two region tasks");
  check(plan->execution_order.size() == 2, "two-task execution order");

  const pl::TaskDesc* t_softmax = task_producing(*plan, "s");
  const pl::TaskDesc* t_region = task_producing(*plan, "z");
  check(t_softmax && t_region, "both regions located by an output tensor");
  if (!(t_softmax && t_region))
    return 1;

  // softmax → NativeTorch, single node.
  check(t_softmax->as_compute() &&
            t_softmax->as_compute()->backend == pl::BackendTag::NativeTorch &&
            t_softmax->as_compute()->op == pl::OpKind::Softmax &&
            t_softmax->as_compute()->graph_nodes.size() == 1,
        "softmax region is native-torch, single node");

  // The three consecutive MLIR nodes merge into one region.
  check(t_region->as_compute() &&
            t_region->as_compute()->backend == pl::BackendTag::Mlir &&
            t_region->as_compute()->graph_nodes.size() == 3,
        "mlir region has three merged nodes");
  check(t_region->as_compute()->op == pl::OpKind::Relu,
        "mlir region representative op is the first node's op (relu)");

  // The region consumes two *distinct* external inputs (s and x) and emits one
  // output (z).
  check(t_region->inputs.size() == 2, "mlir region has two external inputs");
  if (t_region->inputs.size() == 2) {
    const pl::TensorDesc* in0 = plan->find_tensor(t_region->inputs[0]);
    const pl::TensorDesc* in1 = plan->find_tensor(t_region->inputs[1]);
    check(in0 && in1, "region inputs resolve to tensors");
    if (in0 && in1) {
      check(in0->name == "s" && in1->name == "x", "region inputs are s and x");
      check(in0->id != in1->id, "s and x are distinct tensors");
    }
  }
  check(t_region->outputs.size() == 1, "mlir region has one output");
  if (!t_region->outputs.empty()) {
    const pl::TensorDesc* z = plan->find_tensor(t_region->outputs.front());
    check(z && z->name == "z", "region output is z");
  }

  // Region depends on the softmax task (producer of s).
  {
    bool dep_on_softmax = false;
    for (pl::TaskId d : t_region->dependencies)
      if (d == t_softmax->id)
        dep_on_softmax = true;
    check(dep_on_softmax, "mlir region depends on the softmax task");
  }

  // Internal tensors a and b are recorded with shape/dtype, but no task lists
  // them as I/O (they live inside the compiled region).
  for (const char* name : {"a", "b"}) {
    const pl::TensorDesc* t = find_tensor(*plan, name);
    check(t != nullptr, "plan records internal tensor");
    if (t) {
      check(t->shape == (std::vector<int64_t>{2, 3}), "internal shape [2,3]");
      check(t->dtype == sx::DType::Float32, "internal dtype float32");
    }
    check(task_producing(*plan, name) == nullptr,
          "internal tensor is not a task output");
  }

  // --- compile the MLIR region (one slice) and the native backend ----------
  auto region_doc = region_slice(*doc, *g, *t_region);
  check(region_doc.has_value(), "mlir region slices");
  if (!region_doc)
    return 1;
  auto cpu = pl::CpuBackend::compile(*region_doc, {});
  check(cpu.has_value(), "mlir region compiles");
  if (!cpu)
    return 1;

  auto nt = pl::NativeTorchBackend::compile(*g);
  check(nt.has_value(), "native-torch backend compiles");
  if (!nt)
    return 1;

  // Single MLIR region → the Mlir tag is unambiguous; map dispatch suffices.
  pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
      {pl::BackendTag::Mlir, cpu->get()},
      {pl::BackendTag::NativeTorch, nt->get()}});

  // --- execute and check the numeric result --------------------------------
  // x rows are identical ([1,2,3] twice); axis=1 softmax of a row gives
  // [p0,p1,p2], relu is identity on the positive values, so
  //   z = softmax_row + x_row = [p0+1, p1+2, p2+3] per row.
  const std::vector<float> x = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};
  auto result = exec.execute(*plan, *snap, {pack(x)});
  check(result.has_value(), "execute succeeds");
  if (!result) {
    std::cerr << "unexpected: " << result.error().message << "\n";
    return 1;
  }

  check(result->outputs.size() == 1, "one output buffer");
  const float p0 = 0.090030573f, p1 = 0.244728471f, p2 = 0.665240956f;
  const std::vector<float> expected = {p0 + 1.0f, p1 + 2.0f, p2 + 3.0f,
                                       p0 + 1.0f, p1 + 2.0f, p2 + 3.0f};
  std::vector<float> got = unpack(result->outputs[0]);
  check(got.size() == 6, "output has 6 elements");
  for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i)
    check(std::fabs(got[i] - expected[i]) <= 1e-5f, "merge value correct");

  // --- error path: a task with no backend → RuntimeError (no crash) ---------
  {
    pl::RuntimeExecutor missing(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::NativeTorch, nt->get()}});
    // the Mlir region intentionally has no registered backend.
    auto r = missing.execute(*plan, *snap, {pack(x)});
    check(!r.has_value() &&
              r.error().code == pl::RuntimeErrorCode::BackendFailure,
          "missing mlir backend maps to BackendFailure");
  }

  if (g_failures == 0) {
    std::cout << "test_coexecution_merge OK\n";
    return 0;
  }
  std::cerr << "test_coexecution_merge FAILED: " << g_failures << " check(s)\n";
  return 1;
}
