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

// CPU co-execution branch/merge test (Stage 4). The four-node graph
//
//                    ┌─ softmax(x) → s ─ relu → a ─┐
//   x ───────────────┤                             ├─ add(a, b) → z
//                    └─ relu(x) → b ───────────────┘
//
// is planned into four compute tasks: softmax on NativeTorch, and three MLIR
// nodes (relu(s), relu(x), add) each compiled from its own slice_document
// region. The executor is wired per-task so the three MLIR slices dispatch to
// three distinct CpuBackends. It verifies:
//   - four tasks in topological order, with the add task depending on both relu
//     tasks and consuming two *distinct* input tensors (a and b, neither the
//     graph input nor a previous task's output);
//   - intermediate tensors a and b carry their own shape/dtype and are not
//     overwritten across branches;
//   - the numeric result z = relu(softmax(x)) + relu(x) matches an independent
//     hand-computed reference;
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
  // --- plan the four-node graph into four routed compute tasks -------------
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

  check(plan->tasks.size() == 4, "plan has four compute tasks");
  check(plan->execution_order.size() == 4, "four-task execution order");

  const pl::TaskDesc* t_softmax = task_producing(*plan, "s");
  const pl::TaskDesc* t_relu_s = task_producing(*plan, "a");
  const pl::TaskDesc* t_relu_x = task_producing(*plan, "b");
  const pl::TaskDesc* t_add = task_producing(*plan, "z");
  check(t_softmax && t_relu_s && t_relu_x && t_add,
        "all four tasks located by their output tensor");

  if (!(t_softmax && t_relu_s && t_relu_x && t_add))
    return 1;

  // Routing: softmax → NativeTorch, all three MLIR nodes → Mlir.
  check(t_softmax->as_compute() &&
            t_softmax->as_compute()->backend == pl::BackendTag::NativeTorch &&
            t_softmax->as_compute()->op == pl::OpKind::Softmax,
        "softmax task is native-torch");
  check(t_relu_s->as_compute() &&
            t_relu_s->as_compute()->backend == pl::BackendTag::Mlir &&
            t_relu_s->as_compute()->op == pl::OpKind::Relu,
        "relu(s) task is mlir relu");
  check(t_relu_x->as_compute() &&
            t_relu_x->as_compute()->backend == pl::BackendTag::Mlir &&
            t_relu_x->as_compute()->op == pl::OpKind::Relu,
        "relu(x) task is mlir relu");
  check(t_add->as_compute() &&
            t_add->as_compute()->backend == pl::BackendTag::Mlir &&
            t_add->as_compute()->op == pl::OpKind::Add,
        "add task is mlir add");

  // The add task must consume exactly two *distinct* input tensors (a and b),
  // neither the graph input "x" nor the softmax intermediate "s".
  check(t_add->inputs.size() == 2, "add task has two inputs");
  if (t_add->inputs.size() == 2) {
    const pl::TensorDesc* a = plan->find_tensor(t_add->inputs[0]);
    const pl::TensorDesc* b = plan->find_tensor(t_add->inputs[1]);
    check(a && b, "add task inputs resolve to tensors");
    if (a && b) {
      check(a->name == "a", "add task first input is 'a'");
      check(b->name == "b", "add task second input is 'b'");
      check(a->id != b->id, "a and b are distinct tensors");
      check(a->shape == (std::vector<int64_t>{2, 3}) &&
                b->shape == (std::vector<int64_t>{2, 3}),
            "a and b both carry shape [2,3]");
      check(a->dtype == sx::DType::Float32 && b->dtype == sx::DType::Float32,
            "a and b are float32");
    }
  }

  // add depends on both relu tasks (producer/consumer relationships).
  {
    bool dep_on_relu_s = false, dep_on_relu_x = false;
    for (pl::TaskId d : t_add->dependencies) {
      if (d == t_relu_s->id)
        dep_on_relu_s = true;
      if (d == t_relu_x->id)
        dep_on_relu_x = true;
    }
    check(dep_on_relu_s && dep_on_relu_x, "add task depends on both relu tasks");
  }

  // Intermediate tensors are recorded with their own shape/dtype.
  for (const char* name : {"s", "a", "b"}) {
    const pl::TensorDesc* t = find_tensor(*plan, name);
    check(t != nullptr, "plan records intermediate tensor");
    if (t) {
      check(t->shape == (std::vector<int64_t>{2, 3}), "intermediate shape [2,3]");
      check(t->dtype == sx::DType::Float32, "intermediate dtype float32");
    }
  }

  // --- slice each MLIR region and compile a backend per slice ---------------
  auto doc_a = pl::slice_document(*doc, {"a"});
  auto doc_b = pl::slice_document(*doc, {"b"});
  auto doc_z = pl::slice_document(*doc, {"z"});
  check(doc_a.has_value() && doc_b.has_value() && doc_z.has_value(),
        "three MLIR regions slice");
  if (!(doc_a && doc_b && doc_z))
    return 1;

  auto cpu_a = pl::CpuBackend::compile(*doc_a, {});
  auto cpu_b = pl::CpuBackend::compile(*doc_b, {});
  auto cpu_z = pl::CpuBackend::compile(*doc_z, {});
  check(cpu_a.has_value() && cpu_b.has_value() && cpu_z.has_value(),
        "three mlir backends compile");
  if (!(cpu_a && cpu_b && cpu_z))
    return 1;

  auto nt = pl::NativeTorchBackend::compile(*g);
  check(nt.has_value(), "native-torch backend compiles");
  if (!nt)
    return 1;

  // --- wire the executor per-task (the Mlir tag alone is ambiguous) ---------
  pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{});
  exec.set_task_backend(t_softmax->id, nt->get());
  exec.set_task_backend(t_relu_s->id, cpu_a->get());
  exec.set_task_backend(t_relu_x->id, cpu_b->get());
  exec.set_task_backend(t_add->id, cpu_z->get());

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
    pl::RuntimeExecutor missing(std::map<pl::BackendTag, pl::Backend*>{});
    missing.set_task_backend(t_softmax->id, nt->get());
    missing.set_task_backend(t_relu_s->id, cpu_a->get());
    missing.set_task_backend(t_relu_x->id, cpu_b->get());
    // t_add intentionally has no backend registered.
    auto r = missing.execute(*plan, *snap, {pack(x)});
    check(!r.has_value() &&
              r.error().code == pl::RuntimeErrorCode::BackendFailure,
          "missing add backend maps to BackendFailure");
  }

  if (g_failures == 0) {
    std::cout << "test_coexecution_merge OK\n";
    return 0;
  }
  std::cerr << "test_coexecution_merge FAILED: " << g_failures << " check(s)\n";
  return 1;
}
