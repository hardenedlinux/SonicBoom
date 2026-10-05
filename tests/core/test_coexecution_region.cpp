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

// CPU co-execution region-partition tests (Stage 5). Each scenario builds a
// small S-Expr graph, plans it through the region-partitioned mixed path, and
// verifies the region/task structure AND the numeric result:
//   A. consecutive MLIR nodes merge into one region (softmax → relu → relu → relu);
//   B. NativeTorch → MLIR → NativeTorch (three regions in dependency order);
//   C. multi-input MLIR region fed by two native-torch producers;
//   D. graph output that is also an intermediate tensor (two distinguishable
//      outputs);
//   E. two *separate* MLIR regions wired with set_task_backend.
// Error paths are folded into A (slice_region rejects an unproduced region
// output, and a byte-count mismatch) and B (a missing MLIR backend).

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

const pl::TaskDesc* task_producing(const pl::ExecutionPlan& plan,
                                   const std::string& name) {
  for (const auto& t : plan.tasks)
    for (pl::TensorId o : t.outputs)
      if (const auto* td = plan.find_tensor(o))
        if (td->name == name)
          return &t;
  return nullptr;
}

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

// softmax([1,2,3]) and softmax(softmax([1,2,3])) reference values.
const float P0 = 0.090030573f, P1 = 0.244728471f, P2 = 0.665240956f;
const float Q0 = 0.253497652f, Q1 = 0.295909144f, Q2 = 0.450593204f;
const std::vector<float> X = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};

// A. softmax → relu → relu → relu : the three relus merge into one region.
const char* CONSECUTIVE =
    "(sonicboom-s-expr (version 0 1) (graph (name \"consecutive\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"a\" (tensor float32 (shape 2 3)))))"
    "   (node relu (inputs \"a\")"
    "     (outputs (\"b\" (tensor float32 (shape 2 3)))))"
    "   (node relu (inputs \"b\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// B. softmax → relu → softmax : three regions (NT, MLIR, NT).
const char* NT_MLIR_NT =
    "(sonicboom-s-expr (version 0 1) (graph (name \"nt_mlir_nt\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"a\" (tensor float32 (shape 2 3)))))"
    "   (node softmax (inputs \"a\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1)))))))";

// C. softmax(x)→s, softmax(y)→t, add(s,t)→z : MLIR add fed by two native tasks.
const char* MULTI_INPUT =
    "(sonicboom-s-expr (version 0 1) (graph (name \"multi_input\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3)))"
    "          (input \"y\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node softmax (inputs \"y\")"
    "     (outputs (\"t\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node add (inputs \"s\" \"t\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// D. softmax(x)→s, add(s,x)→z, outputs {s, z} : an intermediate that is also a
// graph output, plus a two-input region, with two distinguishable results.
const char* GRAPH_OUTPUT =
    "(sonicboom-s-expr (version 0 1) (graph (name \"graph_output\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"s\") (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node add (inputs \"s\" \"x\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// F. softmax → relu → softmax → relu : two separate MLIR regions.
const char* TWO_REGIONS =
    "(sonicboom-s-expr (version 0 1) (graph (name \"two_regions\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"a\" (tensor float32 (shape 2 3)))))"
    "   (node softmax (inputs \"a\")"
    "     (outputs (\"t\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"t\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

} // namespace

int main() {
  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "snapshot builds");
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel cost(*snap);

  // --- A. consecutive MLIR nodes merge -------------------------------------
  {
    auto doc = sx::parse_document(CONSECUTIVE);
    auto g = pl::adapt_graph(*doc);
    auto plan = pl::plan_execution(*g, *snap, cost);
    check(doc && g && plan, "A: plan succeeds");
    if (!(doc && g && plan))
      return 1;

    check(plan->tasks.size() == 2, "A: two regions (softmax + merged relus)");

    const pl::TaskDesc* t_s = task_producing(*plan, "s");
    const pl::TaskDesc* t_z = task_producing(*plan, "z");
    check(t_s && t_z, "A: both regions located");
    if (!(t_s && t_z))
      return 1;

    check(t_s->as_compute()->backend == pl::BackendTag::NativeTorch &&
              t_s->as_compute()->graph_nodes.size() == 1,
          "A: softmax is a single-node native region");
    check(t_z->as_compute()->backend == pl::BackendTag::Mlir &&
              t_z->as_compute()->graph_nodes.size() == 3 &&
              t_z->as_compute()->op == pl::OpKind::Relu,
          "A: three consecutive relus merged into one mlir region");
    check(t_z->inputs.size() == 1 && t_z->outputs.size() == 1,
          "A: mlir region is 1-in/1-out");
    for (const char* internal : {"a", "b"})
      check(task_producing(*plan, internal) == nullptr &&
                find_tensor(*plan, internal) != nullptr,
            "A: internal tensor stays inside the region");

    auto region_doc = region_slice(*doc, *g, *t_z);
    check(region_doc.has_value(), "A: region slices");
    auto cpu = pl::CpuBackend::compile(*region_doc, {});
    auto nt = pl::NativeTorchBackend::compile(*g);
    check(cpu && nt, "A: backends compile");
    if (!(cpu && nt))
      return 1;

    pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::Mlir, cpu->get()},
        {pl::BackendTag::NativeTorch, nt->get()}});

    auto r = exec.execute(*plan, *snap, {pack(X)});
    check(r.has_value(), "A: execute succeeds");
    if (r) {
      std::vector<float> got = unpack(r->outputs[0]);
      const std::vector<float> want = {P0, P1, P2, P0, P1, P2};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-5f, "A: merged value correct");
    }

    // A error path: slice_region rejects a region output no kept node produces.
    auto bad = pl::slice_region(*doc, {"a"}, {"z"});
    check(!bad.has_value(), "A: slice_region rejects unproduced region output");

    // A error path: byte-count mismatch on the region's graph input.
    sx::Bytes short_buf(10);
    auto bad_run = exec.execute(*plan, *snap, {short_buf});
    check(!bad_run.has_value() &&
              bad_run.error().code == pl::RuntimeErrorCode::BackendFailure,
          "A: byte-count mismatch rejected");
  }

  // --- B. NativeTorch → MLIR → NativeTorch ----------------------------------
  {
    auto doc = sx::parse_document(NT_MLIR_NT);
    auto g = pl::adapt_graph(*doc);
    auto plan = pl::plan_execution(*g, *snap, cost);
    check(doc && g && plan, "B: plan succeeds");
    if (!(doc && g && plan))
      return 1;

    check(plan->tasks.size() == 3, "B: three regions");
    check(plan->execution_order.size() == 3, "B: three-task order");

    const pl::TaskDesc* t_s = task_producing(*plan, "s");
    const pl::TaskDesc* t_a = task_producing(*plan, "a");
    const pl::TaskDesc* t_z = task_producing(*plan, "z");
    check(t_s && t_a && t_z, "B: all three regions located");
    if (!(t_s && t_a && t_z))
      return 1;
    check(t_s->as_compute()->backend == pl::BackendTag::NativeTorch,
          "B: region s is native");
    check(t_a->as_compute()->backend == pl::BackendTag::Mlir &&
              t_a->as_compute()->op == pl::OpKind::Relu,
          "B: region a is mlir relu");
    check(t_z->as_compute()->backend == pl::BackendTag::NativeTorch,
          "B: region z is native");

    auto cpu = pl::CpuBackend::compile(*region_slice(*doc, *g, *t_a), {});
    auto nt = pl::NativeTorchBackend::compile(*g);
    check(cpu && nt, "B: backends compile");
    if (!(cpu && nt))
      return 1;

    pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::Mlir, cpu->get()},
        {pl::BackendTag::NativeTorch, nt->get()}});

    auto r = exec.execute(*plan, *snap, {pack(X)});
    check(r.has_value(), "B: execute succeeds");
    if (r) {
      std::vector<float> got = unpack(r->outputs[0]);
      const std::vector<float> want = {Q0, Q1, Q2, Q0, Q1, Q2};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-4f,
              "B: NT→MLIR→NT value correct");
    }

    // B error path: MLIR backend missing → BackendFailure (no crash).
    pl::RuntimeExecutor missing(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::NativeTorch, nt->get()}});
    auto bad_run = missing.execute(*plan, *snap, {pack(X)});
    check(!bad_run.has_value() &&
              bad_run.error().code == pl::RuntimeErrorCode::BackendFailure,
          "B: missing mlir backend maps to BackendFailure");
  }

  // --- C. multi-input MLIR region fed by two native tasks -------------------
  {
    auto doc = sx::parse_document(MULTI_INPUT);
    auto g = pl::adapt_graph(*doc);
    auto plan = pl::plan_execution(*g, *snap, cost);
    check(doc && g && plan, "C: plan succeeds");
    if (!(doc && g && plan))
      return 1;

    check(plan->tasks.size() == 3, "C: three regions (two native + add)");

    const pl::TaskDesc* t_z = task_producing(*plan, "z");
    check(t_z && t_z->as_compute()->backend == pl::BackendTag::Mlir,
          "C: add is the mlir region");
    if (!t_z)
      return 1;
    check(t_z->inputs.size() == 2, "C: mlir region has two external inputs");
    if (t_z->inputs.size() == 2) {
      const pl::TensorDesc* i0 = plan->find_tensor(t_z->inputs[0]);
      const pl::TensorDesc* i1 = plan->find_tensor(t_z->inputs[1]);
      check(i0 && i1 && i0->name == "s" && i1->name == "t",
            "C: region inputs are s and t from the two native tasks");
      check(i0 && i1 && i0->id != i1->id, "C: s and t are distinct tensors");
    }
    check(t_z->dependencies.size() == 2, "C: region depends on both native tasks");

    auto cpu = pl::CpuBackend::compile(*region_slice(*doc, *g, *t_z), {});
    auto nt = pl::NativeTorchBackend::compile(*g);
    check(cpu && nt, "C: backends compile");
    if (!(cpu && nt))
      return 1;

    pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::Mlir, cpu->get()},
        {pl::BackendTag::NativeTorch, nt->get()}});

    auto r = exec.execute(*plan, *snap, {pack(X), pack(X)});
    check(r.has_value(), "C: execute succeeds");
    if (r) {
      std::vector<float> got = unpack(r->outputs[0]);
      const std::vector<float> want = {2 * P0, 2 * P1, 2 * P2,
                                       2 * P0, 2 * P1, 2 * P2};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-5f, "C: multi-input value correct");
    }
  }

  // --- D. graph output that is also an intermediate -------------------------
  {
    auto doc = sx::parse_document(GRAPH_OUTPUT);
    auto g = pl::adapt_graph(*doc);
    auto plan = pl::plan_execution(*g, *snap, cost);
    check(doc && g && plan, "D: plan succeeds");
    if (!(doc && g && plan))
      return 1;

    check(plan->tasks.size() == 2, "D: two regions");

    const pl::TaskDesc* t_z = task_producing(*plan, "z");
    check(t_z && t_z->as_compute()->backend == pl::BackendTag::Mlir,
          "D: add is the mlir region");
    if (!t_z)
      return 1;

    auto cpu = pl::CpuBackend::compile(*region_slice(*doc, *g, *t_z), {});
    auto nt = pl::NativeTorchBackend::compile(*g);
    check(cpu && nt, "D: backends compile");
    if (!(cpu && nt))
      return 1;

    pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::Mlir, cpu->get()},
        {pl::BackendTag::NativeTorch, nt->get()}});

    auto r = exec.execute(*plan, *snap, {pack(X)});
    check(r.has_value(), "D: execute succeeds");
    if (r) {
      check(r->outputs.size() == 2, "D: two graph outputs returned");
      if (r->outputs.size() == 2) {
        std::vector<float> s = unpack(r->outputs[0]);
        std::vector<float> z = unpack(r->outputs[1]);
        const std::vector<float> want_s = {P0, P1, P2, P0, P1, P2};
        const std::vector<float> want_z = {1 + P0, 2 + P1, 3 + P2,
                                           1 + P0, 2 + P1, 3 + P2};
        for (std::size_t i = 0; i < want_s.size(); ++i) {
          check(std::fabs(s[i] - want_s[i]) <= 1e-5f, "D: intermediate output s");
          check(std::fabs(z[i] - want_z[i]) <= 1e-5f, "D: final output z");
        }
      }
    }
  }

  // --- F. two separate MLIR regions via set_task_backend --------------------
  {
    auto doc = sx::parse_document(TWO_REGIONS);
    auto g = pl::adapt_graph(*doc);
    auto plan = pl::plan_execution(*g, *snap, cost);
    check(doc && g && plan, "F: plan succeeds");
    if (!(doc && g && plan))
      return 1;

    check(plan->tasks.size() == 4, "F: four regions (NT, MLIR, NT, MLIR)");

    const pl::TaskDesc* t_a = task_producing(*plan, "a");
    const pl::TaskDesc* t_z = task_producing(*plan, "z");
    check(t_a && t_z && t_a->as_compute()->backend == pl::BackendTag::Mlir &&
              t_z->as_compute()->backend == pl::BackendTag::Mlir &&
              t_a->id != t_z->id,
          "F: two distinct mlir regions");

    auto cpu_a = pl::CpuBackend::compile(*region_slice(*doc, *g, *t_a), {});
    auto cpu_z = pl::CpuBackend::compile(*region_slice(*doc, *g, *t_z), {});
    auto nt = pl::NativeTorchBackend::compile(*g);
    check(cpu_a && cpu_z && nt, "F: backends compile");
    if (!(cpu_a && cpu_z && nt))
      return 1;

    // NativeTorch via the tag map; each MLIR region via a per-task override.
    pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::NativeTorch, nt->get()}});
    exec.set_task_backend(t_a->id, cpu_a->get());
    exec.set_task_backend(t_z->id, cpu_z->get());

    auto r = exec.execute(*plan, *snap, {pack(X)});
    check(r.has_value(), "F: execute succeeds");
    if (r) {
      std::vector<float> got = unpack(r->outputs[0]);
      const std::vector<float> want = {Q0, Q1, Q2, Q0, Q1, Q2};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-4f, "F: two-region value correct");
    }
  }

  if (g_failures == 0) {
    std::cout << "test_coexecution_region OK\n";
    return 0;
  }
  std::cerr << "test_coexecution_region FAILED: " << g_failures << " check(s)\n";
  return 1;
}
