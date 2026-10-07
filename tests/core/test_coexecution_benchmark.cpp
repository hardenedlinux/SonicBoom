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

// CPU co-execution cost benchmark (Stage 4). Measures the current CPU
// co-execution path along three axes — NativeTorch alone, MLIR alone, and mixed
// (softmax→relu with one intermediate handoff) — plus the one-shot compile cost
// and the first-vs-repeated execution difference. It reports min/mean/max over
// many iterations; it is *not* a correctness test and makes no timing assertion
// (absolute wall-clock is environment-dependent).
//
// Limitation (documented, not fabricated): the current Backend interface only
// exposes a whole-call `execute()`; it cannot separate per-component costs
// (bytes→Tensor copy, dispatch, Tensor→bytes copy, JIT invoke) inside a call.
// The "mixed" row therefore includes the executor's per-iteration validation
// (fingerprint + plan re-validation + TensorId gather), which the alone rows do
// not, so mixed is not directly comparable to the sum of the alone rows.

#include <sonicboom/planner/backend.h>
#include <sonicboom/planner/native_torch_backend.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/runtime_executor.h>
#include <sonicboom/planner/telemetry.h>

#include <sonicboom/sx/parser.h>

#include <algorithm>
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

const pl::TaskDesc* task_producing(const pl::ExecutionPlan& plan,
                                   const std::string& name) {
  for (const auto& t : plan.tasks)
    for (pl::TensorId o : t.outputs)
      if (const auto* td = plan.find_tensor(o))
        if (td->name == name)
          return &t;
  return nullptr;
}

// Slice a region task into a standalone Document (its graph_nodes select the
// kept nodes; its outputs become the graph output list).
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

struct Stats {
  double first_us;
  double min_us;
  double mean_us;
  double max_us;
};

// Run `f` `iters` times, returning first/min/mean/max wall-clock microseconds.
// `f` returns bool (success); a failed iteration is counted but still timed.
template <typename F>
Stats measure(int iters, F&& f) {
  double total = 0.0;
  double min_us = 0.0, max_us = 0.0, first_us = 0.0;
  for (int i = 0; i < iters; ++i) {
    pl::Timer t;
    bool ok = f();
    double us = t.elapsed_us();
    if (!ok)
      ++g_failures;
    if (i == 0) {
      first_us = min_us = max_us = us;
    } else {
      min_us = std::min(min_us, us);
      max_us = std::max(max_us, us);
    }
    total += us;
  }
  return Stats{first_us, min_us, total / iters, max_us};
}

void report(const char* label, const Stats& s, int iters) {
  std::cout << "  " << label << ": first=" << s.first_us
            << "us min=" << s.min_us << "us mean=" << s.mean_us
            << "us max=" << s.max_us << "us (" << iters << " iters)\n";
}

// fixed benchmark shape [128,128] float32 (16384 elements, 65536 bytes).
const char* SOFTMAX_ONLY =
    "(sonicboom-s-expr (version 0 1) (graph (name \"softmax_only\")"
    " (inputs (input \"x\" (tensor float32 (shape 128 128))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes (node softmax (inputs \"x\")"
    "   (outputs (\"y\" (tensor float32 (shape 128 128))))"
    "   (attrs (axis (int 1)))))))";

const char* RELU_ONLY =
    "(sonicboom-s-expr (version 0 1) (graph (name \"relu_only\")"
    " (inputs (input \"x\" (tensor float32 (shape 128 128))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes (node relu (inputs \"x\")"
    "   (outputs (\"y\" (tensor float32 (shape 128 128))))))))";

const char* SOFTMAX_RELU =
    "(sonicboom-s-expr (version 0 1) (graph (name \"softmax_relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 128 128))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"y\" (tensor float32 (shape 128 128))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 128 128))))))))";

// Region-merge comparison graph: softmax → relu → relu → relu → relu. The Stage
// 4 per-node model would emit 5 tasks (1 native + 4 mlir); Stage 5 region
// partitioning emits 2 (1 native + 1 merged mlir region of 4 nodes).
const char* CHAIN =
    "(sonicboom-s-expr (version 0 1) (graph (name \"chain\")"
    " (inputs (input \"x\" (tensor float32 (shape 128 128))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 128 128))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"a\" (tensor float32 (shape 128 128)))))"
    "   (node relu (inputs \"a\")"
    "     (outputs (\"b\" (tensor float32 (shape 128 128)))))"
    "   (node relu (inputs \"b\")"
    "     (outputs (\"c\" (tensor float32 (shape 128 128)))))"
    "   (node relu (inputs \"c\")"
    "     (outputs (\"z\" (tensor float32 (shape 128 128))))))))";
} // namespace

int main() {
  constexpr int kIters = 200;
  constexpr int kDim = 128;

  std::cout << "co-execution benchmark: float32 [" << kDim << "x" << kDim
            << "], " << kIters << " iterations/case\n";

  // Deterministic input: rows are arange(1..128); softmax(axis=1) has a known
  // shape and finite, positive values (sanity-checked once below).
  std::vector<float> x(kDim * kDim);
  for (int r = 0; r < kDim; ++r)
    for (int c = 0; c < kDim; ++c)
      x[static_cast<std::size_t>(r) * kDim + c] = static_cast<float>(c + 1);
  const sx::Bytes xbuf = pack(x);
  const pl::TensorValue xval = pl::TensorValue::from_host(xbuf);

  auto snap = pl::CpuResourceProvider::snapshot();
  check(snap.has_value(), "snapshot builds");
  if (!snap)
    return 1;
  pl::AnalyticalCpuCostModel cost(*snap);

  // --- NativeTorch alone (softmax) ------------------------------------------
  auto doc_softmax = sx::parse_document(SOFTMAX_ONLY);
  check(doc_softmax.has_value(), "softmax doc parses");
  if (!doc_softmax)
    return 1;
  auto g_softmax = pl::adapt_graph(*doc_softmax);
  check(g_softmax.has_value(), "softmax graph adapts");
  if (!g_softmax)
    return 1;
  auto plan_softmax = pl::plan_execution(*g_softmax, *snap, cost);
  check(plan_softmax.has_value(), "softmax plan succeeds");
  if (!plan_softmax)
    return 1;
  auto nt = pl::NativeTorchBackend::compile(*g_softmax);
  check(nt.has_value(), "native backend compiles");
  if (!nt)
    return 1;
  const pl::TaskDesc& softmax_task = plan_softmax->tasks.front();

  // --- MLIR alone (relu) -----------------------------------------------------
  auto doc_relu = sx::parse_document(RELU_ONLY);
  check(doc_relu.has_value(), "relu doc parses");
  if (!doc_relu)
    return 1;
  auto g_relu = pl::adapt_graph(*doc_relu);
  check(g_relu.has_value(), "relu graph adapts");
  if (!g_relu)
    return 1;
  auto plan_relu = pl::plan_execution(*g_relu, *snap, cost);
  check(plan_relu.has_value(), "relu plan succeeds");
  if (!plan_relu)
    return 1;
  auto cpu_relu = pl::CpuBackend::compile(*doc_relu, {});
  check(cpu_relu.has_value(), "mlir backend compiles relu");
  if (!cpu_relu)
    return 1;
  const pl::TaskDesc& relu_task = plan_relu->tasks.front();

  // --- Mixed (softmax → relu) ------------------------------------------------
  auto doc_mixed = sx::parse_document(SOFTMAX_RELU);
  check(doc_mixed.has_value(), "mixed doc parses");
  if (!doc_mixed)
    return 1;
  auto g_mixed = pl::adapt_graph(*doc_mixed);
  check(g_mixed.has_value(), "mixed graph adapts");
  if (!g_mixed)
    return 1;
  auto plan_mixed = pl::plan_execution(*g_mixed, *snap, cost);
  check(plan_mixed.has_value(), "mixed plan succeeds");
  if (!plan_mixed)
    return 1;
  auto relu_slice = pl::slice_document(*doc_mixed, {"z"});
  check(relu_slice.has_value(), "mixed relu region slices");
  if (!relu_slice)
    return 1;
  auto cpu_mixed = pl::CpuBackend::compile(*relu_slice, {});
  auto nt_mixed = pl::NativeTorchBackend::compile(*g_mixed);
  check(cpu_mixed.has_value() && nt_mixed.has_value(), "mixed backends compile");
  if (!cpu_mixed || !nt_mixed)
    return 1;
  pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
      {pl::BackendTag::Mlir, cpu_mixed->get()},
      {pl::BackendTag::NativeTorch, nt_mixed->get()}});

  // --- one-shot compile cost (report, not part of the exec loop) -------------
  {
    pl::Timer t;
    auto e = pl::CpuBackend::compile(*doc_relu, {});
    double us = t.elapsed_us();
    check(e.has_value(), "relu compile re-checks");
    std::cout << "  compile (CpuBackend/relu, one-shot): " << us << "us\n";
  }

  // --- sanity check each path once before timing -----------------------------
  {
    auto r = exec.execute(*plan_mixed, *snap, {xbuf});
    check(r.has_value() && r->outputs.size() == 1, "mixed sanity run");
    auto native = (*nt)->execute(softmax_task, {xval});
    check(native.has_value() && native->size() == 1, "native sanity run");
    auto mlir = (*cpu_relu)->execute(relu_task, {xval});
    check(mlir.has_value() && mlir->size() == 1, "mlir sanity run");
    if (native) {
      // softmax(axis=1) of arange(1..128) is positive and sums to 1 per row.
      const auto& out = native->front();
      const auto* f = reinterpret_cast<const float*>(out.host.data());
      double row_sum = 0.0;
      for (int c = 0; c < kDim; ++c)
        row_sum += static_cast<double>(f[c]);
      // softmax of arange(1..128) is peaked: the tail underflows to 0 in f32,
      // but the row must still sum to 1 and the peak (f[kDim-1]) stay positive.
      check(std::fabs(row_sum - 1.0) < 1e-3 && f[kDim - 1] > 0.0,
            "softmax row sums to 1 and peak is positive");
    }
  }

  // --- timing loops -----------------------------------------------------------
  auto native_stats = measure(kIters, [&] {
    auto r = (*nt)->execute(softmax_task, {xval});
    return r.has_value() && r->size() == 1;
  });
  auto mlir_stats = measure(kIters, [&] {
    auto r = (*cpu_relu)->execute(relu_task, {xval});
    return r.has_value() && r->size() == 1;
  });
  auto mixed_stats = measure(kIters, [&] {
    auto r = exec.execute(*plan_mixed, *snap, {xbuf});
    return r.has_value() && r->outputs.size() == 1;
  });

  report("native-torch alone (softmax)", native_stats, kIters);
  report("mlir alone (relu)", mlir_stats, kIters);
  report("mixed (softmax->relu, 1 handoff)", mixed_stats, kIters);

  // --- Region-merge comparison: per-node (Stage 4 model) vs merged (Stage 5) --
  {
    auto doc = sx::parse_document(CHAIN);
    auto g = pl::adapt_graph(*doc);
    auto plan = pl::plan_execution(*g, *snap, cost);
    check(doc && g && plan, "chain plan succeeds");
    if (!(doc && g && plan))
      return 1;

    const std::size_t n_nodes = g->nodes.size();
    const std::size_t n_regions = plan->tasks.size();
    const pl::TaskDesc* t_softmax = task_producing(*plan, "s");
    const pl::TaskDesc* t_region = task_producing(*plan, "z");
    check(t_softmax && t_region, "chain regions located");
    if (!(t_softmax && t_region))
      return 1;
    const std::size_t merged_mlir_nodes =
        t_region->as_compute()->graph_nodes.size();

    std::cout << "\nregion-merge (softmax -> relu x4, [128x128] f32):\n";
    std::cout << "  structure: " << n_nodes << " nodes -> " << n_regions
              << " regions (merged " << merged_mlir_nodes
              << " mlir nodes); per-node model would emit " << n_nodes
              << " tasks\n";

    auto nt_chain = pl::NativeTorchBackend::compile(*g);
    check(nt_chain.has_value(), "chain native backend compiles");
    if (!nt_chain)
      return 1;

    // "after" (merged): one MLIR region compile + one native = 2 compiles.
    auto merged_slice = region_slice(*doc, *g, *t_region);
    check(merged_slice.has_value(), "chain merged region slices");
    if (!merged_slice)
      return 1;
    double after_mlir_us = 0.0;
    std::unique_ptr<pl::CpuBackend> merged_cpu;
    {
      pl::Timer t;
      auto cpu = pl::CpuBackend::compile(*merged_slice, {});
      after_mlir_us = t.elapsed_us();
      check(cpu.has_value(), "chain merged region compiles");
      if (!cpu)
        return 1;
      merged_cpu = std::move(*cpu);
    }

    // "before" (per-node): four per-node MLIR slices = 4 compiles + 1 native.
    double before_mlir_us = 0.0;
    std::vector<std::unique_ptr<pl::CpuBackend>> per_node;
    {
      const char* outs[4] = {"a", "b", "c", "z"};
      for (const char* o : outs) {
        auto slice = pl::slice_document(*doc, {o});
        check(slice.has_value(), "per-node slice succeeds");
        if (!slice)
          return 1;
        pl::Timer t;
        auto cpu = pl::CpuBackend::compile(*slice, {});
        before_mlir_us += t.elapsed_us();
        check(cpu.has_value(), "per-node mlir compiles");
        if (!cpu)
          return 1;
        per_node.push_back(std::move(*cpu));
      }
    }

    std::cout << "  compile: before=" << (1 + per_node.size())
              << " (1 native + " << per_node.size()
              << " per-node mlir) vs after=2 (1 native + 1 merged mlir)\n";
    std::cout << "  compile time (mlir only, one-shot, no cache): before="
              << before_mlir_us << "us vs after=" << after_mlir_us << "us\n";

    // Exec "after" through the executor (merged region, 2 tasks).
    pl::RuntimeExecutor exec(std::map<pl::BackendTag, pl::Backend*>{
        {pl::BackendTag::Mlir, merged_cpu.get()},
        {pl::BackendTag::NativeTorch, nt_chain->get()}});

    // Exec "before" as a manually-orchestrated per-node chain. This bypasses the
    // executor (no per-iteration fingerprint/plan re-validation/gather), so it
    // is a *lower bound* on the true per-node path cost — conservative (favors
    // "before"). A dummy compute task suffices: CpuBackend only checks the kind.
    pl::TaskDesc dummy;
    dummy.kind = pl::TaskKind::Compute;

    // Sanity: both paths must compute the same result (softmax(x); relu is
    // identity on its positive output).
    {
      auto a = exec.execute(*plan, *snap, {xbuf});
      check(a && a->outputs.size() == 1, "chain merged sanity run");

      auto s = (*nt_chain)->execute(*t_softmax, {xval});
      bool per_node_ok = s && s->size() == 1;
      std::vector<pl::TensorValue> acc =
          per_node_ok ? std::vector<pl::TensorValue>{(*s)[0]}
                      : std::vector<pl::TensorValue>{};
      for (const auto& cpu : per_node) {
        auto r = cpu->execute(dummy, {acc.empty() ? pl::TensorValue{} : acc[0]});
        per_node_ok = per_node_ok && r && r->size() == 1;
        if (!r || r->size() != 1)
          break;
        acc = std::move(*r);
      }
      check(per_node_ok, "chain per-node sanity run");

      if (a && per_node_ok && a->outputs[0].size() == acc[0].host.size()) {
        const auto* f1 = reinterpret_cast<const float*>(a->outputs[0].data());
        const auto* f2 = reinterpret_cast<const float*>(acc[0].host.data());
        bool same = true;
        for (int i = 0; i < kDim * kDim; ++i)
          if (std::fabs(f1[i] - f2[i]) > 1e-5f) {
            same = false;
            break;
          }
        check(same, "merged and per-node chains agree");
      }
    }

    auto merged_stats = measure(kIters, [&] {
      auto r = exec.execute(*plan, *snap, {xbuf});
      return r.has_value() && r->outputs.size() == 1;
    });
    // Merged path also run manually (bypassing the executor) so the two manual
    // rows are comparable: the merge's effect on exec time is 2 backend calls
    // (1 native + 1 region) vs 5 (1 native + 4 per-node), i.e. fewer JIT-call
    // boundaries and fewer intermediate buffer handoffs.
    auto merged_manual_stats = measure(kIters, [&] {
      auto s = (*nt_chain)->execute(*t_softmax, {xval});
      if (!s || s->size() != 1)
        return false;
      auto z = merged_cpu->execute(dummy, {(*s)[0]});
      return z && z->size() == 1;
    });
    auto pernode_stats = measure(kIters, [&] {
      // softmax (native) → relu → relu → relu → relu (4 per-node mlir calls).
      auto s = (*nt_chain)->execute(*t_softmax, {xval});
      if (!s || s->size() != 1)
        return false;
      std::vector<pl::TensorValue> acc = {(*s)[0]};
      for (const auto& cpu : per_node) {
        auto r = cpu->execute(dummy, {acc[0]});
        if (!r || r->size() != 1)
          return false;
        acc = std::move(*r);
      }
      return true;
    });

    report("merged (2 tasks, 1 region, via executor)", merged_stats, kIters);
    report("merged (2 calls, manual, no executor overhead)", merged_manual_stats,
           kIters);
    report("per-node (5 calls, manual, no executor overhead)", pernode_stats,
           kIters);
  }

  if (g_failures == 0) {
    std::cout << "test_coexecution_benchmark OK\n";
    return 0;
  }
  std::cerr << "test_coexecution_benchmark FAILED: " << g_failures << " check(s)\n";
  return 1;
}
