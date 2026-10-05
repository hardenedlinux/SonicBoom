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
    auto native = (*nt)->execute(softmax_task, {xbuf});
    check(native.has_value() && native->size() == 1, "native sanity run");
    auto mlir = (*cpu_relu)->execute(relu_task, {xbuf});
    check(mlir.has_value() && mlir->size() == 1, "mlir sanity run");
    if (native) {
      // softmax(axis=1) of arange(1..128) is positive and sums to 1 per row.
      const auto& out = native->front();
      const auto* f = reinterpret_cast<const float*>(out.data());
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
    auto r = (*nt)->execute(softmax_task, {xbuf});
    return r.has_value() && r->size() == 1;
  });
  auto mlir_stats = measure(kIters, [&] {
    auto r = (*cpu_relu)->execute(relu_task, {xbuf});
    return r.has_value() && r->size() == 1;
  });
  auto mixed_stats = measure(kIters, [&] {
    auto r = exec.execute(*plan_mixed, *snap, {xbuf});
    return r.has_value() && r->outputs.size() == 1;
  });

  report("native-torch alone (softmax)", native_stats, kIters);
  report("mlir alone (relu)", mlir_stats, kIters);
  report("mixed (softmax->relu, 1 handoff)", mixed_stats, kIters);

  if (g_failures == 0) {
    std::cout << "test_coexecution_benchmark OK\n";
    return 0;
  }
  std::cerr << "test_coexecution_benchmark FAILED: " << g_failures << " check(s)\n";
  return 1;
}
