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

// VS1 async-prefetch skeleton. Two halves:
//
//  * AsyncPrefetchPass: a model-free graph with three layers and three
//    model-bound constant weights. The switch (SONICBOOM_ASYNC_PREFETCH) and
//    the threshold (SONICBOOM_VRAM_THRESHOLD_MB) decide whether the markers
//    appear; when they do, the schedule is exactly async(0) at the graph start
//    and sync(L), async(L+1) before layer L's first node. The markers carry the
//    layer, the layer's weight bytes, and its weight operands, and the marked
//    graph round-trips through boomToGraph. The pass-disable ablation removes
//    them.
//
//  * AsyncPrefetchRuntime: with the switch unset/off it is a no-op (no stream,
//    counters at zero); with it on it counts begins/waits and allocates the
//    prefetch stream. LRUEvictionPolicy offers the least-recently-touched
//    resident layer.
//
// Every other pass is disabled through SONICBOOM_IR_PASS_DISABLE so the graph
// the assertions read is the one the prefetch pass wrote.

#include "sonicboom/compiler/BoomConversion.h"
#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomPasses.h"

#include "../../core/src/planner/async_prefetch.h"

#include "mlir/IR/MLIRContext.h"

#include "sonicboom/planner/ir_passes.h"
#include "sonicboom/planner/op_kind.h"
#include "sonicboom/planner/partition.h"
#include "sonicboom/sx/ir.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

namespace boom = sonicboom::compiler::boom;
namespace compiler = sonicboom::compiler;
namespace planner = sonicboom::planner;
namespace sx = sonicboom::sx;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    std::cout << "FAIL: " << what << "\n";
  }
}

void set_env(const char *name, const char *value) {
  if (value)
    ::setenv(name, value, 1);
  else
    ::unsetenv(name);
}

// Every pass but AsyncPrefetchPass, so only the prefetch pass changes the
// graph. The names are the switch's own vocabulary.
const char *const kOtherPasses =
    "PrefillDispatchPass,QuantizeOncePass,RmsNormQuantizePass,"
    "GeluMulQuantizePass,SiluMulQuantizePass,BiasIntoGemvPass,"
    "ResidualAddIntoGemvPass,FuseResidualEpiloguePass,FuseSiblingGemvPass,"
    "FuseGlueRegionsPass,FuseAddRmsNormMmqPass,FuseRmsNormAddPass,"
    "FuseHeadsNormRopeCastPass,FuseKvAppendIntoPrepPass,FuseSliceScalePass,"
    "FuseGeluMulPass,ScaleIntoDensePass,FoldScaleIntoGluePass,RmsNormMmqPass,"
    "GeluMulMmqPass,SiluMulMmqPass,DenseLayoutPropagationPass,BufferReusePass";

// Three layers, each an Add of the running activation and a 1 MiB constant
// weight. The constants are model-bound (nothing produces them) and together
// exceed a 1 MiB threshold by construction.
planner::Graph make_graph() {
  constexpr int64_t kDim = 512;
  constexpr uint64_t kBytes = uint64_t(kDim) * uint64_t(kDim) * 4;  // 1 MiB

  planner::Graph g;
  g.name = "async_prefetch";
  g.tensors.resize(7);
  auto tensor = [&](int i, const char *name, bool input, bool constant,
                    bool output) {
    auto &t = g.tensors[size_t(i)];
    t.id = planner::TensorId{uint32_t(i)};
    t.shape = {kDim, kDim};
    t.dtype = sx::DType::Float32;
    t.size_bytes = kBytes;
    t.is_graph_input = input;
    t.is_constant = constant;
    t.is_graph_output = output;
    t.name = name;
  };
  tensor(0, "x", /*input=*/true, false, false);
  tensor(1, "w0", false, /*constant=*/true, false);
  tensor(2, "w1", false, true, false);
  tensor(3, "w2", false, true, false);
  tensor(4, "y0", false, false, false);
  tensor(5, "y1", false, false, false);
  tensor(6, "y2", false, false, /*output=*/true);

  for (uint32_t l = 0; l < 3; ++l) {
    planner::GraphNodeDesc n;
    n.id = planner::GraphNodeId{l};
    n.op = planner::OpKind::Add;
    const uint32_t base = l == 0 ? 0 : 3 + l;  // x or y(l-1), plus w(l)
    n.inputs = {planner::TensorId{base}, planner::TensorId{l + 1}};
    n.outputs = {planner::TensorId{4 + l}};
    n.attributes = {sx::Attribute{"layer", int64_t(l)}};
    n.name = "add" + std::to_string(l);
    g.nodes.push_back(std::move(n));
  }
  g.inputs = {planner::TensorId{0}};
  g.outputs = {planner::TensorId{6}};
  return g;
}

std::optional<int64_t> attr_int(const std::vector<sx::Attribute> &attrs,
                                const std::string &name) {
  for (const sx::Attribute &a : attrs)
    if (a.name == name)
      if (const auto *v = std::get_if<int64_t>(&a.value)) return *v;
  return std::nullopt;
}

struct RunResult {
  bool ran = false;
  planner::Graph graph;
};

// Build, run the decode phase under (switch, threshold, disable), and convert
// back to a Graph.
RunResult run_pipeline(const char *mode, const char *threshold_mb,
                       const char *disable) {
  set_env("SONICBOOM_ASYNC_PREFETCH", mode);
  set_env("SONICBOOM_VRAM_THRESHOLD_MB", threshold_mb);
  set_env("SONICBOOM_IR_PASS_DISABLE", disable);

  RunResult result;
  mlir::MLIRContext context;
  context.getOrLoadDialect<boom::BoomDialect>();
  auto module = compiler::graphToBoom(make_graph(), &context);
  if (!module) return result;
  if (!compiler::runBoomPasses(*module, planner::IrPassPhase::Decode,
                               /*skip_kv_append_fusion=*/false))
    return result;
  auto rebuilt = compiler::boomToGraph(*module);
  if (!rebuilt) return result;
  result.ran = true;
  result.graph = std::move(*rebuilt);
  return result;
}

size_t count_op(const planner::Graph &g, planner::OpKind op) {
  size_t n = 0;
  for (const auto &node : g.nodes)
    if (node.op == op) ++n;
  return n;
}

} // namespace

int main() {
  // --- the pass: switch and threshold ------------------------------------
  {
    const RunResult off = run_pipeline(/*mode=*/nullptr, "1", kOtherPasses);
    check(off.ran, "switch unset: the pipeline runs");
    check(count_op(off.graph, planner::OpKind::AsyncWeightPrefetch) == 0 &&
              count_op(off.graph, planner::OpKind::SyncPrefetch) == 0,
          "switch unset: no markers");

    const RunResult no = run_pipeline("off", "1", kOtherPasses);
    check(no.ran && count_op(no.graph, planner::OpKind::AsyncWeightPrefetch) == 0,
          "switch off: no markers");

    const RunResult high = run_pipeline("on", "100", kOtherPasses);
    check(high.ran, "threshold above the footprint: the pipeline runs");
    check(count_op(high.graph, planner::OpKind::AsyncWeightPrefetch) == 0 &&
              count_op(high.graph, planner::OpKind::SyncPrefetch) == 0,
          "threshold above the footprint: no markers");

    const RunResult on = run_pipeline("on", "1", kOtherPasses);
    check(on.ran, "threshold below the footprint: the graph round-trips");
    if (on.ran) {
      check(count_op(on.graph, planner::OpKind::AsyncWeightPrefetch) == 3,
            "three async markers (one per layer)");
      check(count_op(on.graph, planner::OpKind::SyncPrefetch) == 3,
            "three sync markers (one per layer)");

      // The schedule, in body order: async(0) first; before layer L's node,
      // sync(L) then async(L+1) for every layer but the last.
      struct Step {
        planner::OpKind op;
        int64_t layer;
      };
      std::vector<Step> seq;
      for (const auto &node : on.graph.nodes) {
        const auto layer = attr_int(node.attributes, "layer");
        if (layer) seq.push_back({node.op, *layer});
      }
      const std::vector<Step> expected = {
          {planner::OpKind::AsyncWeightPrefetch, 0},
          {planner::OpKind::SyncPrefetch, 0},
          {planner::OpKind::AsyncWeightPrefetch, 1},
          {planner::OpKind::Add, 0},
          {planner::OpKind::SyncPrefetch, 1},
          {planner::OpKind::AsyncWeightPrefetch, 2},
          {planner::OpKind::Add, 1},
          {planner::OpKind::SyncPrefetch, 2},
          {planner::OpKind::Add, 2},
      };
      bool same = seq.size() == expected.size();
      for (size_t i = 0; same && i < seq.size(); ++i)
        same = seq[i].op == expected[i].op &&
               seq[i].layer == expected[i].layer;
      check(same, "the prefetch schedule is async(0), sync(L), async(L+1)");

      // Each async marker carries the layer's constant weight as its operand
      // and that weight's byte size as `prefetch_bytes` (1 MiB); the sync
      // markers carry neither.
      for (const auto &node : on.graph.nodes) {
        if (node.op != planner::OpKind::AsyncWeightPrefetch) {
          if (node.op == planner::OpKind::SyncPrefetch)
            check(node.inputs.empty() &&
                      attr_int(node.attributes, "prefetch_bytes") == 0,
                  "a sync marker has no operands and zero bytes");
          continue;
        }
        const auto layer = attr_int(node.attributes, "layer");
        check(layer.has_value(), "an async marker names its layer");
        if (!layer) continue;
        check(node.inputs.size() == 1 &&
                  node.inputs[0].value == uint32_t(1 + *layer),
              "an async marker reads its layer's weight tensor");
        check(attr_int(node.attributes, "prefetch_bytes") ==
                  int64_t(512 * 512 * 4),
              "an async marker records the layer's weight bytes");
      }
    }

    const std::string disabled =
        std::string(kOtherPasses) + ",AsyncPrefetchPass";
    const RunResult ablated = run_pipeline("on", "1", disabled.c_str());
    check(ablated.ran &&
              count_op(ablated.graph, planner::OpKind::AsyncWeightPrefetch) == 0,
          "AsyncPrefetchPass disabled: no markers");
  }

  // --- the runtime: no-op by default, stream when enabled ----------------
  {
    planner::AsyncPrefetchRuntime &rt = planner::AsyncPrefetchRuntime::instance();
    rt.reset_for_test();

    set_env("SONICBOOM_ASYNC_PREFETCH", nullptr);
    check(!rt.enabled(), "runtime: switch unset is disabled");
    rt.begin(0, 4096);
    rt.wait(0);
    check(rt.begins() == 0 && rt.waits() == 0 && rt.prefetched_bytes() == 0,
          "runtime off: begin/wait do nothing");
    check(rt.stream() == nullptr, "runtime off: no prefetch stream");

    set_env("SONICBOOM_ASYNC_PREFETCH", "off");
    rt.begin(0, 4096);
    check(!rt.enabled() && rt.begins() == 0, "runtime 'off': disabled");

    set_env("SONICBOOM_ASYNC_PREFETCH", "on");
    check(rt.enabled(), "runtime 'on': enabled");
    auto policy = std::make_unique<planner::LRUEvictionPolicy>();
    planner::LRUEvictionPolicy *raw = policy.get();
    rt.set_eviction_policy(std::move(policy));
    rt.begin(2, 4096);
    rt.wait(2);
    rt.begin(3, 1024);
    check(rt.begins() == 2 && rt.waits() == 1 && rt.prefetched_bytes() == 5120,
          "runtime on: the markers advance the counters");
    check(rt.eviction_policy() != nullptr && raw->last_touched() == 3,
          "runtime on: the active marker touches the eviction policy");
#ifdef SB_TEST_CUDA
    check(rt.stream() != nullptr, "runtime on: the prefetch stream exists");
#else
    check(rt.stream() == nullptr, "no CUDA build: no prefetch stream");
#endif
    rt.reset_for_test();
    check(rt.begins() == 0 && rt.waits() == 0 && rt.stream() == nullptr &&
              rt.eviction_policy() == nullptr,
          "reset_for_test clears the skeleton");
  }

  // --- the LRU stub ------------------------------------------------------
  {
    planner::LRUEvictionPolicy lru;
    lru.touch(2);
    lru.touch(0);
    lru.touch(1);
    const int resident[] = {0, 1, 2};
    check(lru.evict_candidate(resident, 0) == 2,
          "LRU evicts the earliest-touched resident layer");
    lru.touch(2);  // 2 is now the most recent; 0 is the oldest
    check(lru.evict_candidate(resident, 0) == 0,
          "a re-touch refreshes the layer's recency");
    const int partial[] = {1};
    check(lru.evict_candidate(partial, 0) == 1,
          "LRU skips non-resident order entries");
    check(!lru.evict_candidate({}, 0), "LRU has no candidate for an empty set");
    check(lru.last_touched() == 2, "LRU reports the last touch");
    lru.reset();
    check(lru.last_touched() == -1, "LRU reset forgets the touch order");
  }

  ::unsetenv("SONICBOOM_ASYNC_PREFETCH");
  ::unsetenv("SONICBOOM_VRAM_THRESHOLD_MB");
  ::unsetenv("SONICBOOM_IR_PASS_DISABLE");

  if (failures == 0) {
    std::cout << "test_async_prefetch_noop OK\n";
    return 0;
  }
  std::cout << "test_async_prefetch_noop FAILED: " << failures << " check(s)\n";
  return 1;
}
