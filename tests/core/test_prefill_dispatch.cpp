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

// PrefillDispatchPass (MMQ1): the batched quantized_matmul_q8 projections are
// dispatched to the generated tensor-core kernel, or stay on the ggml mmq
// kernel, under the two switches the pass reads.
//
// Compiler-side and model-free: a hand-built graph with one projection is
// enough to pin the decision. Every other prefill pass is disabled through
// SONICBOOM_IR_PASS_DISABLE so the graph the assertions read is the graph the
// dispatch pass saw.
//
// Cases:
//   * auto within the budget         -> matmul_q8_tc
//   * auto over the budget           -> quantized_matmul_q8 (the ggml path)
//   * on over the budget             -> matmul_q8_tc (the override)
//   * off                            -> quantized_matmul_q8
//   * PrefillDispatchPass disabled   -> quantized_matmul_q8
//   * K not a multiple of 128        -> quantized_matmul_q8, even with =on
//   * attributes and result id       -> carried over unchanged

#include "sonicboom/compiler/BoomBuilder.h"
#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomPasses.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "sonicboom/planner/ir_passes.h"
#include "sonicboom/planner/op_kind.h"
#include "sonicboom/sx/ir.h"

#include <cstdlib>
#include <iostream>
#include <string>
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

// Every prefill pass but the dispatch pass, so only the dispatch pass changes
// the graph. The names are the switch's own vocabulary.
const char *const kOtherPasses =
    "QuantizeOncePass,RmsNormQuantizePass,GeluMulQuantizePass,"
    "SiluMulQuantizePass,BiasIntoGemvPass,ResidualAddIntoGemvPass,"
    "FuseResidualEpiloguePass,FuseSiblingGemvPass,FuseGlueRegionsPass,"
    "FuseAddRmsNormMmqPass,FuseRmsNormAddPass,FuseHeadsNormRopeCastPass,"
    "FuseKvAppendIntoPrepPass,FuseSliceScalePass,FuseGeluMulPass,"
    "ScaleIntoDensePass,FoldScaleIntoGluePass,RmsNormMmqPass,GeluMulMmqPass,"
    "SiluMulMmqPass,DenseLayoutPropagationPass,BufferReusePass";

void set_env(const char *name, const char *value) {
  if (value)
    ::setenv(name, value, 1);
  else
    ::unsetenv(name);
}

// One graph: xq [n, k] -> quantized_matmul_q8 -> y [n, rows], with the batch
// and ds the prefill emitter stamps on a batched projection.
mlir::OwningOpRef<mlir::ModuleOp> build(mlir::MLIRContext &context, int64_t n,
                                        int64_t k, int64_t rows) {
  compiler::BoomBuilder builder(&context, "prefill", 0);
  const auto xq = builder.input("xq", {n, k}, sx::DType::Float32);
  const sx::Attribute batch{"batch", n};
  const sx::Attribute ds{"ds", int64_t(0)};
  const sx::Attribute slot{"slot", int64_t(3)};
  const auto y = builder.node(planner::OpKind::QuantizedMatmulQ8, {xq}, "y",
                              {n, rows}, {batch, ds, slot});
  builder.set_output(y);
  return builder.finish();
}

mlir::Block *graph_body(mlir::ModuleOp module) {
  for (boom::GraphOp graph : module.getOps<boom::GraphOp>())
    return &graph.getBody().front();
  return nullptr;
}

bool has_node(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Block *body = graph_body(module);
  if (!body) return false;
  for (mlir::Operation &op : *body)
    if (op.getName().getStringRef() == name) return true;
  return false;
}

mlir::Operation *find_node(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Block *body = graph_body(module);
  if (!body) return nullptr;
  for (mlir::Operation &op : *body)
    if (op.getName().getStringRef() == name) return &op;
  return nullptr;
}

// Run the prefill phase over a fresh graph under the given switch values and
// report whether the projection came out as matmul_q8_tc.
bool dispatch_rewrites(int64_t n, int64_t k, int64_t rows, const char *mode,
                       const char *max_mb, const char *disable,
                       std::string *why) {
  set_env("SONICBOOM_MMQ_TC", mode);
  set_env("SONICBOOM_MMQ_TC_MAX_MB", max_mb);
  set_env("SONICBOOM_IR_PASS_DISABLE", disable);
  mlir::MLIRContext context;
  auto module = build(context, n, k, rows);
  if (!module) {
    *why = "graph build failed";
    return false;
  }
  if (!compiler::runBoomPasses(*module, planner::IrPassPhase::Prefill,
                               /*skip_kv_append_fusion=*/false)) {
    *why = "prefill passes failed";
    return false;
  }
  const bool tc = has_node(*module, "boom.matmul_q8_tc");
  const bool q8 = has_node(*module, "boom.quantized_matmul_q8");
  if (tc == q8) {
    *why = tc ? "both forms present" : "neither form present";
    return false;
  }
  return tc;
}

} // namespace

int main() {
  // k = 2048, rows = 512: the q8_1 copy is 512*(2048+256) = 1.125 MiB, so a
  // 2 MiB budget covers it and a 1 MiB budget does not.
  const int64_t n = 64, k = 2048, rows = 512;
  std::string why;

  check(dispatch_rewrites(n, k, rows, "auto", "2", kOtherPasses, &why),
        "auto within the budget dispatches to matmul_q8_tc" +
            std::string(why.empty() ? "" : " (" + why + ")"));
  check(!dispatch_rewrites(n, k, rows, "auto", "1", kOtherPasses, &why),
        "auto over the budget keeps quantized_matmul_q8" +
            std::string(why.empty() ? "" : " (" + why + ")"));
  check(dispatch_rewrites(n, k, rows, "on", "1", kOtherPasses, &why),
        "on overrides the budget" +
            std::string(why.empty() ? "" : " (" + why + ")"));
  check(!dispatch_rewrites(n, k, rows, "off", "2", kOtherPasses, &why),
        "off keeps quantized_matmul_q8" +
            std::string(why.empty() ? "" : " (" + why + ")"));
  check(!dispatch_rewrites(n, k, rows, "on", "2",
                           (std::string(kOtherPasses) +
                            ",PrefillDispatchPass").c_str(), &why),
        "the pass disable switch ablates the dispatch" +
            std::string(why.empty() ? "" : " (" + why + ")"));
  check(!dispatch_rewrites(n, 1984, rows, "on", "2", kOtherPasses, &why),
        "K not a multiple of 128 is ineligible" +
            std::string(why.empty() ? "" : " (" + why + ")"));

  // The rewrite changes only the operator name: the node id, the result id and
  // the node attributes are the ones the emitter wrote.
  set_env("SONICBOOM_MMQ_TC", "on");
  set_env("SONICBOOM_MMQ_TC_MAX_MB", nullptr);
  set_env("SONICBOOM_IR_PASS_DISABLE", kOtherPasses);
  {
    mlir::MLIRContext context;
    auto module = build(context, n, k, rows);
    check(bool(module), "the attribute case graph builds");
    if (module) {
      mlir::Operation *before = find_node(*module, "boom.quantized_matmul_q8");
      const int64_t before_id =
          before->getAttrOfType<mlir::IntegerAttr>("node_id").getInt();
      const int64_t before_result =
          before->getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids")[0];
      check(compiler::runBoomPasses(*module, planner::IrPassPhase::Prefill,
                                    false),
            "the attribute case passes");
      mlir::Operation *tc = find_node(*module, "boom.matmul_q8_tc");
      check(tc != nullptr, "the attribute case dispatches");
      if (tc) {
        check(tc->getAttrOfType<mlir::IntegerAttr>("node_id").getInt() ==
                  before_id,
              "the node id is carried over");
        check(tc->getAttrOfType<mlir::IntegerAttr>("batch").getInt() == n,
              "the batch attribute is carried over");
        check(tc->getAttrOfType<mlir::IntegerAttr>("slot").getInt() == 3,
              "the slot attribute is carried over");
        auto ids = tc->getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
        check(ids && ids.size() == 1 && ids[0] == before_result,
              "the output keeps its tensor id");
      }
    }
  }

  if (failures) {
    std::cout << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "prefill dispatch: all checks passed\n";
  return 0;
}
