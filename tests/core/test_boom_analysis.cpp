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

// boom-r4 F2: the Boom data-flow analyses, on hand-checked IR.
//
// Every expectation below is computed by hand in the comment next to it. The
// transfer functions are checked where a mis-signed interval or a swapped
// error term would show; the reduction table is checked against the standard
// forward-error bounds (K*u*Σ|ab| sequential, log2(K)*u*Σ|ab| blocked); the
// liveness comparison is checked on an unsound mark, a sound one, and the case
// where the pass is stricter than liveness. The analysis is read-only, so the
// module text must be identical before and after.

#include "sonicboom/compiler/BoomAnalysis.h"
#include "sonicboom/compiler/BoomBuilder.h"
#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomPasses.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/MLIRContext.h"

#include "sonicboom/model/plan.h"
#include "sonicboom/sx/ir.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

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

void near(double got, double want, const std::string &what, double tol = 1e-9) {
  if (!(std::abs(got - want) <= tol * std::max(1.0, std::abs(want)))) {
    ++failures;
    std::cout << "FAIL: " << what << " (got " << got << ", want " << want
              << ")\n";
  }
}

std::string moduleText(mlir::ModuleOp m) {
  std::string text;
  llvm::raw_string_ostream os(text);
  m.print(os);
  return text;
}

// The first value fact whose node name and op match, or nullptr.
const compiler::ValueFact *fact(const compiler::AnalysisResult &r,
                                const std::string &node,
                                const std::string &op) {
  for (const auto &v : r.values)
    if (v.node == node && v.op == op) return &v;
  return nullptr;
}

const compiler::ReductionReport *reduction(const compiler::AnalysisResult &r,
                                           const std::string &op, int64_t k) {
  for (const auto &x : r.reductions)
    if (x.op == op && x.k == k) return &x;
  return nullptr;
}

// x and y are graph inputs with the ranges in `facts`; the body is the caller's.
struct Built {
  // The context outlives the module: declared first, destroyed last.
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

Built build(std::function<void(compiler::BoomBuilder &)> body) {
  Built b;
  b.context = std::make_unique<mlir::MLIRContext>();
  b.context->getOrLoadDialect<boom::BoomDialect>();
  compiler::BoomBuilder builder(b.context.get(), "analysis_test", 0);
  body(builder);
  b.module = builder.finish();
  return b;
}

} // namespace

int main() {
  compiler::RangeFacts facts;
  // x: every element in [-1, 3], within 0.5 of exact.
  facts.setTensor("x", compiler::RangeError::bounded(-1.0, 3.0, 0.5));
  // y: [-2, 4], within 0.25.
  facts.setTensor("y", compiler::RangeError::bounded(-2.0, 4.0, 0.25));

  // --- elementwise transfer functions ------------------------------------
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto y = b.input("y", {4}, sx::DType::Float32);
      auto add = b.node(planner::OpKind::Add, {x, y}, "added", {4}, {});
      b.set_output(add);
    });
    check(static_cast<bool>(b.module), "add graph builds");
    if (!b.module) return 1;
    const compiler::AnalysisResult r = compiler::analyzeBoom(*b.module, facts);
    const auto *v = fact(r, "add", "boom.add");
    check(v != nullptr, "the add's result has a fact");
    if (v) {
      // [-1,3] + [-2,4] = [-3,7]; error 0.5 + 0.25 plus one f32 rounding of 7.
      near(v->range.lo, -3.0, "add lo");
      near(v->range.hi, 7.0, "add hi");
      check(v->range.err >= 0.75 && v->range.err <= 0.75 + 1e-6,
            "add err = 0.75 + one rounding");
    }
  }
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto s = b.node(planner::OpKind::Scale, {x}, "scaled", {4},
                      {sx::Attribute{"scale", 2.0}});
      b.set_output(s);
    });
    check(static_cast<bool>(b.module), "scale graph builds");
    if (b.module) {
      compiler::RangeFacts f;
      f.setTensor("x", compiler::RangeError::bounded(-1.0, 3.0, 0.5));
      const auto r = compiler::analyzeBoom(*b.module, f);
      const auto *v = fact(r, "scale", "boom.scale");
      check(v != nullptr, "the scale's result has a fact");
      if (v) {
        near(v->range.lo, -2.0, "scale lo");
        near(v->range.hi, 6.0, "scale hi");
        check(v->range.err >= 1.0 && v->range.err <= 2.0,
              "scale err = 2 * 0.5 + rounding");
      }
    }
  }
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto y = b.input("y", {4}, sx::DType::Float32);
      auto m = b.node(planner::OpKind::Mul, {x, y}, "product", {4}, {});
      b.set_output(m);
    });
    if (!b.module) {
      check(false, "mul graph builds");
    } else {
      compiler::RangeFacts f;
      f.setTensor("x", compiler::RangeError::bounded(-1.0, 3.0, 0.5));
      f.setTensor("y", compiler::RangeError::bounded(-2.0, 4.0, 0.25));
      const auto r = compiler::analyzeBoom(*b.module, f);
      const auto *v = fact(r, "mul", "boom.mul");
      check(v != nullptr, "the mul's result has a fact");
      if (v) {
        // Corners: {-1,-2}=2, {-1,4}=-4, {3,-2}=-6, {3,4}=12 -> [-6,12].
        near(v->range.lo, -6.0, "mul lo");
        near(v->range.hi, 12.0, "mul hi");
      }
    }
  }
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto s = b.node(planner::OpKind::Silu, {x}, "silu", {4}, {});
      b.set_output(s);
    });
    if (!b.module) {
      check(false, "silu graph builds");
    } else {
      compiler::RangeFacts f;
      f.setTensor("x", compiler::RangeError::bounded(-1.0, 3.0, 0.1));
      const auto r = compiler::analyzeBoom(*b.module, f);
      const auto *v = fact(r, "silu", "boom.silu");
      check(v != nullptr, "the silu's result has a fact");
      if (v) {
        // SiLU(-1) = -0.26894, SiLU(3) = 2.85772; its minimum (-1.278) is
        // outside [-1, 3], so the endpoints are the hull.
        near(v->range.lo, -1.0 / (1.0 + std::exp(1.0)), "silu lo");
        near(v->range.hi, 3.0 / (1.0 + std::exp(-3.0)), "silu hi");
      }
    }
  }

  // --- a reduction: the quantized matvec ---------------------------------
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {2560}, sx::DType::Float32);
      auto q = b.node(planner::OpKind::QuantizedMatmul, {x}, "q", {2560},
                      {sx::Attribute{"slot",
                                     int64_t(sonicboom::model::WeightSlot::AttnQ)},
                       sx::Attribute{"layer", int64_t(0)}});
      b.set_output(q);
    });
    if (!b.module) {
      check(false, "matvec graph builds");
    } else {
      compiler::RangeFacts f;
      f.setTensor("x", compiler::RangeError::bounded(-2.0, 2.0, 0.1));
      f.setWeight(int64_t(sonicboom::model::WeightSlot::AttnQ), 0,
                  compiler::RangeError::bounded(-0.5, 0.5, 0.01));
      const auto r = compiler::analyzeBoom(*b.module, f);
      const auto *v = fact(r, "quantized_matmul", "boom.quantized_matmul");
      check(v != nullptr, "the matvec's result has a fact");
      if (v) {
        // |out| <= K * max|x| * max|w| = 2560 * 2 * 0.5 = 2560.
        near(v->range.lo, -2560.0, "matvec lo");
        near(v->range.hi, 2560.0, "matvec hi");
        // err >= K*(0.1*0.5 + 2*0.01) = 2560*0.07 = 179.2 (input errors).
        check(v->range.err >= 179.2, "matvec err carries both input budgets");
      }
      const auto *rr = reduction(r, "boom.quantized_matmul", 2560);
      check(rr != nullptr, "the matvec is a reduction row");
      if (rr) {
        // sum|ab| <= 2560*2*0.5 = 2560.
        near(rr->sum_abs, 2560.0, "sum of absolute terms");
        // Sequential: K*u*sum = 2560 * 2^-24 * 2560 = 0.390625.
        near(rr->err_seq, 2560.0 * std::ldexp(1.0, -24) * 2560.0,
             "sequential bound");
        // Blocked: ceil(log2(2560)) = 12 levels: 12 * 2^-24 * 2560.
        near(rr->err_tree, 12.0 * std::ldexp(1.0, -24) * 2560.0,
             "tree bound");
        check(!rr->within_seq, "sequential f32 is not provable at 1e-5 rel");
        check(rr->within_tree, "blocked f32 is provable at 1e-5 rel");
      }
    }
  }

  // --- a norm reduction ---------------------------------------------------
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {2560}, sx::DType::Float32);
      auto n = b.node(planner::OpKind::RmsNorm, {x}, "norm", {2560},
                      {sx::Attribute{"slot",
                                     int64_t(sonicboom::model::WeightSlot::AttnNorm)},
                       sx::Attribute{"layer", int64_t(0)}});
      b.set_output(n);
    });
    if (!b.module) {
      check(false, "norm graph builds");
    } else {
      compiler::RangeFacts f;
      f.setTensor("x", compiler::RangeError::bounded(-2.0, 2.0, 0.0));
      f.setWeight(int64_t(sonicboom::model::WeightSlot::AttnNorm), 0,
                  compiler::RangeError::bounded(-1.0, 1.0, 0.0));
      const auto r = compiler::analyzeBoom(*b.module, f);
      const auto *v = fact(r, "rms_norm", "boom.rms_norm");
      check(v != nullptr, "the norm's result has a fact");
      if (v) {
        // |x_j| / rms(x) <= sqrt(n): 50.5964 for n = 2560; max|w| = 1.
        near(v->range.hi, std::sqrt(2560.0), "norm bound");
      }
      const auto *rr = reduction(r, "boom.rms_norm", 2560);
      check(rr != nullptr, "the norm is a reduction row");
      if (rr) {
        // Terms are x^2: sum <= 2560 * 4 = 10240.
        near(rr->sum_abs, 10240.0, "sum of squares bound");
        check(rr->within_tree, "blocked sum-of-squares is provable at 1e-5");
      }
    }
  }

  // --- liveness vs the in-place marks ----------------------------------
  {
    // add(x,y) -> s; add(s, x) -> t (graph output). Marking the first node
    // in-place borrows s's buffer, and the second node reads s afterwards.
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto y = b.input("y", {4}, sx::DType::Float32);
      auto s = b.node(planner::OpKind::Add, {x, y}, "s", {4}, {});
      auto t = b.node(planner::OpKind::Add, {s, x}, "t", {4}, {});
      b.set_output(t);
    });
    if (!b.module) {
      check(false, "liveness graph builds");
    } else {
      size_t index = 0;
      b.module->walk([&](mlir::Operation *op) {
        if (op->getName().getStringRef() != "boom.add") return;
        if (index == 0)
          op->setAttr("inplace",
                      mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), 1));
        ++index;
      });
      compiler::RangeFacts f;
      const auto r = compiler::analyzeBoom(*b.module, f);
      bool unsound = false, strict = false;
      for (const auto &c : r.liveness) {
        if (c.kind == "unsound") unsound = true;
        if (c.kind == "strict") strict = true;
      }
      check(unsound, "an in-place mark over a live value is reported unsound");
      check(!strict, "no strict note for a single-use, dead-after operand");
    }
  }
  {
    // s is read by both mul and scale; the scale is s's last use. Marking the
    // scale in place is sound under dense liveness but the pass would decline
    // it (operand 0 has two uses): a "strict" note, not an unsoundness.
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto y = b.input("y", {4}, sx::DType::Float32);
      auto s = b.node(planner::OpKind::Add, {x, y}, "s", {4}, {});
      auto m = b.node(planner::OpKind::Mul, {s, x}, "m", {4}, {});
      auto v = b.node(planner::OpKind::Scale, {s}, "v", {4},
                      {sx::Attribute{"scale", 1.0}});
      b.set_output(m);
      (void)v;
    });
    if (!b.module) {
      check(false, "second liveness graph builds");
    } else {
      b.module->walk([&](mlir::Operation *op) {
        if (op->getName().getStringRef() != "boom.scale") return;
        op->setAttr(
            "inplace",
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32), 1));
      });
      compiler::RangeFacts f;
      const auto r = compiler::analyzeBoom(*b.module, f);
      bool strict = false, unsound = false;
      for (const auto &c : r.liveness) {
        if (c.kind == "strict") strict = true;
        if (c.kind == "unsound") unsound = true;
      }
      check(strict, "an earlier-but-not-later use is a strict note");
      check(!unsound, "an earlier-but-not-later use is not unsound");
    }
  }

  // --- the marks the pipeline's pass actually emits ----------------------
  {
    // x, y in; s = add(x, y); t = mul(s, x); u = scale(t) out. The mul's
    // operand 0 (s) is one node result with one use, its own result t is not
    // an output: exactly one mark, and it is sound (nothing reads s after the
    // mul). The scale cannot be marked because its result is the output.
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto y = b.input("y", {4}, sx::DType::Float32);
      auto s = b.node(planner::OpKind::Add, {x, y}, "s", {4}, {});
      auto t = b.node(planner::OpKind::Mul, {s, x}, "tm", {4}, {});
      auto u = b.node(planner::OpKind::Scale, {t}, "u", {4},
                      {sx::Attribute{"scale", 2.0}});
      b.set_output(u);
    });
    if (!b.module) {
      check(false, "marked graph builds");
    } else {
      check(compiler::markBufferReuse(*b.module), "BufferReusePass runs");
      // Running it again must replace the marks, not stack a second copy (the
      // verifier rejects duplicate node_attrs names).
      check(compiler::markBufferReuse(*b.module), "BufferReusePass re-runs");
      bool duplicates = false;
      b.module->walk([&](mlir::Operation *op) {
        if (auto attrs = op->getAttrOfType<mlir::ArrayAttr>("node_attrs")) {
          size_t inplace = 0;
          for (mlir::Attribute a : attrs)
            if (auto na = mlir::dyn_cast<boom::NodeAttr>(a))
              if (na.getName() == "inplace") ++inplace;
          if (inplace > 1) duplicates = true;
        }
      });
      check(!duplicates, "a second pass run leaves one inplace entry per node");
      compiler::RangeFacts f;
      const auto r = compiler::analyzeBoom(*b.module, f);
      bool checked = false;
      for (const auto &n : r.notes)
        if (n.find("inplace marks checked: 1 ") == 0) checked = true;
      check(checked, "the pass's in-place mark reaches the analysis");
      check(r.liveness.empty(), "a sound mark produces no liveness row");
    }
  }

  // --- read-only gate -----------------------------------------------------
  {
    Built b = build([](compiler::BoomBuilder &b) {
      auto x = b.input("x", {4}, sx::DType::Float32);
      auto s = b.node(planner::OpKind::Silu, {x}, "s", {4}, {});
      auto t = b.node(planner::OpKind::Scale, {s}, "t", {4},
                      {sx::Attribute{"scale", 0.5}});
      b.set_output(t);
    });
    if (!b.module) {
      check(false, "read-only graph builds");
    } else {
      const std::string before = moduleText(*b.module);
      compiler::RangeFacts f;
      f.setTensor("x", compiler::RangeError::bounded(-1.0, 1.0, 0.0));
      compiler::analyzeBoom(*b.module, f);
      const std::string after = moduleText(*b.module);
      check(before == after, "the analysis leaves the IR byte-identical");
    }
  }

  // --- facts file round trip ---------------------------------------------
  {
    compiler::RangeFacts f;
    f.setWeight(14, 0, compiler::RangeError::bounded(-0.5, 0.5, 0.01));
    f.setTensor("h0", compiler::RangeError::bounded(-3.0, 3.0, 0.001));
    const std::string path = "/tmp/boom-analysis-facts-test.txt";
    std::string error;
    check(f.save(path, &error), "facts save: " + error);
    compiler::RangeFacts g;
    check(g.load(path, &error), "facts load: " + error);
    check(g.weightCount() == 1 && g.tensorCount() == 1, "facts counts survive");
    const auto w = g.weight(14, 0);
    check(w && w->lo == -0.5 && w->hi == 0.5 && w->err == 0.01,
          "weight facts survive the round trip");
  }

  if (failures != 0) {
    std::cout << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "boom analysis: all checks passed\n";
  return 0;
}
