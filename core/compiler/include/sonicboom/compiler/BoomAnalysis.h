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

//===----------------------------------------------------------------------===//
// Data-flow facts over Boom IR (boom-r4 F2).
//
// Two analyses, both read-only — no pass is run and no attribute is written:
//
//   * A sparse forward value-range / absolute-error analysis
//     (mlir::dataflow::SparseForwardDataFlowAnalysis) carrying, per SSA
//     value, the interval every element is proved to lie in and the absolute
//     error bound against the exact real-arithmetic result. Transfer functions
//     cover the Boom ops the Gemma/Qwen/DeepSeek plans use: the quantized and
//     dense matmuls (with per-block weight scales), the norms, RoPE, softmax,
//     elementwise adds/muls, quantize, and the fused norm/quantize forms.
//     Weight ranges are not in the IR — a quantized weight is named by the
//     node's `slot`/`layer` attributes — so they come from `RangeFacts`, which
//     the caller fills from the loaded weights and from a calibration run.
//
//   * Dense backward liveness over boom.graph's flat node list, which is what
//     BufferReusePass's `inplace=1` / `buf=N` marks must agree with: a mark is
//     sound exactly when the borrowed buffer's value is dead after the marked
//     node. The per-position live-after set is derived from a backward sweep;
//     mlir::dataflow::RunLivenessAnalysis provides the independent per-value
//     liveness it is cross-checked against.
//
// The first consumer of the range facts is the reduction-safety table: for
// every reduction kernel (matmul family, RMS norm, quantize block max) it
// reports whether f32 accumulation is provably within a stated relative
// epsilon, given the propagated operand bounds.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sonicboom::compiler {

// What is proved about one tensor value: every element lies in [lo, hi] and
// differs from the exact (real-arithmetic) result by at most `err`, absolute.
// `known == false` means nothing is proved; every transfer function then
// produces unknown.
struct RangeError {
  double lo = 0.0;
  double hi = 0.0;
  double err = 0.0;
  bool known = false;

  static RangeError unknown();
  static RangeError bounded(double lo, double hi, double err);

  // The join of an SSA value's incoming information: the hull, with the larger
  // error budget (both are sound over-approximations of the same value).
  static RangeError join(const RangeError &a, const RangeError &b);

  bool operator==(const RangeError &o) const;
  bool operator!=(const RangeError &o) const { return !(*this == o); }

  double magnitude() const;  // max(|lo|, |hi|)

  // The lattice's printing hook (Lattice<T>::print calls this).
  void print(llvm::raw_ostream &os) const;
};

// The facts the IR cannot carry: weight ranges by (slot, layer), and — after a
// calibration run — measured activation ranges by tensor name. `err` is the
// value's own error budget (a weight's quantization error, a measured
// activation's measurement slack); the analysis adds it like any other.
class RangeFacts {
public:
  void setWeight(int64_t slot, int64_t layer, const RangeError &r);
  void setTensor(llvm::StringRef name, const RangeError &r);

  std::optional<RangeError> weight(int64_t slot, int64_t layer) const;
  std::optional<RangeError> tensor(llvm::StringRef name) const;

  // One line per fact:
  //   weight <slot> <layer> <lo> <hi> <err>
  //   tensor <name> <lo> <hi> <err>
  // `name` is taken whole (it may contain no whitespace).
  bool load(llvm::StringRef path, std::string *error);
  bool save(llvm::StringRef path, std::string *error) const;

  // Fold every fact of `other` in (later entries win, so a calibrated tensor
  // overrides a weight of the same name).
  void mergeFrom(const RangeFacts &other);

  size_t weightCount() const { return weights_.size(); }
  size_t tensorCount() const { return tensor_names_.size(); }

private:
  struct WeightKey {
    int64_t slot = 0;
    int64_t layer = 0;
    bool operator==(const WeightKey &o) const {
      return slot == o.slot && layer == o.layer;
    }
  };
  struct WeightKeyHash {
    size_t operator()(const WeightKey &k) const {
      return (size_t(k.slot) * 1315423911u) ^ size_t(k.layer);
    }
  };
  // Kept as parallel vectors so the header does not pull in DenseMap; the
  // table is small (one entry per bound weight, a few hundred).
  std::vector<WeightKey> weights_;
  std::vector<RangeError> weight_ranges_;
  std::vector<std::string> tensor_names_;
  std::vector<RangeError> tensor_ranges_;
};

// One reduction kernel's safety row. `sum_abs` is the proved bound on the sum
// of absolute terms the reduction would accumulate (K terms of at most
// max|a|*max|b|); `err_seq` and `err_tree` are the standard forward-error
// bounds for sequential and blocked/tree accumulation of those terms.
struct ReductionReport {
  std::string node;
  std::string op;
  std::string unit;  // "<kind>:<name>" of the graph/step it came from
  int64_t k = 0;
  double sum_abs = 0.0;
  double err_seq = 0.0;
  double err_tree = 0.0;
  bool within_seq = false;
  bool within_tree = false;
};

// A BufferReusePass mark that dense liveness does not justify, or a mark dense
// liveness would allow that the pass skipped (the latter is not a bug: the
// pass is stricter than liveness on purpose).
struct LivenessCheck {
  std::string node;
  std::string kind;  // "unsound" | "strict" | "missed"
  std::string why;
  std::string unit;  // "<kind>:<name>"
};

// What was proved about one node's first result, in node order. The op name is
// the Boom mnemonic and `node` the emitter's node name.
struct ValueFact {
  std::string node;
  std::string op;
  RangeError range;
  std::string unit;  // "<kind>:<name>"
};

struct AnalysisResult {
  std::vector<ValueFact> values;
  std::vector<ReductionReport> reductions;
  std::vector<LivenessCheck> liveness;
  std::vector<std::string> notes;  // values the analysis could not bound
  size_t values_analyzed = 0;
  size_t values_unknown = 0;
  double epsilon = 1e-5;
  bool ok = false;
};

// Run both analyses over every boom.graph in `module`. Analysis only: the IR
// is not modified, so a plan built with this call is byte-identical to one
// built without it.
AnalysisResult analyzeBoom(mlir::ModuleOp module, const RangeFacts &facts,
                           double epsilon = 1e-5);

} // namespace sonicboom::compiler
