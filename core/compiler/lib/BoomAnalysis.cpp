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

// The two analyses of BoomAnalysis.h. Nothing here mutates the IR.

#include "sonicboom/compiler/BoomAnalysis.h"

#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/LivenessAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <utility>

namespace sonicboom::compiler {

namespace {

using mlir::ArrayRef;
using mlir::Operation;
using mlir::Value;

// --- constants --------------------------------------------------------------

// The unit roundoff of the two formats the kernels compute in, and the unit
// roundoff of the accumulation format (f32 for the f32 paths, f32 for the f32
// tail of the dp4a paths). u = 2^-24 is the f32 unit roundoff; the fp16
// representation error is 2^-11.
constexpr double kF32Eps = 5.9604644775390625e-08;  // 2^-24
constexpr double kF16Eps = 4.8828125e-04;           // 2^-11

// log2 ceiling as the depth of a balanced reduction over `k` terms.
double treeDepth(int64_t k) {
  if (k <= 1) return 0.0;
  double d = 0.0;
  while ((int64_t(1) << int64_t(d)) < k) d += 1.0;
  return d;
}

} // namespace

// --- RangeError -------------------------------------------------------------

RangeError RangeError::unknown() { return RangeError{}; }

RangeError RangeError::bounded(double lo, double hi, double err) {
  RangeError r;
  r.lo = lo;
  r.hi = hi;
  r.err = err;
  r.known = true;
  return r;
}

RangeError RangeError::join(const RangeError &a, const RangeError &b) {
  if (!a.known) return b;
  if (!b.known) return a;
  RangeError r;
  r.lo = std::min(a.lo, b.lo);
  r.hi = std::max(a.hi, b.hi);
  r.err = std::max(a.err, b.err);
  r.known = true;
  return r;
}

bool RangeError::operator==(const RangeError &o) const {
  if (known != o.known) return false;
  if (!known) return true;
  return lo == o.lo && hi == o.hi && err == o.err;
}

double RangeError::magnitude() const {
  if (!known) return std::numeric_limits<double>::infinity();
  return std::max(std::abs(lo), std::abs(hi));
}

void RangeError::print(llvm::raw_ostream &os) const {
  if (!known) {
    os << "unknown";
    return;
  }
  os << '[' << lo << ", " << hi << "] +/- " << err;
}

// --- RangeFacts -------------------------------------------------------------

void RangeFacts::setWeight(int64_t slot, int64_t layer, const RangeError &r) {
  for (size_t i = 0; i < weights_.size(); ++i)
    if (weights_[i].slot == slot && weights_[i].layer == layer) {
      weight_ranges_[i] = r;
      return;
    }
  weights_.push_back(WeightKey{slot, layer});
  weight_ranges_.push_back(r);
}

void RangeFacts::setTensor(llvm::StringRef name, const RangeError &r) {
  for (size_t i = 0; i < tensor_names_.size(); ++i)
    if (tensor_names_[i] == name) {
      tensor_ranges_[i] = r;
      return;
    }
  tensor_names_.push_back(name.str());
  tensor_ranges_.push_back(r);
}

std::optional<RangeError> RangeFacts::weight(int64_t slot,
                                             int64_t layer) const {
  for (size_t i = 0; i < weights_.size(); ++i)
    if (weights_[i].slot == slot && weights_[i].layer == layer)
      return weight_ranges_[i];
  return std::nullopt;
}

std::optional<RangeError> RangeFacts::tensor(llvm::StringRef name) const {
  for (size_t i = 0; i < tensor_names_.size(); ++i)
    if (tensor_names_[i] == name) return tensor_ranges_[i];
  return std::nullopt;
}

bool RangeFacts::load(llvm::StringRef path, std::string *error) {
  std::ifstream in(path.str());
  if (!in) {
    if (error) *error = "cannot open facts file '" + path.str() + "'";
    return false;
  }
  std::string line;
  int64_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    RangeError r;
    if (kind == "weight") {
      int64_t slot = 0, layer = 0;
      if (!(ss >> slot >> layer >> r.lo >> r.hi >> r.err)) {
        if (error) *error = path.str() + ":" + std::to_string(lineno);
        return false;
      }
      r.known = true;
      setWeight(slot, layer, r);
    } else if (kind == "tensor") {
      std::string name;
      if (!(ss >> name >> r.lo >> r.hi >> r.err)) {
        if (error) *error = path.str() + ":" + std::to_string(lineno);
        return false;
      }
      r.known = true;
      setTensor(name, r);
    } else {
      if (error) *error = path.str() + ":" + std::to_string(lineno) +
                          ": unknown fact kind '" + kind + "'";
      return false;
    }
  }
  return true;
}

bool RangeFacts::save(llvm::StringRef path, std::string *error) const {
  std::ofstream out(path.str());
  if (!out) {
    if (error) *error = "cannot write facts file '" + path.str() + "'";
    return false;
  }
  out.precision(17);
  for (size_t i = 0; i < weights_.size(); ++i)
    out << "weight " << weights_[i].slot << ' ' << weights_[i].layer << ' '
        << weight_ranges_[i].lo << ' ' << weight_ranges_[i].hi << ' '
        << weight_ranges_[i].err << '\n';
  for (size_t i = 0; i < tensor_names_.size(); ++i)
    out << "tensor " << tensor_names_[i] << ' ' << tensor_ranges_[i].lo << ' '
        << tensor_ranges_[i].hi << ' ' << tensor_ranges_[i].err << '\n';
  return true;
}

void RangeFacts::mergeFrom(const RangeFacts &other) {
  for (size_t i = 0; i < other.weights_.size(); ++i)
    setWeight(other.weights_[i].slot, other.weights_[i].layer,
              other.weight_ranges_[i]);
  for (size_t i = 0; i < other.tensor_names_.size(); ++i)
    setTensor(other.tensor_names_[i], other.tensor_ranges_[i]);
}

// --- IR helpers -------------------------------------------------------------

namespace {

llvm::ArrayRef<int64_t> typeShape(mlir::Type type) {
  if (auto t = mlir::dyn_cast<boom::TensorType>(type)) return t.getShape();
  if (auto t = mlir::dyn_cast<boom::Q8_1Type>(type)) return t.getShape();
  if (boom::isBoomWeightQuantType(type))
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
      return shaped.getShape();
  return {};
}

bool isFloatDType(llvm::StringRef dtype) {
  return dtype == "float32" || dtype == "float16" || dtype == "bfloat16";
}

// The number of elements a value covers.
int64_t numElements(mlir::Type type) {
  const llvm::ArrayRef<int64_t> shape = typeShape(type);
  int64_t n = 1;
  for (int64_t d : shape) {
    if (d <= 0 || mlir::ShapedType::isDynamic(d)) return -1;
    n *= d;
  }
  return n;
}

// Element count of the fastest-moving (reduction) dimension: the K of a dot
// product whose left operand is `type`, and the length of a row reduction.
int64_t lastDim(mlir::Type type) {
  const llvm::ArrayRef<int64_t> shape = typeShape(type);
  if (shape.empty()) return -1;
  const int64_t d = shape.back();
  return d > 0 && !mlir::ShapedType::isDynamic(d) ? d : -1;
}

int64_t firstDim(mlir::Type type) {
  const llvm::ArrayRef<int64_t> shape = typeShape(type);
  if (shape.empty()) return -1;
  const int64_t d = shape.front();
  return d > 0 && !mlir::ShapedType::isDynamic(d) ? d : -1;
}

bool intAttr(Operation *op, llvm::StringRef name, int64_t &out) {
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>(name)) {
    out = a.getInt();
    return true;
  }
  return false;
}

// An integer attribute the emitter writes as a typed attribute or a pass writes
// into the node_attrs record (BufferReusePass's inplace/buf are of the latter
// kind: they are facts about the node, not part of the operator's vocabulary).
bool nodeIntAttr(Operation *op, llvm::StringRef name, int64_t &out) {
  if (intAttr(op, name, out)) return true;
  if (auto attrs = op->getAttrOfType<mlir::ArrayAttr>("node_attrs"))
    for (mlir::Attribute a : attrs)
      if (auto na = mlir::dyn_cast<boom::NodeAttr>(a))
        if (na.getName() == name)
          if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(na.getValue())) {
            out = ia.getInt();
            return true;
          }
  return false;
}

bool floatAttr(Operation *op, llvm::StringRef name, double &out) {
  if (auto a = op->getAttrOfType<mlir::FloatAttr>(name)) {
    out = a.getValueAsDouble();
    return true;
  }
  return false;
}

// A *unit* is an op whose body is a node list carrying a tensor table:
// boom.graph (a decode or prefill segment) and boom.step (the same node list
// carried inside a boom.program's loop). Both spell the table as `tensors` and
// give every node `result_ids`, so one implementation serves both. The analysis
// therefore runs on the posts a plan is lowered from *and* on the step that
// program lifting derives from it.
bool isUnit(Operation *op) {
  return mlir::isa<boom::GraphOp>(op) || mlir::isa<boom::StepOp>(op);
}

// A node's result tensor ids. The record is a DenseI64ArrayAttr (the builder's
// getDenseI64ArrayAttr), not a generic ArrayAttr: reading it as the wrong type
// silently loses every result id.
llvm::ArrayRef<int64_t> resultIds(Operation *node) {
  if (auto a = node->getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids"))
    return a.asArrayRef();
  return {};
}

llvm::ArrayRef<mlir::Attribute> unitTensors(Operation *op) {
  if (auto a = op->getAttrOfType<mlir::ArrayAttr>("tensors"))
    return a.getValue();
  return {};
}

llvm::StringRef unitKind(Operation *op) {
  return mlir::isa<boom::GraphOp>(op) ? "graph" : "step";
}

llvm::StringRef unitName(Operation *op) {
  if (auto n = op->getAttrOfType<mlir::StringAttr>("graph_name"))
    return n.getValue();
  return "step";
}

llvm::ArrayRef<int64_t> unitOutputs(Operation *op) {
  if (auto g = mlir::dyn_cast<boom::GraphOp>(op)) return g.getOutputs();
  if (auto s = mlir::dyn_cast<boom::StepOp>(op)) return s.getOutputIds();
  return {};
}

mlir::Block &unitBody(Operation *op) { return op->getRegion(0).front(); }

// The name of the tensor a block argument stands for: a unit's block arguments
// are its free variables in ascending id order (the order the verifier
// recomputes), with the KV handles after them; the name itself is in the table.
llvm::StringRef blockArgName(mlir::BlockArgument arg) {
  Operation *unit = arg.getOwner()->getParentOp();
  if (!unit || !isUnit(unit)) return {};
  llvm::DenseSet<int64_t> producedSet;
  for (Operation &node : unitBody(unit))
    for (int64_t id : resultIds(&node)) producedSet.insert(id);
  llvm::SmallVector<int64_t, 32> freeIds;
  for (mlir::Attribute a : unitTensors(unit))
    if (auto d = mlir::dyn_cast<boom::TensorDescAttr>(a))
      if (!producedSet.count(d.getId())) freeIds.push_back(d.getId());
  llvm::sort(freeIds);
  const unsigned index = arg.getArgNumber();
  if (index >= freeIds.size()) return {};
  for (mlir::Attribute a : unitTensors(unit))
    if (auto d = mlir::dyn_cast<boom::TensorDescAttr>(a))
      if (d.getId() == freeIds[index]) return d.getName();
  return {};
}

// The tensor-table name of a node's first result: the key a calibration fact
// is stored under.
llvm::StringRef resultName(Operation *node) {
  const llvm::ArrayRef<int64_t> ids = resultIds(node);
  if (ids.empty()) return {};
  Operation *unit = node->getParentOp();
  if (!unit || !isUnit(unit)) return {};
  for (mlir::Attribute a : unitTensors(unit))
    if (auto d = mlir::dyn_cast<boom::TensorDescAttr>(a))
      if (d.getId() == ids[0]) return d.getName();
  return {};
}

Operation *definingNode(Value v) {
  if (auto op = v.getDefiningOp()) return op;
  return nullptr;
}

} // namespace

// --- transfer functions -----------------------------------------------------

namespace {

struct LatticeValue {
  RangeError value;
  bool hasReduction = false;
  ReductionReport reduction;
};

// The weight a node names with `slot` (+ the layer the node names, -1 when it
// names none). An emitter writes the weight as an attribute; a graph built from
// a planner::Graph carries it as a typed operand instead, and the slot still
// identifies it, so the same lookup serves both.
std::optional<RangeError> nodeWeight(Operation *op, const RangeFacts &facts) {
  int64_t slot = 0, layer = -1;
  if (!intAttr(op, "slot", slot)) return std::nullopt;
  intAttr(op, "layer", layer);
  return facts.weight(slot, layer);
}

// Interval arithmetic for one f32 addition.
RangeError addRange(const RangeError &a, const RangeError &b) {
  if (!a.known || !b.known) return RangeError::unknown();
  const double lo = a.lo + b.lo, hi = a.hi + b.hi;
  const double mag = std::max(std::abs(lo), std::abs(hi));
  return RangeError::bounded(lo, hi, a.err + b.err + kF32Eps * mag);
}

RangeError mulRange(const RangeError &a, const RangeError &b) {
  if (!a.known || !b.known) return RangeError::unknown();
  const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
  const double lo = *std::min_element(p, p + 4);
  const double hi = *std::max_element(p, p + 4);
  const double mag = std::max(std::abs(lo), std::abs(hi));
  const double err = a.magnitude() * b.err + b.magnitude() * a.err +
                     a.err * b.err + kF32Eps * mag;
  return RangeError::bounded(lo, hi, err);
}

// A unary monotone-increasing map applied by evaluating the endpoints, with a
// multiplicative Lipschitz factor on the input error.
RangeError monotoneRange(const RangeError &x, double (*f)(double), double lipschitz) {
  if (!x.known) return RangeError::unknown();
  const double lo = f(x.lo), hi = f(x.hi);
  return RangeError::bounded(lo, hi, lipschitz * x.err + kF32Eps * x.magnitude());
}

double siluFn(double x) { return x / (1.0 + std::exp(-x)); }
double geluFn(double x) {
  return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// A non-monotone unary map: the endpoints plus the map's interior extrema that
// fall inside the interval. `extrema` are the critical points; evaluating them
// unconditionally and taking the hull is sound (an out-of-interval value only
// widens the hull), so no membership test is needed.
RangeError sampledRange(const RangeError &x, double (*f)(double),
                        llvm::ArrayRef<double> extrema, double lipschitz) {
  if (!x.known) return RangeError::unknown();
  double lo = std::min(f(x.lo), f(x.hi));
  double hi = std::max(f(x.lo), f(x.hi));
  for (double p : extrema) {
    if (p < x.lo || p > x.hi) continue;  // outside the interval: not reached
    lo = std::min(lo, f(p));
    hi = std::max(hi, f(p));
  }
  return RangeError::bounded(lo, hi, lipschitz * x.err + kF32Eps * x.magnitude());
}

// The value one element of a q8_1 (or other block-quantized activation) can be
// off by: half a quantization step, with the block scale itself an fp16.
double quantizeSinkError(double magnitude, int bits) {
  if (!std::isfinite(magnitude)) return std::numeric_limits<double>::infinity();
  const double step = magnitude / double((1 << (bits - 1)) - 1);
  return 0.5 * step + kF16Eps * magnitude;
}

// A reduction row for an op whose f32 accumulation sums `k` products of at most
// `ma` and `mb` in magnitude.
ReductionReport reductionRow(Operation *op, llvm::StringRef opName, int64_t k,
                             double ma, double mb, double epsilon) {
  ReductionReport r;
  r.node = op->getAttrOfType<mlir::StringAttr>("node_name")
               ? op->getAttrOfType<mlir::StringAttr>("node_name").getValue().str()
               : std::string(opName);
  r.op = opName.str();
  r.k = k;
  if (k <= 0 || !std::isfinite(ma) || !std::isfinite(mb)) {
    r.sum_abs = std::numeric_limits<double>::infinity();
    r.err_seq = r.err_tree = std::numeric_limits<double>::infinity();
    return r;
  }
  r.sum_abs = double(k) * ma * mb;
  r.err_seq = double(k) * kF32Eps * r.sum_abs;
  r.err_tree = treeDepth(k) * kF32Eps * r.sum_abs;
  r.within_seq = r.err_seq <= epsilon * r.sum_abs;
  r.within_tree = r.err_tree <= epsilon * r.sum_abs;
  return r;
}

RangeError transfer(Operation *op, llvm::ArrayRef<const RangeError *> in,
                    const RangeFacts &facts, double epsilon,
                    ReductionReport *red, bool *hasRed) {
  const llvm::StringRef name = op->getName().getStringRef();
  auto inRange = [&](size_t i) {
    return i < in.size() && in[i] ? *in[i] : RangeError::unknown();
  };
  const mlir::Type resType =
      op->getNumResults() != 0 ? op->getResult(0).getType() : mlir::Type();

  // A node whose result dtype is not floating point carries no range the
  // floating-point analysis can use (argmax, index vectors).
  if (resType) {
    if (auto t = mlir::dyn_cast<boom::TensorType>(resType))
      if (!isFloatDType(t.getDtype())) return RangeError::unknown();
  }

  auto reduction = [&](int64_t k, double ma, double mb) {
    if (!hasRed) return;
    *hasRed = true;
    *red = reductionRow(op, name, k, ma, mb, epsilon);
  };

  // A matmul-family node: an activation of `n` elements dotted against a
  // weight row of `k` elements, or a two-operand dense product. The weight is
  // the node's `slot` when the IR carries one (the emitter's form), otherwise
  // the last operand with a quantized type (the graph round trip's form).
  auto matmul = [&]() {
    if (in.empty()) return RangeError::unknown();
    const RangeError a = inRange(0);
    RangeError w = RangeError::unknown();
    if (auto fw = nodeWeight(op, facts)) w = *fw;
    // Gemma 4's per-layer projection (ple_proj) names its weight by op kind,
    // not by a `slot` attribute: it reads the model-wide PerLayerModelProj
    // table, the weight the facts record at layer -1.
    constexpr int64_t kPerLayerModelProj = 3;  // WeightSlot::PerLayerModelProj
    if (!w.known && name == "boom.matvec_bf16")
      if (auto fw = facts.weight(kPerLayerModelProj, -1)) w = *fw;
    for (size_t i = 1; i < in.size(); ++i)
      if (in[i] && boom::isBoomWeightQuantType(op->getOperand(i).getType()))
        w = RangeError::join(w, *in[i]);
    if (name == "boom.matmul" || name == "boom.bmm" || name == "boom.gemm") {
      const RangeError b = inRange(1);
      const int64_t k = lastDim(op->getOperand(0).getType());
      reduction(k, a.magnitude(), b.magnitude());
      if (!a.known || !b.known) return RangeError::unknown();
      const double m = a.magnitude() * b.magnitude();
      const double bound = double(std::max<int64_t>(k, 1)) * m;
      const double err = double(std::max<int64_t>(k, 1)) *
                             (a.err * b.magnitude() + b.err * a.magnitude()) +
                         (hasRed ? red->err_seq : 0.0);
      return RangeError::bounded(-bound, bound, err);
    }
    int64_t k = lastDim(op->getOperand(0).getType());
    // The graph round trip's quantized weight is an operand shaped [K, N]; the
    // emitter's slot names a weight whose first (quantized) dimension is K.
    if (k <= 0) k = firstDim(op->getOperand(in.size() - 1).getType());
    reduction(k, a.magnitude(), w.magnitude());
    if (!a.known || !w.known) return RangeError::unknown();
    const double bound = double(std::max<int64_t>(k, 1)) * a.magnitude() * w.magnitude();
    const double err = double(std::max<int64_t>(k, 1)) *
                           (a.err * w.magnitude() + w.err * a.magnitude()) +
                       (hasRed ? red->err_seq : 0.0);
    return RangeError::bounded(-bound, bound, err);
  };

  // A full-row RMS norm: n elements reduced into the mean of squares.
  auto rmsNorm = [&](bool quantized) {
    const RangeError x = inRange(0);
    const int64_t n = lastDim(op->getOperand(0).getType());
    RangeError w = RangeError::bounded(1.0, 1.0, 0.0);  // unit weight
    if (auto fw = nodeWeight(op, facts)) w = *fw;
    // The accumulation this row reports is the sum of squares, so the "terms"
    // are |x|^2 and the sum of absolute terms is n * max|x|^2.
    reduction(n, 1.0, x.magnitude() * x.magnitude());
    if (!x.known || !w.known) return RangeError::unknown();
    // |x_j| / rms(x) <= sqrt(n) elementwise (Cauchy-Schwarz on the vector), so
    // the normalized value is bounded by sqrt(n) and the output by
    // max|w| * sqrt(n) whatever the input scale.
    const double bound = w.magnitude() * std::sqrt(double(std::max<int64_t>(n, 1)));
    double err = w.magnitude() * 2.0 * x.err +
                 x.magnitude() * w.err + kF32Eps * bound;
    if (quantized) err += quantizeSinkError(bound, 8);
    return RangeError::bounded(-bound, bound, err);
  };

  if (name == "boom.add") return addRange(inRange(0), inRange(1));
  if (name == "boom.mul") return mulRange(inRange(0), inRange(1));
  if (name == "boom.scale") {
    double c = 1.0;
    floatAttr(op, "scale", c);
    const RangeError x = inRange(0);
    if (!x.known) return RangeError::unknown();
    const double lo = c >= 0 ? c * x.lo : c * x.hi;
    const double hi = c >= 0 ? c * x.hi : c * x.lo;
    return RangeError::bounded(lo, hi, std::abs(c) * x.err +
                                            kF32Eps * std::abs(c) * x.magnitude());
  }
  if (name == "boom.silu")
    return sampledRange(inRange(0), siluFn, {0.0, -1.2784645427610738}, 1.1);
  if (name == "boom.gelu_fp16") {
    RangeError r = sampledRange(inRange(0), geluFn, {-0.7517915238}, 1.13);
    if (r.known) r.err += kF16Eps * r.magnitude();
    return r;
  }
  if (name == "boom.silu_mul" || name == "boom.gelu_mul" ||
      name == "boom.add_bias") {
    if (name == "boom.add_bias") {
      RangeError bias = RangeError::bounded(0.0, 0.0, 0.0);
      if (auto fb = nodeWeight(op, facts)) bias = *fb;
      return addRange(inRange(0), bias);
    }
    return mulRange(inRange(0), inRange(1));
  }
  if (name == "boom.softcap") {
    double c = 1.0;
    floatAttr(op, "scale", c);
    const RangeError x = inRange(0);
    if (!x.known) return RangeError::unknown();
    return RangeError::bounded(-std::abs(c), std::abs(c),
                               x.err + kF32Eps * std::abs(c));
  }
  if (name == "boom.rms_norm") return rmsNorm(false);
  if (name == "boom.rms_norm_heads") return rmsNorm(false);
  if (name == "boom.rms_norm_quantize") return rmsNorm(true);
  if (name == "boom.gelu_mul_quantize" || name == "boom.silu_mul_quantize") {
    RangeError r;
    if (name == "boom.gelu_mul_quantize") {
      RangeError g = sampledRange(inRange(0), geluFn, {-0.7517915238}, 1.13);
      if (g.known) g.err += kF16Eps * g.magnitude();
      r = mulRange(g, inRange(1));
    } else {
      r = mulRange(sampledRange(inRange(0), siluFn,
                                {0.0, -1.2784645427610738}, 1.1),
                   inRange(1));
    }
    if (r.known) r.err += quantizeSinkError(r.magnitude(), 8);
    // The quantize's block max is a 32-element reduction.
    reduction(32, r.magnitude(), 1.0);
    return r;
  }
  if (name == "boom.quantize") {
    const RangeError x = inRange(0);
    if (!x.known) return RangeError::unknown();
    // The block max over 32 elements is the reduction; the value is the input
    // rounded to an 8-bit block scale.
    reduction(32, x.magnitude(), 1.0);
    return RangeError::bounded(-x.magnitude(), x.magnitude(),
                               x.err + quantizeSinkError(x.magnitude(), 8));
  }
  // A row lookup: the result is one dequantized row of the table, so the
  // table's own range (and its quantization error) is the result's.
  if (name == "boom.embedding") return nodeWeight(op, facts).value_or(RangeError::unknown());
  if (name == "boom.quantized_matmul" || name == "boom.quantized_matmul_q8" ||
      name == "boom.matmul_q8_tc" || name == "boom.matvec_f32" ||
      name == "boom.matvec_bf16")
    return matmul();
  if (name == "boom.cast_fp16") {
    const RangeError x = inRange(0);
    if (!x.known) return RangeError::unknown();
    return RangeError::bounded(x.lo, x.hi, x.err + kF16Eps * x.magnitude());
  }
  if (name == "boom.rope") {
    const RangeError x = inRange(0);
    if (!x.known) return RangeError::unknown();
    const double bound = std::sqrt(2.0) * x.magnitude();
    return RangeError::bounded(-bound, bound, x.err + kF32Eps * bound);
  }
  if (name == "boom.attention" || name == "boom.flash_attention") {
    // The output is a convex combination of V rows, so it cannot exceed V's
    // magnitude; V is an operand when the graph passes it, and the cache
    // (unbounded at plan time) otherwise.
    const RangeError v = in.size() > 2 ? inRange(2) : RangeError::unknown();
    if (!v.known) return RangeError::unknown();
    return RangeError::bounded(-v.magnitude(), v.magnitude(),
                               v.err + kF32Eps * v.magnitude());
  }
  if (name == "boom.softmax") {
    const RangeError x = inRange(0);
    if (!x.known) return RangeError::unknown();
    return RangeError::bounded(0.0, 1.0, 0.5 * x.err + kF32Eps);
  }
  if (name == "boom.slice" || name == "boom.reshape" || name == "boom.permute" ||
      name == "boom.gqa_broadcast" || name == "boom.whole_graph")
    return inRange(0);
  if (name == "boom.reduce_mean" || name == "boom.reduce_sum" ||
      name == "boom.cross_entropy") {
    const RangeError x = inRange(0);
    const int64_t k = lastDim(op->getOperand(0).getType());
    reduction(k, x.magnitude(), 1.0);
    if (!x.known) return RangeError::unknown();
    const double bound = name == "boom.reduce_mean"
                             ? x.magnitude()
                             : x.magnitude() * double(std::max<int64_t>(k, 1));
    return RangeError::bounded(-bound, bound,
                               x.err * double(std::max<int64_t>(k, 1)) +
                                   (*hasRed ? red->err_seq : 0.0));
  }
  if (name == "boom.argmax") return RangeError::bounded(0.0, 1.0, 0.0);
  if (name == "boom.kv_append" || name == "boom.async_weight_prefetch" ||
      name == "boom.sync_prefetch")
    return RangeError::unknown();
  return RangeError::unknown();
}

} // namespace

// --- the forward analysis ---------------------------------------------------

namespace {

class RangeLattice : public mlir::dataflow::Lattice<RangeError> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RangeLattice)
  using Lattice::Lattice;
};

class RangeAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<RangeLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RangeAnalysis)

  RangeAnalysis(mlir::DataFlowSolver &solver, const RangeFacts &facts,
                double epsilon)
      : mlir::dataflow::SparseForwardDataFlowAnalysis<RangeLattice>(solver),
        facts_(facts),
        epsilon_(epsilon) {}

  void setToEntryState(RangeLattice *lattice) override {
    const Value anchor = lattice->getAnchor();
    RangeError r = RangeError::unknown();
    if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(anchor)) {
      // A calibration fact names the tensor; a weight the IR names by slot
      // reaches its consumer's transfer function instead.
      const llvm::StringRef name = blockArgName(arg);
      if (!name.empty())
        if (auto f = facts_.tensor(name)) r = *f;
    }
    propagateIfChanged(lattice, lattice->join(r));
  }

  mlir::LogicalResult visitOperation(
      Operation *op, llvm::ArrayRef<const RangeLattice *> operands,
      llvm::ArrayRef<RangeLattice *> results) override {
    // A node with no results is a side effect (kv_append, prefetch).
    if (results.empty()) return mlir::success();

    llvm::SmallVector<const RangeError *, 6> in;
    in.reserve(operands.size());
    for (const RangeLattice *l : operands)
      in.push_back(l ? &l->getValue() : nullptr);
    ReductionReport red;
    bool hasRed = false;
    RangeError r = transfer(op, in, facts_, epsilon_, &red, &hasRed);
    // A calibration fact narrows the result to what was measured on a real
    // run. The node's own reduction row still comes from its input bounds.
    const llvm::StringRef name = resultName(op);
    if (!name.empty())
      if (auto f = facts_.tensor(name)) {
        f->known = true;
        r = *f;
      }
    last_[op] = LatticeValue{r, hasRed, red};
    for (RangeLattice *l : results)
      propagateIfChanged(l, l->join(r));
    return mlir::success();
  }

  const llvm::DenseMap<Operation *, LatticeValue> &last() const { return last_; }

private:
  const RangeFacts &facts_;
  double epsilon_;
  llvm::DenseMap<Operation *, LatticeValue> last_;
};

} // namespace

// --- dense backward liveness ------------------------------------------------

namespace {

// The values live after each node, over the graph's flat, ordered node list:
// a value is live after node i iff some later node reads it, or it is a graph
// output.
struct LiveAfter {
  llvm::DenseMap<Operation *, llvm::DenseSet<Value>> after;
  llvm::DenseSet<Value> liveAtEnd;
};

LiveAfter computeLiveAfter(Operation *unit) {
  LiveAfter out;
  llvm::DenseSet<Value> live;
  llvm::DenseSet<int64_t> outputs;
  for (int64_t id : unitOutputs(unit)) outputs.insert(id);
  for (Operation &node : unitBody(unit)) {
    const auto ids = resultIds(&node);
    for (size_t index = 0; index < ids.size() && index < node.getNumResults();
         ++index)
      if (outputs.count(ids[index])) live.insert(node.getResult(index));
  }
  out.liveAtEnd = live;
  llvm::SmallVector<Operation *> nodes;
  for (Operation &node : unitBody(unit)) nodes.push_back(&node);
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    Operation *node = *it;
    // A result is defined here, so it is not live before the node; live after
    // it is what the backward pass has accumulated.
    out.after[node] = live;
    for (Value operand : node->getOperands()) live.insert(operand);
  }
  return out;
}

// The tensor id a node result carries, for graph-output membership.
bool resultIsOutput(Operation *unit, Operation *node, unsigned index) {
  const auto ids = resultIds(node);
  if (index >= ids.size()) return false;
  for (int64_t id : unitOutputs(unit))
    if (id == ids[index]) return true;
  return false;
}

bool operandIsGraphOutput(Operation *unit, Operation *node, unsigned index) {
  if (index >= node->getNumOperands()) return false;
  Value v = node->getOperand(index);
  if (Operation *def = v.getDefiningOp())
    return resultIsOutput(
        unit, def, unsigned(mlir::cast<mlir::OpResult>(v).getResultNumber()));
  return false;
}

} // namespace

// --- the entry point --------------------------------------------------------

AnalysisResult analyzeBoom(mlir::ModuleOp module, const RangeFacts &facts,
                           double epsilon) {
  AnalysisResult result;
  result.epsilon = epsilon;

  llvm::SmallVector<Operation *, 4> units;
  module.walk([&](Operation *op) {
    if (isUnit(op)) units.push_back(op);
  });
  if (units.empty()) {
    result.notes.push_back("no boom.graph or boom.step in the module");
    return result;
  }

  for (Operation *unit : units) {
    // --- forward value ranges ---
    mlir::DataFlowSolver solver;
    solver.load<mlir::dataflow::DeadCodeAnalysis>();
    auto *analysis = solver.load<RangeAnalysis>(facts, epsilon);
    if (mlir::failed(solver.initializeAndRun(unit))) {
      result.notes.push_back("the data-flow solver failed on " +
                             unitKind(unit).str() + " '" + unitName(unit).str() +
                             "'");
      continue;
    }
    const auto &last = analysis->last();
    const std::string unitTag = unitKind(unit).str() + ":" + unitName(unit).str();

    // Walk the node list in order: a stable, deterministic report.
    size_t nodeIndex = 0;
    for (Operation &node : unitBody(unit)) {
      ++nodeIndex;
      if (node.getNumResults() == 0) continue;
      ++result.values_analyzed;
      auto *lattice = solver.lookupState<RangeLattice>(node.getResult(0));
      const RangeError r = lattice ? lattice->getValue() : RangeError::unknown();
      result.values.push_back(
          {node.getAttrOfType<mlir::StringAttr>("node_name")
               ? node.getAttrOfType<mlir::StringAttr>("node_name").getValue().str()
               : std::string(node.getName().getStringRef()),
           std::string(node.getName().getStringRef()), r, unitTag});
      if (!r.known) {
        ++result.values_unknown;
        if (nodeIndex <= 4)
          result.notes.push_back("unbounded: " +
                                 std::string(node.getName().getStringRef()));
      }
      auto it = last.find(&node);
      if (it != last.end() && it->second.hasReduction) {
        ReductionReport rr = it->second.reduction;
        rr.unit = unitTag;
        result.reductions.push_back(std::move(rr));
      }
    }

    // --- dense backward liveness vs BufferReusePass marks ---
    size_t marks = 0;
    const LiveAfter liveAfter = computeLiveAfter(unit);
    mlir::dataflow::RunLivenessAnalysis valueLiveness(unit);

    for (Operation &node : unitBody(unit)) {
      // The independent value-level check: a value the machine says is dead
      // must not be live after any node. A disagreement means the per-position
      // sweep and MLIR's liveness disagree, which is a fact about this analysis
      // worth reporting rather than a pass defect.
      for (Value operand : node.getOperands()) {
        if (operand.getDefiningOp() == nullptr) continue;
        const mlir::dataflow::Liveness *l = valueLiveness.getLiveness(operand);
        if (l && !l->isLive) {
          auto after = liveAfter.after.find(&node);
          if (after != liveAfter.after.end() && after->second.count(operand))
            result.liveness.push_back(
                {std::string(node.getName().getStringRef()), "analysis",
                 "MLIR's liveness calls a value dead that the per-position "
                 "sweep has live after this node",
                 unitTag});
        }
      }

      int64_t inplace = 0;
      if (!nodeIntAttr(&node, "inplace", inplace) || inplace == 0) continue;
      ++marks;
      const std::string nodeName =
          node.getAttrOfType<mlir::StringAttr>("node_name")
              ? node.getAttrOfType<mlir::StringAttr>("node_name").getValue().str()
              : std::string(node.getName().getStringRef());
      if (node.getNumOperands() == 0) {
        result.liveness.push_back({nodeName, "unsound",
                                   "inplace=1 on a node with no operand 0",
                                   unitTag});
        continue;
      }
      Value source = node.getOperand(0);
      if (source.getDefiningOp() == nullptr) {
        result.liveness.push_back(
            {nodeName, "unsound", "inplace=1 borrows a block argument's buffer",
             unitTag});
        continue;
      }
      if (operandIsGraphOutput(unit, &node, 0)) {
        result.liveness.push_back(
            {nodeName, "unsound",
             "inplace=1 overwrites operand 0, which is a graph output", unitTag});
        continue;
      }
      auto after = liveAfter.after.find(&node);
      if (after != liveAfter.after.end() && after->second.count(source)) {
        result.liveness.push_back(
            {nodeName, "unsound",
             "operand 0 is live after this node (a later node reads it)",
             unitTag});
        continue;
      }
      // Sound but stricter than liveness: the pass also requires a single use
      // anywhere in the graph. The value may have earlier uses and still be
      // dead here, which is an opportunity the pass declines on purpose.
      if (!source.hasOneUse())
        result.liveness.push_back(
            {nodeName, "strict",
             "operand 0 has earlier uses; dense liveness allows the reuse, the "
             "pass requires one use",
             unitTag});
    }
    result.notes.push_back("inplace marks checked: " + std::to_string(marks) +
                           " (" + unitTag + ")");
  }

  result.ok = true;
  return result;
}

} // namespace sonicboom::compiler
