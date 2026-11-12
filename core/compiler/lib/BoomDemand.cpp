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

// The backward demand analysis over Boom IR (BoomDemand.h).

#include "sonicboom/compiler/BoomDemand.h"

#include "BoomUnit.h"
#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>

namespace sonicboom::compiler {

//--- IndexSet ----------------------------------------------------------------

IndexSet IndexSet::all() {
  IndexSet s;
  s.kind = Kind::All;
  return s;
}

IndexSet IndexSet::interval(int64_t lo, int64_t hi) {
  IndexSet s;
  if (hi <= lo)
    return s;
  s.kind = Kind::Interval;
  s.lo = lo;
  s.hi = hi;
  return s;
}

IndexSet IndexSet::list(int64_t capacity) {
  IndexSet s;
  if (capacity <= 0)
    return s;
  s.kind = Kind::List;
  s.capacity = capacity;
  return s;
}

IndexSet IndexSet::join(const IndexSet &a, const IndexSet &b) {
  if (a.isEmpty())
    return b;
  if (b.isEmpty())
    return a;
  if (a.isAll() || b.isAll())
    return all();
  if (a.kind == Kind::Interval && b.kind == Kind::Interval)
    return interval(std::min(a.lo, b.lo), std::max(a.hi, b.hi));
  if (a.kind == Kind::List && b.kind == Kind::List && a.capacity == b.capacity)
    return a;
  // Two different lists, or a list and an interval: no set of either form is
  // known to contain both.
  return all();
}

bool IndexSet::operator==(const IndexSet &o) const {
  if (kind != o.kind)
    return false;
  switch (kind) {
  case Kind::Empty:
  case Kind::All:
    return true;
  case Kind::Interval:
    return lo == o.lo && hi == o.hi;
  case Kind::List:
    return capacity == o.capacity;
  }
  return false;
}

void IndexSet::print(llvm::raw_ostream &os) const {
  switch (kind) {
  case Kind::Empty:
    os << "empty";
    return;
  case Kind::All:
    os << "all";
    return;
  case Kind::Interval:
    os << "[" << lo << ", " << hi << ")";
    return;
  case Kind::List:
    os << "list<=" << capacity;
    return;
  }
}

//--- Demand ------------------------------------------------------------------

Demand Demand::none(unsigned rank) {
  Demand d;
  d.dims.assign(rank, IndexSet::empty());
  return d;
}

Demand Demand::whole(unsigned rank, Observation obs) {
  Demand d;
  d.obs = obs;
  d.dims.assign(rank, obs == Observation::None ? IndexSet::empty()
                                               : IndexSet::all());
  return d;
}

namespace {

// Dimension `i` of `d`, for a rank `d` was not built with: a value nothing
// reads has nothing read there either, and an observed one is read whole.
IndexSet dimAt(const Demand &d, unsigned i) {
  if (i < d.dims.size())
    return d.dims[i];
  return d.isNone() ? IndexSet::empty() : IndexSet::all();
}

} // namespace

Demand Demand::join(const Demand &a, const Demand &b) {
  Demand out;
  out.obs = std::max(a.obs, b.obs);
  const unsigned rank = unsigned(std::max(a.dims.size(), b.dims.size()));
  out.dims.reserve(rank);
  for (unsigned i = 0; i < rank; ++i)
    out.dims.push_back(IndexSet::join(dimAt(a, i), dimAt(b, i)));
  return out;
}

bool Demand::isWholeRegion() const {
  return !isNone() &&
         llvm::all_of(dims, [](const IndexSet &s) { return s.isAll(); });
}

bool Demand::operator==(const Demand &o) const {
  return obs == o.obs && dims == o.dims;
}

void Demand::print(llvm::raw_ostream &os) const {
  switch (obs) {
  case Observation::None:
    os << "none";
    break;
  case Observation::Order:
    os << "order";
    break;
  case Observation::Exact:
    os << "exact";
    break;
  }
  os << "<";
  llvm::interleaveComma(dims, os, [&](const IndexSet &s) { s.print(os); });
  os << ">";
}

//--- the analysis ------------------------------------------------------------

namespace {

class DemandLattice : public mlir::dataflow::Lattice<Demand> {
public:
  using Lattice::Lattice;
};

// The rank a value's demand is stated over: a Boom tensor's (an f32 activation
// or its q8_1 layout), and none for everything else.
unsigned rankOf(mlir::Type type) {
  if (auto t = llvm::dyn_cast<boom::TensorType>(type))
    return unsigned(t.getShape().size());
  if (auto t = llvm::dyn_cast<boom::Q8_1Type>(type))
    return unsigned(t.getShape().size());
  return 0;
}

unsigned rankOf(mlir::Value v) { return rankOf(v.getType()); }

// The demand a graph's declared consumer places on its output.
Demand consumerDemand(boom::GraphOp graph, unsigned rank) {
  const std::optional<llvm::StringRef> consumer = graph.getConsumer();
  const Observation obs = consumer && *consumer == "argmax" ? Observation::Order
                                                            : Observation::Exact;
  Demand d = Demand::whole(rank, obs);
  if (auto list = graph.getConsumerList())
    if (rank > 0)
      d.dims.back() = IndexSet::list(int64_t(*list));
  return d;
}

class DemandAnalysis
    : public mlir::dataflow::SparseBackwardDataFlowAnalysis<DemandLattice> {
public:
  DemandAnalysis(mlir::DataFlowSolver &solver,
                 mlir::SymbolTableCollection &symbols)
      : SparseBackwardDataFlowAnalysis(solver, symbols) {}

  mlir::LogicalResult
  visitOperation(mlir::Operation *op, llvm::ArrayRef<DemandLattice *> operands,
                 llvm::ArrayRef<const DemandLattice *> results) override {
    if (auto graph = llvm::dyn_cast<boom::GraphOp>(op)) {
      visitGraph(graph);
      return mlir::success();
    }
    if (auto step = llvm::dyn_cast<boom::StepOp>(op)) {
      visitStep(step, operands, results);
      return mlir::success();
    }
    if (auto sample = llvm::dyn_cast<boom::SampleOp>(op)) {
      visitSample(sample, operands, results);
      return mlir::success();
    }
    if (unit::isNode(op)) {
      visitNode(op, operands, results);
      return mlir::success();
    }
    visitOpaque(op, operands, results);
    return mlir::success();
  }

  // A branch condition, a call argument and a value that leaves the analyzed
  // scope are all read for what they are.
  void visitBranchOperand(mlir::OpOperand &operand) override {
    demandWhole(getLatticeElement(operand.get()));
  }
  void visitCallOperand(mlir::OpOperand &operand) override {
    demandWhole(getLatticeElement(operand.get()));
  }
  void visitNonControlFlowArguments(
      mlir::RegionSuccessor &,
      llvm::ArrayRef<mlir::BlockArgument>) override {}
  void setToExitState(DemandLattice *lattice) override { demandWhole(lattice); }

private:
  void demand(DemandLattice *lattice, const Demand &d) {
    propagateIfChanged(lattice, lattice->meet(d));
  }
  void demandWhole(DemandLattice *lattice) {
    demand(lattice, Demand::whole(rankOf(lattice->getAnchor())));
  }

  // The graph's output is observed by its consumer. Nothing else in a
  // standalone graph is an observer except its effectful nodes, which
  // visitNode handles where they stand.
  void visitGraph(boom::GraphOp graph) {
    mlir::Block &body = graph.getBody().front();
    for (int64_t id : graph.getOutputs()) {
      mlir::Value out = unit::valueOf(body, graph.getTensors(), id);
      if (!out)
        continue;
      demand(getLatticeElement(out), consumerDemand(graph, rankOf(out)));
    }
  }

  // A step is transparent: its result is its body's output, and its body's
  // free variables are its operands, in order.
  void visitStep(boom::StepOp step, llvm::ArrayRef<DemandLattice *> operands,
                 llvm::ArrayRef<const DemandLattice *> results) {
    mlir::Block &body = step.getBody().front();
    for (auto [index, id] : llvm::enumerate(step.getOutputIds())) {
      if (index >= results.size())
        break;
      mlir::Value out = unit::valueOf(body, step.getTensors(), id);
      if (out)
        demand(getLatticeElement(out), results[index]->getValue());
    }
    mlir::ProgramPoint *after = getProgramPointAfter(step.getOperation());
    const unsigned n = std::min<unsigned>(unsigned(operands.size()),
                                          body.getNumArguments());
    for (unsigned i = 0; i < n; ++i) {
      // Read the argument's state as a dependency of this step, so the step is
      // revisited when a body node raises it.
      const DemandLattice *arg =
          getOrCreateFor<DemandLattice>(after, mlir::Value(body.getArgument(i)));
      demand(operands[i], arg->getValue());
    }
  }

  // The sample ranks the logits when it takes their mode and reads them when it
  // draws from them; either way only over its allowed list, if it has one.
  void visitSample(boom::SampleOp sample,
                   llvm::ArrayRef<DemandLattice *> operands,
                   llvm::ArrayRef<const DemandLattice *> results) {
    if (results.empty() || results[0]->getValue().isNone())
      return;
    const bool draws = sample.getStochastic() || sample.getTemperature() ||
                       sample.getTopK() || sample.getTopP();
    const unsigned rank = rankOf(sample.getLogits());
    Demand d =
        Demand::whole(rank, draws ? Observation::Exact : Observation::Order);
    if (auto list = sample.getAllowedList())
      if (rank > 0)
        d.dims.back() = IndexSet::list(int64_t(*list));
    demand(operands[0], d);
  }

  // An op the analysis has no statement about: if it may have an effect it is
  // an observer in its own right, and otherwise it reads its operands whole as
  // soon as anything reads its results.
  void visitOpaque(mlir::Operation *op,
                   llvm::ArrayRef<DemandLattice *> operands,
                   llvm::ArrayRef<const DemandLattice *> results) {
    const bool observed =
        llvm::any_of(results, [](const DemandLattice *r) {
          return !r->getValue().isNone();
        });
    if (!observed && mlir::isMemoryEffectFree(op))
      return;
    for (DemandLattice *operand : operands)
      demandWhole(operand);
  }

  void visitNode(mlir::Operation *node,
                 llvm::ArrayRef<DemandLattice *> operands,
                 llvm::ArrayRef<const DemandLattice *> results) {
    // A node that does not declare itself effect-free (a KV append, a prefetch
    // marker, an optimizer step) is kept whatever reads its results.
    if (!mlir::isMemoryEffectFree(node)) {
      for (DemandLattice *operand : operands)
        demandWhole(operand);
      return;
    }

    Demand out;
    for (const DemandLattice *r : results)
      out = Demand::join(out, r->getValue());
    if (out.isNone())
      return;

    // The shape-carrying rules below are stated for a node with one result,
    // which is every effect-free node the emitters and passes produce.
    const bool single = node->getNumResults() == 1;
    const unsigned outRank = single ? rankOf(node->getResult(0)) : 0;

    if (single && node->hasTrait<boom::OpTrait::Elementwise>()) {
      // A ranking of the results is a ranking of the operand only through a
      // non-decreasing map; any other elementwise map needs the elements.
      Demand in = out;
      if (in.obs == Observation::Order &&
          !node->hasTrait<boom::OpTrait::Monotone>())
        in.obs = Observation::Exact;
      for (auto [index, operand] : llvm::enumerate(operands)) {
        // A broadcast operand (another rank) is read whole.
        if (rankOf(node->getOperand(unsigned(index))) == outRank)
          demand(operand, in);
        else
          demandWhole(operand);
      }
      return;
    }

    if (single && isWeightMatmul(node)) {
      // y[.., j] = W[j, :] . x[.., :]: every output index reads its whole
      // activation row, and only its own rows of the activation batch.
      for (auto [index, operand] : llvm::enumerate(operands)) {
        const unsigned rank = rankOf(node->getOperand(unsigned(index)));
        Demand in = Demand::whole(rank);
        if (rank == outRank)
          for (unsigned d = 0; d + 1 < rank; ++d)
            in.dims[d] = dimAt(out, d);
        demand(operand, in);
      }
      return;
    }

    if (single && node->hasTrait<boom::OpTrait::RowIndependent>() &&
        outRank >= 2) {
      // Row r of the result reads row r of each tensor operand.
      for (auto [index, operand] : llvm::enumerate(operands)) {
        const unsigned rank = rankOf(node->getOperand(unsigned(index)));
        Demand in = Demand::whole(rank);
        if (rank >= 1)
          in.dims[0] = dimAt(out, 0);
        demand(operand, in);
      }
      return;
    }

    const llvm::StringRef name = node->getName().getStringRef();
    if (single && name == "boom.argmax") {
      // The index of the largest element: a ranking of the operand, all of it.
      for (auto [index, operand] : llvm::enumerate(operands))
        demand(operand,
               Demand::whole(rankOf(node->getOperand(unsigned(index))),
                             Observation::Order));
      return;
    }

    for (DemandLattice *operand : operands)
      demandWhole(operand);
  }
};

} // namespace

bool isWeightMatmul(mlir::Operation *node) {
  return node->getName().getStringRef() == "boom.quantized_matmul";
}

Demand DemandFacts::of(mlir::Value v) const {
  auto it = values.find(v);
  if (it == values.end())
    return Demand::none(rankOf(v));
  return it->second;
}

IndexSet DemandFacts::weightRows(mlir::Operation *node) const {
  auto it = weight_rows.find(node);
  return it == weight_rows.end() ? IndexSet::empty() : it->second;
}

DemandFacts analyzeDemand(mlir::Operation *top) {
  DemandFacts facts;
  if (!top)
    return facts;

  // Reachability and demand are solved together: a block the dead-code
  // analysis does not prove live contributes no observer.
  mlir::DataFlowSolver solver;
  mlir::SymbolTableCollection symbols;
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<mlir::dataflow::SparseConstantPropagation>();
  solver.load<DemandAnalysis>(symbols);
  // The solver visits an op only inside a block it has proved live, and it
  // proves the blocks under the op it is started on. Started on `top` itself,
  // a boom.graph nested in a module would sit in a block nobody proved, and
  // its own visit — the one that states the consumer's demand — would be
  // skipped. So the solve covers the enclosing root, and the facts are read
  // back for `top`'s values only.
  mlir::Operation *root = top;
  while (mlir::Operation *parent = root->getParentOp())
    root = parent;
  if (mlir::failed(solver.initializeAndRun(root)))
    return facts;

  // A value the solver reached but nothing demanded still has its initial
  // state, which carries no dimensions; it is recorded in the same form `of`
  // reports an unreached value in.
  auto record = [&](mlir::Value v) {
    if (const auto *state = solver.lookupState<DemandLattice>(v))
      facts.values[v] = state->getValue().isNone() ? Demand::none(rankOf(v))
                                                   : state->getValue();
  };
  top->walk([&](mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      record(result);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument arg : block.getArguments())
          record(arg);
    if (unit::isNode(op) && op->getNumResults() == 1 && isWeightMatmul(op)) {
      const Demand d = facts.of(op->getResult(0));
      facts.weight_rows[op] = d.dims.empty() ? (d.isNone() ? IndexSet::empty()
                                                           : IndexSet::all())
                                             : d.dims.back();
    }
  });
  facts.ok = true;
  return facts;
}

} // namespace sonicboom::compiler
