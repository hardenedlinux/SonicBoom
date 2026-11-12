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

// The forward stage analysis over Boom IR (BoomStage.h).

#include "sonicboom/compiler/BoomStage.h"

#include "BoomUnit.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <map>

namespace sonicboom::compiler {

void printSources(llvm::raw_ostream &os, Sources sources) {
  static constexpr const char *kNames[] = {"prompt", "counter", "token",
                                           "cache"};
  os << "{";
  bool first = true;
  for (unsigned bit = 0; bit < 4; ++bit) {
    if (!((sources.bits >> bit) & 1u))
      continue;
    os << (first ? "" : ",") << kNames[bit];
    first = false;
  }
  os << "}";
}

namespace {

// The lattice value: a source set, in the form mlir::dataflow::Lattice reads.
struct SourceSet {
  Sources sources;

  static SourceSet join(const SourceSet &a, const SourceSet &b) {
    return {Sources::join(a.sources, b.sources)};
  }
  bool operator==(const SourceSet &o) const { return sources == o.sources; }
  void print(llvm::raw_ostream &os) const { printSources(os, sources); }
};

class StageLattice : public mlir::dataflow::Lattice<SourceSet> {
public:
  using Lattice::Lattice;
};

bool isCacheHandle(mlir::Value v) {
  return llvm::isa<boom::KvType>(v.getType());
}

// A standalone graph's free tensor, by the name the program lift binds it by.
Sources graphArgumentSources(mlir::BlockArgument arg) {
  boom::TensorDescAttr desc = unit::descOf(arg);
  if (!desc)
    return Sources::all();
  if (desc.getName() == "token_id")
    return Sources::of(Source::Token);
  if (desc.getName() == "pos")
    return Sources::of(Source::Prompt).with(Source::Counter);
  return Sources::all();
}

class StageAnalysis
    : public mlir::dataflow::SparseForwardDataFlowAnalysis<StageLattice> {
public:
  using SparseForwardDataFlowAnalysis::SparseForwardDataFlowAnalysis;

  // Every loop-carried value an iteration replaces depends on how many
  // iterations have run. The framework joins a loop argument from its initial
  // and its yielded value, neither of which says so; it is stated here, once,
  // and the join can only add to it.
  mlir::LogicalResult initialize(mlir::Operation *top) override {
    if (mlir::failed(SparseForwardDataFlowAnalysis::initialize(top)))
      return mlir::failure();
    top->walk([&](mlir::Operation *op) {
      if (auto loop = llvm::dyn_cast<mlir::scf::WhileOp>(op)) {
        seedLoop(loop);
        return;
      }
      // A region op the analysis has no loop rule for, other than a plain
      // conditional: nothing is assumed about how often its regions run.
      if (llvm::isa<mlir::RegionBranchOpInterface>(op) &&
          !llvm::isa<mlir::scf::IfOp>(op))
        for (mlir::Region &region : op->getRegions())
          for (mlir::Block &block : region)
            for (mlir::BlockArgument arg : block.getArguments())
              add(getLatticeElement(arg), Sources::all());
    });
    return mlir::success();
  }

  mlir::LogicalResult
  visitOperation(mlir::Operation *op,
                 llvm::ArrayRef<const StageLattice *> operands,
                 llvm::ArrayRef<StageLattice *> results) override {
    if (auto step = llvm::dyn_cast<boom::StepOp>(op)) {
      visitStep(step, results);
      return mlir::success();
    }

    // The binding points: a chosen token is known for what it is.
    if (llvm::isa<boom::SampleOp, boom::AcceptOp>(op)) {
      for (StageLattice *result : results)
        add(result, Sources::of(Source::Token));
      return mlir::success();
    }
    // A draft proposes tokens from the history: the prompt and what was
    // accepted so far.
    if (llvm::isa<boom::DraftOp>(op)) {
      for (StageLattice *result : results)
        add(result, Sources::of(Source::Prompt).with(Source::Token));
      return mlir::success();
    }

    // An op that may read state the IR does not name can depend on anything.
    Sources sources =
        mlir::isMemoryEffectFree(op) ? Sources::none() : Sources::all();
    for (const StageLattice *operand : operands)
      sources = Sources::join(sources, operand->getValue().sources);
    for (StageLattice *result : results)
      add(result, sources);
    return mlir::success();
  }

  // A value the framework cannot trace to a definition: a block argument with
  // no control-flow predecessor (the arguments of a program, a graph and a
  // step body), or a result it gave up on.
  void setToEntryState(StageLattice *lattice) override {
    const mlir::Value anchor = lattice->getAnchor();
    auto arg = llvm::dyn_cast<mlir::BlockArgument>(anchor);
    if (!arg) {
      add(lattice, Sources::all());
      return;
    }
    mlir::Block *block = arg.getOwner();
    mlir::Operation *parent = block->getParentOp();

    // A step's body argument is the step's operand.
    if (auto step = llvm::dyn_cast_or_null<boom::StepOp>(parent)) {
      if (arg.getArgNumber() < step->getNumOperands()) {
        const StageLattice *operand = getLatticeElementFor(
            getProgramPointBefore(block), step->getOperand(arg.getArgNumber()));
        add(lattice, operand->getValue().sources);
      } else {
        add(lattice, Sources::all());
      }
      return;
    }
    if (isCacheHandle(anchor)) {
      add(lattice, Sources::of(Source::Cache));
      return;
    }
    if (llvm::isa_and_nonnull<boom::ProgramOp>(parent)) {
      add(lattice, Sources::of(Source::Prompt));
      return;
    }
    if (llvm::isa_and_nonnull<boom::GraphOp>(parent)) {
      add(lattice, graphArgumentSources(arg));
      return;
    }
    add(lattice, Sources::all());
  }

private:
  void add(StageLattice *lattice, Sources sources) {
    propagateIfChanged(lattice, lattice->join(SourceSet{sources}));
  }

  // A step is transparent: its result is its body's output.
  void visitStep(boom::StepOp step, llvm::ArrayRef<StageLattice *> results) {
    mlir::Block &body = step.getBody().front();
    mlir::ProgramPoint *after = getProgramPointAfter(step.getOperation());
    for (auto [index, id] : llvm::enumerate(step.getOutputIds())) {
      if (index >= results.size())
        break;
      mlir::Value out = unit::valueOf(body, step.getTensors(), id);
      if (!out) {
        add(results[index], Sources::all());
        continue;
      }
      // Read as a dependency of the step, so the step is revisited when a
      // body node raises its output.
      add(results[index], getLatticeElementFor(after, out)->getValue().sources);
    }
  }

  // scf.while carries its values through the `before` region's arguments; the
  // `after` region yields the next iteration's. A carried value is replaced
  // when what comes back is not the argument itself.
  void seedLoop(mlir::scf::WhileOp loop) {
    mlir::Block &before = loop.getBefore().front();
    mlir::Block &after = loop.getAfter().front();
    mlir::Operation *condition = before.getTerminator();
    mlir::Operation *yield = after.getTerminator();
    const unsigned carried = before.getNumArguments();
    for (unsigned i = 0; i < carried; ++i) {
      // Operand 0 of the condition is the predicate; the forwarded values
      // follow it.
      const bool forwarded = i + 1 < condition->getNumOperands() &&
                             condition->getOperand(i + 1) ==
                                 mlir::Value(before.getArgument(i));
      const bool returned = i < yield->getNumOperands() &&
                            i < after.getNumArguments() &&
                            yield->getOperand(i) ==
                                mlir::Value(after.getArgument(i));
      if (!(forwarded && returned))
        add(getLatticeElement(before.getArgument(i)),
            Sources::of(Source::Counter));
    }
  }
};

// The units under `top`, `top` included.
void forEachUnit(mlir::Operation *top,
                 llvm::function_ref<void(mlir::Operation *)> visit) {
  top->walk([&](mlir::Operation *op) {
    if (llvm::isa<boom::GraphOp, boom::StepOp>(op))
      visit(op);
  });
}

} // namespace

Sources StageFacts::of(mlir::Value v) const {
  auto it = values.find(v);
  return it == values.end() ? Sources::all() : it->second;
}

StageFacts analyzeStage(mlir::Operation *top) {
  StageFacts facts;
  if (!top)
    return facts;

  // Reachability and stage are solved together, as for demand: a value on a
  // path the dead-code analysis does not prove live joins nothing in.
  mlir::DataFlowSolver solver;
  solver.load<mlir::dataflow::DeadCodeAnalysis>();
  solver.load<mlir::dataflow::SparseConstantPropagation>();
  solver.load<StageAnalysis>();
  // The solve covers the enclosing root for the reason BoomDemand.cpp gives:
  // the solver proves live only the blocks under the op it is started on.
  mlir::Operation *root = top;
  while (mlir::Operation *parent = root->getParentOp())
    root = parent;
  if (mlir::failed(solver.initializeAndRun(root)))
    return facts;

  auto record = [&](mlir::Value v) {
    if (const auto *state = solver.lookupState<StageLattice>(v))
      facts.values[v] = state->getValue().sources;
  };
  top->walk([&](mlir::Operation *op) {
    for (mlir::Value result : op->getResults())
      record(result);
    for (mlir::Region &region : op->getRegions())
      for (mlir::Block &block : region)
        for (mlir::BlockArgument arg : block.getArguments())
          record(arg);
  });
  facts.ok = true;
  return facts;
}

bool recordStages(mlir::Operation *top) {
  if (!top)
    return false;
  const StageFacts facts = analyzeStage(top);
  if (!facts.ok)
    return false;

  forEachUnit(top, [&](mlir::Operation *op) {
    const mlir::ArrayAttr tensors = unit::tensorsOf(op);
    mlir::Block &body = op->getRegion(0).front();

    // What this run computed, by tensor id.
    std::map<int64_t, Sources> computed;
    const llvm::SmallVector<int64_t> free = unit::freeIds(body, tensors);
    for (auto [index, id] : llvm::enumerate(free))
      if (index < body.getNumArguments())
        computed[id] = facts.of(body.getArgument(unsigned(index)));
    for (mlir::Operation &node : body) {
      auto ids = node.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
      if (!ids)
        continue;
      for (auto [index, id] : llvm::enumerate(ids.asArrayRef()))
        if (index < node.getNumResults())
          computed[id] = facts.of(node.getResult(unsigned(index)));
    }

    // An earlier recording and this one are both true of a tensor the unit
    // still has — each contains what the tensor really depends on — so what
    // is in both is true as well, and at least as precise as either. The
    // earlier one is usually the smaller: it was made when more of the graph
    // was readable.
    if (auto earlier = op->getAttrOfType<mlir::DenseI64ArrayAttr>("stages")) {
      const llvm::ArrayRef<int64_t> flat = earlier.asArrayRef();
      for (size_t i = 0; i + 1 < flat.size(); i += 2) {
        auto it = computed.find(flat[i]);
        if (it != computed.end())
          it->second.bits &= uint8_t(flat[i + 1] & Sources::kAll);
      }
    }

    llvm::SmallVector<int64_t> flat;
    flat.reserve(computed.size() * 2);
    for (const auto &[id, sources] : computed) {
      flat.push_back(id);
      flat.push_back(int64_t(sources.bits));
    }
    op->setAttr("stages",
                mlir::DenseI64ArrayAttr::get(op->getContext(), flat));
  });
  return true;
}

Sources recordedSources(mlir::Operation *unit, int64_t id) {
  if (!unit)
    return Sources::all();
  auto stages = unit->getAttrOfType<mlir::DenseI64ArrayAttr>("stages");
  if (!stages)
    return Sources::all();
  const llvm::ArrayRef<int64_t> flat = stages.asArrayRef();
  for (size_t i = 0; i + 1 < flat.size(); i += 2)
    if (flat[i] == id)
      return Sources{uint8_t(flat[i + 1] & Sources::kAll)};
  return Sources::all();
}

} // namespace sonicboom::compiler
