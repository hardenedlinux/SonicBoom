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
// Demand over Boom IR: which part of every value is observed, and how closely.
//
// A backward data-flow analysis (mlir::dataflow::SparseBackwardDataFlowAnalysis)
// run together with MLIR's dead-code and constant-propagation analyses, so a
// value read only on an unreachable path is not demanded. The fact it computes
// per SSA value is a `Demand`:
//
//   * an Observation — nothing reads the value (None), readers only rank its
//     elements (Order), or readers need the elements themselves (Exact);
//   * an IndexSet per dimension — the indices along that dimension some reader
//     needs: none, an interval, a runtime list of bounded length, or all.
//
// Demand starts at the observers and flows to the operands:
//
//   * a boom.graph's output is observed by the consumer the graph declares
//     (GraphOp::consumer / consumer_list); with no declaration it is observed
//     whole and exactly;
//   * inside a boom.program the observers are the program's own ops — the
//     sample (ranking the logits when it takes the mode, reading them when it
//     draws), the stop test, the emission — and a boom.step hands its result's
//     demand to its body's output and its body's free variables' demand to its
//     operands, so demand crosses the step boundary in both directions and
//     around the loop;
//   * a node that writes the KV cache, or declares nothing about its effects,
//     is itself an observer: it stays, and it demands its operands whole.
//
// Through a node, demand follows what the op declares (BoomOps.td, "Demand
// traits"): an elementwise op hands each operand the result's demand, a
// monotone one keeps a ranking observation a ranking, a row-independent op
// keeps the row set, a matmul over a model-bound weight keeps the row set and
// records which weight rows are read. Everything else is opaque — operands
// whole and exact — so an op the analysis does not know can make the result
// less precise but never wrong.
//
// The analysis is read-only. What acts on it is the consumer resolution
// (BoomPasses.h resolveConsumer), and later passes that prune.
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <cstdint>

namespace sonicboom::compiler {

// How closely a value is observed. A chain: None < Order < Exact.
enum class Observation : uint8_t {
  None = 0,   // nothing reads it
  Order = 1,  // readers only rank its elements against each other
  Exact = 2,  // readers need the elements
};

// The indices read along one dimension. Ordered by inclusion, Empty at the
// bottom and All at the top:
//
//   Empty  ⊑  Interval [lo, hi)  ⊑  All
//   Empty  ⊑  List <= capacity   ⊑  All
//
// A List is a set known only at run time, of at most `capacity` indices (a
// grammar's allowed token ids). Two lists are the same set exactly when their
// capacities agree: a unit has one consumer, and so one list. A second list
// source would need its own identity here; until then, lists of different
// capacity — or a list and an interval — join to All.
struct IndexSet {
  enum class Kind : uint8_t { Empty, Interval, List, All };

  Kind kind = Kind::Empty;
  int64_t lo = 0;        // Interval: first index
  int64_t hi = 0;        // Interval: one past the last
  int64_t capacity = 0;  // List: the most indices it can hold

  static IndexSet empty() { return {}; }
  static IndexSet all();
  static IndexSet interval(int64_t lo, int64_t hi);  // empty when hi <= lo
  static IndexSet list(int64_t capacity);            // empty when capacity <= 0

  // The least set containing both.
  static IndexSet join(const IndexSet &a, const IndexSet &b);

  bool isEmpty() const { return kind == Kind::Empty; }
  bool isAll() const { return kind == Kind::All; }
  bool isList() const { return kind == Kind::List; }

  bool operator==(const IndexSet &o) const;
  bool operator!=(const IndexSet &o) const { return !(*this == o); }

  void print(llvm::raw_ostream &os) const;
};

// What is observed of one value: how closely, and where. `dims` has one entry
// per dimension of the value, outermost first (none for a value that is not a
// tensor: a cache handle, a loop counter). A value nothing reads has
// Observation::None and every dimension Empty.
struct Demand {
  Observation obs = Observation::None;
  llvm::SmallVector<IndexSet, 2> dims;

  // Nothing is read.
  static Demand none(unsigned rank = 0);
  // Every element is read (or, with Order, ranked).
  static Demand whole(unsigned rank, Observation obs = Observation::Exact);

  // The least demand covering both: the closer observation, the larger index
  // set per dimension. Ranks that disagree (the same value is never given two)
  // widen to the larger rank with the missing dimensions read whole.
  static Demand join(const Demand &a, const Demand &b);
  // The backward framework accumulates with `meet`; both directions of this
  // lattice accumulate demand, so they are the same operation.
  static Demand meet(const Demand &a, const Demand &b) { return join(a, b); }

  bool isNone() const { return obs == Observation::None; }
  // Every dimension is read whole.
  bool isWholeRegion() const;

  bool operator==(const Demand &o) const;
  bool operator!=(const Demand &o) const { return !(*this == o); }

  void print(llvm::raw_ostream &os) const;
};

// The result of one run. `values` holds every SSA value under the analyzed op
// the solver reached; a value it did not reach (dead code) is absent and reads
// as Demand::none. `weight_rows` holds, for each matmul over a model-bound
// weight, the weight rows its readers need — the fact a row-gathered kernel,
// and later a residency decision, is chosen from.
struct DemandFacts {
  bool ok = false;
  llvm::DenseMap<mlir::Value, Demand> values;
  llvm::DenseMap<mlir::Operation *, IndexSet> weight_rows;

  Demand of(mlir::Value v) const;
  IndexSet weightRows(mlir::Operation *node) const;
};

// Run the analysis over `top` (a module, a boom.graph or a boom.program) and
// everything nested in it. The IR is not modified. `ok` is false when the
// solver failed; the facts are then empty.
DemandFacts analyzeDemand(mlir::Operation *top);

// True for the node ops the analysis treats as a matmul over a model-bound
// weight: the result's innermost dimension indexes the weight's rows.
bool isWeightMatmul(mlir::Operation *node);

} // namespace sonicboom::compiler
