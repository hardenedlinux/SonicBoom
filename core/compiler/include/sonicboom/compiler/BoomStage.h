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
// Stage over Boom IR: what each value depends on, and so when it is known.
//
// A forward data-flow analysis (mlir::dataflow::SparseForwardDataFlowAnalysis)
// run together with MLIR's dead-code and constant-propagation analyses. The
// fact it computes per SSA value is a set of `Source`s — the inputs of a
// generation whose change can change the value:
//
//   Prompt   what a request brings: the prompt, the position it starts at, the
//            first token;
//   Counter  how many loop iterations have run;
//   Token    a token the loop chose and fed back;
//   Cache    the contents of the KV cache.
//
// The empty set is a value that depends on the model alone. Sets are ordered
// by inclusion and joined by union, so the lattice has height four and the
// solve terminates without widening.
//
// Where a source enters:
//
//   * a program's arguments are the request: a cache handle carries Cache,
//     every other argument Prompt;
//   * a loop-carried value that an iteration replaces carries Counter;
//   * the ops that choose a token — sample, draft, accept — are binding
//     points: their results carry Token and nothing else, because once the
//     token is chosen, what it was chosen from no longer has to be known;
//   * a standalone boom.graph has no loop around it to read these from, so its
//     arguments are taken by the names the program lift itself binds
//     (`token_id` carries Token, `pos` carries Prompt and Counter), a cache
//     handle carries Cache, and any other argument is unknown — every source.
//
// Through an op, the result depends on what the operands depend on. An op that
// is not declared free of memory effects may read state the IR does not name,
// so its results carry every source. A boom.step is transparent, as it is for
// demand: its body's arguments are its operands, its result its body's output.
//
// The set orders into a `Stage` — the earliest moment the value can be
// computed:
//
//   Model     at load;
//   Request   once per generation, before the loop;
//   Schedule  for every iteration ahead of time (a function of the iteration
//             number);
//   Token     as soon as the iteration's token is chosen, before anything that
//             reads the cache;
//   State     in the iteration, in order.
//
// A stage is a fact about what a value means, not about who reads it, so it
// stays true of a tensor through every rewrite that keeps the tensor's
// meaning. That is what lets it be recorded on the unit, keyed by tensor id,
// and carried through the passes and the lowering (recordStages below;
// planner/facts.h on the other side), where a fact about readers — demand —
// has to be acted on where it is computed.
//
// The analysis is read-only.
//===----------------------------------------------------------------------===//

#pragma once

#include "sonicboom/planner/facts.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include <cstdint>

namespace sonicboom::compiler {

// The sources and stages are the planner's (planner/facts.h): the same facts
// are read on both sides of the lowering.
using planner::Source;
using planner::Sources;
using planner::Stage;

// "{}" for the empty set, else the members in bit order: "{prompt,counter}".
void printSources(llvm::raw_ostream &os, Sources sources);

// The result of one run. `values` holds every SSA value under the analyzed op
// the solver reached. A value it did not reach reads as every source: nothing
// is known about when dead code could run.
struct StageFacts {
  bool ok = false;
  llvm::DenseMap<mlir::Value, Sources> values;

  Sources of(mlir::Value v) const;
  Stage stage(mlir::Value v) const { return of(v).stage(); }
};

// Run the analysis over `top` (a module, a boom.graph or a boom.program) and
// everything nested in it. The IR is not modified. `ok` is false when the
// solver failed; the facts are then empty.
StageFacts analyzeStage(mlir::Operation *top);

// Record what every tensor of every unit under `top` depends on, on the unit,
// as its `stages` attribute: (tensor id, source bits) pairs, ascending id.
// Where a unit already records a tensor, the entry becomes the sources both
// recordings name: each is true of the tensor, so their intersection is, and a
// pass that fused away the ops the analysis read a tensor through cannot make
// its entry less precise. Entries for tensors the unit no longer has are
// dropped. Returns false when the analysis failed; nothing is recorded then.
bool recordStages(mlir::Operation *top);

// The recorded sources of tensor `id` of a unit (a boom.graph or boom.step);
// every source when the unit records none for it.
Sources recordedSources(mlir::Operation *unit, int64_t id);

} // namespace sonicboom::compiler
