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

// The incrementally-built Boom IR a plan emitter writes
// (docs/boom-ir.md). This is the emitter's side of the dialect:
// the model layer names operators, shapes, dtypes and attributes and gets back
// the SSA values it feeds to the next node. It never names a tensor id, and it
// never builds a planner::Graph — the graph is what the lowering produces from
// this IR, not what the emitter writes.
//
// The IR built here is the IR `graphToBoom` builds from the equivalent graph:
// the same tensor table in the same order, the same block arguments, the same
// node attribute record, the same result ids, the same SSA order — because the
// passes and the plan are derived from it, and two constructors of the same IR
// that could disagree is one too many. `createBoomNode` is the shared half: one
// implementation of the node op record, called by both, so the record cannot
// drift between them.
//
// Every node result is an f32 activation. A q8_1 value is one a *pass* created
// (QuantizeOncePass, BoomPasses.cpp), and a pass creates it in the IR; an
// emitter that wants a quantized activation emits the matmul and leaves the
// quantize to the pipeline. An emitter that needs a q8_1 operand for another
// reason has no spelling here, deliberately: it would be a second, unverified
// producer of a layout only the passes are allowed to invent.

#pragma once

#include "sonicboom/compiler/BoomOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OwningOpRef.h"

#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sonicboom::compiler {

// sx::Attribute's value as the MLIR attribute a #boom.attr carries. Every kind
// sx::Attribute has maps to one MLIR attribute, so a node attribute is never
// dropped; the one corner is an empty vector, which cannot say which element
// type it is (neither could sx::Attribute, whose variant index is the only sign
// of it) and reads back as the int64 vector.
mlir::Attribute toMlirAttribute(mlir::MLIRContext *context,
                                const sx::Attribute &attr);

// The reverse. `name` is carried over unchanged: node_attrs is order-preserving,
// and the name is the key the typed attributes are matched against.
std::optional<sx::Attribute> fromMlirAttribute(llvm::StringRef name,
                                               mlir::Attribute value);

// One `boom.<op>` node: its operand/result SSA list, its `node_name`/`node_id`/
// `result_ids`/`node_attrs` record, the optional backend override, and the typed
// view of its attributes (the names ODS declares — BoomOps.td — which are the
// ones the plan emitters and the passes read).
//
// An attribute in that typed vocabulary whose value is neither an integer nor a
// float has no typed attribute to set, and the graph verifier would reject the
// mismatch, so it is reported as a diagnostic and nullptr is returned rather
// than a node whose two views of the same attribute disagree.
mlir::Operation *createBoomNode(mlir::OpBuilder &builder, mlir::Location loc,
                                planner::OpKind op,
                                llvm::ArrayRef<mlir::Value> operands,
                                llvm::ArrayRef<mlir::Type> results,
                                llvm::StringRef name, int64_t node_id,
                                llvm::ArrayRef<int64_t> result_ids,
                                llvm::ArrayRef<sx::Attribute> attributes,
                                std::optional<planner::BackendTag> backend);

// Lift a module's single boom.graph (a decode graph) into the greedy generation
// program (docs/boom-ir.md): a `boom.program` holding a
// `boom.prefill` call, then an `scf.while` whose body is the decode graph as a
// `boom.step`, a `boom.sample`, a `boom.emit` and a `boom.stop_check`, and whose
// continuation advances the position with `boom.pos_inc`.
//
// The graph op is consumed: its body — the node list and tensor table — moves
// into the step, so the step's block arguments, in order, are the graph's free
// tensor variables (the token and the position) followed by its cache handles.
// The program's own block arguments are the prompt, the decoding start
// position, the first token and position, and one cache handle per layer.
//
// `prompt_shape` types the prompt the prefill consumes (an int64 tensor);
// `eos_tokens` is the stop set and `max_tokens` the emitted-count bound (0 for
// unbounded). The program is verified when this returns.
mlir::LogicalResult addGreedyProgram(mlir::ModuleOp module,
                                     llvm::StringRef program_name,
                                     llvm::StringRef prefill_graph,
                                     llvm::ArrayRef<int64_t> prompt_shape,
                                     llvm::ArrayRef<int64_t> eos_tokens,
                                     int64_t max_tokens);

// The speculative form of the program (B6). The loop body is
//
//   boom.draft -> boom.step / boom.verify -> boom.accept -> boom.emit ->
//   boom.stop_check
//
// with the step kept as the k == 0 fallback the executor runs when the drafter
// proposes nothing. The draft node names the proposal strategy (`"ngram"`:
// the prompt-lookup drafter) and its cap `max_drafts`; `max_drafts + 1` is the
// verify step's row budget (at most 16, the device token slot's rows). The
// continuation advances the carried position by the accept node's run length
// plus one (boom.pos_inc's `by` operand). The graph op is consumed exactly as
// addGreedyProgram consumes it.
mlir::LogicalResult addSpeculativeProgram(mlir::ModuleOp module,
                                          llvm::StringRef program_name,
                                          llvm::StringRef prefill_graph,
                                          llvm::ArrayRef<int64_t> prompt_shape,
                                          llvm::ArrayRef<int64_t> eos_tokens,
                                          int64_t max_tokens,
                                          int64_t max_drafts,
                                          llvm::StringRef draft_strategy);

// A whole graph under construction: one module holding one boom.graph op, whose
// body is the node list and whose block arguments are the inputs the emitter
// declared.
//
// Ids are the builder's business. A tensor's id is its index in the table, which
// is creation order; an input's id is therefore its position among the inputs
// and a node result's is its position among all tensors created before it —
// exactly the ids a graph built in the same order would carry.
class BoomBuilder {
public:
  // The dialect is loaded here: an operation cannot be created by name until it
  // is. `context` must outlive the builder and everything built from it.
  //
  // `extent` is the segment length a dynamic shape (`?`) stands for in this
  // build: the table entry of a dynamic tensor records the byte count at that
  // extent, so the passes (which decide fusions by comparing byte sizes) see
  // the same sizes they would on the concrete graph at that length. The graph
  // op records it as the sizing extent too, which is what the lowering
  // evaluates the affine size from when it binds a (not larger) extent
  // (BoomConversion.cpp). 0 leaves dynamic entries unsized.
  BoomBuilder(mlir::MLIRContext *context, llvm::StringRef graph_name,
              uint64_t extent = 0);

  // A graph input: a free variable, created as a block argument, and listed in
  // the graph's `inputs`. Emitters declare every input before any node, so the
  // block arguments come out in ascending tensor-id order — the order the
  // verifier recomputes them in.
  mlir::Value input(llvm::StringRef name, llvm::ArrayRef<int64_t> shape,
                    sx::DType dtype);

  // The per-layer KV cache handle (`!boom.kv<layer>`): a block argument after
  // every declared input, returned to the cache-touching node as its last
  // operand. Declare the handles before any node, in ascending layer order —
  // that is the block-argument order the verifier recomputes.
  mlir::Value kv(uint32_t layer);

  // Deal each node's single-row output out over `rows` positions: every
  // node() shape is multiplied by `rows` (position-major [rows, dim]) and each
  // node carries `attr_name` = rows as an extra attribute. The verify step
  // (B3) emits one decode graph over k+1 query rows this way; 0 disables.
  void set_row_scale(int64_t rows, llvm::StringRef attr_name);

  // A node and its one result tensor: an f32 activation of `out_shape`. The
  // node's name is its operator's (op_kind_name), the tensor's is `out_name`.
  mlir::Value node(planner::OpKind op, llvm::ArrayRef<mlir::Value> operands,
                   llvm::StringRef out_name, llvm::ArrayRef<int64_t> out_shape,
                   llvm::ArrayRef<sx::Attribute> attributes,
                   planner::BackendTag backend = planner::BackendTag::Sonic);

  // Name `value`'s tensor the graph's output. The last call wins: a graph has
  // one output, and an emitter that set two would be a graph the round trip
  // could not reproduce.
  void set_output(mlir::Value value);

  // The finished graph, verified — the dialect's verifier is what decides
  // whether the emitter wrote a graph at all, and it runs here rather than three
  // passes later. Empty when the build could not be completed: a diagnostic on
  // the context names what was refused.
  mlir::OwningOpRef<mlir::ModuleOp> finish();

private:
  // The tensor table index a value's tensor carries, or nullopt for a value this
  // builder did not create.
  std::optional<int64_t> idOf(mlir::Value value) const;

  // Adds a tensor table entry and returns its id. `is_input` also makes it a
  // block argument.
  int64_t addTensor(llvm::StringRef name, llvm::ArrayRef<int64_t> shape,
                    sx::DType dtype, uint64_t size_bytes, bool is_input);

  mlir::MLIRContext *context_;
  mlir::Location loc_;
  uint64_t extent_ = 0;  // segment length a dynamic shape is sized at
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  boom::GraphOp graph_op_;
  mlir::Block *body_ = nullptr;
  llvm::SmallVector<mlir::Attribute> table_;
  llvm::SmallVector<int64_t> inputs_;
  std::vector<int64_t> outputs_;
  llvm::DenseMap<mlir::Value, int64_t> ids_;
  llvm::DenseMap<uint32_t, mlir::Value> kv_;
  int64_t nodes_ = 0;
  int64_t row_scale_ = 0;        // out_shape multiplier (verify rows); 0 = off
  std::string row_scale_attr_;   // attribute stamped on each node when on
  bool failed_ = false;
};

} // namespace sonicboom::compiler
