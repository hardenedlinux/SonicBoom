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
// (design/mlir-migration.md §7d). This is the emitter's side of the dialect:
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
  BoomBuilder(mlir::MLIRContext *context, llvm::StringRef graph_name);

  // A graph input: a free variable, created as a block argument, and listed in
  // the graph's `inputs`. Emitters declare every input before any node, so the
  // block arguments come out in ascending tensor-id order — the order the
  // verifier recomputes them in.
  mlir::Value input(llvm::StringRef name, llvm::ArrayRef<int64_t> shape,
                    sx::DType dtype);

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
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
  boom::GraphOp graph_op_;
  mlir::Block *body_ = nullptr;
  llvm::SmallVector<mlir::Attribute> table_;
  llvm::SmallVector<int64_t> inputs_;
  std::vector<int64_t> outputs_;
  llvm::DenseMap<mlir::Value, int64_t> ids_;
  int64_t nodes_ = 0;
  bool failed_ = false;
};

} // namespace sonicboom::compiler
