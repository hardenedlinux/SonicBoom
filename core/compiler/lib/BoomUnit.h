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
// A unit's body: the values of a boom.graph or a boom.step.
//
// Both ops carry a node list with its tensor table. The table names every
// tensor by id; the body's SSA values are bound to the ids by the relation the
// verifier states: a node result is the id at the same index of the node's
// `result_ids`, and a tensor no node produces — a free tensor — is the block
// argument at its rank among the free ids, ascending. The cache handles follow
// the free tensors and have no table entry.
//
// The analyses read a unit through these helpers so that relation is written
// once.
//===----------------------------------------------------------------------===//

#pragma once

#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

namespace sonicboom::compiler::unit {

// A node op: one of the per-OpKind ops, which all carry the graph-node record.
inline bool isNode(mlir::Operation *op) {
  return op->getName().getDialectNamespace() == "boom" &&
         op->getAttrOfType<mlir::IntegerAttr>("node_id") &&
         op->getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
}

// The tensor table of a unit op; null for any other op.
inline mlir::ArrayAttr tensorsOf(mlir::Operation *op) {
  if (auto g = llvm::dyn_cast_or_null<boom::GraphOp>(op))
    return g.getTensors();
  if (auto s = llvm::dyn_cast_or_null<boom::StepOp>(op))
    return s.getTensors();
  return {};
}

// The ids of the tensors no node of `body` produces, ascending: the order of
// the body's leading block arguments.
inline llvm::SmallVector<int64_t> freeIds(mlir::Block &body,
                                          mlir::ArrayAttr tensors) {
  llvm::DenseSet<int64_t> produced;
  for (mlir::Operation &node : body)
    if (auto ids = node.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids"))
      for (int64_t id : ids.asArrayRef())
        produced.insert(id);
  llvm::SmallVector<int64_t> free;
  for (mlir::Attribute entry : tensors)
    if (auto desc = llvm::dyn_cast<boom::TensorDescAttr>(entry))
      if (!produced.count(desc.getId()))
        free.push_back(desc.getId());
  llvm::sort(free);
  return free;
}

// The SSA value of tensor `id` in a unit's body; null when the table has no
// such tensor.
inline mlir::Value valueOf(mlir::Block &body, mlir::ArrayAttr tensors,
                           int64_t id) {
  for (mlir::Operation &node : body) {
    auto ids = node.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    if (!ids)
      continue;
    for (auto [index, rid] : llvm::enumerate(ids.asArrayRef()))
      if (rid == id && index < node.getNumResults())
        return node.getResult(unsigned(index));
  }
  const llvm::SmallVector<int64_t> free = freeIds(body, tensors);
  auto it = llvm::find(free, id);
  if (it == free.end())
    return {};
  const unsigned index = unsigned(it - free.begin());
  return index < body.getNumArguments() ? mlir::Value(body.getArgument(index))
                                        : mlir::Value();
}

// The table entry of a unit's free tensor, by its block argument; null for a
// cache handle, and for an argument of a block that is not a unit's body.
inline boom::TensorDescAttr descOf(mlir::BlockArgument arg) {
  mlir::Block *body = arg.getOwner();
  const mlir::ArrayAttr tensors = tensorsOf(body->getParentOp());
  if (!tensors)
    return {};
  const llvm::SmallVector<int64_t> free = freeIds(*body, tensors);
  if (arg.getArgNumber() >= free.size())
    return {};
  for (mlir::Attribute entry : tensors)
    if (auto desc = llvm::dyn_cast<boom::TensorDescAttr>(entry))
      if (desc.getId() == free[arg.getArgNumber()])
        return desc;
  return {};
}

// The table id of a unit body's value: a node result's, or a free tensor's
// block argument's. -1 for a cache handle and for any other value.
inline int64_t idOf(mlir::Value value) {
  if (auto arg = llvm::dyn_cast<mlir::BlockArgument>(value)) {
    boom::TensorDescAttr desc = descOf(arg);
    return desc ? desc.getId() : -1;
  }
  mlir::Operation *node = value.getDefiningOp();
  if (!node || !isNode(node))
    return -1;
  auto ids = node->getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
  const unsigned index = llvm::cast<mlir::OpResult>(value).getResultNumber();
  return index < ids.size() ? ids[index] : -1;
}

} // namespace sonicboom::compiler::unit
