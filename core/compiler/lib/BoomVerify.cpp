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

// Boom IR verification (design/mlir-migration.md §6, "Verifiers for
// shape/dtype/quant consistency").
//
// The dialect's ops already reject a value of a foreign type: every operand and
// result is constrained by Boom_ValueType, which ODS turns into an
// verifyInvariants check. What is left is everything that needs the graph as a
// whole, and that is what boom.graph's verifier checks — the unit of meaning in
// this IR is the graph, not the node:
//
//   * the tensor table is well formed, and every id the graph or a node names
//     is in it and is defined exactly once;
//   * the body's block arguments are the graph's free variables — every tensor
//     the graph references but does not produce, in ascending id order — and
//     each argument's type agrees with that tensor's table entry;
//   * a node's results carry `result_ids` (right arity, unique ids, in the
//     table) and its operands resolve to tensor ids, so every SSA value has
//     exactly one tensor id — the mapping the round trip relies on;
//   * every operand/result type agrees with its tensor table entry (shape and
//     dtype), which is the shape/dtype consistency check;
//   * the typed attributes and the `node_attrs` record are the same set — that
//     is what makes `node_attrs` the authoritative serialization and the typed
//     attributes a checked view of it;
//   * the quantized-type rules: what a q8_1 value is, who may produce one, who
//     may consume one, and the legal values of `ds`;
//   * the elementwise ops' shapes.
//
// Quant consistency is the reason the quantized formats are type-system
// citizens (§6.2). `ds` is the mmq activation layout a weight format selects —
// today an emitter-baked integer that fuse_rms_norm_mmq_pass reads
// (planner/rewrites.cpp:362-377) — so its legality is stated here rather than
// left implicit in whatever the emitter happened to write.

#include "sonicboom/compiler/BoomOps.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>

namespace sonicboom::compiler::boom {

using mlir::Attribute;
using mlir::NamedAttribute;
using mlir::Operation;
using mlir::Type;
using mlir::Value;

//--- the typed attribute vocabulary -----------------------------------------

namespace {

const llvm::StringRef kBoomTypedAttrNames[] = {
    "slot",    "layer", "kv_source_layer", "ds",     "batch", "dim",
    "offset",  "rows",  "length",          "stride", "scale",
};

} // namespace

llvm::ArrayRef<llvm::StringRef> boomTypedAttrNames() {
  return kBoomTypedAttrNames;
}

bool isBoomTypedAttrName(llvm::StringRef name) {
  return llvm::is_contained(kBoomTypedAttrNames, name);
}

llvm::SmallVector<NamedAttribute> typedNodeAttributes(Operation *op) {
  llvm::SmallVector<NamedAttribute> out;
  for (llvm::StringRef name : kBoomTypedAttrNames)
    if (Attribute value = op->getAttr(name))
      out.push_back(NamedAttribute(llvm::StringRef(name), value));
  return out;
}

//--- op classes the quant/shape rules below are stated over ------------------

namespace {

// The ops whose result is `!boom.q8_1`: the activation quantizers. Whether a
// given one of their results is a q8_1 layout is decided by its consumer, not by
// the op — the same quantize output could in principle feed a f32 path.
bool isQ8_1Producer(llvm::StringRef opName) {
  return opName == "boom.quantize" || opName == "boom.rms_norm_quantize" ||
         opName == "boom.gelu_mul_quantize" ||
         opName == "boom.silu_mul_quantize" ||
         opName == "boom.fused_region";
}

// The ops that carry `ds`: the mmq matmuls it selects the activation layout for,
// and the fused quantizers that produce that layout.
bool carriesDs(llvm::StringRef opName) {
  return opName == "boom.quantized_matmul" ||
         opName == "boom.quantized_matmul_q8" ||
         opName == "boom.rms_norm_quantize" ||
         opName == "boom.gelu_mul_quantize" ||
         opName == "boom.silu_mul_quantize";
}

// Elementwise ops whose result has the shape of their first operand. `add_bias`
// is included deliberately: it adds a [width] bias to a [.., width] activation,
// so the broadcast lands on the later dimensions and the result keeps the
// activation's shape. Ops that change shape (argmax collapses, reshape, the
// gathers, the matmuls) are absent.
bool isShapePreserving(llvm::StringRef opName) {
  return opName == "boom.relu" || opName == "boom.mul" ||
         opName == "boom.scale" || opName == "boom.gelu_fp16" ||
         opName == "boom.silu" || opName == "boom.cast_fp16" ||
         opName == "boom.softcap" || opName == "boom.add_bias" ||
         opName == "boom.add";
}

bool isAllSameShapeOp(llvm::StringRef opName) {
  return opName == "boom.add" || opName == "boom.mul";
}

// The logical dimensions of any Boom value type. Every one of them carries a
// shape, but they are separate TypeDefs (not builtin ShapedTypes), so this is
// how the shape is reached without a cast that would be invalid for the
// quantized ones.
llvm::ArrayRef<int64_t> boomValueShape(Type type) {
  return llvm::TypeSwitch<Type, llvm::ArrayRef<int64_t>>(type)
      .Case<TensorType, Q3KType, Q4KType, Q5KType, Q6KType, Q5_0Type, Q8_0Type,
            Q8_1Type>([](auto shaped) { return shaped.getShape(); })
      .Default([](Type) { return llvm::ArrayRef<int64_t>(); });
}

// A tensors id -> its table entry, keyed by the id the table has been checked to
// define exactly once.
using TensorTable = llvm::DenseMap<int64_t, TensorDescAttr>;

// "(slot = 3 : i64, rows = 4 : i64)" — for the attribute-mismatch message.
std::string describeAttrs(llvm::ArrayRef<NamedAttribute> attrs) {
  std::string text;
  llvm::raw_string_ostream os(text);
  llvm::interleaveComma(attrs, os, [&](const NamedAttribute &a) {
    os << a.getName() << " = " << a.getValue();
  });
  os.flush();
  return text;
}

} // namespace

//--- boom.graph --------------------------------------------------------------

mlir::LogicalResult GraphOp::verify() {
  const llvm::StringRef graphName = getGraphName();

  // 1. The tensor table: every entry is a #boom.tensor_desc and the ids are
  //    unique, so an id identifies exactly one tensor.
  TensorTable table;
  for (Attribute entry : getTensors()) {
    auto desc = llvm::dyn_cast<TensorDescAttr>(entry);
    if (!desc)
      return emitOpError() << "graph '" << graphName
                           << "': tensor table entry " << entry
                           << " is not a #boom.tensor_desc";
    if (!table.try_emplace(desc.getId(), desc).second)
      return emitOpError() << "graph '" << graphName << "': tensor id "
                           << desc.getId() << " appears twice in the table";
  }

  auto lookup = [&](int64_t id) -> std::optional<TensorDescAttr> {
    auto it = table.find(id);
    if (it == table.end())
      return std::nullopt;
    return it->second;
  };

  // 2. The graph's input/output id lists name tensors in the table, and those
  //    tensors carry the matching flag. That flag is what the memory planner
  //    reads (planner/memory_planner.cpp:103), so a list entry without it would
  //    be a disagreement about whether the buffer is externally owned.
  for (int64_t id : getInputs()) {
    auto desc = lookup(id);
    if (!desc)
      return emitOpError() << "graph '" << graphName << "': input tensor id "
                           << id << " is not in the tensor table";
    if (!desc->getIsInput())
      return emitOpError()
             << "graph '" << graphName << "': id " << id
             << " is listed as a graph input but its table entry does not set "
                "is_input";
  }
  for (int64_t id : getOutputs()) {
    auto desc = lookup(id);
    if (!desc)
      return emitOpError() << "graph '" << graphName << "': output tensor id "
                           << id << " is not in the tensor table";
    if (!desc->getIsOutput())
      return emitOpError()
             << "graph '" << graphName << "': id " << id
             << " is listed as a graph output but its table entry does not set "
                "is_output";
  }

  mlir::Block &entry = getBody().front();

  // A value's type against its tensor table entry. This is the shape/dtype
  // consistency check; the entry is authoritative for everything a type cannot
  // say.
  auto checkAgainstTable = [&](Type type, int64_t id,
                               const TensorDescAttr &desc) -> mlir::LogicalResult {
    if (auto tensor = llvm::dyn_cast<TensorType>(type)) {
      if (tensor.getShape() != desc.getShape())
        return emitOpError() << "graph '" << graphName << "': tensor id " << id
                             << " is typed " << type
                             << " but the tensor table records the shape "
                             << desc.getShape();
      if (tensor.getDtype() != desc.getDtype())
        return emitOpError() << "graph '" << graphName << "': tensor id " << id
                             << " is typed " << type
                             << " but the tensor table records the dtype "
                             << desc.getDtype();
      return mlir::success();
    }
    // A quantized layout. Its table entry describes the buffer that holds it,
    // whose element type is float32 and whose shape is the logical shape (see
    // quantize_once_pass, planner/rewrites.cpp:111-120: the q8_1 tensor is
    // declared as an f32 intermediate of the input's shape and only the CUDA
    // kernels interpret the bytes as q8_1).
    if (!isBoomValueType(type))
      return emitOpError() << "graph '" << graphName << "': tensor id " << id
                           << " has the type " << type
                           << ", which is not a Boom value type";
    if (desc.getDtype() != "float32")
      return emitOpError() << "graph '" << graphName << "': tensor id " << id
                           << " holds a " << boomQuantFormatName(type)
                           << " layout, but its tensor table entry has dtype "
                           << desc.getDtype() << " (expected float32)";
    if (desc.getShape() != boomValueShape(type))
      return emitOpError() << "graph '" << graphName << "': tensor id " << id
                           << " holds a " << boomQuantFormatName(type)
                           << " layout of shape " << boomValueShape(type)
                           << ", but the tensor table records the shape "
                           << desc.getShape();
    return mlir::success();
  };

  // 3. The nodes' results, first: a tensor produced by a node cannot be a free
  //    variable, so the block arguments cannot be checked until every
  //    `result_ids` has been read.
  llvm::DenseSet<int64_t> produced;
  for (Operation &op : entry) {
    if (op.getName().getDialectNamespace() != "boom")
      return op.emitOpError() << "graph '" << graphName
                              << "' contains a non-Boom operation";
    auto resultIds = op.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    if (!resultIds)
      return op.emitOpError() << "has no result_ids";
    if (resultIds.size() != op.getNumResults())
      return op.emitOpError()
             << "has " << op.getNumResults() << " results but result_ids lists "
             << resultIds.size() << " tensor ids";
    for (int64_t id : resultIds.asArrayRef()) {
      if (!table.count(id))
        return op.emitOpError() << "result id " << id
                                << " is not in the tensor table";
      if (!produced.insert(id).second)
        return op.emitOpError() << "tensor id " << id
                                << " is a result of two nodes";
    }
  }

  // 4. The block arguments are the free variables in ascending id order.
  llvm::SmallVector<int64_t> freeIds;
  for (const auto &[id, desc] : table)
    if (!produced.count(id))
      freeIds.push_back(id);
  llvm::sort(freeIds);

  if (entry.getNumArguments() != freeIds.size())
    return emitOpError() << "graph '" << graphName << "': the body has "
                         << entry.getNumArguments() << " block arguments but "
                         << freeIds.size()
                         << " tensors in the table are not produced by a node";

  llvm::DenseMap<Value, int64_t> tensorOf;
  for (auto [index, id] : llvm::enumerate(freeIds)) {
    Value arg = entry.getArgument(index);
    if (llvm::failed(checkAgainstTable(arg.getType(), id, table[id])))
      return mlir::failure();
    tensorOf[arg] = id;
  }

  // A graph input is fed from outside, so no node may produce it.
  for (int64_t id : getInputs())
    if (produced.count(id))
      return emitOpError() << "graph '" << graphName << "': tensor id " << id
                           << " is a graph input but is also a node's result";

  // 5. Walk the nodes in body order: operands, attributes, quant, shapes.
  for (Operation &op : entry) {
    const llvm::StringRef opName =
        op.getName().getStringRef().drop_front(llvm::StringRef("boom.").size());

    // 5a. Operands resolve to tensor ids, and their types agree with the table.
    for (Value operand : op.getOperands()) {
      auto it = tensorOf.find(operand);
      if (it == tensorOf.end())
        return op.emitOpError()
               << "reads a value that is neither a block argument nor a result "
                  "of an earlier node";
      if (llvm::failed(checkAgainstTable(operand.getType(), it->second,
                                         table[it->second])))
        return mlir::failure();
    }

    // 5b. `node_attrs` is an ordered list of #boom.attr with unique names, and
    //     the typed attributes are exactly its int/float subset.
    llvm::SmallVector<NamedAttribute> recorded;
    for (Attribute entryAttr : op.getAttrOfType<mlir::ArrayAttr>("node_attrs")) {
      auto nodeAttr = llvm::dyn_cast<NodeAttr>(entryAttr);
      if (!nodeAttr)
        return op.emitOpError() << "node_attrs entry " << entryAttr
                                << " is not a #boom.attr";
      for (const NamedAttribute &prior : recorded)
        if (prior.getName() == nodeAttr.getName())
          return op.emitOpError()
                 << "node_attrs carries '" << nodeAttr.getName() << "' twice";
      recorded.push_back(
          NamedAttribute(mlir::StringAttr::get(op.getContext(),
                                               nodeAttr.getName()),
                         nodeAttr.getValue()));
    }

    llvm::SmallVector<NamedAttribute> expected;
    for (const NamedAttribute &attr : recorded) {
      if (!isBoomTypedAttrName(attr.getName()))
        continue;
      const bool isInteger = llvm::isa<mlir::IntegerAttr>(attr.getValue());
      const bool isFloat = llvm::isa<mlir::FloatAttr>(attr.getValue());
      if (!isInteger && !isFloat)
        return op.emitOpError()
               << "attribute '" << attr.getName()
               << "' is in the dialect's typed vocabulary but its value "
               << attr.getValue() << " is neither an integer nor a float";
      expected.push_back(attr);
    }

    llvm::SmallVector<NamedAttribute> typed = typedNodeAttributes(&op);
    auto byName = [](llvm::SmallVector<NamedAttribute> attrs) {
      llvm::sort(attrs, [](const NamedAttribute &a, const NamedAttribute &b) {
        return a.getName().getValue() < b.getName().getValue();
      });
      return attrs;
    };
    if (byName(expected) != byName(typed))
      return op.emitOpError()
             << "the typed attributes (" << describeAttrs(typed)
             << ") are not the int/float subset of node_attrs ("
             << describeAttrs(expected) << ")";

    // 5c. `ds`: only where a pass reads it, and only as one of the three mmq
    //     activation layouts (rewrites.cpp:564-568, gemma4_plan.cpp:564-568).
    if (auto ds = op.getAttrOfType<mlir::IntegerAttr>("ds")) {
      if (!carriesDs(op.getName().getStringRef()))
        return op.emitOpError() << "carries `ds`, but no `ds`-dependent pass "
                                   "reads it on this op";
      const int64_t value = ds.getInt();
      if (value != -1 && value != 0 && value != 1)
        return op.emitOpError()
               << "has ds = " << value
               << ", which is not one of the three mmq activation layouts "
                  "(-1 none, 0 the 32-element formats, 1 the K-quants)";
    }

    // 5d. Quantized values: one consumer, one kind of producer.
    const bool producesQ8_1 = isQ8_1Producer(op.getName().getStringRef());
    for (Value operand : op.getOperands()) {
      if (llvm::isa<TensorType>(operand.getType()))
        continue;
      // The one op whose point is a pre-quantized activation.
      if (opName == "quantized_matmul_q8" && operand == op.getOperand(0))
        continue;
      return op.emitOpError()
             << "reads a " << boomQuantFormatName(operand.getType())
             << " value; only boom.quantized_matmul_q8's first operand may be "
                "quantized";
    }
    for (auto [index, type] : llvm::enumerate(op.getResultTypes())) {
      bool consumedAsQ8_1 = false;
      for (Operation *user : op.getResult(index).getUsers())
        if (user->getName().getStringRef() == "boom.quantized_matmul_q8" &&
            user->getOperand(0) == op.getResult(index))
          consumedAsQ8_1 = true;

      if (consumedAsQ8_1 && !llvm::isa<Q8_1Type>(type))
        return op.emitOpError()
               << "result " << index << " feeds a boom.quantized_matmul_q8 "
               << "but is typed " << type << " instead of !boom.q8_1";
      if (consumedAsQ8_1 && !producesQ8_1)
        return op.emitOpError() << "produces the q8_1 activation a "
                                   "boom.quantized_matmul_q8 consumes, but is "
                                   "not an activation quantizer";
      if (!llvm::isa<TensorType>(type) && !producesQ8_1)
        return op.emitOpError()
               << "produces a " << boomQuantFormatName(type)
               << " value, which none of the ops that produce a quantized "
                  "activation is";
    }

    // 5e. Elementwise shapes.
    if (isShapePreserving(op.getName().getStringRef()) && op.getNumOperands() &&
        op.getNumResults() && boomValueShape(op.getOperand(0).getType()) !=
                                  boomValueShape(op.getResult(0).getType()))
      return op.emitOpError()
             << "is shape-preserving but its result is "
             << boomValueShape(op.getResult(0).getType())
             << " while its first operand is "
             << boomValueShape(op.getOperand(0).getType());
    if (isAllSameShapeOp(op.getName().getStringRef()))
      for (Value operand : llvm::drop_begin(op.getOperands()))
        if (boomValueShape(operand.getType()) !=
            boomValueShape(op.getOperand(0).getType()))
          return op.emitOpError()
                 << "has operands of different shapes ("
                 << boomValueShape(op.getOperand(0).getType()) << " vs "
                 << boomValueShape(operand.getType()) << ")";

    // 5f. Register this op's results, so the next op's operands resolve.
    auto resultIds = op.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef()))
      tensorOf[op.getResult(index)] = id;
  }

  // 6. The graph's outputs are produced by this graph (or are its inputs).
  for (int64_t id : getOutputs())
    if (!llvm::is_contained(getInputs(), id) && !produced.count(id))
      return emitOpError() << "graph '" << graphName << "': output tensor id "
                           << id << " is produced by no node";

  return mlir::success();
}

} // namespace sonicboom::compiler::boom
