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
//     the graph references but does not produce, in ascending id order — each
//     agreeing with its tensor's table entry, followed by one `!boom.kv` handle
//     per cache layer in ascending layer order;
//   * the three cache-touching ops carry their layer's handle as the last
//     operand, and no other op carries one;
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
#include "mlir/Dialect/SCF/IR/SCF.h"

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
         opName == "boom.silu_mul_quantize" ||
         // The prefill add+norm+quantize region carries the norm's ds.
         opName == "boom.fused_region";
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

// The ops whose cache access is stated on the KVCacheResource (BoomDialect.td):
// they carry the layer's `!boom.kv` handle as their last operand.
bool carriesKvHandle(llvm::StringRef opName) {
  return opName == "boom.attention" || opName == "boom.flash_attention" ||
         opName == "boom.kv_append";
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

  // 4. The block arguments are the free variables in ascending id order,
  //    followed by one `!boom.kv` handle per cache layer the graph touches, in
  //    ascending layer order. A handle is not a tensor — it has no table entry
  //    — so a block argument is the only place it can be defined.
  llvm::SmallVector<int64_t> freeIds;
  for (const auto &[id, desc] : table)
    if (!produced.count(id))
      freeIds.push_back(id);
  llvm::sort(freeIds);

  if (entry.getNumArguments() < freeIds.size())
    return emitOpError() << "graph '" << graphName << "': the body has "
                         << entry.getNumArguments() << " block arguments but "
                         << freeIds.size()
                         << " tensors in the table are not produced by a node";

  llvm::DenseMap<Value, int64_t> tensorOf;
  for (auto [index, id] : llvm::enumerate(freeIds)) {
    Value arg = entry.getArgument(index);
    if (llvm::isa<KvType>(arg.getType()))
      return emitOpError()
             << "graph '" << graphName << "': block argument " << index
             << " is a KV cache handle, but the first " << freeIds.size()
             << " arguments must be the graph's free tensor variables";
    if (llvm::failed(checkAgainstTable(arg.getType(), id, table[id])))
      return mlir::failure();
    tensorOf[arg] = id;
  }

  llvm::DenseMap<Value, int64_t> kvLayerOf;
  int64_t previousLayer = -1;
  for (size_t index = freeIds.size(); index < entry.getNumArguments();
       ++index) {
    Value arg = entry.getArgument(index);
    auto kv = llvm::dyn_cast<KvType>(arg.getType());
    if (!kv)
      return emitOpError()
             << "graph '" << graphName << "': block argument " << index
             << " is typed " << arg.getType() << ", but every argument after "
             << "the " << freeIds.size()
             << " free tensor variables must be a !boom.kv handle";
    if (kv.getLayer() <= previousLayer)
      return emitOpError()
             << "graph '" << graphName << "': the KV cache handles are in "
             << "strict ascending layer order, but layer " << kv.getLayer()
             << " follows layer " << previousLayer;
    previousLayer = kv.getLayer();
    kvLayerOf[arg] = kv.getLayer();
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
    //     A `!boom.kv` operand is the cache handle: only the three
    //     cache-touching ops may carry one, it is their last operand, and it
    //     names exactly the layer the op's attributes select.
    bool sawKvHandle = false;
    for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
      if (auto kv = llvm::dyn_cast<KvType>(operand.getType())) {
        if (!carriesKvHandle(op.getName().getStringRef()))
          return op.emitOpError()
                 << "reads a " << kv << " cache handle, but it does not access "
                    "the KV cache";
        if (sawKvHandle)
          return op.emitOpError() << "reads more than one KV cache handle";
        if (index + 1 != op.getNumOperands())
          return op.emitOpError()
                 << "carries its KV cache handle at operand " << index
                 << ", but it must be the last operand";
        sawKvHandle = true;
        auto layerIt = kvLayerOf.find(operand);
        if (layerIt == kvLayerOf.end())
          return op.emitOpError()
                 << "reads a KV cache handle that is not one of the graph's "
                    "block arguments";
        auto sourceAttr = op.getAttrOfType<mlir::IntegerAttr>("kv_source_layer");
        const llvm::StringRef layerAttr = sourceAttr ? "kv_source_layer" : "layer";
        auto named = sourceAttr ? sourceAttr
                                : op.getAttrOfType<mlir::IntegerAttr>("layer");
        if (!named)
          return op.emitOpError()
                 << "reads a KV cache handle without a `layer` or "
                    "`kv_source_layer` attribute to check it against";
        if (layerIt->second != named.getInt())
          return op.emitOpError()
                 << "reads layer " << layerIt->second
                 << "'s KV cache handle, but its `" << layerAttr
                 << "` attribute names layer " << named.getInt();
        continue;
      }
      auto it = tensorOf.find(operand);
      if (it == tensorOf.end())
        return op.emitOpError()
               << "reads a value that is neither a block argument nor a result "
                  "of an earlier node";
      if (llvm::failed(checkAgainstTable(operand.getType(), it->second,
                                         table[it->second])))
        return mlir::failure();
    }
    if (carriesKvHandle(op.getName().getStringRef()) && !sawKvHandle)
      return op.emitOpError()
             << "accesses the KV cache but carries no !boom.kv handle operand";

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
      // The cache handle, checked in 5a.
      if (llvm::isa<KvType>(operand.getType()))
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

//--- the generation program (§10) -------------------------------------------

namespace {

// A value's type against its tensor table entry, for the step's own table. The
// graph verifier states the same rule; this is the same check without the
// graph's free-variable machinery, reported with the step's diagnostics.
mlir::LogicalResult checkStepValue(Operation *step, int64_t id, Type type,
                                   const TensorDescAttr &desc) {
  if (auto tensor = llvm::dyn_cast<TensorType>(type)) {
    if (tensor.getShape() != desc.getShape())
      return step->emitOpError()
             << "tensor id " << id << " is typed " << type
             << " but the tensor table records the shape " << desc.getShape();
    if (tensor.getDtype() != desc.getDtype())
      return step->emitOpError()
             << "tensor id " << id << " is typed " << type
             << " but the tensor table records the dtype " << desc.getDtype();
    return mlir::success();
  }
  if (!isBoomValueType(type))
    return step->emitOpError() << "tensor id " << id << " has the type " << type
                               << ", which is not a Boom value type";
  return mlir::success();
}

bool isArithOp(Operation &op) {
  return op.getName().getDialectNamespace() == "arith";
}

} // namespace

// One decode step: the decode graph's node list inside a region, with the
// position/token/cache as operands instead of an isolated graph's free
// variables. The node-level attribute and quant rules are the graph verifier's;
// a step is checked here for the table/SSA structure the execution needs.
mlir::LogicalResult StepOp::verify() {
  // 1. The tensor table: unique ids, every entry a #boom.tensor_desc.
  TensorTable table;
  for (Attribute entry : getTensors()) {
    auto desc = llvm::dyn_cast<TensorDescAttr>(entry);
    if (!desc)
      return emitOpError() << "tensor table entry " << entry
                           << " is not a #boom.tensor_desc";
    if (!table.try_emplace(desc.getId(), desc).second)
      return emitOpError() << "tensor id " << desc.getId()
                           << " appears twice in the table";
  }
  auto lookup = [&](int64_t id) -> std::optional<TensorDescAttr> {
    auto it = table.find(id);
    if (it == table.end())
      return std::nullopt;
    return it->second;
  };

  // 2. The input/output id lists name table entries that carry the matching
  //    flag (the memory planner reads those flags).
  for (int64_t id : getInputIds()) {
    auto desc = lookup(id);
    if (!desc)
      return emitOpError() << "input tensor id " << id
                           << " is not in the tensor table";
    if (!desc->getIsInput())
      return emitOpError() << "id " << id
                           << " is listed as an input but its table entry does "
                              "not set is_input";
  }
  for (int64_t id : getOutputIds()) {
    auto desc = lookup(id);
    if (!desc)
      return emitOpError() << "output tensor id " << id
                           << " is not in the tensor table";
    if (!desc->getIsOutput())
      return emitOpError() << "id " << id
                           << " is listed as an output but its table entry does "
                              "not set is_output";
  }

  // 3. One result per output, typed as the output's table entry.
  if (getNumResults() != getOutputIds().size())
    return emitOpError() << "has " << getNumResults() << " results but "
                         << getOutputIds().size() << " outputs";
  for (auto [index, id] : llvm::enumerate(getOutputIds()))
    if (llvm::failed(checkStepValue(getOperation(), id, getResult(index).getType(),
                                    *lookup(id))))
      return mlir::failure();

  if (!llvm::hasSingleElement(getBody()))
    return emitOpError() << "must have exactly one block";
  mlir::Block &block = getBody().front();

  // 4. The body's block arguments repeat the graph convention: the free tensor
  //    variables in ascending id order, then one handle per cache layer in
  //    ascending layer order. The block arguments and the step's operands are
  //    the same values — the step is not an isolated graph.
  llvm::DenseSet<int64_t> produced;
  for (Operation &op : block) {
    if (op.getName().getDialectNamespace() != "boom")
      return op.emitOpError() << "step contains a non-Boom operation";
    auto resultIds = op.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    if (!resultIds)
      return op.emitOpError() << "has no result_ids";
    if (resultIds.size() != op.getNumResults())
      return op.emitOpError() << "has " << op.getNumResults()
                              << " results but result_ids lists "
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

  llvm::SmallVector<int64_t> freeIds;
  for (const auto &[id, desc] : table)
    if (!produced.count(id))
      freeIds.push_back(id);
  llvm::sort(freeIds);

  if (block.getNumArguments() < freeIds.size())
    return emitOpError() << "the body has " << block.getNumArguments()
                         << " block arguments but " << freeIds.size()
                         << " tensors in the table are not produced by a node";
  if (block.getNumArguments() != getNumOperands())
    return emitOpError()
           << "has " << getNumOperands() << " operands but its body has "
           << block.getNumArguments()
           << " block arguments; a step's operands are its block arguments";

  llvm::DenseMap<Value, int64_t> tensorOf;
  for (auto [index, id] : llvm::enumerate(freeIds)) {
    Value arg = block.getArgument(index);
    if (llvm::isa<KvType>(arg.getType()))
      return emitOpError() << "block argument " << index
                           << " is a KV cache handle, but the first "
                           << freeIds.size()
                           << " arguments must be the free tensor variables";
    if (llvm::failed(checkStepValue(getOperation(), id, arg.getType(), table[id])))
      return mlir::failure();
    tensorOf[arg] = id;
  }

  llvm::DenseMap<Value, int64_t> kvLayerOf;
  int64_t previousLayer = -1;
  for (size_t index = freeIds.size(); index < block.getNumArguments();
       ++index) {
    Value arg = block.getArgument(index);
    auto kv = llvm::dyn_cast<KvType>(arg.getType());
    if (!kv)
      return emitOpError() << "block argument " << index << " is typed "
                           << arg.getType() << ", but every argument after the "
                           << freeIds.size()
                           << " free tensor variables must be a !boom.kv handle";
    if (kv.getLayer() <= previousLayer)
      return emitOpError() << "the KV cache handles are in strict ascending "
                              "layer order, but layer "
                           << kv.getLayer() << " follows layer "
                           << previousLayer;
    previousLayer = kv.getLayer();
    kvLayerOf[arg] = kv.getLayer();
  }

  // The operands and the block arguments agree, one for one: a step's body is
  // bound to the loop's SSA values, not to anything outside.
  for (auto [index, arg] : llvm::enumerate(block.getArguments()))
    if (arg.getType() != getOperand(index).getType())
      return emitOpError() << "block argument " << index << " is typed "
                           << arg.getType() << " but operand " << index
                           << " is typed " << getOperand(index).getType();

  // 5. Walk the nodes: every operand resolves to a block argument or an earlier
  //    node's result, and a handle operand obeys the cache-op rule.
  for (Operation &op : block) {
    const llvm::StringRef opName = op.getName().getStringRef();
    bool sawKvHandle = false;
    for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
      if (auto kv = llvm::dyn_cast<KvType>(operand.getType())) {
        if (!carriesKvHandle(opName))
          return op.emitOpError()
                 << "reads a " << kv
                 << " cache handle, but it does not access the KV cache";
        if (sawKvHandle)
          return op.emitOpError() << "reads more than one KV cache handle";
        if (index + 1 != op.getNumOperands())
          return op.emitOpError() << "carries its KV cache handle at operand "
                                  << index << ", but it must be the last operand";
        sawKvHandle = true;
        auto layerIt = kvLayerOf.find(operand);
        if (layerIt == kvLayerOf.end())
          return op.emitOpError()
                 << "reads a KV cache handle that is not one of the step's "
                    "block arguments";
        auto sourceAttr = op.getAttrOfType<mlir::IntegerAttr>("kv_source_layer");
        auto named = sourceAttr ? sourceAttr
                                : op.getAttrOfType<mlir::IntegerAttr>("layer");
        if (!named)
          return op.emitOpError()
                 << "reads a KV cache handle without a `layer` or "
                    "`kv_source_layer` attribute to check it against";
        if (layerIt->second != named.getInt())
          return op.emitOpError() << "reads layer " << layerIt->second
                                  << "'s KV cache handle, but its attribute "
                                     "names layer "
                                  << named.getInt();
        continue;
      }
      auto it = tensorOf.find(operand);
      if (it == tensorOf.end())
        return op.emitOpError()
               << "reads a value that is neither a block argument nor a result "
                  "of an earlier node";
      if (llvm::failed(checkStepValue(&op, it->second, operand.getType(),
                                      table[it->second])))
        return mlir::failure();
    }
    if (carriesKvHandle(opName) && !sawKvHandle)
      return op.emitOpError()
             << "accesses the KV cache but carries no !boom.kv handle operand";

    auto resultIds = op.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef()))
      tensorOf[op.getResult(index)] = id;
  }

  // 6. The outputs are computed by this step: a loop that yields its own input
  //    would be a decode step that decoded nothing.
  for (int64_t id : getOutputIds())
    if (!produced.count(id))
      return emitOpError() << "output tensor id " << id
                           << " is produced by no node";

  return mlir::success();
}

// The prefill call: value operands then the cache handles it fills.
mlir::LogicalResult PrefillOp::verify() {
  if (getGraphName().empty())
    return emitOpError() << "names no prefill graph";
  bool sawKvHandle = false;
  int64_t previousLayer = -1;
  for (auto [index, operand] : llvm::enumerate(getInputs())) {
    if (auto kv = llvm::dyn_cast<KvType>(operand.getType())) {
      if (kv.getLayer() <= previousLayer)
        return emitOpError() << "the KV cache handles are in strict ascending "
                                "layer order, but layer "
                             << kv.getLayer() << " follows layer "
                             << previousLayer;
      previousLayer = kv.getLayer();
      sawKvHandle = true;
      continue;
    }
    if (!isBoomValueType(operand.getType()))
      return emitOpError() << "operand " << index << " is typed "
                           << operand.getType() << ", which is neither a Boom "
                           << "value nor a KV cache handle";
    if (sawKvHandle)
      return emitOpError()
             << "operand " << index
             << " is a value after a KV cache handle; the handles come last";
  }
  return mlir::success();
}

// The sampling op: the logits and the chosen token are Boom values, and the
// sampling attributes (when present) are in range. The result's shape is the
// sampler's — the argmax collapses it — so only the value kind is checked.
mlir::LogicalResult SampleOp::verify() {
  if (!isBoomValueType(getLogits().getType()))
    return emitOpError() << "logits are typed " << getLogits().getType()
                         << ", which is not a Boom value";
  if (!isBoomValueType(getToken().getType()))
    return emitOpError() << "the token is typed " << getToken().getType()
                         << ", which is not a Boom value";
  if (auto t = getTemperature())
    if (!(t->convertToDouble() > 0.0))
      return emitOpError() << "temperature must be positive";
  if (auto k = getTopK())
    if (*k == 0)
      return emitOpError() << "top_k must be positive";
  if (auto p = getTopP())
    if (!(p->convertToDouble() > 0.0) || p->convertToDouble() > 1.0)
      return emitOpError() << "top_p must be in (0, 1]";
  return mlir::success();
}

// The program: a prefill call, then a while loop carrying (token, position,
// emitted count) whose body is exactly one step, one sample, one emit and one
// stop check, and whose continuation advances the position.
mlir::LogicalResult ProgramOp::verify() {
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError() << "must have exactly one block";
  mlir::Block &block = getBody().front();

  // The body's arguments: Boom values, then the cache handles in ascending
  // layer order (the same convention the graph body follows).
  bool sawKvHandle = false;
  int64_t previousLayer = -1;
  for (auto [index, arg] : llvm::enumerate(block.getArguments())) {
    if (auto kv = llvm::dyn_cast<KvType>(arg.getType())) {
      if (kv.getLayer() <= previousLayer)
        return emitOpError() << "the KV cache handles are in strict ascending "
                                "layer order, but layer "
                             << kv.getLayer() << " follows layer "
                             << previousLayer;
      previousLayer = kv.getLayer();
      sawKvHandle = true;
      continue;
    }
    if (!isBoomValueType(arg.getType()))
      return emitOpError() << "block argument " << index << " is typed "
                           << arg.getType()
                           << ", which is neither a Boom value nor a KV cache "
                              "handle";
    if (sawKvHandle)
      return emitOpError() << "block argument " << index
                           << " is a value after a KV cache handle; the "
                              "handles come last";
  }

  Operation *prefill = nullptr;
  mlir::scf::WhileOp loop;
  bool sawLoop = false;
  for (Operation &op : block) {
    if (auto p = llvm::dyn_cast<PrefillOp>(&op)) {
      if (prefill)
        return emitOpError() << "holds more than one boom.prefill";
      if (sawLoop)
        return emitOpError() << "its boom.prefill follows the decode loop";
      prefill = &op;
      continue;
    }
    if (auto w = llvm::dyn_cast<mlir::scf::WhileOp>(&op)) {
      if (sawLoop)
        return emitOpError() << "holds more than one decode loop";
      sawLoop = true;
      loop = w;
      continue;
    }
    // The loop's initial counter, and whatever else arith state the program
    // threads into it.
    if (isArithOp(op))
      continue;
    return op.emitOpError()
           << "a boom.program body holds one boom.prefill and one scf.while";
  }
  if (!prefill)
    return emitOpError() << "holds no boom.prefill";
  if (!loop)
    return emitOpError() << "holds no decode loop (scf.while)";

  if (!llvm::hasSingleElement(loop.getBefore()) ||
      !llvm::hasSingleElement(loop.getAfter()))
    return loop.emitOpError() << "must have one block in each region";

  // The loop body: one step, then the sample/emit/stop-check chain; the
  // continuation advances the position.
  StepOp step;
  SampleOp sample;
  EmitOp emit;
  StopCheckOp stop;
  bool sawEmit = false;
  bool sawStop = false;
  llvm::DenseSet<int64_t> order;  // 0 step, 1 sample, 2 emit, 3 stop
  for (Operation &op : loop.getBeforeBody()->getOperations()) {
    if (auto s = llvm::dyn_cast<StepOp>(&op)) {
      if (step)
        return op.emitOpError() << "the loop body holds more than one step";
      step = s;
      order.insert(0);
    } else if (auto s = llvm::dyn_cast<SampleOp>(&op)) {
      if (sample)
        return op.emitOpError() << "the loop body holds more than one sample";
      sample = s;
      order.insert(1);
    } else if (auto e = llvm::dyn_cast<EmitOp>(&op)) {
      if (sawEmit)
        return op.emitOpError() << "the loop body holds more than one emit";
      sawEmit = true;
      emit = e;
      order.insert(2);
    } else if (auto s = llvm::dyn_cast<StopCheckOp>(&op)) {
      if (sawStop)
        return op.emitOpError() << "the loop body holds more than one stop check";
      sawStop = true;
      stop = s;
      order.insert(3);
    } else if (op.getName().getDialectNamespace() == "boom") {
      return op.emitOpError()
             << "a decode loop body holds a step, a sample, an emit and a stop "
                "check, and no other Boom op";
    }
  }
  if (!step)
    return loop.emitOpError() << "its body holds no boom.step";
  if (!sample)
    return loop.emitOpError() << "its body holds no boom.sample";
  if (!sawEmit)
    return loop.emitOpError() << "its body holds no boom.emit";
  if (!sawStop)
    return loop.emitOpError() << "its body holds no boom.stop_check";
  for (int64_t needed : {int64_t(1), int64_t(2), int64_t(3)})
    if (!order.count(needed))
      return loop.emitOpError()
             << "the step, sample, emit and stop check are out of order";

  bool sawPosInc = false;
  for (Operation &op : loop.getAfterBody()->getOperations()) {
    if (llvm::isa<PosIncOp>(&op))
      sawPosInc = true;
    else if (op.getName().getDialectNamespace() == "boom" &&
             !llvm::isa<mlir::scf::YieldOp>(&op))
      return op.emitOpError()
             << "a decode loop's continuation advances the position and yields "
                "the next iteration";
  }
  if (!sawPosInc)
    return loop.emitOpError()
           << "its continuation does not advance the position (boom.pos_inc)";

  return mlir::success();
}

} // namespace sonicboom::compiler::boom
