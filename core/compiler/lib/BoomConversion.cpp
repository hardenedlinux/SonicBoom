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

// planner::Graph <-> Boom IR.

#include "sonicboom/compiler/BoomConversion.h"

#include "sonicboom/compiler/BoomBuilder.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <optional>
#include <string>

namespace sonicboom::compiler {

namespace boom = sonicboom::compiler::boom;
using planner::BackendTag;
using planner::Graph;
using planner::GraphNodeDesc;
using planner::OpKind;
using planner::TensorDesc;
using planner::TensorId;

namespace {

//--- ids ---------------------------------------------------------------------

int64_t raw(TensorId id) { return int64_t(id.value); }
int64_t raw(planner::GraphNodeId id) { return int64_t(id.value); }

TensorId tensorId(int64_t id) { return TensorId{uint32_t(id)}; }
planner::GraphNodeId nodeId(int64_t id) {
  return planner::GraphNodeId{uint32_t(id)};
}

// The inverse of planner::backend_tag_name (planner/partition.cpp). Written here
// rather than next to it because this is its only caller; the round trip is what
// keeps the two in step.
std::optional<BackendTag> backendTagFromName(llvm::StringRef name) {
  if (name == "mlir")
    return BackendTag::Mlir;
  if (name == "native_torch")
    return BackendTag::NativeTorch;
  if (name == "sonic")
    return BackendTag::Sonic;
  return std::nullopt;
}

//--- value types -------------------------------------------------------------

// The tensor ids the graph declares as the mmq activation layout: the first
// operand of a node `op_consumes_q8_1` names (a quantized_matmul_q8 node, or
// the matmul_q8_tc node the prefill dispatch pass may turn it into). The graph
// itself says nothing more — a `quantize` result could in principle feed an f32
// path — so this is the whole rule, and it is the same one the graph verifier
// applies (BoomVerify.cpp). Both ops read the same two layouts (K-quant
// weights, q8_1 activation), so both keep the operand typed the same way, and
// `bladeToBoom` reads the rule from the same predicate.
llvm::DenseSet<int64_t> q8_1Ids(const Graph &g) {
  llvm::DenseSet<int64_t> ids;
  for (const GraphNodeDesc &n : g.nodes) {
    if (!op_consumes_q8_1(n.op) || n.inputs.empty())
      continue;
    ids.insert(raw(n.inputs.front()));
  }
  return ids;
}

bool isCacheOp(const GraphNodeDesc &n);

std::optional<int64_t> intAttr(llvm::ArrayRef<sx::Attribute> attrs,
                               llvm::StringRef name) {
  for (const sx::Attribute &a : attrs)
    if (llvm::StringRef(a.name) == name && a.kind() == sx::AttrValueKind::Int)
      return std::get<int64_t>(a.value);
  return std::nullopt;
}

// The ops whose cache access is a stated effect on the KVCacheResource
// (BoomDialect.td): they carry their layer's `!boom.kv` handle. The attention
// prep region is one exactly when it performs the append itself (`kv_store`,
// Stage 6d; BoomVerify.cpp states the same rule on the IR side).
bool isCacheOp(const GraphNodeDesc &n) {
  return n.op == OpKind::Attention || n.op == OpKind::FlashAttention ||
         n.op == OpKind::KvAppend ||
         (n.op == OpKind::FusedRegion &&
          intAttr(n.attributes, "kv_store").value_or(0) != 0);
}

// The cache layer a cache-touching node reads/writes: a shared-KV attention
// names its donor with `kv_source_layer`; every other cache op acts on its own
// `layer`.
std::optional<int64_t> cacheLayer(const GraphNodeDesc &n) {
  if (n.op == OpKind::Attention || n.op == OpKind::FlashAttention)
    if (auto donor = intAttr(n.attributes, "kv_source_layer"))
      return donor;
  return intAttr(n.attributes, "layer");
}

// A tensor's IR shape. A tensor whose leading dimension is the graph's dynamic
// extent states it as `?`; the lowering binds it back to a concrete extent.
std::vector<int64_t> irShape(const TensorDesc &t) {
  std::vector<int64_t> shape = t.shape;
  if (t.dynamic_stride_bytes != 0 && !shape.empty())
    shape.front() = mlir::ShapedType::kDynamic;
  return shape;
}

// The SSA type of a tensor: !boom.q8_1 when the graph declares it as the mmq
// activation layout, !boom.tensor<shape, dtype> otherwise. Empty on a tensor no
// Boom type can describe — an unknown dtype name, or a q8_1 shape that is not a
// whole number of 32-element blocks — after diagnosing it.
mlir::Type valueType(mlir::MLIRContext *context, mlir::Location loc,
                     const TensorDesc &t, llvm::ArrayRef<int64_t> shape,
                     bool is_q8_1) {
  if (is_q8_1) {
    auto type = boom::Q8_1Type::getChecked(
        mlir::detail::getDefaultDiagnosticEmitFn(loc), context, shape);
    if (!type)
      mlir::emitError(loc) << "tensor id " << raw(t.id) << " ('" << t.name
                           << "') is a q8_1 activation of shape " << shape
                           << ", which is not a whole number of 32-element "
                              "blocks";
    return type;
  }
  const llvm::StringRef dtypeName = sx::dtype_name(t.dtype);
  if (!boom::boomDTypeFromName(dtypeName)) {
    mlir::emitError(loc) << "tensor id " << raw(t.id) << " ('" << t.name
                         << "') has the dtype '" << dtypeName
                         << "', which is not one of the frozen sx::DType names";
    return {};
  }
  return boom::TensorType::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(loc), context, shape, dtypeName);
}

// Every id a graph lists, as the plain int64s the array attributes carry.
llvm::SmallVector<int64_t> flatten(llvm::ArrayRef<TensorId> ids) {
  llvm::SmallVector<int64_t> out;
  out.reserve(ids.size());
  for (TensorId id : ids)
    out.push_back(raw(id));
  return out;
}

} // namespace

//--- Graph -> Boom IR --------------------------------------------------------

mlir::OwningOpRef<mlir::ModuleOp> graphToBoom(const Graph &graph,
                                              mlir::MLIRContext *context) {
  const mlir::Location loc = mlir::UnknownLoc::get(context);
  mlir::OpBuilder builder(context);

  // The module is the graph's container, not a second level of meaning: the
  // graph op is IsolatedFromAbove, so it needs a scope to be isolated from.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  // The table entry of a dynamic tensor states its extent as `?` and records
  // the byte size at the extent the graph was planned for, plus the fixed part
  // of its affine size in that extent — the passes compare those sizes, and
  // they must see what the concrete graph at that extent carries. The lowering
  // rebinds the extent, so the value is a shape-function record, not a hard
  // bound.
  llvm::SmallVector<mlir::Attribute> table;
  table.reserve(graph.tensors.size());
  for (const TensorDesc &t : graph.tensors) {
    const std::vector<int64_t> shape = irShape(t);
    table.push_back(boom::TensorDescAttr::get(
        context, raw(t.id), shape, sx::dtype_name(t.dtype), t.size_bytes,
        t.dynamic_base_bytes, t.is_graph_input, t.is_graph_output,
        t.is_constant, t.is_external, t.name));
  }

  // The extent the dynamic entries were sized at is the concretization the
  // graph carries in its leading dimension. All dynamic tensors carry the same
  // one extent, so the first sighting defines it.
  int64_t build_extent = 0;
  for (const TensorDesc &t : graph.tensors) {
    if (t.dynamic_stride_bytes == 0 || t.shape.empty())
      continue;
    if (build_extent == 0)
      build_extent = t.shape.front();
    else if (t.shape.front() != build_extent) {
      mlir::emitError(loc)
          << "tensor '" << t.name << "' carries dynamic extent "
          << t.shape.front() << ", but the graph's other dynamic tensors carry "
          << build_extent;
      return {};
    }
  }
  auto graphOp = boom::GraphOp::create(
      builder, loc, builder.getStringAttr(graph.name), builder.getArrayAttr(table),
      builder.getDenseI64ArrayAttr(flatten(graph.inputs)),
      builder.getDenseI64ArrayAttr(flatten(graph.outputs)),
      builder.getI64IntegerAttr(build_extent));

  mlir::Block *body = builder.createBlock(&graphOp.getBody());

  // The free variables: every tensor no node produces, in ascending id order, as
  // the verifier recomputes them. A model-bound weight is one of these — it has
  // to be an operand, and with the graph isolated there is nothing outside to
  // name it.
  llvm::DenseSet<int64_t> produced;
  for (const GraphNodeDesc &n : graph.nodes)
    for (TensorId id : n.outputs)
      produced.insert(raw(id));

  llvm::DenseMap<int64_t, const TensorDesc *> byId;
  for (const TensorDesc &t : graph.tensors)
    byId[raw(t.id)] = &t;

  const llvm::DenseSet<int64_t> q8_1 = q8_1Ids(graph);

  llvm::SmallVector<int64_t> freeIds;
  for (const TensorDesc &t : graph.tensors)
    if (!produced.count(raw(t.id)))
      freeIds.push_back(raw(t.id));
  llvm::sort(freeIds);

  llvm::DenseMap<int64_t, mlir::Value> valueOf;
  for (int64_t id : freeIds) {
    const TensorDesc *t = byId.at(id);
    mlir::Type type = valueType(context, loc, *t, irShape(*t), q8_1.count(id));
    if (!type)
      return {};
    valueOf[id] = body->addArgument(type, loc);
  }

  // The cache handles, one per layer the graph touches, appended after the free
  // tensor variables in ascending layer order. A handle names backend state
  // (BoomDialect.td), so it is a block argument rather than a tensor-table
  // entry; the cache itself stays keyed by layer in the backend.
  llvm::SmallVector<int64_t> cacheLayers;
  for (const GraphNodeDesc &n : graph.nodes) {
    if (!isCacheOp(n))
      continue;
    auto layer = cacheLayer(n);
    if (!layer) {
      mlir::emitError(loc) << "node '" << n.name << "' ("
                           << planner::op_kind_name(n.op)
                           << ") accesses the KV cache but carries neither a "
                              "`layer` nor a `kv_source_layer` attribute";
      return {};
    }
    cacheLayers.push_back(*layer);
  }
  llvm::sort(cacheLayers);
  cacheLayers.erase(llvm::unique(cacheLayers), cacheLayers.end());

  llvm::DenseMap<int64_t, mlir::Value> kvOf;
  for (int64_t layer : cacheLayers) {
    auto type = boom::KvType::getChecked(
        mlir::detail::getDefaultDiagnosticEmitFn(loc), context, layer);
    if (!type)
      return {};
    kvOf[layer] = body->addArgument(type, loc);
  }

  for (const GraphNodeDesc &n : graph.nodes) {
    llvm::SmallVector<mlir::Value> operands;
    for (TensorId in : n.inputs) {
      auto it = valueOf.find(raw(in));
      if (it == valueOf.end()) {
        mlir::emitError(loc)
            << "node '" << n.name << "' reads tensor id " << raw(in)
            << ", which no earlier node produces and no block argument holds";
        return {};
      }
      operands.push_back(it->second);
    }

    // The cache handle is the cache-touching op's last operand (BoomVerify.cpp
    // checks the placement and the layer).
    if (isCacheOp(n))
      operands.push_back(kvOf[*cacheLayer(n)]);

    llvm::SmallVector<mlir::Type> results;
    for (TensorId out : n.outputs) {
      auto it = byId.find(raw(out));
      if (it == byId.end()) {
        mlir::emitError(loc) << "node '" << n.name << "' produces tensor id "
                             << raw(out) << ", which is not in the table";
        return {};
      }
      mlir::Type type =
          valueType(context, loc, *it->second, irShape(*it->second),
                    q8_1.count(raw(out)));
      if (!type)
        return {};
      results.push_back(type);
    }

    // The node op record is built by the one implementation both this
    // conversion and the emitter's builder go through (BoomBuilder.h), so the
    // IR the runtime runs its passes on is the same whichever wrote it.
    mlir::Operation *node =
        createBoomNode(builder, loc, n.op, operands, results, n.name, raw(n.id),
                       flatten(n.outputs), n.attributes, n.backend);
    if (!node)
      return {};

    for (auto [index, id] : llvm::enumerate(n.outputs))
      valueOf[raw(id)] = node->getResult(index);
  }

  // The conversion is responsible for the IR it hands over: whatever it built
  // must satisfy the dialect's verifier, here rather than three passes later.
  if (mlir::failed(mlir::verify(module->getOperation())))
    return {};

  return module;
}

//--- Boom IR -> Graph --------------------------------------------------------

std::optional<Graph> boomToGraph(mlir::ModuleOp module,
                                 uint64_t dynamic_extent) {
  boom::GraphOp graphOp;
  for (boom::GraphOp candidate : module.getOps<boom::GraphOp>()) {
    if (graphOp) {
      module.emitError() << "the module holds more than one boom.graph op";
      return std::nullopt;
    }
    graphOp = candidate;
  }
  if (!graphOp) {
    module.emitError() << "the module holds no boom.graph op";
    return std::nullopt;
  }

  Graph graph;
  graph.name = graphOp.getGraphName().str();

  for (mlir::Attribute entry : graphOp.getTensors()) {
    auto desc = llvm::dyn_cast<boom::TensorDescAttr>(entry);
    if (!desc) {
      graphOp.emitError() << "tensor table entry " << entry
                          << " is not a #boom.tensor_desc";
      return std::nullopt;
    }
    const auto dtype = boom::boomDTypeFromName(desc.getDtype());
    if (!dtype) {
      graphOp.emitError() << "tensor id " << desc.getId() << " has the dtype '"
                          << desc.getDtype()
                          << "', which is not one of the frozen sx::DType names";
      return std::nullopt;
    }
    TensorDesc t;
    t.id = tensorId(desc.getId());
    t.shape.assign(desc.getShape().begin(), desc.getShape().end());
    t.dtype = *dtype;

    // Bind a dynamic extent. Only the leading dimension may be dynamic: the
    // graph's extent is one scalar (the segment length), and every activation
    // that carries it does so as row 0 of a position-major shape. Binding is
    // the shape function: a tensor's byte size is the bound byte size of its
    // static per-extent shape, and the per-unit stride is that divided by the
    // extent.
    bool dynamic = false;
    for (size_t axis = 0; axis < t.shape.size(); ++axis) {
      if (t.shape[axis] != mlir::ShapedType::kDynamic)
        continue;
      if (axis != 0) {
        graphOp.emitError()
            << "tensor id " << desc.getId()
            << " has a dynamic dimension at axis " << axis
            << ", but only the leading dimension is the graph's extent";
        return std::nullopt;
      }
      dynamic = true;
    }
    if (dynamic) {
      if (dynamic_extent == 0) {
        graphOp.emitError()
            << "tensor id " << desc.getId()
            << " has a dynamic leading dimension, but no extent was bound to "
               "the graph";
        return std::nullopt;
      }
      t.shape.front() = int64_t(dynamic_extent);
      // The table's byte size is the size at the extent the IR was sized at
      // (the graph op's `dynamic_extent`), not a hard bound, and it is
      // authoritative there: the planner passes rewrite the byte size of a
      // packed q8_1 activation to its packed count, which shape + dtype cannot
      // re-derive. The size is affine in the extent (the packed activation's
      // tail guard does not scale), so the table also records the fixed part:
      //
      //   size(N) = stride * N + base
      //
      // Binding evaluates that at the requested extent; a larger extent than
      // the sized one would overrun the buffers the plan reserved.
      const int64_t sized_i = graphOp.getDynamicExtent();
      const uint64_t sized = uint64_t(sized_i < 0 ? 0 : sized_i);
      if (sized == 0 || dynamic_extent > sized) {
        graphOp.emitError()
            << "tensor id " << desc.getId() << " ('" << desc.getName()
            << "') has a dynamic dimension to bind at extent " << dynamic_extent
            << ", but the IR records no sizing extent at or above it (sized "
            << sized << ")";
        return std::nullopt;
      }
      const uint64_t bound = desc.getByteSize();
      const uint64_t base = desc.getDynamicBaseBytes();
      if (bound <= base || (bound - base) % sized != 0) {
        graphOp.emitError()
            << "tensor id " << desc.getId() << " ('" << desc.getName()
            << "') has byte size " << bound << " at extent " << sized
            << " with a fixed part of " << base
            << " bytes, which is not a whole multiple of the extent";
        return std::nullopt;
      }
      const uint64_t stride = (bound - base) / sized;
      t.size_bytes = stride * dynamic_extent + base;
      t.dynamic_base_bytes = base;
      t.dynamic_stride_bytes = stride;
    } else {
      t.size_bytes = desc.getByteSize();
    }
    t.is_graph_input = desc.getIsInput();
    t.is_graph_output = desc.getIsOutput();
    t.is_constant = desc.getIsConstant();
    t.is_external = desc.getIsExternal();
    t.name = desc.getName().str();
    graph.tensors.push_back(std::move(t));
  }

  for (int64_t id : graphOp.getInputs())
    graph.inputs.push_back(tensorId(id));
  for (int64_t id : graphOp.getOutputs())
    graph.outputs.push_back(tensorId(id));

  mlir::Block &body = graphOp.getBody().front();

  // Values are identified the way the verifier identifies them: a node's results
  // by its `result_ids`, then the block arguments by the free variables the
  // tensor table implies.
  llvm::DenseSet<int64_t> produced;
  for (mlir::Operation &node : body) {
    auto resultIds = node.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    if (!resultIds) {
      node.emitError() << "has no result_ids";
      return std::nullopt;
    }
    for (int64_t id : resultIds.asArrayRef())
      produced.insert(id);
  }
  llvm::SmallVector<int64_t> freeIds;
  for (const TensorDesc &t : graph.tensors)
    if (!produced.count(raw(t.id)))
      freeIds.push_back(raw(t.id));
  llvm::sort(freeIds);
  if (body.getNumArguments() < freeIds.size()) {
    graphOp.emitError() << "the body has " << body.getNumArguments()
                        << " block arguments but " << freeIds.size()
                        << " tensors in the table are not produced by a node";
    return std::nullopt;
  }

  llvm::DenseMap<mlir::Value, int64_t> idOfValue;
  for (auto [index, id] : llvm::enumerate(freeIds))
    idOfValue[body.getArgument(index)] = id;
  // The trailing arguments are the cache handles. They name backend state, not
  // tensors, so the rebuilt graph carries that state in the ops' layer
  // attributes — the same graph the IR was converted from.
  for (size_t index = freeIds.size(); index < body.getNumArguments(); ++index)
    if (!llvm::isa<boom::KvType>(body.getArgument(unsigned(index)).getType())) {
      graphOp.emitError()
          << "block argument " << index << " is typed "
          << body.getArgument(unsigned(index)).getType()
          << ", but every argument after the free tensor variables must be a "
             "!boom.kv handle";
      return std::nullopt;
    }

  for (mlir::Operation &node : body) {
    GraphNodeDesc n;
    const auto kind = planner::op_kind_from_name(
        node.getName()
            .getStringRef()
            .drop_front(llvm::StringRef("boom.").size())
            .str());
    if (!kind) {
      node.emitError() << "is not one of the planner's operators";
      return std::nullopt;
    }
    n.op = *kind;

    auto nodeIdAttr = node.getAttrOfType<mlir::IntegerAttr>("node_id");
    auto nameAttr = node.getAttrOfType<mlir::StringAttr>("node_name");
    if (!nodeIdAttr || !nameAttr) {
      node.emitError() << "carries neither a node_id nor a node_name";
      return std::nullopt;
    }
    n.id = nodeId(nodeIdAttr.getInt());
    n.name = nameAttr.getValue().str();

    for (mlir::Value operand : node.getOperands()) {
      // The cache handle is backend state, not a graph tensor: the rebuilt node
      // names the layer with its attributes (checked by the dialect verifier).
      if (llvm::isa<boom::KvType>(operand.getType()))
        continue;
      auto it = idOfValue.find(operand);
      if (it == idOfValue.end()) {
        node.emitError() << "reads a value that no block argument holds and no "
                            "earlier node produces";
        return std::nullopt;
      }
      n.inputs.push_back(tensorId(it->second));
    }

    auto resultIds = node.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids");
    if (resultIds.size() != node.getNumResults()) {
      node.emitError() << "has " << node.getNumResults()
                       << " results but result_ids lists " << resultIds.size()
                       << " tensor ids";
      return std::nullopt;
    }
    for (int64_t id : resultIds.asArrayRef())
      n.outputs.push_back(tensorId(id));

    auto nodeAttrs = node.getAttrOfType<mlir::ArrayAttr>("node_attrs");
    if (!nodeAttrs) {
      node.emitError() << "has no node_attrs";
      return std::nullopt;
    }
    for (mlir::Attribute entry : nodeAttrs) {
      auto attr = llvm::dyn_cast<boom::NodeAttr>(entry);
      if (!attr) {
        node.emitError() << "node_attrs entry " << entry
                         << " is not a #boom.attr";
        return std::nullopt;
      }
      auto sxAttr = fromMlirAttribute(attr.getName(), attr.getValue());
      if (!sxAttr) {
        node.emitError() << "attribute '" << attr.getName() << "' has the value "
                         << attr.getValue()
                         << ", which sx::Attribute cannot carry";
        return std::nullopt;
      }
      n.attributes.push_back(std::move(*sxAttr));
    }

    if (auto backend = node.getAttrOfType<mlir::StringAttr>("backend")) {
      auto tag = backendTagFromName(backend.getValue());
      if (!tag) {
        node.emitError() << "names the unknown backend '" << backend.getValue()
                         << "'";
        return std::nullopt;
      }
      n.backend = *tag;
    }

    graph.nodes.push_back(std::move(n));

    for (auto [index, id] : llvm::enumerate(resultIds.asArrayRef()))
      idOfValue[node.getResult(index)] = id;
  }

  return graph;
}

//--- inspection --------------------------------------------------------------

std::string dumpBoom(mlir::Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  op->print(os);
  os.flush();
  return text;
}

} // namespace sonicboom::compiler
