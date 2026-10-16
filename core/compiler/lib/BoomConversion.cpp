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

#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"

#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>

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

//--- node attributes ---------------------------------------------------------

// sx::Attribute's value -> an MLIR attribute. Every kind sx::Attribute has maps
// to one MLIR attribute, so a node attribute is never dropped; the one corner is
// an empty vector, which cannot say which element type it is (neither could
// sx::Attribute, whose variant index is the only sign of it) and reads back as
// the int64 vector.
mlir::Attribute toMlirAttribute(mlir::MLIRContext *context,
                                const sx::Attribute &attr) {
  using K = sx::AttrValueKind;
  const auto &v = attr.value;
  switch (attr.kind()) {
  case K::Int:
    return mlir::IntegerAttr::get(
        mlir::IntegerType::get(context, 64), std::get<int64_t>(v));
  case K::Float:
    return mlir::FloatAttr::get(mlir::Float64Type::get(context),
                               std::get<double>(v));
  case K::String:
    return mlir::StringAttr::get(context, std::get<std::string>(v));
  case K::Bool:
    return mlir::BoolAttr::get(context, std::get<bool>(v));
  case K::Ints: {
    llvm::SmallVector<mlir::Attribute> items;
    for (int64_t x : std::get<std::vector<int64_t>>(v))
      items.push_back(mlir::IntegerAttr::get(
          mlir::IntegerType::get(context, 64), x));
    return mlir::ArrayAttr::get(context, items);
  }
  case K::Floats: {
    llvm::SmallVector<mlir::Attribute> items;
    for (double x : std::get<std::vector<double>>(v))
      items.push_back(
          mlir::FloatAttr::get(mlir::Float64Type::get(context), x));
    return mlir::ArrayAttr::get(context, items);
  }
  case K::Strings: {
    llvm::SmallVector<mlir::Attribute> items;
    for (const std::string &x : std::get<std::vector<std::string>>(v))
      items.push_back(mlir::StringAttr::get(context, x));
    return mlir::ArrayAttr::get(context, items);
  }
  case K::Bools: {
    llvm::SmallVector<mlir::Attribute> items;
    for (bool x : std::get<std::vector<bool>>(v))
      items.push_back(mlir::BoolAttr::get(context, x));
    return mlir::ArrayAttr::get(context, items);
  }
  }
  return {};
}

// The reverse. `name` is carried over unchanged: node_attrs is order-preserving,
// and the name is the key the typed attributes are matched against.
std::optional<sx::Attribute> fromMlirAttribute(mlir::StringRef name,
                                              mlir::Attribute value) {
  sx::Attribute out;
  out.name = name.str();

  if (auto i = llvm::dyn_cast<mlir::IntegerAttr>(value)) {
    out.value = i.getInt();
    return out;
  }
  if (auto f = llvm::dyn_cast<mlir::FloatAttr>(value)) {
    out.value = f.getValueAsDouble();
    return out;
  }
  if (auto s = llvm::dyn_cast<mlir::StringAttr>(value)) {
    out.value = s.getValue().str();
    return out;
  }
  if (auto b = llvm::dyn_cast<mlir::BoolAttr>(value)) {
    out.value = b.getValue();
    return out;
  }
  if (auto a = llvm::dyn_cast<mlir::ArrayAttr>(value)) {
    // The element type is the first element's. An empty array is the one
    // ambiguous case (see toMlirAttribute) and reads back as int64.
    if (a.empty()) {
      out.value = std::vector<int64_t>{};
      return out;
    }
    if (llvm::isa<mlir::IntegerAttr>(a[0])) {
      std::vector<int64_t> items;
      for (mlir::Attribute x : a)
        items.push_back(llvm::cast<mlir::IntegerAttr>(x).getInt());
      out.value = std::move(items);
      return out;
    }
    if (llvm::isa<mlir::FloatAttr>(a[0])) {
      std::vector<double> items;
      for (mlir::Attribute x : a)
        items.push_back(llvm::cast<mlir::FloatAttr>(x).getValueAsDouble());
      out.value = std::move(items);
      return out;
    }
    if (llvm::isa<mlir::StringAttr>(a[0])) {
      std::vector<std::string> items;
      for (mlir::Attribute x : a)
        items.push_back(llvm::cast<mlir::StringAttr>(x).getValue().str());
      out.value = std::move(items);
      return out;
    }
    if (llvm::isa<mlir::BoolAttr>(a[0])) {
      std::vector<bool> items;
      for (mlir::Attribute x : a)
        items.push_back(llvm::cast<mlir::BoolAttr>(x).getValue());
      out.value = std::move(items);
      return out;
    }
  }
  return std::nullopt;
}

//--- value types -------------------------------------------------------------

// The tensor ids the graph declares as the mmq activation layout: the first
// operand of a quantized_matmul_q8 node. The graph itself says nothing more — a
// `quantize` result could in principle feed an f32 path — so this is the whole
// rule, and it is the same one the graph verifier applies (BoomVerify.cpp).
llvm::DenseSet<int64_t> q8_1Ids(const Graph &g) {
  llvm::DenseSet<int64_t> ids;
  for (const GraphNodeDesc &n : g.nodes) {
    if (n.op != OpKind::QuantizedMatmulQ8 || n.inputs.empty())
      continue;
    ids.insert(raw(n.inputs.front()));
  }
  return ids;
}

// The SSA type of a tensor: !boom.q8_1 when the graph declares it as the mmq
// activation layout, !boom.tensor<shape, dtype> otherwise. Empty on a tensor no
// Boom type can describe — an unknown dtype name, or a q8_1 shape that is not a
// whole number of 32-element blocks — after diagnosing it.
mlir::Type valueType(mlir::MLIRContext *context, mlir::Location loc,
                     const TensorDesc &t, bool is_q8_1) {
  // ArrayRef, not the vector: the generated getChecked is a concrete overload
  // next to the base's catch-all template, and only an exact match picks it.
  const llvm::ArrayRef<int64_t> shape(t.shape);
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

// The node attributes, in order, as the `node_attrs` array.
llvm::SmallVector<mlir::Attribute> nodeAttrs(mlir::MLIRContext *context,
                                             const GraphNodeDesc &n) {
  llvm::SmallVector<mlir::Attribute> out;
  for (const sx::Attribute &a : n.attributes)
    out.push_back(boom::NodeAttr::get(
        context, a.name, toMlirAttribute(context, a)));
  return out;
}

// The typed view of a node's attributes: the ones ODS declares, which are the
// names the plan emitters and the fusion passes read. An attribute in that
// vocabulary whose value is neither an integer nor a float has no typed
// attribute to set, and the graph verifier would reject the mismatch, so it is
// reported here instead.
bool addTypedAttributes(mlir::MLIRContext *context, mlir::Location loc,
                        const GraphNodeDesc &n,
                        llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs) {
  for (llvm::StringRef name : boom::boomTypedAttrNames()) {
    const sx::Attribute *found = nullptr;
    for (const sx::Attribute &a : n.attributes)
      if (llvm::StringRef(a.name) == name)
        found = &a;
    if (!found)
      continue;
    switch (found->kind()) {
    case sx::AttrValueKind::Int:
      attrs.emplace_back(name, mlir::IntegerAttr::get(
                                   mlir::IntegerType::get(context, 64),
                                   std::get<int64_t>(found->value)));
      break;
    case sx::AttrValueKind::Float:
      attrs.emplace_back(name, mlir::FloatAttr::get(
                                   mlir::Float64Type::get(context),
                                   std::get<double>(found->value)));
      break;
    default:
      mlir::emitError(loc)
          << "node '" << n.name << "' carries the attribute '" << name
          << "', which the dialect declares with an integer or float type";
      return false;
    }
  }
  return true;
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

  llvm::SmallVector<mlir::Attribute> table;
  table.reserve(graph.tensors.size());
  for (const TensorDesc &t : graph.tensors) {
    table.push_back(boom::TensorDescAttr::get(
        context, raw(t.id), t.shape, sx::dtype_name(t.dtype), t.size_bytes,
        t.is_graph_input, t.is_graph_output, t.is_constant, t.is_external,
        t.name));
  }

  auto graphOp = boom::GraphOp::create(
      builder, loc, builder.getStringAttr(graph.name), builder.getArrayAttr(table),
      builder.getDenseI64ArrayAttr(flatten(graph.inputs)),
      builder.getDenseI64ArrayAttr(flatten(graph.outputs)));

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
    mlir::Type type = valueType(context, loc, *t, q8_1.count(id));
    if (!type)
      return {};
    valueOf[id] = body->addArgument(type, loc);
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

    llvm::SmallVector<mlir::Type> results;
    for (TensorId out : n.outputs) {
      auto it = byId.find(raw(out));
      if (it == byId.end()) {
        mlir::emitError(loc) << "node '" << n.name << "' produces tensor id "
                             << raw(out) << ", which is not in the table";
        return {};
      }
      mlir::Type type = valueType(context, loc, *it->second, q8_1.count(raw(out)));
      if (!type)
        return {};
      results.push_back(type);
    }

    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.emplace_back("node_name", builder.getStringAttr(n.name));
    attrs.emplace_back("node_id", builder.getI64IntegerAttr(raw(n.id)));
    attrs.emplace_back("result_ids",
                       builder.getDenseI64ArrayAttr(flatten(n.outputs)));
    attrs.emplace_back("node_attrs",
                       builder.getArrayAttr(nodeAttrs(context, n)));
    if (n.backend)
      attrs.emplace_back(
          "backend",
          builder.getStringAttr(planner::backend_tag_name(*n.backend)));
    if (!addTypedAttributes(context, loc, n, attrs))
      return {};

    mlir::OperationState state(
        loc, std::string("boom.") + planner::op_kind_name(n.op));
    state.addOperands(operands);
    state.addTypes(results);
    state.propertiesAttr = mlir::DictionaryAttr::get(context, attrs);
    mlir::Operation *node = builder.create(state);

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

std::optional<Graph> boomToGraph(mlir::ModuleOp module) {
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
    t.size_bytes = desc.getByteSize();
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
  if (body.getNumArguments() != freeIds.size()) {
    graphOp.emitError() << "the body has " << body.getNumArguments()
                        << " block arguments but " << freeIds.size()
                        << " tensors in the table are not produced by a node";
    return std::nullopt;
  }

  llvm::DenseMap<mlir::Value, int64_t> idOfValue;
  for (auto [index, id] : llvm::enumerate(freeIds))
    idOfValue[body.getArgument(index)] = id;

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
