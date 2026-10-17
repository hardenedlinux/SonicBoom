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

// The emitter's side of Boom IR: the shared node record, and the builder the
// plan emitters write a graph with. `graphToBoom` (BoomConversion.cpp) goes
// through the same `createBoomNode`, so the emitter's IR and the converter's IR
// cannot say different things about a node.

#include "sonicboom/compiler/BoomBuilder.h"

#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include <string>

namespace sonicboom::compiler {

namespace boom = sonicboom::compiler::boom;

namespace {

// The typed view of a node's attributes: the names ODS declares, which are the
// ones the plan emitters and the fusion passes read. An attribute in that
// vocabulary whose value is neither an integer nor a float has no typed
// attribute to set, and the graph verifier would reject the mismatch, so it is
// reported here — naming the node and the attribute — instead of producing an op
// whose two views of the same attribute disagree.
bool addTypedAttributes(mlir::MLIRContext *context, mlir::Location loc,
                        llvm::StringRef node_name,
                        llvm::ArrayRef<sx::Attribute> attributes,
                        llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs) {
  for (llvm::StringRef name : boom::boomTypedAttrNames()) {
    const sx::Attribute *found = nullptr;
    for (const sx::Attribute &a : attributes)
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
          << "node '" << node_name << "' carries the attribute '" << name
          << "', which the dialect declares with an integer or float type";
      return false;
    }
  }
  return true;
}

// The `node_attrs` array: every attribute, in order, whatever its kind.
llvm::SmallVector<mlir::Attribute>
nodeAttrs(mlir::MLIRContext *context, llvm::ArrayRef<sx::Attribute> attributes) {
  llvm::SmallVector<mlir::Attribute> out;
  out.reserve(attributes.size());
  for (const sx::Attribute &a : attributes)
    out.push_back(
        boom::NodeAttr::get(context, a.name, toMlirAttribute(context, a)));
  return out;
}

// What a tensor needs to become: the SSA type of its value and the byte count
// its table entry records. Both are refused rather than guessed when the dtype
// is outside the frozen names or the shape's element count is not
// representable — the table entry is what the round trip reads back, so a wrong
// size there is a plan that differs from the graph it came from.
//
// A shape with a dynamic dimension (the graph's extent) is sized at the extent
// the builder was given: the byte count in the table is then the one the
// concrete graph at that length carries, which is what the passes compare.
// The lowering recomputes the size from the extent it binds
// (BoomConversion.cpp), so the recorded value is informative, not binding; a
// builder given no extent records 0.
struct TensorForm {
  mlir::Type type;
  uint64_t size_bytes;
};

std::optional<TensorForm> tensorForm(mlir::MLIRContext *context,
                                     mlir::Location loc, int64_t id,
                                     llvm::StringRef name,
                                     llvm::ArrayRef<int64_t> shape,
                                     sx::DType dtype, uint64_t extent) {
  const llvm::StringRef dtypeName = sx::dtype_name(dtype);
  if (!boom::boomDTypeFromName(dtypeName)) {
    mlir::emitError(loc) << "tensor id " << id << " ('" << name
                         << "') has the dtype '" << dtypeName
                         << "', which is not one of the frozen sx::DType names";
    return std::nullopt;
  }
  const bool dynamic = llvm::is_contained(shape, mlir::ShapedType::kDynamic);
  std::vector<int64_t> sized(shape.begin(), shape.end());
  for (int64_t &d : sized)
    if (d == mlir::ShapedType::kDynamic) d = int64_t(extent);
  const std::optional<int64_t> size =
      (dynamic && extent == 0)
          ? std::optional<int64_t>(0)
          : sx::tensor_byte_size(sx::TensorType{dtype,
                                                sx::Shape{std::move(sized)}});
  if (!size) {
    mlir::emitError(loc) << "tensor id " << id << " ('" << name
                         << "') has the shape " << shape
                         << ", whose element count is not representable";
    return std::nullopt;
  }
  // ArrayRef, not the vector: the generated getChecked is a concrete overload
  // next to the base's catch-all template, and only an exact match picks it.
  auto type = boom::TensorType::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(loc), context, shape, dtypeName);
  if (!type)
    return std::nullopt;
  return TensorForm{type, uint64_t(*size)};
}

} // namespace

//--- attribute values --------------------------------------------------------

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

std::optional<sx::Attribute> fromMlirAttribute(llvm::StringRef name,
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

//--- one node -----------------------------------------------------------------

mlir::Operation *createBoomNode(mlir::OpBuilder &builder, mlir::Location loc,
                                planner::OpKind op,
                                llvm::ArrayRef<mlir::Value> operands,
                                llvm::ArrayRef<mlir::Type> results,
                                llvm::StringRef name, int64_t node_id,
                                llvm::ArrayRef<int64_t> result_ids,
                                llvm::ArrayRef<sx::Attribute> attributes,
                                std::optional<planner::BackendTag> backend) {
  mlir::MLIRContext *context = builder.getContext();

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.emplace_back("node_name", builder.getStringAttr(name));
  attrs.emplace_back("node_id", builder.getI64IntegerAttr(node_id));
  attrs.emplace_back("result_ids", builder.getDenseI64ArrayAttr(result_ids));
  attrs.emplace_back("node_attrs",
                     builder.getArrayAttr(nodeAttrs(context, attributes)));
  if (backend)
    attrs.emplace_back(
        "backend",
        builder.getStringAttr(planner::backend_tag_name(*backend)));
  if (!addTypedAttributes(context, loc, name, attributes, attrs))
    return nullptr;

  mlir::OperationState state(
      loc, std::string("boom.") + planner::op_kind_name(op));
  state.addOperands(operands);
  state.addTypes(results);
  state.propertiesAttr = mlir::DictionaryAttr::get(context, attrs);
  return builder.create(state);
}

//--- the builder --------------------------------------------------------------

BoomBuilder::BoomBuilder(mlir::MLIRContext *context, llvm::StringRef graph_name,
                         uint64_t extent)
    : context_(context), loc_(mlir::UnknownLoc::get(context)), extent_(extent),
      builder_(context) {
  // An op cannot be created by name until its dialect is loaded, and every node
  // here is created by name.
  context->getOrLoadDialect<boom::BoomDialect>();

  // The module is the graph's container, not a second level of meaning: the
  // graph op is IsolatedFromAbove, so it needs a scope to be isolated from.
  // The table and the input/output lists are empty for now — they are set once,
  // at finish(), when they are complete and cannot be read half-built.
  module_ = mlir::ModuleOp::create(loc_);
  builder_.setInsertionPointToStart(module_->getBody());
  graph_op_ = boom::GraphOp::create(
      builder_, loc_, builder_.getStringAttr(graph_name),
      builder_.getArrayAttr(llvm::ArrayRef<mlir::Attribute>{}),
      builder_.getDenseI64ArrayAttr(llvm::ArrayRef<int64_t>{}),
      builder_.getDenseI64ArrayAttr(llvm::ArrayRef<int64_t>{}),
      builder_.getI64IntegerAttr(int64_t(extent_)));
  body_ = builder_.createBlock(&graph_op_.getBody());
}

std::optional<int64_t> BoomBuilder::idOf(mlir::Value value) const {
  auto it = ids_.find(value);
  if (it == ids_.end())
    return std::nullopt;
  return it->second;
}

int64_t BoomBuilder::addTensor(llvm::StringRef name, llvm::ArrayRef<int64_t> shape,
                               sx::DType dtype, uint64_t size_bytes,
                               bool is_input) {
  const int64_t id = int64_t(table_.size());
  table_.push_back(boom::TensorDescAttr::get(
      context_, id, shape, sx::dtype_name(dtype), size_bytes,
      /*dynamic_base_bytes=*/0, /*is_graph_input=*/is_input,
      /*is_graph_output=*/false, /*is_constant=*/false,
      /*is_external=*/false, name));
  if (is_input)
    inputs_.push_back(id);
  return id;
}

mlir::Value BoomBuilder::input(llvm::StringRef name, llvm::ArrayRef<int64_t> shape,
                               sx::DType dtype) {
  const int64_t id = int64_t(table_.size());
  auto form = tensorForm(context_, loc_, id, name, shape, dtype, extent_);
  if (!form) {
    failed_ = true;
    return {};
  }
  addTensor(name, shape, dtype, form->size_bytes, /*is_input=*/true);
  mlir::Value value = body_->addArgument(form->type, loc_);
  ids_[value] = id;
  return value;
}

mlir::Value BoomBuilder::node(planner::OpKind op,
                              llvm::ArrayRef<mlir::Value> operands,
                              llvm::StringRef out_name,
                              llvm::ArrayRef<int64_t> out_shape,
                              llvm::ArrayRef<sx::Attribute> attributes,
                              planner::BackendTag backend) {
  const int64_t id = int64_t(table_.size());
  auto form = tensorForm(context_, loc_, id, out_name, out_shape,
                         sx::DType::Float32, extent_);
  if (!form) {
    failed_ = true;
    return {};
  }
  addTensor(out_name, out_shape, sx::DType::Float32, form->size_bytes,
            /*is_input=*/false);

  // The node's name is its operator's — the graph node's identity is the
  // operator plus the ids it carries, and the tensor's name is its own.
  const int64_t node_id = nodes_++;
  const llvm::SmallVector<mlir::Type, 1> results{form->type};
  const llvm::SmallVector<int64_t, 1> result_ids{id};
  mlir::Operation *node = createBoomNode(builder_, loc_, op, operands, results,
                                         planner::op_kind_name(op), node_id,
                                         result_ids, attributes, backend);
  if (!node) {
    failed_ = true;
    return {};
  }
  mlir::Value value = node->getResult(0);
  ids_[value] = id;
  return value;
}

void BoomBuilder::set_output(mlir::Value value) {
  auto id = idOf(value);
  if (!id) {
    mlir::emitError(loc_)
        << "the graph's output is not a value this builder created";
    failed_ = true;
    return;
  }
  auto desc = llvm::cast<boom::TensorDescAttr>(table_[*id]);
  table_[*id] = boom::TensorDescAttr::get(
      context_, *id, desc.getShape(), desc.getDtype(), desc.getByteSize(),
      desc.getDynamicBaseBytes(), desc.getIsInput(),
      /*is_graph_output=*/true, desc.getIsConstant(), desc.getIsExternal(),
      desc.getName());
  outputs_.assign(1, *id);
}

mlir::OwningOpRef<mlir::ModuleOp> BoomBuilder::finish() {
  if (failed_)
    return {};
  graph_op_.setTensorsAttr(builder_.getArrayAttr(table_));
  graph_op_.setInputsAttr(builder_.getDenseI64ArrayAttr(inputs_));
  graph_op_.setOutputsAttr(builder_.getDenseI64ArrayAttr(outputs_));

  // The builder is responsible for the IR it hands over: whatever it built must
  // satisfy the dialect's verifier, here rather than three passes later.
  if (mlir::failed(mlir::verify(module_->getOperation()))) {
    failed_ = true;
    return {};
  }
  return std::move(module_);
}

} // namespace sonicboom::compiler
