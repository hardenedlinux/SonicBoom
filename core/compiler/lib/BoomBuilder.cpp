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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

  // A non-empty array is an ArrayAttr whose element attributes name its kind.
  // An *empty* array has no element to carry that: an empty ArrayAttr reads
  // back as whichever kind the decoder guesses, so the encoder gives it the
  // element kind's typed zero-length container instead (DenseI64ArrayAttr …
  // DenseStringElementsAttr). The value slot of #boom.attr is any MLIR
  // attribute, so this is an encoding of the value, not a change to the
  // attribute's format.
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
    const auto &values = std::get<std::vector<int64_t>>(v);
    if (values.empty())
      return mlir::DenseI64ArrayAttr::get(context, {});
    llvm::SmallVector<mlir::Attribute> items;
    for (int64_t x : values)
      items.push_back(mlir::IntegerAttr::get(
          mlir::IntegerType::get(context, 64), x));
    return mlir::ArrayAttr::get(context, items);
  }
  case K::Floats: {
    const auto &values = std::get<std::vector<double>>(v);
    if (values.empty())
      return mlir::DenseF64ArrayAttr::get(context, {});
    llvm::SmallVector<mlir::Attribute> items;
    for (double x : values)
      items.push_back(
          mlir::FloatAttr::get(mlir::Float64Type::get(context), x));
    return mlir::ArrayAttr::get(context, items);
  }
  case K::Strings: {
    const auto &values = std::get<std::vector<std::string>>(v);
    if (values.empty()) {
      // MLIR has no string element type, and a dense string attribute needs a
      // shaped type. With no elements the type is never inspected; it must
      // only be a non-integer, non-float type, because the parser turns
      // `dense<…> : tensor<Nxi8>` into a dense integer attribute — the very
      // ambiguity this encoding avoids. A [1]xi8 vector element is that
      // placeholder.
      auto element =
          mlir::VectorType::get({1}, mlir::IntegerType::get(context, 8));
      auto type = mlir::RankedTensorType::get({0}, element);
      return mlir::DenseStringElementsAttr::get(
          type, llvm::ArrayRef<llvm::StringRef>());
    }
    llvm::SmallVector<mlir::Attribute> items;
    for (const std::string &x : values)
      items.push_back(mlir::StringAttr::get(context, x));
    return mlir::ArrayAttr::get(context, items);
  }
  case K::Bools: {
    const auto &values = std::get<std::vector<bool>>(v);
    if (values.empty())
      return mlir::DenseBoolArrayAttr::get(context, {});
    llvm::SmallVector<mlir::Attribute> items;
    for (bool x : values)
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

  // BoolAttr *is* an IntegerAttr (it derives from it), so it must be tested
  // first: the integer branch would read it as 0/1 and change the attribute's
  // kind, and the kind is part of the model fingerprint.
  if (auto b = llvm::dyn_cast<mlir::BoolAttr>(value)) {
    out.value = b.getValue();
    return out;
  }
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
  // The zero-length containers toMlirAttribute gives an empty array. Each
  // names its element kind, so the kind survives the round trip; they are read
  // before any generic attribute test because they carry no elements to infer
  // from.
  if (auto a = llvm::dyn_cast<mlir::DenseBoolArrayAttr>(value)) {
    llvm::ArrayRef<bool> items = a.asArrayRef();
    out.value = std::vector<bool>(items.begin(), items.end());
    return out;
  }
  if (auto a = llvm::dyn_cast<mlir::DenseI64ArrayAttr>(value)) {
    llvm::ArrayRef<int64_t> items = a.asArrayRef();
    out.value = std::vector<int64_t>(items.begin(), items.end());
    return out;
  }
  if (auto a = llvm::dyn_cast<mlir::DenseF64ArrayAttr>(value)) {
    llvm::ArrayRef<double> items = a.asArrayRef();
    out.value = std::vector<double>(items.begin(), items.end());
    return out;
  }
  if (auto a = llvm::dyn_cast<mlir::DenseStringElementsAttr>(value)) {
    std::vector<std::string> items;
    for (llvm::StringRef x : a.getValues<llvm::StringRef>())
      items.push_back(x.str());
    out.value = std::move(items);
    return out;
  }
  if (auto a = llvm::dyn_cast<mlir::ArrayAttr>(value)) {
    // The element type is the first element's. A zero-length ArrayAttr has
    // none; the encoder never produces one (see toMlirAttribute), so this
    // reads it as the pre-encoding behaviour did — an int64 array.
    if (a.empty()) {
      out.value = std::vector<int64_t>{};
      return out;
    }
    // As above: a BoolAttr is an IntegerAttr, so the bool array is the more
    // specific test and comes first.
    if (llvm::isa<mlir::BoolAttr>(a[0])) {
      std::vector<bool> items;
      for (mlir::Attribute x : a)
        items.push_back(llvm::cast<mlir::BoolAttr>(x).getValue());
      out.value = std::move(items);
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

mlir::Value BoomBuilder::kv(uint32_t layer) {
  auto it = kv_.find(layer);
  if (it != kv_.end())
    return it->second;
  auto type = boom::KvType::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(loc_), context_, int64_t(layer));
  if (!type) {
    failed_ = true;
    return {};
  }
  mlir::Value value = body_->addArgument(type, loc_);
  kv_[layer] = value;
  return value;
}

void BoomBuilder::set_row_scale(int64_t rows, llvm::StringRef attr_name) {
  if (rows < 0) {
    failed_ = true;
    return;
  }
  row_scale_ = rows;
  row_scale_attr_ = attr_name.str();
}

mlir::Value BoomBuilder::node(planner::OpKind op,
                              llvm::ArrayRef<mlir::Value> operands,
                              llvm::StringRef out_name,
                              llvm::ArrayRef<int64_t> out_shape,
                              llvm::ArrayRef<sx::Attribute> attributes,
                              planner::BackendTag backend) {
  // Row-dealt emission (verify step): the emitter writes one query row's
  // shapes, the builder lays them out over `rows` positions and stamps the
  // row count on every node so the backend dispatches per-row decode kernels.
  llvm::SmallVector<int64_t, 4> scaled_shape;
  llvm::SmallVector<sx::Attribute, 6> scaled_attrs;
  if (row_scale_ > 0) {
    scaled_shape.assign(out_shape.begin(), out_shape.end());
    for (int64_t& v : scaled_shape) v *= row_scale_;
    scaled_attrs.assign(attributes.begin(), attributes.end());
    scaled_attrs.push_back(
        sx::Attribute{row_scale_attr_, int64_t(row_scale_)});
    out_shape = scaled_shape;
    attributes = scaled_attrs;
  }

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

//--- the generation program ---------------------------------------------------

// Build the program around the module's decode graph. `max_drafts == 0` is the
// greedy loop (step -> sample -> emit -> stop_check); a positive count is the
// speculative loop (draft -> step/verify -> accept -> emit -> stop_check, with
// the step as the drafter's k == 0 fallback).
static mlir::LogicalResult buildGenerationProgram(
    mlir::ModuleOp module, llvm::StringRef program_name,
    llvm::StringRef prefill_graph, llvm::ArrayRef<int64_t> prompt_shape,
    llvm::ArrayRef<int64_t> eos_tokens, int64_t max_tokens,
    int64_t max_drafts, llvm::StringRef draft_strategy) {
  mlir::MLIRContext *context = module.getContext();
  const mlir::Location loc = mlir::UnknownLoc::get(context);
  // The program holds SCF control flow and arith step arithmetic, which cannot
  // be created before their dialects are loaded.
  context->getOrLoadDialect<mlir::scf::SCFDialect>();
  context->getOrLoadDialect<mlir::arith::ArithDialect>();

  // The decode graph this program lifts. Exactly one: a program is one
  // generation, and two decode graphs in a module would have to be chosen
  // between.
  boom::GraphOp graph;
  int64_t graph_count = 0;
  for (mlir::Operation &op : module.getBody()->getOperations())
    if (auto g = llvm::dyn_cast<boom::GraphOp>(&op)) {
      if (graph_count > 0)
        return module.emitError()
               << "a generation program lifts one decode graph, but the module "
                  "holds more than one";
      graph = g;
      ++graph_count;
    }
  if (!graph)
    return module.emitError()
           << "a generation program needs a decode graph in its module";
  if (!llvm::hasSingleElement(graph.getBody()))
    return graph.emitOpError() << "must have exactly one block";

  mlir::Block &graphBody = graph.getBody().front();

  // The graph's free tensor variables, ascending id — the block-argument order
  // the graph verifier recomputes. The step keeps them as its block arguments.
  llvm::DenseSet<int64_t> produced;
  for (mlir::Operation &op : graphBody)
    for (int64_t id : op.getAttrOfType<mlir::DenseI64ArrayAttr>("result_ids")
                          .asArrayRef())
      produced.insert(id);
  llvm::SmallVector<int64_t> freeIds;
  for (mlir::Attribute entry : graph.getTensorsAttr()) {
    auto desc = llvm::cast<boom::TensorDescAttr>(entry);
    if (!produced.count(desc.getId()))
      freeIds.push_back(desc.getId());
  }
  llvm::sort(freeIds);

  auto argOfId = [&](int64_t id) -> mlir::Value {
    auto it = llvm::find(freeIds, id);
    if (it == freeIds.end())
      return {};
    return graphBody.getArgument(size_t(std::distance(freeIds.begin(), it)));
  };
  auto descOfId = [&](int64_t id) -> boom::TensorDescAttr {
    for (mlir::Attribute entry : graph.getTensorsAttr()) {
      auto desc = llvm::cast<boom::TensorDescAttr>(entry);
      if (desc.getId() == id)
        return desc;
    }
    return {};
  };

  // The token and position the loop carries. The decode emitter declares them
  // first (token_id, pos); by name, so a graph that lists its inputs in another
  // order still lifts correctly.
  mlir::Value firstTokenArg, posArg;
  for (int64_t id : graph.getInputs()) {
    auto desc = descOfId(id);
    if (desc.getName() == "token_id")
      firstTokenArg = argOfId(id);
    else if (desc.getName() == "pos")
      posArg = argOfId(id);
  }
  if (!firstTokenArg || !posArg)
    return graph.emitOpError()
           << "a decode graph lifted into a program must have `token_id` and "
              "`pos` inputs";
  const mlir::Type tokenType = firstTokenArg.getType();
  const mlir::Type posType = posArg.getType();

  llvm::SmallVector<mlir::Type> kvTypes;
  for (size_t i = freeIds.size(); i < graphBody.getNumArguments(); ++i)
    kvTypes.push_back(graphBody.getArgument(i).getType());

  // The decode graph's output type is the step's result type.
  if (graph.getOutputs().size() != 1)
    return graph.emitOpError()
           << "a decode graph lifted into a program must have one output";
  auto outDesc = descOfId(graph.getOutputs().front());
  if (!outDesc)
    return graph.emitOpError() << "its output is not in the tensor table";
  const mlir::Type outType =
      boom::TensorType::get(context, outDesc.getShape(), outDesc.getDtype());

  auto promptType =
      boom::TensorType::get(context, prompt_shape, sx::dtype_name(sx::DType::Int64));

  mlir::OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  auto program = boom::ProgramOp::create(
      builder, loc, builder.getStringAttr(program_name),
      builder.getStringAttr(prefill_graph));
  mlir::Block *body = builder.createBlock(&program.getBody());
  mlir::Value prompt = body->addArgument(promptType, loc);
  mlir::Value startPos = body->addArgument(posType, loc);
  mlir::Value firstToken = body->addArgument(tokenType, loc);
  mlir::Value pos0 = body->addArgument(posType, loc);
  llvm::SmallVector<mlir::Value> kvArgs;
  for (mlir::Type kvType : kvTypes)
    kvArgs.push_back(body->addArgument(kvType, loc));

  builder.setInsertionPointToEnd(body);
  llvm::SmallVector<mlir::Value> prefillOperands{prompt, startPos};
  prefillOperands.append(kvArgs);
  boom::PrefillOp::create(builder, loc, prefillOperands,
                          builder.getStringAttr(prefill_graph));

  const mlir::Type i64Type = builder.getI64Type();
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);

  if (max_drafts <= 0) {
    // The greedy loop: one decode step, then sample, emit and stop.
    mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{tokenType, posType, i64Type},
        mlir::ValueRange{firstToken, pos0, zero},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          llvm::SmallVector<mlir::Value> stepOperands{args[0], args[1]};
          stepOperands.append(kvArgs);
          auto step = boom::StepOp::create(
              b, l, mlir::TypeRange{outType}, stepOperands,
              graph.getTensorsAttr(), graph.getInputsAttr(),
              graph.getOutputsAttr(), graph.getDynamicExtentAttr());
          // The decode graph's body becomes the step's: the same block
          // arguments, now bound to the loop's SSA values.
          step.getBody().takeBody(graph.getBody());

          auto sample = boom::SampleOp::create(b, l, tokenType, step.getResult(0),
                                               mlir::FloatAttr(),
                                               mlir::IntegerAttr(),
                                               mlir::FloatAttr());
          boom::EmitOp::create(b, l, mlir::ValueRange{sample.getToken()});
          auto stop = boom::StopCheckOp::create(
              b, l, b.getI1Type(), sample.getToken(), args[2],
              llvm::ArrayRef<int64_t>(eos_tokens), uint64_t(max_tokens));
          mlir::Value one = mlir::arith::ConstantIntOp::create(b, l, 1, 1);
          mlir::Value go = mlir::arith::XOrIOp::create(b, l, stop.getStop(), one);
          mlir::scf::ConditionOp::create(
              b, l, go, mlir::ValueRange{sample.getToken(), args[1], args[2]});
        },
        // The continuation: advance the position and the emitted count.
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value nextPos = boom::PosIncOp::create(
              b, l, posType, args[1], mlir::Value(), b.getI64IntegerAttr(1));
          mlir::Value one = mlir::arith::ConstantIntOp::create(b, l, 1, 64);
          mlir::Value nextCount =
              mlir::arith::AddIOp::create(b, l, args[2], one);
          mlir::scf::YieldOp::create(
              b, l, mlir::ValueRange{args[0], nextPos, nextCount});
        });
    graph.erase();
  } else {
    // The speculative loop. The carried tuple is (token, position, emitted
    // count, accepted+1): the continuation advances the position by the accept
    // node's run length, which the condition region computes and the
    // continuation needs, so it rides along.
    const int64_t rows = max_drafts + 1;
    const mlir::Type draftsType = boom::TensorType::get(
        context, llvm::ArrayRef<int64_t>{max_drafts}, sx::dtype_name(sx::DType::Int64));
    const mlir::Type rowsType = boom::TensorType::get(
        context, llvm::ArrayRef<int64_t>{rows}, sx::dtype_name(sx::DType::Int64));
    const mlir::Type runType = boom::TensorType::get(
        context, llvm::ArrayRef<int64_t>{rows}, sx::dtype_name(sx::DType::Int64));
    const mlir::StringAttr graphName = graph.getGraphNameAttr();

    mlir::scf::WhileOp::create(
        builder, loc, mlir::TypeRange{tokenType, posType, i64Type, i64Type},
        mlir::ValueRange{firstToken, pos0, zero, zero},
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          // Draft: up to max_drafts tokens following the accepted history.
          auto draft = boom::DraftOp::create(
              b, l, mlir::TypeRange{draftsType, i64Type},
              b.getStringAttr(draft_strategy),
              b.getI64IntegerAttr(max_drafts));

          // The plain step: the k == 0 path, and the decode fallback the
          // executor runs instead of a zero-row verify. It owns the graph body.
          llvm::SmallVector<mlir::Value> stepOperands{args[0], args[1]};
          stepOperands.append(kvArgs);
          auto step = boom::StepOp::create(
              b, l, mlir::TypeRange{outType}, stepOperands,
              graph.getTensorsAttr(), graph.getInputsAttr(),
              graph.getOutputsAttr(), graph.getDynamicExtentAttr());
          step.getBody().takeBody(graph.getBody());

          // Verify: the last token and the drafts, at the base position, over
          // the cache handles. Row i's argmax is the plain greedy token at
          // start_pos + i + 1.
          llvm::SmallVector<mlir::Value> verifyOperands{
              args[0], draft.getDrafts().front(), args[1]};
          verifyOperands.append(kvArgs);
          auto verify = boom::VerifyOp::create(
              b, l, mlir::TypeRange{rowsType}, verifyOperands, graphName,
              b.getI64IntegerAttr(rows));

          // Accept: the longest matching draft prefix plus row a's argmax.
          auto accept = boom::AcceptOp::create(
              b, l, mlir::TypeRange{runType, tokenType, i64Type},
              mlir::ValueRange{draft.getDrafts().front(), step.getResult(0),
                               verify.getRows().front()});

          boom::EmitOp::create(b, l, accept.getTokens());
          auto stop = boom::StopCheckOp::create(
              b, l, b.getI1Type(), accept.getTokens().front(), args[2],
              llvm::ArrayRef<int64_t>(eos_tokens), uint64_t(max_tokens));
          mlir::Value one = mlir::arith::ConstantIntOp::create(b, l, 1, 1);
          mlir::Value go = mlir::arith::XOrIOp::create(b, l, stop.getStop(), one);
          mlir::Value one64 = mlir::arith::ConstantIntOp::create(b, l, 1, 64);
          mlir::Value advance =
              mlir::arith::AddIOp::create(b, l, accept.getAccepted(), one64);
          mlir::scf::ConditionOp::create(
              b, l, go,
              mlir::ValueRange{accept.getNext(), args[1], args[2], advance});
        },
        // The continuation: advance by the accepted run's length and count the
        // run's tokens; the next iteration starts with no pending advance.
        [&](mlir::OpBuilder &b, mlir::Location l, mlir::ValueRange args) {
          mlir::Value nextPos = boom::PosIncOp::create(
              b, l, posType, args[1], args[3], b.getI64IntegerAttr(1));
          mlir::Value nextCount =
              mlir::arith::AddIOp::create(b, l, args[2], args[3]);
          mlir::Value zero64 = mlir::arith::ConstantIntOp::create(b, l, 0, 64);
          mlir::scf::YieldOp::create(
              b, l, mlir::ValueRange{args[0], nextPos, nextCount, zero64});
        });

    // The graph's body now lives in the step; the graph op is spent.
    graph.erase();
  }

  if (mlir::failed(mlir::verify(module.getOperation())))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult addGreedyProgram(mlir::ModuleOp module,
                                     llvm::StringRef program_name,
                                     llvm::StringRef prefill_graph,
                                     llvm::ArrayRef<int64_t> prompt_shape,
                                     llvm::ArrayRef<int64_t> eos_tokens,
                                     int64_t max_tokens) {
  return buildGenerationProgram(module, program_name, prefill_graph,
                                prompt_shape, eos_tokens, max_tokens,
                                /*max_drafts=*/0, /*draft_strategy=*/{});
}

mlir::LogicalResult addSpeculativeProgram(mlir::ModuleOp module,
                                          llvm::StringRef program_name,
                                          llvm::StringRef prefill_graph,
                                          llvm::ArrayRef<int64_t> prompt_shape,
                                          llvm::ArrayRef<int64_t> eos_tokens,
                                          int64_t max_tokens,
                                          int64_t max_drafts,
                                          llvm::StringRef draft_strategy) {
  if (max_drafts <= 0)
    return module.emitError()
           << "a speculative program needs a positive draft length";
  return buildGenerationProgram(module, program_name, prefill_graph,
                                prompt_shape, eos_tokens, max_tokens,
                                max_drafts, draft_strategy);
}

} // namespace sonicboom::compiler
