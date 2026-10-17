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

// Blade (sx::Document) -> Boom IR.
//
// This is the migration's step (d) lowering for the Blade route: the document
// is read directly into the dialect, without the planner::Graph that
// adapt_graph built for it. The ids, the table order, the free-variable block
// arguments and the node records are the ones `graphToBoom(adapt_graph(doc))`
// produces, which is what lets the test compare the two IRs byte for byte.

#include "sonicboom/compiler/BladeToBoom.h"

#include "sonicboom/compiler/BoomBuilder.h"
#include "sonicboom/compiler/BoomDialect.h"
#include "sonicboom/compiler/BoomTypes.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"

#include <sonicboom/planner/op_kind.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sonicboom::compiler {

namespace boom = sonicboom::compiler::boom;

namespace {

// One tensor as the document defines it. The flags are the TensorDesc fields
// adapt_graph derives from the same document: an input is a graph input, a
// parameter is a constant (external when its data is a sidecar reference), and
// a graph output is marked after the output list resolves.
struct Entry {
  int64_t id = 0;
  const sx::TensorType *type = nullptr;
  std::string name;
  bool is_input = false;
  bool is_constant = false;
  bool is_external = false;
  bool is_output = false;
};

// The SSA type of a tensor. This is the same rule graphToBoom applies: a
// q8_1 activation layout where the graph says one is (the first operand of a
// quantized-matmul node), a Boom tensor otherwise; a dtype no Boom type names
// is refused rather than guessed.
mlir::Type valueType(mlir::MLIRContext *context, mlir::Location loc,
                     int64_t id, llvm::StringRef name,
                     const sx::TensorType &type, bool is_q8_1) {
  llvm::ArrayRef<int64_t> shape(type.shape.dims);
  if (is_q8_1) {
    auto q = boom::Q8_1Type::getChecked(
        mlir::detail::getDefaultDiagnosticEmitFn(loc), context, shape);
    if (!q)
      mlir::emitError(loc) << "tensor id " << id << " ('" << name
                           << "') is a q8_1 activation of shape " << shape
                           << ", which is not a whole number of 32-element "
                              "blocks";
    return q;
  }
  const llvm::StringRef dtypeName = sx::dtype_name(type.dtype);
  if (!boom::boomDTypeFromName(dtypeName)) {
    mlir::emitError(loc) << "tensor id " << id << " ('" << name
                         << "') has the dtype '" << dtypeName
                         << "', which is not one of the frozen sx::DType names";
    return {};
  }
  return boom::TensorType::getChecked(
      mlir::detail::getDefaultDiagnosticEmitFn(loc), context, shape, dtypeName);
}

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> bladeToBoom(const sx::Document &doc,
                                              mlir::MLIRContext *context) {
  const sx::Graph &sg = doc.graph;
  const mlir::Location loc = mlir::UnknownLoc::get(context);
  // An op cannot be created by name until its dialect is loaded. Loading here
  // (idempotently) keeps the lowering independent of what the caller did.
  context->getOrLoadDialect<boom::BoomDialect>();
  mlir::OpBuilder builder(context);

  // The module is the graph's container, not a second level of meaning: the
  // graph op is IsolatedFromAbove, so it needs a scope to be isolated from.
  mlir::OwningOpRef<mlir::ModuleOp> module = mlir::ModuleOp::create(loc);
  builder.setInsertionPointToStart(module->getBody());

  std::vector<Entry> entries;
  std::unordered_map<std::string, int64_t> by_name;

  // Ids are the table position: inputs, parameters, then node outputs in
  // document order. This is adapt_graph's assignment, and the reason the two
  // lowerings agree on `result_ids`.
  auto add_tensor =
      [&](const std::string &name, const sx::TensorType &type, bool is_input,
          bool is_constant, bool is_external) -> std::optional<int64_t> {
    if (by_name.count(name)) {
      mlir::emitError(loc) << "duplicate value definition '" << name << "'";
      return std::nullopt;
    }
    if (!sx::tensor_byte_size(type)) {
      mlir::emitError(loc) << "tensor '" << name << "' byte size overflows";
      return std::nullopt;
    }
    Entry e;
    e.id = int64_t(entries.size());
    e.type = &type;
    e.name = name;
    e.is_input = is_input;
    e.is_constant = is_constant;
    e.is_external = is_external;
    by_name.emplace(name, e.id);
    entries.push_back(std::move(e));
    return entries.back().id;
  };

  // 1. Graph inputs.
  std::vector<int64_t> input_ids;
  for (const auto &in : sg.inputs) {
    auto id = add_tensor(in.name, in.type, /*is_input=*/true, false, false);
    if (!id)
      return {};
    input_ids.push_back(*id);
  }

  // 2. Parameters (weights and constants).
  for (const auto &p : sg.parameters) {
    const bool is_external = std::holds_alternative<sx::ExternalData>(p.data);
    if (!add_tensor(p.name, p.type, /*is_input=*/false, /*is_constant=*/true,
                    is_external))
      return {};
  }

  // 3. Nodes: validate the operator, resolve operands to prior values, and
  //    define the node's outputs.
  struct LoweredNode {
    planner::OpKind op;
    const sx::Node *source = nullptr;
    std::vector<int64_t> inputs;
    std::vector<int64_t> outputs;
  };
  std::vector<LoweredNode> nodes;
  nodes.reserve(sg.nodes.size());
  for (const sx::Node &node : sg.nodes) {
    auto op = planner::op_kind_from_name(node.op_name);
    if (!op) {
      mlir::emitError(loc) << "unsupported operator '" << node.op_name << "'";
      return {};
    }
    LoweredNode ln;
    ln.op = *op;
    ln.source = &node;
    for (const auto &in : node.inputs) {
      auto it = by_name.find(in);
      if (it == by_name.end()) {
        mlir::emitError(loc) << "node '" << node.op_name
                             << "' references unknown value '" << in << "'";
        return {};
      }
      ln.inputs.push_back(it->second);
    }
    for (const auto &o : node.outputs) {
      auto id = add_tensor(o.name, o.type, /*is_input=*/false, false, false);
      if (!id)
        return {};
      ln.outputs.push_back(*id);
    }
    nodes.push_back(std::move(ln));
  }

  // 4. Graph outputs resolve, and the table entries they name are marked.
  std::vector<int64_t> output_ids;
  for (const auto &out_name : sg.outputs) {
    auto it = by_name.find(out_name);
    if (it == by_name.end()) {
      mlir::emitError(loc) << "graph output references unknown value '"
                           << out_name << "'";
      return {};
    }
    output_ids.push_back(it->second);
    entries[it->second].is_output = true;
  }

  // The tensor table, in id order.
  llvm::SmallVector<mlir::Attribute> table;
  table.reserve(entries.size());
  for (const Entry &e : entries) {
    table.push_back(boom::TensorDescAttr::get(
        context, e.id, llvm::ArrayRef<int64_t>(e.type->shape.dims),
        sx::dtype_name(e.type->dtype), uint64_t(*sx::tensor_byte_size(*e.type)),
        /*dynamic_base_bytes=*/0, e.is_input, e.is_output, e.is_constant,
        e.is_external, e.name));
  }

  // A Blade tensor is static, so there is no extent to size the dynamic
  // entries at (adapt_graph produces none either; graphToBoom records 0).
  auto graphOp = boom::GraphOp::create(
      builder, loc, builder.getStringAttr(sg.name), builder.getArrayAttr(table),
      builder.getDenseI64ArrayAttr(input_ids),
      builder.getDenseI64ArrayAttr(output_ids), builder.getI64IntegerAttr(0));

  mlir::Block *body = builder.createBlock(&graphOp.getBody());

  // The free variables: every tensor no node produces, in ascending id order —
  // the order the verifier recomputes. In a Blade document those are the
  // inputs and the parameters.
  llvm::DenseSet<int64_t> produced;
  for (const LoweredNode &n : nodes)
    for (int64_t id : n.outputs)
      produced.insert(id);

  // The q8_1 activations the graph declares: the first operand of a quantized
  // matmul. This is graphToBoom's rule, read from the document's nodes.
  llvm::DenseSet<int64_t> q8_1;
  for (const LoweredNode &n : nodes)
    if (n.op == planner::OpKind::QuantizedMatmulQ8 && !n.inputs.empty())
      q8_1.insert(n.inputs.front());

  llvm::SmallVector<int64_t> free_ids;
  for (const Entry &e : entries)
    if (!produced.count(e.id))
      free_ids.push_back(e.id);
  llvm::sort(free_ids);

  std::vector<mlir::Value> value_of(entries.size());
  for (int64_t id : free_ids) {
    const Entry &e = entries[id];
    mlir::Type type = valueType(context, loc, id, e.name, *e.type, q8_1.count(id) != 0);
    if (!type)
      return {};
    value_of[id] = body->addArgument(type, loc);
  }

  // The nodes, in document order.
  for (size_t index = 0; index < nodes.size(); ++index) {
    const LoweredNode &n = nodes[index];
    const sx::Node &source = *n.source;

    llvm::SmallVector<mlir::Value> operands;
    operands.reserve(n.inputs.size());
    for (int64_t id : n.inputs) {
      if (!value_of[id]) {
        mlir::emitError(loc) << "node '" << source.op_name << "' reads tensor id "
                             << id
                             << ", which no earlier node produces and no block "
                                "argument holds";
        return {};
      }
      operands.push_back(value_of[id]);
    }

    llvm::SmallVector<mlir::Type> results;
    llvm::SmallVector<int64_t> result_ids;
    results.reserve(n.outputs.size());
    result_ids.reserve(n.outputs.size());
    for (int64_t id : n.outputs) {
      const Entry &e = entries[id];
      mlir::Type type =
          valueType(context, loc, id, e.name, *e.type, q8_1.count(id) != 0);
      if (!type)
        return {};
      results.push_back(type);
      result_ids.push_back(id);
    }

    // The shared node record: the same constructor graphToBoom calls, with the
    // node name the document carries (adapt_graph preserves it verbatim) and
    // no backend override (adapt_graph leaves it unset).
    mlir::Operation *node = createBoomNode(
        builder, loc, n.op, operands, results, source.op_name, int64_t(index),
        result_ids, source.attributes, std::nullopt);
    if (!node)
      return {};

    for (auto [i, id] : llvm::enumerate(n.outputs))
      value_of[id] = node->getResult(i);
  }

  // Whatever was built must satisfy the dialect's verifier, here rather than
  // three passes later.
  if (mlir::failed(mlir::verify(module->getOperation())))
    return {};
  return module;
}

} // namespace sonicboom::compiler
