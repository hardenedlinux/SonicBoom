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

#include <sonicboom/planner/cpu_kernel_backend.h>

#include <sonicboom/planner/op_kind.h>
#include <sonicboom/sx/ir.h>

#include <string>
#include <utility>
#include <vector>

namespace sonicboom::planner {

namespace {

RuntimeError rt_err(std::string msg) {
  return RuntimeError(RuntimeErrorCode::BackendFailure, std::move(msg),
                      Phase::Execution);
}

RuntimeError plan_err(std::string msg) {
  return RuntimeError(RuntimeErrorCode::InvalidPlan, std::move(msg),
                      Phase::Execution);
}

sx::TensorType tensor_type(const TensorDesc& t) {
  return sx::TensorType{t.dtype, sx::Shape{t.shape}};
}

// One graph node as a one-node Blade document: graph inputs/outputs are the
// node's operands/results in node order, the operator name is the canonical
// S-Expr spelling of the op kind, and the attributes are carried verbatim.
// Parameter operands are rejected by the caller, so the document needs no
// parameters and compiles with an empty weight map.
sx::Document node_document(const Graph& graph, const GraphNodeDesc& node) {
  sx::Document doc{};
  doc.version_major = 0;
  doc.version_minor = 1;

  sx::Graph& g = doc.graph;
  g.name = graph.name.empty() ? node.name : graph.name + "/" + node.name;
  g.opsets.push_back(sx::Graph::Opset{"default", 1});

  sx::Node n;
  n.op_name = op_kind_name(node.op);
  n.attributes = node.attributes;
  for (const TensorId in : node.inputs) {
    const TensorDesc* t = graph.find_tensor(in);
    g.inputs.push_back(sx::Graph::Input{t->name, tensor_type(*t)});
    n.inputs.push_back(t->name);
  }
  for (const TensorId out : node.outputs) {
    const TensorDesc* t = graph.find_tensor(out);
    n.outputs.push_back(sx::Node::Output{t->name, tensor_type(*t)});
    g.outputs.push_back(t->name);
  }
  g.nodes.push_back(std::move(n));
  return doc;
}

} // namespace

std::expected<std::unique_ptr<CpuKernelBackend>, RuntimeError>
CpuKernelBackend::compile(const Graph& graph) {
  return std::unique_ptr<CpuKernelBackend>(new CpuKernelBackend(graph));
}

CpuKernelBackend::CpuKernelBackend(Graph graph) : graph_(std::move(graph)) {}

std::expected<std::vector<TensorValue>, RuntimeError>
CpuKernelBackend::execute(const TaskDesc& task,
                          const std::vector<TensorValue>& inputs) {
  const auto* compute = task.as_compute();
  if (!compute)
    return std::unexpected(plan_err("cpu kernel backend requires a compute task"));

  // One node per compute task, as the partition guarantees.
  if (compute->graph_nodes.size() != 1)
    return std::unexpected(
        plan_err("cpu kernel backend expects a single-node task"));
  const GraphNodeDesc* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(plan_err("cpu kernel task references unknown node"));
  if (node->inputs.size() != inputs.size())
    return std::unexpected(
        plan_err("cpu kernel task input buffer count mismatch"));

  // 1. Bind the raw inputs to the node's operands. Shapes and dtypes come from
  //    the plan's TensorDescs, never from the byte length.
  std::vector<sx::Bytes> operands;
  operands.reserve(inputs.size());
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    const TensorDesc* t = graph_.find_tensor(node->inputs[i]);
    if (!t)
      return std::unexpected(
          plan_err("cpu kernel task input tensor has no descriptor"));
    if (t->is_constant)
      return std::unexpected(plan_err("cpu kernel task binds a constant as an "
                                      "input operand ('" +
                                      t->name + "')"));
    if (inputs[i].on_device)
      return std::unexpected(
          rt_err("cpu kernel backend received a device input"));
    if (inputs[i].host.size() != static_cast<std::size_t>(t->size_bytes))
      return std::unexpected(rt_err("input byte count does not match tensor '" +
                                    t->name + "'"));
    operands.push_back(inputs[i].host);
  }

  // 2. Compile the single-node document on first use. A node that cannot be
  //    lowered reports the failure as an execution error, once, on every
  //    subsequent call as well.
  auto& program = programs_[node->id.value];
  if (!program.exe && program.error.empty()) {
    const sx::Document doc = node_document(graph_, *node);
    auto compiled = sx::Executable::compile(doc, {});
    if (!compiled) {
      program.error = "cpu kernel backend cannot compile node '" + node->name +
                      "': " + compiled.error().message;
    } else {
      program.exe = std::move(*compiled);
    }
  }
  if (!program.exe) return std::unexpected(rt_err(program.error));

  // 3. Run over host buffers; the executor copies in and out.
  std::vector<sx::Bytes> results;
  auto run = program.exe->run(operands, results);
  if (!run) return std::unexpected(rt_err(run.error().message));

  if (results.size() != node->outputs.size())
    return std::unexpected(
        rt_err("cpu kernel executor produced an unexpected result count"));

  std::vector<TensorValue> outputs;
  outputs.reserve(results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    const TensorDesc* od = graph_.find_tensor(node->outputs[i]);
    if (!od)
      return std::unexpected(
          plan_err("cpu kernel task output tensor has no descriptor"));
    if (results[i].size() != static_cast<std::size_t>(od->size_bytes))
      return std::unexpected(
          rt_err("cpu kernel result byte count does not match plan"));
    outputs.push_back(TensorValue::from_host(std::move(results[i])));
  }
  return outputs;
}

} // namespace sonicboom::planner
