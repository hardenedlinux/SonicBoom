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

#include <sonicboom/planner/native_torch_backend.h>

#include <sonicboom/operator_handle.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/tensor.h>
#include <sonicboom/value.h>

#include <cstring>
#include <optional>
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

// v0 bridge supports the float-family and integer dtypes that softmax (and the
// immediate follow-up ops) actually need. Anything else is an explicit error,
// never a silent guess.
std::optional<nt::ScalarType> sx_to_nt(sx::DType d) {
  switch (d) {
    case sx::DType::Float32: return nt::ScalarType::Float;
    case sx::DType::Float64: return nt::ScalarType::Double;
    case sx::DType::Int64:   return nt::ScalarType::Long;
    case sx::DType::Int32:   return nt::ScalarType::Int;
    case sx::DType::Bool:    return nt::ScalarType::Bool;
    default:                 return std::nullopt;
  }
}

const sx::Attribute* find_attr(const std::vector<sx::Attribute>& attrs,
                               const std::string& name) {
  for (const auto& a : attrs)
    if (a.name == name)
      return &a;
  return nullptr;
}

int64_t numel_of(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape)
    n *= d;
  return n;
}

} // namespace

std::expected<std::unique_ptr<NativeTorchBackend>, RuntimeError>
NativeTorchBackend::compile(const Graph& graph) {
  return std::unique_ptr<NativeTorchBackend>(new NativeTorchBackend(graph));
}

NativeTorchBackend::NativeTorchBackend(Graph graph) : graph_(std::move(graph)) {}

std::expected<std::vector<TensorValue>, RuntimeError> NativeTorchBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& inputs) {
  const auto* compute = task.as_compute();
  if (!compute)
    return std::unexpected(plan_err("native torch backend requires a compute task"));

  // v0: exactly one node per native-torch compute task.
  if (compute->graph_nodes.size() != 1)
    return std::unexpected(
        plan_err("native torch backend expects a single-node task (v0)"));
  const GraphNodeDesc* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(plan_err("native torch task references unknown node"));
  if (node->inputs.size() != inputs.size())
    return std::unexpected(
        plan_err("native torch task input buffer count mismatch"));

  // 1. sx::Bytes → nt::Tensor (explicit copy). Shape/dtype come from the plan's
  //    TensorDesc, never inferred from byte length.
  std::vector<nt::Tensor> tensors;
  tensors.reserve(node->inputs.size());
  for (std::size_t i = 0; i < node->inputs.size(); ++i) {
    const TensorDesc* t = graph_.find_tensor(node->inputs[i]);
    if (!t)
      return std::unexpected(
          plan_err("native torch task input tensor has no descriptor"));
    auto nt_dtype = sx_to_nt(t->dtype);
    if (!nt_dtype)
      return std::unexpected(rt_err("unsupported dtype for native torch bridge"));
    if (inputs[i].on_device)
      return std::unexpected(rt_err("native torch backend received a device input"));
    if (inputs[i].host.size() != static_cast<std::size_t>(t->size_bytes))
      return std::unexpected(rt_err("input byte count does not match tensor '" +
                                    t->name + "'"));
    nt::Tensor tt = nt::empty(t->shape, *nt_dtype);
    std::memcpy(tt.data_ptr(), inputs[i].host.data(), inputs[i].host.size());
    tensors.push_back(std::move(tt));
  }

  // 2. Op name + attribute mapping (v0: softmax only), planner-owned.
  std::string aten_name;
  nt::ArgumentList args;
  if (node->op == OpKind::Softmax) {
    aten_name = "aten::_softmax";
    int64_t axis = -1;  // ONNX Softmax default axis
    if (const sx::Attribute* a = find_attr(node->attributes, "axis"))
      if (std::holds_alternative<int64_t>(a->value))
        axis = std::get<int64_t>(a->value);
    args.push_back(nt::Value(tensors[0]));      // self
    args.push_back(nt::Value(axis));            // dim
    args.push_back(nt::Value(false));           // half_to_float
  } else {
    return std::unexpected(rt_err("native torch backend has no mapping for op '" +
                                  std::string(op_kind_name(node->op)) + "'"));
  }

  // 3. Dispatch through the Layer 2 interface; map any c10 exception.
  nt::ResultList result;
  try {
    nt::OperatorHandle op = nt::find_operator(nt::OperatorName{aten_name});
    result = op.call(args);
  } catch (const std::exception& e) {
    return std::unexpected(rt_err(std::string("native torch dispatch failed: ") +
                                  e.what()));
  }

  // 4. nt::Tensor → sx::Bytes (explicit copy), with shape/dtype sanity checks.
  if (result.size() != node->outputs.size())
    return std::unexpected(
        plan_err("native torch op produced an unexpected result count"));
  if (!result[0].isTensor())
    return std::unexpected(rt_err("native torch op did not return a tensor"));

  nt::Tensor out_t = result[0].toTensor();
  const TensorDesc* od = graph_.find_tensor(node->outputs.front());
  if (!od)
    return std::unexpected(
        plan_err("native torch task output tensor has no descriptor"));
  auto expect_dtype = sx_to_nt(od->dtype);
  if (!expect_dtype || out_t.dtype() != *expect_dtype)
    return std::unexpected(rt_err("native torch result dtype does not match plan"));
  if (out_t.numel() != numel_of(od->shape))
    return std::unexpected(rt_err("native torch result numel does not match plan"));

  sx::Bytes out_bytes(static_cast<std::size_t>(od->size_bytes));
  std::memcpy(out_bytes.data(), out_t.data_ptr(), out_bytes.size());

  std::vector<TensorValue> outputs;
  outputs.push_back(TensorValue::from_host(std::move(out_bytes)));
  return outputs;
}

} // namespace sonicboom::planner
