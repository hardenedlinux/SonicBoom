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

#include <sonicboom/planner/graph.h>

#include "fingerprint.h"

#include <bit>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace sonicboom::planner {

const TensorDesc* Graph::find_tensor(TensorId id) const noexcept {
  for (const auto& t : tensors)
    if (t.id == id)
      return &t;
  return nullptr;
}

const GraphNodeDesc* Graph::find_node(GraphNodeId id) const noexcept {
  for (const auto& n : nodes)
    if (n.id == id)
      return &n;
  return nullptr;
}

namespace {

void hash_attr(detail::Fnv1a& h, const sx::Attribute& a) {
  h.str(a.name);
  h.u8(static_cast<uint8_t>(a.kind()));
  using K = sx::AttrValueKind;
  switch (a.kind()) {
    case K::Int:
      h.u64(static_cast<uint64_t>(std::get<int64_t>(a.value)));
      break;
    case K::Float:
      h.u64(std::bit_cast<uint64_t>(std::get<double>(a.value)));
      break;
    case K::String:
      h.str(std::get<std::string>(a.value));
      break;
    case K::Bool:
      h.u8(std::get<bool>(a.value) ? 1 : 0);
      break;
    case K::Ints: {
      const auto& v = std::get<std::vector<int64_t>>(a.value);
      h.u32(static_cast<uint32_t>(v.size()));
      for (int64_t x : v)
        h.u64(static_cast<uint64_t>(x));
      break;
    }
    case K::Floats: {
      const auto& v = std::get<std::vector<double>>(a.value);
      h.u32(static_cast<uint32_t>(v.size()));
      for (double x : v)
        h.u64(std::bit_cast<uint64_t>(x));
      break;
    }
    case K::Strings: {
      const auto& v = std::get<std::vector<std::string>>(a.value);
      h.u32(static_cast<uint32_t>(v.size()));
      for (const auto& s : v)
        h.str(s);
      break;
    }
    case K::Bools: {
      const auto& v = std::get<std::vector<bool>>(a.value);
      h.u32(static_cast<uint32_t>(v.size()));
      for (bool b : v)
        h.u8(b ? 1 : 0);
      break;
    }
  }
}

} // namespace

uint64_t model_fingerprint(const Graph& g) noexcept {
  detail::Fnv1a h;
  h.str(g.name);
  h.u32(static_cast<uint32_t>(g.tensors.size()));
  for (const auto& t : g.tensors) {
    h.u32(t.id.value);
    h.u32(static_cast<uint32_t>(t.shape.size()));
    for (int64_t d : t.shape)
      h.u64(static_cast<uint64_t>(d));
    h.u8(static_cast<uint8_t>(t.dtype));
    h.u64(t.size_bytes);
    h.u8(t.is_graph_input ? 1 : 0);
    h.u8(t.is_graph_output ? 1 : 0);
    h.u8(t.is_constant ? 1 : 0);
    h.u8(t.is_external ? 1 : 0);
    h.str(t.name);
  }
  h.u32(static_cast<uint32_t>(g.nodes.size()));
  for (const auto& n : g.nodes) {
    h.u32(n.id.value);
    h.u8(static_cast<uint8_t>(n.op));
    h.u32(static_cast<uint32_t>(n.inputs.size()));
    for (TensorId id : n.inputs)
      h.u32(id.value);
    h.u32(static_cast<uint32_t>(n.outputs.size()));
    for (TensorId id : n.outputs)
      h.u32(id.value);
    h.u32(static_cast<uint32_t>(n.attributes.size()));
    for (const auto& a : n.attributes)
      hash_attr(h, a);
    h.str(n.name);
  }
  h.u32(static_cast<uint32_t>(g.inputs.size()));
  for (TensorId id : g.inputs)
    h.u32(id.value);
  h.u32(static_cast<uint32_t>(g.outputs.size()));
  for (TensorId id : g.outputs)
    h.u32(id.value);
  return h.get();
}

std::expected<Graph, PlannerError> adapt_graph(const sx::Document& doc) {
  const sx::Graph& sg = doc.graph;
  Graph g;
  g.name = sg.name;

  std::unordered_map<std::string, TensorId> by_name;
  uint32_t next_tensor = 0;
  uint32_t next_node = 0;

  auto make_tensor =
      [&](const std::string& name, const sx::TensorType& type, bool is_input,
          bool is_constant, bool is_external)
      -> std::expected<TensorId, PlannerError> {
    if (by_name.count(name))
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidGraph,
          "duplicate value definition '" + name + "'", Phase::GraphAnalysis));
    auto sz = sx::tensor_byte_size(type);
    if (!sz)
      return std::unexpected(PlannerError(
          PlannerErrorCode::ArithmeticOverflow,
          "tensor '" + name + "' byte size overflows", Phase::GraphAnalysis));
    TensorDesc td;
    td.id = TensorId{next_tensor};
    td.shape = type.shape.dims;
    td.dtype = type.dtype;
    td.size_bytes = static_cast<uint64_t>(*sz);
    td.is_graph_input = is_input;
    td.is_constant = is_constant;
    td.is_external = is_external;
    td.name = name;
    TensorId id = td.id;
    g.tensors.push_back(std::move(td));
    by_name.emplace(name, id);
    if (!next_id(next_tensor))
      return std::unexpected(PlannerError(PlannerErrorCode::InternalError,
                                          "tensor id counter overflow",
                                          Phase::GraphAnalysis));
    return id;
  };

  // 1. Graph inputs (deterministic ids, document order).
  std::vector<TensorId> input_ids;
  for (const auto& in : sg.inputs) {
    auto r = make_tensor(in.name, in.type, /*is_input=*/true, false, false);
    if (!r)
      return std::unexpected(r.error());
    input_ids.push_back(*r);
  }

  // 2. Parameters (weights/constants), document order.
  for (const auto& p : sg.parameters) {
    bool is_external = std::holds_alternative<sx::ExternalData>(p.data);
    auto r = make_tensor(p.name, p.type, /*is_input=*/false, true, is_external);
    if (!r)
      return std::unexpected(r.error());
  }

  // 3. Node outputs (deterministic ids, node order); inputs resolve to prior
  //    values (inputs, parameters, or earlier node outputs).
  for (const auto& node : sg.nodes) {
    auto op = op_kind_from_name(node.op_name);
    if (!op)
      return std::unexpected(PlannerError(
          PlannerErrorCode::UnsupportedOperator,
          "unsupported operator '" + node.op_name + "'", Phase::GraphAnalysis));

    GraphNodeDesc nd;
    nd.id = GraphNodeId{next_node};
    nd.op = *op;
    nd.name = node.op_name;

    for (const auto& in : node.inputs) {
      auto it = by_name.find(in);
      if (it == by_name.end())
        return std::unexpected(PlannerError(
            PlannerErrorCode::InvalidGraph,
            "node '" + node.op_name + "' references unknown value '" + in + "'",
            Phase::GraphAnalysis));
      nd.inputs.push_back(it->second);
    }
    for (const auto& o : node.outputs) {
      auto r = make_tensor(o.name, o.type, /*is_input=*/false, false, false);
      if (!r)
        return std::unexpected(r.error());
      nd.outputs.push_back(*r);
    }
    nd.attributes = node.attributes;
    g.nodes.push_back(std::move(nd));
    if (!next_id(next_node))
      return std::unexpected(PlannerError(PlannerErrorCode::InternalError,
                                          "node id counter overflow",
                                          Phase::GraphAnalysis));
  }

  // 4. Graph outputs resolve; mark them.
  for (const auto& out_name : sg.outputs) {
    auto it = by_name.find(out_name);
    if (it == by_name.end())
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidGraph,
          "graph output references unknown value '" + out_name + "'",
          Phase::GraphAnalysis));
    g.outputs.push_back(it->second);
  }
  for (TensorId id : g.outputs)
    for (auto& t : g.tensors)
      if (t.id == id)
        t.is_graph_output = true;

  g.inputs = std::move(input_ids);
  return g;
}

} // namespace sonicboom::planner
