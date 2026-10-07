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

#include <sonicboom/planner/partition.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sonicboom::planner {

const char* backend_tag_name(BackendTag t) noexcept {
  switch (t) {
    case BackendTag::Mlir: return "mlir";
    case BackendTag::NativeTorch: return "native_torch";
    case BackendTag::Sonic: return "sonic";
  }
  return "?";
}

namespace {
PlannerError slice_err(std::string msg) {
  return PlannerError(PlannerErrorCode::InvalidGraph, std::move(msg),
                      Phase::GraphAnalysis);
}

// Shared core for slice_document and slice_region. Keeps the nodes that produce
// any name in `kept_outputs`; exposes `region_outputs` (a subset of kept-node
// outputs) as the graph output list. Everything else (external inputs,
// parameter carry-over, node ordering) is identical between the two callers.
std::expected<sx::Document, PlannerError> slice_impl(
    const sx::Document& doc, const std::vector<std::string>& kept_outputs,
    const std::vector<std::string>& region_outputs) {
  const sx::Graph& sg = doc.graph;

  // 1. Value name → type, gathered from every SSA definition source.
  std::unordered_map<std::string, const sx::TensorType*> type_of;
  for (const auto& in : sg.inputs)
    type_of[in.name] = &in.type;
  for (const auto& p : sg.parameters)
    type_of[p.name] = &p.type;
  for (const auto& n : sg.nodes)
    for (const auto& o : n.outputs)
      type_of[o.name] = &o.type;

  // 2. Kept nodes = those producing at least one requested output.
  std::unordered_set<std::string> keep(kept_outputs.begin(), kept_outputs.end());
  std::vector<const sx::Node*> kept;
  for (const auto& n : sg.nodes) {
    bool is_kept = false;
    for (const auto& o : n.outputs)
      if (keep.count(o.name)) {
        is_kept = true;
        break;
      }
    if (is_kept)
      kept.push_back(&n);
  }
  if (kept.empty())
    return std::unexpected(slice_err("slice keeps no nodes"));

  // Values produced *inside the slice* = outputs of kept nodes only. Outputs of
  // excluded nodes (e.g. the native-torch softmax feeding this region) are NOT
  // produced here, so they surface as graph inputs below.
  std::unordered_set<std::string> produced;
  for (const auto* n : kept)
    for (const auto& o : n->outputs)
      produced.insert(o.name);

  for (const auto& o : region_outputs)
    if (!produced.count(o))
      return std::unexpected(slice_err("slice output '" + o +
                                       "' is not produced by a kept node"));

  // 3. Graph inputs = kept-node inputs not produced inside the slice, in first
  //    reference order.
  std::vector<sx::Graph::Input> ins;
  std::unordered_set<std::string> seen_in;
  for (const auto* n : kept)
    for (const auto& in : n->inputs) {
      if (produced.count(in) || seen_in.count(in))
        continue;
      auto it = type_of.find(in);
      if (it == type_of.end())
        return std::unexpected(slice_err("slice input '" + in +
                                         "' has no type"));
      ins.push_back(sx::Graph::Input{in, *it->second});
      seen_in.insert(in);
    }

  // 4. Parameters referenced by kept nodes (and not produced inside the slice).
  std::unordered_set<std::string> used;
  for (const auto* n : kept)
    for (const auto& in : n->inputs)
      used.insert(in);
  std::vector<sx::Parameter> params;
  for (const auto& p : sg.parameters)
    if (used.count(p.name) && !produced.count(p.name))
      params.push_back(p);

  // 5. Kept nodes verbatim, in original order.
  std::vector<sx::Node> nodes;
  nodes.reserve(kept.size());
  for (const auto* n : kept)
    nodes.push_back(*n);

  sx::Graph out;
  out.name = sg.name;
  out.opsets = sg.opsets;
  out.inputs = std::move(ins);
  out.outputs = region_outputs;
  out.parameters = std::move(params);
  out.nodes = std::move(nodes);

  sx::Document d;
  d.version_major = doc.version_major;
  d.version_minor = doc.version_minor;
  d.graph = std::move(out);
  return d;
}
} // namespace

std::expected<sx::Document, PlannerError> slice_document(
    const sx::Document& doc, const std::vector<std::string>& keep_outputs) {
  return slice_impl(doc, keep_outputs, keep_outputs);
}

std::expected<sx::Document, PlannerError> slice_region(
    const sx::Document& doc,
    const std::vector<std::string>& kept_node_outputs,
    const std::vector<std::string>& region_outputs) {
  return slice_impl(doc, kept_node_outputs, region_outputs);
}

} // namespace sonicboom::planner
