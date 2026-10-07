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

#pragma once

// Planner-facing graph IR. This is a *view* over the frozen S-Expr v0.1 IR
// (sx::Document), not a replacement: the adapter (adapt_graph) preserves node
// identity, producer/consumer relationships, attributes, and static shape/dtype
// information while assigning stable strongly-typed IDs for planning. The
// frozen sx IR is never mutated.

#include <sonicboom/planner/errors.h>
#include <sonicboom/planner/ids.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/planner/partition.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace sonicboom::planner {

// A tensor in the planner vocabulary, identified by a stable TensorId. Identity
// and byte size are computed from the source sx type (checked for overflow).
struct TensorDesc {
  TensorId id;
  std::vector<int64_t> shape;
  sx::DType dtype;
  uint64_t size_bytes = 0;

  bool is_graph_input = false;
  bool is_graph_output = false;
  bool is_constant = false;  // a parameter (weight or inline constant)
  bool is_external = false;  // an external weight-sidecar parameter

  std::string name;  // original sx value name (identity/diagnostics)
};

// A single operator node. Attributes are preserved verbatim so the lowering
// path (conv pads/strides, gemm transA/transB, …) loses no information.
struct GraphNodeDesc {
  GraphNodeId id;
  OpKind op;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;
  std::vector<sx::Attribute> attributes;
  std::string name;  // original sx op name (identity/diagnostics)

  // Optional per-node backend override. When set, the planner routes this node
  // to `backend` regardless of `route_op(op)`. The Gemma 4 plan emitter pins
  // every node to BackendTag::Sonic (its `Add` residual nodes would otherwise
  // route to Mlir, the frozen S-Expr default for the shared `Add` operator).
  // The sx::Document → adapt_graph path leaves this unset, so route_op applies.
  std::optional<BackendTag> backend;
};

struct Graph {
  std::string name;
  std::vector<TensorDesc> tensors;
  std::vector<GraphNodeDesc> nodes;
  std::vector<TensorId> inputs;
  std::vector<TensorId> outputs;

  const TensorDesc* find_tensor(TensorId id) const noexcept;
  const GraphNodeDesc* find_node(GraphNodeId id) const noexcept;
};

// Deterministic 64-bit model fingerprint (FNV-1a over tensors, nodes, and
// attributes). Non-cryptographic; documented as such.
uint64_t model_fingerprint(const Graph& g) noexcept;

// Adapt a parsed sx::Document into a planner-facing Graph. Assigns deterministic
// TensorIds (inputs, then parameters, then node outputs, in document order) and
// GraphNodeIds (node order). Rejects (with a PlannerError):
//   - an operator outside the planner vocabulary (UnsupportedOperator);
//   - a tensor whose byte size overflows (ArithmeticOverflow);
//   - an unresolved input/output reference or duplicate definition
//     (InvalidGraph), as a defensive re-check over a hand-built Document.
std::expected<Graph, PlannerError> adapt_graph(const sx::Document& doc);

} // namespace sonicboom::planner
