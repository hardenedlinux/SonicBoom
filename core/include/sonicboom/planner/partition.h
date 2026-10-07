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

// Co-execution partitioning seam. `BackendTag` names the execution backend that
// a compute task is dispatched to; `route_op` is the single, replaceable
// decision point that maps an operator to its default backend. `slice_document`
// extracts a contiguous S-Expr sub-graph into a standalone 1-in/1-out Document
// so the MLIR backend can compile a native-torch-bounded region independently.

#include <sonicboom/planner/errors.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <expected>
#include <string>
#include <vector>

namespace sonicboom::planner {

// The backend a compute task is dispatched to. Distinct from DeviceKind: the
// Mlir and NativeTorch backends both run on CPU in v0, so device kind alone
// cannot route a task; Sonic is the Phase 6 transformer backend (also CPU in
// M2, later GPU in M3).
enum class BackendTag : uint8_t {
  Mlir,        // sx::Executable (CpuBackend), the 7-op lowering set
  NativeTorch, // nt::OperatorHandle dispatcher (NativeTorchBackend)
  Sonic,       // nn::*/quant::* kernels (SonicBackend), the transformer path
};
const char* backend_tag_name(BackendTag t) noexcept;

// Default backend for an operator. This is the *entire* backend-selection seam:
// replacing the routing strategy means replacing this one function, not
// threading conditionals through the planner and executor. A graph node may
// override this default via GraphNodeDesc::backend (used by the Gemma 4 plan
// emitter to pin every node — including the shared `Add` — to Sonic); when no
// override is present, this function is the routing rule.
inline BackendTag route_op(OpKind op) noexcept {
  switch (op) {
    case OpKind::Softmax:
      return BackendTag::NativeTorch;
    // Phase 6 transformer operators. These are not part of the frozen S-Expr
    // v0.1 vocabulary, so this branch is only reachable through a model-aware
    // plan emitter (e.g. the Gemma 4 decode graph); it exists so the routing
    // seam stays total over OpKind.
    case OpKind::QuantizedMatmul:
    case OpKind::RmsNorm:
    case OpKind::RmsNormHeads:
    case OpKind::GeluFp16:
    case OpKind::Rope:
    case OpKind::Attention:
    case OpKind::AttentionShared:
    case OpKind::GqaBroadcast:
    case OpKind::Embedding:
    case OpKind::MatvecF32:
    case OpKind::MatvecBf16:
    case OpKind::Mul:
    case OpKind::Scale:
    case OpKind::CastFp16:
    case OpKind::Softcap:
    case OpKind::Argmax:
    case OpKind::LayerCombine:
      return BackendTag::Sonic;
    default:
      return BackendTag::Mlir;
  }
}

// Extract the nodes that produce `keep_outputs` (and only those nodes) into a
// standalone Document. Every SSA value a kept node references but does not
// produce becomes a graph input; `keep_outputs` becomes the graph output list.
// Parameters referenced by the slice are carried over verbatim. This is a pure
// sx::Document transform — it never touches the frozen IR format.
std::expected<sx::Document, PlannerError> slice_document(
    const sx::Document& doc, const std::vector<std::string>& keep_outputs);

// Extract an explicit node set (a backend region) into a standalone Document.
// `kept_node_outputs` names one output per kept node (v0 nodes are
// single-output), selecting exactly those nodes regardless of transitive
// dataflow; `region_outputs` (a subset of kept-node outputs) becomes the graph
// output list. Any value a kept node references but no kept node produces
// becomes a graph input; parameters referenced by the region are carried over
// verbatim. Used by the region-partitioned mixed path, where consecutive
// same-backend nodes are compiled as one unit.
std::expected<sx::Document, PlannerError> slice_region(
    const sx::Document& doc,
    const std::vector<std::string>& kept_node_outputs,
    const std::vector<std::string>& region_outputs);

} // namespace sonicboom::planner
