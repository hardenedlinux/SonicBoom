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

// The backend a compute task is dispatched to. Distinct from DeviceKind: both
// backends run on CPU in v0, so device kind alone cannot route a task.
enum class BackendTag : uint8_t {
  Mlir,        // sx::Executable (CpuBackend), the 7-op lowering set
  NativeTorch, // nt::OperatorHandle dispatcher (NativeTorchBackend)
};
const char* backend_tag_name(BackendTag t) noexcept;

// Default backend for an operator. This is the *entire* backend-selection seam:
// replacing the routing strategy means replacing this one function, not
// threading conditionals through the planner and executor.
inline BackendTag route_op(OpKind op) noexcept {
  switch (op) {
    case OpKind::Softmax:
      return BackendTag::NativeTorch;
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

} // namespace sonicboom::planner
