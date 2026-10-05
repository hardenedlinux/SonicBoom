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

// The planner's operator vocabulary. The concrete operators mirror the frozen
// S-Expr v0.1 supported operator set (used for per-node capability and cost
// analysis); `Softmax` is a native-torch-routed operator (no MLIR lowering);
// `WholeGraph` is a task-granularity marker for the single whole-graph compute
// task the v0 compiler emits, not a node-level operator.

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace sonicboom::planner {

enum class OpKind : uint8_t {
  Conv,
  Relu,
  Add,
  MaxPool,
  ReduceMean,
  Reshape,
  Gemm,
  Softmax,
  WholeGraph,
};

// Map a canonical S-Expr op name (e.g. "conv") to its OpKind; nullopt if the
// name is not one of the seven supported operators.
std::optional<OpKind> op_kind_from_name(std::string_view name) noexcept;

const char* op_kind_name(OpKind k) noexcept;

} // namespace sonicboom::planner
