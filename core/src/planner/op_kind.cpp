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

#include <sonicboom/planner/op_kind.h>

namespace sonicboom::planner {

std::optional<OpKind> op_kind_from_name(std::string_view name) noexcept {
  if (name == "conv")
    return OpKind::Conv;
  if (name == "relu")
    return OpKind::Relu;
  if (name == "add")
    return OpKind::Add;
  if (name == "max_pool")
    return OpKind::MaxPool;
  if (name == "reduce_mean")
    return OpKind::ReduceMean;
  if (name == "reshape")
    return OpKind::Reshape;
  if (name == "gemm")
    return OpKind::Gemm;
  return std::nullopt;
}

const char* op_kind_name(OpKind k) noexcept {
  switch (k) {
    case OpKind::Conv: return "conv";
    case OpKind::Relu: return "relu";
    case OpKind::Add: return "add";
    case OpKind::MaxPool: return "max_pool";
    case OpKind::ReduceMean: return "reduce_mean";
    case OpKind::Reshape: return "reshape";
    case OpKind::Gemm: return "gemm";
    case OpKind::WholeGraph: return "whole_graph";
  }
  return "?";
}

} // namespace sonicboom::planner
