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

#include <cstdint>
#include <string>
#include <utility>

namespace nt {

// Native Torch-owned operator argument kind. v0 required kinds (manifest M2).
enum class ArgKind : uint8_t {
  Tensor,
  OptionalTensor,
  TensorList,
  OptionalTensorList,
  Scalar,
  Int,
  IntList,
  Float,
  Bool,
  String,
  Device,
  ScalarType,
  Layout,
  MemoryFormat,
  None,
};

struct Argument {
  std::string name;
  ArgKind kind = ArgKind::None;

  Argument() = default;
  Argument(std::string n, ArgKind k) : name(std::move(n)), kind(k) {}
};

} // namespace nt
