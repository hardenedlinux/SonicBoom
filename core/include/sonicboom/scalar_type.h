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

namespace nt {

// Layer 2 scalar type (dtype). A Native Torch-owned enum; it does not alias
// c10::ScalarType. The Layer 1 adapter maps it (see core/layer1/adapter).
// v0 subset: the dtypes the minimal runtime needs; float8 / unsigned dtypes are
// deferred per the manifest.
enum class ScalarType : uint8_t {
  Byte,
  Char,
  Short,
  Int,
  Long,
  Half,
  Float,
  Double,
  ComplexFloat,
  ComplexDouble,
  Bool,
  BFloat16,
};

} // namespace nt
