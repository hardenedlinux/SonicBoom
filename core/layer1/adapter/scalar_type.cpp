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

#include "adapter.h"

namespace nt {
namespace detail {

c10::ScalarType to_aten(ScalarType s) {
  switch (s) {
    case ScalarType::Byte:
      return c10::ScalarType::Byte;
    case ScalarType::Char:
      return c10::ScalarType::Char;
    case ScalarType::Short:
      return c10::ScalarType::Short;
    case ScalarType::Int:
      return c10::ScalarType::Int;
    case ScalarType::Long:
      return c10::ScalarType::Long;
    case ScalarType::Half:
      return c10::ScalarType::Half;
    case ScalarType::Float:
      return c10::ScalarType::Float;
    case ScalarType::Double:
      return c10::ScalarType::Double;
    case ScalarType::ComplexFloat:
      return c10::ScalarType::ComplexFloat;
    case ScalarType::ComplexDouble:
      return c10::ScalarType::ComplexDouble;
    case ScalarType::Bool:
      return c10::ScalarType::Bool;
    case ScalarType::BFloat16:
      return c10::ScalarType::BFloat16;
  }
  TORCH_CHECK(false, "unsupported nt::ScalarType");
  return c10::ScalarType::Float; // unreachable
}

ScalarType from_aten(c10::ScalarType s) {
  switch (s) {
    case c10::ScalarType::Byte:
      return ScalarType::Byte;
    case c10::ScalarType::Char:
      return ScalarType::Char;
    case c10::ScalarType::Short:
      return ScalarType::Short;
    case c10::ScalarType::Int:
      return ScalarType::Int;
    case c10::ScalarType::Long:
      return ScalarType::Long;
    case c10::ScalarType::Half:
      return ScalarType::Half;
    case c10::ScalarType::Float:
      return ScalarType::Float;
    case c10::ScalarType::Double:
      return ScalarType::Double;
    case c10::ScalarType::ComplexFloat:
      return ScalarType::ComplexFloat;
    case c10::ScalarType::ComplexDouble:
      return ScalarType::ComplexDouble;
    case c10::ScalarType::Bool:
      return ScalarType::Bool;
    case c10::ScalarType::BFloat16:
      return ScalarType::BFloat16;
    default:
      break;
  }
  TORCH_CHECK(false, "unsupported c10::ScalarType");
  return ScalarType::Float; // unreachable
}

} // namespace detail
} // namespace nt
