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

#include <sonicboom/sx/ir.h>

namespace sonicboom::sx {

const char* dtype_name(DType d) {
  switch (d) {
    case DType::Float32: return "float32";
    case DType::Float16: return "float16";
    case DType::BFloat16: return "bfloat16";
    case DType::Float64: return "float64";
    case DType::Int8: return "int8";
    case DType::UInt8: return "uint8";
    case DType::Int16: return "int16";
    case DType::Int32: return "int32";
    case DType::Int64: return "int64";
    case DType::Bool: return "bool";
  }
  return "?";
}

} // namespace sonicboom::sx
