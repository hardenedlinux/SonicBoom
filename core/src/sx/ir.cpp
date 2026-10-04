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

std::size_t dtype_size(DType d) {
  switch (d) {
    case DType::Float32: return 4;
    case DType::Float16: return 2;
    case DType::BFloat16: return 2;
    case DType::Float64: return 8;
    case DType::Int8:  return 1;
    case DType::UInt8: return 1;
    case DType::Bool:  return 1;
    case DType::Int16: return 2;
    case DType::Int32: return 4;
    case DType::Int64: return 8;
  }
  return 0;
}

std::optional<int64_t> numel_checked(const Shape& s) {
  int64_t n = 1;
  for (int64_t d : s.dims) {
    if (d < 0)
      return std::nullopt;
    int64_t next = 0;
    if (__builtin_mul_overflow(n, d, &next))
      return std::nullopt;
    n = next;
  }
  return n;
}

int64_t numel(const Shape& s) {
  return numel_checked(s).value_or(-1);
}

int64_t numel(const TensorType& t) {
  return numel(t.shape);
}

std::optional<int64_t> tensor_byte_size(const TensorType& t) {
  auto n = numel_checked(t.shape);
  if (!n)
    return std::nullopt;
  int64_t es = static_cast<int64_t>(dtype_size(t.dtype));
  if (es <= 0)
    return std::nullopt;
  int64_t bytes = 0;
  if (__builtin_mul_overflow(*n, es, &bytes))
    return std::nullopt;
  return bytes;
}

} // namespace sonicboom::sx
