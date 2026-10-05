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

c10::IValue to_aten(const Value& v) {
  switch (v.kind()) {
    case ValueKind::None:
      return c10::IValue();
    case ValueKind::Tensor:
      return c10::IValue(to_aten(v.toTensor()));
    case ValueKind::Scalar:
      return c10::IValue(to_aten(v.toScalar()));
    case ValueKind::Int:
      return c10::IValue(v.toInt());
    case ValueKind::Float:
      return c10::IValue(v.toFloat());
    case ValueKind::Bool:
      return c10::IValue(v.toBool());
    case ValueKind::String:
      return c10::IValue(v.toString());
    case ValueKind::Device:
      return c10::IValue(to_aten(v.toDevice()));
    case ValueKind::ScalarType:
      return c10::IValue(to_aten(v.toScalarType()));
    case ValueKind::Layout:
      return c10::IValue(to_aten(v.toLayout()));
    case ValueKind::MemoryFormat:
      return c10::IValue(to_aten(v.toMemoryFormat()));
    case ValueKind::TensorList: {
      std::vector<at::Tensor> vec;
      vec.reserve(v.toTensorList().size());
      for (const auto& t : v.toTensorList()) {
        vec.push_back(to_aten(t));
      }
      return c10::IValue(std::move(vec));
    }
    case ValueKind::IntList:
      // A plain int64 list. For operators whose schema declares SymInt[]
      // (e.g. conv2d stride/padding/dilation), the boxed kernel unboxes via
      // IValue::toSymIntList(), which accepts an int list (constant SymInts
      // decay to plain ints), so no SymInt representation is required in v0.
      return c10::IValue(v.toIntList());
  }
  TORCH_CHECK(false, "unsupported nt::ValueKind");
  return c10::IValue(); // unreachable
}

Value from_aten(const c10::IValue& iv) {
  if (iv.isNone()) {
    return Value();
  }
  if (iv.isTensor()) {
    return Value(from_aten(iv.toTensor()));
  }
  if (iv.isInt()) {
    return Value(iv.toInt());
  }
  if (iv.isDouble()) {
    return Value(iv.toDouble());
  }
  if (iv.isBool()) {
    return Value(iv.toBool());
  }
  if (iv.isString()) {
    return Value(iv.toStringRef());
  }
  if (iv.isDevice()) {
    return Value(from_aten(iv.toDevice()));
  }
  if (iv.isTensorList()) {
    std::vector<Tensor> vec;
    for (const auto& t : iv.toTensorVector()) {
      vec.push_back(from_aten(t));
    }
    return Value(std::move(vec));
  }
  if (iv.isIntList()) {
    return Value(iv.toIntVector());
  }
  if (iv.isScalar()) {
    return Value(from_aten(iv.toScalar()));
  }
  TORCH_CHECK(false, "unsupported c10::IValue kind");
  return Value(); // unreachable
}

} // namespace detail
} // namespace nt
