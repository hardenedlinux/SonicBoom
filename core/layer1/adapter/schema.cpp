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

namespace {

c10::TypePtr arg_type(ArgKind k) {
  switch (k) {
    case ArgKind::Tensor:
      return c10::TensorType::get();
    case ArgKind::OptionalTensor:
      return c10::OptionalType::ofTensor();
    case ArgKind::TensorList:
      return c10::ListType::ofTensors();
    case ArgKind::OptionalTensorList:
      return c10::OptionalType::create(c10::ListType::ofTensors());
    case ArgKind::Scalar:
      return c10::NumberType::get();
    case ArgKind::Int:
    case ArgKind::ScalarType:
    case ArgKind::Layout:
    case ArgKind::MemoryFormat:
      return c10::IntType::get();
    case ArgKind::IntList:
      return c10::ListType::ofInts();
    case ArgKind::Float:
      return c10::FloatType::get();
    case ArgKind::Bool:
      return c10::BoolType::get();
    case ArgKind::String:
      return c10::StringType::get();
    case ArgKind::Device:
      return c10::DeviceObjType::get();
    case ArgKind::None:
      return c10::NoneType::get();
  }
  TORCH_CHECK(false, "unsupported nt::ArgKind");
  return c10::TensorType::get(); // unreachable
}

ArgKind arg_kind_from_type(const c10::Type& t) {
  using c10::TypeKind;
  switch (t.kind()) {
    case TypeKind::TensorType:
      return ArgKind::Tensor;
    case TypeKind::ListType: {
      const auto& lt = static_cast<const c10::ListType&>(t);
      if (lt.getElementType()->kind() == TypeKind::TensorType) {
        return ArgKind::TensorList;
      }
      if (lt.getElementType()->kind() == TypeKind::IntType) {
        return ArgKind::IntList;
      }
      TORCH_CHECK(false, "unsupported list element type");
      return ArgKind::TensorList; // unreachable
    }
    case TypeKind::OptionalType: {
      const auto& ot = static_cast<const c10::OptionalType&>(t);
      if (ot.getElementType()->kind() == TypeKind::TensorType) {
        return ArgKind::OptionalTensor;
      }
      if (ot.getElementType()->kind() == TypeKind::ListType) {
        return ArgKind::OptionalTensorList;
      }
      TORCH_CHECK(false, "unsupported optional element type");
      return ArgKind::OptionalTensor; // unreachable
    }
    case TypeKind::NumberType:
      return ArgKind::Scalar;
    case TypeKind::IntType:
      return ArgKind::Int;
    case TypeKind::FloatType:
      return ArgKind::Float;
    case TypeKind::BoolType:
      return ArgKind::Bool;
    case TypeKind::StringType:
      return ArgKind::String;
    case TypeKind::DeviceObjType:
      return ArgKind::Device;
    case TypeKind::NoneType:
      return ArgKind::None;
    default:
      TORCH_CHECK(false, "unsupported c10::Type kind");
      return ArgKind::None; // unreachable
  }
}

} // namespace

c10::Argument to_aten(const Argument& a) {
  return c10::Argument(a.name, arg_type(a.kind));
}

c10::FunctionSchema to_aten(const OperatorSchema& s) {
  std::vector<c10::Argument> args;
  std::vector<c10::Argument> rets;
  args.reserve(s.arguments().size());
  rets.reserve(s.returns().size());
  for (const auto& a : s.arguments()) {
    args.push_back(to_aten(a));
  }
  for (const auto& r : s.returns()) {
    rets.push_back(to_aten(r));
  }
  return c10::FunctionSchema(s.name(), s.overload_name(), std::move(args),
                             std::move(rets));
}

Argument from_aten(const c10::Argument& a) {
  return Argument(a.name(), arg_kind_from_type(*a.type()));
}

OperatorSchema from_aten(const c10::FunctionSchema& s) {
  std::vector<Argument> args;
  std::vector<Argument> rets;
  args.reserve(s.arguments().size());
  rets.reserve(s.returns().size());
  for (const auto& a : s.arguments()) {
    args.push_back(from_aten(a));
  }
  for (const auto& r : s.returns()) {
    rets.push_back(from_aten(r));
  }
  return OperatorSchema(s.name(), s.overload_name(), std::move(args),
                        std::move(rets));
}

} // namespace detail
} // namespace nt
