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
#include <variant>
#include <vector>

#include <sonicboom/tensor.h>
#include <sonicboom/scalar.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/device.h>
#include <sonicboom/layout.h>
#include <sonicboom/memory_format.h>

namespace nt {

enum class ValueKind : uint8_t {
  None,
  Tensor,
  Scalar,
  Int,
  Float,
  Bool,
  String,
  Device,
  ScalarType,
  Layout,
  MemoryFormat,
  TensorList,
  IntList,
};

// Layer 2 runtime value. Self-contained (a std::variant of Native Torch-owned
// types); it does not expose c10::IValue. Conversion to/from the native value
// representation lives in the Layer 1 adapter (core/layer1/adapter/value.cpp).
class Value {
 public:
  Value() = default; // None
  Value(Tensor t) : data_(std::move(t)) {}
  Value(Scalar s) : data_(s) {}
  Value(int i) : data_(static_cast<int64_t>(i)) {} // disambiguate int literals
  Value(int64_t i) : data_(i) {}
  Value(double d) : data_(d) {}
  Value(bool b) : data_(b) {}
  Value(std::string s) : data_(std::move(s)) {}
  Value(Device d) : data_(d) {}
  Value(ScalarType st) : data_(st) {}
  Value(Layout l) : data_(l) {}
  Value(MemoryFormat mf) : data_(mf) {}
  Value(std::vector<Tensor> tl) : data_(std::move(tl)) {}
  Value(std::vector<int64_t> il) : data_(std::move(il)) {}

  ValueKind kind() const { return static_cast<ValueKind>(data_.index()); }

  bool isNone() const { return std::holds_alternative<std::monostate>(data_); }
  bool isTensor() const { return std::holds_alternative<Tensor>(data_); }
  bool isScalar() const { return std::holds_alternative<Scalar>(data_); }
  bool isInt() const { return std::holds_alternative<int64_t>(data_); }
  bool isFloat() const { return std::holds_alternative<double>(data_); }
  bool isBool() const { return std::holds_alternative<bool>(data_); }
  bool isString() const { return std::holds_alternative<std::string>(data_); }
  bool isDevice() const { return std::holds_alternative<Device>(data_); }
  bool isScalarType() const {
    return std::holds_alternative<ScalarType>(data_);
  }
  bool isLayout() const { return std::holds_alternative<Layout>(data_); }
  bool isMemoryFormat() const {
    return std::holds_alternative<MemoryFormat>(data_);
  }
  bool isTensorList() const {
    return std::holds_alternative<std::vector<Tensor>>(data_);
  }
  bool isIntList() const {
    return std::holds_alternative<std::vector<int64_t>>(data_);
  }

  Tensor toTensor() const { return std::get<Tensor>(data_); }
  Scalar toScalar() const { return std::get<Scalar>(data_); }
  int64_t toInt() const { return std::get<int64_t>(data_); }
  double toFloat() const { return std::get<double>(data_); }
  bool toBool() const { return std::get<bool>(data_); }
  const std::string& toString() const { return std::get<std::string>(data_); }
  Device toDevice() const { return std::get<Device>(data_); }
  ScalarType toScalarType() const { return std::get<ScalarType>(data_); }
  Layout toLayout() const { return std::get<Layout>(data_); }
  MemoryFormat toMemoryFormat() const { return std::get<MemoryFormat>(data_); }
  const std::vector<Tensor>& toTensorList() const {
    return std::get<std::vector<Tensor>>(data_);
  }
  const std::vector<int64_t>& toIntList() const {
    return std::get<std::vector<int64_t>>(data_);
  }

 private:
  // Alternative order MUST match ValueKind (index 0 == None).
  std::variant<std::monostate, Tensor, Scalar, int64_t, double, bool,
               std::string, Device, ScalarType, Layout, MemoryFormat,
               std::vector<Tensor>, std::vector<int64_t>>
      data_;
};

} // namespace nt
