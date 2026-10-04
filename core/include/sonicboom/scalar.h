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
#include <variant>

namespace nt {

// Type-erased scalar value (int / float / bool). Complex is deferred (v0).
class Scalar {
 public:
  Scalar() = default;
  Scalar(int i) : data_(static_cast<int64_t>(i)) {} // disambiguate int literals
  Scalar(int64_t i) : data_(i) {}
  Scalar(double d) : data_(d) {}
  Scalar(bool b) : data_(b) {}

  bool isInt() const { return std::holds_alternative<int64_t>(data_); }
  bool isDouble() const { return std::holds_alternative<double>(data_); }
  bool isBool() const { return std::holds_alternative<bool>(data_); }

  int64_t toInt() const { return std::get<int64_t>(data_); }
  double toDouble() const { return std::get<double>(data_); }
  bool toBool() const { return std::get<bool>(data_); }

 private:
  std::variant<int64_t, double, bool> data_;
};

} // namespace nt
