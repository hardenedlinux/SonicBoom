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

c10::Scalar to_aten(const Scalar& s) {
  if (s.isInt()) {
    return c10::Scalar(s.toInt());
  }
  if (s.isDouble()) {
    return c10::Scalar(s.toDouble());
  }
  return c10::Scalar(s.toBool());
}

Scalar from_aten(const c10::Scalar& s) {
  if (s.isBoolean()) {
    return Scalar(s.to<bool>());
  }
  if (s.isFloatingPoint()) {
    return Scalar(s.to<double>());
  }
  return Scalar(s.to<int64_t>());
}

} // namespace detail
} // namespace nt
