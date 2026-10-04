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

c10::Layout to_aten(Layout l) {
  switch (l) {
    case Layout::Strided:
      return c10::Layout::Strided;
    case Layout::Sparse:
      return c10::Layout::Sparse;
  }
  TORCH_CHECK(false, "unsupported nt::Layout");
  return c10::Layout::Strided; // unreachable
}

Layout from_aten(c10::Layout l) {
  switch (l) {
    case c10::Layout::Strided:
      return Layout::Strided;
    case c10::Layout::Sparse:
      return Layout::Sparse;
    default:
      break;
  }
  TORCH_CHECK(false, "unsupported c10::Layout");
  return Layout::Strided; // unreachable
}

} // namespace detail
} // namespace nt
