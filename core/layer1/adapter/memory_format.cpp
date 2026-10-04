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

c10::MemoryFormat to_aten(MemoryFormat m) {
  switch (m) {
    case MemoryFormat::Contiguous:
      return c10::MemoryFormat::Contiguous;
    case MemoryFormat::Preserve:
      return c10::MemoryFormat::Preserve;
    case MemoryFormat::ChannelsLast:
      return c10::MemoryFormat::ChannelsLast;
  }
  TORCH_CHECK(false, "unsupported nt::MemoryFormat");
  return c10::MemoryFormat::Contiguous; // unreachable
}

MemoryFormat from_aten(c10::MemoryFormat m) {
  switch (m) {
    case c10::MemoryFormat::Contiguous:
      return MemoryFormat::Contiguous;
    case c10::MemoryFormat::Preserve:
      return MemoryFormat::Preserve;
    case c10::MemoryFormat::ChannelsLast:
      return MemoryFormat::ChannelsLast;
    default:
      break;
  }
  TORCH_CHECK(false, "unsupported c10::MemoryFormat");
  return MemoryFormat::Contiguous; // unreachable
}

} // namespace detail
} // namespace nt
