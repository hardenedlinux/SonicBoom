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

c10::Device to_aten(const Device& d) {
  TORCH_CHECK(d.type == DeviceType::CPU, "v0 supports the CPU device only");
  return c10::Device(c10::DeviceType::CPU, static_cast<int8_t>(d.index));
}

Device from_aten(const c10::Device& d) {
  TORCH_CHECK(d.type() == c10::DeviceType::CPU, "v0 supports the CPU device only");
  // c10 uses index -1 to mean "current/default" device; v0 CPU is
  // single-device, so normalize to index 0 (matching Device::cpu()).
  return Device{DeviceType::CPU, 0};
}

} // namespace detail
} // namespace nt
