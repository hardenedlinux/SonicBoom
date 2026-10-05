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
  switch (d.type) {
    case DeviceType::CPU:
      return c10::Device(c10::DeviceType::CPU, static_cast<int8_t>(d.index));
    case DeviceType::CUDA:
      // c10::DeviceIndex is int8_t; the CUDA device ordinal is 0 for v0.
      return c10::Device(c10::DeviceType::CUDA, static_cast<int8_t>(d.index));
  }
  TORCH_CHECK(false, "unsupported nt::DeviceType");
  return c10::Device(c10::DeviceType::CPU); // unreachable
}

Device from_aten(const c10::Device& d) {
  switch (d.type()) {
    case c10::DeviceType::CPU:
      // c10 uses index -1 to mean "current/default" device; v0 CPU is
      // single-device, so normalize to index 0 (matching Device::cpu()).
      return Device{DeviceType::CPU, 0};
    case c10::DeviceType::CUDA:
      // Preserve the CUDA device ordinal (-1 means "current device" in c10;
      // v0 uses the explicit index, normalizing -1 to 0).
      return Device{DeviceType::CUDA,
                    d.index() < 0 ? 0 : static_cast<int32_t>(d.index())};
  }
  TORCH_CHECK(false, "unsupported c10::DeviceType");
  return Device{DeviceType::CPU, 0}; // unreachable
}

} // namespace detail
} // namespace nt
