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

namespace nt {

// CPU and CUDA. ROCm / other accelerators are deferred manifest amendments and
// must not appear here.
enum class DeviceType : uint8_t {
  CPU,
  CUDA,
};

struct Device {
  DeviceType type = DeviceType::CPU;
  int32_t index = 0; // device ordinal; 0 for CPU, the CUDA device index for CUDA.

  static constexpr Device cpu() { return Device{DeviceType::CPU, 0}; }
  static constexpr Device cuda(int32_t index = 0) {
    return Device{DeviceType::CUDA, index};
  }
};

inline bool operator==(const Device& a, const Device& b) {
  return a.type == b.type && a.index == b.index;
}
inline bool operator!=(const Device& a, const Device& b) { return !(a == b); }

} // namespace nt
