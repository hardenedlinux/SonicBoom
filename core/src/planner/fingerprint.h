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

// PRIVATE header (not installed) implementing a deterministic FNV-1a digest for
// resource and model fingerprints. The digest is non-cryptographic: it is only
// a collision-resistant-enough deterministic identity, documented as such. All
// integers are serialized little-endian (endian-independent); strings are
// length-prefixed so distinct sequences cannot alias.

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace sonicboom::planner::detail {

class Fnv1a {
public:
  // Feed a scalar as its little-endian byte sequence (endian-independent).
  void u8(uint8_t v) noexcept {
    h_ ^= v;
    h_ *= kPrime;
  }
  void u16(uint16_t v) noexcept {
    for (int i = 0; i < 2; ++i)
      u8(static_cast<uint8_t>(v >> (i * 8)));
  }
  void u32(uint32_t v) noexcept {
    for (int i = 0; i < 4; ++i)
      u8(static_cast<uint8_t>(v >> (i * 8)));
  }
  void u64(uint64_t v) noexcept {
    for (int i = 0; i < 8; ++i)
      u8(static_cast<uint8_t>(v >> (i * 8)));
  }
  // Length-prefixed string (u64 length, then bytes).
  void str(std::string_view s) noexcept {
    u64(static_cast<uint64_t>(s.size()));
    for (unsigned char c : s)
      u8(c);
  }

  uint64_t get() const noexcept { return h_; }

private:
  uint64_t h_ = kOffset;
  static constexpr uint64_t kOffset = 14695981039346656037ULL;
  static constexpr uint64_t kPrime = 1099511628211ULL;
};

} // namespace sonicboom::planner::detail
