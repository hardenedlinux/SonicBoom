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

// Strongly-typed identifiers for the static execution planner. Each kind is a
// distinct C++ type so that GraphNodeId / TaskId / TensorId / BufferId /
// JitEntryId cannot be silently mixed up; they share a numeric storage layout
// only for compactness. IDs are unique within their own scope and stable for
// the lifetime of one plan.

#include <cstdint>

namespace sonicboom::planner {

struct DeviceId {
  uint32_t value;
  auto operator<=>(const DeviceId&) const = default;
};
struct MemorySpaceId {
  uint32_t value;
  auto operator<=>(const MemorySpaceId&) const = default;
};
struct TensorId {
  uint32_t value;
  auto operator<=>(const TensorId&) const = default;
};
struct BufferId {
  uint32_t value;
  auto operator<=>(const BufferId&) const = default;
};
struct TaskId {
  uint32_t value;
  auto operator<=>(const TaskId&) const = default;
};
struct GraphNodeId {
  uint32_t value;
  auto operator<=>(const GraphNodeId&) const = default;
};
struct JitEntryId {
  uint32_t value;
  auto operator<=>(const JitEntryId&) const = default;
};

// Deterministic, overflow-checked counter increment. Returns false (leaving
// `id` untouched) if the increment would wrap past UINT32_MAX, so a wrapped ID
// can never alias an existing one. Callers must propagate the failure.
inline bool next_id(uint32_t& id) noexcept {
  if (id == UINT32_MAX)
    return false;
  ++id;
  return true;
}

} // namespace sonicboom::planner
