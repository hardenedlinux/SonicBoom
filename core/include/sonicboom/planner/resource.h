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

// Unified physical resource model. A *device* (compute) and a *memory space*
// (storage) are distinct concepts: a CPU is a device, host RAM is a memory
// space. GPU/accelerator kinds and non-host memory kinds exist so a capability
// can be *declared* honestly, but in v0 only CPU compute and Host memory are
// actually implemented and reported as available; nothing fabricates GPU,
// pinned-memory, or asynchronous capabilities from CPU data.

#include <sonicboom/planner/errors.h>
#include <sonicboom/planner/ids.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace sonicboom::planner {

enum class DeviceKind : uint8_t { CPU, GPU, Accelerator };
const char* device_kind_name(DeviceKind k) noexcept;

enum class MemoryKind : uint8_t {
  Host,
  DeviceLocal,
  PinnedHost,
  PersistentStorage,
};
const char* memory_kind_name(MemoryKind k) noexcept;

// What a device can actually do. Every flag defaults to false; a provider sets
// a flag only when the corresponding implementation exists and is verified.
struct DeviceCapability {
  bool can_execute = false;
  bool supports_async = false;
  bool supports_concurrent_copy = false;

  std::vector<OpKind> supported_ops;
  std::vector<sx::DType> supported_dtypes;

  // True iff the device can execute `op` on `dtype` (compute capability only).
  bool supports(OpKind op, sx::DType dtype) const noexcept;
};

struct Device {
  DeviceId id;
  DeviceKind kind;
  std::string name;
  DeviceCapability capability;
};

struct MemorySpace {
  MemorySpaceId id;
  DeviceId owner;
  MemoryKind kind;

  uint64_t capacity_bytes = 0;  // 0 == unknown (documented, never "infinite")
  uint64_t reserved_bytes = 0;
  uint64_t alignment_bytes = 1;

  // capacity - reserved; nullopt when capacity is unknown. Reserved > capacity
  // is an invalid configuration detected by validate_snapshot, never wrapped.
  std::optional<uint64_t> effective_budget_bytes() const noexcept;
};

// Immutable snapshot of the resources a plan is validated and executed against.
struct ResourceSnapshot {
  uint32_t version = 0;
  uint64_t fingerprint = 0;

  std::vector<Device> devices;
  std::vector<MemorySpace> memory_spaces;

  const Device* find_device(DeviceId id) const noexcept;
  const MemorySpace* find_memory_space(MemorySpaceId id) const noexcept;
};

// Deterministic 64-bit fingerprint (FNV-1a over a canonical serialization of
// every planning-relevant property). Not cryptographic; documented as such.
uint64_t resource_fingerprint(const ResourceSnapshot& s) noexcept;

// Validate a snapshot's structural invariants. Returns the specific
// PlannerError on failure; success means the snapshot is internally consistent.
std::expected<void, PlannerError> validate_snapshot(const ResourceSnapshot& s);

// ---------------------------------------------------------------------------
// CPU v0 resource provider
// ---------------------------------------------------------------------------

struct CpuProviderConfig {
  // Host memory budget available to this execution. 0 = auto: a conservative
  // fraction of physical RAM reported by the OS (never "all RAM"). An explicit
  // value must be non-zero.
  uint64_t host_memory_budget_bytes = 0;
  uint64_t reserved_bytes = 0;   // runtime/JIT reservations, configurable
  uint64_t alignment_bytes = 64; // host allocation alignment
};

class CpuResourceProvider {
public:
  // Build the v0 CPU snapshot: one CPU device (can_execute, all 7 ops,
  // float32 compute dtype, no async / concurrent copy) + one Host memory space.
  // Returns a PlannerError if the configuration is impossible (e.g. reserved >
  // budget, or the OS could not report a usable budget for auto mode).
  static std::expected<ResourceSnapshot, PlannerError> snapshot(
      const CpuProviderConfig& cfg = {});
};

} // namespace sonicboom::planner
