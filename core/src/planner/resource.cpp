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

#include <sonicboom/planner/resource.h>

#include "fingerprint.h"

#include <algorithm>
#include <cstdint>
#include <unordered_set>
#include <utility>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

namespace sonicboom::planner {

const char* device_kind_name(DeviceKind k) noexcept {
  switch (k) {
    case DeviceKind::CPU: return "cpu";
    case DeviceKind::GPU: return "gpu";
    case DeviceKind::Accelerator: return "accelerator";
  }
  return "?";
}

const char* memory_kind_name(MemoryKind k) noexcept {
  switch (k) {
    case MemoryKind::Host: return "host";
    case MemoryKind::DeviceLocal: return "device_local";
    case MemoryKind::PinnedHost: return "pinned_host";
    case MemoryKind::PersistentStorage: return "persistent_storage";
  }
  return "?";
}

bool DeviceCapability::supports(OpKind op, sx::DType dtype) const noexcept {
  if (!can_execute)
    return false;
  bool op_ok = false;
  for (OpKind o : supported_ops)
    if (o == op) {
      op_ok = true;
      break;
    }
  if (!op_ok)
    return false;
  for (sx::DType d : supported_dtypes)
    if (d == dtype)
      return true;
  return false;
}

std::optional<uint64_t> MemorySpace::effective_budget_bytes() const noexcept {
  if (capacity_bytes == 0)
    return std::nullopt;  // unknown capacity, documented
  if (reserved_bytes > capacity_bytes)
    return std::nullopt;  // invalid (validated elsewhere); never wrap
  return capacity_bytes - reserved_bytes;
}

const Device* ResourceSnapshot::find_device(DeviceId id) const noexcept {
  for (const auto& d : devices)
    if (d.id == id)
      return &d;
  return nullptr;
}

const MemorySpace* ResourceSnapshot::find_memory_space(MemorySpaceId id) const noexcept {
  for (const auto& m : memory_spaces)
    if (m.id == id)
      return &m;
  return nullptr;
}

uint64_t resource_fingerprint(const ResourceSnapshot& s) noexcept {
  detail::Fnv1a h;
  h.u32(s.version);
  h.u32(static_cast<uint32_t>(s.devices.size()));
  for (const auto& d : s.devices) {
    h.u32(d.id.value);
    h.u8(static_cast<uint8_t>(d.kind));
    h.str(d.name);
    h.u8(d.capability.can_execute ? 1 : 0);
    h.u8(d.capability.supports_async ? 1 : 0);
    h.u8(d.capability.supports_concurrent_copy ? 1 : 0);
    h.u32(static_cast<uint32_t>(d.capability.supported_ops.size()));
    for (OpKind o : d.capability.supported_ops)
      h.u8(static_cast<uint8_t>(o));
    h.u32(static_cast<uint32_t>(d.capability.supported_dtypes.size()));
    for (sx::DType t : d.capability.supported_dtypes)
      h.u8(static_cast<uint8_t>(t));
  }
  h.u32(static_cast<uint32_t>(s.memory_spaces.size()));
  for (const auto& m : s.memory_spaces) {
    h.u32(m.id.value);
    h.u32(m.owner.value);
    h.u8(static_cast<uint8_t>(m.kind));
    h.u64(m.capacity_bytes);
    h.u64(m.reserved_bytes);
    h.u64(m.alignment_bytes);
  }
  return h.get();
}

std::expected<void, PlannerError> validate_snapshot(const ResourceSnapshot& s) {
  if (s.devices.empty())
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidConfiguration, "resource snapshot has no devices",
        Phase::Resource));

  std::unordered_set<uint32_t> device_ids;
  bool has_exec = false;
  for (const auto& d : s.devices) {
    if (!device_ids.insert(d.id.value).second)
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidConfiguration,
          "duplicate device id " + std::to_string(d.id.value), Phase::Resource));
    if (d.capability.can_execute)
      has_exec = true;
  }
  if (!has_exec)
    return std::unexpected(PlannerError(
        PlannerErrorCode::UnsupportedCapability,
        "resource snapshot has no device capable of execution", Phase::Resource));

  std::unordered_set<uint32_t> space_ids;
  for (const auto& m : s.memory_spaces) {
    if (!space_ids.insert(m.id.value).second)
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidConfiguration,
          "duplicate memory space id " + std::to_string(m.id.value),
          Phase::Resource));
    if (!s.find_device(m.owner))
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidConfiguration,
          "memory space " + std::to_string(m.id.value) + " references unknown device " +
              std::to_string(m.owner.value),
          Phase::Resource));
    if (m.alignment_bytes == 0)
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidConfiguration,
          "memory space " + std::to_string(m.id.value) + " has zero alignment",
          Phase::Resource));
    if (m.capacity_bytes != 0 && m.reserved_bytes > m.capacity_bytes)
      return std::unexpected(PlannerError(
          PlannerErrorCode::InvalidConfiguration,
          "memory space " + std::to_string(m.id.value) + " reserves " +
              std::to_string(m.reserved_bytes) + " bytes over capacity " +
              std::to_string(m.capacity_bytes),
          Phase::Resource));
  }
  return {};
}

// --- CPU provider -----------------------------------------------------------

namespace {

// Conservative default host budget when the OS cannot report physical memory.
constexpr uint64_t kFallbackHostBudgetBytes = 1ULL << 30;  // 1 GiB

uint64_t auto_host_budget_bytes() {
#if defined(__unix__) || defined(__APPLE__)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    uint64_t phys = static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    // 80% of physical RAM: conservative (never "all RAM"), leaves headroom for
    // the OS, the JIT, and other processes.
    return phys / 5 * 4;
  }
#endif
  return kFallbackHostBudgetBytes;
}

} // namespace

std::expected<ResourceSnapshot, PlannerError> CpuResourceProvider::snapshot(
    const CpuProviderConfig& cfg) {
  uint64_t budget = cfg.host_memory_budget_bytes;
  if (budget == 0)
    budget = auto_host_budget_bytes();
  if (cfg.reserved_bytes > budget)
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidConfiguration,
        "reserved " + std::to_string(cfg.reserved_bytes) + " bytes exceeds host budget " +
            std::to_string(budget),
        Phase::Resource));
  if (cfg.alignment_bytes == 0)
    return std::unexpected(PlannerError(
        PlannerErrorCode::InvalidConfiguration, "alignment must be non-zero",
        Phase::Resource));

  ResourceSnapshot s;
  s.version = 1;

  Device cpu;
  cpu.id = DeviceId{0};
  cpu.kind = DeviceKind::CPU;
  cpu.name = "cpu";
  cpu.capability.can_execute = true;
  cpu.capability.supports_async = false;           // honest: no async impl
  cpu.capability.supports_concurrent_copy = false; // honest: no concurrent copy
  cpu.capability.supported_ops = {
      OpKind::Conv,   OpKind::Relu,    OpKind::Add,       OpKind::MaxPool,
      OpKind::ReduceMean, OpKind::Reshape, OpKind::Gemm,
  };
  // v0 execution dtype: float32 compute (input/output). Integer parameters
  // (reshape/reduce axes) are baked constants, not computed tensors.
  cpu.capability.supported_dtypes = {sx::DType::Float32};
  s.devices.push_back(std::move(cpu));

  MemorySpace host;
  host.id = MemorySpaceId{0};
  host.owner = DeviceId{0};
  host.kind = MemoryKind::Host;
  host.capacity_bytes = budget;
  host.reserved_bytes = cfg.reserved_bytes;
  host.alignment_bytes = cfg.alignment_bytes;
  s.memory_spaces.push_back(std::move(host));

  s.fingerprint = resource_fingerprint(s);
  return s;
}

} // namespace sonicboom::planner
