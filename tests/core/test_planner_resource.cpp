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

// Resource model tests (P1): valid snapshot, invalid configurations, honest
// capability reporting, effective-budget arithmetic, and deterministic
// fingerprinting.

#include <sonicboom/planner/resource.h>

#include <sonicboom/quant/quantized_matmul.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace pl = sonicboom::planner;

namespace {
int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}
} // namespace

int main() {
  // --- valid CPU/Host snapshot ---------------------------------------------
  auto snap = pl::CpuResourceProvider::snapshot({.host_memory_budget_bytes = 1024,
                                                 .reserved_bytes = 64,
                                                 .alignment_bytes = 64});
  check(snap.has_value(), "CPU snapshot builds");
  if (!snap)
    return 1;
  // M3: the CPU provider also enumerates a CUDA device + device-local space
  // when one is present, so the device/space counts follow cuda_available().
  const bool has_cuda = sonicboom::quant::cuda_available();
  check(snap->devices.size() == (has_cuda ? 2u : 1u),
        "device count (CPU + optional GPU)");
  check(snap->memory_spaces.size() == (has_cuda ? 2u : 1u),
        "memory space count (host + optional device-local)");
  check(snap->devices[0].kind == pl::DeviceKind::CPU, "device is CPU");
  check(snap->devices[0].capability.can_execute, "CPU can execute");
  check(!snap->devices[0].capability.supports_async, "no async claimed");
  check(!snap->devices[0].capability.supports_concurrent_copy,
        "no concurrent copy claimed");
  check(snap->memory_spaces[0].kind == pl::MemoryKind::Host, "host space");
  check(snap->memory_spaces[0].owner == pl::DeviceId{0}, "host owned by CPU");

  auto budget = snap->memory_spaces[0].effective_budget_bytes();
  check(budget.has_value() && *budget == 1024 - 64, "effective budget = 960");

  // --- capability: supported vs unsupported --------------------------------
  auto& cap = snap->devices[0].capability;
  check(cap.supports(pl::OpKind::Conv, sonicboom::sx::DType::Float32),
        "supports conv/float32");
  check(!cap.supports(pl::OpKind::Conv, sonicboom::sx::DType::Int64),
        "does not support conv/int64");
  check(!cap.supports(pl::OpKind::WholeGraph, sonicboom::sx::DType::Float32),
        "WholeGraph is not a device op capability");

  // --- validate_snapshot: valid --------------------------------------------
  auto v = pl::validate_snapshot(*snap);
  check(v.has_value(), "valid snapshot validates");

  // --- reserved > capacity rejected ----------------------------------------
  pl::ResourceSnapshot bad = *snap;
  bad.memory_spaces[0].capacity_bytes = 64;
  bad.memory_spaces[0].reserved_bytes = 128;
  auto bv = pl::validate_snapshot(bad);
  check(!bv.has_value(), "reserved > capacity rejected");

  // provider-level reserved > budget rejected
  auto badcfg = pl::CpuResourceProvider::snapshot(
      {.host_memory_budget_bytes = 64, .reserved_bytes = 128});
  check(!badcfg.has_value(), "provider reserved > budget rejected");

  // --- unknown capacity (0) is representable, not "infinite" ---------------
  pl::ResourceSnapshot unk = *snap;
  unk.memory_spaces[0].capacity_bytes = 0;
  check(!unk.memory_spaces[0].effective_budget_bytes().has_value(),
        "unknown capacity -> no effective budget");
  auto uv = pl::validate_snapshot(unk);
  check(uv.has_value(), "unknown-capacity snapshot still valid");

  // --- invalid (zero) alignment rejected -----------------------------------
  pl::ResourceSnapshot al = *snap;
  al.memory_spaces[0].alignment_bytes = 0;
  check(!pl::validate_snapshot(al).has_value(), "zero alignment rejected");

  // --- duplicate device id rejected ----------------------------------------
  pl::ResourceSnapshot dup = *snap;
  dup.devices.push_back(dup.devices[0]);
  check(!pl::validate_snapshot(dup).has_value(), "duplicate device id rejected");

  // --- deterministic fingerprint -------------------------------------------
  uint64_t fp1 = pl::resource_fingerprint(*snap);
  uint64_t fp2 = pl::resource_fingerprint(*snap);
  check(fp1 == fp2, "fingerprint deterministic");

  auto snap2 = pl::CpuResourceProvider::snapshot(
      {.host_memory_budget_bytes = 2048, .reserved_bytes = 64, .alignment_bytes = 64});
  check(snap2.has_value() &&
            pl::resource_fingerprint(*snap2) != fp1,
        "fingerprint changes with capacity");

  pl::ResourceSnapshot cap_changed = *snap;
  cap_changed.devices[0].capability.supported_ops.push_back(pl::OpKind::WholeGraph);
  check(pl::resource_fingerprint(cap_changed) != fp1,
        "fingerprint changes with capability");

  if (g_failures == 0) {
    std::cout << "test_planner_resource OK\n";
    return 0;
  }
  std::cerr << "test_planner_resource FAILED: " << g_failures << " check(s)\n";
  return 1;
}
