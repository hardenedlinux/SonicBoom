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

// Cost value types. Time is in microseconds; bandwidth is in bytes per
// microsecond; sizes are in bytes. These are deterministic estimates, kept
// strictly separate from measured telemetry. Invalid values (negative latency/
// bandwidth, NaN/Inf, out-of-range confidence) are rejected by the cost model
// rather than being propagated as fictitious zero or garbage costs.

#include <cstdint>

namespace sonicboom::planner {

struct ComputeCost {
  double estimated_latency_us = 0.0;
  double confidence = 1.0;  // in [0, 1]; 1.0 = fully determined

  bool valid() const noexcept;
};

struct TransferCost {
  double fixed_latency_us = 0.0;
  double effective_bandwidth_bytes_per_us = 0.0;  // must be > 0 for a transfer

  bool valid() const noexcept;
};

struct MemoryCost {
  uint64_t allocation_overhead_bytes = 0;
  double allocation_latency_us = 0.0;

  bool valid() const noexcept;
};

struct PlanCostSummary {
  double estimated_compute_us = 0.0;
  double estimated_transfer_us = 0.0;
  double estimated_memory_us = 0.0;
  double estimated_total_us = 0.0;

  // Sum the components into estimated_total_us, rejecting arithmetic that would
  // overflow or produce a non-finite total. Returns false on overflow/non-finite
  // (leaving estimated_total_us untouched).
  bool summarize() noexcept;
};

} // namespace sonicboom::planner
