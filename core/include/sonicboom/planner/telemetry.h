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

// Measured runtime telemetry. Kept strictly separate from the cost model's
// analytical estimates (cost.h): those are predictions, these are what the CPU
// executor actually observed. v0 records only what it can honestly measure —
// wall-clock time via a monotonic clock — and never fabricates counters for
// unimplemented capabilities (async, transfers, device memory).

#include <chrono>
#include <cstdint>

namespace sonicboom::planner {

// Monotonic wall-clock timer (std::chrono::steady_clock). Measures elapsed
// microseconds; non-copyable semantics are unnecessary since it holds only a
// single time point.
class Timer {
public:
  Timer() : t0_(std::chrono::steady_clock::now()) {}

  void start() noexcept { t0_ = std::chrono::steady_clock::now(); }

  // Microseconds since the last start() (or construction).
  double elapsed_us() const noexcept {
    return std::chrono::duration<double, std::micro>(
               std::chrono::steady_clock::now() - t0_)
        .count();
  }

  // Return elapsed microseconds and restart from now.
  double lap_us() noexcept {
    double e = elapsed_us();
    t0_ = std::chrono::steady_clock::now();
    return e;
  }

private:
  std::chrono::steady_clock::time_point t0_;
};

// Aggregate of measured execution runs.
struct ExecutionTelemetry {
  uint64_t runs = 0;
  double total_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
  double last_us = 0.0;

  // Fold one measured run into the aggregate.
  void record(double elapsed_us) noexcept {
    ++runs;
    total_us += elapsed_us;
    last_us = elapsed_us;
    if (runs == 1) {
      min_us = max_us = elapsed_us;
      return;
    }
    if (elapsed_us < min_us)
      min_us = elapsed_us;
    if (elapsed_us > max_us)
      max_us = elapsed_us;
  }
};

} // namespace sonicboom::planner
