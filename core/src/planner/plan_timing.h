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

// Scoped wall-clock timers for the planning pipeline. Enabled by
// SONICBOOM_PLAN_TIMING (any value); disabled runs pay nothing beyond one
// std::getenv on first use. Output is one line per phase on stderr:
//
//   [plan] <phase> <ms>
//
// Used to keep the planning pipeline honest: `segment_plan` must stay in the
// low milliseconds so plan construction never shows up in prefill time.

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace sonicboom::planner::detail {

inline bool plan_timing_enabled() noexcept {
  static const bool enabled = std::getenv("SONICBOOM_PLAN_TIMING") != nullptr;
  return enabled;
}

class PlanTimer {
 public:
  explicit PlanTimer(const char* phase) : phase_(phase) {
    if (plan_timing_enabled()) start_ = Clock::now();
  }

  PlanTimer(const PlanTimer&) = delete;
  PlanTimer& operator=(const PlanTimer&) = delete;

  ~PlanTimer() {
    if (!plan_timing_enabled()) return;
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    std::fprintf(stderr, "[plan] %-24s %8.2f ms\n", phase_, ms);
  }

 private:
  using Clock = std::chrono::steady_clock;
  const char* phase_;
  Clock::time_point start_{};
};

} // namespace sonicboom::planner::detail
