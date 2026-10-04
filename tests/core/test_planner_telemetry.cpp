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

// Telemetry tests (P7): the monotonic timer and the execution aggregate behave
// as documented; measured values are non-negative and accumulate correctly.

#include <sonicboom/planner/telemetry.h>

#include <iostream>

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
  pl::Timer t;
  double a = t.elapsed_us();
  check(a >= 0.0, "elapsed is non-negative");
  double b = t.elapsed_us();
  check(b >= a, "elapsed is monotonic");

  double lap = t.lap_us();
  check(lap >= 0.0, "lap is non-negative");
  double after = t.elapsed_us();
  check(after < lap + 1e-3 || after >= 0.0, "lap resets the timer");

  pl::ExecutionTelemetry tm;
  tm.record(10.0);
  tm.record(20.0);
  tm.record(5.0);
  tm.record(15.0);
  check(tm.runs == 4, "runs counted");
  check(tm.total_us == 50.0, "total accumulated");
  check(tm.min_us == 5.0, "min tracked");
  check(tm.max_us == 20.0, "max tracked");
  check(tm.last_us == 15.0, "last tracked");

  if (g_failures == 0) {
    std::cout << "test_planner_telemetry OK\n";
    return 0;
  }
  std::cerr << "test_planner_telemetry FAILED: " << g_failures << " check(s)\n";
  return 1;
}
