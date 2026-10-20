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

// Q1 runner: with synthetic compile closures, the runner must pick the
// fastest config that reproduces the reference output bit for bit, skip a
// config that does not compile, and reject a faster config whose output
// differs. It must do nothing unless SONICBOOM_AUTOTUNE=1.

#include "../../core/src/planner/mlir_autotune.h"

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>

namespace mt = sonicboom::mlir_tune;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

void spin_us(int us) {
  const auto until = std::chrono::steady_clock::now() +
                     std::chrono::microseconds(us);
  while (std::chrono::steady_clock::now() < until) {
  }
}

} // namespace

int main() {
  // slow: reference output, 150 us/launch. fast: identical output, 60 us.
  // liar: fastest of all but writes a different value. broken: does not
  // compile.
  const mt::TileConfig slow{512, 1, 32, 1, 1};
  const mt::TileConfig fast{128, 1, 32, 1, 1};
  const mt::TileConfig liar{256, 1, 32, 1, 1};
  const mt::TileConfig broken{999, 1, 32, 1, 1};

  uint64_t out = 0;
  mt::AutotuneRunner::Problem p;
  p.candidates = {slow, fast, liar, broken};
  p.output = &out;
  p.output_bytes = sizeof(out);
  p.iters = 5;
  p.compile = [&](const mt::TileConfig& c)
      -> std::optional<mt::AutotuneRunner::Launch> {
    if (c.tile_m == broken.tile_m) return std::nullopt;
    if (c.tile_m == liar.tile_m) {
      return mt::AutotuneRunner::Launch([&] {
        spin_us(5);
        out = 43;
      });
    }
    const bool is_fast = c.tile_m == fast.tile_m;
    return mt::AutotuneRunner::Launch([&, is_fast] {
      spin_us(is_fast ? 60 : 150);
      out = 42;
    });
  };

  // Gate: unset and 0 do nothing (no compile, no timing).
  {
    int compiled = 0;
    mt::AutotuneRunner::Problem gated = p;
    gated.compile = [&](const mt::TileConfig& c) {
      ++compiled;
      return p.compile(c);
    };
    ::unsetenv("SONICBOOM_AUTOTUNE");
    check(!mt::AutotuneRunner::run(gated), "gate unset: no run");
    ::setenv("SONICBOOM_AUTOTUNE", "0", 1);
    check(!mt::AutotuneRunner::run(gated), "gate 0: no run");
    check(compiled == 0, "gate off compiles nothing");
  }

  ::setenv("SONICBOOM_AUTOTUNE", "1", 1);

  // Repeat: timing noise must not change the winner.
  for (int r = 0; r < 3; ++r) {
    out = 0;
    const auto best = mt::AutotuneRunner::run(p);
    check(best && *best == fast,
          "the fastest bit-identical config wins");
    check(out == 42, "the output window holds the winner's result");
  }

  // A problem whose reference does not compile has no answer.
  {
    mt::AutotuneRunner::Problem no_ref = p;
    no_ref.candidates = {broken, fast};
    check(!mt::AutotuneRunner::run(no_ref), "no compilable reference: none");
  }

  // A single candidate: it is the reference and the answer.
  {
    mt::AutotuneRunner::Problem one = p;
    one.candidates = {slow};
    const auto best = mt::AutotuneRunner::run(one);
    check(best && *best == slow, "one candidate returns that candidate");
  }

  if (g_failures == 0) {
    std::cout << "test_autotune_runner OK\n";
    return 0;
  }
  std::cerr << "test_autotune_runner FAILED: " << g_failures << " check(s)\n";
  return 1;
}
