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

// Q1 cache round trip: the mmap'd per-SM tile-config file stores and returns
// records keyed by (op, shape class), updates a key in place, misses cleanly,
// and lives at $XDG_CACHE_HOME/sonicboom/autotune/sm_<NN>.bin. TileConfig's
// text form round-trips and rejects malformed input. The cache itself is not
// gated (the runner is), so this works with SONICBOOM_AUTOTUNE unset or 0.

#include "../../core/src/planner/mlir_autotune.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

namespace mt = sonicboom::mlir_tune;

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
  namespace fs = std::filesystem;

  char tmpl[] = "/tmp/sb-autotune-cache-XXXXXX";
  if (::mkdtemp(tmpl) == nullptr) {
    std::cerr << "mkdtemp failed\n";
    return 1;
  }
  const fs::path root(tmpl);
  ::setenv("XDG_CACHE_HOME", root.c_str(), 1);
  ::unsetenv("SONICBOOM_AUTOTUNE");

  auto& cache = mt::AutotuneCache::instance();
  cache.reset_for_test();

  const std::string kOp = "gemv";
  const std::string kShapeA = "bpr40-r1024";
  const std::string kShapeB = "bpr10-r256";

  check(!cache.lookup(kOp, kShapeA), "an empty cache misses");

  const mt::TileConfig a{128, 1, 64, 2, 4};
  const mt::TileConfig b{512, 1, 32, 1, 8};
  check(cache.store(kOp, kShapeA, a), "store shape A");
  check(cache.store(kOp, kShapeB, b), "store shape B");

  const auto ra = cache.lookup(kOp, kShapeA);
  check(ra && *ra == a, "shape A round-trips");
  const auto rb = cache.lookup(kOp, kShapeB);
  check(rb && *rb == b, "shape B round-trips");
  check(!cache.lookup(kOp, "bpr8-r128"), "an unrelated shape still misses");
  check(!cache.lookup("mmq", kShapeA), "an unrelated op misses");

  // Updating a key in place must replace, not append a second record.
  const mt::TileConfig a2{512, 1, 32, 1, 1};
  check(cache.store(kOp, kShapeA, a2), "update shape A");
  const auto ra2 = cache.lookup(kOp, kShapeA);
  check(ra2 && *ra2 == a2, "the update is what reads back");
  const auto rb2 = cache.lookup(kOp, kShapeB);
  check(rb2 && *rb2 == b, "the update leaves other keys alone");

  // Path: <cache>/sonicboom/autotune/sm_<NN>.bin.
  const std::string path = cache.path();
  const std::string suffix =
      std::string("/sonicboom/autotune/sm_") +
      (mt::sm_version() < 10 ? "0" : "") + std::to_string(mt::sm_version()) +
      ".bin";
  check(path.size() >= suffix.size() &&
            path.compare(path.size() - suffix.size(), suffix.size(), suffix) ==
                0,
        "cache path is .../sonicboom/autotune/sm_NN.bin");
  check(fs::exists(path), "the cache file exists after a store");

  // Serialization: every candidate and the default round-trip; malformed text
  // is rejected rather than silently defaulted.
  {
    const mt::TileConfig def = mt::default_config("gemv");
    const auto rt = mt::TileConfig::from_string(def.to_string());
    check(rt && *rt == def, "default config round-trips");
    for (const mt::TileConfig& c : mt::gemv_candidates()) {
      const auto r = mt::TileConfig::from_string(c.to_string());
      check(r && *r == c, "gemv candidate round-trips");
    }
    for (const char* bad : {"", "1,2,3", "1,2,3,4,0", "1,2,3,4,5x",
                            "1,2,3,4,5,6", "a,2,3,4,5"}) {
      check(!mt::TileConfig::from_string(bad), "malformed text is rejected");
    }
  }

  // Shape classes are stable and bucket rows, not exact sizes.
  check(mt::gemv_shape_class(40, 1000) == mt::gemv_shape_class(40, 1024),
        "the same row bucket has the same shape class");
  check(mt::gemv_shape_class(40, 1024) != mt::gemv_shape_class(40, 4096),
        "a different row bucket has a different shape class");
  check(mt::gemv_shape_class(8, 512) != mt::gemv_shape_class(40, 512),
        "different blocks-per-row differ");
  check(mt::gemv_shape_class(10, 10240) == "bpr10-r16384" &&
            mt::gemv_shape_class(10, 262144) == "bpr10-r262144",
        "rows above 16384 bucket separately (FFN vs logits)");
  check(mt::gemv_shape_class(10, 262144) == mt::gemv_shape_class(10, 400000),
        "the tallest bucket is clamped at r262144");

  // The runner gate, and the gemv lookup that honors it.
  {
    const std::string shape = mt::gemv_shape_class(99, 3000);  // bpr99-r4096
    check(cache.store(kOp, shape, b), "store the gate probe");
    ::setenv("SONICBOOM_AUTOTUNE", "1", 1);
    check(mt::enabled(), "SONICBOOM_AUTOTUNE=1 enables");
    check(mt::gemv_config(99, 3000) == b, "gate on: cached winner is used");
    ::setenv("SONICBOOM_AUTOTUNE", "0", 1);
    check(!mt::enabled(), "SONICBOOM_AUTOTUNE=0 disables");
    ::setenv("SONICBOOM_AUTOTUNE", "1", 1);
    ::setenv("SONICBOOM_AUTOTUNE_JIT", "0", 1);
    check(!mt::enabled(), "SONICBOOM_AUTOTUNE_JIT=0 disables the JIT service");
    ::unsetenv("SONICBOOM_AUTOTUNE_JIT");
    ::setenv("SONICBOOM_AUTOTUNE", "0", 1);
    check(mt::gemv_config(99, 3000) == mt::default_config("gemv"),
          "gate off: the static default, never the cache");
    ::unsetenv("SONICBOOM_AUTOTUNE");
    check(mt::enabled(), "gate unset: enabled by default");
    check(mt::gemv_config(99, 3000) == b,
          "gate unset: the cached winner is used");
  }

  // The cold budget: an exhausted budget resolves a fresh class to the static
  // default (Skipped, so the caller does not probe) instead of measuring it.
  {
    ::unsetenv("SONICBOOM_AUTOTUNE");
    ::setenv("SONICBOOM_AUTOTUNE_BUDGET_MS", "1", 1);
    mt::reset_budget_for_test();
    check(mt::budget_remaining_ms() <= 1, "the budget counts down from 1 ms");
    ::usleep(5000);
    check(mt::budget_exhausted(), "the budget is exhausted after 5 ms");
    const mt::GemvChoice skipped = mt::gemv_choice(77, 3000);
    check(skipped.resolved() && skipped.source == mt::GemvSource::Skipped &&
              skipped.cfg == mt::default_config("gemv"),
          "gemv: an exhausted budget resolves to the static default");
    const mt::MmqTcChoice mskip = mt::mmq_tc_choice(1024, 896, 896);
    check(mskip.resolved() && mskip.source == mt::MmqTcSource::Skipped &&
              mskip.cfg == mt::mmq_tc_default_config(),
          "mmq: an exhausted budget resolves to the static default");
    const mt::AttnPrefillChoice as =
        mt::attn_prefill_choice(128, 8192, 16);
    check(as.resolved() && as.source == mt::AttnPrefillSource::Skipped &&
              as.cfg.tile_m == 0,
          "attn: an exhausted budget resolves to the dispatcher shape");
    // A later, larger budget still cannot revive the classes this process
    // already resolved as Skipped, nor is the clock restarted.
    ::setenv("SONICBOOM_AUTOTUNE_BUDGET_MS", "10000", 1);
    check(mt::budget_exhausted(), "the budget is a one-shot process clock");
    check(mt::gemv_choice(77, 3000).source == mt::GemvSource::Skipped,
          "a skipped class stays skipped");
    ::unsetenv("SONICBOOM_AUTOTUNE_BUDGET_MS");
    mt::reset_budget_for_test();
  }

  // MMQ2: the generated int8 tensor-core matmul's shape class, candidate grid,
  // gate and persisted winner.
  {
    const std::string mkey = mt::mmq_tc_shape_class(512, 4864, 896);
    check(mt::mmq_tc_shape_class(4, 4864, 896) ==
              mt::mmq_tc_shape_class(7, 4864, 896),
          "mmq: M 4-7 share a bucket");
    check(mt::mmq_tc_shape_class(8, 4864, 896) !=
              mt::mmq_tc_shape_class(7, 4864, 896),
          "mmq: a different M bucket differs");
    check(mt::mmq_tc_shape_class(1024, 4864, 896) ==
              mt::mmq_tc_shape_class(4096, 4864, 896),
          "mmq: 1024+ is one bucket");
    check(mt::mmq_tc_shape_class(512, 4864, 896) !=
              mt::mmq_tc_shape_class(512, 896, 896),
          "mmq: N stays exact");
    check(mt::mmq_tc_shape_class(512, 4864, 896) !=
              mt::mmq_tc_shape_class(512, 4864, 4864),
          "mmq: K stays exact");

    const auto grid = mt::mmq_tc_candidates();
    check(grid.size() == 12, "mmq: 12 tile candidates (K=32 only)");
    check(grid.front() == (mt::TileConfig{16, 16, 32, 1, 1}),
          "mmq: the smallest config is the reference");
    bool k32 = true, unique = true, in_range = true;
    for (size_t i = 0; i < grid.size(); ++i) {
      k32 = k32 && grid[i].tile_k == 32 && grid[i].unroll == 1 &&
            grid[i].vectorize == 1;
      in_range = in_range && (grid[i].tile_m == 16 || grid[i].tile_m == 32 ||
                              grid[i].tile_m == 64 || grid[i].tile_m == 128) &&
                 (grid[i].tile_n == 16 || grid[i].tile_n == 32 ||
                  grid[i].tile_n == 64);
      for (size_t j = i + 1; j < grid.size(); ++j)
        unique = unique && !(grid[i] == grid[j]);
    }
    check(k32, "mmq: every candidate stages 32-K chunks");
    check(in_range, "mmq: every candidate is on the tile grid");
    check(unique, "mmq: the candidates are distinct");
    check(mt::mmq_tc_default_config() == (mt::TileConfig{64, 64, 32, 1, 1}),
          "mmq: the static default is the MMQ1 tile");

    ::setenv("SONICBOOM_AUTOTUNE", "0", 1);
    check(mt::mmq_tc_choice(512, 4864, 896).cfg == mt::mmq_tc_default_config(),
          "mmq gate off: the static default");
    check(!mt::mmq_tc_choice(512, 4864, 896).resolved(),
          "mmq gate off: nothing is resolved");

    ::setenv("SONICBOOM_AUTOTUNE", "1", 1);
    const mt::TileConfig winner{128, 32, 32, 1, 1};
    check(mt::mmq_tc_choice_store(512, 4864, 896, winner),
          "mmq: the winner is persisted");
    const mt::MmqTcChoice choice = mt::mmq_tc_choice(512, 4864, 896);
    check(choice.resolved() && choice.cfg == winner &&
              choice.source == mt::MmqTcSource::Measured,
          "mmq: the measured winner is memoized");
    const auto mstored = cache.lookup("matmul_q8_tc", mkey);
    check(mstored && *mstored == winner, "mmq: the winner reads back");

    // force() skips the disk cache, so a class the process has not memoized
    // yet resolves to the default and the caller retunes it.
    const std::string other = mt::mmq_tc_shape_class(512, 896, 896);
    check(cache.store("matmul_q8_tc", other, winner),
          "mmq: seed a second class on disk");
    ::setenv("SONICBOOM_AUTOTUNE", "force", 1);
    check(!mt::mmq_tc_choice(512, 896, 896).resolved(),
          "mmq force: the cache is ignored");
    ::unsetenv("SONICBOOM_AUTOTUNE");
  }

  std::error_code ec;
  fs::remove_all(root, ec);

  if (g_failures == 0) {
    std::cout << "test_autotune_cache OK\n";
    return 0;
  }
  std::cerr << "test_autotune_cache FAILED: " << g_failures << " check(s)\n";
  return 1;
}
