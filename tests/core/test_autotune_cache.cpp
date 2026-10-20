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
// gated (the runner is), so this works with SONICBOOM_AUTOTUNE unset.

#include "../../core/src/planner/mlir_autotune.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

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

  // The runner gate, and the gemv lookup that honors it.
  {
    const std::string shape = mt::gemv_shape_class(99, 3000);  // bpr99-r4096
    check(cache.store(kOp, shape, b), "store the gate probe");
    ::setenv("SONICBOOM_AUTOTUNE", "1", 1);
    check(mt::enabled(), "SONICBOOM_AUTOTUNE=1 enables");
    check(mt::gemv_config(99, 3000) == b, "gate on: cached winner is used");
    ::setenv("SONICBOOM_AUTOTUNE", "0", 1);
    check(!mt::enabled(), "SONICBOOM_AUTOTUNE=0 disables");
    check(mt::gemv_config(99, 3000) == mt::default_config("gemv"),
          "gate off: the static default, never the cache");
    ::unsetenv("SONICBOOM_AUTOTUNE");
    check(mt::gemv_config(99, 3000) == mt::default_config("gemv"),
          "gate unset: the static default");
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
