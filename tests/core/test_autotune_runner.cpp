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

// Q1/Q2 runner: with synthetic compile closures, the runner must pick the
// fastest config that reproduces the reference output bit for bit, skip a
// config that does not compile, and reject a faster config whose output
// differs. It must do nothing under SONICBOOM_AUTOTUNE=0.
//
// Q2 adds the real kernel: the first q8_1 gemv call for a shape class under the
// gate times tile_m {256, 128, 512} on the live buffers, accepts a width only
// when its output matches the 256-thread launch bit for bit, and persists the
// winner per SM. The three widths are the (32, 4/8/16) block shapes; each row is
// computed by its own warp either way, so the tuned launch must reproduce the
// default path byte for byte.

#include "../../core/src/planner/mlir_autotune.h"

#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace mt = sonicboom::mlir_tune;
namespace sbquant = sonicboom::quant;

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

uint16_t rand_finite_half(std::mt19937& rng) {
  uint16_t h;
  do { h = uint16_t(rng()); } while (((h >> 10) & 0x1F) == 0x1F);
  return h;
}

// The synthetic q3_K weight of tools/baseline/gemv_bench.cpp: random block
// bytes, a finite half scale per block, so the dequant arithmetic is the same
// work as real GGUF bytes and the result is deterministic for a fixed seed.
std::vector<std::byte> build_q3k(uint64_t rows, uint64_t cols,
                                 std::mt19937& rng) {
  const uint64_t bpr = cols / 256;
  std::vector<std::byte> bytes(rows * bpr * 110);
  for (auto& b : bytes) b = std::byte(uint8_t(rng()));
  for (uint64_t i = 0; i < rows * bpr; ++i) {
    const uint16_t d = rand_finite_half(rng);
    bytes[i * 110 + 108] = std::byte(d & 0xFF);
    bytes[i * 110 + 109] = std::byte(d >> 8);
  }
  return bytes;
}

// --- Q2: the runner on the real q8_1 gemv -----------------------------------

void real_gemv_section() {
  if (!sbquant::cuda::available()) {
    std::cout << "  (skipped: CUDA unavailable)\n";
    return;
  }
  char tmpl[] = "/tmp/sb-autotune-gemv-XXXXXX";
  if (::mkdtemp(tmpl) == nullptr) {
    std::cerr << "  FAIL: mkdtemp\n";
    ++g_failures;
    return;
  }
  ::setenv("XDG_CACHE_HOME", tmpl, 1);
  // This test measures several sections in one process; the 10 s cold budget
  // is a production bound, not a test one.
  ::setenv("SONICBOOM_AUTOTUNE_BUDGET_MS", "0", 1);
  mt::AutotuneCache::instance().reset_for_test();

  const uint64_t cols = 2560, rows = 2048;  // q3_K, blocks-per-row 10
  const int bpr = int(cols / 256);
  const std::string cls = mt::gemv_shape_class(bpr, int(rows));
  std::mt19937 rng(20260820);
  const std::vector<std::byte> bytes = build_q3k(rows, cols, rng);
  sbquant::QuantizedTensor W(sbquant::QuantType::Q3_K, {cols, rows}, bytes);
  check(W.valid(), "the synthetic q3_K weight is well formed");
  if (!W.valid()) return;

  std::vector<float> x(cols), y_off(rows), y_on(rows), y_again(rows);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& v : x) v = ud(rng);

  // Gate off: the pre-Q1 launch, and nothing is resolved or stored.
  ::setenv("SONICBOOM_AUTOTUNE", "0", 1);
  check(sbquant::cuda::matvec_f32(W, x, y_off), "gate off: the gemv runs");
  check(mt::gemv_choice(bpr, int(rows)).source == mt::GemvSource::Default,
        "gate off: the class is not resolved");
  check(!mt::AutotuneCache::instance().lookup("gemv", cls),
        "gate off: nothing is stored");

  // Gate on: the first call measures the class through the runner.
  ::setenv("SONICBOOM_AUTOTUNE", "1", 1);
  check(sbquant::cuda::matvec_f32(W, x, y_on), "gate on: the gemv runs");
  const mt::GemvChoice choice = mt::gemv_choice(bpr, int(rows));
  check(choice.source == mt::GemvSource::Measured,
        "the first tuned call measures the class");
  check(choice.cfg.tile_m == 128 || choice.cfg.tile_m == 256 ||
            choice.cfg.tile_m == 512,
        "the winner is one of the candidate widths");
  const auto stored = mt::AutotuneCache::instance().lookup("gemv", cls);
  check(stored && *stored == choice.cfg, "the winner is persisted");
  check(std::memcmp(y_off.data(), y_on.data(), rows * sizeof(float)) == 0,
        "the tuned width is bit-identical to the default");

  check(sbquant::cuda::matvec_f32(W, x, y_again), "the second call runs");
  check(std::memcmp(y_off.data(), y_again.data(), rows * sizeof(float)) == 0,
        "the cached width is bit-identical too");
  check(mt::gemv_choice(bpr, int(rows)).cfg == choice.cfg,
        "the class stays resolved");
  ::unsetenv("SONICBOOM_AUTOTUNE");
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

  // Gate: 0 does nothing (no compile, no timing); unset is on by default.
  {
    int compiled = 0;
    mt::AutotuneRunner::Problem gated = p;
    gated.compile = [&](const mt::TileConfig& c) {
      ++compiled;
      return p.compile(c);
    };
    ::unsetenv("SONICBOOM_AUTOTUNE");
    check(mt::enabled(), "gate unset: enabled by default");
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

  real_gemv_section();

  if (g_failures == 0) {
    std::cout << "test_autotune_runner OK\n";
    return 0;
  }
  std::cerr << "test_autotune_runner FAILED: " << g_failures << " check(s)\n";
  return 1;
}
