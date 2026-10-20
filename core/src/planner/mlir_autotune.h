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

// Q1 — tuning skeleton for MLIR-generated kernels.
//
// A TileConfig is the tuning unit of a generated kernel (tile shape, unroll,
// vector width). AutotuneCache is the persistent, cross-process store of the
// winning config per (op_type, shape_class, sm_version): a fixed-record binary
// file, mmap'd, protected by flock. AutotuneRunner is the in-process search:
// compile each candidate, time N iterations, keep a candidate only if its
// output matches the reference bit for bit, and return the fastest.
//
// Gate: the runner is enabled only by SONICBOOM_AUTOTUNE=1 (off by default).
// With the gate off nothing is compiled, timed, or stored, and callers keep
// their static default. This is deliberately the opposite default of the
// older sonicboom::tune service (autotune.h, which tunes already-compiled
// hand-written kernels at runtime and is on unless SONICBOOM_AUTOTUNE=0):
// compiling and launching JIT candidates is opt-in.
//
// Cache location: $XDG_CACHE_HOME/sonicboom/autotune/sm_<NN>.bin, else
// $HOME/.cache/sonicboom/autotune/sm_<NN>.bin, where <NN> is the device
// compute capability × 10 (sm_86 for an RTX 3050; sm_00 without CUDA).
//
// Layer placement: internal runtime machinery (core/src/planner), not part of
// the Layer 2 public headers; no MLIR/LLVM/CUDA types in this interface.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::mlir_tune {

struct TileConfig {
  int32_t tile_m = 0;
  int32_t tile_n = 0;
  int32_t tile_k = 0;
  int32_t unroll = 1;
  int32_t vectorize = 1;

  bool operator==(const TileConfig&) const = default;

  // Serializable form: "tile_m,tile_n,tile_k,unroll,vectorize". Parsing is
  // strict: five integers, no trailing text, every value > 0.
  std::string to_string() const;
  static std::optional<TileConfig> from_string(std::string_view s);
};

// The static per-shape-class default. Callers use it on a cache miss and
// whenever the gate is off, so AUTOTUNE=0 is the pre-Q1 code path.
TileConfig default_config(std::string_view op);

// SONICBOOM_AUTOTUNE=1 (exact string) enables the runner. Read per call.
bool enabled();

// Device compute capability × 10 (86, 89, 120); 0 when the build has no CUDA
// or no device is visible. Cached after the first call.
int sm_version();

// Candidate grid for the decode gemv: tile_m {128, 256, 512} × tile_k {32, 64},
// tile_n = 1 (one row of output per tile), unroll = 1, vectorize = 1.
std::span<const TileConfig> gemv_candidates();

class AutotuneCache {
 public:
  static AutotuneCache& instance();

  // The record for (op, shape_class, sm_version()), or nullopt on a miss.
  std::optional<TileConfig> lookup(std::string_view op,
                                   std::string_view shape_class);
  // Insert or update the record. False when the file cannot be created or
  // locked; the caller keeps its default then.
  bool store(std::string_view op, std::string_view shape_class,
             const TileConfig& cfg);

  // File this process uses (diagnostics and tests).
  std::string path() const;
  void reset_for_test();

 private:
  AutotuneCache() = default;
  mutable std::string path_;  // resolved on the first path() call
};

// V2 spec auto-detect: a scalar ratio record in the same per-SM file. The spec
// probe measures one plain decode step against one k=1 verify step and stores
// verify/plain here, so the decision is made once per GPU rather than once per
// process. The record is encoded as a TileConfig ratio in thousandths
// ({tile_m = round(ratio*1000), 1, 1, 1, 1}); the op name keeps it apart from
// every real tile choice. Not gated by enabled(): this is a probe result, not
// a JIT candidate. lookup_ratio returns nullopt on a miss or a stored value
// outside [0.001, 1048.575].
bool store_ratio(std::string_view op, std::string_view shape, double ratio);
std::optional<double> lookup_ratio(std::string_view op,
                                   std::string_view shape);

// The per-session gemv choice. The first call for a shape class resolves it
// once and memoizes it for the life of the process; every later call with the
// same shape class is a map lookup. With the gate off, or on a cache miss,
// this is default_config("gemv").
TileConfig gemv_config(int blocks_per_row, int num_rows);

// A shape class is a coarse, bucketized description of the problem: settings
// are per class, never per batch size or KV position, so a reused prefix and a
// full prefill resolve the same way.
std::string gemv_shape_class(int blocks_per_row, int num_rows);

class AutotuneRunner {
 public:
  using Launch = std::function<void()>;

  struct Problem {
    // candidates[0] is the reference (the default config). All other
    // candidates are accepted only if their output matches it bit for bit.
    std::vector<TileConfig> candidates;
    // Compile a candidate to a launch closure; nullopt when the config does
    // not compile or is unsupported, which skips it.
    std::function<std::optional<Launch>(const TileConfig&)> compile;
    // Output window a launch writes; compared across candidates.
    void* output = nullptr;
    size_t output_bytes = 0;
    // Called after each launch when set (a device stream sync for async
    // launches; synchronous launches leave it empty).
    std::function<void()> sync;
    // Timed iterations per candidate (the brief's N=5).
    int iters = 5;
  };

  // Fastest bit-identical candidate, or nullopt when the gate is off, no
  // candidate compiles, or every candidate diverges from the reference.
  static std::optional<TileConfig> run(const Problem& problem);
};

} // namespace sonicboom::mlir_tune
