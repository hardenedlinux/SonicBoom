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
// Gate: the runner is enabled by default, like the older sonicboom::tune
// service (autotune.h). SONICBOOM_AUTOTUNE=0 disables it (nothing is compiled,
// timed, or stored; callers keep their static default), and
// SONICBOOM_AUTOTUNE=force retimes every class, ignoring the disk cache.
// Disabling the runtime `autotune` feature does the same as =0.
// SONICBOOM_AUTOTUNE_JIT=0 disables only this JIT-candidate service and leaves
// the hand-written tuner on; it is the ablation for "generated kernel vs the
// hand-written kernel the other tuner chose", and the mode the plan-record
// goldens use so their text stays the hand-written plan.
//
// Cold budget: cold tuning (every class not in the disk cache) shares one
// process-wide wall-clock budget, SONICBOOM_AUTOTUNE_BUDGET_MS (default 10000;
// <= 0 means unlimited). When it runs out, the remaining classes keep their
// static default for the life of the process and are not probed again, while
// the classes tuned before it expired are persisted as usual. The clock starts
// at the first unresolved class, so a process whose classes are all cached
// spends nothing. The budget bounds a first run on a new device to ~10 s of
// measurement instead of minutes; every choice is still decided once per
// device and cached.
//
// Determinism: a cache hit is deterministic — the same record is applied on
// every launch and in every process on the device. A cold measurement is
// timing dependent, so two cold processes can disagree within the tuner's
// resolution; the runner only ever picks a candidate whose output is
// bit-identical to the static default, so a different pick changes the kernel's
// launch geometry, never the token stream or the numerics.
//
// Cache location: $XDG_CACHE_HOME/sonicboom/autotune/sm_<NN>.bin, else
// $HOME/.cache/sonicboom/autotune/sm_<NN>.bin, where <NN> is the device
// compute capability × 10 (sm_86 for an RTX 3050; sm_00 without CUDA). The
// file is a versioned fixed-record array written in place under an exclusive
// flock, so concurrent processes never see a torn update and a version bump
// invalidates every record (the first store rewrites the file).
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

// True unless SONICBOOM_AUTOTUNE=0, SONICBOOM_AUTOTUNE_JIT=0, or the
// `autotune` runtime feature is disabled. Read per call, so a test can flip it
// between sections.
bool enabled();
// SONICBOOM_AUTOTUNE=force: every per-class lookup skips the disk cache, so
// each class is retimed in this process.
bool force();

// Wall-clock budget left for cold tuning in this process, in milliseconds
// (0 when exhausted). Starts the clock on its first call; SONICBOOM_AUTOTUNE_
// BUDGET_MS overrides the 10 s default (<= 0 = unlimited).
int64_t budget_remaining_ms();
bool budget_exhausted();

// Restart the cold-tuning clock (tests only; the disk cache is untouched).
void reset_budget_for_test();

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

// Every record in this SM's cache file, sorted by (op, shape class). Read
// only: sb-explain prints them, and calling this never tunes or creates a
// file. An absent file, an older version, or a corrupt header yields nothing.
struct DiskRecord {
  std::string op;
  std::string shape_class;
  TileConfig cfg;
};
std::vector<DiskRecord> disk_records();

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
// same shape class is a map lookup. With the gate off this is
// default_config("gemv") and nothing is resolved.
enum class GemvSource {
  Default,   // gate off, or the class is not resolved yet
  Cache,     // loaded from the per-SM cache file
  Measured,  // the runner timed it in this process
  Skipped,   // the cold-tuning budget is exhausted: static default, no probe
};

struct GemvChoice {
  TileConfig cfg = TileConfig{256, 1, 32, 1, 1};
  GemvSource source = GemvSource::Default;
  bool resolved() const { return source != GemvSource::Default; }
};

// Resolve (memoized). An unresolved result means the caller may tune the class
// with AutotuneRunner and then call gemv_choice_store with the winner.
GemvChoice gemv_choice(int blocks_per_row, int num_rows);

// Memoize a measured winner for the process and persist it for the SM. The
// process memo is updated even when the file write fails, so a class is never
// retimed on every launch. False means the file write failed.
bool gemv_choice_store(int blocks_per_row, int num_rows,
                       const TileConfig& cfg);

// The resolved config alone (callers that never tune).
TileConfig gemv_config(int blocks_per_row, int num_rows);

// A shape class is a coarse, bucketized description of the problem: settings
// are per class, never per batch size or KV position, so a reused prefix and a
// full prefill resolve the same way. Rows bucket to the next power of two in
// [128, 262144]; blocks-per-row is kept exact.
std::string gemv_shape_class(int blocks_per_row, int num_rows);

// MMQ2 — the generated int8 tensor-core prefill matmul (op "matmul_q8_tc").
//
// TileConfig mapping: tile_m is the weight rows per block (MmqTcShape::bm),
// tile_n the tokens per block (MmqTcShape::bn), tile_k the K values per staged
// chunk. unroll and vectorize are fixed at 1.
//
// Shape class: the token count (the activation's batched dimension, "M" in the
// BLAS sense) is bucketed; the weight rows (N, the output features) and K are
// kept exact. A bucket spans every token count that resolves to the same tile
// choice, so a class is tuned once per GPU and reused across launches.
std::string mmq_tc_shape_class(int m_tokens, int n_rows, int k);

// The pre-MMQ2 launch shape: 64 weight rows x 64 tokens, 32-K chunks
// (whatever the shape class, so SONICBOOM_AUTOTUNE=0 is the MMQ1 path).
TileConfig mmq_tc_default_config();

// The candidate grid: tile_m {16,32,64,128} x tile_n {16,32,64} x tile_k {32}.
// Pruning: tile_k 16 is not a candidate because the kernel's unit is the
// m16n8k32 mma (a 16-K chunk cannot feed it); the runner additionally skips a
// config the generator does not support. candidates[0] is the smallest config
// (16 x 16 x 32) and is the bit-identity reference.
std::span<const TileConfig> mmq_tc_candidates();

enum class MmqTcSource {
  Default,   // gate off, or the class is not resolved yet
  Cache,     // loaded from the per-SM cache file
  Measured,  // the runner timed it in this process
  Skipped,   // the cold-tuning budget is exhausted: static default, no probe
};

struct MmqTcChoice {
  TileConfig cfg = TileConfig{64, 64, 32, 1, 1};
  MmqTcSource source = MmqTcSource::Default;
  bool resolved() const { return source != MmqTcSource::Default; }
};

// Resolve (memoized per process). An unresolved result means the caller may
// tune the class with AutotuneRunner (candidates from mmq_tc_candidates() and
// the smallest config as reference) and then call mmq_tc_choice_store.
MmqTcChoice mmq_tc_choice(int m_tokens, int n_rows, int k);

// Memoize a measured winner and persist it for the SM. The process memo is
// updated even when the file write fails, so a class is never retimed on every
// launch. False means the file write failed.
bool mmq_tc_choice_store(int m_tokens, int n_rows, int k,
                         const TileConfig& cfg);

// Q3 — prefill attention shape classes (the generated FA-2 kernel).
//
// TileConfig mapping: tile_m is the query rows per block (the generator's
// `br`), tile_n the head_dim columns a warp owns (`dc` = head_dim / dps),
// tile_k the key columns per tile (`bc`), unroll the cp.async pipeline depth
// (`as`: 0 = synchronous staging, 2 = double buffered); vectorize is unused.
// dps stays at the production shape's value in every candidate: another
// split changes the S reduction order and fails the runner's bit-identity
// gate. The grid is (br x bc x as) around the production config, which is
// candidates[0] and the reference both the bit check and the timing target.
// Classes cover the generator's defaulted head_dims {32,64,128,256,512}; the
// production dps for 256/512 is 4/8 (dc = 64 in every shipped class).
//
// Shape class: head_dim exact; the query length (the prefill chunk size) and
// the query-head count are bucketed, so a class is tuned once per GPU and
// reused across launches in the same bucket.
std::string attn_prefill_seq_bucket(int seq_len);
std::string attn_prefill_heads_bucket(int n_heads);

// "<head_dim>/<seq_len_bucket>/<num_heads_bucket>" (the cache's shape field;
// the op field is "attn_prefill", so the combined name is
// "attn_prefill/<head_dim>/<seq_len_bucket>/<num_heads_bucket>").
std::string attn_prefill_shape_class(int head_dim, int seq_len, int n_heads);

// Head dims the generated prefill kernel has defaulted shape classes for.
bool attn_prefill_tunable(int head_dim);

// Grid around `production` (which must be candidates[0]): br in
// {production.br, 32, 64} (16 also for head_dim >= 512), bc in
// {production.bc, 32}, as in {production.async_stages, 0, 2}. The caller
// rejects candidates the generator does not support (shared-memory budget)
// in its compile step.
std::vector<TileConfig> attn_prefill_candidates(int head_dim,
                                                const TileConfig &production);

enum class AttnPrefillSource {
  Default,   // gate off, or the class is not resolved yet
  Cache,     // loaded from the per-SM cache file
  Measured,  // the runner timed it in this process
  Skipped,   // the cold-tuning budget is exhausted: static default, no probe
};

struct AttnPrefillChoice {
  TileConfig cfg{};
  AttnPrefillSource source = AttnPrefillSource::Default;
  bool resolved() const { return source != AttnPrefillSource::Default; }
};

// Resolve (memoized per process). An unresolved result means the caller may
// tune the class with AutotuneRunner (candidates from
// attn_prefill_candidates(), the first as reference) and then call
// attn_prefill_choice_store.
AttnPrefillChoice attn_prefill_choice(int head_dim, int seq_len, int n_heads);

// Memoize a measured winner and persist it for the SM. True when the file
// write succeeded.
bool attn_prefill_choice_store(int head_dim, int seq_len, int n_heads,
                               const TileConfig& cfg);

// Drop the in-process memo (tests; the disk cache is untouched).
void attn_prefill_reset_for_test();

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
    // The wait used while timing, when set: a lighter sync than `sync` (e.g. a
    // stream sync without the device-to-host copy the bit-identity check
    // needs), so the measurement is the launch rather than the launch plus an
    // identical-for-every-candidate copy. Empty uses `sync`. The check runs and
    // the final winner run always use `sync`.
    std::function<void()> timing_sync;
    // Timed iterations per candidate (the brief's N=5).
    int iters = 5;
  };

  // Fastest bit-identical candidate, or nullopt when the gate is off, the
  // cold budget is exhausted before any round completes, no candidate
  // compiles, or every candidate diverges from the reference. An expired
  // budget stops the search and ranks the rounds already measured.
  static std::optional<TileConfig> run(const Problem& problem);
};

} // namespace sonicboom::mlir_tune
