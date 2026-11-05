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

// One autotune service for the device kernels that have a per-shape-class
// configuration choice (attention tiles, MMQ J, ...): a disk cache keyed by
// GPU + op + shape class, shared by every tuner, plus the timing loop they
// all use.
//
// Cache location:
//   $SONICBOOM_TUNE_DIR, else $XDG_CACHE_HOME/sonicboom/tune, else
//   $HOME/.cache/sonicboom/tune
//   /<gpu-name>-sm<arch>-<driver>/<op>.json
// Each file is a JSON object of shape-class -> object-of-int. A missing or
// unreadable file is an empty cache; the first tuner run rewrites the whole
// file. Choices are per (op, shape class, GPU). A key never carries a live
// batch size or position: the same shape class must resolve the same way for a
// reused prefix and a full prefill. The one exception is a knob whose every
// setting writes identical bytes (MMQ's J tile): the tuner may add a fixed
// token-count bucket to that key, because the value cannot change the numerics
// while the optimal tile does move with batch scale. A knob that changes the
// summation order never gets such a bucket.
//
// Switches (ablation groups):
//   SONICBOOM_AUTOTUNE=0      use the static per-class defaults, never tune
//   SONICBOOM_AUTOTUNE=force  ignore cached choices and retune every class
//   unset                     use cached choices, tune on a miss
// The tuner only runs outside CUDA-graph capture (host timing is illegal
// there); a capture uses the cached/default choice.
//
// Layer placement: internal runtime machinery (core/src/planner), not part
// of the Layer 2 public headers.

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <functional>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::tune {

// SONICBOOM_AUTOTUNE=0 disables tuning (the static defaults stay in force).
bool enabled();

// SONICBOOM_AUTOTUNE=force: ignore the disk cache, retune every class.
bool force();

// The device directory name (<gpu-name>-sm<arch>-<driver>). Computed once.
// "cpu" when the build has no CUDA support.
const std::string &device_dir();

// Full path of an op's cache file (for diagnostics/reporting).
std::string cache_path(std::string_view op);

// A record with this fingerprint value is not fingerprint-checked by lookup()
// and may be replaced by store(); it marks a numerics-neutral table.
inline constexpr long long kNoFingerprint = -1;

// The build fingerprint: the version string and the identity (size+mtime) of
// the running shared object. It is the only fingerprint a tune record carries;
// per-process switches belong in the key, never here.
const std::string &build_fingerprint();
long long build_fingerprint_key();

// SONICBOOM_TUNE_TRACE=1: one line per tune-table event on stderr:
//   [tune] <event> file=<op>.json key=<key> fp=<fp> pid=<pid> exe=<argv0>
//          caller=<function>
// events: hit, miss, fp-mismatch, measure, write (a new record),
// overwrite (an existing record replaced), overwrite-refused (write-once
// refused).
bool trace_enabled();

// The trace line names the function that touched the table. Both g++ and the
// CUDA host compiler provide __builtin_FUNCTION(), and a default argument is
// evaluated at the call site, so a caller that does not name itself is still
// attributed to the function performing the lookup or the store.
void trace_event(std::string_view event, std::string_view op,
                 std::string_view key, std::string_view fp,
                 const char *caller = __builtin_FUNCTION());

// Cached choice for (op, key); empty when absent or when SONICBOOM_AUTOTUNE=0.
// `force()` makes this always return empty so the caller retunes. With a
// non-negative `expect_fp`, a record whose "k" value differs (or is absent)
// counts as absent: it is traced as fp-mismatch and an empty map is returned.
std::map<std::string, int> lookup(std::string_view op, std::string_view key,
                                  long long expect_fp = kNoFingerprint,
                                  const char *caller = __builtin_FUNCTION());

// Store a choice, updating the in-memory map and rewriting `<op>.json`.
// A no-op when tuning is disabled. Writes are read-modify-write under an
// exclusive flock on `<op>.json.lock`, temp file plus rename, and a record
// whose "k" equals the new record's (`fp`, or the values' own "k") is never
// replaced; the refused write is traced as overwrite-refused. An identical record is
// likewise not rewritten, so a file's digest only moves when a value does.
void store(std::string_view op, std::string_view key,
           const std::map<std::string, int> &values,
           long long fp = kNoFingerprint,
           const char *caller = __builtin_FUNCTION());

// The model-load resolution epoch. Every numerics-affecting choice the loaded
// model can use is resolved while the epoch is open (measurements and writes
// are allowed); seal() closes it. After seal(), a lookup miss on a
// numerics-affecting table is a bug: it is reported and recorded, and the
// runtime build must fail rather than measure. A write to a numerics-affecting
// table after seal() is refused the same way; timing tables (and prefill-path
// tables that resolve on first use) keep their write path. open() starts a new
// epoch for the next model load in the same process.
void open();
void seal();
bool sealed();
bool sealed_missed();
std::string sealed_miss_key();
void clear_sealed_miss();

// Drop the in-memory maps so the next lookup re-reads from disk.
void clear_memory_cache();

// Every choice recorded on disk for this device, sorted by (op, key), read
// without tuning anything. sb-explain prints these; the format marker entry is
// not a choice and is skipped.
struct DiskChoice {
  std::string op;
  std::string key;
  std::map<std::string, int> values;
};
std::vector<DiskChoice> disk_choices();

#ifdef SONICBOOM_USE_CUDA
// Median of `reps` individually timed launches after `warmups` untimed ones.
// The events are recorded on `stream`, which must be the stream the launch
// lambda uses (the caller's current stream). Returns a negative value when no
// valid measurement could be taken (event failure, launch failure); the caller
// keeps its default then.
double median_launch_ms(const std::function<bool()> &launch, cudaStream_t stream,
                        int warmups = 2, int reps = 5);
#else
double median_launch_ms(const std::function<bool()> &launch, int warmups = 2,
                        int reps = 5);
#endif

} // namespace sonicboom::tune
