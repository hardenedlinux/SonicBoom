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

// The decision registry (design/decision-registry.md).
//
// Every automatic choice — a reduction mode, a hand-written tuner's pick, a
// JIT candidate's tile — is decided once per device or per static shape class
// and recorded here with its inputs, the candidates considered, the reason it
// won, its resource costs and the environment variable that can override it.
// The registry is the in-process truth that sb-explain and the C API print;
// the disk file keeps the same records across processes, in the per-device
// decision directory the autotune caches use.
//
// The first record for a (name, scope) pair in a process wins. Recording the
// same value again is a no-op; recording a different value is a conflict: the
// original stays and the conflict is counted, so a nondeterministic choice
// shows up instead of silently changing the plan mid-session. A record loaded
// from disk is historical: this process may resolve the same (name, scope)
// differently (another env, another measurement) and its record replaces it.
//
// fingerprint() hashes the recorded (name, scope, value) triples. The plan
// cache carries it, so a decision that changes invalidates cached plans keyed
// under the old set.
//
// This is internal runtime machinery (core/src/planner), not part of the
// Layer 2 public headers.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::planner::decisions {

enum class Source {
  Default,   // the static default: no measurement, no cached pick
  Env,       // an environment variable overrode the choice
  Cached,    // a recorded per-device / per-class pick
  Measured,  // timed in this process
  Verdict,   // an acceptance verdict recorded by a measurement run
  Skipped,   // the cold-tuning budget expired; the static default stands
};

std::string_view source_name(Source source);

struct Entry {
  std::string name;   // "reduce.mode", "tune/gemv", ...
  std::string scope;  // "device", "cpu", a shape class, ...
  std::string value;  // the chosen value, in its serialized form
  Source source = Source::Default;
  std::string env;    // override variable, empty when there is none

  // Why the choice was made, for sb-explain.
  std::string inputs;      // device facts / shape class the decision used
  std::string candidates;  // alternatives considered, "|"-separated
  std::string reason;      // the rule that picked the value
  std::string numerics;    // numerics class (bit-identical, fp64 bound, ...)

  // Declared resource costs, so a solver can choose jointly later.
  int64_t vram_bytes = 0;
  int64_t tuning_ms = 0;

  // False when the entry was loaded from the persisted registry: this process
  // may replace it.
  bool current = true;
};

// Record the decision for (name, scope) with the short form (no inputs,
// candidates or costs). True when it was stored or already present with the
// same value; false on a conflicting value, which is counted and dropped.
bool record(std::string_view name, std::string_view scope,
            std::string_view value, Source source,
            std::string_view env = {});

// Record the full decision. `current` is ignored (records are always current).
// Same return convention as the short form; a historical entry is replaced.
bool record(const Entry& entry);

// Every decision (current over historical), sorted by (name, scope).
std::vector<Entry> snapshot();

// The number of conflicting current-process records dropped.
std::size_t conflict_count();

// FNV-1a of the sorted (name, scope, value) triples; 0 when no decision has
// been recorded.
uint64_t fingerprint();

// The human table and the JSON form sb-explain / sb_explain_table and
// sb_explain_json print:
//   name scope = value [source] env=VAR reason="..."
// and one JSON object per decision.
std::string dump_text();
std::string dump_json();

// Record the algorithm-selecting environment knobs (the T0b table) for this
// device, once per process: the env value when set, the static default
// otherwise. sb-explain and planner_env() call this; the resolved per-class
// choices the knobs select are recorded where they are made.
void seed_static_knobs();

// The persisted registry for this device:
// <decision root>/<device>/decisions.jsonl.
std::string path();

// Tests only: drop the in-memory registry and this device's persisted file.
void reset_for_test();

// Tests only: drop the in-memory registry but keep the persisted file, so the
// next call reloads it exactly as a new process would.
void drop_memory_for_test();

} // namespace sonicboom::planner::decisions
