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
// Every automatic choice — the reduction mode, a hand-written tuner's pick, a
// JIT candidate's tile — is decided once per device or per static shape class
// and recorded here with the source that decided it and the environment
// variable that can override it. The registry is the in-process truth that
// sb-explain prints; the disk caches hold the same decisions across processes.
//
// The first record for a (name, scope) pair wins. Recording the same value
// again is a no-op; recording a different value is a conflict: the original
// stays and the conflict is counted, so a nondeterministic choice shows up in
// sb-explain instead of silently changing the plan mid-session.
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
};

// Record the decision for (name, scope). True when it was stored or already
// present with the same value; false on a conflicting value, which is counted
// and dropped.
bool record(std::string_view name, std::string_view scope,
            std::string_view value, Source source,
            std::string_view env = {});

// Every recorded decision, sorted by (name, scope).
std::vector<Entry> snapshot();

// The number of conflicting records dropped since the process started.
std::size_t conflict_count();

// FNV-1a of the sorted (name, scope, value) triples; 0 when no decision has
// been recorded. Stable across processes for the same decision set.
uint64_t fingerprint();

// The textual dump sb-explain prints for the in-process decisions:
//   <name> <scope> = <value> [source] [env=<VAR>]
std::string dump_text();

// Tests only.
void reset_for_test();

} // namespace sonicboom::planner::decisions
