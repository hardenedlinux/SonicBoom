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

// Facts the compiler proved about a graph's tensors, carried into the planner.
//
// A plan is derived from a graph's tensors and nodes; these are not part of
// either. They say something true of a tensor's value that the node list alone
// no longer shows once nodes have been fused, and a consumer of the graph —
// the planner, an executor, a scheduler — may read them to decide when work
// can run. They are never needed to run the graph correctly, they take no part
// in the model fingerprint, and a graph with none is a graph nothing was
// proved about.
//
// The one fact carried today is a tensor's stage: what the value depends on,
// and so the earliest moment it can be computed (the stage analysis over
// Boom IR computes it).

#include <sonicboom/planner/ids.h>

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <vector>

namespace sonicboom::planner {

// One input of a generation a value can depend on. The numbering is the bit
// position in `Sources::bits`; it does not change.
enum class Source : uint8_t {
  Prompt = 0,   // what a request brings: prompt, start position, first token
  Counter = 1,  // how many loop iterations have run
  Token = 2,    // a token the loop chose and fed back
  Cache = 3,    // the contents of the KV cache
};

// The earliest moment a value can be computed. A chain, earliest first.
enum class Stage : uint8_t {
  Model = 0,     // at load: depends on the model alone
  Request = 1,   // once per generation, before the loop
  Schedule = 2,  // for every iteration ahead of time
  Token = 3,     // once the iteration's token is chosen, before any cache read
  State = 4,     // in the iteration, in order
};

constexpr std::string_view stage_name(Stage stage) noexcept {
  switch (stage) {
    case Stage::Model: return "model";
    case Stage::Request: return "request";
    case Stage::Schedule: return "schedule";
    case Stage::Token: return "token";
    case Stage::State: return "state";
  }
  return "state";
}

// A set of sources, ordered by inclusion; the empty set depends on the model
// alone.
struct Sources {
  uint8_t bits = 0;

  static constexpr uint8_t kAll = 0x0f;

  static constexpr Sources none() noexcept { return {}; }
  static constexpr Sources all() noexcept { return {kAll}; }
  static constexpr Sources of(Source s) noexcept {
    return {uint8_t(1u << unsigned(s))};
  }
  // The least set containing both.
  static constexpr Sources join(Sources a, Sources b) noexcept {
    return {uint8_t(a.bits | b.bits)};
  }

  constexpr bool has(Source s) const noexcept {
    return ((bits >> unsigned(s)) & 1u) != 0;
  }
  constexpr Sources with(Source s) const noexcept { return join(*this, of(s)); }

  // The latest source decides: Cache, then Token, then Counter, then Prompt.
  constexpr Stage stage() const noexcept {
    if (has(Source::Cache)) return Stage::State;
    if (has(Source::Token)) return Stage::Token;
    if (has(Source::Counter)) return Stage::Schedule;
    if (has(Source::Prompt)) return Stage::Request;
    return Stage::Model;
  }

  constexpr bool operator==(const Sources&) const noexcept = default;
};

struct TensorSources {
  TensorId tensor;
  Sources sources;
};

struct GraphFacts {
  // Ascending tensor id; at most one entry per tensor.
  std::vector<TensorSources> stages;

  bool empty() const noexcept { return stages.empty(); }

  // What tensor `id` depends on; every source when nothing is recorded for
  // it, which is always a true answer.
  Sources sources(TensorId id) const noexcept {
    auto it = std::lower_bound(
        stages.begin(), stages.end(), id,
        [](const TensorSources& e, TensorId key) { return e.tensor < key; });
    return it != stages.end() && it->tensor == id ? it->sources
                                                  : Sources::all();
  }
  Stage stage(TensorId id) const noexcept { return sources(id).stage(); }
};

} // namespace sonicboom::planner
