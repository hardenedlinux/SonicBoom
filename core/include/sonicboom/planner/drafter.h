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

// Prompt-lookup / n-gram drafter (B5). A CPU-only, model-agnostic drafter over
// the accepted token history: index every 4-, 3- and 2-gram by the position at
// which it most recently ended, and propose the tokens that followed that
// occurrence when the history's suffix matches again. It is the cheap drafter
// speculative decoding wants on structured text (a pasted file, a code edit),
// where the continuation is usually a copy of an earlier run of tokens.
//
// The tables must describe accepted tokens only. The caller pushes a token when
// it becomes part of the session history (an accepted draft, a verifier token,
// a prefill token); a rejected draft is never pushed. Proposals are the
// hash-table hits' following tokens, so the drafter is deterministic for a
// given history and has no dependency on any model or device.
//
// Cost: one hash insert per (n = 4, 3, 2) per pushed token, and one lookup per
// (n = 4, 3, 2) per proposal — O(1) in the history length, which is what makes
// the drafter < 2% of a decode step.

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace sonicboom::planner {

class NGramDrafter {
 public:
  // The longest / shortest n-gram indexed. Lookups try 4, then 3, then 2.
  static constexpr size_t kMaxN = 4;
  static constexpr size_t kMinN = 2;

  // `max_drafts` is the largest proposal this drafter returns (a speculative
  // step's draft length cap; the caller may lower it between steps).
  explicit NGramDrafter(size_t max_drafts = 7) : max_drafts_(max_drafts) {}

  void reset() noexcept;
  void push(uint64_t token);
  void push(std::span<const uint64_t> tokens);

  size_t max_drafts() const noexcept { return max_drafts_; }
  void set_max_drafts(size_t k) noexcept { max_drafts_ = k; }

  // Propose up to min(out.size(), max_drafts()) tokens following the current
  // history, from the most recent earlier occurrence of the longest matching
  // suffix n-gram. Writes them to the front of `out` and returns the count; 0
  // means no n-gram in the tables matches (the caller should take a plain
  // decode step). A match yields at most the tokens that followed it in the
  // history, so the proposal never invents a token.
  size_t propose(std::span<uint64_t> out) const;

  std::span<const uint64_t> history() const noexcept { return hist_; }
  size_t size() const noexcept { return hist_.size(); }

 private:
  // Index the n-grams ending at history position `end` (their following token
  // already exists, so a hit can propose at least one token).
  void index_ending_at(size_t end);

  std::vector<uint64_t> hist_;
  std::unordered_map<uint64_t, size_t> ngram_[kMaxN + 1];  // [n] = key -> end
  size_t max_drafts_ = 7;
};

} // namespace sonicboom::planner
