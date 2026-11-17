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

// The final-segment split policy (round 9, part B; round 10 keeps both parts in
// the batched class).
//
// A prompt's last row is the only one whose hidden state feeds lm_head, so a
// final segment can be split into interior rows — planned for their cache
// writes alone — and a final segment of the last `batched_min` rows. Both parts
// stay at or above the batch threshold, so every row is computed by the same
// kernels as in the unsplit segment. Only the trailing blocks that own neither
// a cache nor a state are dead in every interior segment; the final part runs
// them once (about one decode step's model read). The split pays when the rows
// it makes interior repay kSplitRowsPerModelRead rows of that read.
//
// The policy is a pure function of the recipe and the row count so the suite
// can drive it with synthetic stacks. The environment override lives in
// gemma4_plan.cpp (SONICBOOM_PREFILL_SPLIT_LAST).

#ifndef SONICBOOM_MODEL_PREFILL_SPLIT_H
#define SONICBOOM_MODEL_PREFILL_SPLIT_H

#include <sonicboom/model/recipe.h>

#include <cstdint>
#include <variant>

namespace sonicboom::model::detail {

// The last few rows as their own segment read the whole model once, about as
// expensive as batching 64 rows.
constexpr uint64_t kSplitRowsPerModelRead = 64;

// `auto` applies the static rule below, `on` splits a final segment of at least
// two batched segments regardless of the block count (the test's forcing arm)
// and `off` keeps the pre-round plan.
enum class SplitLastMode : uint8_t { Auto, On, Off };

// On by default: both parts of the split stay in the batched class, so every
// row is computed by the same kernels as in the unsplit segment. The Gemma
// gate (round 10, part B3) compares eight prompt lengths, 200 to 6000, and the
// first-token logits and 64 greedy tokens are bitwise identical either way.
constexpr SplitLastMode kPrefillSplitDefault = SplitLastMode::Auto;

// Trailing blocks whose mixer owns neither a cache nor a state, counted from
// the end of the stack. A block that writes neither contributes nothing a later
// position reads, so demand pruning removes it from an interior segment. A
// layer that shares another layer's cache does not own one; a GatedDeltaNet
// advances the state it owns.
inline uint32_t trailing_dead_blocks(const Recipe& r) {
  uint32_t dead = 0;
  for (auto it = r.blocks.rbegin(); it != r.blocks.rend(); ++it) {
    bool owns = false;
    if (const auto* at = std::get_if<SelfAttention>(&it->mixer))
      owns = at->owns_cache;
    else
      owns = std::holds_alternative<GatedDeltaNet>(it->mixer);
    if (owns) break;
    ++dead;
  }
  return dead;
}

// Rows held back from a final segment of `nseg` rows to run as their own
// final segment; 0 = no split. `batched_min` is the smallest segment the
// batched kernels take: both parts stay at or above it, so every row is
// computed by the same kernels as in the unsplit segment.
inline uint64_t split_tail_rows(const Recipe& r, uint64_t nseg,
                                uint64_t batched_min,
                                SplitLastMode mode) {
  if (mode == SplitLastMode::Off) return 0;
  if (batched_min == 0 || nseg < 2 * batched_min) return 0;
  if (mode == SplitLastMode::On) return batched_min;
  const uint64_t dead = trailing_dead_blocks(r);
  return dead > 0 && nseg - batched_min >=
                         kSplitRowsPerModelRead * r.blocks.size() / dead
             ? batched_min
             : 0;
}

}  // namespace sonicboom::model::detail

#endif  // SONICBOOM_MODEL_PREFILL_SPLIT_H
