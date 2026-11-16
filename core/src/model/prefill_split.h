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

// The final-segment split policy (round 9, part B).
//
// A prompt's last row is the only one whose hidden state feeds lm_head, so a
// final segment can be split into `rows - 1` interior rows — planned for their
// cache writes alone — and a one-row final segment, an existing shape class.
// Only the trailing blocks that own neither a cache nor a state are dead in
// every interior segment; the one-row segment runs them once (about one decode
// step's model read). The split pays when the rows it makes interior repay
// kSplitRowsPerModelRead rows of that read.
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

constexpr uint64_t kSplitRowsPerModelRead = 64;

// `auto` applies the static rule below, `on` splits a final segment of two or
// more rows regardless of the block count (the test's forcing arm) and `off`
// keeps the pre-round plan.
enum class SplitLastMode : uint8_t { Auto, On, Off };

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

// Whether a final segment of `rows` rows splits into `rows - 1` interior rows
// and a one-row final segment. The rule is static (no timing): `auto` splits
// only where whole blocks are dead and the rows made interior repay the
// one-row segment's model read.
inline bool split_last_row(const Recipe& r, uint64_t rows,
                           SplitLastMode mode) {
  if (mode == SplitLastMode::Off || rows <= 1) return false;
  if (mode == SplitLastMode::On) return true;
  const uint64_t dead = trailing_dead_blocks(r);
  return dead > 0 &&
         rows - 1 >= kSplitRowsPerModelRead * r.blocks.size() / dead;
}

}  // namespace sonicboom::model::detail

#endif  // SONICBOOM_MODEL_PREFILL_SPLIT_H
