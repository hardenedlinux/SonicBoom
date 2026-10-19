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

#include <sonicboom/planner/drafter.h>

#include <algorithm>

namespace sonicboom::planner {

namespace {

// A 64-bit key for an n-token n-gram. Two different n-grams may collide; a
// collision only yields a wrong proposal, which the verifier rejects, so the
// drafter stays correct and deterministic — the mix just needs to make a
// collision as unlikely as a 64-bit hash can.
uint64_t ngram_key(const uint64_t* toks, size_t n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ uint64_t(n);
  for (size_t i = 0; i < n; ++i) {
    h ^= toks[i] + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBull;
    h ^= h >> 31;
  }
  return h;
}

} // namespace

void NGramDrafter::reset() noexcept {
  hist_.clear();
  for (auto& m : ngram_) m.clear();
}

void NGramDrafter::index_ending_at(size_t end) {
  for (size_t n = kMinN; n <= kMaxN; ++n) {
    if (end + 1 < n) continue;  // the occurrence would start before position 0
    // Most recent occurrence wins: a later push overwrites the end position.
    ngram_[n][ngram_key(&hist_[end + 1 - n], n)] = end;
  }
}

void NGramDrafter::push(uint64_t token) {
  hist_.push_back(token);
  // The n-gram ending at the new token's predecessor now has a known following
  // token (the one just pushed) — exactly what a hit needs to propose. Indexing
  // one position behind the end also keeps the history's own last n-gram out of
  // the tables, so a lookup always finds an *earlier* occurrence rather than
  // the one that ends where the lookup starts.
  if (hist_.size() >= 2) index_ending_at(hist_.size() - 2);
}

void NGramDrafter::push(std::span<const uint64_t> tokens) {
  for (uint64_t t : tokens) push(t);
}

size_t NGramDrafter::propose(std::span<uint64_t> out) const {
  const size_t k = std::min(out.size(), max_drafts_);
  if (k == 0 || hist_.size() < kMinN + 1) return 0;
  // Longest context first: a 4-gram hit is a more specific prediction than a
  // 3- or 2-gram hit, and the order it is tried in is fixed (deterministic).
  for (size_t n = kMaxN; n >= kMinN; --n) {
    if (hist_.size() < n) continue;
    const uint64_t* suffix = &hist_[hist_.size() - n];
    const auto it = ngram_[n].find(ngram_key(suffix, n));
    if (it == ngram_[n].end()) continue;
    const size_t end = it->second;
    // The proposal is what followed that occurrence: history[end+1 .. L-1].
    const size_t count = std::min(k, hist_.size() - 1 - end);
    if (count == 0) continue;
    for (size_t i = 0; i < count; ++i) out[i] = hist_[end + 1 + i];
    return count;
  }
  return 0;
}

} // namespace sonicboom::planner
