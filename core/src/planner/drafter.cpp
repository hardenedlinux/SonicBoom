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

// The long-match drafter's n-gram classes and the polynomial rolling hash.
// The base is odd so the modulo-2^64 product stays invertible (an even base
// would drop low bits as the window slides); kLongPow[i] is B^(n-1) for class
// i, the shift the departing token sits at when the window slides.
constexpr size_t kLongNs[8] = {32, 24, 16, 12, 8, 6, 4, 3};
constexpr uint64_t kRollBase = 0x00000100000001B3ull;  // FNV prime

uint64_t roll_pow(size_t e) {
  uint64_t p = 1;
  for (size_t i = 0; i < e; ++i) p *= kRollBase;
  return p;
}

const uint64_t kLongPow[8] = {
    roll_pow(kLongNs[0] - 1), roll_pow(kLongNs[1] - 1),
    roll_pow(kLongNs[2] - 1), roll_pow(kLongNs[3] - 1),
    roll_pow(kLongNs[4] - 1), roll_pow(kLongNs[5] - 1),
    roll_pow(kLongNs[6] - 1), roll_pow(kLongNs[7] - 1)};

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
  return propose(out, nullptr);
}

size_t NGramDrafter::propose(std::span<uint64_t> out, size_t* matched_n) const {
  if (matched_n) *matched_n = 0;
  const size_t k = std::min(out.size(), max_drafts_);
  if (k == 0 || hist_.size() < kMinN + 1) return 0;
  // Longest context first: a 4-gram hit is a more specific prediction than a
  // 3- or 2-gram hit, and the order it is tried in is fixed (deterministic).
  for (size_t n = kMaxN; n >= kMinN; --n) {
    if (hist_.size() < n) continue;
    const uint64_t* suffix = &hist_[hist_.size() - n];
    const auto it = ngram_[n].find(ngram_key(suffix, n));
    if (it == ngram_[n].end()) continue;
    if (matched_n) *matched_n = n;
    const size_t end = it->second;
    // The proposal is what followed that occurrence: history[end+1 .. L-1].
    const size_t count = std::min(k, hist_.size() - 1 - end);
    if (count == 0) continue;
    for (size_t i = 0; i < count; ++i) out[i] = hist_[end + 1 + i];
    return count;
  }
  return 0;
}

void NGramLongDrafter::reset() noexcept {
  hist_.clear();
  for (auto& m : ngram_) m.clear();
  for (auto& h : roll_) h = 0;
}

void NGramLongDrafter::push(uint64_t token) {
  const size_t s = hist_.size();  // position of the token being pushed
  hist_.push_back(token);
  for (size_t i = 0; i < kClasses; ++i) {
    const size_t n = kLongNs[i];
    if (s >= n) {
      // The rolling state covers the window ending at s-1 — exactly the n-gram
      // whose following token is the one just pushed. Record it, then slide the
      // window one token forward.
      ngram_[i][roll_[i]] = s - 1;
      roll_[i] = (roll_[i] - hist_[s - n] * kLongPow[i]) * kRollBase + token;
    } else {
      roll_[i] = roll_[i] * kRollBase + token;
    }
  }
}

void NGramLongDrafter::push(std::span<const uint64_t> tokens) {
  for (uint64_t t : tokens) push(t);
}

size_t NGramLongDrafter::propose(std::span<uint64_t> out) const {
  return propose(out, nullptr);
}

size_t NGramLongDrafter::propose(std::span<uint64_t> out,
                                 size_t* matched_n) const {
  if (matched_n) *matched_n = 0;
  const size_t k = std::min(out.size(), max_drafts_);
  if (k == 0) return 0;
  const size_t size = hist_.size();
  // Longest context first: a 32-gram hit repeats a whole run of tokens, so its
  // continuation is far more specific than a 3-gram hit's.
  for (size_t i = 0; i < kClasses; ++i) {
    const size_t n = kLongNs[i];
    if (size < n) continue;
    const auto it = ngram_[i].find(roll_[i]);
    if (it == ngram_[i].end()) continue;
    const size_t end = it->second;
    // The table is a hash index: compare the occurrence's tokens with the
    // current suffix before trusting it, so a collision falls through to the
    // next shorter class instead of proposing an unrelated continuation.
    const size_t occ = end + 1 - n;
    const auto at = [this](size_t i) {
      return hist_.begin() + static_cast<std::ptrdiff_t>(i);
    };
    if (!std::equal(at(occ), at(occ + n), at(size - n))) continue;
    const size_t count = std::min(k, size - 1 - end);
    if (count == 0) continue;
    if (matched_n) *matched_n = n;
    for (size_t j = 0; j < count; ++j) out[j] = hist_[end + 1 + j];
    return count;
  }
  return 0;
}

} // namespace sonicboom::planner
