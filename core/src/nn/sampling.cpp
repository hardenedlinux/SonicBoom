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

#include <sonicboom/nn/sampling.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

namespace sonicboom::nn {

namespace {

constexpr uint32_t kPhiloxM0 = 0xD2511F53u;
constexpr uint32_t kPhiloxM1 = 0xCD9E8D57u;
constexpr uint32_t kPhiloxW0 = 0x9E3779B9u;
constexpr uint32_t kPhiloxW1 = 0xBB67AE85u;

inline void mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
  const uint64_t p = uint64_t(a) * uint64_t(b);
  hi = uint32_t(p >> 32);
  lo = uint32_t(p);
}

} // namespace

std::array<uint32_t, 4> philox4x32_10(std::array<uint32_t, 4> counter,
                                      uint64_t key) {
  uint32_t c0 = counter[0], c1 = counter[1], c2 = counter[2], c3 = counter[3];
  uint32_t k0 = uint32_t(key), k1 = uint32_t(key >> 32);
  for (int round = 0; round < 10; ++round) {
    uint32_t hi0, lo0, hi1, lo1;
    mulhilo(kPhiloxM0, c0, hi0, lo0);
    mulhilo(kPhiloxM1, c2, hi1, lo1);
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
    k0 += kPhiloxW0;
    k1 += kPhiloxW1;
  }
  return {c0, c1, c2, c3};
}

bool softcap(std::span<const float> x, float cap, std::span<float> y) {
  if (cap <= 0.0f) return false;
  if (y.size() < x.size()) return false;
  const float inv_cap = 1.0f / cap;
  for (size_t i = 0; i < x.size(); ++i)
    y[i] = cap * std::tanh(x[i] * inv_cap);
  return true;
}

int64_t argmax(std::span<const float> x) {
  if (x.empty()) return -1;
  int64_t best = 0;
  for (size_t i = 1; i < x.size(); ++i)
    if (x[i] > x[best]) best = int64_t(i);
  return best;
}

int64_t argmax_masked(std::span<const float> x, std::span<const uint8_t> mask) {
  int64_t best = -1;
  for (size_t i = 0; i < x.size(); ++i) {
    const size_t byte = i >> 3;
    if (byte >= mask.size()) break;
    if (((mask[byte] >> (i & 7)) & 1u) == 0) continue;
    if (best < 0 || x[i] > x[size_t(best)]) best = int64_t(i);
  }
  return best;
}

namespace {

struct Cand {
  int64_t id;
  float z;
};

// The shared front half of sample_token: penalties, mask, and (for
// temperature > 0) the temperature window, top-k, min-p and top-p filters.
// `adv` holds the penalized logits; `m` the pre-window maximum (0 when greedy).
struct CandidateSet {
  std::vector<float> adv;
  std::vector<Cand> cands;
  float m = 0.0f;
};

CandidateSet build_candidates(std::span<const float> logits,
                              const SamplingParams& p,
                              std::span<const int64_t> history,
                              std::span<const uint8_t> mask) {
  const int64_t vocab = int64_t(logits.size());
  CandidateSet out;
  out.adv.assign(logits.begin(), logits.end());

  // (b) Repetition / presence / frequency penalties over the trailing window.
  // Only in-vocabulary ids are touched; an id is penalized once for its count
  // in the window.
  if (!history.empty() && (p.repeat_penalty != 1.0f ||
                           p.presence_penalty != 0.0f ||
                           p.frequency_penalty != 0.0f)) {
    const size_t window =
        std::min<size_t>(history.size(), size_t(p.repeat_last_n));
    std::unordered_map<int64_t, uint32_t> count;
    count.reserve(window);
    for (size_t i = history.size() - window; i < history.size(); ++i) {
      const int64_t t = history[i];
      if (t >= 0 && t < vocab) ++count[t];
    }
    for (const auto& [t, c] : count) {
      if (p.repeat_penalty != 1.0f) {
        float& v = out.adv[size_t(t)];
        v = v > 0.0f ? v / p.repeat_penalty : v * p.repeat_penalty;
      }
      out.adv[size_t(t)] -=
          p.frequency_penalty * float(c) + p.presence_penalty;
    }
  }

  // (c) Grammar mask: disallowed tokens leave the candidate set. Candidates are
  // built in ascending id order, so every later tie-break toward the lowest id
  // is the stable order.
  out.cands.reserve(size_t(std::max<int64_t>(vocab, 0)));
  for (int64_t i = 0; i < vocab; ++i) {
    if (!mask.empty()) {
      const size_t byte = size_t(i) >> 3;
      if (byte >= mask.size()) break;  // past the bitmap: disallowed
      if (((mask[byte] >> (i & 7)) & 1u) == 0) continue;
    }
    out.cands.push_back({i, 0.0f});
  }
  if (out.cands.empty() || !(p.temperature > 0.0f)) return out;

  // (e) Temperature scaling and the logit window.
  const float inv_t = 1.0f / p.temperature;
  out.m = -std::numeric_limits<float>::infinity();
  for (Cand& c : out.cands) {
    c.z = out.adv[size_t(c.id)] * inv_t;
    out.m = std::max(out.m, c.z);
  }
  const float floor_z = out.m - kSampleLogitWindow;
  out.cands.erase(std::remove_if(out.cands.begin(), out.cands.end(),
                                 [floor_z](const Cand& c) {
                                   return c.z < floor_z;
                                 }),
                  out.cands.end());

  auto by_z = [](const Cand& a, const Cand& b) {
    if (a.z != b.z) return a.z > b.z;
    return a.id < b.id;
  };

  // (f) Top-k under (z descending, id ascending).
  if (p.top_k > 0 && out.cands.size() > p.top_k) {
    std::sort(out.cands.begin(), out.cands.end(), by_z);
    out.cands.resize(p.top_k);
  }

  // (g) Min-p.
  if (p.min_p > 0.0f) {
    const float cutoff = out.m + std::log(p.min_p);
    out.cands.erase(std::remove_if(out.cands.begin(), out.cands.end(),
                                   [cutoff](const Cand& c) {
                                     return c.z < cutoff;
                                   }),
                    out.cands.end());
  }

  // (h) Nucleus: the shortest descending prefix whose normalized probability
  // reaches top_p.
  if (p.top_p > 0.0f && p.top_p < 1.0f) {
    std::sort(out.cands.begin(), out.cands.end(), by_z);
    double total = 0.0;
    for (const Cand& c : out.cands)
      total += std::exp(double(c.z) - double(out.m));
    if (total > 0.0) {
      double acc = 0.0;
      size_t keep = 0;
      for (; keep < out.cands.size(); ++keep) {
        acc += std::exp(double(out.cands[keep].z) - double(out.m));
        if (acc / total >= double(p.top_p)) {
          ++keep;
          break;
        }
      }
      if (keep == 0) keep = 1;
      out.cands.resize(std::min(keep, out.cands.size()));
    }
  }
  return out;
}

} // namespace

int64_t sample_token(std::span<const float> logits, const SamplingParams& p,
                     uint64_t position, std::span<const int64_t> history,
                     std::span<const uint8_t> mask) {
  CandidateSet set = build_candidates(logits, p, history, mask);
  if (set.cands.empty()) return -1;

  // (d) Greedy: argmax of the penalized logits (ties -> lowest id).
  if (!(p.temperature > 0.0f)) {
    int64_t best = set.cands.front().id;
    float best_v = set.adv[size_t(best)];
    for (size_t k = 1; k < set.cands.size(); ++k) {
      const float v = set.adv[size_t(set.cands[k].id)];
      if (v > best_v) {
        best = set.cands[k].id;
        best_v = v;
      }
    }
    return best;
  }

  // (i) Gumbel-max: per-candidate deterministic draw keyed by
  // (seed, position, token id).
  const uint32_t pos_lo = uint32_t(position);
  const uint32_t pos_hi = uint32_t(position >> 32);
  int64_t best = -1;
  double best_score = -std::numeric_limits<double>::infinity();
  for (const Cand& c : set.cands) {
    const auto bits =
        philox4x32_10({pos_lo, pos_hi, uint32_t(c.id), 0u}, p.seed);
    const double u = (double(bits[0]) + 0.5) / 4294967296.0;
    const double g = -std::log(-std::log(u));
    const double score = double(c.z) + g;
    if (best < 0 || score > best_score ||
        (score == best_score && c.id < best)) {
      best = c.id;
      best_score = score;
    }
  }
  return best;
}

std::vector<int64_t> sample_candidates(std::span<const float> logits,
                                       const SamplingParams& p,
                                       std::span<const int64_t> history,
                                       std::span<const uint8_t> mask) {
  CandidateSet set = build_candidates(logits, p, history, mask);
  std::vector<int64_t> ids;
  ids.reserve(set.cands.size());
  for (const Cand& c : set.cands) ids.push_back(c.id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

} // namespace sonicboom::nn
