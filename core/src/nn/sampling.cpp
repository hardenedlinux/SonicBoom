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

// Repetition / presence / frequency penalties over the trailing window, in the
// historic float expression. Used by the greedy path only, whose output must
// stay bit-identical to the pre-sampling implementation.
void apply_penalties_legacy(std::span<const float> logits,
                            const SamplingParams& p,
                            std::span<const int64_t> history,
                            std::vector<float>& adv) {
  const int64_t vocab = int64_t(logits.size());
  if (history.empty() || (p.repeat_penalty == 1.0f &&
                          p.presence_penalty == 0.0f &&
                          p.frequency_penalty == 0.0f))
    return;
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
      float& v = adv[size_t(t)];
      v = v > 0.0f ? v / p.repeat_penalty : v * p.repeat_penalty;
    }
    adv[size_t(t)] -=
        p.frequency_penalty * float(c) + p.presence_penalty;
  }
}

// The same penalties with explicit single-rounding operations only
// (temperature-corrected paths, including the CUDA select kernel):
//   repeat:  v > 0 ? v / rp : v * rp
//   adjust:  v - (freq * count + presence)
// Each arithmetic result is one correctly-rounded IEEE operation; no
// multiply-add contraction is possible across the explicit fma calls, so host
// and device agree bitwise.
void apply_penalties_exact(std::span<const float> logits,
                           const SamplingParams& p,
                           std::span<const int64_t> history,
                           std::vector<float>& adv) {
  const int64_t vocab = int64_t(logits.size());
  if (history.empty() || (p.repeat_penalty == 1.0f &&
                          p.presence_penalty == 0.0f &&
                          p.frequency_penalty == 0.0f))
    return;
  const size_t window =
      std::min<size_t>(history.size(), size_t(p.repeat_last_n));
  std::unordered_map<int64_t, uint32_t> count;
  count.reserve(window);
  for (size_t i = history.size() - window; i < history.size(); ++i) {
    const int64_t t = history[i];
    if (t >= 0 && t < vocab) ++count[t];
  }
  for (const auto& [t, c] : count) {
    float v = adv[size_t(t)];
    if (p.repeat_penalty != 1.0f) {
      // fma(v, rp, 0) rounds exactly like a plain multiply but cannot be
      // fused with the subtraction below.
      v = v > 0.0f ? v / p.repeat_penalty : std::fma(v, p.repeat_penalty, 0.0f);
    }
    const float adjust =
        std::fma(p.frequency_penalty, float(c), p.presence_penalty);
    adv[size_t(t)] = std::fma(-1.0f, adjust, v);
  }
}

auto by_z_desc_id_asc = [](const Candidate& a, const Candidate& b) {
  if (a.z != b.z) return a.z > b.z;
  return a.id < b.id;
};

// Stages (e)-(h) of the sampler: the logit window, top-k, min-p and the
// nucleus. `in` is ordered by (z descending, id ascending) and every filter
// preserves that order.
std::vector<Candidate> filter_candidates(std::span<const Candidate> in,
                                         const SamplingParams& p) {
  std::vector<Candidate> out;
  if (in.empty()) return out;
  const float m = in.front().z;

  // (e) Logit window.
  const float floor_z = m - kSampleLogitWindow;
  out.reserve(in.size());
  for (const Candidate& c : in)
    if (c.z >= floor_z) out.push_back(c);

  // (f) Top-k (the input order is already (z descending, id ascending)).
  if (p.top_k > 0 && out.size() > p.top_k) out.resize(p.top_k);

  // (g) Min-p.
  if (p.min_p > 0.0f) {
    const float cutoff = m + std::log(p.min_p);
    out.erase(std::remove_if(out.begin(), out.end(),
                             [cutoff](const Candidate& c) {
                               return c.z < cutoff;
                             }),
              out.end());
  }

  // (h) Nucleus: the shortest descending prefix whose normalized probability
  // reaches top_p.
  if (p.top_p > 0.0f && p.top_p < 1.0f && !out.empty()) {
    double total = 0.0;
    for (const Candidate& c : out)
      total += std::exp(double(c.z) - double(m));
    if (total > 0.0) {
      double acc = 0.0;
      size_t keep = 0;
      for (; keep < out.size(); ++keep) {
        acc += std::exp(double(out[keep].z) - double(m));
        if (acc / total >= double(p.top_p)) {
          ++keep;
          break;
        }
      }
      if (keep == 0) keep = 1;
      out.resize(std::min(keep, out.size()));
    }
  }
  return out;
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

std::vector<Candidate> select_candidates_host(std::span<const float> logits,
                                              const SamplingParams& p,
                                              std::span<const int64_t> history,
                                              std::span<const uint8_t> mask) {
  const int64_t vocab = int64_t(logits.size());
  std::vector<Candidate> out;
  if (!(p.temperature > 0.0f) || vocab <= 0) return out;

  std::vector<float> adv(logits.begin(), logits.end());
  apply_penalties_exact(logits, p, history, adv);

  out.reserve(size_t(vocab));
  for (int64_t i = 0; i < vocab; ++i) {
    if (!mask.empty()) {
      const size_t byte = size_t(i) >> 3;
      if (byte >= mask.size()) break;  // past the bitmap: disallowed
      if (((mask[byte] >> (i & 7)) & 1u) == 0) continue;
    }
    // z = l_adj / T as one float division (the CUDA kernel's __fdiv_rn).
    out.push_back({int32_t(i), adv[size_t(i)] / p.temperature});
  }
  if (int(out.size()) > kSampleCandidates) {
    std::nth_element(out.begin(), out.begin() + kSampleCandidates, out.end(),
                     by_z_desc_id_asc);
    out.resize(size_t(kSampleCandidates));
  }
  std::sort(out.begin(), out.end(), by_z_desc_id_asc);
  return out;
}

int64_t sample_from_candidates(std::span<const Candidate> cands,
                               const SamplingParams& p, uint64_t position) {
  if (cands.empty()) return -1;
  const std::vector<Candidate> kept = filter_candidates(cands, p);
  if (kept.empty()) return -1;

  // (i) Gumbel-max: per-candidate deterministic draw keyed by
  // (seed, position, token id).
  const uint32_t pos_lo = uint32_t(position);
  const uint32_t pos_hi = uint32_t(position >> 32);
  int64_t best = -1;
  double best_score = -std::numeric_limits<double>::infinity();
  for (const Candidate& c : kept) {
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

int64_t sample_token(std::span<const float> logits, const SamplingParams& p,
                     uint64_t position, std::span<const int64_t> history,
                     std::span<const uint8_t> mask) {
  if (logits.empty()) return -1;

  // Greedy: the argmax of the penalized logits (ties -> lowest id), with the
  // historic arithmetic so temperature-0 streams stay bit-identical.
  if (!(p.temperature > 0.0f)) {
    std::vector<float> adv(logits.begin(), logits.end());
    apply_penalties_legacy(logits, p, history, adv);
    int64_t best = -1;
    float best_v = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
      if (!mask.empty()) {
        const size_t byte = i >> 3;
        if (byte >= mask.size()) break;
        if (((mask[byte] >> (i & 7)) & 1u) == 0) continue;
      }
      const float v = adv[i];
      if (best < 0 || v > best_v) {
        best = int64_t(i);
        best_v = v;
      }
    }
    return best;
  }

  const std::vector<Candidate> cands =
      select_candidates_host(logits, p, history, mask);
  return sample_from_candidates(cands, p, position);
}

std::vector<int64_t> sample_candidates(std::span<const float> logits,
                                       const SamplingParams& p,
                                       std::span<const int64_t> history,
                                       std::span<const uint8_t> mask) {
  std::vector<int64_t> ids;
  if (logits.empty()) return ids;
  if (!(p.temperature > 0.0f)) {
    for (size_t i = 0; i < logits.size(); ++i) {
      const size_t byte = i >> 3;
      if (byte >= mask.size()) break;
      if (((mask[byte] >> (i & 7)) & 1u) == 0) continue;
      ids.push_back(int64_t(i));
    }
    return ids;
  }
  const std::vector<Candidate> cands =
      select_candidates_host(logits, p, history, mask);
  const std::vector<Candidate> kept = filter_candidates(cands, p);
  ids.reserve(kept.size());
  for (const Candidate& c : kept) ids.push_back(c.id);
  std::sort(ids.begin(), ids.end());
  return ids;
}

} // namespace sonicboom::nn
