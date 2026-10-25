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

#include <array>
#include <cstdint>
#include <span>
#include <vector>

// Logit softcapping, greedy token selection and seeded sampling for the decode
// loop (float32).

namespace sonicboom::nn {

// Gemma final-logit softcapping (final_logit_softcapping = 30 for Gemma 4):
//   y[i] = cap * tanh(x[i] / cap)
// Requires cap > 0 and y.size() >= x.size(); returns false otherwise.
bool softcap(std::span<const float> x, float cap, std::span<float> y);

// Index of the maximum element (ties broken toward the lowest index). Returns
// -1 for an empty span. This is the greedy (temperature-0) token selection.
int64_t argmax(std::span<const float> x);

// Constrained variant: the same argmax, but only ids whose bit is set in
// `mask` (bit i of mask[i / 8], least-significant bit first) are candidates.
// Returns -1 when the mask has no set bit among x's indices. A mask shorter
// than ceil(x.size() / 8) is treated as disallowing anything past its end.
int64_t argmax_masked(std::span<const float> x, std::span<const uint8_t> mask);

// Sampling parameters. The defaults are the greedy/neutral configuration:
// temperature 0 returns the argmax, every filter is a no-op and every penalty
// is inert.
struct SamplingParams {
  float temperature = 0.0f;
  uint32_t top_k = 0;              // 0 = no top-k truncation
  float top_p = 1.0f;              // 1 = no nucleus truncation
  float min_p = 0.0f;
  uint64_t seed = 0;
  float repeat_penalty = 1.0f;
  uint32_t repeat_last_n = 64;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
};

// Candidates whose z = l/T is more than this below the maximum are dropped
// before top-k/min-p/top-p: the probability ratio there is below 1e-13.
inline constexpr float kSampleLogitWindow = 30.0f;

// Sampling draws only from the kSampleCandidates highest-z candidates
// (z = penalized logit / temperature, ties toward the lower id). This is exact
// whenever top_k <= kSampleCandidates or the nucleus lies inside the top
// kSampleCandidates; otherwise the tail below the 256th candidate is never
// selected.
inline constexpr int kSampleCandidates = 256;

// One sampling candidate: the token id and its temperature-scaled, penalized
// logit z = l_adj / T (float, computed as a single division).
struct Candidate {
  int32_t id;
  float z;
};

// Philox4x32-10 with the standard Random123 constants and a 64-bit key
// (key0 = low 32 bits, key1 = high 32 bits). Known answer:
// philox4x32_10({0,0,0,0}, 0) == {0x6627e8d5, 0xe169c58d, 0xbc57ac4c,
// 0x9b00dbd8}.
std::array<uint32_t, 4> philox4x32_10(std::array<uint32_t, 4> counter,
                                      uint64_t key);

// Deterministic token sampling.
//
//   logits  already softcapped where the model has a logit softcap (the values
//           argmax sees in the greedy path).
//   p       the sampling configuration.
//   position absolute sequence position of the token being produced.
//   history prompt + generated tokens so far, in order (penalties only).
//   mask    constrained-decoding bitmap, may be empty (see argmax_masked).
//
// Order: penalties over the last repeat_last_n history tokens -> mask ->
// temperature <= 0 argmax -> select_candidates_host -> sample_from_candidates.
// For temperature <= 0 the result is the argmax of the penalized logits (ties
// toward the lowest id), unchanged from the historic greedy path. For
// temperature > 0 sampling draws from the top kSampleCandidates candidates.
// Returns -1 when no candidate remains.
int64_t sample_token(std::span<const float> logits, const SamplingParams& p,
                     uint64_t position, std::span<const int64_t> history,
                     std::span<const uint8_t> mask);

// Stage (1) of sampling: penalties over the last repeat_last_n history tokens,
// the grammar mask, and z = l_adj / temperature in float. The
// kSampleCandidates highest z survive (ties toward the lower id) and are
// returned in (z descending, id ascending) order. The penalties use explicit
// single-rounding operations so the CUDA select kernel produces bitwise
// identical z values. Requires p.temperature > 0.
std::vector<Candidate> select_candidates_host(std::span<const float> logits,
                                              const SamplingParams& p,
                                              std::span<const int64_t> history,
                                              std::span<const uint8_t> mask);

// Stage (2) of sampling: kSampleLogitWindow, top_k, min_p and top_p over the
// given candidate order, then a Gumbel-max draw with a Philox4x32-10 keyed by
// (seed, position, candidate id). Returns -1 for an empty candidate list.
int64_t sample_from_candidates(std::span<const Candidate> cands,
                               const SamplingParams& p, uint64_t position);

// Test seam for sample_token's filter stages (penalties, mask, the temperature
// window, top_k, min_p, top_p): the surviving candidate ids in ascending order.
// Temperature <= 0 returns every masked id, because the greedy stage does not
// filter. The Gumbel draw is not part of the set.
std::vector<int64_t> sample_candidates(std::span<const float> logits,
                                       const SamplingParams& p,
                                       std::span<const int64_t> history,
                                       std::span<const uint8_t> mask);

} // namespace sonicboom::nn
