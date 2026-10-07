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

#include <sonicboom/nn/attention.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sonicboom::nn {

bool scaled_dot_product_attention(std::span<const float> q, uint64_t n_q,
                                  std::span<const float> k,
                                  std::span<const float> v, uint64_t n_k,
                                  uint64_t n_heads_q, uint64_t n_heads_kv,
                                  uint64_t head_dim, bool causal,
                                  uint64_t sliding_window, float softcap,
                                  float scale, std::span<float> out) {
  if (head_dim == 0 || n_heads_q == 0 || n_heads_kv == 0) return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  const uint64_t group = n_heads_q / n_heads_kv;

  if (q.size() != n_q * n_heads_q * head_dim) return false;
  if (k.size() != n_k * n_heads_kv * head_dim) return false;
  if (v.size() != n_k * n_heads_kv * head_dim) return false;
  if (out.size() != n_q * n_heads_q * head_dim) return false;

  std::vector<float> scores(n_k);

  for (uint64_t i = 0; i < n_q; ++i) {
    for (uint64_t h = 0; h < n_heads_q; ++h) {
      const uint64_t kv_h = h / group;
      const float* qrow = q.data() + (i * n_heads_q + h) * head_dim;

      // Scores over all keys.
      for (uint64_t j = 0; j < n_k; ++j) {
        const float* krow = k.data() + (j * n_heads_kv + kv_h) * head_dim;
        float dot = 0.0f;
        for (uint64_t d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
        scores[j] = dot * scale;
      }

      // Softcap (Gemma attention logit softcapping): score -> cap*tanh(score/cap).
      if (softcap > 0.0f) {
        for (uint64_t j = 0; j < n_k; ++j)
          scores[j] = softcap * std::tanh(scores[j] / softcap);
      }

      // Mask and softmax in one pass (max-subtraction for stability).
      float max_score = -std::numeric_limits<float>::infinity();
      for (uint64_t j = 0; j < n_k; ++j) {
        const bool masked =
            (causal && j > i) ||
            (sliding_window > 0 && i >= j + sliding_window);
        if (masked) continue;
        max_score = std::max(max_score, scores[j]);
      }

      float* orow = out.data() + (i * n_heads_q + h) * head_dim;
      if (max_score == -std::numeric_limits<float>::infinity()) {
        // No unmasked keys: the reference yields a zero output row.
        for (uint64_t d = 0; d < head_dim; ++d) orow[d] = 0.0f;
        continue;
      }

      float sum = 0.0f;
      for (uint64_t j = 0; j < n_k; ++j) {
        const bool masked =
            (causal && j > i) ||
            (sliding_window > 0 && i >= j + sliding_window);
        if (masked) continue;
        const float e = std::exp(scores[j] - max_score);
        scores[j] = e;  // reuse scores as the (unnormalized) probabilities
        sum += e;
      }
      const float inv = 1.0f / sum;

      for (uint64_t d = 0; d < head_dim; ++d) orow[d] = 0.0f;
      for (uint64_t j = 0; j < n_k; ++j) {
        const bool masked =
            (causal && j > i) ||
            (sliding_window > 0 && i >= j + sliding_window);
        if (masked) continue;
        const float prob = scores[j] * inv;
        const float* vrow = v.data() + (j * n_heads_kv + kv_h) * head_dim;
        for (uint64_t d = 0; d < head_dim; ++d) orow[d] += prob * vrow[d];
      }
    }
  }
  return true;
}

bool gqa_broadcast(std::span<const float> src, std::span<float> dst,
                   uint64_t n_heads_q, uint64_t n_heads_kv, uint64_t head_dim) {
  if (head_dim == 0 || n_heads_q == 0 || n_heads_kv == 0) return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  if (src.size() < n_heads_kv * head_dim) return false;
  if (dst.size() < n_heads_q * head_dim) return false;

  for (uint64_t hq = 0; hq < n_heads_q; ++hq) {
    const uint64_t kv_h = hq * n_heads_kv / n_heads_q;
    std::copy_n(src.begin() + kv_h * head_dim, head_dim, dst.begin() + hq * head_dim);
  }
  return true;
}

bool decode_attention(std::span<const float> q, std::span<const float> k_cache,
                      std::span<const float> v_cache, uint64_t pos,
                      uint64_t n_slots, uint64_t n_heads_q, uint64_t n_heads_kv,
                      uint64_t head_dim, uint64_t sliding_window, float scale,
                      std::span<float> out) {
  if (head_dim == 0 || n_heads_q == 0 || n_heads_kv == 0 || n_slots == 0)
    return false;
  if (n_heads_q % n_heads_kv != 0) return false;
  if (q.size() != n_heads_q * head_dim) return false;
  if (k_cache.size() != n_slots * n_heads_kv * head_dim) return false;
  if (v_cache.size() != n_slots * n_heads_kv * head_dim) return false;
  if (out.size() != n_heads_q * head_dim) return false;

  const uint64_t group = n_heads_q / n_heads_kv;

  // Valid key positions [start, pos]: causal, further windowed for SWA layers.
  const uint64_t start =
      (sliding_window > 0 && pos >= sliding_window) ? pos - sliding_window + 1 : 0;
  const uint64_t n_keys = pos - start + 1;

  std::vector<float> scores(n_keys);

  for (uint64_t h = 0; h < n_heads_q; ++h) {
    const uint64_t kv_h = h / group;
    const float* qrow = q.data() + h * head_dim;

    for (uint64_t idx = 0; idx < n_keys; ++idx) {
      const uint64_t slot = (start + idx) % n_slots;
      const float* krow = k_cache.data() + (slot * n_heads_kv + kv_h) * head_dim;
      float dot = 0.0f;
      for (uint64_t d = 0; d < head_dim; ++d) dot += qrow[d] * krow[d];
      scores[idx] = dot * scale;
    }

    // Softmax over the valid keys (max-subtraction for stability).
    float max_score = -std::numeric_limits<float>::infinity();
    for (uint64_t idx = 0; idx < n_keys; ++idx)
      max_score = std::max(max_score, scores[idx]);

    float* orow = out.data() + h * head_dim;
    if (max_score == -std::numeric_limits<float>::infinity()) {
      for (uint64_t d = 0; d < head_dim; ++d) orow[d] = 0.0f;
      continue;
    }

    float sum = 0.0f;
    for (uint64_t idx = 0; idx < n_keys; ++idx) {
      const float e = std::exp(scores[idx] - max_score);
      scores[idx] = e;
      sum += e;
    }
    const float inv = 1.0f / sum;

    for (uint64_t d = 0; d < head_dim; ++d) orow[d] = 0.0f;
    for (uint64_t idx = 0; idx < n_keys; ++idx) {
      const uint64_t slot = (start + idx) % n_slots;
      const float prob = scores[idx] * inv;
      const float* vrow = v_cache.data() + (slot * n_heads_kv + kv_h) * head_dim;
      for (uint64_t d = 0; d < head_dim; ++d) orow[d] += prob * vrow[d];
    }
  }
  return true;
}

} // namespace sonicboom::nn
