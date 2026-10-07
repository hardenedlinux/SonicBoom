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

#include <sonicboom/nn/rope.h>

#include <atomic>
#include <cmath>

#include <sonicboom/thread_pool.h>

namespace sonicboom::nn {

bool rope_neox(std::span<float> x, uint64_t pos, float theta_base,
               float freq_scale, std::span<const float> freq_factors) {
  const size_t n = x.size();
  if (n == 0 || (n & 1) != 0) return false;

  const size_t half = n / 2;
  if (!freq_factors.empty() && freq_factors.size() < half) return false;

  const float base = theta_base;
  const float log_base = std::log(base);

  for (size_t i = 0; i < half; ++i) {
    // theta = pos * base^(-2*i/n) = pos * exp(-2*i/n * log(base)). Computing the
    // exponent in float (rather than pow per element) keeps this a streaming
    // kernel; for the head dims Gemma uses (<= 256) float precision is ample.
    float theta = freq_scale * float(pos) *
                  std::exp(-2.0f * float(i) / float(n) * log_base);
    // Proportional RoPE: a per-pair factor divides the angle; a 1e30 sentinel
    // drives it to ~0, disabling the rotation for that pair.
    if (!freq_factors.empty()) theta /= freq_factors[i];

    const float c = std::cos(theta);
    const float s = std::sin(theta);

    const float x0 = x[i];
    const float x1 = x[i + half];
    x[i] = x0 * c - x1 * s;
    x[i + half] = x0 * s + x1 * c;
  }
  return true;
}

bool rope_neox_heads(std::span<float> x, uint64_t head_dim, uint64_t pos,
                     float theta_base, float freq_scale,
                     std::span<const float> freq_factors) {
  if (head_dim == 0 || (head_dim & 1) != 0 || x.size() == 0 ||
      x.size() % head_dim != 0)
    return false;
  if (!freq_factors.empty() && freq_factors.size() < head_dim / 2) return false;

  const uint64_t heads = x.size() / head_dim;
  // Each head writes its own span, so the result is deterministic even though the
  // heads run in parallel.
  std::atomic<bool> ok{true};
  ThreadPool::instance().parallel_for(0, heads, [&](uint64_t b, uint64_t e) {
    for (uint64_t h = b; h < e; ++h)
      if (!rope_neox(x.subspan(h * head_dim, head_dim), pos, theta_base,
                     freq_scale, freq_factors))
        ok = false;
  });
  return ok;
}

} // namespace sonicboom::nn
