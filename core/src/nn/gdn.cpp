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

// The gated DeltaNet scan, CPU (nn/gdn.h).

#include <sonicboom/nn/gdn.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace sonicboom::nn {

bool gdn_scan(const GdnDims& dims, const GdnWeights& weights,
              std::span<const float> qkv, std::span<const float> beta_logit,
              std::span<const float> alpha_logit, std::span<float> hist,
              std::span<float> mem, std::span<float> out) {
  const uint64_t hk = dims.heads_k;
  const uint64_t hv = dims.heads_v;
  const uint64_t S = dims.dim;
  const uint64_t K = dims.kernel;
  if (hk == 0 || hv == 0 || S == 0 || K == 0 || hv % hk != 0)
    return false;
  const uint64_t key_dim = hk * S;
  const uint64_t value_dim = hv * S;
  const uint64_t C = 2 * key_dim + value_dim;
  if (qkv.size() % C != 0)
    return false;
  const uint64_t n = qkv.size() / C;
  if (beta_logit.size() != n * hv || alpha_logit.size() != n * hv ||
      hist.size() != (K - 1) * C || mem.size() != hv * S * S ||
      out.size() != n * value_dim)
    return false;
  if (weights.conv.size() != C * K || weights.dt.size() != hv ||
      weights.a.size() != hv)
    return false;

  const float inv_sqrt_S = 1.0f / std::sqrt(static_cast<float>(S));
  std::vector<float> y(C);  // one row's convolution and activation output

  for (uint64_t t = 0; t < n; ++t) {
    const float* const row = qkv.data() + t * C;

    // (a) Convolution over the last K raw rows and SiLU, channel by channel.
    for (uint64_t c = 0; c < C; ++c) {
      float sum = 0.0f;
      for (uint64_t i = 0; i < K; ++i) {
        const float x = (i < K - 1) ? hist[i * C + c] : row[c];
        sum += x * weights.conv[c * K + i];
      }
      y[c] = sum / (1.0f + std::exp(-sum));
    }
    // The histogram keeps RAW rows: drop the oldest, append this row.
    if (K > 1) {
      for (uint64_t j = 0; j + 2 < K; ++j)
        for (uint64_t c = 0; c < C; ++c)
          hist[j * C + c] = hist[(j + 1) * C + c];
      for (uint64_t c = 0; c < C; ++c)
        hist[(K - 2) * C + c] = row[c];
    }

    // (b) Split and (c) normalize q and k, each key head separately. The
    // square sum accumulates in double; the two scales are applied in the
    // order the reference applies them.
    for (uint64_t g = 0; g < hk; ++g) {
      for (uint64_t half = 0; half < 2; ++half) {
        float* x = y.data() + half * key_dim + g * S;
        double sumsq = 0.0;
        for (uint64_t i = 0; i < S; ++i)
          sumsq += double(x[i]) * double(x[i]);
        const double mean = sumsq / double(S);
        const float inv =
            static_cast<float>(1.0 / std::sqrt(mean + double(dims.eps) /
                                                          double(S)));
        for (uint64_t i = 0; i < S; ++i) {
          x[i] = x[i] * inv;
          x[i] = x[i] * inv_sqrt_S;
        }
      }
    }

    // (d) Gates and (e) the memory update, one value head at a time.
    for (uint64_t h = 0; h < hv; ++h) {
      const uint64_t g = h % hk;
      const float* const q = y.data() + g * S;
      const float* const k = y.data() + key_dim + g * S;
      const float* const v = y.data() + 2 * key_dim + h * S;

      const float beta = 1.0f / (1.0f + std::exp(-beta_logit[t * hv + h]));
      const float t_gate = alpha_logit[t * hv + h] + weights.dt[h];
      const float sp = t_gate > 20.0f
                           ? t_gate
                           : std::log(1.0f + std::exp(t_gate));
      const float eg = std::exp(sp * weights.a[h]);

      float* const M = mem.data() + h * S * S;
      float* const o = out.data() + t * value_dim + h * S;
      for (uint64_t j = 0; j < S; ++j) {
        float* const Mj = M + j * S;
        for (uint64_t i = 0; i < S; ++i) Mj[i] *= eg;
        float sk = 0.0f;
        for (uint64_t i = 0; i < S; ++i) sk += Mj[i] * k[i];
        const float delta = (v[j] - sk) * beta;
        for (uint64_t i = 0; i < S; ++i) Mj[i] += k[i] * delta;
        float acc = 0.0f;
        for (uint64_t i = 0; i < S; ++i) acc += Mj[i] * q[i];
        o[j] = acc * inv_sqrt_S;
      }
    }
  }
  return true;
}

} // namespace sonicboom::nn
