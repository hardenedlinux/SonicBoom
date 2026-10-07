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

#include <sonicboom/nn/norm.h>

#include <cmath>

namespace sonicboom::nn {

bool rms_norm(std::span<const float> x, std::span<const float> weight, float eps,
              std::span<float> y) {
  if (y.size() < x.size()) return false;
  if (!weight.empty() && weight.size() < x.size()) return false;

  // mean(x^2). Accumulate in double (ggml's ggml_float), but each product is the
  // f32 x[i]*x[i] (ggml casts that rounded f32 product to double before adding),
  // so the sum reproduces llama.cpp's rms_norm bit-for-bit.
  double sum_sq = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    const float v = x[i];
    sum_sq += double(v * v);
  }
  const float inv = 1.0f / std::sqrt(float(sum_sq / double(x.size())) + eps);

  if (weight.empty()) {
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv;
  } else {
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * weight[i];
  }
  return true;
}

bool rms_norm_heads(std::span<float> x, std::span<const float> weight,
                    uint64_t head_dim, float eps) {
  if (head_dim == 0 || x.size() == 0 || x.size() % head_dim != 0) return false;
  if (!weight.empty() && weight.size() < head_dim) return false;

  const uint64_t heads = x.size() / head_dim;
  for (uint64_t h = 0; h < heads; ++h) {
    std::span<float> xh = x.subspan(h * head_dim, head_dim);
    // mean(x^2), accumulated in double exactly as rms_norm does (each product is
    // the f32 x[i]*x[i]).
    double sum_sq = 0.0;
    for (uint64_t i = 0; i < head_dim; ++i) {
      const float v = xh[i];
      sum_sq += double(v * v);
    }
    const float inv = 1.0f / std::sqrt(float(sum_sq / double(head_dim)) + eps);
    if (weight.empty()) {
      for (uint64_t i = 0; i < head_dim; ++i) xh[i] *= inv;
    } else {
      for (uint64_t i = 0; i < head_dim; ++i) xh[i] = xh[i] * inv * weight[i];
    }
  }
  return true;
}

} // namespace sonicboom::nn
