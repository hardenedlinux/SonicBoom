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

#include <sonicboom/training/clip_grad_norm.h>

#include <cmath>

namespace sonicboom::training {

double grad_global_norm(std::span<float* const> grads,
                        std::span<const std::size_t> sizes) noexcept {
  double sumsq = 0.0;
  const std::size_t n = grads.size() < sizes.size() ? grads.size() : sizes.size();
  for (std::size_t t = 0; t < n; ++t) {
    const float* g = grads[t];
    if (g == nullptr) continue;
    for (std::size_t i = 0; i < sizes[t]; ++i) {
      const double v = static_cast<double>(g[i]);
      sumsq += v * v;
    }
  }
  return std::sqrt(sumsq);
}

double clip_grad_norm_inplace(std::span<float* const> grads,
                              std::span<const std::size_t> sizes,
                              double max_norm) noexcept {
  const double norm = grad_global_norm(grads, sizes);
  if (!(norm > max_norm)) return norm; // also false for a NaN norm: no clip
  const double scale = max_norm / (norm + kClipGradNormEps);
  const std::size_t n = grads.size() < sizes.size() ? grads.size() : sizes.size();
  for (std::size_t t = 0; t < n; ++t) {
    float* g = grads[t];
    if (g == nullptr) continue;
    for (std::size_t i = 0; i < sizes[t]; ++i)
      g[i] = static_cast<float>(static_cast<double>(g[i]) * scale);
  }
  return norm;
}

} // namespace sonicboom::training
