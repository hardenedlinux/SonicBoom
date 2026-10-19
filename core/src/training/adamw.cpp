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

// AdamW host reference. This is the CPU fallback for a host-resident Param and
// the formula the device kernel in adamw.cu is checked against.

#include <sonicboom/training/adamw.h>

#include <cmath>

namespace sonicboom::training {

void adamw_f32(float* param, const float* grad, float* m1, float* m2,
               std::size_t n, const AdamWConfig& cfg, std::uint64_t step) {
  const double b1 = cfg.beta1;
  const double b2 = cfg.beta2;
  const double bc1 = 1.0 - std::pow(b1, static_cast<double>(step));
  const double bc2 = 1.0 - std::pow(b2, static_cast<double>(step));
  const double lr = cfg.lr;
  const double eps = cfg.eps;
  const double wd = cfg.weight_decay;

  for (std::size_t i = 0; i < n; ++i) {
    const double g = grad[i];
    const double mi = b1 * m1[i] + (1.0 - b1) * g;
    const double vi = b2 * m2[i] + (1.0 - b2) * g * g;
    // The moment buffers are float32 state: round now, so the next step starts
    // from exactly the value the device path would have stored.
    m1[i] = static_cast<float>(mi);
    m2[i] = static_cast<float>(vi);

    const double mh = mi / bc1;
    const double vh = vi / bc2;
    const double p = param[i];
    param[i] =
        static_cast<float>(p - lr * (mh / (std::sqrt(vh) + eps) + wd * p));
  }
}

} // namespace sonicboom::training
