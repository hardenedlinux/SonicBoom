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

#include <cmath>

namespace sonicboom::nn {

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

} // namespace sonicboom::nn
