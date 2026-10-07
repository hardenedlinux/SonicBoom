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

#include <sonicboom/nn/softmax.h>

#include <cmath>
#include <limits>

namespace sonicboom::nn {

namespace {

// max-subtraction softmax over indices [0, n). Returns false if every element
// is masked (max == -inf), leaving y untouched.
bool softmax_range(std::span<const float> x, uint64_t n, std::span<float> y) {
  float m = -std::numeric_limits<float>::infinity();
  for (uint64_t i = 0; i < n; ++i) m = std::max(m, x[i]);
  if (m == -std::numeric_limits<float>::infinity()) return false;

  float sum = 0.0f;
  for (uint64_t i = 0; i < n; ++i) {
    const float e = std::exp(x[i] - m);
    y[i] = e;
    sum += e;
  }
  const float inv = 1.0f / sum;
  for (uint64_t i = 0; i < n; ++i) y[i] *= inv;
  return true;
}

} // namespace

bool softmax(std::span<const float> x, std::span<float> y) {
  if (y.size() < x.size()) return false;
  return softmax_range(x, x.size(), y);
}

bool softmax_causal(std::span<const float> x, uint64_t key_pos, std::span<float> y) {
  if (y.size() < x.size()) return false;
  // Allowed keys are indices [0, key_pos]. key_pos < x.size() is not required:
  // when the query position exceeds the key count (fully-cached decode), every
  // key is attended to, i.e. the mask is empty.
  const uint64_t n = std::min<uint64_t>(x.size(), key_pos + 1);
  // Mask (only) the positions that are never read, so a subsequent caller can't
  // accidentally observe stale y values for masked slots.
  for (uint64_t i = n; i < x.size(); ++i) y[i] = 0.0f;
  return softmax_range(x, n, y);
}

} // namespace sonicboom::nn
