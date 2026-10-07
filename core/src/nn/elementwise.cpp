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

#include <sonicboom/nn/elementwise.h>

#include <stdfloat>

namespace sonicboom::nn {

bool mul(std::span<const float> a, std::span<const float> b, std::span<float> y) {
  if (a.size() != b.size() || y.size() < a.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] * b[i];
  return true;
}

bool scale(std::span<const float> x, float s, std::span<float> y) {
  if (y.size() < x.size()) return false;
  for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s;
  return true;
}

bool add(std::span<const float> a, std::span<const float> b, std::span<float> y) {
  if (a.size() != b.size() || y.size() < a.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) y[i] = a[i] + b[i];
  return true;
}

bool axpy(std::span<const float> a, float alpha, std::span<float> y) {
  if (y.size() < a.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) y[i] += alpha * a[i];
  return true;
}

void cast_fp16(std::span<float> v) {
  for (float& x : v) x = float(std::float16_t(x));
}

} // namespace sonicboom::nn
