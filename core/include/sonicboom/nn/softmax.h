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

#include <cstdint>
#include <span>

// Softmax kernels for the native transformer path (float32).

namespace sonicboom::nn {

// Row softmax over a single contiguous slice, numerically stable:
//   y[i] = exp(x[i] - max_j x[j]) / sum_k exp(x[k] - max_j x[j])
//
// Requires y.size() >= x.size(); returns false (writing nothing) otherwise.
bool softmax(std::span<const float> x, std::span<float> y);

// Causal-masked softmax over a contiguous row. `key_pos` is the 0-based
// absolute position of this query token; positions with key index > key_pos are
// masked out (contribute 0). This is the single-query causal mask — multi-query
// / sliding-window masks are the attention layer's job, not this primitive.
//
// Requires y.size() >= x.size(); returns false otherwise. An all-masked row
// (key_pos == 0 with only one element, i.e. key index 0 is allowed, is never
// all-masked because index 0 <= key_pos always holds) is well-defined.
bool softmax_causal(std::span<const float> x, uint64_t key_pos, std::span<float> y);

} // namespace sonicboom::nn
