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

// The gated DeltaNet scan (float32): one kernel serves decode (one row), verify
// (M rows) and prefill (n rows). A scan over n rows is n single-row updates in
// order — row t+1 sees the histogram and memory row t left — so the state after
// any scan is the state the same inputs give token by token.
//
// The bit-relevant decisions are part of the contract:
//   - the key-head normalization accumulates its square sum in double;
//   - the memory pass accumulates the dot products in float, ascending i;
//   - the convolution reads the histogram oldest tap first and appends the RAW
//     qkv row (not the activated one) after the row is computed.

#include <cstdint>
#include <span>

namespace sonicboom::nn {

// The recurrence's geometry. `dim` is the head size S, keys and values alike;
// `kernel` is the convolution's tap count K; `eps` is the model's RMS-norm
// epsilon.
struct GdnDims {
  uint32_t heads_k = 0;
  uint32_t heads_v = 0;
  uint32_t dim = 0;
  uint32_t kernel = 0;
  float eps = 0.0f;
};

// The per-block recurrence weights. `conv` is [channels * kernel] with tap i of
// channel c at c * kernel + i; `dt` and `a` are [heads_v].
struct GdnWeights {
  std::span<const float> conv;
  std::span<const float> dt;
  std::span<const float> a;
};

// Runs rows 0..n-1 of one block's recurrence, updating `hist` and `mem` in
// place. Shapes: qkv [n*C], beta_logit [n*heads_v], alpha_logit [n*heads_v],
// hist [(kernel-1)*C] with hist[j*C+c] (j=0 oldest), mem
// [heads_v*dim*dim] with mem[(h*dim + j)*dim + i] (j the value index, i the key
// index), out [n*heads_v*dim].
//
// Returns false when a span's size does not match the geometry.
bool gdn_scan(const GdnDims& dims, const GdnWeights& weights,
              std::span<const float> qkv, std::span<const float> beta_logit,
              std::span<const float> alpha_logit, std::span<float> hist,
              std::span<float> mem, std::span<float> out);

} // namespace sonicboom::nn
