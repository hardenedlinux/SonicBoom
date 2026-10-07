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

// Normalization kernels for the native transformer path (float32).

namespace sonicboom::nn {

// Root-mean-square normalization (Gemma's pre/post-attention norm).
//
//   y[i] = (x[i] / sqrt(mean_j(x[j]^2) + eps)) * weight[i]
//
// `x` and `y` have length n. `weight` is the learned per-dimension scale; pass
// an empty span for unit weights. `eps` must be >= 0 (the model's
// attention_layer_norm_rms_epsilon).
//
// Returns false (writing nothing) when y.size() < x.size(), or `weight` is
// non-empty and weight.size() < x.size().
bool rms_norm(std::span<const float> x, std::span<const float> weight, float eps,
              std::span<float> y);

// In-place per-head RMSNorm over a concatenated multi-head buffer: `x` is
// `heads` heads of `head_dim` floats laid back-to-back, and head h is normalized
// in place with the same (shared) weight vector:
//   x[h*head_dim + i] = (x[h*head_dim + i] / sqrt(mean_i(x[h*head_dim+i]^2) + eps)) * weight[i]
// `weight` is a single head_dim-length vector shared across heads (empty =>
// unit). This is the Q/K/V per-head norm of the decode block, hoisted to a named
// kernel; each head's sum is accumulated before its write, so the in-place write
// is bit-identical to the original norm-into-scratch-then-copy.
//
// Returns false (writing nothing) when head_dim == 0, x.size() == 0,
// x.size() % head_dim != 0, or `weight` is non-empty and weight.size() < head_dim.
bool rms_norm_heads(std::span<float> x, std::span<const float> weight,
                    uint64_t head_dim, float eps);

} // namespace sonicboom::nn
