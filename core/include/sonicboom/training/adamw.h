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

// AdamW (decoupled weight decay) — the training update kernel's shared
// definition. One fused pass over a flat float32 parameter:
//
//   m1 = b1*m1 + (1-b1)*g
//   m2 = b2*m2 + (1-b2)*g*g
//   p -= lr * (m1/(1-b1^t) / (sqrt(m2/(1-b2^t)) + eps) + wd*p)
//
// where t is the 1-based step. `adamw_f32` is the host reference (and the CPU
// fallback for a parameter that is not device-resident). The device kernel in
// core/src/training/adamw.cu computes the same formula in float32; the CUDA
// path is the one training on a GPU uses, and the host reference is what its
// numerics are checked against.

#include <cstddef>
#include <cstdint>

namespace sonicboom::training {

struct AdamWConfig {
  float lr = 1e-3f;
  float beta1 = 0.9f;        // first-moment decay
  float beta2 = 0.999f;      // second-moment decay
  float eps = 1e-8f;         // added to sqrt(v_hat)
  float weight_decay = 0.0f; // decoupled, applied to the parameter itself
};

// One host-side update of `n` elements. `param`, `grad`, `m1` and `m2` each
// hold `n` floats; `param`, `m1` and `m2` are updated in place. `step` is the
// 1-based update index used for bias correction. The per-element arithmetic is
// carried in double and rounded once to float on store, matching the
// numpy/torch reference the check in tests/core/test_adamw.cpp uses.
void adamw_f32(float* param, const float* grad, float* m1, float* m2,
               std::size_t n, const AdamWConfig& cfg, std::uint64_t step);

} // namespace sonicboom::training
