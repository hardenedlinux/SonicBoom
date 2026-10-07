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

#include <sonicboom/nn/activation.h>

#include <sonicboom/thread_pool.h>

#include <cmath>
#include <stdfloat>

namespace sonicboom::nn {

namespace {

// ggml's ggml_gelu_f32 (ggml/src/ggml-cpu/vec.h), reproduced verbatim in both the
// constants and the operation order so the tanh argument agrees bit-for-bit with
// llama.cpp's kernel:
//   0.5f * x * (1 + tanhf(SQRT_2_OVER_PI * x * (1 + GELU_COEF_A * x * x))).
float gelu_tanh_f32(float x) {
  constexpr float sqrt_2_over_pi =
      0.79788456080286535587989211986876f;  // sqrt(2/pi)
  constexpr float gelu_coef_a = 0.044715f;
  return 0.5f * x *
         (1.0f + std::tanh(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
}

}  // namespace

bool silu(std::span<const float> x, std::span<float> y) {
  if (y.size() < x.size()) return false;
  for (size_t i = 0; i < x.size(); ++i) {
    // x / (1 + exp(-x)): exp(-x) stays in (0,1] for x >= 0; for x < 0 it grows
    // but remains finite (and the quotient -> 0) down to ~ -88, beyond which
    // exp(-x) overflows to inf and x/inf -> ±0, which is the correct float value.
    y[i] = x[i] / (1.0f + std::exp(-x[i]));
  }
  return true;
}

bool gelu(std::span<const float> x, std::span<float> y) {
  if (y.size() < x.size()) return false;
  for (size_t i = 0; i < x.size(); ++i) y[i] = gelu_tanh_f32(x[i]);
  return true;
}

bool gelu_fp16(std::span<const float> x, std::span<float> y) {
  if (y.size() < x.size()) return false;
  // Each element depends only on x[i], so the loop is split across the pool
  // deterministically (identical to the serial path). The per-element tanh
  // (gelu_tanh_f32) is heavy enough that the 10k-element ffn-gate case needs the
  // pool, despite the handshake cost.
  ThreadPool::instance().parallel_for(0, x.size(), [&](uint64_t b, uint64_t e) {
    for (uint64_t i = b; i < e; ++i) {
      const float v = x[i];
      // ggml_vec_gelu_f32 under GGML_GELU_FP16: clamp outside [-10, 10], otherwise
      // look up ggml_table_gelu_f16[fp16(v)] = fp16(gelu_tanh_f32(fp16(v))).
      if (v <= -10.0f) {
        y[i] = 0.0f;
      } else if (v >= 10.0f) {
        y[i] = v;
      } else {
        const float vf = float(std::float16_t(v));  // input -> fp16 (table index)
        y[i] = float(std::float16_t(gelu_tanh_f32(vf)));  // result -> fp16
      }
    }
  });
  return true;
}

} // namespace sonicboom::nn
