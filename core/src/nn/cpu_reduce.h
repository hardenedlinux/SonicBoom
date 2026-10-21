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

// The CPU reference kernels' reduction accumulators, for the ReduceMode the
// planner selected once for this host (planner/reduce_mode.h measures f64
// against df32 and applies the same cost-ratio rule the device uses). The f64
// branch is the historical code, byte-for-byte: each product is the f32
// rounding of the multiply, widened to double before the add, which is what
// makes the CPU path match llama.cpp's rms_norm and vec_dot.

#include <sonicboom/compiler/ReduceMode.h>

#include "../planner/reduce_mode.h"

#include <sonicboom/dtype.h>

#include <cstdint>
#include <span>

namespace sonicboom::nn {

// mean(x[i]^2) into the n divisor, as a float.
inline float cpu_mean_sq(std::span<const float> x) {
  const double n = double(x.size());
  switch (planner::cpu_reduce_mode()) {
    case compiler::ReduceMode::Df32: {
      compiler::Df32 s;
      for (const float v : x) s = compiler::df32_add_float(s, v * v);
      const float nf = float(x.size());
      return (s.hi / nf) + (s.lo / nf);
    }
    case compiler::ReduceMode::F32: {
      float s = 0.0f;
      for (const float v : x) s += v * v;
      return s / float(x.size());
    }
    default:
      break;
  }
  double sum_sq = 0.0;
  for (const float v : x) sum_sq += double(v * v);
  return float(sum_sq / n);
}

// sum(w[i] * x[i]).
inline float cpu_dot(std::span<const float> w, std::span<const float> x) {
  switch (planner::cpu_reduce_mode()) {
    case compiler::ReduceMode::Df32: {
      compiler::Df32 s;
      for (std::size_t i = 0; i < x.size(); ++i)
        s = compiler::df32_add_float(s, w[i] * x[i]);
      return s.hi + s.lo;
    }
    case compiler::ReduceMode::F32: {
      float s = 0.0f;
      for (std::size_t i = 0; i < x.size(); ++i) s += w[i] * x[i];
      return s;
    }
    default:
      break;
  }
  double acc = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) acc += double(w[i] * x[i]);
  return float(acc);
}

// sum(bf16_to_f32(w[i]) * x[i]): the weights are widened to f32 before the
// product, then the same accumulator policy applies.
inline float cpu_dot_bf16(std::span<const uint16_t> w,
                          std::span<const float> x) {
  switch (planner::cpu_reduce_mode()) {
    case compiler::ReduceMode::Df32: {
      compiler::Df32 s;
      for (std::size_t i = 0; i < x.size(); ++i)
        s = compiler::df32_add_float(s, bf16_to_f32(w[i]) * x[i]);
      return s.hi + s.lo;
    }
    case compiler::ReduceMode::F32: {
      float s = 0.0f;
      for (std::size_t i = 0; i < x.size(); ++i)
        s += bf16_to_f32(w[i]) * x[i];
      return s;
    }
    default:
      break;
  }
  double acc = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i)
    acc += double(bf16_to_f32(w[i]) * x[i]);
  return float(acc);
}

} // namespace sonicboom::nn
