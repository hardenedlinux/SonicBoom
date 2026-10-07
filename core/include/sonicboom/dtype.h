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

#include <bit>
#include <cstdint>

// Scalar floating-point type conversions shared across Layer 2. bf16 is not a
// first-class SonicBoom dtype in v0; it appears only as the storage type of the
// single bf16 tensor in the target model (`per_layer_model_proj.weight`), whose
// matvec the nn layer computes. Both the model layer (which rounds the f32
// activation to bf16 before the dot) and the nn kernel (which widens each raw
// weight half back to f32) need the same conversion, so they live here rather
// than in model/ or nn/.

namespace sonicboom {

// IEEE bf16 (1 sign / 8 exp / 7 mantissa) -> f32. bf16 shares f32's exponent
// and is a truncated mantissa, so the conversion is a pure left-shift.
inline float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(uint32_t(h) << 16); }

// f32 -> bf16, round-to-nearest-even (the same rounding ggml's
// ggml_compute_fp32_to_bf16 applies when it packs an f32 tensor into bf16 for a
// bf16 vec_dot). NaN is quieted to match ggml. The bf16 matvec must round its
// f32 input through bf16 — llama.cpp packs src1 (the activation) to the weight's
// vec_dot_type (bf16) before the dot product, so a full-f32 input would not
// reproduce the baseline's rounding.
inline uint16_t f32_to_bf16(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  if ((u & 0x7fffffff) > 0x7f800000) return uint16_t((u >> 16) | 64);  // NaN
  return uint16_t((u + (0x7fff + ((u >> 16) & 1))) >> 16);
}

} // namespace sonicboom
