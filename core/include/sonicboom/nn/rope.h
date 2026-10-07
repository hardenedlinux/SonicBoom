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

// Rotary position embedding (RoPE) for the native transformer path (float32).

namespace sonicboom::nn {

// GPT-NeoX-style rotary embedding, applied in place to a single query or key
// vector of `head_dim` floats at absolute position `pos`.
//
// The rotation pairs dimension i with dimension i + head_dim/2, using frequency
// index i for both. This is the layout Gemma uses (llama.cpp LLAMA_ROPE_TYPE_NEOX),
// not the Llama/GPT-J interleaved (2i, 2i+1) layout.
//
//   theta_i = freq_scale * pos * theta_base^(-2*i/head_dim) / freq_factors[i]
//   (x[i], x[i+half]) -> (x[i]*cos - x[i+half]*sin, x[i]*sin + x[i+half]*cos)
//
// `theta_base` is the model's rope_theta (Gemma: 10000). `freq_scale` is the
// model's rope_freq_scale (Gemma: 1.0), applied as a multiplicative factor on
// the frequency.
//
// `freq_factors` is an optional per-pair divisor (llama.cpp `rope_freqs`
// "proportional RoPE"); empty means 1.0 for every pair. Gemma 4 uses it only on
// the global (full-attention) layers, where `rope_freqs = [64×1.0, 192×1e30]`
// divides the angle of the tail pairs to ~0, disabling their rotation (partial
// RoPE). A factor of 1.0 leaves a pair's rotation unchanged.
//
// Returns false (writing nothing) when x.size() is 0 or odd, or freq_factors is
// non-empty and shorter than x.size()/2.
bool rope_neox(std::span<float> x, uint64_t pos, float theta_base,
               float freq_scale = 1.0f,
               std::span<const float> freq_factors = {});

// Same NEOX rotation over a concatenated multi-head buffer: `x` is `heads`
// heads of `head_dim` floats laid back-to-back, and each head is rotated in
// place by rope_neox. `freq_factors` (if non-empty, length head_dim/2) is shared
// across heads — indexed by the intra-head offset, not the flat index. This is
// the decode block's per-head RoPE hoisted to one kernel, split across the
// thread pool (trig-heavy RoPE is big enough to amortise the handshake, unlike
// the trivial elementwise ops).
//
// Returns false (writing nothing) when head_dim is 0 or odd, x.size() is 0 or
// not a multiple of head_dim, or freq_factors is non-empty and shorter than
// head_dim/2.
bool rope_neox_heads(std::span<float> x, uint64_t head_dim, uint64_t pos,
                     float theta_base, float freq_scale,
                     std::span<const float> freq_factors);

} // namespace sonicboom::nn
