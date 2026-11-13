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

// Mixture-of-experts feed-forward kernels (float32 activations, quantized or
// float32 weights). The routing and the routed SwiGLU sum are one position's
// work: the route picks `used` of `experts` and renormalises their weights,
// the experts node applies exactly those and accumulates.
//
// Two decisions that change bits are part of the contract, not the caller's
// choice:
//   - a top-k tie takes the lower expert index;
//   - the routed sum accumulates in ascending expert index, so the result does
//     not depend on the order the selection found the experts in.

#include <sonicboom/quant/quantized_tensor.h>

#include <cstdint>
#include <span>

namespace sonicboom::nn {

// The activation between the routed gate and up projections.
enum class MoeActivation : uint8_t {
  Silu,
  GeluFp16,  // the same fp16-table GELU the dense gated FFN uses
};

// The routed experts' projections, one view per expert (index = expert).
struct MoeExpertWeights {
  std::span<const quant::QuantizedTensor> gate;  // [d, ff_e] each
  std::span<const quant::QuantizedTensor> up;    // [d, ff_e] each
  std::span<const quant::QuantizedTensor> down;  // [ff_e, d] each
};

// y = sum over the selected experts of w_e * down_e(act(gate_e x) * (up_e x)).
//
// `ids` holds the selected expert indices and `weights` their weights, one per
// id, in ascending expert index (what moe_route returns). `x` and `y` are one
// position's activation; the routed experts' projections are evaluated with the
// same CPU quantized matvec the dense projections use (q8_K activation).
bool moe_experts(std::span<const float> x, std::span<const int64_t> ids,
                 std::span<const float> weights,
                 const MoeExpertWeights& experts, MoeActivation activation,
                 std::span<float> y);

// Router: logits = W x, probs = softmax over all experts, the top `used`
// experts by probability (ties to the lower index), and
// w_e = probs[e] / max(sum of the selected probs, 2^-14) * weight_scale.
// The weights are computed in ascending expert index and `ids` is written in
// that order. The router weight is the float32 matrix when `router_f32` is not
// empty ([d, experts]), the quantized one otherwise (`router`).
bool moe_route(std::span<const float> x, const quant::QuantizedTensor* router,
               std::span<const float> router_f32, uint32_t experts,
               uint32_t used, float weight_scale, std::span<int64_t> ids,
               std::span<float> weights);

} // namespace sonicboom::nn
