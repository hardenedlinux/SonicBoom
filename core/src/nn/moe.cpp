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

#include <sonicboom/nn/moe.h>

#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace sonicboom::nn {

namespace {

// The lower bound on the routed weight denominator (2^-14), so a router whose
// selected probabilities underflow cannot divide by zero.
constexpr float kMoeWeightSumFloor = 6.103515625e-5f;

} // namespace

bool moe_route(std::span<const float> x, const quant::QuantizedTensor *router,
               std::span<const float> router_f32, uint32_t experts,
               uint32_t used, float weight_scale, std::span<int64_t> ids,
               std::span<float> weights) {
  if (experts == 0 || used == 0 || used > experts || ids.size() < used ||
      weights.size() < used || x.empty())
    return false;

  std::vector<float> logits(experts, 0.0f);
  const uint64_t d = x.size();
  if (!router_f32.empty()) {
    if (router_f32.size() < uint64_t(experts) * d) return false;
    if (!matvec_f32(router_f32.first(uint64_t(experts) * d), d, x, logits))
      return false;
  } else if (router && router->valid()) {
    if (!quant::matvec(*router, x, logits, quant::MatmulBackend::CpuQ8K))
      return false;
  } else {
    return false;
  }

  // Stable softmax over all experts, accumulated in ascending index.
  float mx = logits[0];
  for (float l : logits) mx = std::max(mx, l);
  float sum = 0.0f;
  std::vector<float> probs(experts, 0.0f);
  for (uint32_t e = 0; e < experts; ++e) {
    probs[e] = std::exp(logits[e] - mx);
    sum += probs[e];
  }
  if (!(sum > 0.0f)) return false;
  const float inv = 1.0f / sum;
  for (uint32_t e = 0; e < experts; ++e) probs[e] *= inv;

  // Top-k: the larger probability, and on equal probabilities the lower index
  // (ascending scan with a strict comparison keeps the first of a tie).
  std::vector<bool> taken(experts, false);
  std::vector<uint32_t> selected;
  selected.reserve(used);
  for (uint32_t k = 0; k < used; ++k) {
    uint32_t best = experts;  // none yet
    for (uint32_t e = 0; e < experts; ++e) {
      if (taken[e]) continue;
      if (best == experts || probs[e] > probs[best]) best = e;
    }
    if (best == experts) return false;
    taken[best] = true;
    selected.push_back(best);
  }

  // Ascending expert index: the ids are sorted and the weight denominator is
  // summed in that order, so the result does not depend on selection order.
  std::sort(selected.begin(), selected.end());
  float denom = 0.0f;
  for (uint32_t e : selected) denom += probs[e];
  denom = std::max(denom, kMoeWeightSumFloor);
  for (uint32_t k = 0; k < used; ++k) {
    ids[k] = int64_t(selected[k]);
    weights[k] = probs[selected[k]] / denom * weight_scale;
  }
  return true;
}

bool moe_experts(std::span<const float> x, std::span<const int64_t> ids,
                 std::span<const float> weights,
                 const MoeExpertWeights &experts, MoeActivation activation,
                 std::span<float> y) {
  if (ids.size() != weights.size() || x.empty() || y.size() < x.size())
    return false;
  if (experts.gate.size() != experts.up.size() ||
      experts.gate.size() != experts.down.size())
    return false;

  const uint64_t d = x.size();
  const uint64_t e_count = experts.gate.size();

  std::vector<float> gate, up, hidden;
  std::vector<float> down(d, 0.0f);
  for (size_t k = 0; k < ids.size(); ++k) {
    const int64_t e = ids[k];
    if (e < 0 || uint64_t(e) >= e_count) return false;
    const quant::QuantizedTensor &g = experts.gate[size_t(e)];
    const quant::QuantizedTensor &u = experts.up[size_t(e)];
    const quant::QuantizedTensor &dn = experts.down[size_t(e)];
    if (!g.valid() || !u.valid() || !dn.valid()) return false;
    const uint64_t ff_dim = g.dims().size() > 1 ? g.dims()[1] : 0;
    if (ff_dim == 0) return false;
    gate.assign(ff_dim, 0.0f);
    up.assign(ff_dim, 0.0f);
    hidden.assign(ff_dim, 0.0f);
    if (!quant::matvec(g, x, gate, quant::MatmulBackend::CpuQ8K)) return false;
    if (!quant::matvec(u, x, up, quant::MatmulBackend::CpuQ8K)) return false;
    const bool act_ok = activation == MoeActivation::Silu
                            ? silu(gate, gate)
                            : gelu_fp16(gate, gate);
    if (!act_ok) return false;
    for (uint64_t i = 0; i < ff_dim; ++i) hidden[i] = gate[i] * up[i];
    // The weight is applied to the expert's output, then summed ascending.
    if (!quant::matvec(dn, hidden, down, quant::MatmulBackend::CpuQ8K))
      return false;
    const float w = weights[k];
    for (uint64_t i = 0; i < d; ++i) y[i] += w * down[i];
  }
  return true;
}

} // namespace sonicboom::nn
