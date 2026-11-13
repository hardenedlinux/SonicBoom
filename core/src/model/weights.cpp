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

// Where a weight's address leads (model/weights.h).

#include <sonicboom/model/weights.h>

namespace sonicboom::model {

namespace {

// The block a key addresses; null when the layer is out of range.
const BlockWeights* block_of(const Gemma4Weights& weights,
                             WeightKey key) noexcept {
  return key.layer < weights.blocks.size() ? &weights.blocks[key.layer]
                                           : nullptr;
}

} // namespace

std::span<const float> weight_f32(const Gemma4Weights& weights,
                                  WeightKey key) noexcept {
  // No f32 role has more than one weight per block (the experts are all
  // quantized).
  if (key.index != 0)
    return {};
  switch (key.slot) {
    case WeightSlot::PerLayerProjNorm: return weights.per_layer_proj_norm;
    case WeightSlot::OutputNorm: return weights.output_norm;
    case WeightSlot::RopeFreqs: return weights.rope_freqs;
    default: break;
  }
  const BlockWeights* block = block_of(weights, key);
  if (!block)
    return {};
  switch (key.slot) {
    case WeightSlot::AttnNorm: return block->attn_norm;
    case WeightSlot::FfnNorm: return block->ffn_norm;
    case WeightSlot::AttnQNorm: return block->attn_q_norm;
    case WeightSlot::AttnKNorm: return block->attn_k_norm;
    case WeightSlot::PostAttentionNorm: return block->post_attention_norm;
    case WeightSlot::PostFfwNorm: return block->post_ffw_norm;
    case WeightSlot::PostNorm: return block->post_norm;
    case WeightSlot::AttnQBias: return block->attn_q_bias;
    case WeightSlot::AttnKBias: return block->attn_k_bias;
    case WeightSlot::AttnVBias: return block->attn_v_bias;
    case WeightSlot::AttnOutputBias: return block->attn_output_bias;
    case WeightSlot::InpGate: return block->inp_gate;
    case WeightSlot::Proj: return block->proj;
    case WeightSlot::LayerOutputScale: return block->layer_output_scale;
    case WeightSlot::ExpertRouter: return block->expert_router_f32;
    case WeightSlot::SharedGateIn: return block->shared_gate_in;
    default: return {};
  }
}

const quant::QuantizedTensor& weight_quant(const Gemma4Weights& weights,
                                           WeightKey key) noexcept {
  static const quant::QuantizedTensor kEmpty;
  // The three expert roles hold one weight per expert, addressed by `index`;
  // every other role holds at most one weight per block (index 0).
  const auto expert = [&](const std::vector<quant::QuantizedTensor>& v)
      -> const quant::QuantizedTensor& {
    return key.index < v.size() ? v[key.index] : kEmpty;
  };
  switch (key.slot) {
    case WeightSlot::TokenEmbd:
      return key.index == 0 ? weights.token_embd : kEmpty;
    case WeightSlot::PerLayerTokenEmbd:
      return key.index == 0 ? weights.per_layer_token_embd : kEmpty;
    case WeightSlot::Output:
      return key.index == 0 ? weights.output : kEmpty;
    default: break;
  }
  const BlockWeights* block = block_of(weights, key);
  if (!block)
    return kEmpty;
  const bool is_expert = key.slot == WeightSlot::ExpertGate ||
                         key.slot == WeightSlot::ExpertUp ||
                         key.slot == WeightSlot::ExpertDown;
  if (key.index != 0 && !is_expert)
    return kEmpty;
  switch (key.slot) {
    case WeightSlot::AttnQ: return block->attn_q;
    case WeightSlot::AttnK: return block->attn_k;
    case WeightSlot::AttnV: return block->attn_v;
    case WeightSlot::AttnOutput: return block->attn_output;
    case WeightSlot::FfnGate: return block->ffn_gate;
    case WeightSlot::FfnUp: return block->ffn_up;
    case WeightSlot::FfnDown: return block->ffn_down;
    case WeightSlot::ExpertRouter:
      return key.index == 0 ? block->expert_router : kEmpty;
    case WeightSlot::ExpertGate: return expert(block->expert_gate);
    case WeightSlot::ExpertUp: return expert(block->expert_up);
    case WeightSlot::ExpertDown: return expert(block->expert_down);
    case WeightSlot::SharedGate:
      return key.index == 0 ? block->shared_gate : kEmpty;
    case WeightSlot::SharedUp:
      return key.index == 0 ? block->shared_up : kEmpty;
    case WeightSlot::SharedDown:
      return key.index == 0 ? block->shared_down : kEmpty;
    default: return kEmpty;
  }
}

std::expected<std::vector<quant::QuantizedTensor>, std::string>
expert_views(uint32_t ggml_type, const std::vector<uint64_t>& dims,
             std::span<const std::byte> data, uint64_t n_expert,
             const std::vector<uint64_t>& per_expert_dims) {
  const auto type = quant::quant_type_from_ggml(ggml_type);
  if (!type)
    return std::unexpected(
        std::string("the expert tensor is not a supported quantized type"));
  if (n_expert == 0 || per_expert_dims.empty())
    return std::unexpected(std::string("no expert slices"));
  std::vector<uint64_t> want(per_expert_dims);
  want.push_back(n_expert);
  if (dims != want)
    return std::unexpected(
        std::string("the tensor's shape does not end in the expert count"));
  const quant::QuantTypeInfo* info = quant::quant_type_info(*type);
  uint64_t per_elements = 1;
  for (uint64_t v : per_expert_dims) per_elements *= v;
  if (per_elements % info->block_size != 0)
    return std::unexpected(
        std::string("one expert's slice is not a whole number of " +
                    std::to_string(info->block_size) + "-element blocks"));
  const uint64_t per_bytes =
      per_elements / info->block_size * info->bytes_per_block;
  if (data.size() != per_bytes * n_expert)
    return std::unexpected(
        std::string("the tensor's bytes are not exactly ") +
        std::to_string(n_expert) + " expert slices of " +
        std::to_string(per_bytes) + " bytes");
  std::vector<quant::QuantizedTensor> out;
  out.reserve(n_expert);
  for (uint64_t e = 0; e < n_expert; ++e) {
    quant::QuantizedTensor view(*type, per_expert_dims,
                                data.subspan(e * per_bytes, per_bytes));
    if (!view.valid())
      return std::unexpected(std::string("expert ") + std::to_string(e) +
                             " is not a valid view");
    out.push_back(std::move(view));
  }
  return out;
}

std::expected<ExpertGateUpViews, std::string>
expert_gate_up_views(uint32_t ggml_type, const std::vector<uint64_t>& dims,
                     std::span<const std::byte> data, uint64_t n_expert,
                     uint64_t n_embd, uint64_t ff) {
  const auto type = quant::quant_type_from_ggml(ggml_type);
  if (!type)
    return std::unexpected(
        std::string("the expert tensor is not a supported quantized type"));
  if (n_expert == 0 || n_embd == 0 || ff == 0)
    return std::unexpected(std::string("empty expert slices"));
  const std::vector<uint64_t> want{n_embd, ff * 2, n_expert};
  if (dims != want)
    return std::unexpected(
        std::string("the merged gate+up tensor's shape is not "
                    "[n_embd, 2*ff, n_expert]"));
  const quant::QuantTypeInfo* info = quant::quant_type_info(*type);
  const uint64_t half_elements = n_embd * ff;
  if (half_elements % info->block_size != 0)
    return std::unexpected(
        std::string("one expert's gate/up half is not a whole number of " +
                    std::to_string(info->block_size) + "-element blocks"));
  const uint64_t half_bytes =
      half_elements / info->block_size * info->bytes_per_block;
  if (data.size() != 2 * half_bytes * n_expert)
    return std::unexpected(
        std::string("the merged tensor's bytes are not exactly ") +
        std::to_string(n_expert) + " gate+up expert slices");
  const std::vector<uint64_t> half_dims{n_embd, ff};
  ExpertGateUpViews out;
  out.gate.reserve(n_expert);
  out.up.reserve(n_expert);
  for (uint64_t e = 0; e < n_expert; ++e) {
    const uint64_t base = e * 2 * half_bytes;
    quant::QuantizedTensor g(*type, half_dims, data.subspan(base, half_bytes));
    quant::QuantizedTensor u(*type, half_dims,
                             data.subspan(base + half_bytes, half_bytes));
    if (!g.valid() || !u.valid())
      return std::unexpected(std::string("expert ") + std::to_string(e) +
                             " has no valid gate/up view");
    out.gate.push_back(std::move(g));
    out.up.push_back(std::move(u));
  }
  return out;
}

} // namespace sonicboom::model
