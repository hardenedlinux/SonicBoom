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
  // No role has more than one f32 weight per block.
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
    default: return {};
  }
}

const quant::QuantizedTensor& weight_quant(const Gemma4Weights& weights,
                                           WeightKey key) noexcept {
  static const quant::QuantizedTensor kEmpty;
  // No role has more than one quantized weight per block.
  if (key.index != 0)
    return kEmpty;
  switch (key.slot) {
    case WeightSlot::TokenEmbd: return weights.token_embd;
    case WeightSlot::PerLayerTokenEmbd: return weights.per_layer_token_embd;
    case WeightSlot::Output: return weights.output;
    default: break;
  }
  const BlockWeights* block = block_of(weights, key);
  if (!block)
    return kEmpty;
  switch (key.slot) {
    case WeightSlot::AttnQ: return block->attn_q;
    case WeightSlot::AttnK: return block->attn_k;
    case WeightSlot::AttnV: return block->attn_v;
    case WeightSlot::AttnOutput: return block->attn_output;
    case WeightSlot::FfnGate: return block->ffn_gate;
    case WeightSlot::FfnUp: return block->ffn_up;
    case WeightSlot::FfnDown: return block->ffn_down;
    default: return kEmpty;
  }
}

} // namespace sonicboom::model
