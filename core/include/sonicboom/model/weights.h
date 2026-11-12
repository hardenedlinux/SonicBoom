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
#include <vector>

#include <sonicboom/dtype.h>
#include <sonicboom/quant/quantized_tensor.h>

// Bound Gemma 4 weights (Phase 5A). A `Gemma4Weights` is a set of views over
// the GGUF reader's backing buffer: quantized tensors keep their packed bytes
// (QuantizedTensor borrows the byte span), f32 tensors are exposed as
// std::span<const float>, and the single bf16 tensor is exposed as raw halves.
//
// Alignment note: GGUF aligns every tensor's data offset to `general.alignment`
// (32 in the target model), so reinterpret_cast-ing the byte span to float (4)
// or uint16_t (2) is well-defined. The reader owns the buffer and outlives the
// model, so all spans are borrowed, never owned.

namespace sonicboom::model {

// bf16 <-> f32 conversions are shared with the nn kernels (whose bf16 matvec
// widens the same raw halves) and re-exported here for the model layer.
using sonicboom::bf16_to_f32;
using sonicboom::f32_to_bf16;

// The role a weight plays. A graph node that reads a weight carries its role as
// the `slot` attribute (and its block as `layer`); weight_f32 and weight_quant
// below are how a role is found in the loaded weights. Block roles
// (AttnNorm .. LayerOutputScale, and the biases) are one per block; the rest
// are one per model. A value never changes once assigned: it is written into
// plans.
enum class WeightSlot : uint8_t {
  None = 0,
  TokenEmbd,
  PerLayerTokenEmbd,
  PerLayerModelProj,
  PerLayerProjNorm,
  OutputNorm,
  RopeFreqs,
  AttnNorm,
  FfnNorm,
  AttnQNorm,
  AttnKNorm,
  PostAttentionNorm,
  PostFfwNorm,
  PostNorm,
  AttnQ,
  AttnK,
  AttnV,
  AttnOutput,
  FfnGate,
  FfnUp,
  FfnDown,
  InpGate,
  Proj,
  LayerOutputScale,
  // Optional linear-layer biases, f32 [width]. An architecture either carries
  // them (Qwen2's Q/K/V) or does not (Gemma 4); the loader binds what the file
  // has and the emitter emits an AddBias node only for a bound span, so an
  // absent bias costs nothing. Added after the original slots so a slot's
  // numeric value never moves.
  AttnQBias,
  AttnKBias,
  AttnVBias,
  AttnOutputBias,
  // The untied output projection (lm head); empty when the model ties its
  // embedding and output weights, in which case the emitter reuses TokenEmbd.
  Output,
};

// One transformer block's weights (block_count of these, layers 0..41).
//
// All 17 per-block tensor groups are bound. The 7 added in Phase 5C-1
// (attn_q_norm/attn_k_norm/post_attention_norm/post_ffw_norm/post_norm/
// inp_gate/proj) are bound + shape/type-validated but NOT yet consumed by
// run_block; their exact arithmetic (q/k norm convention, per-layer gating,
// post-norm residual order) is unresolved pending a baseline.
// Bound-but-not-consumed spans are empty() only
// when the loader failed to bind them, which the loader already reports.
struct BlockWeights {
  std::span<const float> attn_norm;   // [embedding_length]
  std::span<const float> ffn_norm;    // [embedding_length]
  std::span<const float> attn_q_norm; // [head_dim] (256 SWA / 512 global); scalar broadcast in this GGUF
  std::span<const float> attn_k_norm; // [head_dim]; scalar broadcast
  std::span<const float> post_attention_norm; // [embedding_length]
  std::span<const float> post_ffw_norm;       // [embedding_length]
  std::span<const float> post_norm;           // [embedding_length]
  quant::QuantizedTensor attn_q;      // [embedding_length, n_heads_q*head_dim]
  quant::QuantizedTensor attn_k;      // [embedding_length, n_heads_kv*head_dim]
  quant::QuantizedTensor attn_v;      // [embedding_length, n_heads_kv*head_dim]
  quant::QuantizedTensor attn_output; // [n_heads_q*head_dim, embedding_length]
  // Optional projection biases (empty when the architecture has none, as Gemma
  // 4 does). Qwen2 carries the Q/K/V ones.
  std::span<const float> attn_q_bias;      // [n_heads_q*head_dim]
  std::span<const float> attn_k_bias;      // [n_heads_kv*head_dim]
  std::span<const float> attn_v_bias;      // [n_heads_kv*head_dim]
  std::span<const float> attn_output_bias; // [embedding_length]
  quant::QuantizedTensor ffn_gate;    // [embedding_length, ff]
  quant::QuantizedTensor ffn_up;      // [embedding_length, ff]
  quant::QuantizedTensor ffn_down;    // [ff, embedding_length]
  std::span<const float> inp_gate;    // [embedding_length, per_layer_input] (f32)
  std::span<const float> proj;        // [per_layer_input, embedding_length] (f32)
  std::span<const float> layer_output_scale;  // [1]
};

struct Gemma4Weights {
  quant::QuantizedTensor token_embd;            // [embedding_length, vocab]
  quant::QuantizedTensor per_layer_token_embd;  // [block_count*per_layer_input, vocab]
  std::span<const float> per_layer_proj_norm;   // [per_layer_input]
  std::span<const uint16_t> per_layer_model_proj;  // bf16 [embedding_length, block_count*per_layer_input]
  std::span<const float> output_norm;           // [embedding_length]
  std::span<const float> rope_freqs;            // [256] (precomputed RoPE table)
  // Owned storage behind `rope_freqs` when the table is synthesized from
  // `{arch}.rope.scaling.*` metadata (generic linear/YaRN scaling) instead of
  // read from a `rope_freqs.weight` tensor. Metadata keys cannot borrow the
  // reader's tensor buffer the way `rope_freqs.weight` can, so the model owns
  // these floats; `rope_freqs` points into this vector and stays valid across
  // moves (a vector move transfers its heap buffer).
  std::vector<float> rope_freqs_table;
  // YaRN magnitude scaling, expressed as a multiplier on the ATTENTION SCORES.
  // YaRN rotates q and k by cos/sin multiplied by m = 0.1*ln(factor)+1 (times
  // the GGUF attn_factor), so the q·k dot product is m^2 times the unscaled
  // one. With full rope every component of q/k is rotated, which makes scaling
  // the scores by m^2 exactly equivalent; folding it into the emitter's
  // attention temperature avoids a per-pair magnitude argument in every rope
  // kernel. 1.0 = no magnitude scaling (linear scaling, or no scaling at all).
  float rope_attn_scale = 1.0f;

  // The lm head. Empty when the model ties its output to the token embedding
  // (Gemma 4), in which case the emitter falls back to token_embd.
  quant::QuantizedTensor output;                // [embedding_length, vocab]

  std::vector<BlockWeights> blocks;  // size == block_count

  bool empty() const noexcept { return blocks.empty(); }
};

// The address of one weight: its role, the block it belongs to, and — for a
// role a block holds several weights of — which of them. Every role today has
// one weight per block (or per model), so `index` is 0; it is part of the
// address so that a layer kind with many weights of one role (the experts of a
// mixture) is addressed the same way as every other weight, by one lookup,
// instead of by a second mechanism beside it.
struct WeightKey {
  WeightSlot slot = WeightSlot::None;
  uint32_t layer = 0;  // ignored by a per-model role
  uint32_t index = 0;

  constexpr bool operator==(const WeightKey&) const noexcept = default;
};

// The f32 weight at `key`; empty when the role is not an f32 role, the model
// has no such weight, or the address is out of range.
std::span<const float> weight_f32(const Gemma4Weights& weights,
                                  WeightKey key) noexcept;

// The quantized weight at `key`; a tensor of no elements under the same
// conditions.
const quant::QuantizedTensor& weight_quant(const Gemma4Weights& weights,
                                           WeightKey key) noexcept;

} // namespace sonicboom::model
