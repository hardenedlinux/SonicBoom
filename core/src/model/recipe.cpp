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

// The recipes of the architectures the loader reads (model/recipe.h).

#include <sonicboom/model/recipe.h>

#include <cmath>

namespace sonicboom::model {

namespace {

// What an architecture does that the config's numbers do not say. Gemma 4
// norms Q, K and V per head, norms after each sub-layer as well as before,
// scales the embedding and each block's output, ties the head to the
// embedding and softcaps its logits; Qwen2 biases Q, K and V, gates with SiLU,
// and scales the scores by 1/sqrt(head_dim).
struct Traits {
  bool scale_embedding = false;
  bool qk_norm = false;
  bool v_norm = false;
  bool qkv_bias = false;
  bool post_mixer_norm = false;
  bool post_ffn_norm = false;
  Activation activation = Activation::GeluFp16;
  bool per_layer_input = false;
  bool output_scale = false;
  bool softcap = false;
  bool tied_head = false;
  float score_scale = 1.0f;
};

Traits traits_of(const Gemma4Config& cfg) {
  Traits t;
  switch (cfg.arch) {
    case ModelArch::Gemma4:
      t.scale_embedding = true;
      t.qk_norm = true;
      t.v_norm = true;
      t.post_mixer_norm = true;
      t.post_ffn_norm = true;
      t.per_layer_input = cfg.per_layer_input > 0;
      t.output_scale = true;
      t.softcap = cfg.final_logit_softcapping > 0.0f;
      t.tied_head = true;
      t.score_scale = 1.0f;
      break;
    case ModelArch::Qwen2:
      t.qkv_bias = true;
      t.activation = Activation::Silu;
      t.tied_head = false;
      t.score_scale = 1.0f / std::sqrt(float(cfg.head_dim_global));
      break;
  }
  return t;
}

} // namespace

Recipe recipe_for(const Gemma4Model& m) {
  const Gemma4Config& cfg = m.config;
  const Traits t = traits_of(cfg);
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;

  Recipe r;
  r.width = d;
  r.vocab = cfg.vocab_size;
  r.context_length = cfg.context_length;
  r.scale_embedding = t.scale_embedding;
  if (t.per_layer_input)
    r.per_layer_input = PerLayerInput{cfg.per_layer_input};
  r.caches = cfg.n_layer_kv();

  auto bias = [](bool has, WeightSlot slot) {
    return has ? slot : WeightSlot::None;
  };

  r.blocks.reserve(cfg.block_count);
  for (uint32_t l = 0; l < cfg.block_count; ++l) {
    const LayerConfig lc = cfg.layer_config(l);
    const BlockWeights& w = m.weights.blocks[l];

    SelfAttention at;
    at.heads_q = lc.n_heads_q;
    at.heads_kv = lc.n_heads_kv;
    at.head_dim = lc.head_dim;
    at.window = lc.sliding_window;
    const uint32_t qw = at.q_width();
    const uint32_t kvw = at.kv_width();
    at.q = {WeightSlot::AttnQ, bias(t.qkv_bias, WeightSlot::AttnQBias),
            w.attn_q.type(), d, qw};
    at.k = {WeightSlot::AttnK, bias(t.qkv_bias, WeightSlot::AttnKBias),
            w.attn_k.type(), d, kvw};
    at.v = {WeightSlot::AttnV, bias(t.qkv_bias, WeightSlot::AttnVBias),
            w.attn_v.type(), d, kvw};
    at.out = {WeightSlot::AttnOutput, WeightSlot::None, w.attn_output.type(),
              qw, d};
    at.qk_norm = t.qk_norm;
    at.v_norm = t.v_norm;
    // The architecture's temperature times YaRN's magnitude term (1.0 for a
    // model without YaRN metadata).
    at.scale = double(t.score_scale) * m.weights.rope_attn_scale;
    at.cache = cfg.kv_donor_layer(l);
    at.owns_cache = cfg.has_kv(l);

    GatedFfn ffn;
    ffn.up = {WeightSlot::FfnUp, WeightSlot::None, w.ffn_up.type(), d, ff};
    ffn.gate = {WeightSlot::FfnGate, WeightSlot::None, w.ffn_gate.type(), d, ff};
    ffn.down = {WeightSlot::FfnDown, WeightSlot::None, w.ffn_down.type(), ff, d};
    ffn.activation = t.activation;

    Block block;
    block.mixer = at;
    block.post_mixer_norm = t.post_mixer_norm;
    block.ffn = ffn;
    block.post_ffn_norm = t.post_ffn_norm;
    block.per_layer_gate = t.per_layer_input;
    if (t.output_scale && !w.layer_output_scale.empty())
      block.output_scale = double(w.layer_output_scale[0]);
    r.blocks.push_back(std::move(block));
  }

  r.head.weight = t.tied_head ? WeightSlot::TokenEmbd : WeightSlot::Output;
  r.head.type =
      t.tied_head ? m.weights.token_embd.type() : m.weights.output.type();
  r.head.softcap = t.softcap ? double(cfg.final_logit_softcapping) : 0.0;
  return r;
}

} // namespace sonicboom::model
