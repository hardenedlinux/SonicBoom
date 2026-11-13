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
// and scales the scores by 1/sqrt(head_dim); Qwen3.6-MoE gates its attention
// output through the second half of the Q projection and scales the scores the
// same way, and its feed-forward is a mixture of experts (the recipe below).
struct Traits {
  bool scale_embedding = false;
  bool qk_norm = false;
  bool v_norm = false;
  bool qkv_bias = false;
  bool output_gate = false;
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
    case ModelArch::Qwen35Moe:
      t.qk_norm = true;
      t.output_gate = true;
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
  for (bool recurrent : cfg.recurrent_layers)
    r.states += recurrent ? 1 : 0;

  auto bias = [](bool has, WeightSlot slot) {
    return has ? slot : WeightSlot::None;
  };

  r.blocks.reserve(cfg.block_count);
  uint32_t next_state = 0;
  for (uint32_t l = 0; l < cfg.block_count; ++l) {
    const LayerConfig lc = cfg.layer_config(l);
    const BlockWeights& w = m.weights.blocks[l];

    Block block;
    const bool recurrent =
        l < cfg.recurrent_layers.size() && cfg.recurrent_layers[l];
    if (recurrent) {
      // Gated DeltaNet: the four projections the scan consumes, the output
      // projection, and the convolution/dt/a kernels the scan reads by slot.
      GatedDeltaNet gdn;
      gdn.heads_k = cfg.ssm_group_count;
      gdn.heads_v = cfg.ssm_time_step_rank;
      gdn.head_dim = cfg.ssm_state_size;
      gdn.conv_kernel = cfg.ssm_conv_kernel;
      // A projection stored as f32 in the file takes the f32 matvec; one
      // stored quantized keeps its format.
      auto projection = [](WeightSlot slot, const quant::QuantizedTensor& q,
                           std::span<const float> f, uint32_t in,
                           uint32_t out) {
        Projection p{slot, WeightSlot::None, q.type(), in, out};
        p.f32 = !f.empty();
        return p;
      };
      const uint32_t kd = gdn.key_dim();
      const uint32_t vd = gdn.value_dim();
      gdn.qkv = projection(WeightSlot::GdnQkv, w.gdn_qkv, w.gdn_qkv_f32, d,
                           2 * kd + vd);
      gdn.gate =
          projection(WeightSlot::GdnGate, w.gdn_gate, w.gdn_gate_f32, d, vd);
      gdn.beta = projection(WeightSlot::GdnBeta, w.gdn_beta, w.gdn_beta_f32, d,
                            gdn.heads_v);
      gdn.alpha = projection(WeightSlot::GdnAlpha, w.gdn_alpha,
                             w.gdn_alpha_f32, d, gdn.heads_v);
      gdn.out =
          projection(WeightSlot::GdnOut, w.gdn_out, w.gdn_out_f32, vd, d);
      gdn.state = next_state++;
      block.mixer = gdn;
    } else {
      SelfAttention at;
      at.heads_q = lc.n_heads_q;
      at.heads_kv = lc.n_heads_kv;
      at.head_dim = lc.head_dim;
      at.window = lc.sliding_window;
      const uint32_t qw = at.q_width();
      const uint32_t kvw = at.kv_width();
      at.q = {WeightSlot::AttnQ, bias(t.qkv_bias, WeightSlot::AttnQBias),
              w.attn_q.type(), d, t.output_gate ? 2 * qw : qw};
      at.k = {WeightSlot::AttnK, bias(t.qkv_bias, WeightSlot::AttnKBias),
              w.attn_k.type(), d, kvw};
      at.v = {WeightSlot::AttnV, bias(t.qkv_bias, WeightSlot::AttnVBias),
              w.attn_v.type(), d, kvw};
      at.out = {WeightSlot::AttnOutput, WeightSlot::None, w.attn_output.type(),
                qw, d};
      at.qk_norm = t.qk_norm;
      at.v_norm = t.v_norm;
      at.output_gate = t.output_gate;
      at.rotary_dims = cfg.rope_dimension_count;
      // The architecture's temperature times YaRN's magnitude term (1.0 for a
      // model without YaRN metadata).
      at.scale = double(t.score_scale) * m.weights.rope_attn_scale;
      at.cache = cfg.kv_donor_layer(l);
      at.owns_cache = cfg.has_kv(l);
      block.mixer = at;
      block.post_mixer_norm = t.post_mixer_norm;
    }

    FeedForward ffn;
    if (cfg.arch == ModelArch::Qwen35Moe) {
      MoeFfn moe;
      moe.router = {WeightSlot::ExpertRouter, WeightSlot::None,
                    w.expert_router.type(), d, cfg.n_expert};
      moe.experts = cfg.n_expert;
      moe.used = cfg.n_expert_used;
      const quant::QuantType et = w.expert_gate.empty()
                                      ? quant::QuantType{}
                                      : w.expert_gate[0].type();
      moe.routed.gate = {WeightSlot::ExpertGate, WeightSlot::None, et, d,
                         cfg.expert_ff};
      moe.routed.up = {WeightSlot::ExpertUp, WeightSlot::None, et, d,
                       cfg.expert_ff};
      moe.routed.down = {WeightSlot::ExpertDown, WeightSlot::None, et,
                         cfg.expert_ff, d};
      moe.weight_scale = double(cfg.expert_weights_scale);
      moe.activation = Activation::Silu;
      if (cfg.shared_expert_ff > 0) {
        Expert shared;
        shared.gate = {WeightSlot::SharedGate, WeightSlot::None,
                       w.shared_gate.type(), d, cfg.shared_expert_ff};
        shared.up = {WeightSlot::SharedUp, WeightSlot::None, w.shared_up.type(),
                     d, cfg.shared_expert_ff};
        shared.down = {WeightSlot::SharedDown, WeightSlot::None,
                       w.shared_down.type(), cfg.shared_expert_ff, d};
        moe.shared = shared;
        moe.shared_gate = WeightSlot::SharedGateIn;
      }
      ffn = moe;
    } else {
      GatedFfn dense;
      dense.up = {WeightSlot::FfnUp, WeightSlot::None, w.ffn_up.type(), d, ff};
      dense.gate = {WeightSlot::FfnGate, WeightSlot::None, w.ffn_gate.type(), d,
                    ff};
      dense.down = {WeightSlot::FfnDown, WeightSlot::None, w.ffn_down.type(), ff,
                    d};
      dense.activation = t.activation;
      ffn = dense;
    }

    block.ffn = std::move(ffn);
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
