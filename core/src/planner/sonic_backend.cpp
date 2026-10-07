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

#include <sonicboom/planner/sonic_backend.h>

#include <sonicboom/dtype.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/attention.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/embedding.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sonicboom::planner {

namespace {

using sonicboom::bf16_to_f32;
using sonicboom::f32_to_bf16;

RuntimeError fail(std::string msg) {
  return RuntimeError(RuntimeErrorCode::BackendFailure, std::move(msg),
                      Phase::Execution);
}

// Decode the little-endian 64-bit scalar graph inputs (token id / position).
uint64_t read_i64(const sx::Bytes& b) {
  uint64_t v = 0;
  std::memcpy(&v, b.data(), sizeof(v));
  return v;
}

// Interpret a raw byte buffer as a float32 span (activations).
std::span<const float> f32(const sx::Bytes& b) {
  return {reinterpret_cast<const float*>(b.data()), b.size() / sizeof(float)};
}

// Interpret a raw byte buffer as a mutable float32 span (outputs).
std::span<float> f32_mut(sx::Bytes& b) {
  return {reinterpret_cast<float*>(b.data()), b.size() / sizeof(float)};
}

// Read an Int / Float attribute by name; false when absent or the wrong kind.
bool attr_i(const GraphNodeDesc& n, std::string_view name, int64_t& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::Int) {
      out = std::get<int64_t>(a.value);
      return true;
    }
  return false;
}

bool attr_f(const GraphNodeDesc& n, std::string_view name, double& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::Float) {
      out = std::get<double>(a.value);
      return true;
    }
  return false;
}

} // namespace

SonicBackend::SonicBackend(const model::Gemma4Model& model, const Graph& graph,
                           uint64_t n_ctx)
    : model_(model), graph_(graph) {
  ensure_caches(n_ctx);
}

void SonicBackend::ensure_caches(uint64_t n_ctx) {
  const uint32_t n_kv = model_.config.n_layer_kv();
  caches_.clear();
  caches_.resize(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l) {
    const model::LayerConfig& lc = model_.config.layer_config(l);
    KvCache& c = caches_[l];
    c.n_heads_kv = lc.n_heads_kv;
    c.head_dim = lc.head_dim;
    // SWA layers are a fixed-size ring (sliding_window slots); global layers are
    // the full decode length (n_ctx slots, no wrap).
    c.n_slots = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx;
    const size_t total = (size_t)c.n_slots * c.n_heads_kv * c.head_dim;
    c.k.assign(total, 0.0f);
    c.v.assign(total, 0.0f);
  }
}

std::expected<std::vector<TensorValue>, RuntimeError> SonicBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& in) {
  const auto* compute = task.as_compute();
  if (!compute || compute->graph_nodes.empty())
    return std::unexpected(fail("Sonic task has no compute node"));
  const auto* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(fail("Sonic task references an unknown graph node"));
  if (node->outputs.empty())
    return std::unexpected(fail("Sonic node has no output"));
  const auto* out_td = graph_.find_tensor(node->outputs.front());
  if (!out_td)
    return std::unexpected(fail("Sonic node output tensor is unknown"));

  // The CPU SonicBackend runs on host float32 bytes; device-resident values
  // never reach it (the GPU SonicBackend handles those).
  std::vector<sx::Bytes> inputs;
  inputs.reserve(in.size());
  for (const auto& v : in) {
    if (v.on_device)
      return std::unexpected(fail("CPU Sonic backend received a device input"));
    inputs.push_back(v.host);
  }

  const auto& cfg = model_.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t per = cfg.per_layer_input;
  const float eps = cfg.rms_epsilon;

  int64_t layer = -1;
  attr_i(*node, "layer", layer);
  int64_t slot_i = 0;
  attr_i(*node, "slot", slot_i);
  const model::WeightSlot slot = static_cast<model::WeightSlot>(slot_i);
  double scale_d = 0.0;
  attr_f(*node, "scale", scale_d);

  const model::BlockWeights& block = model_.weights.blocks[layer >= 0 ? uint32_t(layer) : 0];

  // f32 weight accessor (block slots index blocks[layer], the rest weights.*).
  auto fweight = [&](model::WeightSlot s) -> std::span<const float> {
    switch (s) {
      case model::WeightSlot::AttnNorm: return block.attn_norm;
      case model::WeightSlot::FfnNorm: return block.ffn_norm;
      case model::WeightSlot::AttnQNorm: return block.attn_q_norm;
      case model::WeightSlot::AttnKNorm: return block.attn_k_norm;
      case model::WeightSlot::PostAttentionNorm: return block.post_attention_norm;
      case model::WeightSlot::PostFfwNorm: return block.post_ffw_norm;
      case model::WeightSlot::PostNorm: return block.post_norm;
      case model::WeightSlot::InpGate: return block.inp_gate;
      case model::WeightSlot::Proj: return block.proj;
      case model::WeightSlot::LayerOutputScale: return block.layer_output_scale;
      case model::WeightSlot::PerLayerProjNorm: return model_.weights.per_layer_proj_norm;
      case model::WeightSlot::OutputNorm: return model_.weights.output_norm;
      case model::WeightSlot::RopeFreqs: return model_.weights.rope_freqs;
      default: return {};
    }
  };

  // Quantized weight accessor.
  auto qweight = [&](model::WeightSlot s) -> const quant::QuantizedTensor& {
    static const quant::QuantizedTensor kEmpty;
    switch (s) {
      case model::WeightSlot::AttnQ: return block.attn_q;
      case model::WeightSlot::AttnK: return block.attn_k;
      case model::WeightSlot::AttnV: return block.attn_v;
      case model::WeightSlot::AttnOutput: return block.attn_output;
      case model::WeightSlot::FfnGate: return block.ffn_gate;
      case model::WeightSlot::FfnUp: return block.ffn_up;
      case model::WeightSlot::FfnDown: return block.ffn_down;
      case model::WeightSlot::TokenEmbd: return model_.weights.token_embd;
      case model::WeightSlot::PerLayerTokenEmbd: return model_.weights.per_layer_token_embd;
      default: return kEmpty;
    }
  };

  sx::Bytes out(out_td->size_bytes);
  bool ok = true;

  switch (node->op) {
    case OpKind::Embedding: {
      const uint64_t tok = read_i64(inputs[0]);
      const quant::QuantizedTensor& W =
          slot == model::WeightSlot::PerLayerTokenEmbd
              ? model_.weights.per_layer_token_embd
              : model_.weights.token_embd;
      ok = nn::embedding_f32(W, tok, f32_mut(out));
      break;
    }
    case OpKind::Scale: {
      ok = nn::scale(f32(inputs[0]), float(scale_d), f32_mut(out));
      break;
    }
    case OpKind::RmsNorm: {
      ok = nn::rms_norm(f32(inputs[0]), fweight(slot), eps, f32_mut(out));
      break;
    }
    case OpKind::RmsNormHeads: {
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      std::span<const float> w;
      if (slot == model::WeightSlot::AttnQNorm)
        w = block.attn_q_norm;
      else if (slot == model::WeightSlot::AttnKNorm)
        w = block.attn_k_norm;
      std::span<float> o = f32_mut(out);
      std::span<const float> x = f32(inputs[0]);
      std::copy(x.begin(), x.end(), o.begin());
      ok = nn::rms_norm_heads(o, w, lc.head_dim, eps);
      break;
    }
    case OpKind::Rope: {
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      const uint64_t pos = read_i64(inputs[1]);
      const std::span<const float> ff =
          lc.attention == model::AttentionKind::Global
              ? model_.weights.rope_freqs
              : std::span<const float>{};
      std::span<float> o = f32_mut(out);
      std::span<const float> x = f32(inputs[0]);
      std::copy(x.begin(), x.end(), o.begin());
      ok = nn::rope_neox_heads(o, lc.head_dim, pos, lc.rope_base, 1.0f, ff);
      break;
    }
    case OpKind::CastFp16: {
      std::span<float> o = f32_mut(out);
      std::span<const float> x = f32(inputs[0]);
      std::copy(x.begin(), x.end(), o.begin());
      nn::cast_fp16(o);
      ok = true;
      break;
    }
    case OpKind::Attention: {
      // KV layer: append the fp16-rounded K/V to this layer's cache at slot
      // `pos % n_slots`, then run incremental decode attention over the cache.
      // inputs: {q (RoPE'd f32), kc (fp16 K), vc (fp16 V), pos}.
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      const uint64_t pos = read_i64(inputs[3]);
      KvCache& cache = caches_[uint32_t(layer)];
      const size_t row = (size_t)cache.n_heads_kv * cache.head_dim;
      const size_t slot = (size_t)(pos % cache.n_slots) * row;
      std::span<const float> kc = f32(inputs[1]);
      std::span<const float> vc = f32(inputs[2]);
      std::copy_n(kc.begin(), row, cache.k.begin() + slot);
      std::copy_n(vc.begin(), row, cache.v.begin() + slot);
      ok = nn::decode_attention(f32(inputs[0]), cache.k, cache.v, pos,
                                cache.n_slots, lc.n_heads_q, cache.n_heads_kv,
                                cache.head_dim, lc.sliding_window, 1.0f,
                                f32_mut(out));
      break;
    }
    case OpKind::AttentionShared: {
      // Shared layer (24..41): attend over the donor KV layer's cache (no
      // append). inputs: {q (RoPE'd f32), pos}.
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      const uint64_t pos = read_i64(inputs[1]);
      const uint32_t donor = cfg.kv_donor_layer(uint32_t(layer));
      const KvCache& cache = caches_[donor];
      ok = nn::decode_attention(f32(inputs[0]), cache.k, cache.v, pos,
                                cache.n_slots, lc.n_heads_q, cache.n_heads_kv,
                                cache.head_dim, lc.sliding_window, 1.0f,
                                f32_mut(out));
      break;
    }
    case OpKind::GqaBroadcast: {
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      ok = nn::gqa_broadcast(f32(inputs[0]), f32_mut(out), lc.n_heads_q,
                             lc.n_heads_kv, lc.head_dim);
      break;
    }
    case OpKind::QuantizedMatmul: {
      ok = quant::matvec(qweight(slot), f32(inputs[0]), f32_mut(out),
                         quant::MatmulBackend::CpuQ8K);
      break;
    }
    case OpKind::MatvecF32: {
      std::span<const float> W = fweight(slot);
      std::span<const float> x = f32(inputs[0]);
      ok = nn::matvec_f32(W, x.size(), x, f32_mut(out));
      break;
    }
    case OpKind::MatvecBf16: {
      const uint64_t layer_base = (uint64_t(layer) * per) * d;
      const std::span<const uint16_t> W = model_.weights.per_layer_model_proj.subspan(
          layer_base, uint64_t(per) * d);
      // llama.cpp packs the f32 activation to bf16 (the weight's vec_dot_type)
      // before the dot; reproduce that rounding so the projection matches the
      // oracle (Phase 5D finding).
      std::span<const float> x = f32(inputs[0]);
      std::vector<float> x_bf16(x.size());
      for (size_t i = 0; i < x.size(); ++i)
        x_bf16[i] = bf16_to_f32(f32_to_bf16(x[i]));
      ok = nn::matvec_bf16(W, x.size(), x_bf16, f32_mut(out));
      break;
    }
    case OpKind::GeluFp16: {
      ok = nn::gelu_fp16(f32(inputs[0]), f32_mut(out));
      break;
    }
    case OpKind::Mul: {
      ok = nn::mul(f32(inputs[0]), f32(inputs[1]), f32_mut(out));
      break;
    }
    case OpKind::Add: {
      ok = nn::add(f32(inputs[0]), f32(inputs[1]), f32_mut(out));
      break;
    }
    case OpKind::LayerCombine: {
      const uint64_t col = uint64_t(layer) * per;
      const float ple_scale = std::sqrt(float(per));
      const float combine_scale = 1.0f / std::sqrt(2.0f);
      ok = nn::layer_combine(f32(inputs[0]),
                             f32(inputs[1]).subspan(col, per), ple_scale,
                             combine_scale, f32_mut(out));
      break;
    }
    case OpKind::Softcap: {
      ok = nn::softcap(f32(inputs[0]), float(scale_d), f32_mut(out));
      break;
    }
    default:
      ok = false;
      break;
  }

  if (!ok)
    return std::unexpected(fail("Sonic kernel failed for node '" + node->name +
                                "'"));

  std::vector<TensorValue> res;
  res.push_back(TensorValue::from_host(std::move(out)));
  return res;
}

} // namespace sonicboom::planner
