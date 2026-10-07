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

#include <sonicboom/planner/sonic_cuda_backend.h>

#include <sonicboom/model/plan.h>

#include "../nn/cuda_resident.h"
#include "../quant/cuda_resident.h"
#include "device_pool.h"

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

const float* dev(const TensorValue& v) {
  return reinterpret_cast<const float*>(v.device.handle);
}

float* dev_mut(uint64_t handle) {
  return reinterpret_cast<float*>(handle);
}

} // namespace

SonicCudaBackend::SonicCudaBackend(const model::Gemma4Model& model,
                                   const Graph& graph, uint64_t n_ctx)
    : model_(model), graph_(graph) {
  ensure_caches(n_ctx);
}

SonicCudaBackend::~SonicCudaBackend() {
  for (const auto& c : caches_) {
    if (c.k_handle) device::release(c.k_handle, c.bytes);
    if (c.v_handle) device::release(c.v_handle, c.bytes);
  }
}

void SonicCudaBackend::ensure_caches(uint64_t n_ctx) {
  const uint32_t n_kv = model_.config.n_layer_kv();
  caches_.clear();
  caches_.resize(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l) {
    const model::LayerConfig& lc = model_.config.layer_config(l);
    KvCacheDev& c = caches_[l];
    c.n_heads_kv = lc.n_heads_kv;
    c.head_dim = lc.head_dim;
    // SWA layers are a fixed-size ring; global layers the full decode length.
    c.n_slots = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx;
    c.bytes = c.n_slots * c.n_heads_kv * c.head_dim * sizeof(float);
    c.k_handle = device::acquire(c.bytes);
    c.v_handle = device::acquire(c.bytes);
  }
}

std::expected<std::vector<TensorValue>, RuntimeError> SonicCudaBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& in) {
  const auto* compute = task.as_compute();
  if (!compute || compute->graph_nodes.empty())
    return std::unexpected(fail("Sonic CUDA task has no compute node"));
  const auto* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(fail("Sonic CUDA task references an unknown graph node"));
  if (node->outputs.empty())
    return std::unexpected(fail("Sonic CUDA node has no output"));
  const auto* out_td = graph_.find_tensor(node->outputs.front());
  if (!out_td)
    return std::unexpected(fail("Sonic CUDA node output tensor is unknown"));

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

  const model::BlockWeights& block =
      model_.weights.blocks[layer >= 0 ? uint32_t(layer) : 0];
  const model::LayerConfig& lc = cfg.layer_config(layer >= 0 ? uint32_t(layer) : 0);

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
      case model::WeightSlot::PerLayerProjNorm: return model_.weights.per_layer_proj_norm;
      case model::WeightSlot::OutputNorm: return model_.weights.output_norm;
      case model::WeightSlot::RopeFreqs: return model_.weights.rope_freqs;
      default: return {};
    }
  };

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
      default: return kEmpty;
    }
  };

  // Allocate the device output buffer from the pool. Device activations never
  // reach this backend on the host, so every input is device-resident except the
  // Int64 `pos` scalar consumed by Rope.
  const uint64_t out_bytes = out_td->size_bytes;
  const uint64_t out_handle = device::acquire(out_bytes);
  if (!out_handle)
    return std::unexpected(fail("device buffer allocation failed"));
  float* out = dev_mut(out_handle);
  const int out_n = int(out_bytes / sizeof(float));

  bool ok = true;
  switch (node->op) {
    case OpKind::Scale: {
      ok = nn::cuda::scale_dev(dev(in[0]), float(scale_d), out, out_n);
      break;
    }
    case OpKind::RmsNorm: {
      ok = nn::cuda::rms_norm_dev(fweight(slot), dev(in[0]), eps, out, out_n);
      break;
    }
    case OpKind::RmsNormHeads: {
      std::span<const float> w;
      if (slot == model::WeightSlot::AttnQNorm)
        w = block.attn_q_norm;
      else if (slot == model::WeightSlot::AttnKNorm)
        w = block.attn_k_norm;
      ok = nn::cuda::rms_norm_heads_dev(w, dev(in[0]), eps, out, int(lc.head_dim),
                                        out_n / int(lc.head_dim));
      break;
    }
    case OpKind::Rope: {
      const uint64_t pos = read_i64(in[1].host);
      const std::span<const float> ff =
          lc.attention == model::AttentionKind::Global
              ? model_.weights.rope_freqs
              : std::span<const float>{};
      if (!device::device_copy(in[0].device.handle, out_handle, out_bytes)) {
        ok = false;
        break;
      }
      ok = nn::cuda::rope_neox_heads_dev(out, int(lc.head_dim), out_n / int(lc.head_dim),
                                         pos, lc.rope_base, 1.0f, ff);
      break;
    }
    case OpKind::CastFp16: {
      if (!device::device_copy(in[0].device.handle, out_handle, out_bytes)) {
        ok = false;
        break;
      }
      ok = nn::cuda::cast_fp16_dev(out, out_n);
      break;
    }
    case OpKind::Attention: {
      // KV layer: append the fp16-rounded K/V to this layer's device cache at
      // slot `pos % n_slots`, then run incremental decode attention over it.
      // inputs: {q (RoPE'd f32, device), kc (device), vc (device), pos (host)}.
      const uint64_t pos = read_i64(in[3].host);
      KvCacheDev& cache = caches_[uint32_t(layer)];
      const uint64_t row_bytes =
          cache.n_heads_kv * cache.head_dim * sizeof(float);
      const uint64_t off = (pos % cache.n_slots) * row_bytes;
      if (!cache.k_handle || !cache.v_handle ||
          !device::device_copy(in[1].device.handle, cache.k_handle + off, row_bytes) ||
          !device::device_copy(in[2].device.handle, cache.v_handle + off, row_bytes)) {
        ok = false;
        break;
      }
      ok = nn::cuda::decode_attention_dev(
          dev(in[0]), dev_mut(cache.k_handle), dev_mut(cache.v_handle), pos,
          cache.n_slots, int(lc.n_heads_q), int(cache.n_heads_kv),
          int(cache.head_dim), lc.sliding_window, 1.0f, out);
      break;
    }
    case OpKind::AttentionShared: {
      // Shared layer (24..41): attend over the donor KV layer's cache (no
      // append). inputs: {q (device), pos (host)}.
      const uint64_t pos = read_i64(in[1].host);
      const uint32_t donor = cfg.kv_donor_layer(uint32_t(layer));
      const KvCacheDev& cache = caches_[donor];
      ok = nn::cuda::decode_attention_dev(
          dev(in[0]), dev_mut(cache.k_handle), dev_mut(cache.v_handle), pos,
          cache.n_slots, int(lc.n_heads_q), int(cache.n_heads_kv),
          int(cache.head_dim), lc.sliding_window, 1.0f, out);
      break;
    }
    case OpKind::GqaBroadcast: {
      ok = nn::cuda::gqa_broadcast_dev(dev(in[0]), out, int(lc.n_heads_q),
                                       int(lc.n_heads_kv), int(lc.head_dim));
      break;
    }
    case OpKind::QuantizedMatmul: {
      ok = quant::cuda::matvec_f32_dev(qweight(slot), dev(in[0]), out);
      break;
    }
    case OpKind::MatvecF32: {
      const int cols = int(in[0].device.size_bytes / sizeof(float));
      ok = nn::cuda::matvec_f32_dev(fweight(slot), cols, dev(in[0]), out, out_n);
      break;
    }
    case OpKind::MatvecBf16: {
      const uint64_t layer_base = (uint64_t(layer) * per) * d;
      const std::span<const uint16_t> W =
          model_.weights.per_layer_model_proj.subspan(layer_base, uint64_t(per) * d);
      const int cols = int(in[0].device.size_bytes / sizeof(float));
      // The CPU path rounds the activation through bf16 before the dot; do the
      // same on a scratch copy so the shared `inp` tensor is left untouched.
      const uint64_t xbytes = in[0].device.size_bytes;
      uint64_t xtmp = device::acquire(xbytes);
      if (!xtmp) {
        ok = false;
        break;
      }
      if (!device::device_copy(in[0].device.handle, xtmp, xbytes) ||
          !nn::cuda::cast_bf16_dev(dev_mut(xtmp), cols)) {
        device::release(xtmp, xbytes);
        ok = false;
        break;
      }
      ok = nn::cuda::matvec_bf16_dev(W, cols, reinterpret_cast<const float*>(xtmp),
                                     out, out_n);
      device::release(xtmp, xbytes);
      break;
    }
    case OpKind::GeluFp16: {
      ok = nn::cuda::gelu_fp16_dev(dev(in[0]), out, out_n);
      break;
    }
    case OpKind::Mul: {
      ok = nn::cuda::mul_dev(dev(in[0]), dev(in[1]), out, out_n);
      break;
    }
    case OpKind::Add: {
      ok = nn::cuda::add_dev(dev(in[0]), dev(in[1]), out, out_n);
      break;
    }
    case OpKind::LayerCombine: {
      const uint64_t col = uint64_t(layer) * per;
      const float ple_scale = std::sqrt(float(per));
      const float combine_scale = 1.0f / std::sqrt(2.0f);
      ok = nn::cuda::layer_combine_dev(dev(in[0]), dev(in[1]) + col, ple_scale,
                                       combine_scale, out, int(per));
      break;
    }
    default:
      // Embedding / Softcap are placed on the CPU SonicBackend; hitting them
      // here means the placement pass routed a host-only op to the device.
      ok = false;
      break;
  }

  if (!ok) {
    device::release(out_handle, out_bytes);
    return std::unexpected(fail("Sonic CUDA kernel failed for node '" + node->name +
                                "'"));
  }

  std::vector<TensorValue> res;
  res.push_back(TensorValue::from_device(out_handle, out_bytes));
  return res;
}

} // namespace sonicboom::planner
