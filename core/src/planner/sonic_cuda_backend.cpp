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

#include <sonicboom/model/exec.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

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

bool SonicCudaBackend::prefill(std::span<const uint64_t> tokens, uint64_t start_pos,
                               std::span<float> out_hidden) {
  const auto& cfg = model_.config;
  if (model_.weights.blocks.empty()) return false;
  if (!quant::cuda::available() || !nn::cuda::available()) return false;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const float eps = cfg.rms_epsilon;
  const uint32_t n = uint32_t(tokens.size());
  if (n == 0 || out_hidden.size() < d) return false;
  if (caches_.size() != cfg.n_layer_kv()) return false;

  // The prompt's positions must land in a straight, wrap-free run of each KV
  // layer's cache (SWA ring n_slots == sliding_window, global n_slots == n_ctx).
  for (uint32_t l = 0; l < cfg.n_layer_kv(); ++l) {
    const KvCacheDev& c = caches_[l];
    if (c.n_slots == 0 || (start_pos % c.n_slots) + n > c.n_slots) return false;
  }

  // Activations are [dim, n] column-major (n innermost) to match the batched
  // f32 matmul layout; Q/K/V are transposed to [n, dim] position-major for the
  // per-position head norm / RoPE / flash attention, then back. Mirrors the CPU
  // SonicBackend::prefill; only the heavy kernels (batched matmul + flash
  // attention) run on the device.
  auto transpose = [&](const float* src, float* dst, uint32_t dim) {
    for (uint32_t c = 0; c < n; ++c)
      for (uint32_t k = 0; k < dim; ++k) dst[c * dim + k] = src[k * n + c];
  };
  auto transpose_back = [&](const float* src, float* dst, uint32_t dim) {
    for (uint32_t c = 0; c < n; ++c)
      for (uint32_t k = 0; k < dim; ++k) dst[k * n + c] = src[c * dim + k];
  };
  auto rms_cols = [&](const float* x, std::span<const float> w, uint32_t dim,
                      float* y) {
    std::vector<float> col(dim);
    for (uint32_t c = 0; c < n; ++c) {
      for (uint32_t k = 0; k < dim; ++k) col[k] = x[k * n + c];
      if (!nn::rms_norm(col, w, eps, col)) return false;
      for (uint32_t k = 0; k < dim; ++k) y[k * n + c] = col[k];
    }
    return true;
  };

  // Batched token embedding: res[k*n+c] = token_embd[tokens[c]][k] * sqrt(d).
  std::vector<float> res(uint64_t(d) * n);
  {
    std::vector<float> col(d);
    for (uint32_t c = 0; c < n; ++c) {
      if (!model::embed_token(model_, tokens[c], col)) return false;
      for (uint32_t k = 0; k < d; ++k) res[k * n + c] = col[k];
    }
  }

  const uint32_t n_kv = cfg.n_layer_kv();
  const uint32_t donor_swa_layer = n_kv - 2;
  const uint32_t donor_global_layer = n_kv - 1;
  std::vector<float> donor_swa_k, donor_swa_v, donor_global_k, donor_global_v;

  std::vector<float> next(uint64_t(d) * n);

  for (uint32_t l = 0; l < cfg.block_count; ++l) {
    const model::LayerConfig& lc = model_.plan[l];
    const model::BlockWeights& bw = model_.weights.blocks[l];
    const uint32_t n_q = lc.n_heads_q;
    const uint32_t n_kvh = lc.n_heads_kv;
    const uint32_t hd = lc.head_dim;
    const uint32_t qw = n_q * hd;
    const uint32_t kvw = n_kvh * hd;
    const bool is_kv = cfg.has_kv(l);
    const std::span<const float> freq =
        lc.attention == model::AttentionKind::Global ? model_.weights.rope_freqs
                                                     : std::span<const float>{};

    // --- attention -----------------------------------------------------------
    std::vector<float> h(uint64_t(d) * n);
    if (!rms_cols(res.data(), bw.attn_norm, d, h.data())) return false;

    std::vector<float> q_col(uint64_t(qw) * n);
    if (!quant::cuda::matmul_f32(bw.attn_q, h, q_col, n)) return false;
    std::vector<float> q(uint64_t(qw) * n);
    transpose(q_col.data(), q.data(), qw);
    for (uint32_t c = 0; c < n; ++c) {
      std::span<float> qc(q.data() + uint64_t(c) * qw, qw);
      if (!nn::rms_norm_heads(qc, bw.attn_q_norm, hd, eps)) return false;
      if (!nn::rope_neox_heads(qc, hd, start_pos + c, lc.rope_base, 1.0f, freq))
        return false;
    }

    std::vector<float> attn(uint64_t(qw) * n);
    if (is_kv) {
      std::vector<float> k_col(uint64_t(kvw) * n), v_col(uint64_t(kvw) * n);
      if (!quant::cuda::matmul_f32(bw.attn_k, h, k_col, n)) return false;
      if (!quant::cuda::matmul_f32(bw.attn_v, h, v_col, n)) return false;

      std::vector<float> k(uint64_t(kvw) * n), v(uint64_t(kvw) * n);
      transpose(k_col.data(), k.data(), kvw);
      transpose(v_col.data(), v.data(), kvw);

      KvCacheDev& cache = caches_[l];
      const size_t row = (size_t)cache.n_heads_kv * cache.head_dim;  // == kvw
      for (uint32_t c = 0; c < n; ++c) {
        std::span<float> kc(k.data() + uint64_t(c) * kvw, kvw);
        std::span<float> vc(v.data() + uint64_t(c) * kvw, kvw);
        if (!nn::rms_norm_heads(kc, bw.attn_k_norm, hd, eps)) return false;
        if (!nn::rms_norm_heads(vc, {}, hd, eps)) return false;
        nn::cast_fp16(vc);  // V is fp16-rounded without RoPE
        if (!nn::rope_neox_heads(kc, hd, start_pos + c, lc.rope_base, 1.0f, freq))
          return false;
        nn::cast_fp16(kc);  // K is fp16-rounded after RoPE (spine cache repr)
        const size_t off = (size_t)((start_pos + c) % cache.n_slots) * row;
        if (!cache.k_handle || !cache.v_handle ||
            !device::upload(kc.data(), cache.k_handle + off * sizeof(float),
                            kvw * sizeof(float)) ||
            !device::upload(vc.data(), cache.v_handle + off * sizeof(float),
                            kvw * sizeof(float)))
          return false;
      }

      if (!nn::cuda::flash_attention(q, n, k, v, n, n_q, n_kvh, hd,
                                     lc.sliding_window, 1.0f, attn))
        return false;

      if (l == donor_swa_layer) {
        donor_swa_k = std::move(k);
        donor_swa_v = std::move(v);
      } else if (l == donor_global_layer) {
        donor_global_k = std::move(k);
        donor_global_v = std::move(v);
      }
    } else {
      const uint32_t donor = cfg.kv_donor_layer(l);
      const std::vector<float>& dk =
          cfg.sliding_window_pattern[l] ? donor_swa_k : donor_global_k;
      const std::vector<float>& dv =
          cfg.sliding_window_pattern[l] ? donor_swa_v : donor_global_v;
      if (dk.empty() || dv.empty()) return false;
      if (!nn::cuda::flash_attention(q, n, dk, dv, n, n_q, n_kvh, hd,
                                     lc.sliding_window, 1.0f, attn))
        return false;
    }

    std::vector<float> attn_col(uint64_t(qw) * n);
    transpose_back(attn.data(), attn_col.data(), qw);
    std::vector<float> ap(uint64_t(d) * n);
    if (!quant::cuda::matmul_f32(bw.attn_output, attn_col, ap, n)) return false;

    std::vector<float> o(uint64_t(d) * n);
    if (!rms_cols(ap.data(), bw.post_attention_norm, d, o.data())) return false;
    std::vector<float> x1(uint64_t(d) * n);
    if (!nn::add(o, res, x1)) return false;

    // --- gated feed-forward --------------------------------------------------
    std::vector<float> f(uint64_t(d) * n);
    if (!rms_cols(x1.data(), bw.ffn_norm, d, f.data())) return false;
    std::vector<float> up(uint64_t(ff) * n), gate(uint64_t(ff) * n);
    if (!quant::cuda::matmul_f32(bw.ffn_up, f, up, n)) return false;
    if (!quant::cuda::matmul_f32(bw.ffn_gate, f, gate, n)) return false;
    if (!nn::gelu_fp16(gate, gate)) return false;
    if (!nn::mul(gate, up, gate)) return false;
    std::vector<float> ffn_out(uint64_t(d) * n);
    if (!quant::cuda::matmul_f32(bw.ffn_down, gate, ffn_out, n)) return false;
    std::vector<float> f2(uint64_t(d) * n);
    if (!rms_cols(ffn_out.data(), bw.post_ffw_norm, d, f2.data())) return false;
    std::vector<float> x2(uint64_t(d) * n);
    if (!nn::add(f2, x1, x2)) return false;

    // --- per-layer gate (per position; matches the spine's embed_per_layer) --
    std::vector<float> x2_col(d), g0(per), g1(d), g2(d), x3_col(d), ple(per);
    for (uint32_t c = 0; c < n; ++c) {
      for (uint32_t k = 0; k < d; ++k) x2_col[k] = x2[k * n + c];
      if (!model::embed_per_layer(model_, l, tokens[c], ple, false)) return false;
      if (!nn::matvec_f32(bw.inp_gate, d, x2_col, g0)) return false;
      if (!nn::gelu_fp16(g0, g0)) return false;
      if (!nn::mul(g0, ple, g0)) return false;
      if (!nn::matvec_f32(bw.proj, per, g0, g1)) return false;
      if (!nn::rms_norm(g1, bw.post_norm, eps, g2)) return false;
      if (!nn::add(x2_col, g2, x3_col)) return false;
      if (!bw.layer_output_scale.empty() &&
          !nn::scale(x3_col, bw.layer_output_scale[0], x3_col))
        return false;
      for (uint32_t k = 0; k < d; ++k) next[k * n + c] = x3_col[k];
    }

    std::swap(res, next);
  }

  // The final residual column (position n-1) is the post-prefill hidden state.
  for (uint32_t k = 0; k < d; ++k) out_hidden[k] = res[k * n + (n - 1)];
  return true;
}

} // namespace sonicboom::planner
