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
#include <sonicboom/model/exec.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/attention.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/embedding.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/moe.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/quant/quantized_matmul.h>

#include "async_prefetch.h"
#include "calibration.h"
#include "node_dump.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
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

// The string an attribute record carries for `name`, or false.
bool attr_s(const GraphNodeDesc& n, std::string_view name,
            std::string_view& out) {
  for (const auto& a : n.attributes)
    if (a.name == name && a.kind() == sx::AttrValueKind::String) {
      out = std::get<std::string>(a.value);
      return true;
    }
  return false;
}

} // namespace

SonicBackend::SonicBackend(const model::Gemma4Model& model, const Graph& graph,
                           uint64_t n_ctx)
    : model_(model), graph_(graph), dump_(NodeDump::create()) {
  ensure_caches(n_ctx);
}

SonicBackend::~SonicBackend() = default;

void SonicBackend::bind_segment_extent(uint64_t extent) {
  segment_extent_ = extent;
  if (dump_) dump_->begin_graph();
}

void SonicBackend::share_caches(const SonicBackend& other) {
  caches_ = other.caches_;
  states_ = other.states_;
}

void SonicBackend::ensure_caches(uint64_t n_ctx) {
  const uint32_t n_kv = model_.config.n_layer_kv();
  caches_ = std::make_shared<std::vector<KvCache>>();
  caches_->resize(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l) {
    const model::LayerConfig& lc = model_.config.layer_config(l);
    KvCache& c = (*caches_)[l];
    c.n_heads_kv = lc.n_heads_kv;
    c.head_dim = lc.head_dim;
    // SWA layers are a fixed-size ring (sliding_window slots); global layers are
    // the full decode length (n_ctx slots, no wrap).
    c.n_slots = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx;
    const size_t total = (size_t)c.n_slots * c.n_heads_kv * c.head_dim;
    c.k.assign(total, 0.0f);
    c.v.assign(total, 0.0f);
  }

  // One zeroed state per recurrent block, in the order the blocks appear: the
  // state index a gdn_scan node carries numbers them from 0 the same way. A
  // model with no recurrent block gets an empty set.
  const model::Gemma4Config& mc = model_.config;
  size_t n_states = 0;
  for (bool recurrent : mc.recurrent_layers)
    n_states += recurrent ? 1 : 0;
  const nn::GdnDims dims{mc.ssm_group_count, mc.ssm_time_step_rank,
                         mc.ssm_state_size, mc.ssm_conv_kernel,
                         mc.rms_epsilon};
  states_ = std::make_shared<StateSet>(
      std::vector<nn::GdnDims>(n_states, dims));
}

std::expected<std::vector<TensorValue>, RuntimeError> SonicBackend::execute(
    const TaskDesc& task, const std::vector<TensorValue>& in) {
  const auto* compute = task.as_compute();
  if (!compute || compute->graph_nodes.empty())
    return std::unexpected(fail("Sonic task has no compute node"));
  const auto* node = graph_.find_node(compute->graph_nodes.front());
  if (!node)
    return std::unexpected(fail("Sonic task references an unknown graph node"));

  // VS1 prefetch markers: host-side bookkeeping, not a kernel (the CPU backend
  // sees them when the plan runs without a device). No-op unless the switch
  // is on.
  if (node->op == OpKind::AsyncWeightPrefetch || node->op == OpKind::SyncPrefetch) {
    int64_t layer = -1;
    attr_i(*node, "layer", layer);
    int64_t bytes = 0;
    attr_i(*node, "prefetch_bytes", bytes);
    AsyncPrefetchRuntime& prefetch = AsyncPrefetchRuntime::instance();
    if (node->op == OpKind::AsyncWeightPrefetch)
      prefetch.begin(int(layer), bytes > 0 ? uint64_t(bytes) : 0);
    else
      prefetch.wait(int(layer));
    return std::vector<TensorValue>{};
  }

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

  // The weights this node's roles address (model/weights.h): a block role in
  // the node's layer, a model role wherever the layer points.
  const uint32_t weight_layer = layer >= 0 ? uint32_t(layer) : 0;
  auto fweight = [&](model::WeightSlot s) -> std::span<const float> {
    return model::weight_f32(model_.weights, {s, weight_layer});
  };
  auto qweight = [&](model::WeightSlot s) -> const quant::QuantizedTensor& {
    return model::weight_quant(model_.weights, {s, weight_layer});
  };

  // The number of positions this execution covers: the node's `batch` (the
  // extent the plan was built at) overridden by the extent bound for this
  // execution, so one plan serves every segment of its shape class.
  auto batch_attr = [&](int64_t& nb) {
    if (!attr_i(*node, "batch", nb))
      return false;
    if (segment_extent_ != 0)
      nb = int64_t(segment_extent_);
    return true;
  };

  // A tensor that scales with the plan's dynamic extent occupies its per-unit
  // stride times this execution's extent, plus its fixed part.
  if (out_td->dynamic_stride_bytes != 0 && segment_extent_ == 0)
    return std::unexpected(fail("dynamic tensor executed without an extent"));
  const uint64_t out_bytes =
      out_td->dynamic_stride_bytes != 0
          ? out_td->dynamic_stride_bytes * segment_extent_ +
                out_td->dynamic_base_bytes
          : out_td->size_bytes;
  sx::Bytes out(out_bytes);
  // A node with two results (MoeRoute: the expert ids and their weights) fills
  // this second buffer; it is returned after the first output.
  sx::Bytes out2;
  bool ok = true;

  // The buffer `idx`-th output of this node needs for this execution. A tensor
  // that scales with the plan's dynamic extent is sized from the extent.
  auto output_bytes = [&](size_t idx) -> uint64_t {
    const auto* td = graph_.find_tensor(node->outputs[idx]);
    if (!td) return 0;
    return td->dynamic_stride_bytes != 0
               ? td->dynamic_stride_bytes * segment_extent_ +
                     td->dynamic_base_bytes
               : td->size_bytes;
  };

  // The positions this execution covers: a verify step's row count, else the
  // plan's batch, else one decode position.
  auto row_count = [&]() -> uint64_t {
    int64_t vr = 0;
    if (attr_i(*node, "verify", vr) && vr > 0) return uint64_t(vr);
    int64_t nb = 0;
    if (batch_attr(nb) && nb > 0) return uint64_t(nb);
    return 1;
  };

  switch (node->op) {
    case OpKind::Embedding: {
      const quant::QuantizedTensor& W =
          slot == model::WeightSlot::PerLayerTokenEmbd
              ? model_.weights.per_layer_token_embd
              : model_.weights.token_embd;
      int64_t nb = 0;
      if (batch_attr(nb)) {
        // Batched: inputs[0] is the Int64 [n] token-id buffer (prefill).
        const uint64_t n = uint64_t(nb);
        std::span<float> o = f32_mut(out);  // [n, dim] position-major
        const uint64_t dim = n ? o.size() / n : 0;
        std::vector<float> row(dim);
        for (uint64_t c = 0; c < n; ++c) {
          uint64_t tok = 0;
          std::memcpy(&tok, inputs[0].data() + c * sizeof(uint64_t), sizeof(uint64_t));
          if (!nn::embedding_f32(W, tok, row)) { ok = false; break; }
          for (uint64_t k = 0; k < dim; ++k) o[c * dim + k] = row[k];
        }
      } else {
        const uint64_t tok = read_i64(inputs[0]);
        ok = nn::embedding_f32(W, tok, f32_mut(out));
      }
      break;
    }
    case OpKind::Scale: {
      ok = nn::scale(f32(inputs[0]), float(scale_d), f32_mut(out));
      break;
    }
    case OpKind::RmsNorm: {
      int64_t nb = 0, dim_i = 0;
      if (batch_attr(nb)) {
        // Batched row norm over [n, dim]: normalize each position with the
        // full-width weight (dim baked in as a node attribute by the emitter).
        attr_i(*node, "dim", dim_i);
        const uint64_t n = uint64_t(nb), dim = uint64_t(dim_i);
        std::span<const float> x = f32(inputs[0]);
        std::span<float> o = f32_mut(out);
        std::span<const float> w = fweight(slot);
        for (uint64_t c = 0; c < n; ++c) {
          if (!nn::rms_norm(x.subspan(c * dim, dim), w, eps, o.subspan(c * dim, dim))) {
            ok = false;
            break;
          }
        }
      } else {
        ok = nn::rms_norm(f32(inputs[0]), fweight(slot), eps, f32_mut(out));
      }
      break;
    }
    case OpKind::RmsNormHeads: {
      // Per-head (or per-segment) rms_norm. The segment width is the node's
      // `dim` attribute (defaults to the layer's head_dim); the weight is the
      // node's `slot` (empty => unit), so this also covers the model-wide
      // per-layer projection norm (slot == PerLayerProjNorm, dim == per).
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      const std::span<const float> w = fweight(slot);
      int64_t dim_i = lc.head_dim;
      attr_i(*node, "dim", dim_i);
      const uint64_t head_dim = uint64_t(dim_i);
      std::span<float> o = f32_mut(out);
      std::span<const float> x = f32(inputs[0]);
      std::copy(x.begin(), x.end(), o.begin());
      int64_t nb = 0;
      if (batch_attr(nb)) {
        const uint64_t n = uint64_t(nb);
        const uint64_t dim = n ? x.size() / n : 0;
        for (uint64_t c = 0; c < n; ++c) {
          if (!nn::rms_norm_heads(o.subspan(c * dim, dim), w, head_dim, eps)) {
            ok = false;
            break;
          }
        }
      } else {
        ok = nn::rms_norm_heads(o, w, head_dim, eps);
      }
      break;
    }
    case OpKind::Rope: {
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      const uint64_t pos = read_i64(inputs[1]);
      const std::span<const float> ff =
          lc.attention == model::AttentionKind::Global
              ? model_.weights.rope_freqs
              : std::span<const float>{};
      // 0 = the whole head; a partial-rope model rotates only the leading
      // values of each head (the angle denominator becomes rot_dims).
      int64_t rot_dims = 0;
      attr_i(*node, "rot_dims", rot_dims);
      std::span<float> o = f32_mut(out);
      std::span<const float> x = f32(inputs[0]);
      std::copy(x.begin(), x.end(), o.begin());
      int64_t nb = 0;
      if (batch_attr(nb)) {
        const uint64_t n = uint64_t(nb);
        const uint64_t dim = n ? x.size() / n : 0;
        for (uint64_t c = 0; c < n; ++c) {
          if (!nn::rope_neox_heads(o.subspan(c * dim, dim), lc.head_dim, pos + c,
                                   lc.rope_base, 1.0f, ff,
                                   uint64_t(rot_dims))) {
            ok = false;
            break;
          }
        }
      } else {
        ok = nn::rope_neox_heads(o, lc.head_dim, pos, lc.rope_base, 1.0f, ff,
                                 uint64_t(rot_dims));
      }
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
      // Attends over the KV-cache layer named by the node's `kv_source_layer`
      // attribute (identity for a KV layer, the donor for a shared-KV layer).
      // A KV layer appends its fp16-rounded K/V to the cache at slot
      // `pos % n_slots` before attending (inputs {q, kc, vc, pos}); a shared
      // layer omits K/V and does not append (inputs {q, pos}).
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      int64_t ksrc = layer;
      attr_i(*node, "kv_source_layer", ksrc);
      const uint32_t cache_layer = ksrc >= 0 ? uint32_t(ksrc) : uint32_t(layer);
      const bool append = inputs.size() >= 4;
      const uint64_t pos = read_i64(append ? inputs[3] : inputs[1]);
      KvCache& cache = (*caches_)[cache_layer];
      if (append) {
        const size_t row = (size_t)cache.n_heads_kv * cache.head_dim;
        const size_t slot = (size_t)(pos % cache.n_slots) * row;
        std::span<const float> kc = f32(inputs[1]);
        std::span<const float> vc = f32(inputs[2]);
        std::copy_n(kc.begin(), row, cache.k.begin() + slot);
        std::copy_n(vc.begin(), row, cache.v.begin() + slot);
      }
      double attn_scale = 1.0;
      attr_f(*node, "scale", attn_scale);
      ok = nn::decode_attention(f32(inputs[0]), cache.k, cache.v, pos,
                                cache.n_slots, lc.n_heads_q, cache.n_heads_kv,
                                cache.head_dim, lc.sliding_window, float(attn_scale),
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
      int64_t nb = 0;
      if (batch_attr(nb))
        ok = quant::matmul_f32_q8_K(qweight(slot), f32(inputs[0]), f32_mut(out),
                                    uint64_t(nb));
      else
        ok = quant::matvec(qweight(slot), f32(inputs[0]), f32_mut(out),
                           quant::MatmulBackend::CpuQ8K);
      break;
    }
    case OpKind::MatvecF32: {
      std::span<const float> W = fweight(slot);
      std::span<const float> x = f32(inputs[0]);
      int64_t nb = 0;
      if (batch_attr(nb)) {
        const uint64_t n = uint64_t(nb);
        std::span<float> o = f32_mut(out);  // [n, rows]
        const uint64_t dim = n ? x.size() / n : 0;
        const uint64_t rows = n ? o.size() / n : 0;
        std::vector<float> y(rows);
        for (uint64_t c = 0; c < n; ++c) {
          if (!nn::matvec_f32(W, dim, x.subspan(c * dim, dim), y)) {
            ok = false;
            break;
          }
          for (uint64_t r = 0; r < rows; ++r) o[c * rows + r] = y[r];
        }
      } else {
        ok = nn::matvec_f32(W, x.size(), x, f32_mut(out));
      }
      break;
    }
    case OpKind::MatvecBf16: {
      // Dense bf16 projection against a slice of per_layer_model_proj, named by
      // `offset` (start row) and `rows` (output width). The default (offset ==
      // layer*per, rows == per) is the per-layer projection; the hoisted
      // full-width projection uses offset == 0, rows == total_per.
      int64_t offset = (layer >= 0 ? int64_t(layer) : 0) * per;
      int64_t rows_i = per;
      attr_i(*node, "offset", offset);
      attr_i(*node, "rows", rows_i);
      const uint32_t rows = uint32_t(rows_i);
      const std::span<const uint16_t> W = model_.weights.per_layer_model_proj.subspan(
          uint64_t(offset) * d, uint64_t(rows) * d);
      // llama.cpp packs the f32 activation to bf16 (the weight's vec_dot_type)
      // before the dot; reproduce that rounding so the projection matches the
      // baseline (Phase 5D finding).
      std::span<const float> x = f32(inputs[0]);
      int64_t nb = 0;
      if (batch_attr(nb)) {
        const uint64_t n = uint64_t(nb);
        std::span<float> o = f32_mut(out);  // [n, rows]
        std::vector<float> x_bf16(d), y(rows);
        for (uint64_t c = 0; c < n; ++c) {
          for (uint32_t k = 0; k < d; ++k)
            x_bf16[k] = bf16_to_f32(f32_to_bf16(x[c * d + k]));
          if (!nn::matvec_bf16(W, d, x_bf16, y)) { ok = false; break; }
          for (uint32_t r = 0; r < rows; ++r) o[c * rows + r] = y[r];
        }
      } else {
        std::vector<float> x_bf16(x.size());
        for (size_t i = 0; i < x.size(); ++i)
          x_bf16[i] = bf16_to_f32(f32_to_bf16(x[i]));
        ok = nn::matvec_bf16(W, x.size(), x_bf16, f32_mut(out));
      }
      break;
    }
    case OpKind::GeluFp16: {
      ok = nn::gelu_fp16(f32(inputs[0]), f32_mut(out));
      break;
    }
    case OpKind::Silu: {
      ok = nn::silu(f32(inputs[0]), f32_mut(out));
      break;
    }
    case OpKind::Sigmoid: {
      ok = nn::sigmoid(f32(inputs[0]), f32_mut(out));
      break;
    }
    case OpKind::AddBias: {
      // inputs[0] is the projection output ([width] when decoding, [n, width]
      // when prefilling); the bias is a [width] weight vector broadcast over the
      // rows. A bias the architecture does not carry is never emitted, so an
      // empty span here is a plan/emit mismatch, not a benign no-op.
      const std::span<const float> b = fweight(slot);
      ok = !b.empty() && nn::add_bias(f32(inputs[0]), b, f32_mut(out));
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
    case OpKind::ScaleBy: {
      // [rows, width] * [rows]: each row scaled by its own value.
      ok = nn::mul_row(f32(inputs[0]), f32(inputs[1]), f32_mut(out));
      break;
    }
    case OpKind::Slice: {
      // Sub-span copy. `offset`/`length` name the contiguous span; when `stride`
      // is present it is a strided row slice over `rows` rows of `length`
      // elements each (rows == o.size() / length), used to extract a layer's
      // per-layer-embedding column from a [n, total_per] row.
      int64_t offset = 0, length = 0, stride = 0;
      attr_i(*node, "offset", offset);
      attr_i(*node, "length", length);
      attr_i(*node, "stride", stride);
      const std::span<const float> x = f32(inputs[0]);
      std::span<float> o = f32_mut(out);
      ok = (offset >= 0 && length >= 0);
      if (!ok) break;
      const uint64_t len = uint64_t(length);
      if (stride > 0) {
        ok = (o.size() % len == 0);
        if (!ok) break;
        const uint64_t rows = o.size() / len;
        for (uint64_t r = 0; r < rows; ++r) {
          const uint64_t src = uint64_t(offset) + r * uint64_t(stride);
          if (src + len > x.size()) { ok = false; break; }
          std::copy_n(x.begin() + src, len, o.begin() + r * len);
        }
      } else {
        ok = (uint64_t(offset) + len <= x.size() && o.size() >= len);
        if (!ok) break;
        std::copy_n(x.begin() + offset, len, o.begin());
      }
      break;
    }
    case OpKind::Softcap: {
      ok = nn::softcap(f32(inputs[0]), float(scale_d), f32_mut(out));
      break;
    }
    case OpKind::FlashAttention: {
      // Full-sequence causal SDPA over this chunk's fp16-rounded K/V (no append
      // — the append is the deferred KvAppend node). For a shared-KV layer the
      // graph passes the donor layer's K/V tensors (the `kv_source_layer`
      // attribute is unused here because the CPU prefill never reads the cache).
      // The CPU prefill runs a single wrap-free chunk (start_pos == 0, n <=
      // ring), so there is no prior cache to attend over.
      // inputs: {q, kc/dk, vc/dv, start_pos}.
      const model::LayerConfig& lc = cfg.layer_config(uint32_t(layer));
      int64_t nb = 0;
      batch_attr(nb);
      const uint64_t n = uint64_t(nb);
      double attn_scale = 1.0;
      attr_f(*node, "scale", attn_scale);
      ok = nn::scaled_dot_product_attention(f32(inputs[0]), n, f32(inputs[1]),
                                            f32(inputs[2]), n,
                                            lc.n_heads_q, lc.n_heads_kv,
                                            lc.head_dim, true, lc.sliding_window,
                                            0.0f, float(attn_scale), f32_mut(out));
      break;
    }
    case OpKind::KvAppend: {
      // Deferred cache append: copy the chunk's n fp16-rounded K/V rows into the
      // layer's cache at slots (start_pos + c) % n_slots. inputs: {kc, vc, start_pos}.
      const uint64_t start = read_i64(inputs[2]);
      int64_t nb = 0;
      batch_attr(nb);
      const uint64_t n = uint64_t(nb);
      KvCache& cache = (*caches_)[uint32_t(layer)];
      const size_t row = (size_t)cache.n_heads_kv * cache.head_dim;
      std::span<const float> k = f32(inputs[0]);
      std::span<const float> v = f32(inputs[1]);
      for (uint64_t c = 0; c < n; ++c) {
        const size_t slot = (size_t)((start + c) % cache.n_slots) * row;
        std::copy_n(k.data() + c * row, row, cache.k.begin() + slot);
        std::copy_n(v.data() + c * row, row, cache.v.begin() + slot);
      }
      ok = true;
      break;
    }
    case OpKind::MoeRoute: {
      // inputs: {activation}. The router's ids (int64) and weights (f32) come
      // out as this node's two results, one pair per position.
      int64_t experts = 0, used = 0;
      if (!attr_i(*node, "experts", experts) || !attr_i(*node, "used", used) ||
          experts <= 0 || used <= 0 || used > experts) {
        ok = false;
        break;
      }
      double route_scale = 1.0;
      attr_f(*node, "scale", route_scale);
      const uint64_t rows = row_count();
      const std::span<const float> x = f32(inputs[0]);
      if (rows == 0 || x.size() % rows != 0) {
        ok = false;
        break;
      }
      const uint64_t dim = x.size() / rows;
      const uint64_t k = uint64_t(used);
      const uint64_t ids_bytes = rows * k * sizeof(int64_t);
      const uint64_t w_bytes = rows * k * sizeof(float);
      if (out.size() < ids_bytes || output_bytes(1) < w_bytes) {
        ok = false;
        break;
      }
      out2.assign(w_bytes, std::byte(0));
      auto* ids_out = reinterpret_cast<int64_t*>(out.data());
      float* w_out = reinterpret_cast<float*>(out2.data());
      const quant::QuantizedTensor* router_q = nullptr;
      if (qweight(slot).valid()) router_q = &qweight(slot);
      const std::span<const float> router_f32 = fweight(slot);
      if (!router_q && router_f32.empty()) {
        ok = false;
        break;
      }
      for (uint64_t r = 0; r < rows && ok; ++r)
        ok = nn::moe_route(x.subspan(r * dim, dim), router_q, router_f32,
                           uint32_t(experts), uint32_t(used),
                           float(route_scale), {ids_out + r * k, k},
                           {w_out + r * k, k});
      break;
    }
    case OpKind::MoeExperts: {
      // inputs: {activation, ids, weights}. Each position's selected experts
      // are evaluated and summed in ascending expert index.
      int64_t experts = 0, slot_up_i = 0, slot_down_i = 0;
      if (!attr_i(*node, "experts", experts) ||
          !attr_i(*node, "slot_up", slot_up_i) ||
          !attr_i(*node, "slot_down", slot_down_i) || experts <= 0) {
        ok = false;
        break;
      }
      std::string_view activation = "silu";
      attr_s(*node, "activation", activation);
      const nn::MoeActivation act = activation == "gelu"
                                        ? nn::MoeActivation::GeluFp16
                                        : nn::MoeActivation::Silu;
      const uint64_t rows = row_count();
      const std::span<const float> x = f32(inputs[0]);
      if (rows == 0 || x.size() % rows != 0 ||
          inputs[1].size() % (rows * sizeof(int64_t)) != 0 ||
          inputs[2].size() != inputs[1].size() / sizeof(int64_t) * sizeof(float)) {
        ok = false;
        break;
      }
      const uint64_t dim = x.size() / rows;
      const uint64_t k = inputs[1].size() / sizeof(int64_t) / rows;
      std::span<float> y = f32_mut(out);
      if (k == 0 || y.size() < rows * dim) {
        ok = false;
        break;
      }
      const auto* ids_in = reinterpret_cast<const int64_t*>(inputs[1].data());
      const auto* w_in = reinterpret_cast<const float*>(inputs[2].data());
      const auto up_slot = static_cast<model::WeightSlot>(slot_up_i);
      const auto down_slot = static_cast<model::WeightSlot>(slot_down_i);
      std::vector<quant::QuantizedTensor> gates, ups, downs;
      gates.reserve(size_t(experts));
      ups.reserve(size_t(experts));
      downs.reserve(size_t(experts));
      for (int64_t e = 0; e < experts; ++e) {
        gates.push_back(model::weight_quant(
            model_.weights, {slot, weight_layer, uint32_t(e)}));
        ups.push_back(model::weight_quant(
            model_.weights, {up_slot, weight_layer, uint32_t(e)}));
        downs.push_back(model::weight_quant(
            model_.weights, {down_slot, weight_layer, uint32_t(e)}));
      }
      const nn::MoeExpertWeights views{std::span<const quant::QuantizedTensor>(
                                           gates.data(), gates.size()),
                                       std::span<const quant::QuantizedTensor>(
                                           ups.data(), ups.size()),
                                       std::span<const quant::QuantizedTensor>(
                                           downs.data(), downs.size())};
      std::fill(y.begin(), y.begin() + rows * dim, 0.0f);
      for (uint64_t r = 0; r < rows && ok; ++r)
        ok = nn::moe_experts(x.subspan(r * dim, dim), {ids_in + r * k, k},
                             {w_in + r * k, k}, views, act,
                             y.subspan(r * dim, dim));
      break;
    }
    case OpKind::GdnScan: {
      // The gated DeltaNet recurrence of one block over this execution's
      // positions, committed to the sequence's state: inputs {qkv [n, C],
      // beta logits [n, heads_v], alpha logits [n, heads_v], pos}. The
      // geometry and the state's index are the node's attributes; the
      // convolution, dt and a weights are the node's `slot`, `slot_dt` and
      // `slot_a` roles in its layer.
      int64_t state_index = -1, heads_k = 0, heads_v = 0, gdim = 0, kernel = 0;
      int64_t slot_dt_i = 0, slot_a_i = 0;
      if (!attr_i(*node, "state", state_index) || state_index < 0 ||
          !attr_i(*node, "heads_k", heads_k) ||
          !attr_i(*node, "heads_v", heads_v) || !attr_i(*node, "dim", gdim) ||
          !attr_i(*node, "kernel", kernel) ||
          !attr_i(*node, "slot_dt", slot_dt_i) ||
          !attr_i(*node, "slot_a", slot_a_i)) {
        ok = false;
        break;
      }
      if (!states_ || size_t(state_index) >= states_->size()) {
        ok = false;
        break;
      }
      const nn::GdnWeights w{
          fweight(slot),
          fweight(static_cast<model::WeightSlot>(slot_dt_i)),
          fweight(static_cast<model::WeightSlot>(slot_a_i))};
      const uint64_t start = read_i64(inputs[3]);
      const auto r = states_->commit(size_t(state_index), uint32_t(layer), start,
                                     w, f32(inputs[0]), f32(inputs[1]),
                                     f32(inputs[2]), f32_mut(out));
      if (!r)
        return std::unexpected(fail(r.error()));
      ok = true;
      break;
    }
    default:
      ok = false;
      break;
  }

  if (!ok)
    return std::unexpected(fail("Sonic kernel failed for node '" + node->name +
                                "'"));

  // boom-r4 F2: an enabled calibration run folds this tensor's extrema into the
  // process-wide sink, keyed by the graph tensor name the Boom IR also carries.
  if (out_td->dtype == sx::DType::Float32)
    calibration::record(out_td->name.c_str(),
                        std::span<const float>(
                            reinterpret_cast<const float *>(out.data()),
                            out.size() / sizeof(float)));

  // Side-effect-only nodes (KvAppend) have an orphaned dummy output the
  // planner drops from task.outputs; return zero outputs to match the task's
  // declared output count (mirrors the CUDA backend's identical handling).
  if (task.outputs.empty())
    return std::vector<TensorValue>{};

  // Debug dump (node_dump.h): the first output, host bytes as they are. The
  // dump names the node by its first output's tensor name (the emitter's
  // unique name; the runtime node name is the shared op kind).
  if (dump_)
    dump_->write(out_td->name, out.data(), out.size(), out_td->dtype);

  std::vector<TensorValue> res;
  res.push_back(TensorValue::from_host(std::move(out)));
  if (!out2.empty())
    res.push_back(TensorValue::from_host(std::move(out2)));
  return res;
}

} // namespace sonicboom::planner
