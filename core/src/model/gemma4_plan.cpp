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

#include <sonicboom/model/plan.h>

#include <sonicboom/model/exec.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/rewrites.h>
#include <sonicboom/planner/runtime_executor.h>
#include <sonicboom/planner/sonic_backend.h>
#include <sonicboom/planner/sonic_cuda_backend.h>
#include <sonicboom/sx/exec.h>

#ifdef SONICBOOM_USE_CUDA
#include <nvToolsExt.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sonicboom::model {

namespace {

#ifdef SONICBOOM_USE_CUDA
// NVTX step markers are opt-in (see SonicCudaBackend::nvtx_enabled for why):
// they perturb host timing and shift the split-K gemv's non-deterministic
// atomicAdd order, so they stay OFF for normal runs and ON only when
// SONICBOOM_NVTX is set (the nsys measurement runs).
bool nvtx_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("SONICBOOM_NVTX");
    return e && *e && std::string_view(e) != "0";
  }();
  return on;
}
#endif

// Encode a scalar graph input as a little-endian 64-bit buffer (host x86-64).
sx::Bytes encode_i64(uint64_t v) {
  sx::Bytes b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// Encode the prompt token ids as a little-endian 64-bit-per-token buffer (the
// prefill graph's `tokens` Int64[n] input, decoded back by EmbeddingBatched).
sx::Bytes encode_tokens(std::span<const uint64_t> tokens) {
  sx::Bytes b(tokens.size() * sizeof(uint64_t));
  std::memcpy(b.data(), tokens.data(), b.size());
  return b;
}

sx::Attribute attr_i(const char* name, int64_t v) { return sx::Attribute{name, v}; }
sx::Attribute attr_f(const char* name, double v) { return sx::Attribute{name, v}; }

} // namespace

// ---------------------------------------------------------------------------
// Plan emitter. Mirrors forward()/lm_head in exec.cpp — the same per-stage
// arithmetic, re-expressed as a static acyclic graph of transformer OpKind
// nodes. The graph is stateless (acyclic): the multi-token KV cache lives in the
// SonicBackend, keyed by layer. The `Attention` node appends its RoPE'd fp16 K
// and normed fp16 V to the layer's cache and attends over the accumulated keys;
// the `AttentionShared` node (layers 24..41) attends over its donor layer's
// (22 SWA / 23 global) cache. The residual stream is the *scaled* l_out of the
// previous block.
// ---------------------------------------------------------------------------
planner::Graph build_gemma4_graph(const Gemma4Model& m, bool greedy) {
  using namespace planner;

  const auto& cfg = m.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const uint32_t block_count = cfg.block_count;
  const uint32_t total_per = block_count * per;
  const uint64_t vocab = cfg.vocab_size;

  struct Builder {
    Graph g;
    uint32_t next_tensor = 0;
    uint32_t next_node = 0;

    TensorId tensor(std::string name, std::vector<int64_t> shape, sx::DType dt,
                    bool is_in, bool is_out = false) {
      sx::TensorType ty{dt, sx::Shape{shape}};
      auto sz = sx::tensor_byte_size(ty);  // always valid for these fixed shapes
      TensorDesc td;
      td.id = TensorId{next_tensor++};
      td.shape = std::move(shape);
      td.dtype = dt;
      td.size_bytes = static_cast<uint64_t>(*sz);
      td.is_graph_input = is_in;
      td.is_graph_output = is_out;
      td.name = std::move(name);
      g.tensors.push_back(std::move(td));
      return g.tensors.back().id;
    }

    TensorId node(OpKind op, std::vector<TensorId> ins, std::string out_name,
                  std::vector<int64_t> out_shape,
                  std::vector<sx::Attribute> attrs) {
      TensorId o = tensor(std::move(out_name), std::move(out_shape),
                          sx::DType::Float32, /*is_in=*/false);
      GraphNodeDesc nd;
      nd.id = GraphNodeId{next_node++};
      nd.op = op;
      nd.inputs = std::move(ins);
      nd.outputs = {o};
      nd.attributes = std::move(attrs);
      nd.backend = BackendTag::Sonic;
      nd.name = op_kind_name(op);
      g.nodes.push_back(std::move(nd));
      return o;
    }
  } b;

  b.g.name = "gemma4_decode";

  // Graph inputs: token id + absolute position (Int64 scalars).
  const TensorId token_id = b.tensor("token_id", {}, sx::DType::Int64, /*is_in=*/true);
  const TensorId pos = b.tensor("pos", {}, sx::DType::Int64, /*is_in=*/true);
  b.g.inputs = {token_id, pos};

  auto slot = [](WeightSlot s) { return attr_i("slot", int64_t(s)); };
  auto layer_a = [](uint32_t l) { return attr_i("layer", int64_t(l)); };

  // Preamble: the shared token embedding (scaled), and the full per-layer token
  // embedding row (dequantized once; each block's LayerCombine slices its `per`
  // elements). Both are deterministic, so hoisting them out of the block loop is
  // bit-identical to exec.cpp recomputing them per block.
  const TensorId embd = b.node(OpKind::Embedding, {token_id}, "embd", {int64_t(d)},
                               {slot(WeightSlot::TokenEmbd)});
  const TensorId inp = b.node(OpKind::Scale, {embd}, "inp", {int64_t(d)},
                              {attr_f("scale", std::sqrt(float(d)))});
  const TensorId ple_row = b.node(OpKind::Embedding, {token_id}, "ple_row",
                                  {int64_t(total_per)},
                                  {slot(WeightSlot::PerLayerTokenEmbd)});

  // Hoisted per-layer input (Phase 2.3a): one full-width 2560→10752 bf16 gemv +
  // scale + segmented rms_norm + combine per token, done once over [total_per].
  // Each block's gate then slices its [per] segment instead of recomputing the
  // 2560→256 projection. Backends fold the proj_scale (1/sqrt(d)), segmented
  // rms_norm (shared per_layer_proj_norm), and combine (1/sqrt(2)) into this op.
  const TensorId ple_all = b.node(OpKind::PerLayerInput, {inp, ple_row}, "ple_all",
                                  {int64_t(total_per)}, {});

  TensorId residual = inp;
  for (uint32_t l = 0; l < block_count; ++l) {
    const LayerConfig& lc = cfg.layer_config(l);
    const uint32_t n_q = lc.n_heads_q;
    const uint32_t n_kv = lc.n_heads_kv;
    const uint32_t hd = lc.head_dim;
    const uint32_t qw = n_q * hd;
    const uint32_t kvw = n_kv * hd;
    const std::string suff = std::to_string(l);

    // --- attention ---------------------------------------------------------
    // Q is computed for every layer (llama.cpp projects Q before the has_kv
    // branch). KV layers append their RoPE'd K (fp16) and normed V (fp16) to the
    // per-layer cache then attend over the accumulated keys; shared layers
    // (24..41) attend over their donor layer's cache instead.
    TensorId h = b.node(OpKind::RmsNorm, {residual}, "h" + suff, {int64_t(d)},
                        {layer_a(l), slot(WeightSlot::AttnNorm)});
    TensorId q = b.node(OpKind::QuantizedMatmul, {h}, "q" + suff, {int64_t(qw)},
                        {layer_a(l), slot(WeightSlot::AttnQ)});
    q = b.node(OpKind::RmsNormHeads, {q}, "qn" + suff, {int64_t(qw)},
               {layer_a(l), slot(WeightSlot::AttnQNorm)});
    q = b.node(OpKind::Rope, {q, pos}, "qr" + suff, {int64_t(qw)}, {layer_a(l)});

    TensorId attn;
    if (cfg.has_kv(l)) {
      TensorId k = b.node(OpKind::QuantizedMatmul, {h}, "k" + suff, {int64_t(kvw)},
                          {layer_a(l), slot(WeightSlot::AttnK)});
      TensorId v = b.node(OpKind::QuantizedMatmul, {h}, "v" + suff, {int64_t(kvw)},
                          {layer_a(l), slot(WeightSlot::AttnV)});
      k = b.node(OpKind::RmsNormHeads, {k}, "kn" + suff, {int64_t(kvw)},
                 {layer_a(l), slot(WeightSlot::AttnKNorm)});
      v = b.node(OpKind::RmsNormHeads, {v}, "vn" + suff, {int64_t(kvw)},
                 {layer_a(l)});  // V norm: unit weight
      const TensorId vc = b.node(OpKind::CastFp16, {v}, "vc" + suff, {int64_t(kvw)},
                                 {layer_a(l)});
      k = b.node(OpKind::Rope, {k, pos}, "kr" + suff, {int64_t(kvw)}, {layer_a(l)});
      const TensorId kc = b.node(OpKind::CastFp16, {k}, "kc" + suff, {int64_t(kvw)},
                                 {layer_a(l)});
      attn = b.node(OpKind::Attention, {q, kc, vc, pos}, "attn" + suff,
                    {int64_t(qw)}, {layer_a(l)});
    } else {
      attn = b.node(OpKind::AttentionShared, {q, pos}, "attn" + suff,
                    {int64_t(qw)}, {layer_a(l)});
    }

    TensorId ap = b.node(OpKind::QuantizedMatmul, {attn}, "ap" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::AttnOutput)});
    TensorId o = b.node(OpKind::RmsNorm, {ap}, "o" + suff, {int64_t(d)},
                        {layer_a(l), slot(WeightSlot::PostAttentionNorm)});
    TensorId x1 = b.node(OpKind::Add, {o, residual}, "x1" + suff, {int64_t(d)},
                         {layer_a(l)});

    // --- gated feed-forward ------------------------------------------------
    TensorId f = b.node(OpKind::RmsNorm, {x1}, "f" + suff, {int64_t(d)},
                        {layer_a(l), slot(WeightSlot::FfnNorm)});
    TensorId up = b.node(OpKind::QuantizedMatmul, {f}, "up" + suff, {int64_t(ff)},
                         {layer_a(l), slot(WeightSlot::FfnUp)});
    TensorId gate = b.node(OpKind::QuantizedMatmul, {f}, "gate" + suff, {int64_t(ff)},
                           {layer_a(l), slot(WeightSlot::FfnGate)});
    gate = b.node(OpKind::GeluFp16, {gate}, "gelu" + suff, {int64_t(ff)}, {layer_a(l)});
    TensorId fh = b.node(OpKind::Mul, {gate, up}, "fh" + suff, {int64_t(ff)},
                         {layer_a(l)});
    TensorId fo = b.node(OpKind::QuantizedMatmul, {fh}, "fo" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::FfnDown)});
    TensorId f2 = b.node(OpKind::RmsNorm, {fo}, "f2" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::PostFfwNorm)});
    TensorId x2 = b.node(OpKind::Add, {f2, x1}, "x2" + suff, {int64_t(d)},
                         {layer_a(l)});

    // --- per-layer gate ----------------------------------------------------
    // The per-layer input (proj/scale/rms/combine) was hoisted to the preamble;
    // each block slices its [per] segment of ple_all.
    TensorId ple = b.node(OpKind::Slice, {ple_all}, "ple" + suff, {int64_t(per)},
                          {attr_i("offset", int64_t(l) * per),
                           attr_i("length", int64_t(per))});
    TensorId g0 = b.node(OpKind::MatvecF32, {x2}, "g0" + suff, {int64_t(per)},
                         {layer_a(l), slot(WeightSlot::InpGate)});
    g0 = b.node(OpKind::GeluFp16, {g0}, "g0g" + suff, {int64_t(per)}, {layer_a(l)});
    g0 = b.node(OpKind::Mul, {g0, ple}, "g0m" + suff, {int64_t(per)}, {layer_a(l)});
    TensorId g1 = b.node(OpKind::MatvecF32, {g0}, "g1" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::Proj)});
    TensorId g2 = b.node(OpKind::RmsNorm, {g1}, "g2" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::PostNorm)});
    TensorId x3 = b.node(OpKind::Add, {x2, g2}, "x3" + suff, {int64_t(d)},
                         {layer_a(l)});

    // --- whole-block output scale ------------------------------------------
    TensorId out;
    if (!m.weights.blocks[l].layer_output_scale.empty())
      out = b.node(OpKind::Scale, {x3}, "out" + suff, {int64_t(d)},
                   {attr_f("scale", double(m.weights.blocks[l].layer_output_scale[0]))});
    else
      out = x3;
    residual = out;
  }

  // --- lm head -------------------------------------------------------------
  TensorId hn = b.node(OpKind::RmsNorm, {residual}, "result_norm", {int64_t(d)},
                       {slot(WeightSlot::OutputNorm)});
  TensorId logits = b.node(OpKind::QuantizedMatmul, {hn}, "logits", {int64_t(vocab)},
                           {slot(WeightSlot::TokenEmbd)});

  // Greedy decode: softcapping (sc * tanh(x/sc)) is monotonic so it does not
  // change argmax — skip it and reduce the [vocab] logits to a single index on
  // the device, so the per-token host transfer shrinks from the full 1 MiB
  // logits to 4 bytes. Sampling mode keeps the softcapped full logits (the
  // sampler needs the distribution).
  TensorId out;
  if (greedy) {
    out = b.node(OpKind::Argmax, {logits}, "argmax", {int64_t(1)}, {});
  } else {
    out = b.node(OpKind::Softcap, {logits}, "logits_sc", {int64_t(vocab)},
                 {attr_f("scale", double(cfg.final_logit_softcapping))});
  }

  for (auto& t : b.g.tensors)
    if (t.id == out)
      t.is_graph_output = true;
  b.g.outputs = {out};

  return std::move(b.g);
}

// ---------------------------------------------------------------------------
// Prefill graph emitter (Phase 7). Mirrors build_gemma4_graph per block, but on
// [dim, n] column-major / [n, dim] position-major batched tensors (n == the
// prompt length) so the whole prompt is processed in one graph pass through the
// spine. The batch size n and per-node `dim` are baked into node attributes; the
// elementwise nodes (Scale/Add/Mul/GeluFp16/CastFp16) reuse their per-token
// OpKind on the flat batched buffers. FlashAttention appends the n fp16-rounded
// K/V rows to the per-layer cache and runs full-sequence causal SDPA;
// FlashAttentionShared attends over the donor layer's fp16-rounded K/V graph
// tensors (no append), so the memory planner keeps those alive across layers
// 24..41 via ordinary tensor liveness. Graph inputs are `tokens` (Int64 [n]) and
// `start_pos` (Int64); the graph output is the final residual [dim, n].
// ---------------------------------------------------------------------------
planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n) {
  using namespace planner;

  const auto& cfg = m.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const uint32_t block_count = cfg.block_count;
  const uint32_t total_per = block_count * per;

  struct Builder {
    Graph g;
    uint32_t next_tensor = 0;
    uint32_t next_node = 0;

    TensorId tensor(std::string name, std::vector<int64_t> shape, sx::DType dt,
                    bool is_in, bool is_out = false) {
      sx::TensorType ty{dt, sx::Shape{shape}};
      auto sz = sx::tensor_byte_size(ty);  // always valid for these fixed shapes
      TensorDesc td;
      td.id = TensorId{next_tensor++};
      td.shape = std::move(shape);
      td.dtype = dt;
      td.size_bytes = static_cast<uint64_t>(*sz);
      td.is_graph_input = is_in;
      td.is_graph_output = is_out;
      td.name = std::move(name);
      g.tensors.push_back(std::move(td));
      return g.tensors.back().id;
    }

    TensorId node(OpKind op, std::vector<TensorId> ins, std::string out_name,
                  std::vector<int64_t> out_shape,
                  std::vector<sx::Attribute> attrs) {
      TensorId o = tensor(std::move(out_name), std::move(out_shape),
                          sx::DType::Float32, /*is_in=*/false);
      GraphNodeDesc nd;
      nd.id = GraphNodeId{next_node++};
      nd.op = op;
      nd.inputs = std::move(ins);
      nd.outputs = {o};
      nd.attributes = std::move(attrs);
      nd.backend = BackendTag::Sonic;
      nd.name = op_kind_name(op);
      g.nodes.push_back(std::move(nd));
      return o;
    }
  } b;

  b.g.name = "gemma4_prefill";

  // Graph inputs: the prompt token ids (Int64 [n]) and the absolute start
  // position (Int64 scalar).
  const TensorId tokens = b.tensor("tokens", {int64_t(n)}, sx::DType::Int64, /*is_in=*/true);
  const TensorId start_pos = b.tensor("start_pos", {}, sx::DType::Int64, /*is_in=*/true);
  b.g.inputs = {tokens, start_pos};

  auto slot = [](WeightSlot s) { return attr_i("slot", int64_t(s)); };
  auto layer_a = [](uint32_t l) { return attr_i("layer", int64_t(l)); };
  auto batch_a = [](uint64_t v) { return attr_i("batch", int64_t(v)); };
  auto dim_a = [](uint64_t v) { return attr_i("dim", int64_t(v)); };
  auto dir_a = [](int64_t v) { return attr_i("dir", v); };
  auto shape2 = [](uint64_t r, uint64_t c) {
    return std::vector<int64_t>{int64_t(r), int64_t(c)};
  };

  const TensorId embd = b.node(OpKind::EmbeddingBatched, {tokens}, "embd",
                               shape2(d, n), {slot(WeightSlot::TokenEmbd), batch_a(n)});
  const TensorId inp = b.node(OpKind::Scale, {embd}, "inp", shape2(d, n),
                              {attr_f("scale", std::sqrt(float(d)))});
  const TensorId ple_row = b.node(OpKind::EmbeddingBatched, {tokens}, "ple_row",
                                  shape2(total_per, n),
                                  {slot(WeightSlot::PerLayerTokenEmbd), batch_a(n)});

  const float proj_scale = 1.0f / std::sqrt(float(d));
  const uint32_t n_kv = cfg.n_layer_kv();
  const uint32_t donor_swa_layer = n_kv - 2;
  const uint32_t donor_global_layer = n_kv - 1;
  TensorId donor_swa_k{}, donor_swa_v{}, donor_global_k{}, donor_global_v{};

  TensorId residual = inp;
  for (uint32_t l = 0; l < block_count; ++l) {
    const LayerConfig& lc = cfg.layer_config(l);
    const uint32_t n_q = lc.n_heads_q;
    const uint32_t n_kvh = lc.n_heads_kv;
    const uint32_t hd = lc.head_dim;
    const uint32_t qw = n_q * hd;
    const uint32_t kvw = n_kvh * hd;
    const std::string suff = std::to_string(l);

    // --- attention ---------------------------------------------------------
    TensorId h = b.node(OpKind::RmsNormCols, {residual}, "h" + suff, shape2(d, n),
                        {layer_a(l), slot(WeightSlot::AttnNorm), batch_a(n), dim_a(d)});
    TensorId q_col = b.node(OpKind::QuantizedMatmulBatched, {h}, "q_col" + suff,
                            shape2(qw, n), {layer_a(l), slot(WeightSlot::AttnQ), batch_a(n)});
    TensorId q = b.node(OpKind::Transpose, {q_col}, "q" + suff, shape2(n, qw),
                        {batch_a(n), dim_a(qw), dir_a(0)});
    q = b.node(OpKind::RmsNormHeadsBatched, {q}, "qn" + suff, shape2(n, qw),
               {layer_a(l), slot(WeightSlot::AttnQNorm), batch_a(n)});
    q = b.node(OpKind::RopeBatched, {q, start_pos}, "qr" + suff, shape2(n, qw),
               {layer_a(l), batch_a(n)});

    TensorId attn;
    if (cfg.has_kv(l)) {
      TensorId k_col = b.node(OpKind::QuantizedMatmulBatched, {h}, "k_col" + suff,
                              shape2(kvw, n), {layer_a(l), slot(WeightSlot::AttnK), batch_a(n)});
      TensorId k = b.node(OpKind::Transpose, {k_col}, "k" + suff, shape2(n, kvw),
                          {batch_a(n), dim_a(kvw), dir_a(0)});
      k = b.node(OpKind::RmsNormHeadsBatched, {k}, "kn" + suff, shape2(n, kvw),
                 {layer_a(l), slot(WeightSlot::AttnKNorm), batch_a(n)});
      k = b.node(OpKind::RopeBatched, {k, start_pos}, "kr" + suff, shape2(n, kvw),
                 {layer_a(l), batch_a(n)});
      const TensorId kc = b.node(OpKind::CastFp16, {k}, "kc" + suff, shape2(n, kvw),
                                 {layer_a(l)});
      TensorId v_col = b.node(OpKind::QuantizedMatmulBatched, {h}, "v_col" + suff,
                              shape2(kvw, n), {layer_a(l), slot(WeightSlot::AttnV), batch_a(n)});
      TensorId v = b.node(OpKind::Transpose, {v_col}, "v" + suff, shape2(n, kvw),
                          {batch_a(n), dim_a(kvw), dir_a(0)});
      v = b.node(OpKind::RmsNormHeadsBatched, {v}, "vn" + suff, shape2(n, kvw),
                 {layer_a(l), batch_a(n)});  // V norm: unit weight
      const TensorId vc = b.node(OpKind::CastFp16, {v}, "vc" + suff, shape2(n, kvw),
                                 {layer_a(l)});
      attn = b.node(OpKind::FlashAttention, {q, kc, vc, start_pos}, "attn" + suff,
                    shape2(n, qw), {layer_a(l), batch_a(n)});
      if (l == donor_swa_layer) {
        donor_swa_k = kc;
        donor_swa_v = vc;
      } else if (l == donor_global_layer) {
        donor_global_k = kc;
        donor_global_v = vc;
      }
    } else {
      const uint32_t donor = cfg.kv_donor_layer(l);
      const TensorId dk = (donor == donor_swa_layer) ? donor_swa_k : donor_global_k;
      const TensorId dv = (donor == donor_swa_layer) ? donor_swa_v : donor_global_v;
      attn = b.node(OpKind::FlashAttentionShared, {q, dk, dv}, "attn" + suff,
                    shape2(n, qw), {layer_a(l), batch_a(n)});
    }

    TensorId attn_col = b.node(OpKind::Transpose, {attn}, "attn_col" + suff,
                               shape2(qw, n), {batch_a(n), dim_a(qw), dir_a(1)});
    TensorId ap = b.node(OpKind::QuantizedMatmulBatched, {attn_col}, "ap" + suff,
                         shape2(d, n), {layer_a(l), slot(WeightSlot::AttnOutput), batch_a(n)});
    TensorId o = b.node(OpKind::RmsNormCols, {ap}, "o" + suff, shape2(d, n),
                        {layer_a(l), slot(WeightSlot::PostAttentionNorm), batch_a(n), dim_a(d)});
    TensorId x1 = b.node(OpKind::Add, {o, residual}, "x1" + suff, shape2(d, n), {layer_a(l)});

    // --- gated feed-forward ------------------------------------------------
    TensorId f = b.node(OpKind::RmsNormCols, {x1}, "f" + suff, shape2(d, n),
                        {layer_a(l), slot(WeightSlot::FfnNorm), batch_a(n), dim_a(d)});
    TensorId up = b.node(OpKind::QuantizedMatmulBatched, {f}, "up" + suff, shape2(ff, n),
                         {layer_a(l), slot(WeightSlot::FfnUp), batch_a(n)});
    TensorId gate = b.node(OpKind::QuantizedMatmulBatched, {f}, "gate" + suff, shape2(ff, n),
                           {layer_a(l), slot(WeightSlot::FfnGate), batch_a(n)});
    gate = b.node(OpKind::GeluFp16, {gate}, "gelu" + suff, shape2(ff, n), {layer_a(l)});
    TensorId fh = b.node(OpKind::Mul, {gate, up}, "fh" + suff, shape2(ff, n), {layer_a(l)});
    TensorId fo = b.node(OpKind::QuantizedMatmulBatched, {fh}, "fo" + suff, shape2(d, n),
                         {layer_a(l), slot(WeightSlot::FfnDown), batch_a(n)});
    TensorId f2 = b.node(OpKind::RmsNormCols, {fo}, "f2" + suff, shape2(d, n),
                         {layer_a(l), slot(WeightSlot::PostFfwNorm), batch_a(n), dim_a(d)});
    TensorId x2 = b.node(OpKind::Add, {f2, x1}, "x2" + suff, shape2(d, n), {layer_a(l)});

    // --- per-layer gate ----------------------------------------------------
    TensorId proj = b.node(OpKind::MatvecBf16Batched, {inp}, "proj" + suff, shape2(per, n),
                           {layer_a(l), batch_a(n)});
    proj = b.node(OpKind::Scale, {proj}, "projs" + suff, shape2(per, n),
                  {attr_f("scale", proj_scale)});
    proj = b.node(OpKind::RmsNormCols, {proj}, "projn" + suff, shape2(per, n),
                  {layer_a(l), slot(WeightSlot::PerLayerProjNorm), batch_a(n), dim_a(per)});
    TensorId ple = b.node(OpKind::LayerCombineBatched, {proj, ple_row}, "ple" + suff,
                          shape2(per, n), {layer_a(l), batch_a(n)});
    TensorId g0 = b.node(OpKind::MatvecF32Batched, {x2}, "g0" + suff, shape2(per, n),
                         {layer_a(l), slot(WeightSlot::InpGate), batch_a(n)});
    g0 = b.node(OpKind::GeluFp16, {g0}, "g0g" + suff, shape2(per, n), {layer_a(l)});
    g0 = b.node(OpKind::Mul, {g0, ple}, "g0m" + suff, shape2(per, n), {layer_a(l)});
    TensorId g1 = b.node(OpKind::MatvecF32Batched, {g0}, "g1" + suff, shape2(d, n),
                         {layer_a(l), slot(WeightSlot::Proj), batch_a(n)});
    TensorId g2 = b.node(OpKind::RmsNormCols, {g1}, "g2" + suff, shape2(d, n),
                         {layer_a(l), slot(WeightSlot::PostNorm), batch_a(n), dim_a(d)});
    TensorId x3 = b.node(OpKind::Add, {x2, g2}, "x3" + suff, shape2(d, n), {layer_a(l)});

    TensorId out;
    if (!m.weights.blocks[l].layer_output_scale.empty())
      out = b.node(OpKind::Scale, {x3}, "out" + suff, shape2(d, n),
                   {attr_f("scale", double(m.weights.blocks[l].layer_output_scale[0]))});
    else
      out = x3;
    residual = out;
  }

  for (auto& t : b.g.tensors)
    if (t.id == residual)
      t.is_graph_output = true;
  b.g.outputs = {residual};

  return std::move(b.g);
}

// ---------------------------------------------------------------------------
// Spine decode loop.
// ---------------------------------------------------------------------------
uint64_t generate_spine(const Gemma4Model& m, uint64_t start_token,
                        uint64_t start_pos, std::span<int64_t> tokens,
                        uint64_t n_ctx) {
  if (m.weights.blocks.empty() || start_token >= m.config.vocab_size) return 0;
  if (n_ctx == 0) n_ctx = start_pos + tokens.size();

  planner::Graph graph = build_gemma4_graph(m, /*greedy=*/false);

  // CPU-only spine: enumerate_cuda = false keeps the snapshot host-only so the
  // planner places every Sonic node on the CPU and the transfer scheduler emits
  // no H2D/D2H tasks. The SonicBackend then reads/writes host bytes directly.
  auto snap = planner::CpuResourceProvider::snapshot({.enumerate_cuda = false});
  if (!snap) return 0;
  planner::AnalyticalCpuCostModel cost(*snap);
  auto plan = planner::plan_execution(graph, *snap, cost);
  if (!plan) {
    std::cerr << "[generate_spine] plan failed: " << plan.error().message << "\n";
    return 0;
  }

  planner::SonicBackend backend(m, graph, n_ctx);
  planner::RuntimeExecutor exec(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, &backend}});

  const uint64_t vocab = m.config.vocab_size;
  uint64_t token = start_token;
  uint64_t pos = start_pos;
  uint64_t step = 0;
  for (; step < tokens.size(); ++step) {
    std::vector<sx::Bytes> in;
    in.push_back(encode_i64(token));
    in.push_back(encode_i64(pos));

    auto res = exec.execute(*plan, *snap, in);
    if (!res || res->outputs.empty()) {
      if (!res)
        std::cerr << "[spine] step " << step << " execute failed: "
                  << res.error().message << "\n";
      else
        std::cerr << "[spine] step " << step << " no outputs\n";
      break;
    }
    const sx::Bytes& logits = res->outputs[0];
    if (logits.size() < vocab * sizeof(float)) {
      std::cerr << "[spine] step " << step << " logits too small: " << logits.size()
                << " < " << vocab * sizeof(float) << "\n";
      break;
    }
    std::span<const float> lf(reinterpret_cast<const float*>(logits.data()), vocab);

    const int64_t next = nn::argmax(lf);
    if (next < 0) break;
    tokens[step] = next;
    token = static_cast<uint64_t>(next);
    ++pos;
  }
  return step;
}

// ---------------------------------------------------------------------------
// Prefill + decode (CPU spine).
// ---------------------------------------------------------------------------
uint64_t generate_prefill_spine(const Gemma4Model& m,
                                std::span<const uint64_t> prompt,
                                uint64_t start_pos, std::span<int64_t> tokens,
                                uint64_t n_ctx, bool use_prefill) {
  if (m.weights.blocks.empty() || prompt.empty() || tokens.empty()) return 0;
  if (n_ctx == 0) n_ctx = start_pos + prompt.size() + tokens.size();

  planner::Graph graph = build_gemma4_graph(m, /*greedy=*/false);
  auto snap = planner::CpuResourceProvider::snapshot({.enumerate_cuda = false});
  if (!snap) return 0;
  planner::AnalyticalCpuCostModel cost(*snap);
  auto plan = planner::plan_execution(graph, *snap, cost);
  if (!plan) return 0;

  planner::SonicBackend backend(m, graph, n_ctx);
  planner::RuntimeExecutor exec(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, &backend}});

  const uint64_t vocab = m.config.vocab_size;
  const uint32_t d = m.config.embedding_length;
  std::vector<float> logits(vocab);

  uint64_t token = 0;
  uint64_t pos = start_pos;
  uint64_t step = 0;

  if (use_prefill) {
    // One batched graph pass over the prompt through the spine, then sample
    // tokens[0] from the final hidden column's lm-head logits (the same lm_head
    // arithmetic the per-token spine's tail reproduces).
    const uint64_t n = prompt.size();
    // The prompt must land in a straight, wrap-free run of each KV cache (SWA
    // ring n_slots == sliding_window, global n_slots == n_ctx).
    for (uint32_t l = 0; l < m.config.n_layer_kv(); ++l) {
      const LayerConfig& lc = m.config.layer_config(l);
      const uint64_t n_slots = (lc.sliding_window > 0) ? lc.sliding_window : n_ctx;
      if (n_slots == 0 || (start_pos % n_slots) + n > n_slots) return 0;
    }

    planner::Graph pgraph = build_gemma4_prefill_graph(m, n);
    auto pplan = planner::plan_execution(pgraph, *snap, cost);
    if (!pplan) return 0;

    // The prefill backend is bound to the prefill graph (for node lookup) but
    // shares the decode backend's KV cache: FlashAttention fills it, and the
    // decode loop below reads it.
    planner::SonicBackend prefill_backend(m, pgraph, n_ctx);
    prefill_backend.share_caches(backend);
    planner::RuntimeExecutor prefill_exec(
        std::map<planner::BackendTag, planner::Backend*>{
            {planner::BackendTag::Sonic, &prefill_backend}});

    std::vector<sx::Bytes> in;
    in.push_back(encode_tokens(prompt));
    in.push_back(encode_i64(start_pos));
    auto res = prefill_exec.execute(*pplan, *snap, in);
    if (!res || res->outputs.empty()) return 0;
    const sx::Bytes& hidden_b = res->outputs[0];
    if (hidden_b.size() < uint64_t(d) * n * sizeof(float)) return 0;
    std::span<const float> hf(reinterpret_cast<const float*>(hidden_b.data()),
                              uint64_t(d) * n);
    std::vector<float> hidden(d);
    for (uint32_t k = 0; k < d; ++k) hidden[k] = hf[k * n + (n - 1)];
    if (!lm_head(m, hidden, logits, quant::MatmulBackend::CpuQ8K)) return 0;
  } else {
    // Reference: feed the prompt one token at a time through the per-token
    // spine, keeping only the final logits.
    for (uint64_t p : prompt) {
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(p));
      in.push_back(encode_i64(pos));
      auto res = exec.execute(*plan, *snap, in);
      if (!res || res->outputs.empty()) return 0;
      const sx::Bytes& lo = res->outputs[0];
      if (lo.size() < vocab * sizeof(float)) return 0;
      std::memcpy(logits.data(), lo.data(), vocab * sizeof(float));
      ++pos;
    }
  }

  const int64_t first = nn::argmax(logits);
  if (first < 0) return 0;
  tokens[0] = first;
  token = static_cast<uint64_t>(first);
  pos = start_pos + prompt.size();

  // Decode the remaining tokens with the same planner/backend/cache.
  for (step = 1; step < tokens.size(); ++step) {
    std::vector<sx::Bytes> in;
    in.push_back(encode_i64(token));
    in.push_back(encode_i64(pos));
    auto res = exec.execute(*plan, *snap, in);
    if (!res || res->outputs.empty()) {
      if (!res)
        std::cerr << "[prefill-spine] step " << step << " execute failed: "
                  << res.error().message << "\n";
      else
        std::cerr << "[prefill-spine] step " << step << " no outputs\n";
      break;
    }
    const sx::Bytes& lo = res->outputs[0];
    if (lo.size() < vocab * sizeof(float)) break;
    std::span<const float> lf(reinterpret_cast<const float*>(lo.data()), vocab);
    const int64_t next = nn::argmax(lf);
    if (next < 0) break;
    tokens[step] = next;
    token = static_cast<uint64_t>(next);
    ++pos;
  }
  return step;
}

// ---------------------------------------------------------------------------
// CUDA spine decode loop. Same token-stream contract as generate_spine, but the
// plan is built against a CUDA-aware snapshot so every Sonic node with a *_dev
// kernel is placed on the GPU (device-local) and only Softcap stays on the CPU
// (it has no *_dev kernel; the greedy graph never emits it). Embedding and
// Argmax also run on the device (Phase 3.2), so the per-token transfers drop to
// a single 4-byte D2H of the winning token index — no embedding H2D uploads and
// no full-logits download. The transfer scheduler inserts H2D/D2H tasks at the
// remaining host/device boundary; the executor registers the CUDA backend as the
// default for Sonic and overrides host-memory-space compute tasks back to the
// CPU SonicBackend. Greedy argmax is identical to generate_spine.
// ---------------------------------------------------------------------------
#ifdef SONICBOOM_USE_CUDA
uint64_t generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                             uint64_t start_pos, std::span<int64_t> tokens,
                             uint64_t n_ctx) {
  if (m.weights.blocks.empty() || start_token >= m.config.vocab_size) return 0;
  if (n_ctx == 0) n_ctx = start_pos + tokens.size();

  const bool gen_timing = std::getenv("SONICBOOM_GEN_TIMING") != nullptr;
  const auto t_plan0 = std::chrono::steady_clock::now();
  planner::Graph graph = build_gemma4_graph(m, /*greedy=*/true);
  // T1 quantize-once: extract the activation quantize from each K-quant gemv
  // into a shared explicit Quantize node (CUDA-only; the CPU reference keeps the
  // unfused QuantizedMatmul path with its q8_K semantics). Gated on q8_1 so the
  // SONICBOOM_GEMV_F32 fp32 fallback stays reachable.
  if (quant::cuda::q8_1_enabled()) {
    planner::quantize_once_pass(graph);
    // T3: fold each rms_norm -> quantize pair into one RmsNormQuantize kernel so
    // the f32 normalized activation is never materialized.
    planner::fuse_rms_norm_quantize_pass(graph);
  }

  auto snap = planner::CpuResourceProvider::snapshot();
  if (!snap) return 0;
  planner::AnalyticalCpuCostModel cost(*snap);
  auto plan = planner::plan_execution(graph, *snap, cost);
  if (!plan) return 0;

  planner::SonicBackend cpu_backend(m, graph);
  planner::SonicCudaBackend cuda_backend(m, graph, n_ctx);
  planner::RuntimeExecutor exec(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, &cuda_backend}});

  // Route host-memory-space compute tasks (Softcap, in sampling mode) to the
  // CPU SonicBackend; device-memory-space tasks keep the default CUDA backend.
  // The transfer scheduler has already emitted the H2D/D2H tasks that move their
  // operands across the boundary.
  for (const auto& task : plan->tasks) {
    if (task.kind != planner::TaskKind::Compute)
      continue;
    const planner::MemorySpace* ms = nullptr;
    if (task.memory_space)
      ms = plan->find_memory_space(*task.memory_space);
    if (!ms || ms->kind == planner::MemoryKind::Host)
      exec.set_task_backend(task.id, &cpu_backend);
  }

  const auto t_loop0 = std::chrono::steady_clock::now();
  auto t_first = t_loop0;
  uint64_t token = start_token;
  uint64_t pos = start_pos;
  uint64_t step = 0;

  // Seed the device-resident token-id slot (Phase 3.2 B4). Argmax overwrites it
  // in place each step, and the Embedding gather reads it back on the device, so
  // the token id never round-trips through the host.
  cuda_backend.seed_token(start_token);

  // --- CUDA Graph capture (HAL record_step) ----------------------------------
  // Replace the ~2600 per-step device-op launches (each a host-side launch call)
  // with a single pre-recorded program (a cudaGraphLaunch inside the CUDA
  // backend's record_step). Opt-out via SONICBOOM_CUDA_GRAPHS=0; any capture
  // failure returns nullptr and falls back to the per-step dispatch path below,
  // so the graph is a pure launch-overhead optimization with no numerics change.
  // The CUDA stream / graph machinery lives behind SonicCudaBackend::record_step;
  // this spine never touches it.
  const bool use_graphs =
      std::getenv("SONICBOOM_CUDA_GRAPHS") == nullptr ||
      std::string_view(std::getenv("SONICBOOM_CUDA_GRAPHS")) != "0";
  std::unique_ptr<hal::StepProgram> step_prog;
  if (use_graphs) {
    // Warm one real step so every device-pool buffer, kernel scratch buffer and
    // weight upload is allocated before capture (cudaMalloc / synchronous
    // cudaMemcpy cannot be recorded into a graph). The capture pass only records.
    {
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(token));
      in.push_back(encode_i64(pos));
      cuda_backend.set_pos(pos);
      auto res = exec.execute(*plan, *snap, in);
      if (res) {
        cuda_backend.synchronize();
        // The warm step's Argmax overwrote the token slot; reset it so the first
        // replay gathers the same start token the warm step did (the warm K/V
        // append at this position is overwritten by the first replay's append).
        cuda_backend.seed_token(start_token);
        step_prog = cuda_backend.record_step([&]() {
          std::vector<sx::Bytes> cin;
          cin.push_back(encode_i64(token));
          cin.push_back(encode_i64(pos));
          auto cres = exec.execute(*plan, *snap, cin);
          return cres.has_value();
        });
      }
    }
  }

  if (step_prog) {
    // Replay the captured step: stage the position, launch the graph, read the
    // winning token back from the persistent device slot. The synchronous D2H
    // read also drains the graph stream before the next step.
    for (; step < tokens.size(); ++step) {
      const std::string step_name = "decode_step_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(step_name.c_str());
      cuda_backend.set_pos(pos);
      if (!step_prog->replay()) {
        if (nvtx_enabled()) nvtxRangePop();
        std::cerr << "[spine] step " << step << " graph replay failed\n";
        break;
      }
      float next_f = -1.0f;
      cuda_backend.read_token(&next_f);
      if (nvtx_enabled()) nvtxRangePop();
      const int64_t next = int64_t(next_f);
      if (next < 0) break;
      tokens[step] = next;
      token = static_cast<uint64_t>(next);
      ++pos;
      if (step == 0) t_first = std::chrono::steady_clock::now();
    }
  } else {
    // Fallback: per-step executor dispatch (the pre-graph decode path).
    for (; step < tokens.size(); ++step) {
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(token));
      in.push_back(encode_i64(pos));

      // Stage the position into the device slot read by RoPE / KV-append /
      // attention (Phase 3.2 B4 / CUDA Graphs): the kernels read it in-kernel.
      cuda_backend.set_pos(pos);

      // NVTX step marker: brackets one decode step so nsys can isolate the
      // steady-state per-token window (instead of dividing whole-run totals).
      const std::string step_name = "decode_step_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(step_name.c_str());
      auto res = exec.execute(*plan, *snap, in);
      if (nvtx_enabled()) nvtxRangePop();
      if (!res || res->outputs.empty()) {
        if (!res)
          std::cerr << "[spine] step " << step << " execute failed: "
                    << res.error().message << "\n";
        else
          std::cerr << "[spine] step " << step << " no outputs\n";
        break;
      }
      const sx::Bytes& arg = res->outputs[0];
      if (arg.size() < sizeof(float)) {
        std::cerr << "[spine] step " << step << " argmax too small: "
                  << arg.size() << " < " << sizeof(float) << "\n";
        break;
      }
      const int64_t next = int64_t(*reinterpret_cast<const float*>(arg.data()));
      if (next < 0) break;
      tokens[step] = next;
      token = static_cast<uint64_t>(next);
      ++pos;
      if (step == 0) t_first = std::chrono::steady_clock::now();
    }
  }

  if (gen_timing) {
    using ms = std::chrono::duration<double, std::milli>;
    const auto t_loop1 = std::chrono::steady_clock::now();
    const double plan_ms = ms(t_loop0 - t_plan0).count();
    const double loop_ms = ms(t_loop1 - t_loop0).count();
    const double first_ms = ms(t_first - t_loop0).count();
    const double rest_ms = loop_ms - first_ms;
    std::cerr << "[cuda-timing] plan " << plan_ms << " ms | decode " << loop_ms
              << " ms / " << step << " steps ("
              << (loop_ms > 0 ? step * 1000.0 / loop_ms : 0.0)
              << " tok/s) | first " << first_ms << " ms | steady "
              << (rest_ms > 0 && step > 1 ? (step - 1) * 1000.0 / rest_ms : 0.0)
              << " tok/s (excl plan+first)\n";
  }
  return step;
}
#else
uint64_t generate_spine_cuda(const Gemma4Model&, uint64_t, uint64_t,
                             std::span<int64_t>, uint64_t) {
  return 0;  // built without CUDA: no device spine is available
}
#endif

// ---------------------------------------------------------------------------
// Prefill + decode (CUDA spine). Same token-stream contract as the CPU
// generate_prefill_spine, but the prefill fills the device-resident KV caches
// (SonicCudaBackend::prefill: batched f32 matmul + flash attention) and the
// decode runs through the CUDA spine loop. lm_head uses the CUDA backend so the
// first-token logits match the device spine's lm-head arithmetic.
// ---------------------------------------------------------------------------
#ifdef SONICBOOM_USE_CUDA
uint64_t generate_prefill_spine_cuda(const Gemma4Model& m,
                                     std::span<const uint64_t> prompt,
                                     uint64_t start_pos, std::span<int64_t> tokens,
                                     uint64_t n_ctx, bool use_prefill) {
  if (m.weights.blocks.empty() || prompt.empty() || tokens.empty()) return 0;
  if (n_ctx == 0) n_ctx = start_pos + prompt.size() + tokens.size();

  planner::Graph graph = build_gemma4_graph(m, /*greedy=*/true);
  // T1 quantize-once (CUDA decode): shared explicit Quantize nodes for the
  // K-quant gemvs, CSE-merged per activation (see generate_spine_cuda). Gated on
  // q8_1 so the SONICBOOM_GEMV_F32 fp32 fallback stays reachable.
  if (quant::cuda::q8_1_enabled()) {
    planner::quantize_once_pass(graph);
    planner::fuse_rms_norm_quantize_pass(graph);
  }

  auto snap = planner::CpuResourceProvider::snapshot();
  if (!snap) return 0;
  planner::AnalyticalCpuCostModel cost(*snap);
  auto plan = planner::plan_execution(graph, *snap, cost);
  if (!plan) return 0;

  planner::SonicBackend cpu_backend(m, graph);
  planner::SonicCudaBackend cuda_backend(m, graph, n_ctx);
  planner::RuntimeExecutor exec(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, &cuda_backend}});

  for (const auto& task : plan->tasks) {
    if (task.kind != planner::TaskKind::Compute)
      continue;
    const planner::MemorySpace* ms = nullptr;
    if (task.memory_space)
      ms = plan->find_memory_space(*task.memory_space);
    if (!ms || ms->kind == planner::MemoryKind::Host)
      exec.set_task_backend(task.id, &cpu_backend);
  }

  const uint64_t vocab = m.config.vocab_size;
  const uint32_t d = m.config.embedding_length;
  std::vector<float> logits(vocab);

  uint64_t token = 0;
  uint64_t pos = start_pos;
  uint64_t step = 0;

  int64_t first = -1;
  if (use_prefill) {
    // One batched pass over the prompt, then sample tokens[0] from the final
    // hidden state's lm-head logits (same arithmetic as the CUDA spine's lm-head).
    std::vector<float> hidden(d);
    if (!cuda_backend.prefill(prompt, start_pos, hidden)) return 0;
    if (!lm_head(m, hidden, logits, quant::MatmulBackend::Cuda)) return 0;
    first = nn::argmax(logits);
  } else {
    // Reference: feed the prompt one token at a time through the CUDA spine
    // (greedy graph -> per-token argmax output), keeping the final token's index.
    // Each prompt token is seeded into the device token slot before its step
    // (Phase 3.2 B4), since the Embedding gather now reads the token id on the
    // device rather than the host graph input.
    float last_argmax = -1.0f;
    for (uint64_t p : prompt) {
      cuda_backend.seed_token(p);
      cuda_backend.set_pos(pos);
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(p));
      in.push_back(encode_i64(pos));
      auto res = exec.execute(*plan, *snap, in);
      if (!res || res->outputs.empty()) return 0;
      const sx::Bytes& lo = res->outputs[0];
      if (lo.size() < sizeof(float)) return 0;
      last_argmax = *reinterpret_cast<const float*>(lo.data());
      ++pos;
    }
    first = int64_t(last_argmax);
  }

  if (first < 0) return 0;
  tokens[0] = first;
  token = static_cast<uint64_t>(first);
  pos = start_pos + prompt.size();
  // The decode loop below reads the next token id from the device slot, which the
  // reference path already left at `first`; the prefill path never ran an
  // Embedding/Argmax step, so seed it explicitly (Phase 3.2 B4).
  cuda_backend.seed_token(token);

  // Decode the remaining tokens with the same planner/backend/cache.
  for (step = 1; step < tokens.size(); ++step) {
    cuda_backend.set_pos(pos);
    std::vector<sx::Bytes> in;
    in.push_back(encode_i64(token));
    in.push_back(encode_i64(pos));
    auto res = exec.execute(*plan, *snap, in);
    if (!res || res->outputs.empty()) {
      if (!res)
        std::cerr << "[prefill-spine-cuda] step " << step << " execute failed: "
                  << res.error().message << "\n";
      else
        std::cerr << "[prefill-spine-cuda] step " << step << " no outputs\n";
      break;
    }
    const sx::Bytes& lo = res->outputs[0];
    if (lo.size() < sizeof(float)) break;
    const int64_t next = int64_t(*reinterpret_cast<const float*>(lo.data()));
    if (next < 0) break;
    tokens[step] = next;
    token = static_cast<uint64_t>(next);
    ++pos;
  }
  return step;
}
#else
uint64_t generate_prefill_spine_cuda(const Gemma4Model&,
                                     std::span<const uint64_t>, uint64_t,
                                     std::span<int64_t>, uint64_t, bool) {
  return 0;  // built without CUDA: no device prefill is available
}
#endif

} // namespace sonicboom::model
