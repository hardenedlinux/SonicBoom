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

#include <sonicboom/nn/sampling.h>
#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/runtime_executor.h>
#include <sonicboom/planner/sonic_backend.h>
#include <sonicboom/planner/sonic_cuda_backend.h>
#include <sonicboom/sx/exec.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sonicboom::model {

namespace {

// Encode a scalar graph input as a little-endian 64-bit buffer (host x86-64).
sx::Bytes encode_i64(uint64_t v) {
  sx::Bytes b(8);
  std::memcpy(b.data(), &v, 8);
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
planner::Graph build_gemma4_graph(const Gemma4Model& m) {
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

  const float proj_scale = 1.0f / std::sqrt(float(d));

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
    TensorId proj = b.node(OpKind::MatvecBf16, {inp}, "proj" + suff, {int64_t(per)},
                           {layer_a(l), slot(WeightSlot::PerLayerModelProj)});
    proj = b.node(OpKind::Scale, {proj}, "projs" + suff, {int64_t(per)},
                  {attr_f("scale", proj_scale)});
    proj = b.node(OpKind::RmsNorm, {proj}, "projn" + suff, {int64_t(per)},
                  {layer_a(l), slot(WeightSlot::PerLayerProjNorm)});
    TensorId ple = b.node(OpKind::LayerCombine, {proj, ple_row}, "ple" + suff,
                          {int64_t(per)}, {layer_a(l)});
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
  TensorId logits_sc = b.node(OpKind::Softcap, {logits}, "logits_sc", {int64_t(vocab)},
                              {attr_f("scale", double(cfg.final_logit_softcapping))});

  for (auto& t : b.g.tensors)
    if (t.id == logits_sc)
      t.is_graph_output = true;
  b.g.outputs = {logits_sc};

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

  planner::Graph graph = build_gemma4_graph(m);

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
// CUDA spine decode loop. Same token-stream contract as generate_spine, but the
// plan is built against a CUDA-aware snapshot so every Sonic node with a *_dev
// kernel is placed on the GPU (device-local) and only Embedding/Softcap stay on
// the CPU (they have no *_dev kernel). The transfer scheduler inserts H2D/D2H
// tasks at the three host/device boundaries; the executor registers the CUDA
// backend as the default for Sonic and overrides host-memory-space compute tasks
// back to the CPU SonicBackend. Greedy argmax is identical to generate_spine.
// ---------------------------------------------------------------------------
#ifdef SONICBOOM_USE_CUDA
uint64_t generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                             uint64_t start_pos, std::span<int64_t> tokens,
                             uint64_t n_ctx) {
  if (m.weights.blocks.empty() || start_token >= m.config.vocab_size) return 0;
  if (n_ctx == 0) n_ctx = start_pos + tokens.size();

  planner::Graph graph = build_gemma4_graph(m);

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

  // Route host-memory-space compute tasks (Embedding, Softcap) to the CPU
  // SonicBackend; device-memory-space tasks keep the default CUDA backend. The
  // transfer scheduler has already emitted the H2D/D2H tasks that move their
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
#else
uint64_t generate_spine_cuda(const Gemma4Model&, uint64_t, uint64_t,
                             std::span<int64_t>, uint64_t) {
  return 0;  // built without CUDA: no device spine is available
}
#endif

} // namespace sonicboom::model
