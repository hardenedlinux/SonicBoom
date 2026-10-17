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

#include <sonicboom/compiler/BoomBuilder.h>
#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomPasses.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/ir_passes.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/runtime_executor.h>
#include <sonicboom/planner/sonic_backend.h>
#include <sonicboom/planner/sonic_cuda_backend.h>
#include <sonicboom/sx/exec.h>

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>  // cudaGetDeviceProperties (record device line)
#include <nvToolsExt.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <list>
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
// prefill graph's `tokens` Int64[n] input, decoded back by the batched
// Embedding node).
sx::Bytes encode_tokens(std::span<const uint64_t> tokens) {
  sx::Bytes b(tokens.size() * sizeof(uint64_t));
  std::memcpy(b.data(), tokens.data(), b.size());
  return b;
}

sx::Attribute attr_i(const char* name, int64_t v) { return sx::Attribute{name, v}; }
sx::Attribute attr_f(const char* name, double v) { return sx::Attribute{name, v}; }

// Device-prefill segment size for a model with no sliding window. The chunk
// exists to keep a prefill segment a wrap-free run inside the KV ring; with no
// window there is no ring to stay inside, so any positive value is correct and
// the choice is a pure capacity/launch-geometry one (it only bounds the prefill
// graph's token buffer). 512 matches the segment size the windowed models use.
constexpr uint64_t kDefaultPrefillChunk = 512;

// Which generic pieces of the transformer block an architecture uses. This is
// the ONE architecture-conditional in SonicBoom: the emitter (together with the
// loader) is the model-specific front end, so an architecture's structural
// differences are expressed here and nowhere else — the planner, the
// partitioning seam, and both backends stay model-agnostic, and every field
// below selects between existing generic OpKinds/attributes rather than
// introducing a model-specific one.
//
// Gemma 4 turns every field on except the projection biases and the untied
// output; Qwen2 turns on the projection bias, the SiLU gate, the attention
// temperature, and the untied output, and leaves the rest off.
struct BlockProfile {
  bool embd_scale = false;     // token embedding × sqrt(d)
  bool qk_norm = false;        // per-head RMSNorm on Q and K
  bool v_norm = false;         // unit-weight per-head RMSNorm on V
  bool proj_bias = false;      // Q/K/V projections carry a bias
  bool attn_out_bias = false;  // attention output projection carries a bias
  bool post_attn_norm = false; // norm between the attn projection and its residual
  bool post_ffw_norm = false;  // norm between the FFN down-projection and its residual
  bool silu_ffn = false;       // SiLU (Qwen2) vs fp16 GELU (Gemma 4) gate
  bool per_layer_input = false; // per-layer embedding gate + its whole block
  bool layer_scale = false;    // whole-block output scale
  bool kv_shared = false;      // some layers attend over another layer's KV cache
  bool softcap = false;        // logit softcapping
  bool tied_output = false;    // lm head reuses the token embedding
  float attn_scale = 1.0f;     // logit temperature (1.0 with Q/K RMSNorm)
};

BlockProfile profile_for(const Gemma4Config& cfg) {
  BlockProfile p;
  switch (cfg.arch) {
    case ModelArch::Gemma4:
      p.embd_scale = true;
      p.qk_norm = true;
      p.v_norm = true;
      p.post_attn_norm = true;
      p.post_ffw_norm = true;
      p.per_layer_input = cfg.per_layer_input > 0;
      p.layer_scale = true;
      p.kv_shared = cfg.shared_kv_layers > 0;
      p.softcap = cfg.final_logit_softcapping > 0.0f;
      p.tied_output = true;
      p.attn_scale = 1.0f;  // Q/K are unit-normed, so the dot product is a cosine
      break;
    case ModelArch::Qwen2:
      p.proj_bias = true;
      p.silu_ffn = true;
      p.tied_output = false;
      p.attn_scale = 1.0f / std::sqrt(float(cfg.head_dim_global));
      break;
  }
  return p;
}

} // namespace

// ---------------------------------------------------------------------------
// Model-agnostic KV-cache geometry (Section C). Extracts the cache-relevant
// facts from the Gemma 4 config into a planner::KvCacheSpec, keeping the
// Gemma-specific per-layer mapping (layer_config / kv_donor_layer) inside the
// model layer. The CUDA backend then allocates and indexes its cache from the
// spec alone.
// ---------------------------------------------------------------------------
planner::KvCacheSpec kv_cache_spec(const Gemma4Config& cfg) {
  planner::KvCacheSpec spec;
  spec.n_layers = cfg.n_layer_kv();
  // Max prefill segment. A windowed model uses the ring size (so a segment never
  // wraps the ring); a window-less model has no ring constraint, so it gets the
  // fixed default segment instead of 0 — the backend sizes its prefill token
  // buffer from this and rejects a segment larger than it.
  spec.chunk = (cfg.sliding_window > 0) ? cfg.sliding_window : kDefaultPrefillChunk;
  spec.layers.reserve(spec.n_layers);
  for (uint32_t l = 0; l < spec.n_layers; ++l) {
    const LayerConfig& lc = cfg.layer_config(l);
    planner::KvCacheSpec::Layer e;
    e.kv_heads = lc.n_heads_kv;
    e.head_dim = lc.head_dim;
    e.window = lc.sliding_window;
    spec.layers.push_back(e);
  }
  return spec;
}

// ---------------------------------------------------------------------------
// The emitter's IR, and the one lowering.
//
// An emitter writes Boom IR (compiler/BoomBuilder.h) and returns the module it
// lives in; the phase's passes rewrite that IR (compiler/BoomPasses.h); the
// lowering below is the only place a planner::Graph is produced from it, and it
// exists because the plan builder's input format is a graph, not a second
// construction of the graph the emitter meant. Nothing between the emitter and
// the lowering names a Graph.
// ---------------------------------------------------------------------------
namespace {

struct GraphIr {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

// Boom IR -> the graph the plan is derived from. A lowering that fails is
// reported and leaves an empty graph, which fails the plan — there is
// deliberately no fallback that would plan something else.
planner::Graph lower_graph(mlir::ModuleOp module) {
  auto graph = compiler::boomToGraph(module);
  if (!graph) {
    std::cerr << "[plan] Boom IR -> graph failed (diagnostic above)\n";
    return {};
  }
  return std::move(*graph);
}

// What every plan site runs: the emitter's IR, the phase's passes, the
// lowering. `rewrite` is the caller's condition — whether the quantized path is
// available at all is a property of the device, not of the pipeline, so the
// caller keeps it. A failed pass fails the plan rather than letting an unfused
// graph be planned as if the phase had run.
planner::Graph lower_graph(GraphIr& ir, planner::IrPassPhase phase, bool rewrite) {
  if (!ir.module)
    return {};
  if (rewrite && !compiler::runBoomPasses(*ir.module, phase))
    return {};
  return lower_graph(*ir.module);
}

}  // namespace

// ---------------------------------------------------------------------------
// Plan emitter. Mirrors forward()/lm_head in exec.cpp — the same per-stage
// arithmetic, re-expressed as a static acyclic graph of transformer OpKind
// nodes. The graph is stateless (acyclic): the multi-token KV cache lives in the
// SonicBackend, keyed by layer. The `Attention` node appends its RoPE'd fp16 K
// and normed fp16 V to the layer's cache and attends over the accumulated keys;
// a shared-KV layer (24..41) omits the K/V inputs and attends over its donor
// layer's (22 SWA / 23 global) cache, selected by the node's `kv_source_layer`
// attribute. The residual stream is the *scaled* l_out of the previous block.
// ---------------------------------------------------------------------------
GraphIr emit_decode_ir(const Gemma4Model& m, bool greedy) {
  using namespace planner;

  const auto& cfg = m.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const uint32_t block_count = cfg.block_count;
  const uint32_t total_per = block_count * per;
  const uint64_t vocab = cfg.vocab_size;

  GraphIr ir;
  ir.context = std::make_unique<mlir::MLIRContext>();
  compiler::BoomBuilder b(ir.context.get(), "gemma4_decode");

  // Graph inputs: token id + absolute position (Int64 scalars), declared to the
  // builder — they are the graph's free variables, and a free variable is a block
  // argument in the IR the emitter writes.
  const mlir::Value token_id = b.input("token_id", {}, sx::DType::Int64);
  const mlir::Value pos = b.input("pos", {}, sx::DType::Int64);

  const BlockProfile prof = profile_for(cfg);

  auto slot = [](WeightSlot s) { return attr_i("slot", int64_t(s)); };
  auto layer_a = [](uint32_t l) { return attr_i("layer", int64_t(l)); };
  auto kv_src = [&cfg](uint32_t l) {
    return attr_i("kv_source_layer", int64_t(cfg.kv_donor_layer(l)));
  };
  // The attention temperature is emitted only when it is not the identity, so a
  // model whose Q/K are unit-normed (Gemma 4) gets a byte-identical graph to the
  // one before the generalization.
  auto attn_attrs = [&](uint32_t l) {
    std::vector<sx::Attribute> a = {layer_a(l), kv_src(l)};
    if (prof.attn_scale != 1.0f) a.push_back(attr_f("scale", double(prof.attn_scale)));
    return a;
  };

  // Preamble: the shared token embedding (scaled where the architecture scales
  // it), and — only for a model with per-layer inputs — the full per-layer token
  // embedding row (dequantized once; each block slices its `per`-element column
  // out of the combined `ple_all`). Both are deterministic, so hoisting them out
  // of the block loop is bit-identical to exec.cpp recomputing them per block.
  const mlir::Value embd = b.node(OpKind::Embedding, {token_id}, "embd", {int64_t(d)},
                               {slot(WeightSlot::TokenEmbd)});
  const mlir::Value inp = prof.embd_scale
                           ? b.node(OpKind::Scale, {embd}, "inp", {int64_t(d)},
                                    {attr_f("scale", std::sqrt(float(d)))})
                           : embd;

  // Hoisted per-layer input (Phase 2.3a): one full-width 2560→10752 bf16 gemv +
  // scale + segmented rms_norm + combine per token, done once over [total_per].
  // Each block's gate then slices its [per] segment instead of recomputing the
  // 2560→256 projection. Decomposed (C.5) into the general ops below: a
  // full-width MatvecBf16 (offset 0, rows total_per) + Scale(1/sqrt(d)) +
  // RmsNormHeads(dim == per, weight == per_layer_proj_norm) + the combine
  // (proj + ple*sqrt(per)) / sqrt(2) as Scale/Add/Scale. A model without
  // per-layer inputs (Qwen2) emits none of this and runs the block alone.
  mlir::Value ple_all{};
  if (prof.per_layer_input) {
    const mlir::Value ple_row = b.node(OpKind::Embedding, {token_id}, "ple_row",
                                    {int64_t(total_per)},
                                    {slot(WeightSlot::PerLayerTokenEmbd)});
    const float proj_scale = 1.0f / std::sqrt(float(d));
    const float ple_scale = std::sqrt(float(per));
    const float combine_scale = 1.0f / std::sqrt(2.0f);
    ple_all = b.node(OpKind::MatvecBf16, {inp}, "ple_proj",
                     {int64_t(total_per)},
                     {attr_i("offset", int64_t(0)),
                      attr_i("rows", int64_t(total_per))});
    ple_all = b.node(OpKind::Scale, {ple_all}, "ple_projs", {int64_t(total_per)},
                     {attr_f("scale", double(proj_scale))});
    ple_all = b.node(OpKind::RmsNormHeads, {ple_all}, "ple_projn",
                     {int64_t(total_per)},
                     {layer_a(0), slot(WeightSlot::PerLayerProjNorm),
                      attr_i("dim", int64_t(per))});
    const mlir::Value ple_s = b.node(OpKind::Scale, {ple_row}, "ple_s",
                                  {int64_t(total_per)},
                                  {attr_f("scale", double(ple_scale))});
    const mlir::Value ple_add = b.node(OpKind::Add, {ple_all, ple_s}, "ple_add",
                                    {int64_t(total_per)}, {});
    ple_all = b.node(OpKind::Scale, {ple_add}, "ple_all", {int64_t(total_per)},
                     {attr_f("scale", double(combine_scale))});
  }

  mlir::Value residual = inp;
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
    mlir::Value h = b.node(OpKind::RmsNorm, {residual}, "h" + suff, {int64_t(d)},
                        {layer_a(l), slot(WeightSlot::AttnNorm)});
    mlir::Value q = b.node(OpKind::QuantizedMatmul, {h}, "q" + suff, {int64_t(qw)},
                        {layer_a(l), slot(WeightSlot::AttnQ)});
    if (prof.proj_bias)  // bias is added before RoPE (llama.cpp build_qkv order)
      q = b.node(OpKind::AddBias, {q}, "qb" + suff, {int64_t(qw)},
                 {layer_a(l), slot(WeightSlot::AttnQBias)});
    if (prof.qk_norm)
      q = b.node(OpKind::RmsNormHeads, {q}, "qn" + suff, {int64_t(qw)},
                 {layer_a(l), slot(WeightSlot::AttnQNorm)});
    q = b.node(OpKind::Rope, {q, pos}, "qr" + suff, {int64_t(qw)}, {layer_a(l)});

    mlir::Value attn;
    if (cfg.has_kv(l)) {
      mlir::Value k = b.node(OpKind::QuantizedMatmul, {h}, "k" + suff, {int64_t(kvw)},
                          {layer_a(l), slot(WeightSlot::AttnK)});
      mlir::Value v = b.node(OpKind::QuantizedMatmul, {h}, "v" + suff, {int64_t(kvw)},
                          {layer_a(l), slot(WeightSlot::AttnV)});
      if (prof.proj_bias) {
        k = b.node(OpKind::AddBias, {k}, "kb" + suff, {int64_t(kvw)},
                   {layer_a(l), slot(WeightSlot::AttnKBias)});
        v = b.node(OpKind::AddBias, {v}, "vb" + suff, {int64_t(kvw)},
                   {layer_a(l), slot(WeightSlot::AttnVBias)});
      }
      if (prof.qk_norm)
        k = b.node(OpKind::RmsNormHeads, {k}, "kn" + suff, {int64_t(kvw)},
                   {layer_a(l), slot(WeightSlot::AttnKNorm)});
      if (prof.v_norm)
        v = b.node(OpKind::RmsNormHeads, {v}, "vn" + suff, {int64_t(kvw)},
                   {layer_a(l)});  // V norm: unit weight
      const mlir::Value vc = b.node(OpKind::CastFp16, {v}, "vc" + suff, {int64_t(kvw)},
                                 {layer_a(l)});
      k = b.node(OpKind::Rope, {k, pos}, "kr" + suff, {int64_t(kvw)}, {layer_a(l)});
      const mlir::Value kc = b.node(OpKind::CastFp16, {k}, "kc" + suff, {int64_t(kvw)},
                                 {layer_a(l)});
      attn = b.node(OpKind::Attention, {q, kc, vc, pos}, "attn" + suff,
                    {int64_t(qw)}, attn_attrs(l));
    } else {
      attn = b.node(OpKind::Attention, {q, pos}, "attn" + suff,
                    {int64_t(qw)}, attn_attrs(l));
    }

    mlir::Value ap = b.node(OpKind::QuantizedMatmul, {attn}, "ap" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::AttnOutput)});
    if (prof.attn_out_bias)
      ap = b.node(OpKind::AddBias, {ap}, "apb" + suff, {int64_t(d)},
                  {layer_a(l), slot(WeightSlot::AttnOutputBias)});
    mlir::Value o = prof.post_attn_norm
                     ? b.node(OpKind::RmsNorm, {ap}, "o" + suff, {int64_t(d)},
                              {layer_a(l), slot(WeightSlot::PostAttentionNorm)})
                     : ap;
    mlir::Value x1 = b.node(OpKind::Add, {o, residual}, "x1" + suff, {int64_t(d)},
                         {layer_a(l)});

    // --- gated feed-forward ------------------------------------------------
    mlir::Value f = b.node(OpKind::RmsNorm, {x1}, "f" + suff, {int64_t(d)},
                        {layer_a(l), slot(WeightSlot::FfnNorm)});
    mlir::Value up = b.node(OpKind::QuantizedMatmul, {f}, "up" + suff, {int64_t(ff)},
                         {layer_a(l), slot(WeightSlot::FfnUp)});
    mlir::Value gate = b.node(OpKind::QuantizedMatmul, {f}, "gate" + suff, {int64_t(ff)},
                           {layer_a(l), slot(WeightSlot::FfnGate)});
    gate = prof.silu_ffn
               ? b.node(OpKind::Silu, {gate}, "silu" + suff, {int64_t(ff)}, {layer_a(l)})
               : b.node(OpKind::GeluFp16, {gate}, "gelu" + suff, {int64_t(ff)},
                        {layer_a(l)});
    mlir::Value fh = b.node(OpKind::Mul, {gate, up}, "fh" + suff, {int64_t(ff)},
                         {layer_a(l)});
    mlir::Value fo = b.node(OpKind::QuantizedMatmul, {fh}, "fo" + suff, {int64_t(d)},
                         {layer_a(l), slot(WeightSlot::FfnDown)});
    mlir::Value f2 = prof.post_ffw_norm
                      ? b.node(OpKind::RmsNorm, {fo}, "f2" + suff, {int64_t(d)},
                               {layer_a(l), slot(WeightSlot::PostFfwNorm)})
                      : fo;
    mlir::Value x2 = b.node(OpKind::Add, {f2, x1}, "x2" + suff, {int64_t(d)},
                         {layer_a(l)});

    // --- per-layer gate ----------------------------------------------------
    // The per-layer input (proj/scale/rms/combine) was hoisted to the preamble;
    // each block slices its [per] segment of ple_all. A model without per-layer
    // inputs (Qwen2) ends the block here.
    mlir::Value x3 = x2;
    if (prof.per_layer_input) {
      mlir::Value ple = b.node(OpKind::Slice, {ple_all}, "ple" + suff, {int64_t(per)},
                            {attr_i("offset", int64_t(l) * per),
                             attr_i("length", int64_t(per))});
      mlir::Value g0 = b.node(OpKind::MatvecF32, {x2}, "g0" + suff, {int64_t(per)},
                           {layer_a(l), slot(WeightSlot::InpGate)});
      g0 = b.node(OpKind::GeluFp16, {g0}, "g0g" + suff, {int64_t(per)}, {layer_a(l)});
      g0 = b.node(OpKind::Mul, {g0, ple}, "g0m" + suff, {int64_t(per)}, {layer_a(l)});
      mlir::Value g1 = b.node(OpKind::MatvecF32, {g0}, "g1" + suff, {int64_t(d)},
                           {layer_a(l), slot(WeightSlot::Proj)});
      mlir::Value g2 = b.node(OpKind::RmsNorm, {g1}, "g2" + suff, {int64_t(d)},
                           {layer_a(l), slot(WeightSlot::PostNorm)});
      x3 = b.node(OpKind::Add, {x2, g2}, "x3" + suff, {int64_t(d)},
                  {layer_a(l)});
    }

    // --- whole-block output scale ------------------------------------------
    mlir::Value out;
    if (prof.layer_scale && !m.weights.blocks[l].layer_output_scale.empty())
      out = b.node(OpKind::Scale, {x3}, "out" + suff, {int64_t(d)},
                   {attr_f("scale", double(m.weights.blocks[l].layer_output_scale[0]))});
    else
      out = x3;
    residual = out;
  }

  // --- lm head -------------------------------------------------------------
  mlir::Value hn = b.node(OpKind::RmsNorm, {residual}, "result_norm", {int64_t(d)},
                       {slot(WeightSlot::OutputNorm)});
  mlir::Value logits = b.node(OpKind::QuantizedMatmul, {hn}, "logits", {int64_t(vocab)},
                           {slot(prof.tied_output ? WeightSlot::TokenEmbd
                                                  : WeightSlot::Output)});

  // Greedy decode: softcapping (sc * tanh(x/sc)) is monotonic so it does not
  // change argmax — skip it and reduce the [vocab] logits to a single index on
  // the device, so the per-token host transfer shrinks from the full 1 MiB
  // logits to 4 bytes. Sampling mode keeps the softcapped full logits (the
  // sampler needs the distribution).
  mlir::Value out;
  if (greedy) {
    out = b.node(OpKind::Argmax, {logits}, "argmax", {int64_t(1)}, {});
  } else if (prof.softcap) {
    out = b.node(OpKind::Softcap, {logits}, "logits_sc", {int64_t(vocab)},
                 {attr_f("scale", double(cfg.final_logit_softcapping))});
  } else {
    out = logits;  // no softcapping: the raw logits are the graph output
  }

  b.set_output(out);

  ir.module = b.finish();
  return ir;
}

// The emitter's IR, lowered with no passes: the graph as the emitter wrote it,
// which is what a caller that runs the pipeline itself — or records what the
// emitter wrote — wants (model/plan.h).
planner::Graph build_gemma4_graph(const Gemma4Model& m, bool greedy) {
  GraphIr ir = emit_decode_ir(m, greedy);
  if (!ir.module)
    return {};
  return lower_graph(*ir.module);
}

// ---------------------------------------------------------------------------
// Prefill graph emitter (Phase 7 / Section C.3). Mirrors build_gemma4_graph per
// block, but on [n, dim] position-major batched tensors (n == the prompt
// length, dim innermost) so the whole prompt is processed in one graph pass
// through the spine. The batch size n and per-node `dim` are baked into node
// attributes; the per-token OpKinds (Embedding, QuantizedMatmul, RmsNorm,
// RmsNormHeads, Rope, MatvecF32, MatvecBf16, Slice) carry the `batch`
// attribute and the SonicBackend loops their per-token kernel over the batch
// dim, while the elementwise nodes (Scale/Add/Mul/GeluFp16/CastFp16) are reused
// unchanged on the flat batched buffers. C.3 removed the [dim, n] column-major
// layout and its Transpose nodes: every batched matmul/gather/norm writes
// [n, dim] directly, so the attention path (which always consumed [n, dim]) no
// longer needs a data-movement transpose on either side of it. FlashAttention
// runs full-sequence causal SDPA over prior cache + the current chunk's n
// fp16-rounded K/V rows (no append — the deferred KvAppend node does that); a
// shared-KV layer passes its donor's fp16-rounded K/V graph tensors instead of
// its own, and the node's `kv_source_layer` attribute selects the donor cache,
// so the memory planner keeps the donor K/V alive across layers 24..41 via
// ordinary tensor liveness. Graph inputs are `tokens` (Int64 [n]) and
// `start_pos` (Int64); the graph output is the final residual [n, dim].
// ---------------------------------------------------------------------------
GraphIr emit_prefill_ir(const Gemma4Model& m, uint64_t n) {
  using namespace planner;

  const auto& cfg = m.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const uint32_t block_count = cfg.block_count;
  const uint32_t total_per = block_count * per;

  GraphIr ir;
  ir.context = std::make_unique<mlir::MLIRContext>();
  compiler::BoomBuilder b(ir.context.get(), "gemma4_prefill");

  // Graph inputs: the prompt token ids (Int64 [n]) and the absolute start
  // position (Int64 scalar).
  const mlir::Value tokens = b.input("tokens", {int64_t(n)}, sx::DType::Int64);
  const mlir::Value start_pos = b.input("start_pos", {}, sx::DType::Int64);

  auto slot = [](WeightSlot s) { return attr_i("slot", int64_t(s)); };
  auto layer_a = [](uint32_t l) { return attr_i("layer", int64_t(l)); };
  auto kv_src = [&cfg](uint32_t l) {
    return attr_i("kv_source_layer", int64_t(cfg.kv_donor_layer(l)));
  };
  auto batch_a = [](uint64_t v) { return attr_i("batch", int64_t(v)); };
  auto dim_a = [](uint64_t v) { return attr_i("dim", int64_t(v)); };
  // The fused mmq quantize's activation scale layout follows the consuming
  // matmul's weight type: Q3_K and the legacy 32-element formats Q5_0/Q8_0 ->
  // ggml D4 (float scales), Q4_K/Q5_K -> DS4 (half2 scales). Every format the
  // mmq path executes therefore has a layout, so the C.4 planner passes may
  // fold a quantize into any of them.
  // Q6_K gets -1: its mmq weight kernel is disabled (TODO(q6k-mmq)), so it has
  // no mmq activation layout and must never be folded into a fused emitter —
  // -1 is the "no layout" marker the C.4 passes skip on (see
  // fuse_act_mul_mmq_pass / fuse_rms_norm_mmq_pass), which keeps the down
  // projection on its dequant + cuBLAS route rather than quantizing its
  // activation to int8.
  // `ds` answers one question only: does this weight type have an mmq activation
  // layout the fused emitter can produce? Q6_K does not (above). Every other
  // format does, including the legacy 32-element ones, whose D4 norm+quantize
  // emitter is now bandwidth-bound at any d_model (its quantize phase uses the
  // standalone quantizer's 128-thread/one-128-block-per-warp geometry — see
  // rms_norm_quantize_mmq_q8_1_dev). It used to be keyed out by quant type as a
  // proxy for "slow at small dim"; that was a property of the old emitter, not
  // of the format, so the decision belongs to the emitter, not here.
  // Baked here (not in the planner pass) because only the emitter knows the
  // weight type; the C.4 planner pass reads `ds` to decide fusion eligibility.
  auto ds_a = [](quant::QuantType qt) {
    const int64_t ds = (qt == quant::QuantType::Q4_K || qt == quant::QuantType::Q5_K) ? 1
                       : (qt == quant::QuantType::Q6_K) ? -1 : 0;
    return attr_i("ds", ds);
  };
  auto shape2 = [](uint64_t r, uint64_t c) {
    return std::vector<int64_t>{int64_t(r), int64_t(c)};
  };

  const BlockProfile prof = profile_for(cfg);
  // The attention temperature is emitted only when it is not the identity (see
  // build_gemma4_graph).
  auto attn_attrs = [&](uint32_t l) {
    std::vector<sx::Attribute> a = {layer_a(l), batch_a(n), kv_src(l)};
    if (prof.attn_scale != 1.0f) a.push_back(attr_f("scale", double(prof.attn_scale)));
    return a;
  };

  const mlir::Value embd = b.node(OpKind::Embedding, {tokens}, "embd",
                               shape2(n, d), {slot(WeightSlot::TokenEmbd), batch_a(n)});
  const mlir::Value inp = prof.embd_scale
                           ? b.node(OpKind::Scale, {embd}, "inp", shape2(n, d),
                                    {attr_f("scale", std::sqrt(float(d)))})
                           : embd;
  // Only a model with per-layer inputs (Gemma 4) reads the per-layer embedding
  // row; the whole gate — including this gather — is skipped otherwise.
  mlir::Value ple_row{};
  if (prof.per_layer_input)
    ple_row = b.node(OpKind::Embedding, {tokens}, "ple_row", shape2(n, total_per),
                     {slot(WeightSlot::PerLayerTokenEmbd), batch_a(n)});

  const float proj_scale = 1.0f / std::sqrt(float(d));
  const uint32_t n_kv = cfg.n_layer_kv();
  const uint32_t donor_swa_layer = n_kv - 2;
  const uint32_t donor_global_layer = n_kv - 1;
  mlir::Value donor_swa_k{}, donor_swa_v{}, donor_global_k{}, donor_global_v{};
  // The fp16-rounded K/V of each KV layer, kept alive past the layer loop so the
  // deferred cache append can run after every layer (see the KvAppend nodes
  // emitted below).
  std::vector<mlir::Value> kv_k(n_kv), kv_v(n_kv);

  mlir::Value residual = inp;
  for (uint32_t l = 0; l < block_count; ++l) {
    const LayerConfig& lc = cfg.layer_config(l);
    const uint32_t n_q = lc.n_heads_q;
    const uint32_t n_kvh = lc.n_heads_kv;
    const uint32_t hd = lc.head_dim;
    const uint32_t qw = n_q * hd;
    const uint32_t kvw = n_kvh * hd;
    const std::string suff = std::to_string(l);

    // --- attention ---------------------------------------------------------
    mlir::Value h = b.node(OpKind::RmsNorm, {residual}, "h" + suff, shape2(n, d),
                        {layer_a(l), slot(WeightSlot::AttnNorm), batch_a(n), dim_a(d)});
    mlir::Value q = b.node(OpKind::QuantizedMatmul, {h}, "q" + suff,
                        shape2(n, qw), {layer_a(l), slot(WeightSlot::AttnQ), batch_a(n),
                        dim_a(d), ds_a(m.weights.blocks[l].attn_q.type())});
    if (prof.proj_bias)
      q = b.node(OpKind::AddBias, {q}, "qb" + suff, shape2(n, qw),
                 {layer_a(l), slot(WeightSlot::AttnQBias), batch_a(n)});
    if (prof.qk_norm)
      q = b.node(OpKind::RmsNormHeads, {q}, "qn" + suff, shape2(n, qw),
                 {layer_a(l), slot(WeightSlot::AttnQNorm), batch_a(n)});
    q = b.node(OpKind::Rope, {q, start_pos}, "qr" + suff, shape2(n, qw),
               {layer_a(l), batch_a(n)});

    mlir::Value attn;
    if (cfg.has_kv(l)) {
      mlir::Value k = b.node(OpKind::QuantizedMatmul, {h}, "k" + suff,
                          shape2(n, kvw), {layer_a(l), slot(WeightSlot::AttnK), batch_a(n),
                          dim_a(d), ds_a(m.weights.blocks[l].attn_k.type())});
      mlir::Value v = b.node(OpKind::QuantizedMatmul, {h}, "v" + suff,
                          shape2(n, kvw), {layer_a(l), slot(WeightSlot::AttnV), batch_a(n),
                          dim_a(d), ds_a(m.weights.blocks[l].attn_v.type())});
      if (prof.proj_bias) {
        k = b.node(OpKind::AddBias, {k}, "kb" + suff, shape2(n, kvw),
                   {layer_a(l), slot(WeightSlot::AttnKBias), batch_a(n)});
        v = b.node(OpKind::AddBias, {v}, "vb" + suff, shape2(n, kvw),
                   {layer_a(l), slot(WeightSlot::AttnVBias), batch_a(n)});
      }
      if (prof.qk_norm)
        k = b.node(OpKind::RmsNormHeads, {k}, "kn" + suff, shape2(n, kvw),
                   {layer_a(l), slot(WeightSlot::AttnKNorm), batch_a(n)});
      if (prof.v_norm)
        v = b.node(OpKind::RmsNormHeads, {v}, "vn" + suff, shape2(n, kvw),
                   {layer_a(l), batch_a(n)});  // V norm: unit weight
      const mlir::Value vc = b.node(OpKind::CastFp16, {v}, "vc" + suff, shape2(n, kvw),
                                 {layer_a(l)});
      k = b.node(OpKind::Rope, {k, start_pos}, "kr" + suff, shape2(n, kvw),
                 {layer_a(l), batch_a(n)});
      const mlir::Value kc = b.node(OpKind::CastFp16, {k}, "kc" + suff, shape2(n, kvw),
                                 {layer_a(l)});
      attn = b.node(OpKind::FlashAttention, {q, kc, vc, start_pos}, "attn" + suff,
                    shape2(n, qw), attn_attrs(l));
      kv_k[l] = kc;
      kv_v[l] = vc;
      if (l == donor_swa_layer) {
        donor_swa_k = kc;
        donor_swa_v = vc;
      } else if (l == donor_global_layer) {
        donor_global_k = kc;
        donor_global_v = vc;
      }
    } else {
      const uint32_t donor = cfg.kv_donor_layer(l);
      const mlir::Value dk = (donor == donor_swa_layer) ? donor_swa_k : donor_global_k;
      const mlir::Value dv = (donor == donor_swa_layer) ? donor_swa_v : donor_global_v;
      attn = b.node(OpKind::FlashAttention, {q, dk, dv, start_pos}, "attn" + suff,
                    shape2(n, qw), attn_attrs(l));
    }

    mlir::Value ap = b.node(OpKind::QuantizedMatmul, {attn}, "ap" + suff,
                         shape2(n, d), {layer_a(l), slot(WeightSlot::AttnOutput), batch_a(n),
                         dim_a(qw), ds_a(m.weights.blocks[l].attn_output.type())});
    if (prof.attn_out_bias)
      ap = b.node(OpKind::AddBias, {ap}, "apb" + suff, shape2(n, d),
                  {layer_a(l), slot(WeightSlot::AttnOutputBias), batch_a(n)});
    mlir::Value o = prof.post_attn_norm
                     ? b.node(OpKind::RmsNorm, {ap}, "o" + suff, shape2(n, d),
                              {layer_a(l), slot(WeightSlot::PostAttentionNorm), batch_a(n),
                               dim_a(d)})
                     : ap;
    mlir::Value x1 = b.node(OpKind::Add, {o, residual}, "x1" + suff, shape2(n, d), {layer_a(l)});

    // --- gated feed-forward ------------------------------------------------
    mlir::Value f = b.node(OpKind::RmsNorm, {x1}, "f" + suff, shape2(n, d),
                        {layer_a(l), slot(WeightSlot::FfnNorm), batch_a(n), dim_a(d)});
    mlir::Value up = b.node(OpKind::QuantizedMatmul, {f}, "up" + suff, shape2(n, ff),
                         {layer_a(l), slot(WeightSlot::FfnUp), batch_a(n),
                         dim_a(d), ds_a(m.weights.blocks[l].ffn_up.type())});
    mlir::Value gate = b.node(OpKind::QuantizedMatmul, {f}, "gate" + suff, shape2(n, ff),
                           {layer_a(l), slot(WeightSlot::FfnGate), batch_a(n),
                           dim_a(d), ds_a(m.weights.blocks[l].ffn_gate.type())});
    gate = prof.silu_ffn
               ? b.node(OpKind::Silu, {gate}, "silu" + suff, shape2(n, ff), {layer_a(l)})
               : b.node(OpKind::GeluFp16, {gate}, "gelu" + suff, shape2(n, ff),
                        {layer_a(l)});
    mlir::Value fh = b.node(OpKind::Mul, {gate, up}, "fh" + suff, shape2(n, ff), {layer_a(l)});
    mlir::Value fo = b.node(OpKind::QuantizedMatmul, {fh}, "fo" + suff, shape2(n, d),
                         {layer_a(l), slot(WeightSlot::FfnDown), batch_a(n),
                         dim_a(ff), ds_a(m.weights.blocks[l].ffn_down.type())});
    mlir::Value f2 = prof.post_ffw_norm
                      ? b.node(OpKind::RmsNorm, {fo}, "f2" + suff, shape2(n, d),
                               {layer_a(l), slot(WeightSlot::PostFfwNorm), batch_a(n),
                                dim_a(d)})
                      : fo;
    mlir::Value x2 = b.node(OpKind::Add, {f2, x1}, "x2" + suff, shape2(n, d), {layer_a(l)});

    // --- per-layer gate ----------------------------------------------------
    mlir::Value x3 = x2;
    if (prof.per_layer_input) {
      mlir::Value proj = b.node(OpKind::MatvecBf16, {inp}, "proj" + suff, shape2(n, per),
                             {layer_a(l), batch_a(n),
                              attr_i("offset", int64_t(l) * per),
                              attr_i("rows", int64_t(per))});
      proj = b.node(OpKind::Scale, {proj}, "projs" + suff, shape2(n, per),
                    {attr_f("scale", proj_scale)});
      proj = b.node(OpKind::RmsNorm, {proj}, "projn" + suff, shape2(n, per),
                    {layer_a(l), slot(WeightSlot::PerLayerProjNorm), batch_a(n), dim_a(per)});
      // Layer combine (C.5): ple = (projn + ple_row[:, l*per:(l+1)*per] * sqrt(per))
      // / sqrt(2), decomposed into a strided Slice of the per-layer embedding row
      // + Scale/Add/Scale.
      const mlir::Value ple_col = b.node(OpKind::Slice, {ple_row}, "ple_col" + suff,
                                      shape2(n, per),
                                      {attr_i("offset", int64_t(l) * per),
                                       attr_i("length", int64_t(per)),
                                       attr_i("stride", int64_t(total_per))});
      const mlir::Value ple_scaled = b.node(OpKind::Scale, {ple_col}, "ple_scaled" + suff,
                                         shape2(n, per),
                                         {attr_f("scale", double(std::sqrt(float(per))))});
      const mlir::Value ple_sum = b.node(OpKind::Add, {proj, ple_scaled}, "ple_sum" + suff,
                                      shape2(n, per), {});
      mlir::Value ple = b.node(OpKind::Scale, {ple_sum}, "ple" + suff, shape2(n, per),
                            {attr_f("scale", 1.0 / std::sqrt(2.0))});
      mlir::Value g0 = b.node(OpKind::MatvecF32, {x2}, "g0" + suff, shape2(n, per),
                           {layer_a(l), slot(WeightSlot::InpGate), batch_a(n)});
      g0 = b.node(OpKind::GeluFp16, {g0}, "g0g" + suff, shape2(n, per), {layer_a(l)});
      g0 = b.node(OpKind::Mul, {g0, ple}, "g0m" + suff, shape2(n, per), {layer_a(l)});
      mlir::Value g1 = b.node(OpKind::MatvecF32, {g0}, "g1" + suff, shape2(n, d),
                           {layer_a(l), slot(WeightSlot::Proj), batch_a(n)});
      mlir::Value g2 = b.node(OpKind::RmsNorm, {g1}, "g2" + suff, shape2(n, d),
                           {layer_a(l), slot(WeightSlot::PostNorm), batch_a(n), dim_a(d)});
      x3 = b.node(OpKind::Add, {x2, g2}, "x3" + suff, shape2(n, d), {layer_a(l)});
    }

    mlir::Value out;
    if (prof.layer_scale && !m.weights.blocks[l].layer_output_scale.empty())
      out = b.node(OpKind::Scale, {x3}, "out" + suff, shape2(n, d),
                   {attr_f("scale", double(m.weights.blocks[l].layer_output_scale[0]))});
    else
      out = x3;
    residual = out;
  }

  // Defer the KV-cache append to after every layer: a shared layer's attention
  // reads its donor's *prior* cache, so the donor's own chunk must not overwrite
  // it until all layers have consumed it. This is what lets a chunk that wraps
  // the SWA ring (start_pos >= sliding_window) still attend over the prior
  // window. The output is a dummy scalar — the append is a cache side effect.
  for (uint32_t l = 0; l < n_kv; ++l) {
    b.node(OpKind::KvAppend, {kv_k[l], kv_v[l], start_pos},
           "kv_append" + std::to_string(l), {}, {layer_a(l), batch_a(n)});
  }

  b.set_output(residual);

  ir.module = b.finish();
  return ir;
}

// The prefill emitter's IR, lowered with no passes, like build_gemma4_graph.
planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n) {
  GraphIr ir = emit_prefill_ir(m, n);
  if (!ir.module)
    return {};
  return lower_graph(*ir.module);
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
    if (!pplan) { std::cerr << "[prefill-spine] plan failed: " << pplan.error().message << "\n"; return 0; }

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
    if (!res) { std::cerr << "[prefill-spine] execute failed: " << res.error().message << "\n"; return 0; }
    if (res->outputs.empty()) { std::cerr << "[prefill-spine] no outputs\n"; return 0; }
    const sx::Bytes& hidden_b = res->outputs[0];
    if (hidden_b.size() < uint64_t(d) * n * sizeof(float)) { std::cerr << "[prefill-spine] hidden too small: " << hidden_b.size() << " < " << uint64_t(d)*n*sizeof(float) << "\n"; return 0; }
    std::span<const float> hf(reinterpret_cast<const float*>(hidden_b.data()),
                              uint64_t(d) * n);
    std::vector<float> hidden(d);
    for (uint32_t k = 0; k < d; ++k) hidden[k] = hf[(n - 1) * d + k];
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

// ---------------------------------------------------------------------------
// Persistent CUDA spine decode runtime (Section C.2). The greedy decode
// graph/plan/backends and the captured decode-step program persist across
// generate/prefill calls so a later call only stages device-resident params
// (pos / token / start_pos) and replays — no plan rebuild or re-capture per
// call. Keyed by (model, n_ctx); the model must outlive the cache (the
// backends hold a raw model pointer). The decode cache self-corrects across
// reuse (each step overwrites the slot the next step reads), so caching it is
// safe.
//
// Prefill segments are cached the same way, minus their CUDA-graph capture:
// a capture is ~0.8% neutral (see "prefill-cuda-graph-neutral") but a
// captured 512-token segment graph costs ~190 MiB that is never freed when
// cached, and replaying a stale capture would write through the device pointers
// it baked in. So the plan side of a segment — graph, rewrites, execution plan,
// backends, executor, all a pure function of (model, n_ctx, segment length,
// planner environment) — is cached, and the capture stays per call.
// ---------------------------------------------------------------------------

// A prefill segment's build product. `cuda` holds the shared KV cache (adopted
// via share_caches, re-adopted on every cache hit) plus three small persistent
// slots (token 4 B, pos 8 B, segment token ids chunk*8 = 4096 B at the SWA
// chunk); the plan itself owns no device memory — its buffers are logical and
// the device pool allocates them per execute. The segment-build path also
// allocates a full KV cache in the backend constructor, which share_caches then
// discards, so the retained device cost of a cache entry is those 4 KiB.
struct PrefillSegment {
  planner::Graph graph;
  planner::ExecutionPlan plan;
  std::unique_ptr<planner::SonicBackend> cpu;
  std::unique_ptr<planner::SonicCudaBackend> cuda;
  std::unique_ptr<planner::RuntimeExecutor> exec;
  // The captured full-chunk CUDA graph, kept with the segment so a later call
  // can replay it instead of paying the capture again. A capture bakes in the
  // KV cache's device pointers, so it is reused only while `cuda` shares the
  // very cache vector it was recorded against — stated by `chunk_prog_cache`
  // (the cache_identity() at capture time). The segment's token/pos staging
  // buffers are carried by `cuda` and re-staged per call, so the captured
  // values themselves are not part of the capture.
  std::unique_ptr<hal::StepProgram> chunk_prog;
  std::shared_ptr<void> chunk_prog_cache;
};

// The planner environment a plan is a function of. Everything else the prefill
// segment build reads — the model, n_ctx, the resource snapshot, the cost
// model, the segment length — is fixed per decode runtime except the segment
// length, which is the cache key. A change in any of these switches changes the
// graph the plan is derived from (the q8_1/mmq gates decide whether the mmq
// fusions run at all), so they are part of the key rather than assumed constant
// for the process.
struct PlannerEnv {
  bool q8_1;
  bool mmq;
  int mmq_min_n;
  friend bool operator==(const PlannerEnv&, const PlannerEnv&) = default;
};

static PlannerEnv planner_env() {
  return PlannerEnv{quant::cuda::q8_1_enabled(), quant::cuda::mmq_enabled(),
                    quant::cuda::mmq_min_n()};
}

// SONICBOOM_PLAN_CACHE=0 disables the prefill segment cache (segments then go
// back to being built per call), for A/B and for ruling the cache out of a
// divergence. Any other value, including unset, keeps it on.
static bool plan_cache_enabled() {
  const char* e = std::getenv("SONICBOOM_PLAN_CACHE");
  return e == nullptr || std::string_view(e) != "0";
}

// Cache cap. A single prefill call needs at most two distinct segment lengths
// (the full chunk, then the final shorter remainder — the per-call map is keyed
// by length, so two full-chunk segments share one entry). Two entries are
// therefore enough for the call in flight, but not across a multi-turn session:
// every turn adds up to two new lengths (its prompt and its reuse suffix), so a
// cap of two rebuilds lengths a session used one turn earlier (~115 ms of
// planning each: the 8-turn workload trace rebuilds 36 segments for 10 hits).
// Eight entries keep about four turns of history; the entry holds the plan and
// the segment backends, whose device activations come from the global device
// pool, so the cap bounds host-side planning state rather than VRAM.
static constexpr std::size_t kMaxCachedSegments = 8;

struct DecodeRuntime {
  planner::Graph graph;
  planner::ExecutionPlan plan;
  planner::ResourceSnapshot snap;
  std::unique_ptr<planner::SonicBackend> cpu;         // host boundary tasks
  std::unique_ptr<planner::SonicCudaBackend> cuda;    // device-resident compute
  std::unique_ptr<planner::RuntimeExecutor> exec;
  std::unique_ptr<hal::StepProgram> prog;             // captured greedy decode step
  // Prefill segments by length, most-recently used first, capped at
  // kMaxCachedSegments.
  std::list<std::pair<uint64_t, std::unique_ptr<PrefillSegment>>> prefills;
  // The planner environment `prefills` was built under; a change clears it.
  PlannerEnv prefill_env{};
  bool prefill_env_set = false;
};

static std::map<std::pair<const Gemma4Model*, uint64_t>,
                std::unique_ptr<DecodeRuntime>> g_decode_runtimes;

static bool cuda_graphs_enabled() {
  const char* e = std::getenv("SONICBOOM_CUDA_GRAPHS");
  return e == nullptr || std::string_view(e) != "0";
}

// Route host-memory-space compute tasks (Embedding / Softcap) to the CPU backend
// and leave device-memory-space tasks on the CUDA backend. Shared by the decode
// and prefill runtimes so each executor gets the same placement overrides.
static void route_host_tasks(planner::RuntimeExecutor& exec,
                             const planner::ExecutionPlan& plan,
                             planner::SonicBackend& cpu) {
  for (const auto& task : plan.tasks) {
    if (task.kind != planner::TaskKind::Compute)
      continue;
    const planner::MemorySpace* ms = nullptr;
    if (task.memory_space)
      ms = plan.find_memory_space(*task.memory_space);
    if (!ms || ms->kind == planner::MemoryKind::Host)
      exec.set_task_backend(task.id, &cpu);
  }
}

// Build the greedy decode graph, plan it, and wire up its backends — everything
// a decode runtime is except the captured step. Split out of
// get_decode_runtime so the plan recorder can run the very same plan *without*
// capture: a captured step replays as one graph launch and bypasses the
// executor, so no per-step launches would be visible to record. Returns nullptr
// on a build/plan failure (the caller falls back to 0).
static std::unique_ptr<DecodeRuntime> build_decode_runtime(
    const Gemma4Model& m, uint64_t n_ctx,
    const GraphTransform* transform = nullptr) {
  auto rt = std::make_unique<DecodeRuntime>();
  GraphIr ir = emit_decode_ir(m, /*greedy=*/true);
  rt->graph = lower_graph(ir, planner::IrPassPhase::Decode,
                          quant::cuda::q8_1_enabled());
  // After every graph-level pass, so the transform sees the graph the plan is
  // actually derived from (plan.h: GraphTransform).
  if (transform && *transform) (*transform)(rt->graph);
  auto snap = planner::CpuResourceProvider::snapshot();
  if (!snap)
    return nullptr;
  rt->snap = std::move(*snap);
  {
    planner::AnalyticalCpuCostModel cost(rt->snap);
    auto plan = planner::plan_execution(rt->graph, rt->snap, cost);
    if (!plan)
      return nullptr;
    rt->plan = std::move(*plan);
  }
  rt->cpu = std::make_unique<planner::SonicBackend>(m, rt->graph);
  rt->cuda = std::make_unique<planner::SonicCudaBackend>(
      kv_cache_spec(m.config), rt->graph, n_ctx);
  rt->cuda->bind_model(m);
  rt->exec = std::make_unique<planner::RuntimeExecutor>(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, rt->cuda.get()}});
  route_host_tasks(*rt->exec, rt->plan, *rt->cpu);
  return rt;
}

// Build (or fetch) the cached greedy decode runtime for (m, n_ctx). The first
// call builds the graph + quantize passes + plan + backends and captures one
// decode step (HAL record_step); later calls return the same runtime. The
// capture reads token/pos from the device slots, so it is replay-agnostic to the
// real start token/position (the caller re-seeds/re-stages before each replay).
// Returns nullptr on a build/plan failure (the caller falls back to 0).
static DecodeRuntime* get_decode_runtime(const Gemma4Model& m, uint64_t n_ctx) {
  auto key = std::make_pair(&m, n_ctx);
  auto it = g_decode_runtimes.find(key);
  if (it != g_decode_runtimes.end())
    return it->second.get();

  auto rt = build_decode_runtime(m, n_ctx);
  if (!rt) return nullptr;

  // Capture the greedy decode step once (the warm-up run allocates every pooled
  // device buffer and weight-upload so the capture pass only records).
  if (cuda_graphs_enabled()) {
    rt->cuda->seed_token(0);
    rt->cuda->set_pos(0);
    std::vector<sx::Bytes> in;
    in.push_back(encode_i64(0));
    in.push_back(encode_i64(0));
    auto res = rt->exec->execute(rt->plan, rt->snap, in);
    if (res) {
      rt->cuda->synchronize();
      rt->cuda->seed_token(0);
      rt->prog = rt->cuda->record_step([&]() {
        std::vector<sx::Bytes> cin;
        cin.push_back(encode_i64(0));
        cin.push_back(encode_i64(0));
        auto cres = rt->exec->execute(rt->plan, rt->snap, cin);
        return cres.has_value();
      });
    }
  }

  DecodeRuntime* raw = rt.get();
  g_decode_runtimes.emplace(key, std::move(rt));
  return raw;
}

bool warmup_spine_cuda(const Gemma4Model& m, uint64_t n_ctx) {
  if (m.weights.blocks.empty() || n_ctx == 0) return false;
  return get_decode_runtime(m, n_ctx) != nullptr;
}

uint64_t generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                             uint64_t start_pos, std::span<int64_t> tokens,
                             uint64_t n_ctx) {
  if (m.weights.blocks.empty() || start_token >= m.config.vocab_size) return 0;
  if (n_ctx == 0) n_ctx = start_pos + tokens.size();

  const bool gen_timing = std::getenv("SONICBOOM_GEN_TIMING") != nullptr;
  const auto t_plan0 = std::chrono::steady_clock::now();
  // C.2: the greedy decode graph/plan/backends and the captured decode step are
  // cached per (model, n_ctx); the first call builds + captures, later calls
  // only re-seed/re-stage the device params and replay.
  DecodeRuntime* rt = get_decode_runtime(m, n_ctx);
  if (!rt) return 0;

  const auto t_loop0 = std::chrono::steady_clock::now();
  auto t_first = t_loop0;
  uint64_t token = start_token;
  uint64_t pos = start_pos;
  uint64_t step = 0;

  // Seed the device-resident token-id slot (Phase 3.2 B4). The cached capture was
  // warmed with a placeholder token, so every call re-seeds the real start token.
  rt->cuda->seed_token(start_token);

  if (rt->prog) {
    // Replay the cached captured step: stage the position, launch the graph, read
    // the winning token back from the persistent device slot. The synchronous D2H
    // read also drains the graph stream before the next step.
    for (; step < tokens.size(); ++step) {
      const std::string step_name = "decode_step_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(step_name.c_str());
      rt->cuda->set_pos(pos);
      if (!rt->prog->replay()) {
        if (nvtx_enabled()) nvtxRangePop();
        std::cerr << "[spine] step " << step << " graph replay failed\n";
        break;
      }
      float next_f = -1.0f;
      rt->cuda->read_token(&next_f);
      if (nvtx_enabled()) nvtxRangePop();
      const int64_t next = int64_t(next_f);
      if (next < 0) break;
      tokens[step] = next;
      token = static_cast<uint64_t>(next);
      ++pos;
      if (step == 0) t_first = std::chrono::steady_clock::now();
    }
  } else {
    // Fallback: per-step executor dispatch (capture disabled or failed).
    for (; step < tokens.size(); ++step) {
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(token));
      in.push_back(encode_i64(pos));

      // Stage the position into the device slot read by RoPE / KV-append /
      // attention (Phase 3.2 B4 / CUDA Graphs): the kernels read it in-kernel.
      rt->cuda->set_pos(pos);

      // NVTX step marker: brackets one decode step so nsys can isolate the
      // steady-state per-token window (instead of dividing whole-run totals).
      const std::string step_name = "decode_step_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(step_name.c_str());
      auto res = rt->exec->execute(rt->plan, rt->snap, in);
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

// RAII: install `rec` as this thread's plan recorder for the scope, so an early
// return cannot leave a dangling recorder installed.
namespace {
struct RecorderScope {
  explicit RecorderScope(planner::PlanRecorder* rec) { planner::set_thread_recorder(rec); }
  ~RecorderScope() { planner::set_thread_recorder(nullptr); }
  RecorderScope(const RecorderScope&) = delete;
  RecorderScope& operator=(const RecorderScope&) = delete;
};

const char* arch_name(ModelArch a) {
  return a == ModelArch::Gemma4 ? "gemma4" : "qwen2";
}

// The device a record was taken on, for the record's `device` line. Part of the
// record is device-dependent (the decode-attention split-KV grid is sized from
// the SM count), so a stored record has to carry which GPU produced it. Queried
// here because this is the translation unit that owns the CUDA context.
std::string device_identity() {
  int dev = 0;
  cudaDeviceProp p{};
  if (cudaGetDevice(&dev) != cudaSuccess ||
      cudaGetDeviceProperties(&p, dev) != cudaSuccess)
    return "unknown";
  char buf[288];
  std::snprintf(buf, sizeof buf, "%s sm %d cc %d.%d", p.name,
                p.multiProcessorCount, p.major, p.minor);
  return buf;
}

// One decode step through the executor, with the same device-slot staging the
// non-captured generate_spine_cuda loop does. Returns false when the step failed
// (the caller stops and keeps whatever was recorded so far).
bool decode_step_through_executor(DecodeRuntime& rt, uint64_t token, uint64_t pos,
                                  int64_t* next_out) {
  const std::vector<sx::Bytes> in{encode_i64(token), encode_i64(pos)};
  rt.cuda->set_pos(pos);
  auto res = rt.exec->execute(rt.plan, rt.snap, in);
  if (!res || res->outputs.empty()) return false;
  const sx::Bytes& arg = res->outputs[0];
  if (arg.size() < sizeof(float)) return false;
  *next_out = int64_t(*reinterpret_cast<const float*>(arg.data()));
  return true;
}
}  // namespace

std::string cuda_device_identity() { return device_identity(); }

std::optional<planner::PlanRunRecord> record_decode_plan(const Gemma4Model& m,
                                                         uint64_t n_ctx,
                                                         uint64_t steps) {
  return record_decode_plan(m, n_ctx, steps, GraphTransform{});
}

std::optional<planner::PlanRunRecord> record_decode_plan(
    const Gemma4Model& m, uint64_t n_ctx, uint64_t steps,
    const GraphTransform& transform) {
  // Deliberately not get_decode_runtime: that runtime is captured, and a
  // captured step replays as a single graph launch with no executor walk, so
  // there would be no per-step kernels to record.
  auto rt = build_decode_runtime(m, n_ctx, &transform);
  if (!rt) return std::nullopt;

  planner::PlanRecorder rec;
  RecorderScope scope(&rec);
  rec.set_run(arch_name(m.config.arch), "decode", n_ctx, steps, rt->plan.plan_id,
              rt->plan.model_fingerprint, planner::backend_tag_name(
                                              planner::BackendTag::Sonic));
  rec.set_device(device_identity());

  int64_t token = 0;
  for (uint64_t step = 0; step < steps; ++step) {
    if (!decode_step_through_executor(*rt, uint64_t(token), step, &token)) break;
    if (token < 0) break;  // model-empty: no further step to record
  }
  return rec.finish();
}

std::optional<planner::PlanRunRecord> record_prefill_plan(const Gemma4Model& m,
                                                          uint64_t n) {
  return record_prefill_plan(m, n, GraphTransform{});
}

std::optional<planner::PlanRunRecord> record_prefill_plan(
    const Gemma4Model& m, uint64_t n, const GraphTransform& transform) {
  if (n == 0) return std::nullopt;
  GraphIr ir = emit_prefill_ir(m, n);
  // The same mmq fusion the prefill segments use, under the same conditions, so
  // the recorded steps are the ones a real prefill segment would run.
  planner::Graph graph = lower_graph(
      ir, planner::IrPassPhase::Prefill,
      quant::cuda::q8_1_enabled() && quant::cuda::mmq_enabled() &&
          n >= uint64_t(quant::cuda::mmq_min_n()));
  // After every graph-level pass (plan.h: GraphTransform).
  if (transform) transform(graph);
  auto snap = planner::CpuResourceProvider::snapshot();
  if (!snap) return std::nullopt;
  planner::AnalyticalCpuCostModel cost(*snap);
  auto plan = planner::plan_execution(graph, *snap, cost);
  if (!plan) return std::nullopt;

  // A prefill segment is one batched pass over the whole [n, dim] activation, so
  // it executes as a single step whose launches are the segment's kernels. The
  // cache has to be large enough to hold the run without wrapping a sliding
  // window: size it to the longest ring the model declares (and at least n), so
  // the segment lands in a straight, wrap-free run exactly as a real one must.
  uint64_t n_ctx = n;
  for (uint32_t l = 0; l < m.config.n_layer_kv(); ++l)
    n_ctx = std::max<uint64_t>(n_ctx, m.config.layer_config(l).sliding_window);
  planner::SonicCudaBackend cuda(kv_cache_spec(m.config), graph, n_ctx);
  cuda.bind_model(m);
  planner::SonicBackend cpu(m, graph);
  planner::RuntimeExecutor exec(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, &cuda}});
  route_host_tasks(exec, *plan, cpu);

  planner::PlanRecorder rec;
  RecorderScope scope(&rec);
  rec.set_run(arch_name(m.config.arch), "prefill", 0, n, plan->plan_id,
              plan->model_fingerprint,
              planner::backend_tag_name(planner::BackendTag::Sonic));
  rec.set_device(device_identity());

  // Token ids are staged into the device slot the batched Embedding reads; the
  // ids themselves do not affect the plan (only the [n, dim] shapes do).
  const std::vector<uint64_t> prompt(n, 0);
  cuda.set_prefill_tokens(prompt);
  cuda.set_pos(0);
  const std::vector<sx::Bytes> in{encode_tokens(prompt), encode_i64(0)};
  (void)exec.execute(*plan, *snap, in);
  return rec.finish();
}
#else
uint64_t generate_spine_cuda(const Gemma4Model&, uint64_t, uint64_t,
                             std::span<int64_t>, uint64_t) {
  return 0;  // built without CUDA: no device spine is available
}

bool warmup_spine_cuda(const Gemma4Model&, uint64_t) {
  return false;  // built without CUDA: there is no runtime to build
}

std::optional<planner::PlanRunRecord> record_decode_plan(const Gemma4Model&,
                                                         uint64_t, uint64_t) {
  return std::nullopt;  // built without CUDA: nothing launches
}

std::optional<planner::PlanRunRecord> record_decode_plan(const Gemma4Model&,
                                                         uint64_t, uint64_t,
                                                         const GraphTransform&) {
  return std::nullopt;  // built without CUDA: nothing launches
}

std::optional<planner::PlanRunRecord> record_prefill_plan(const Gemma4Model&,
                                                          uint64_t) {
  return std::nullopt;  // built without CUDA: nothing launches
}

std::optional<planner::PlanRunRecord> record_prefill_plan(const Gemma4Model&,
                                                          uint64_t,
                                                          const GraphTransform&) {
  return std::nullopt;  // built without CUDA: nothing launches
}

std::string cuda_device_identity() { return "unknown"; }
#endif

// ---------------------------------------------------------------------------
// Prefill + decode (CUDA spine). Same token-stream contract as the CPU
// generate_prefill_spine, but the prefill fills the device-resident KV caches
// (SonicCudaBackend::prefill: batched f32 matmul + flash attention) and the
// decode runs through the CUDA spine loop. lm_head uses the CUDA backend so the
// first-token logits match the device spine's lm-head arithmetic.
// ---------------------------------------------------------------------------
#ifdef SONICBOOM_USE_CUDA
// Env-gated logit dump (C.4 isolation measurement). When SONICBOOM_DUMP_LOGITS
// is set, the decode graph is built non-greedy (Softcap output) so each step's
// softcapped logits reach the host, and the top-2 (with their logits) are
// appended to SONICBOOM_DUMP_LOGITS_FILE (default stderr). Argmax(softcapped) ==
// Argmax(raw), so the token stream is unchanged — this is measurement only.
static void dump_logits_top2(const char* file, uint64_t step, uint64_t pos,
                             std::span<const float> logits) {
  int64_t i1 = -1, i2 = -1;
  float l1 = -INFINITY, l2 = -INFINITY;
  for (size_t i = 0; i < logits.size(); ++i) {
    const float v = logits[i];
    if (v > l1) {
      l2 = l1;
      i2 = i1;
      l1 = v;
      i1 = int64_t(i);
    } else if (v > l2) {
      l2 = v;
      i2 = int64_t(i);
    }
  }
  FILE* f = file ? std::fopen(file, "a") : stderr;
  if (!f) f = stderr;
  std::fprintf(f,
               "[dump-logits] step=%llu pos=%llu top1=%lld logit1=%.6f top2=%lld "
               "logit2=%.6f diff=%.6f\n",
               (unsigned long long)step, (unsigned long long)pos, (long long)i1,
               double(l1), (long long)i2, double(l2), double(l1 - l2));
  if (file && f != stderr) std::fclose(f);
}

// Shared prefill + decode body. `prompt` is the tokens to actually prefill
// (already the suffix when prefix reuse skipped a shared prefix); `session`, when
// non-null, is the persistent prefix-cache session this run adopts/publishes its
// device cache to. The full prompt prefill (generate_prefill_spine_cuda) passes
// null; the prefix-reuse path passes the live session.
static uint64_t prefill_decode_cuda_impl(const Gemma4Model& m,
                                         std::span<const uint64_t> prompt,
                                         uint64_t start_pos,
                                         std::span<int64_t> tokens,
                                         uint64_t n_ctx, bool use_prefill,
                                         planner::KvCacheSession* session) {
  if (m.weights.blocks.empty() || prompt.empty() || tokens.empty()) return 0;
  if (n_ctx == 0) n_ctx = start_pos + prompt.size() + tokens.size();

  const bool gen_timing = std::getenv("SONICBOOM_GEN_TIMING") != nullptr;
  const bool dump_logits = std::getenv("SONICBOOM_DUMP_LOGITS") != nullptr;
  const char* dump_logits_file = std::getenv("SONICBOOM_DUMP_LOGITS_FILE");
  // Host-side phase breakdown for the prefill host-overhead investigation (step
  // 1). Opt-in via SONICBOOM_PREFILL_PHASE_TIMING: each Phase prints its wall ms
  // on destruction and (when SONICBOOM_NVTX is set) brackets the region with an
  // NVTX range of the same name so nsys attributes cudaMalloc/cudaMemcpy/
  // cudaLaunchKernel host time to the phase. Measurement only — OFF by default.
  const bool phase_timing = std::getenv("SONICBOOM_PREFILL_PHASE_TIMING") != nullptr;
  struct Phase {
    const char* name;
    bool report;
    std::chrono::steady_clock::time_point t0;
    Phase(const char* n, bool rep)
        : name(n), report(rep), t0(std::chrono::steady_clock::now()) {
      if (nvtx_enabled()) nvtxRangePushA(n);
    }
    ~Phase() {
      if (nvtx_enabled()) nvtxRangePop();
      if (report) {
        using ms = std::chrono::duration<double, std::milli>;
        std::cerr << "[phase] " << name << " "
                  << ms(std::chrono::steady_clock::now() - t0).count() << " ms\n";
      }
    }
  };
  // Decode runtime. The normal path reuses the cached greedy runtime (and its
  // captured decode step) keyed by (model, n_ctx); the logit-dump path builds a
  // non-greedy throwaway runtime because it must reach the full softcapped logits
  // on the host (argmax(softcapped) == argmax(raw), so only measurement differs).
  DecodeRuntime* rt = nullptr;
  std::unique_ptr<DecodeRuntime> dump_rt;
  if (!dump_logits) {
    Phase _p("prefill:get_decode_runtime", phase_timing);
    rt = get_decode_runtime(m, n_ctx);
    if (!rt) return 0;
  } else {
    Phase _p("prefill:build_decode_graph", phase_timing);
    dump_rt = std::make_unique<DecodeRuntime>();
    GraphIr dump_ir = emit_decode_ir(m, /*greedy=*/false);
    dump_rt->graph = lower_graph(dump_ir, planner::IrPassPhase::Decode,
                                 quant::cuda::q8_1_enabled());
    auto s = planner::CpuResourceProvider::snapshot();
    if (!s) return 0;
    dump_rt->snap = std::move(*s);
    {
      Phase _pp("prefill:plan_execution", phase_timing);
      planner::AnalyticalCpuCostModel cost(dump_rt->snap);
      auto p = planner::plan_execution(dump_rt->graph, dump_rt->snap, cost);
      if (!p) return 0;
      dump_rt->plan = std::move(*p);
    }
    dump_rt->cpu = std::make_unique<planner::SonicBackend>(m, dump_rt->graph);
    dump_rt->cuda = std::make_unique<planner::SonicCudaBackend>(
        kv_cache_spec(m.config), dump_rt->graph, n_ctx);
    dump_rt->cuda->bind_model(m);
    dump_rt->exec = std::make_unique<planner::RuntimeExecutor>(
        std::map<planner::BackendTag, planner::Backend*>{
            {planner::BackendTag::Sonic, dump_rt->cuda.get()}});
    route_host_tasks(*dump_rt->exec, dump_rt->plan, *dump_rt->cpu);
    rt = dump_rt.get();
  }
  const planner::ResourceSnapshot& snap = rt->snap;
  // Cost model for planning a prefill segment graph the first time it is seen
  // (Section C.2's decode runtime is cached; a segment is planned once and then
  // reused from rt->prefills — see segment_for below).
  planner::AnalyticalCpuCostModel cost(snap);

  // Prefix reuse: adopt the session's persistent cache if it has one, otherwise
  // publish this backend's fresh cache into the session (first call) so it
  // survives this run. The prefill segment backends then share it transitively
  // via share_caches(*rt->cuda).
  if (session) {
    if (session->caches)
      rt->cuda->share_caches(*session);
    else
      rt->cuda->publish_cache(*session);
  }

  const uint64_t vocab = m.config.vocab_size;
  const uint32_t d = m.config.embedding_length;
  std::vector<float> logits(vocab);

  uint64_t token = 0;
  uint64_t pos = start_pos;
  uint64_t step = 0;

  int64_t first = -1;
  if (use_prefill) {
    // Chunked device-resident prefill: the prompt is processed in segments of
    // `chunk` tokens (the SWA ring size, so a segment is a wrap-free run and the
    // batched kernels' consecutive-slot assumption holds). Each segment runs the
    // batched prefill graph; within a segment the attention reads the segment's
    // own K/V from the batch buffer plus the prior context from the shared KV
    // cache, and the cache append is deferred until every layer — including the
    // shared donors — has read the prior window, so a segment that wraps the SWA
    // ring still attends correctly. Only the last segment's final hidden column
    // feeds lm_head.
    const uint64_t n = prompt.size();
    // Segment size == the KV spec's chunk (see kv_cache_spec): the ring size where
    // the model has a window, else the fixed default segment.
    const uint64_t chunk = (m.config.sliding_window > 0) ? m.config.sliding_window
                                                        : kDefaultPrefillChunk;

    // One prefill segment per distinct segment size (usually just `chunk`; the
    // final segment may be shorter). Each is bound to its own graph but shares
    // the decode backend's KV cache, so prior segments' context is visible. The
    // full-chunk segment is captured into a CUDA Graph (prog) and replayed per
    // non-final segment; the final (possibly shorter) segment runs uncaptured so
    // its hidden column reaches the host through the executor's output path.
    //
    // The segment build (graph + rewrites + plan + backends) is a pure function
    // of (model, n_ctx, nseg, planner environment) and costs ~115 ms of planning
    // per 512-token Gemma segment, so it is cached across calls in the decode
    // runtime (see the DecodeRuntime note). The offset a segment is executed at
    // is not part of it: pos reaches the kernels as an execute-time binding
    // (set_pos + the `in` argument), and a plan records no position. The
    // CUDA-graph capture is kept with the segment and reused across calls for
    // the one thing it does bake in — the KV cache's device pointers: it is
    // replayed only while the segment shares the cache vector it was recorded
    // against (chunk_prog_cache), otherwise it is captured again. The replay
    // path below is reachable when a capture was made now or reused.
    const PlannerEnv env_now = planner_env();
    if (!rt->prefill_env_set || !(rt->prefill_env == env_now)) {
      rt->prefills.clear();  // planner environment changed: every entry is stale
      rt->prefill_env = env_now;
      rt->prefill_env_set = true;
    }
    const bool cache_segments = plan_cache_enabled();
    // Per-call owners: segments when the cache is off. A captured program is
    // owned by its segment (which the replay loop reads through `progs`), so
    // it outlives the call with the segment that recorded it.
    std::map<uint64_t, std::unique_ptr<PrefillSegment>> call_segments;
    std::map<uint64_t, hal::StepProgram*> progs;
    auto segment_for = [&](uint64_t nseg) -> PrefillSegment* {
      if (cache_segments) {
        auto it = std::find_if(
            rt->prefills.begin(), rt->prefills.end(),
            [nseg](const auto& e) { return e.first == nseg; });
        if (it != rt->prefills.end()) {
          Phase _h("prefill:segment_cache_hit", phase_timing);
          PrefillSegment* hit = it->second.get();
          rt->prefills.splice(rt->prefills.begin(), rt->prefills, it);  // LRU
          // Re-adopt the runtime's KV cache: the entry may have been built (or
          // last used) against a different session's cache, and a segment must
          // never keep an old session's cache alive.
          hit->cuda->share_caches(*rt->cuda);
          return hit;
        }
      }
      auto seg = std::make_unique<PrefillSegment>();
      {
        Phase _b("prefill:segment_build_graph", phase_timing);
        GraphIr seg_ir = emit_prefill_ir(m, nseg);
        // C.4: fold the mmq activation quantize into the norm/gelu kernels when
        // the mmq tensor-core path is active for this segment (same conditions
        // the batched matmul uses, so fused and unfused positions never disagree
        // about which kernel a position runs).
        seg->graph = lower_graph(
            seg_ir, planner::IrPassPhase::Prefill,
            quant::cuda::q8_1_enabled() && quant::cuda::mmq_enabled() &&
                nseg >= uint64_t(quant::cuda::mmq_min_n()));
      }
      std::expected<planner::ExecutionPlan, planner::PlannerError> pplan;
      {
        Phase _p("prefill:segment_plan", phase_timing);
        pplan = planner::plan_execution(seg->graph, snap, cost);
      }
      if (!pplan) {
        std::cerr << "[prefill-cuda] n=" << nseg << " plan failed\n";
        return nullptr;
      }
      seg->plan = std::move(*pplan);
      seg->cpu = std::make_unique<planner::SonicBackend>(m, seg->graph);
      seg->cuda = std::make_unique<planner::SonicCudaBackend>(
          kv_cache_spec(m.config), seg->graph, n_ctx);
      seg->cuda->bind_model(m);
      seg->cuda->share_caches(*rt->cuda);
      seg->exec = std::make_unique<planner::RuntimeExecutor>(
          std::map<planner::BackendTag, planner::Backend*>{
              {planner::BackendTag::Sonic, seg->cuda.get()}});
      route_host_tasks(*seg->exec, seg->plan, *seg->cpu);
      PrefillSegment* raw = seg.get();
      if (cache_segments) {
        rt->prefills.emplace_front(nseg, std::move(seg));
        while (rt->prefills.size() > kMaxCachedSegments) rt->prefills.pop_back();
      } else {
        // Uncached: keep it alive for the rest of the call (a later segment of
        // the same length must reuse it, not rebuild it).
        call_segments.emplace(nseg, std::move(seg));
      }
      return raw;
    };

    // CUDA Graph capture of the full-chunk segment (HAL record_step), replayed
    // per non-final segment so the ~420 per-segment device-op launches collapse
    // into one graph launch. Capture only when there is at least one non-final
    // full-chunk segment (n > chunk); a single segment is always final and runs
    // uncaptured. The capture warm-up runs the full-chunk segment once and its
    // deferred KvAppend overwrites the shared KV cache — safe only when
    // the cache is empty before the run (start_pos == 0, a fresh full prefill).
    // A prefix-reuse suffix (start_pos > 0) already holds the shared prefix in
    // the cache, and the warm-up append would clobber the prefix positions the
    // suffix's first segment still attends over, so reuse runs uncaptured
    // (correctness over the launch-overhead win, which only matters for long
    // prompts). Opt-out via SONICBOOM_CUDA_GRAPHS=0; a capture failure leaves
    // `prog` null and falls back to per-segment dispatch.
    //
    // A previous call's capture is reused when the segment still shares the
    // cache vector it was recorded against. That holds for every call on the
    // same decode runtime: a fresh session publishes (or adopts) the runtime's
    // persistent cache, so the device pointers do not move. Reuse skips the
    // warm-up too — it exists to initialize the kernels and allocations the
    // capture records, and those are already initialized once the graph exists.
    const bool use_graphs = cuda_graphs_enabled();
    if (use_graphs && n > chunk && start_pos == 0) {
      Phase _cap("prefill:capture_graph", phase_timing);
      PrefillSegment* full = segment_for(chunk);
      if (full) {
        const std::shared_ptr<void> cache_id = full->cuda->cache_identity();
        if (full->chunk_prog && full->chunk_prog_cache.get() == cache_id.get()) {
          Phase _reuse("prefill:capture_reuse", phase_timing);
          progs[chunk] = full->chunk_prog.get();
        } else {
          const std::span<const uint64_t> first_chunk = prompt.subspan(0, chunk);
          full->cuda->set_prefill_tokens(first_chunk);
          full->cuda->set_pos(start_pos);
          std::vector<sx::Bytes> in;
          in.push_back(encode_tokens(first_chunk));
          in.push_back(encode_i64(start_pos));
          std::expected<planner::ExecutionResult, planner::RuntimeError> res;
          {
            Phase _w("prefill:capture_warmup", phase_timing);
            res = full->exec->execute(full->plan, snap, in);
          }
          if (res) {
            full->cuda->synchronize();
            std::unique_ptr<hal::StepProgram> prog;
            {
              Phase _r("prefill:capture_record", phase_timing);
              prog = full->cuda->record_step([&]() {
                std::vector<sx::Bytes> cin;
                cin.push_back(encode_tokens(first_chunk));
                cin.push_back(encode_i64(start_pos));
                auto cres = full->exec->execute(full->plan, snap, cin);
                return cres.has_value();
              });
            }
            if (prog) {
              full->chunk_prog = std::move(prog);
              full->chunk_prog_cache = cache_id;
              progs[chunk] = full->chunk_prog.get();
            }
          }
        }
      }
    }

    const auto t_pf0 = std::chrono::steady_clock::now();
    for (uint64_t off = 0; off < n; ) {
      Phase _seg("prefill:segment", phase_timing);
      const uint64_t pos = start_pos + off;
      uint64_t nseg = std::min(chunk, n - off);
      // Align the segment boundary to an absolute multiple of `chunk` so a
      // segment never spans a ring-wraparound point: a KV slot is `pos %
      // n_slots` with n_slots == chunk for the SWA layers, so a segment whose
      // start_pos is not a multiple of `chunk` would wrap the ring and its
      // deferred append would overwrite slots a later position still reads. The
      // cap also makes a prefix-reuse suffix (start_pos not a multiple of
      // `chunk`) split at the same boundaries a full prefill would, so the two
      // paths write the same per-position KV.
      const uint64_t rem = pos % chunk;
      if (rem != 0 && chunk - rem < nseg)
        nseg = chunk - rem;
      const bool is_final = (off + nseg == n);
      PrefillSegment* seg = segment_for(nseg);
      if (!seg)
        return 0;

      auto prog_it = progs.find(nseg);
      hal::StepProgram* prog =
          (prog_it == progs.end()) ? nullptr : prog_it->second;
      if (!is_final && prog) {
        // Replay the captured full-chunk graph: stage this segment's token ids
        // and start position into the persistent device slots on the program's
        // own stream (async H2D, ordered before the replay), then launch the
        // graph on that same stream. Staging + replay both live on the program's
        // non-default stream, so consecutive segments serialize on the GPU while
        // the host never blocks (a synchronous H2D on the legacy default stream
        // would wait for every prior replay — the original per-segment host stall).
        seg->cuda->stage_prefill_tokens(*prog, prompt.subspan(off, nseg));
        seg->cuda->stage_pos(*prog, pos);
        if (!prog->replay()) {
          std::cerr << "[prefill-cuda] segment off=" << off
                    << " graph replay failed\n";
          return 0;
        }
        off += nseg;
        continue;
      }

      // Uncached (final segment, or capture disabled/fell back). The KV cache
      // is already shared — at build for a fresh segment, re-adopted on a cache
      // hit — so only the per-position bindings are set here.
      seg->cuda->set_prefill_tokens(prompt.subspan(off, nseg));
      seg->cuda->set_pos(pos);
      std::vector<sx::Bytes> in;
      in.push_back(encode_tokens(prompt.subspan(off, nseg)));
      in.push_back(encode_i64(pos));
      auto res = seg->exec->execute(seg->plan, snap, in);
      if (!res) {
        std::cerr << "[prefill-cuda] segment off=" << off << " execute failed: "
                  << res.error().message << "\n";
        return 0;
      }
      if (res->outputs.empty()) {
        std::cerr << "[prefill-cuda] segment off=" << off << " no outputs\n";
        return 0;
      }

      if (is_final) {
        // Final segment: its last hidden column is the prompt's final state.
        const sx::Bytes& hidden_b = res->outputs[0];
        if (hidden_b.size() < uint64_t(d) * nseg * sizeof(float)) {
          std::cerr << "[prefill-cuda] n=" << n << " hidden too small: "
                    << hidden_b.size() << " < " << uint64_t(d) * nseg * sizeof(float)
                    << "\n";
          return 0;
        }
        std::span<const float> hf(reinterpret_cast<const float*>(hidden_b.data()),
                                  uint64_t(d) * nseg);
        std::vector<float> hidden(d);
        for (uint32_t k = 0; k < d; ++k)
          hidden[k] = hf[(nseg - 1) * d + k];
        if (!lm_head(m, hidden, logits, quant::MatmulBackend::Cuda)) {
          std::cerr << "[prefill-cuda] n=" << n << " lm_head failed\n";
          return 0;
        }
        first = nn::argmax(logits);
        if (dump_logits) {
          std::vector<float> capped(vocab);
          nn::softcap(logits, m.config.final_logit_softcapping, capped);
          dump_logits_top2(dump_logits_file, /*step=*/0, start_pos + n, capped);
        }
      }
      off += nseg;
    }
    if (gen_timing) {
      using ms = std::chrono::duration<double, std::milli>;
      const double pf_ms = ms(std::chrono::steady_clock::now() - t_pf0).count();
      const double pp_tps = pf_ms > 0.0 ? double(n) * 1000.0 / pf_ms : 0.0;
      std::cerr << "[prefill-cuda] n=" << n << " prefill " << pf_ms << " ms ("
                << pp_tps << " tok/s)\n";
    }
  } else {
    // Reference: feed the prompt one token at a time through the CUDA spine
    // (greedy graph -> per-token argmax output), keeping the final token's index.
    // Each prompt token is seeded into the device token slot before its step
    // (Phase 3.2 B4), since the Embedding gather now reads the token id on the
    // device rather than the host graph input.
    float last_argmax = -1.0f;
    std::vector<float> last_capped;  // softcapped logits of the final prompt token
    for (uint64_t p : prompt) {
      rt->cuda->seed_token(p);
      rt->cuda->set_pos(pos);
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(p));
      in.push_back(encode_i64(pos));
      auto res = rt->exec->execute(rt->plan, snap, in);
      if (!res || res->outputs.empty()) return 0;
      const sx::Bytes& lo = res->outputs[0];
      if (dump_logits) {
        if (lo.size() < vocab * sizeof(float)) return 0;
        last_capped.assign(reinterpret_cast<const float*>(lo.data()),
                           reinterpret_cast<const float*>(lo.data()) + vocab);
        const int64_t a = nn::argmax(last_capped);
        if (a < 0) return 0;
        last_argmax = float(a);
      } else {
        if (lo.size() < sizeof(float)) return 0;
        last_argmax = *reinterpret_cast<const float*>(lo.data());
      }
      ++pos;
    }
    first = int64_t(last_argmax);
    if (dump_logits)
      dump_logits_top2(dump_logits_file, /*step=*/0, pos, last_capped);
  }

  if (first < 0) return 0;
  tokens[0] = first;
  token = static_cast<uint64_t>(first);
  pos = start_pos + prompt.size();
  step = 1;  // tokens[0] is `first`; the decode loop continues from index 1
  // The decode loop below reads the next token id from the device slot, which the
  // reference path already left at `first`; the prefill path never ran an
  // Embedding/Argmax step, so seed it explicitly (Phase 3.2 B4). The cached
  // decode runtime's capture is warmed with a placeholder token, so every call
  // re-seeds the real one (C.2: no re-capture per call).
  rt->cuda->seed_token(token);

  // Decode the remaining tokens with the same planner/backend/cache. When the
  // cached greedy decode step is captured (and we are not dumping logits, which
  // needs the full softcapped vector rather than the argmax scalar), replay it
  // per step: stage the position, launch the graph, read the winning token back
  // from the persistent device slot (the synchronous D2H also drains the graph
  // stream). Otherwise fall back to per-step executor dispatch.
  if (!dump_logits && rt->prog) {
    for (; step < tokens.size(); ++step) {
      rt->cuda->set_pos(pos);
      if (!rt->prog->replay()) {
        std::cerr << "[prefill-spine-cuda] step " << step
                  << " graph replay failed\n";
        break;
      }
      float next_f = -1.0f;
      rt->cuda->read_token(&next_f);
      const int64_t next = int64_t(next_f);
      if (next < 0) break;
      tokens[step] = next;
      token = static_cast<uint64_t>(next);
      ++pos;
    }
  } else {
    for (; step < tokens.size(); ++step) {
      rt->cuda->set_pos(pos);
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(token));
      in.push_back(encode_i64(pos));
      auto res = rt->exec->execute(rt->plan, snap, in);
      if (!res || res->outputs.empty()) {
        if (!res)
          std::cerr << "[prefill-spine-cuda] step " << step << " execute failed: "
                    << res.error().message << "\n";
        else
          std::cerr << "[prefill-spine-cuda] step " << step << " no outputs\n";
        break;
      }
      const sx::Bytes& lo = res->outputs[0];
      int64_t next;
      if (dump_logits) {
        if (lo.size() < vocab * sizeof(float)) break;
        std::span<const float> lf(reinterpret_cast<const float*>(lo.data()), vocab);
        next = nn::argmax(lf);
        if (next < 0) break;
        dump_logits_top2(dump_logits_file, step, pos, lf);
      } else {
        if (lo.size() < sizeof(float)) break;
        next = int64_t(*reinterpret_cast<const float*>(lo.data()));
        if (next < 0) break;
      }
      tokens[step] = next;
      token = static_cast<uint64_t>(next);
      ++pos;
    }
  }
  return step;
}

uint64_t generate_prefill_spine_cuda(const Gemma4Model& m,
                                     std::span<const uint64_t> prompt,
                                     uint64_t start_pos, std::span<int64_t> tokens,
                                     uint64_t n_ctx, bool use_prefill) {
  return prefill_decode_cuda_impl(m, prompt, start_pos, tokens, n_ctx,
                                  use_prefill, /*session=*/nullptr);
}

uint64_t generate_prefill_spine_cuda_prefix(const Gemma4Model& m,
                                            std::span<const uint64_t> prompt,
                                            uint64_t start_pos,
                                            std::span<int64_t> tokens,
                                            planner::KvCacheSession& session,
                                            bool use_prefill,
                                            uint64_t* n_reused_out) {
  if (n_reused_out) *n_reused_out = 0;
  if (m.weights.blocks.empty() || prompt.empty() || tokens.empty()) return 0;

  // Prefix reuse only anchors at position 0 (the session's token history is a
  // contiguous run from 0); a non-zero start_pos disables reuse.
  uint64_t prefix_len =
      (start_pos == 0) ? session.common_prefix_len(prompt) : 0;

  // A prompt the session already holds in full leaves an empty suffix, and
  // prefilling nothing produces no logits at all (the backend returns 0, which
  // the C ABI reads as a failure). The activation of the *last* prompt token is
  // what yields the next-token logits, so that token always has to be
  // processed: keep all but it resident and prefill it as a one-token suffix.
  // The reported reuse count stays honest — this is exactly what was reused.
  if (prefix_len == prompt.size())
    prefix_len = prompt.size() - 1;
  if (n_reused_out) *n_reused_out = prefix_len;

  // The cache size is the session's n_ctx (pre-set by the caller for its largest
  // request); the first call sizes it from its own prompt when session.n_ctx is
  // zero. A request larger than the session's cache is rejected outright rather
  // than silently wrapping the global layers' ring.
  const uint64_t required = start_pos + prompt.size() + tokens.size();
  const uint64_t n_ctx = (session.n_ctx != 0) ? session.n_ctx : required;
  if (n_ctx < required) return 0;

  const uint64_t prefill_start = start_pos + prefix_len;
  const std::span<const uint64_t> suffix = prompt.subspan(prefix_len);

  const uint64_t done = prefill_decode_cuda_impl(m, suffix, prefill_start,
                                                 tokens, n_ctx, use_prefill,
                                                 &session);
  if (done == 0) return 0;

  // Record every token whose KV this call actually wrote, for the next
  // request's prefix match: the prompt, then the generated tokens the decode
  // loop fed back. The loop at `step` fed tokens[0..step-2] (tokens[0] is fed at
  // step 1 and the last generated token is never fed — it was only sampled), so
  // the cache holds prompt + tokens[0..done-2]. Recording only the prompt (the
  // old behaviour) re-prefilled a session's own generation on every later turn.
  if (start_pos == 0) {
    session.filled_tokens.assign(prompt.begin(), prompt.end());
    const uint64_t fed = (done > 1) ? done - 1 : 0;
    for (uint64_t j = 0; j < fed; ++j)
      session.filled_tokens.push_back(static_cast<uint64_t>(tokens[j]));
  } else {
    session.filled_tokens.clear();
  }
  return done;
}
#else
uint64_t generate_prefill_spine_cuda(const Gemma4Model&,
                                     std::span<const uint64_t>, uint64_t,
                                     std::span<int64_t>, uint64_t, bool) {
  return 0;  // built without CUDA: no device prefill is available
}
uint64_t generate_prefill_spine_cuda_prefix(const Gemma4Model&,
                                            std::span<const uint64_t>, uint64_t,
                                            std::span<int64_t>,
                                            planner::KvCacheSession&, bool,
                                            uint64_t* n_reused_out) {
  if (n_reused_out) *n_reused_out = 0;
  return 0;  // built without CUDA: no device prefill is available
}
#endif

} // namespace sonicboom::model
