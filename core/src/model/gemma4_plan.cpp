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

#include <sonicboom/runtime_features.h>
#include <sonicboom/compiler/BoomBuilder.h>
#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomPasses.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/numerics.h>
#include <sonicboom/planner/cost_model.h>
#include <sonicboom/planner/draft_model.h>
#include <sonicboom/planner/ir_passes.h>
#include <sonicboom/planner/pipeline.h>
#include <sonicboom/planner/resource.h>
#include <sonicboom/planner/runtime_executor.h>
#include <sonicboom/planner/sonic_backend.h>
#include <sonicboom/planner/sonic_cuda_backend.h>
#include <sonicboom/sx/exec.h>

#include "../planner/autotune.h"        // B7 draft-length cost cache
#include "../planner/decision_registry.h"  // the plan-cache decision fingerprint
#include "../planner/mlir_autotune.h"    // V2 spec probe ratio cache (Q1 file)
#include "../planner/reduce_mode.h"      // the per-device reduction mode
#include "mlir/IR/BuiltinTypes.h"

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>  // cudaGetDeviceProperties (record device line)
#include <nvtx3/nvToolsExt.h>  // header-only NVTX (no libnvtx link)
#include <sonicboom/quant/quantized_matmul_cuda.h>
#endif

#include <algorithm>
#include <array>
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
#include <tuple>
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
sx::Attribute attr_s(const char* name, std::string v) {
  return sx::Attribute{name, std::move(v)};
}

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
planner::Graph lower_graph(mlir::ModuleOp module,
                           uint64_t dynamic_extent = 0) {
  auto graph = compiler::boomToGraph(module, dynamic_extent);
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
planner::Graph lower_graph(GraphIr& ir, planner::IrPassPhase phase, bool rewrite,
                           uint64_t dynamic_extent = 0,
                           bool skip_kv_append_fusion = false) {
  if (!ir.module)
    return {};
  if (rewrite &&
      !compiler::runBoomPasses(*ir.module, phase, skip_kv_append_fusion))
    return {};
  return lower_graph(*ir.module, dynamic_extent);
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
// `rows` > 1 emits the same graph dealt over that many query positions
// (position-major [rows, dim] activations, each node stamped with a `verify`
// attribute): the row-invariant verify step of §B3. The backend then runs the
// decode kernel per row, with the row-invariant batched mmvq for the quantized
// matmuls — so row i is bit-identical to a plain decode step at pos+i.
// `softcap_output` false keeps the raw logits as the graph output even when the
// model has a logit softcap. The softcap is monotonic (argmax unchanged), but it
// is a lossy transform, so the perplexity scorer needs the raw distribution
// while the logit dump and the sampler use the softcapped one.
GraphIr emit_decode_ir(const Gemma4Model& m, bool greedy, uint64_t rows = 1,
                       bool softcap_output = true) {
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
  compiler::BoomBuilder b(ir.context.get(),
                          rows > 1 ? "gemma4_verify" : "gemma4_decode");
  if (rows > 1) b.set_row_scale(int64_t(rows), "verify");

  // Graph inputs: token id + absolute position (Int64 scalars), declared to the
  // builder — they are the graph's free variables, and a free variable is a block
  // argument in the IR the emitter writes.
  const mlir::Value token_id = b.input("token_id", {}, sx::DType::Int64);
  const mlir::Value pos = b.input("pos", {}, sx::DType::Int64);

  // The per-layer KV cache handles: the cache a node reads or appends to is a
  // `!boom.kv<layer>` operand stating the access as an effect, not a pointer
  // bound at plan time. A shared layer's attention takes its donor's handle.
  const uint32_t n_kv = cfg.n_layer_kv();
  std::vector<mlir::Value> kv_handles(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l)
    kv_handles[l] = b.kv(l);

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
      attn = b.node(OpKind::Attention, {q, kc, vc, pos, kv_handles[l]},
                    "attn" + suff, {int64_t(qw)}, attn_attrs(l));
    } else {
      attn = b.node(OpKind::Attention,
                    {q, pos, kv_handles[cfg.kv_donor_layer(l)]}, "attn" + suff,
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
  } else if (prof.softcap && softcap_output) {
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
GraphIr emit_prefill_ir(const Gemma4Model& m, uint64_t n,
                        bool dynamic_extent = false) {
  using namespace planner;

  const auto& cfg = m.config;
  const uint32_t d = cfg.embedding_length;
  const uint32_t ff = cfg.feed_forward_length;
  const uint32_t per = cfg.per_layer_input;
  const uint32_t block_count = cfg.block_count;
  const uint32_t total_per = block_count * per;

  GraphIr ir;
  ir.context = std::make_unique<mlir::MLIRContext>();
  // Sized at `n`: a dynamic (symbolic-extent) build still records every byte
  // count at the segment length it was emitted for, so the passes see the same
  // graph the concrete build at `n` carries and choose the same fusions. The
  // lowering re-derives the sizes from the extent it binds.
  compiler::BoomBuilder b(ir.context.get(), "gemma4_prefill", n);

  // Graph inputs: the prompt token ids (Int64 [n]) and the absolute start
  // position (Int64 scalar).
  const mlir::Value tokens = b.input(
      "tokens",
      {dynamic_extent ? mlir::ShapedType::kDynamic : int64_t(n)},
      sx::DType::Int64);
  const mlir::Value start_pos = b.input("start_pos", {}, sx::DType::Int64);

  // The per-layer KV cache handles (see emit_decode_ir): the cache a node reads
  // or appends to is a `!boom.kv<layer>` operand stating the access as an
  // effect, not a pointer bound at plan time.
  const uint32_t n_kv = cfg.n_layer_kv();
  std::vector<mlir::Value> kv_handles(n_kv);
  for (uint32_t l = 0; l < n_kv; ++l)
    kv_handles[l] = b.kv(l);

  auto slot = [](WeightSlot s) { return attr_i("slot", int64_t(s)); };
  auto layer_a = [](uint32_t l) { return attr_i("layer", int64_t(l)); };
  auto kv_src = [&cfg](uint32_t l) {
    return attr_i("kv_source_layer", int64_t(cfg.kv_donor_layer(l)));
  };
  auto batch_a = [](uint64_t v) { return attr_i("batch", int64_t(v)); };
  auto dim_a = [](uint64_t v) { return attr_i("dim", int64_t(v)); };
  // The MMQ work decomposition for this weight shape (Stage 1): the runtime's
  // policy (mmq_class_decomp) over the projection's (rows, K, quant type) and
  // this device's SM count. Deliberately independent of the segment length n —
  // split-K changes the summation order, so a batch-size-dependent choice would
  // make prefix-cache reuse diverge from a full prefill. Recorded on every
  // batched matmul so the plan says which kernel the class runs; the runtime
  // resolves Auto (and unannotated nodes) with the same policy. A CPU-only
  // build has no MMQ to classify and records tiled.
  auto decomp_a = [](uint64_t rows, uint64_t cols, quant::QuantType type) {
#ifdef SONICBOOM_USE_CUDA
    const bool stream_k = quant::cuda::mmq_class_decomp(rows, cols, type) ==
                          quant::cuda::MmqDecomp::StreamK;
    return attr_s("decomp", stream_k ? "stream_k" : "tiled");
#else
    (void)rows;
    (void)cols;
    (void)type;
    return attr_s("decomp", "tiled");
#endif
  };
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
  // Every activation is [n, width]. A symbolic segment emits the token
  // dimension as `?` (the graph's dynamic extent) and leaves the width static:
  // the shape function that makes this a valid graph is the verifier's
  // shape-preserving rule plus the emitter's [n, width] layout, and the
  // lowering binds `?` to the shape class's maximum (see boomToGraph).
  auto shape2 = [dynamic_extent](uint64_t r, uint64_t c) {
    return std::vector<int64_t>{
        dynamic_extent ? mlir::ShapedType::kDynamic : int64_t(r), int64_t(c)};
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
                        dim_a(d), ds_a(m.weights.blocks[l].attn_q.type()), decomp_a(qw, d, m.weights.blocks[l].attn_q.type())});
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
                          dim_a(d), ds_a(m.weights.blocks[l].attn_k.type()), decomp_a(kvw, d, m.weights.blocks[l].attn_k.type())});
      mlir::Value v = b.node(OpKind::QuantizedMatmul, {h}, "v" + suff,
                          shape2(n, kvw), {layer_a(l), slot(WeightSlot::AttnV), batch_a(n),
                          dim_a(d), ds_a(m.weights.blocks[l].attn_v.type()), decomp_a(kvw, d, m.weights.blocks[l].attn_v.type())});
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
      attn = b.node(OpKind::FlashAttention,
                    {q, kc, vc, start_pos, kv_handles[l]}, "attn" + suff,
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
      attn = b.node(OpKind::FlashAttention,
                    {q, dk, dv, start_pos, kv_handles[donor]}, "attn" + suff,
                    shape2(n, qw), attn_attrs(l));
    }

    mlir::Value ap = b.node(OpKind::QuantizedMatmul, {attn}, "ap" + suff,
                         shape2(n, d), {layer_a(l), slot(WeightSlot::AttnOutput), batch_a(n),
                         dim_a(qw), ds_a(m.weights.blocks[l].attn_output.type()), decomp_a(d, qw, m.weights.blocks[l].attn_output.type())});
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
                         dim_a(d), ds_a(m.weights.blocks[l].ffn_up.type()), decomp_a(ff, d, m.weights.blocks[l].ffn_up.type())});
    mlir::Value gate = b.node(OpKind::QuantizedMatmul, {f}, "gate" + suff, shape2(n, ff),
                           {layer_a(l), slot(WeightSlot::FfnGate), batch_a(n),
                           dim_a(d), ds_a(m.weights.blocks[l].ffn_gate.type()), decomp_a(ff, d, m.weights.blocks[l].ffn_gate.type())});
    gate = prof.silu_ffn
               ? b.node(OpKind::Silu, {gate}, "silu" + suff, shape2(n, ff), {layer_a(l)})
               : b.node(OpKind::GeluFp16, {gate}, "gelu" + suff, shape2(n, ff),
                        {layer_a(l)});
    mlir::Value fh = b.node(OpKind::Mul, {gate, up}, "fh" + suff, shape2(n, ff), {layer_a(l)});
    mlir::Value fo = b.node(OpKind::QuantizedMatmul, {fh}, "fo" + suff, shape2(n, d),
                         {layer_a(l), slot(WeightSlot::FfnDown), batch_a(n),
                         dim_a(ff), ds_a(m.weights.blocks[l].ffn_down.type()), decomp_a(d, ff, m.weights.blocks[l].ffn_down.type())});
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
    b.node(OpKind::KvAppend, {kv_k[l], kv_v[l], start_pos, kv_handles[l]},
           "kv_append" + std::to_string(l), {}, {layer_a(l), batch_a(n)});
  }

  b.set_output(residual);

  ir.module = b.finish();
  return ir;
}

// The prefill emitter's IR, lowered with no passes, like build_gemma4_graph.
// `symbolic` emits the token dimension as `?` and binds it to `n`, so the graph
// is the same graph a concrete `n` produces plus the dynamic extent record a
// smaller execution can bind (TensorDesc::dynamic_stride_bytes).
planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n,
                                          bool symbolic) {
  GraphIr ir = emit_prefill_ir(m, n, symbolic);
  if (!ir.module)
    return {};
  return lower_graph(*ir.module, symbolic ? n : 0);
}

planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n) {
  return build_gemma4_prefill_graph(m, n, /*symbolic=*/false);
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

// B6: the accept/emit decision (plan.h). Model- and device-independent, so it
// lives outside the CUDA section and is unit-tested directly.
SpecAccept spec_accept_run(std::span<const uint64_t> drafts,
                           std::span<const int64_t> row_argmax,
                           std::span<const int64_t> eos_tokens,
                           uint64_t emitted_before, uint64_t limit,
                           std::span<uint64_t> out) {
  SpecAccept r;
  if (drafts.empty() || row_argmax.size() < drafts.size() + 1 ||
      out.size() < drafts.size() + 1)
    return r;
  while (r.accepted < drafts.size() &&
         row_argmax[size_t(r.accepted)] == int64_t(drafts[size_t(r.accepted)]))
    ++r.accepted;
  for (uint64_t i = 0; i <= r.accepted && emitted_before + i < limit; ++i) {
    const uint64_t t =
        i < r.accepted ? drafts[size_t(i)] : uint64_t(row_argmax[size_t(r.accepted)]);
    out[size_t(r.emitted++)] = t;
    if (std::find(eos_tokens.begin(), eos_tokens.end(), int64_t(t)) !=
        eos_tokens.end()) {
      r.stopped = true;
      break;
    }
  }
  return r;
}

// B7: the adaptive draft-length policy (plan.h). The objective is emitted
// tokens per unit of iteration time: with acceptance rate alpha, k drafts yield
// the expected run E(k) = 1 + alpha + ... + alpha^k, so choose the k that
// maximizes E(k) / c(k), c(k) being one iteration's measured cost.
int64_t spec_choose_drafts(double alpha, std::span<const double> cost_us,
                           int64_t max_k) {
  if (max_k < 0) return 0;
  alpha = std::clamp(alpha, 0.0, 1.0);
  const auto expected = [alpha](int64_t k) {
    if (alpha >= 1.0 - 1e-9) return double(k) + 1.0;  // the limit of the sum
    return (1.0 - std::pow(alpha, double(k) + 1.0)) / (1.0 - alpha);
  };
  int64_t best_k = max_k;  // nothing measured: the fixed cap
  double best = 0.0;
  bool any = false;
  for (int64_t k = 0; k <= max_k && size_t(k) < cost_us.size(); ++k) {
    if (!(cost_us[size_t(k)] > 0.0)) continue;  // unmeasured
    const double score = expected(k) / cost_us[size_t(k)];
    if (!any || score > best) {
      any = true;
      best = score;
      best_k = k;
    }
  }
  return best_k;
}

double spec_alpha_update(double alpha, uint64_t accepted, uint64_t drafted) {
  if (drafted == 0) return alpha;
  const double x = double(accepted) / double(drafted);
  return 0.5 * alpha + 0.5 * x;
}

namespace {
SpecCounters g_spec_counters;
} // namespace

SpecCounters spec_counters() { return g_spec_counters; }
void spec_counters_reset() { g_spec_counters = SpecCounters{}; }

// ---------------------------------------------------------------------------
// S4: per-step speculative diagnostics. SONICBOOM_SPEC_DIAG=1 records every
// iteration of a speculative generation call into a 1024-entry ring — the
// iteration's position, the proposal length k, how many drafts the verifier
// accepted, how many tokens were emitted, and the n-gram length that produced
// the proposal (0 for a miss, the forced bootstrap/warm-up steps and the
// repeat-drafter seam) — plus a histogram of the call's emitted tokens, and
// dumps both to stderr when the call returns. Recording is off by default and
// allocates nothing; when on it is a handful of integer stores per iteration,
// and the histogram is the only allocation (diagnostic runs are short).
//
// The point is the diagnosis the policy cannot make from its counters alone:
// did the drafter miss (match_length 0), hit a short n-gram, or hit a long one
// the policy chose not to verify? The token entropy separates a copy-heavy
// stream (low entropy: the same few tokens recur, prompt lookup should hit) from
// free-form text (high entropy: no repetition to look up).
enum class SpecDiagKind : uint8_t {
  Plain = 0,      // k == 0: the captured single-token decode step
  Bootstrap,      // S2 forced exploration (k = kSpecBootstrapDrafts)
  Warm,           // B7 cost warm-up (forced k)
  Policy,         // the adaptive policy chose k
  Fixed,          // spec=on: the drafter's fixed cap chose k
};

struct SpecDiagEntry {
  uint64_t step, pos;
  uint32_t k, accepted, emitted;
  uint8_t ngram;  // the n-gram length that hit (2..4), 0 = no proposal
  uint8_t kind;
  uint32_t alpha_milli;  // the policy's acceptance EMA at decision time
  uint64_t dt_us;        // wall time of this iteration (propose + step)
};

constexpr size_t kSpecDiagRing = 1024;

struct SpecDiagState {
  bool init = false;
  bool enabled = false;
  uint64_t total = 0;    // entries recorded this call (the ring keeps the last
  uint64_t calls = 0;    // kSpecDiagRing of them)
  SpecDiagEntry ring[kSpecDiagRing]{};
  std::map<uint64_t, uint64_t> hist;  // emitted-token counts, this call
};

SpecDiagState& spec_diag() {
  static SpecDiagState d;
  return d;
}

static bool spec_diag_enabled() {
  SpecDiagState& d = spec_diag();
  if (!d.init) {
    const char* e = std::getenv("SONICBOOM_SPEC_DIAG");
    d.enabled = e != nullptr && std::string_view(e) == "1";
    d.init = true;
  }
  return d.enabled;
}

static void spec_diag_record(uint64_t step, uint64_t pos, size_t k,
                             size_t accepted, size_t emitted, size_t ngram,
                             SpecDiagKind kind, double alpha,
                             std::chrono::steady_clock::time_point iter_start) {
  SpecDiagState& d = spec_diag();
  SpecDiagEntry& e = d.ring[size_t(d.total % kSpecDiagRing)];
  const double a = std::clamp(alpha, 0.0, 1.0);
  const uint64_t dt_us = uint64_t(std::chrono::duration<double, std::micro>(
                                      std::chrono::steady_clock::now() -
                                      iter_start)
                                      .count());
  e = SpecDiagEntry{step, pos, uint32_t(k), uint32_t(accepted),
                    uint32_t(emitted), uint8_t(ngram), uint8_t(kind),
                    uint32_t(a * 1000.0 + 0.5), dt_us};
  ++d.total;
}

static void spec_diag_emit(uint64_t token) {
  ++spec_diag().hist[token];
}

// Dump this call's ring (oldest first) and summary. The entropy is Shannon
// entropy in bits over the call's emitted token ids.
static void spec_diag_dump(const char* strategy) {
  SpecDiagState& d = spec_diag();
  if (!d.enabled) return;
  const char* kinds[] = {"plain", "bootstrap", "warm", "policy", "fixed"};
  const uint64_t shown = std::min<uint64_t>(d.total, kSpecDiagRing);
  const uint64_t first = d.total - shown;
  std::fprintf(stderr, "[spec-diag] call=%llu strategy=%s steps=%llu shown=%llu\n",
               (unsigned long long)++d.calls, strategy,
               (unsigned long long)d.total, (unsigned long long)shown);
  for (uint64_t i = first; i < d.total; ++i) {
    const SpecDiagEntry& e = d.ring[size_t(i % kSpecDiagRing)];
    std::fprintf(stderr,
                 "[spec-diag] step=%llu pos=%llu k=%u acc=%u emit=%u n=%u "
                 "alpha=%.3f dt_us=%llu kind=%s\n",
                 (unsigned long long)e.step, (unsigned long long)e.pos, e.k,
                 e.accepted, e.emitted, unsigned(e.ngram),
                 double(e.alpha_milli) / 1000.0, (unsigned long long)e.dt_us,
                 kinds[e.kind]);
  }
  uint64_t tokens = 0;
  for (const auto& [id, count] : d.hist) tokens += count;
  double entropy = 0.0;
  if (tokens > 0) {
    for (const auto& [id, count] : d.hist) {
      const double p = double(count) / double(tokens);
      entropy -= p * std::log2(p);
    }
  }
  std::fprintf(stderr,
               "[spec-diag] summary tokens=%llu distinct=%zu entropy_bits=%.3f "
               "dropped=%llu\n",
               (unsigned long long)tokens, d.hist.size(), entropy,
               (unsigned long long)(d.total - shown));
  d.total = 0;
  d.hist.clear();
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
  // MMQ1's dispatch switch (compiler/BoomPasses.h): the prefill graph differs
  // between the generated tensor-core kernel and the ggml mmq path, so a
  // segment planned under one is stale under the other.
  compiler::MmqTcMode mmq_tc;
  int mmq_tc_max_mb;
  // VS1's prefetch switch (compiler/BoomPasses.h): the markers the pass inserts
  // are part of the graph, and the threshold decides whether they appear, so a
  // segment planned under one setting is stale under another.
  compiler::AsyncPrefetchMode async_prefetch;
  int vram_threshold_mb;
  // The norm kernels' reduction mode (compiler/ReduceMode.h). The graph is the
  // same, but a cached segment's launches and captured graph belong to the
  // mode they were built under, so a mode change invalidates the cache.
  compiler::ReduceMode reduce;
  // The numerics profile (sonicboom/numerics.h). It selects the KV-cache
  // element type, which is part of the graphs' buffers, so a segment planned
  // under one profile is stale under the other.
  NumericsProfile numerics;
  // Fingerprint of every automatic decision recorded so far
  // (planner/decision_registry.h). A cached segment's launches encode the
  // decisions they were built under; when a later decision changes the set,
  // the fingerprint changes and the entries above are dropped.
  uint64_t decisions;
  friend bool operator==(const PlannerEnv&, const PlannerEnv&) = default;
};

static PlannerEnv planner_env() {
  // The getters first (they record the decisions they resolve), then the
  // fingerprint over the resulting set.
  PlannerEnv env{quant::cuda::q8_1_enabled(), quant::cuda::mmq_enabled(),
                 quant::cuda::mmq_min_n(), compiler::mmq_tc_mode(),
                 compiler::mmq_tc_max_mb(), compiler::async_prefetch_mode(),
                 compiler::vram_threshold_mb(),
                 planner::device_reduce_mode(), numerics_profile(), 0};
  planner::decisions::seed_static_knobs();
  env.decisions = planner::decisions::fingerprint();
  return env;
}

// The prefill segment cache is a runtime feature (plan_cache), so it can be
// ablated with SONICBOOM_FEATURE_DISABLE=plan_cache; the historical
// SONICBOOM_PLAN_CACHE=0 remains as its alias.
static bool plan_cache_enabled() {
  return runtime::feature_enabled(runtime::Feature::PlanCache);
}

// Cache cap. A symbolic plan serves its whole shape class, so a session needs
// at most two entries per model (the full-chunk class and the one-token class)
// no matter how many distinct segment lengths it visits. The cap still bounds
// entries across models and planner environments; the entry holds the plan and
// the segment backends, whose device activations come from the global device
// pool, so the cap bounds host-side planning state rather than VRAM.
static constexpr std::size_t kMaxCachedSegments = 8;

// The greedy generation program (§10), read out of the `boom.program` the
// module's decode graph lifts into. It is the loop's structure — a prefill
// call, then a body of step / sample / emit / stop_check and a pos_inc
// continuation — not a second plan: the step's device work stays the captured
// StepProgram the plan produced, and the executor below walks the program to
// decide when to replay it, which token to emit, and when to stop.
//
// The desc is plain data so the runtime never holds an MLIR module alive; the
// lift runs once at plan-build time and its diagnostics fail the plan there.
struct GreedyProgramDesc {
  bool valid = false;
  // A speculative program (boom.draft / verify / accept in place of the
  // sample). The step node is still present: it is the drafter's k == 0
  // fallback, replayed as the captured decode step.
  bool speculative = false;
  std::string prefill_graph;
  std::string verify_graph;    // the graph the verify node deals over rows
  int64_t max_rows = 0;        // its row budget (max_drafts + 1)
  std::string draft_strategy;  // "ngram"
  int64_t max_drafts = 0;      // the drafter's proposal cap
  std::vector<int64_t> eos_tokens;
  int64_t max_tokens = 0;  // 0 = the caller's token span is the bound
};

struct DecodeRuntime {
  planner::Graph graph;
  planner::ExecutionPlan plan;
  planner::ResourceSnapshot snap;
  std::unique_ptr<planner::SonicBackend> cpu;         // host boundary tasks
  std::unique_ptr<planner::SonicCudaBackend> cuda;    // device-resident compute
  std::unique_ptr<planner::RuntimeExecutor> exec;
  std::unique_ptr<hal::StepProgram> prog;             // captured greedy decode step
  // The cache vector `prog` was recorded against. A capture bakes the KV cache's
  // device pointers in, and a prefill session can adopt a different vector; a
  // caller that replays `prog` must state that the cache it shares is this one
  // (decode_steps_spine_cuda does; B3's reference).
  std::shared_ptr<void> prog_cache;
  GreedyProgramDesc program;                          // the lowered loop (§10)
  // Prefill segments by shape-class sizing extent, most-recently used first,
  // capped at kMaxCachedSegments.
  std::list<std::pair<uint64_t, std::unique_ptr<PrefillSegment>>> prefills;
  // The planner environment `prefills` was built under; a change clears it.
  PlannerEnv prefill_env{};
  bool prefill_env_set = false;
};

// SONICBOOM_IR_PROGRAM=0 runs the hand-written decode loop instead of the
// program-driven one (ablation group "ir-program"). Default: the program.
static bool ir_program_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("SONICBOOM_IR_PROGRAM");
    return !(e && std::string_view(e) == "0");
  }();
  return v;
}

// The speculative draft cap (B6). One speculative step runs at most
// kSpecMaxDrafts + 1 verify rows, which is the envelope the row-invariant
// verify step was built and tested for (B3's verify_k1..k7 classes); B7's
// per-step length policy stays inside it.
constexpr int64_t kSpecMaxDrafts = 7;

// S2 bootstrap draft length: the adaptive policy's forced exploration length.
// S3 starts at k = 1: the bootstrap only has to supply the EMA with accept /
// reject data, and on a workload without matches every extra forced draft is
// verify cost the abort cannot refund.
constexpr int64_t kSpecBootstrapDrafts = 1;

// SONICBOOM_SPEC selects the generation loop:
//   off  the plain greedy loop (B1's program, step per token);
//   on   the speculative program: draft, verify, accept;
//   auto the same program with the draft length decided by policy (B7).
// Unset is V2's probe: the first generation call measures one plain decode
// step against one k=1 verify step on this GPU (cached per SM in the Q1
// autotune file) and picks auto below the threshold, off at or above it. The
// probe is a startup cost, once per GPU; an explicit SONICBOOM_SPEC bypasses
// it. Until the probe has decided, Undecided answers like Off, so the greedy
// runtime is built (and the probe then rebuilds it if it selects auto).
enum class SpecMode { Undecided, Off, On, Auto };

// The explicit environment choice. Undecided means "no override": the probe
// decides, once, and stores the result in g_spec_mode.
static SpecMode env_spec_mode() {
  static const SpecMode mode = [] {
    const char* e = std::getenv("SONICBOOM_SPEC");
    if (!e || !*e) return SpecMode::Undecided;
    if (std::string_view(e) == "off") return SpecMode::Off;
    if (std::string_view(e) == "on") return SpecMode::On;
    if (std::string_view(e) == "auto") return SpecMode::Auto;
    std::cerr << "[spec] unknown SONICBOOM_SPEC='" << e
              << "' (expected off|on|auto); using off\n";
    return SpecMode::Off;
  }();
  return mode;
}

// V2's resolved choice (see spec_resolve_probe). Only read after the
// environment left the mode undecided.
static SpecMode g_spec_mode = SpecMode::Undecided;

static SpecMode spec_mode() {
  const SpecMode env = env_spec_mode();
  return env != SpecMode::Undecided ? env : g_spec_mode;
}

static bool spec_enabled() {
  const SpecMode m = spec_mode();
  return m == SpecMode::On || m == SpecMode::Auto;
}

// Read the program back out of the lifted module. The program verifier has
// already checked the shape; this only recovers the attributes the executor
// needs (the prefill call's name, the stop set and the emitted-count bound, and
// the speculation pipeline's drafter/verify description) and confirms that one
// of each loop op is present.
static GreedyProgramDesc read_greedy_program(mlir::ModuleOp module) {
  namespace boom = compiler::boom;
  GreedyProgramDesc desc;
  boom::ProgramOp program;
  for (mlir::Operation& op : module.getBody()->getOperations()) {
    if (auto p = llvm::dyn_cast<boom::ProgramOp>(&op)) {
      if (program) return desc;  // more than one program: not a generation
      program = p;
    }
  }
  if (!program) return desc;

  int prefills = 0, steps = 0, samples = 0, drafts = 0, verifies = 0;
  int accepts = 0, emits = 0, stops = 0, incs = 0, dynamic_incs = 0;
  int64_t delta = 0;
  program.walk([&](mlir::Operation* op) {
    if (auto p = llvm::dyn_cast<boom::PrefillOp>(op)) {
      ++prefills;
      desc.prefill_graph = p.getGraphName().str();
    } else if (llvm::isa<boom::StepOp>(op)) {
      ++steps;
    } else if (llvm::isa<boom::SampleOp>(op)) {
      ++samples;
    } else if (auto d = llvm::dyn_cast<boom::DraftOp>(op)) {
      ++drafts;
      desc.draft_strategy = d.getStrategy().str();
      desc.max_drafts = int64_t(d.getMaxDrafts());
    } else if (auto v = llvm::dyn_cast<boom::VerifyOp>(op)) {
      ++verifies;
      desc.verify_graph = v.getGraphName().str();
      desc.max_rows = int64_t(v.getMaxRows());
    } else if (llvm::isa<boom::AcceptOp>(op)) {
      ++accepts;
    } else if (llvm::isa<boom::EmitOp>(op)) {
      ++emits;
    } else if (auto s = llvm::dyn_cast<boom::StopCheckOp>(op)) {
      ++stops;
      desc.eos_tokens.assign(s.getEosTokens().begin(), s.getEosTokens().end());
      desc.max_tokens = int64_t(s.getMaxTokens());
    } else if (auto p = llvm::dyn_cast<boom::PosIncOp>(op)) {
      ++incs;
      delta = int64_t(p.getDelta());
      if (p.getBy()) ++dynamic_incs;
    }
  });
  const bool spec = drafts == 1 && verifies == 1 && accepts == 1 &&
                    samples == 0 && dynamic_incs == 1 && desc.max_drafts > 0 &&
                    desc.max_rows == desc.max_drafts + 1;
  const bool greedy = drafts == 0 && verifies == 0 && accepts == 0 &&
                      samples == 1 && dynamic_incs == 0 && delta == 1;
  desc.valid = prefills == 1 && steps == 1 && emits == 1 && stops == 1 &&
               incs == 1 && (spec || greedy);
  desc.speculative = desc.valid && spec;
  return desc;
}

// The stop_check node: EOS membership, and the emitted-count bound. The count
// the program carries is the number of tokens emitted before this one, so the
// bound is tested against it.
static bool program_stop(const GreedyProgramDesc& p, int64_t token,
                         uint64_t emitted_before) {
  if (p.max_tokens > 0 && emitted_before >= uint64_t(p.max_tokens))
    return true;
  return std::find(p.eos_tokens.begin(), p.eos_tokens.end(), token) !=
         p.eos_tokens.end();
}

// Execute the program's decode loop from `step` (the caller has already emitted
// `step` tokens and `token` is the next input, at `pos`). The step node is one
// StepProgram replay — k chained steps per replay in chunk mode — the sample
// node is the argmax the replay wrote, emit writes the host token, stop_check
// tests the program's stop set and pos_inc advances the position. Returns the
// number of tokens written.
static uint64_t run_program_steps(
    DecodeRuntime& rt, const GreedyProgramDesc& p, uint64_t token, uint64_t pos,
    std::span<int64_t> tokens, uint64_t step,
    std::chrono::steady_clock::time_point* first_out = nullptr) {
  const uint64_t limit = p.max_tokens > 0
                             ? std::min<uint64_t>(uint64_t(p.max_tokens),
                                                  tokens.size())
                             : tokens.size();
  const uint32_t chunk = rt.cuda->decode_chunk();
  if (rt.prog && chunk > 1) {
    // One replay per k loop iterations: the step node is executed k times, and
    // sample/emit/stop_check/pos_inc run on each of its k results in order.
    float carry = float(token);
    std::vector<uint64_t> positions(chunk);
    std::vector<float> out(chunk);
    while (step < limit) {
      for (uint32_t j = 0; j < chunk; ++j) positions[j] = pos + j;
      const std::string name = "decode_chunk_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(name.c_str());
      const bool ok = rt.cuda->stage_chunk(*rt.prog, carry, positions) &&
                      rt.prog->replay() &&
                      rt.cuda->read_chunk_tokens(out.data(), chunk);
      if (nvtx_enabled()) nvtxRangePop();
      if (!ok) {
        std::cerr << "[program] chunk at step " << step << " failed\n";
        break;
      }
      bool stopped = false;
      for (uint32_t j = 0; j < chunk && step < limit; ++j) {
        const int64_t next = int64_t(out[j]);
        if (next < 0) { stopped = true; break; }
        tokens[step] = next;
        const bool stop = program_stop(p, next, step);
        carry = out[j];
        ++pos;
        ++step;
        if (first_out && step == 1) *first_out = std::chrono::steady_clock::now();
        if (stop) { stopped = true; break; }
      }
      if (stopped) break;
    }
  } else if (rt.prog) {
    for (; step < limit; ++step) {
      rt.cuda->set_pos(pos);
      if (!rt.prog->replay()) {
        std::cerr << "[program] step " << step << " graph replay failed\n";
        break;
      }
      float next_f = -1.0f;
      rt.cuda->read_token(&next_f);
      const int64_t next = int64_t(next_f);
      if (next < 0) break;
      tokens[step] = next;
      const bool stop = program_stop(p, next, step);
      token = static_cast<uint64_t>(next);
      ++pos;
      if (first_out && step == 0) *first_out = std::chrono::steady_clock::now();
      if (stop) { ++step; break; }
    }
  } else {
    // No re-playable graph: the step node is one executor dispatch.
    for (; step < limit; ++step) {
      rt.cuda->set_pos(pos);
      std::vector<sx::Bytes> in;
      in.push_back(encode_i64(token));
      in.push_back(encode_i64(pos));
      auto res = rt.exec->execute(rt.plan, rt.snap, in);
      if (!res || res->outputs.empty()) {
        if (!res)
          std::cerr << "[program] step " << step << " execute failed: "
                    << res.error().message << "\n";
        else
          std::cerr << "[program] step " << step << " no outputs\n";
        break;
      }
      const sx::Bytes& arg = res->outputs[0];
      if (arg.size() < sizeof(float)) {
        std::cerr << "[program] step " << step << " argmax too small: "
                  << arg.size() << " < " << sizeof(float) << "\n";
        break;
      }
      const int64_t next = int64_t(*reinterpret_cast<const float*>(arg.data()));
      if (next < 0) break;
      tokens[step] = next;
      const bool stop = program_stop(p, next, step);
      token = static_cast<uint64_t>(next);
      ++pos;
      if (first_out && step == 0) *first_out = std::chrono::steady_clock::now();
      if (stop) { ++step; break; }
    }
  }
  return step;
}

// B7: state of the adaptive draft-length policy for one generation call.
// `cost_us[k]` is the measured cost of an iteration that verifies k drafts
// (0 = unmeasured), `alpha` the acceptance-rate EMA, and `warm_next` the next
// draft length the warm-up still has to measure (negative when it is done).
// Without a cost cache (autotune disabled) every cost stays unmeasured and the
// policy reduces to the fixed cap, i.e. `on`'s behavior.
struct SpecLengthPolicy {
  static constexpr int kMax = int(kSpecMaxDrafts);
  std::string key;
  std::array<double, kMax + 1> cost_us{};
  double alpha = 0.5;          // alpha0
  int64_t warm_next = -1;      // next k to warm up, < 0 when done
  bool warm_primed = false;    // the first pass at warm_next has run
  // S2 bootstrap: forced k = 1 iterations still owed after the warm-up. The
  // cost model cannot break its own cold start: with alpha at its 0.5 prior,
  // k = 0 wins for any nearly-linear cost curve, no draft is ever proposed, so
  // alpha is never updated. The bootstrap supplies the drafts the EMA needs.
  uint64_t bootstrap_left = 0;
  // S3 early abort: consecutive bootstrap verify steps that accepted nothing.
  // A workload the drafter cannot match will never teach the EMA anything, so
  // five rejects in a row end the bootstrap instead of paying 50 iterations.
  uint64_t consecutive_zero_accept = 0;
};

// SONICBOOM_SPEC_BOOTSTRAP (default 50, 0 disables): number of adaptive
// iterations after the cost warm-up that force k = 1. Parsed once per process,
// like SONICBOOM_SPEC.
//
// S4 tried exploring with real drafter proposals and a lower default (8). It
// was reverted with the rest of the S4 behavior changes: the measured cost of
// one k=1 verify is 2.06 plain steps, so exploration only pays above ~100%
// acceptance, and on the coding trace the experiments cost more than the
// k=7 verifies they occasionally unlocked (see the S4 report section).
static uint64_t spec_bootstrap_steps() {
  static const uint64_t n = [] {
    const char* e = std::getenv("SONICBOOM_SPEC_BOOTSTRAP");
    if (!e || !*e) return uint64_t(50);
    const long v = std::strtol(e, nullptr, 10);
    return v > 0 ? uint64_t(v) : uint64_t(0);
  }();
  return n;
}

// SONICBOOM_SPEC_BOOTSTRAP_ABORT (default 5, 0 disables): consecutive
// bootstrap verify steps that accept no draft before the bootstrap gives up
// and the policy takes over. A workload without drafter matches never updates
// alpha usefully, so the remaining forced iterations are pure verify cost.
// 0 restores the fixed SONICBOOM_SPEC_BOOTSTRAP stepping (the S2 ablation).
static uint64_t spec_bootstrap_abort() {
  static const uint64_t n = [] {
    const char* e = std::getenv("SONICBOOM_SPEC_BOOTSTRAP_ABORT");
    if (!e || !*e) return uint64_t(5);
    const long v = std::strtol(e, nullptr, 10);
    return v > 0 ? uint64_t(v) : uint64_t(0);
  }();
  return n;
}

// The warm-up measures c(k) once per (model shape, context class) on this GPU:
// k = 0..min(7, room) in turn, each iteration timed whole (the second pass at
// each k: the first absorbs that row count's runtime construction and capture).
// The costs are cached in the autotune cache under "spec-length"; later runs
// load them and skip the warm-up. The context class keeps a long-context
// session's KV traffic from being priced with a short-context measurement (the
// cache class only widens the key; the kernel/tile choices stay keyed by shape
// class as before).
static void spec_policy_init(const Gemma4Model& m, uint64_t n_ctx,
                             SpecLengthPolicy& p) {
  if (!tune::enabled()) return;
  const char* cls = n_ctx <= 1024 ? "s" : n_ctx <= 4096 ? "m" : "l";
  // v1: the key version separates costs measured on this loop's code from an
  // older binary's (a cost is only meaningful for the kernels that produced it).
  p.key = std::string("v1|") +
          (m.config.arch == ModelArch::Gemma4 ? "gemma4" : "qwen2") + "|e" +
          std::to_string(m.config.embedding_length) + "|ctx=" + cls;
  const std::map<std::string, int> cached = tune::lookup("spec-length", p.key);
  bool any = false;
  for (int k = 0; k <= SpecLengthPolicy::kMax; ++k) {
    const auto it = cached.find("c" + std::to_string(k));
    if (it != cached.end() && it->second > 0) {
      p.cost_us[size_t(k)] = double(it->second);
      any = true;
    }
  }
  if (!any) p.warm_next = 0;
}

static void spec_policy_store(const SpecLengthPolicy& p) {
  std::map<std::string, int> values;
  for (int k = 0; k <= SpecLengthPolicy::kMax; ++k) {
    if (p.cost_us[size_t(k)] > 0.0) {
      values["c" + std::to_string(k)] =
          std::max(1, int(std::lround(p.cost_us[size_t(k)])));
    }
  }
  if (!values.empty()) tune::store("spec-length", p.key, values);
}

// Execute the program's speculative loop from (token, pos, step). One
// iteration is the program body: draft up to max_drafts tokens from the
// accepted history, then either replay the captured single decode step (the
// drafter proposed nothing) or run the row-invariant verify step over
// [last token, drafts...]; accept keeps the longest draft prefix that matches
// the verifier's per-row argmax plus the verifier's own next token; emit writes
// the accepted run, truncated at the first stop token exactly as the plain
// loop truncates; the position advances by the run's length.
//
// The sequence is bit-identical to the plain loop: by the verify step's row
// invariance, row i's argmax is the token the plain decode step at position
// L+i+1 produces, so the accepted run is the plain continuation and a rejected
// draft only discards work. `history` is the accepted token history known to
// the caller (the prompt, or empty when the prefill was not seen here); it
// seeds the drafter, and the drafter only ever proposes — acceptance is the
// verifier's decision. In `auto` mode the policy above chooses each
// iteration's draft length.
static uint64_t run_spec_program_steps(
    const Gemma4Model& m, DecodeRuntime& rt, const GreedyProgramDesc& p,
    uint64_t token, uint64_t pos, std::span<int64_t> tokens, uint64_t step,
    planner::KvCacheSession& session, std::span<const uint64_t> history,
    std::chrono::steady_clock::time_point* first_out = nullptr,
    const char* strategy_override = nullptr) {
  const uint64_t limit = p.max_tokens > 0
                             ? std::min<uint64_t>(uint64_t(p.max_tokens),
                                                  tokens.size())
                             : tokens.size();
  const size_t cap =
      size_t(std::min<int64_t>(p.max_drafts, kSpecMaxDrafts));
  if (cap == 0 || !session.caches || session.n_ctx == 0) {
    std::cerr << "[spec] the speculative loop needs a drafter cap and a "
                 "session cache\n";
    return step;
  }
  // "ngram" is the program's prompt-lookup drafter; "repeat" is the test seam's
  // always-proposing drafter (the last token, cap times). Either way the
  // verifier decides, so the accepted stream is the plain greedy stream.
  // Own the string: the ternary between a const char* and a std::string
  // materializes a temporary, so a string_view here would dangle as soon as the
  // declaration ends. (The old code only read it in the next two lines, which
  // hid the bug; the S4 diagnostic dump reads it at function end.)
  const std::string strategy =
      strategy_override ? strategy_override : p.draft_strategy;
  const bool repeat = strategy == "repeat";
  if (!repeat && strategy != "ngram") {
    std::cerr << "[spec] unknown draft strategy '" << strategy << "'\n";
    return step;
  }
  // Drafter selection: the prompt-lookup n-gram drafter by default, or the
  // model-backed drafter named by SONICBOOM_DRAFT_MODEL (SD1). Everything below
  // is drafter-agnostic: the verify graph, the accept rule and the emitted
  // stream depend only on the proposals, so a different drafter plugs in here
  // with the verification path reused unchanged. LlmDraftModel still throws
  // "LlmDraftModel not yet implemented" from its constructor, so asking for it
  // fails loudly instead of silently running n-gram.
  std::unique_ptr<planner::DraftModel> drafter =
      planner::make_draft_model(cap);
  drafter->push(history);
  // SONICBOOM_SPEC=auto picks k per iteration (B7). The forced-strategy seam
  // keeps the fixed cap so a test can pin every iteration's shape.
  const bool adaptive = !strategy_override && spec_mode() == SpecMode::Auto;
  SpecLengthPolicy policy;
  if (adaptive) {
    spec_policy_init(m, session.n_ctx, policy);
    policy.bootstrap_left = spec_bootstrap_steps();
  }
  // The valid KV length: everything before `pos` the caller prefilled is
  // resident, and every later step appends exactly the tokens it feeds.
  uint64_t L = pos;
  uint64_t drafts[kSpecMaxDrafts];
  std::vector<uint64_t> feed;
  std::vector<int64_t> rows;
  if (spec_diag_enabled()) {
    spec_diag().total = 0;
    spec_diag().hist.clear();
  }

  while (step < limit) {
    ++g_spec_counters.iterations;
    const auto iter_start = std::chrono::steady_clock::now();
    if (L + 1 > session.n_ctx) {
      std::cerr << "[spec] the session cache is full at position " << L << "\n";
      break;
    }
    // The drafter may propose at most what the cache can verify in one step:
    // k+1 rows at positions L..L+k must fit.
    drafter->push(token);
    const uint64_t room = std::min<uint64_t>(
        uint64_t(cap), session.n_ctx - L - 1);
    size_t k = 0;
    bool warm = false;         // a warm-up iteration (forced k drafts)
    bool measure = false;      // ... whose time is c(k) (its second pass)
    bool finish_warm = false;  // the last measurement: persist after timing
    bool bootstrap_iter = false;  // a forced bootstrap verify (S3 abort input)
    size_t match_n = 0;        // S4: n-gram length behind this proposal
    SpecDiagKind diag_kind = SpecDiagKind::Fixed;
    if (repeat) {
      k = size_t(room);
      for (uint64_t i = 0; i < room; ++i) drafts[i] = token;
    } else if (adaptive) {
      if (policy.warm_next >= 0) {
        if (uint64_t(policy.warm_next) <= room) {
          // Warm-up: force k drafts so the iteration's cost is c(k). Each k is
          // run twice: the first pass absorbs the one-time cost of building and
          // capturing that row-count's verify runtime, the second is timed.
          // Forced drafts are proposals only; the verifier decides the run.
          k = size_t(policy.warm_next);
          warm = true;
          diag_kind = SpecDiagKind::Warm;
          measure = policy.warm_primed;
          if (policy.warm_primed) {
            policy.warm_primed = false;
            if (++policy.warm_next > kSpecMaxDrafts) finish_warm = true;
          } else {
            policy.warm_primed = true;
          }
          for (size_t i = 0; i < k; ++i) drafts[i] = token;
        } else {
          // No room left for the next length: finish with what is measured.
          spec_policy_store(policy);
          policy.warm_next = -1;
        }
      }
      if (!warm) {
        const uint64_t max_k = std::min<uint64_t>(uint64_t(cap), room);
        if (policy.bootstrap_left > 0) {
          // S2: force exploration so the EMA sees real acceptance data. Like
          // the warm-up's forced drafts these are proposals only; the verifier
          // still decides the emitted run. S3: one draft and an abort on a
          // run of rejects keep the exploration cheap on unmatched workloads.
          k = size_t(std::min<uint64_t>(uint64_t(kSpecBootstrapDrafts), max_k));
          for (size_t i = 0; i < k; ++i) drafts[i] = token;
          --policy.bootstrap_left;
          ++g_spec_counters.bootstrap_steps;
          bootstrap_iter = true;
          diag_kind = SpecDiagKind::Bootstrap;
        } else {
          k = drafter->propose(std::span<uint64_t>(
                                  drafts, size_t(spec_choose_drafts(
                                              policy.alpha, policy.cost_us,
                                              int64_t(max_k)))),
                              &match_n);
          ++g_spec_counters.policy_steps;
          diag_kind = SpecDiagKind::Policy;
        }
      }
    } else {
      k = drafter->propose(std::span<uint64_t>(drafts, size_t(room)),
                           &match_n);
    }

    // S4: when the policy chooses k = 0 it never asks the drafter, which leaves
    // "the tables had no match" and "a match existed but was not worth a
    // verify" indistinguishable in the record. Probe anyway: propose reads the
    // tables and does not mutate them, so the shadow lookup changes nothing.
    if (spec_diag_enabled() && k == 0 && !repeat) {
      uint64_t probe[kSpecMaxDrafts];
      const size_t probe_room =
          size_t(std::min<uint64_t>(uint64_t(cap), room));
      (void)drafter->propose(std::span<uint64_t>(probe, probe_room),
                             &match_n);
    }

    if (k == 0) {
      // (k == 0 ? step : verify): the captured single decode step at L.
      if (!rt.prog) {
        std::cerr << "[spec] no captured decode step for the k == 0 path\n";
        break;
      }
      rt.cuda->set_pos(L);
      if (!rt.cuda->seed_token(token) || !rt.prog->replay()) {
        std::cerr << "[spec] the k == 0 decode step failed at position " << L
                  << "\n";
        break;
      }
      float next_f = -1.0f;
      rt.cuda->read_token(&next_f);
      const int64_t next = int64_t(next_f);
      if (next < 0) break;
      // S4: a plain step is a proposal of length 0 (the drafter missed or the
      // policy chose k = 0) that emits the verifier's own token.
      if (spec_diag_enabled()) {
        spec_diag_record(step, L, /*k=*/0, /*accepted=*/0, /*emitted=*/1,
                         match_n, SpecDiagKind::Plain, policy.alpha,
                         iter_start);
        spec_diag_emit(uint64_t(next));
      }
      // The fed token is now resident; the sampled one is emitted but not fed.
      ++L;
      tokens[step] = next;
      if (first_out && step == 1)
        *first_out = std::chrono::steady_clock::now();
      const bool stop = program_stop(p, next, step);
      token = static_cast<uint64_t>(next);
      ++step;
      ++g_spec_counters.plain_steps;
      ++g_spec_counters.emitted_tokens;
      if (measure) {
        policy.cost_us[0] = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - iter_start).count();
        if (finish_warm) {
          policy.warm_next = -1;
          spec_policy_store(policy);
        }
      }
      if (stop) break;
      continue;
    }

    // verify: [last token, drafts...] at L..L+k, one row per position.
    feed.assign(size_t(k) + 1, 0);
    feed[0] = token;
    std::copy_n(drafts, k, feed.begin() + 1);
    rows.assign(size_t(k) + 1, -1);
    if (!verify_step_spine_cuda(m, feed, L, session, rows)) {
      std::cerr << "[spec] the verify step failed at position " << L << "\n";
      break;
    }

    // accept + emit: the run is the accepted drafts plus the verifier's token,
    // truncated at a stop token or the caller's bound (spec_accept_run). The
    // valid KV length advances by the emitted count: the verify appended k+1
    // rows, but only `token` and the emitted run's fed tokens are live (the
    // last emitted token is sampled, not fed). Rows past them are dead entries
    // no later attention extent reaches (an extent at p reads only positions
    // <= p) and the next steps overwrite in order — the same rollback the
    // rejected drafts take.
    uint64_t run[kSpecMaxDrafts + 1];
    const SpecAccept decision =
        spec_accept_run(std::span<const uint64_t>(drafts, k),
                        std::span<const int64_t>(rows.data(), rows.size()),
                        p.eos_tokens, step, limit,
                        std::span<uint64_t>(run, size_t(k) + 1));
    if (spec_diag_enabled()) {
      spec_diag_record(step, L, k, decision.accepted, decision.emitted,
                       match_n, diag_kind, policy.alpha, iter_start);
      for (size_t i = 0; i < decision.emitted; ++i) spec_diag_emit(run[i]);
    }
    ++g_spec_counters.verify_steps;
    g_spec_counters.draft_tokens += k;
    g_spec_counters.accepted_tokens += decision.accepted;
    g_spec_counters.emitted_tokens += decision.emitted;
    // B7 bookkeeping: a warm-up iteration prices c(k); every other one feeds
    // the acceptance-rate EMA (warm-up drafts are forced, so their acceptance
    // says nothing about the drafter).
    if (measure) {
      policy.cost_us[k] = std::chrono::duration<double, std::micro>(
          std::chrono::steady_clock::now() - iter_start).count();
      if (finish_warm) {
        policy.warm_next = -1;
        spec_policy_store(policy);
      }
    } else if (adaptive && !warm) {
      policy.alpha = spec_alpha_update(policy.alpha, decision.accepted, k);
    }
    // S3 early abort: a bootstrap verify that accepts nothing is a drafter
    // miss. A run of them means this workload has nothing to match, so the
    // remaining forced iterations can only add verify cost — switch to the
    // normal policy now.
    if (bootstrap_iter) {
      const uint64_t abort_after = spec_bootstrap_abort();
      if (abort_after > 0) {
        policy.consecutive_zero_accept =
            decision.accepted == 0 ? policy.consecutive_zero_accept + 1 : 0;
        if (policy.consecutive_zero_accept >= abort_after)
          policy.bootstrap_left = 0;
      }
    }
    if (decision.emitted == 0) break;  // the bound was already reached
    for (size_t i = 0; i < decision.emitted; ++i) {
      tokens[step + i] = int64_t(run[i]);
      if (first_out && step + i == 1)
        *first_out = std::chrono::steady_clock::now();
    }
    step += decision.emitted;
    L += decision.emitted;
    for (size_t i = 0; i + 1 < decision.emitted; ++i) drafter->push(run[i]);
    token = run[decision.emitted - 1];
    if (decision.stopped) break;
  }
  // A run that ended before the warm-up finished stores what it measured; the
  // policy then compares only the lengths it has prices for.
  if (adaptive && policy.warm_next >= 0) spec_policy_store(policy);
  spec_diag_dump(strategy.c_str());
  return step;
}

static std::map<std::pair<const Gemma4Model*, uint64_t>,
                std::unique_ptr<DecodeRuntime>> g_decode_runtimes;

// Decode and prefill captures are separate features so an ablation can turn
// off either one; SONICBOOM_CUDA_GRAPHS=0 still disables both.
static bool cuda_graph_decode_enabled() {
  return runtime::feature_enabled(runtime::Feature::CudaGraphDecode);
}

static bool cuda_graph_prefill_enabled() {
  return runtime::feature_enabled(runtime::Feature::CudaGraphPrefill);
}

static bool capture_reuse_enabled() {
  return runtime::feature_enabled(runtime::Feature::CaptureReuse);
}

// The number of greedy decode steps one captured graph runs (§10). Autotuned
// over {1,4,8,16} on the target GPU: 1 is the fastest by 0.2-0.5% (d0/d512,
// 3-run medians), because the decode step is GPU-bound at ~94% busy — the host
// round-trips chunking removes were already hidden — so the measured default
// is 1 (the per-step replay loop). k > 1 is the ablation switch (tools/
// bench.sh --ablate, group "decode-chunk"): one graph launch + one host
// round-trip per k tokens.
static uint32_t decode_chunk_size() {
  // Speculation and chunking are alternative launch-overhead reducers: a
  // speculative step needs the captured graph to mean exactly one decode step
  // (the k == 0 fallback replays it once), so the chunk capture is off.
  if (spec_enabled()) return 1;
  static const uint32_t k = [] {
    const char* e = std::getenv("SONICBOOM_DECODE_CHUNK");
    if (!e || !*e) return uint32_t(1);
    const int requested = std::atoi(e);
    if (requested <= 1) return uint32_t(1);
    return uint32_t(std::min(requested, 16));
  }();
  return k;
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
  // §10: lift the post-pass decode graph into the generation program and read
  // the loop back. Done on a clone so the module that produced the plan is left
  // exactly as it was; the clone is discarded once the desc is extracted. A
  // program that does not construct fails the runtime: the loop's structure is
  // not an optional decoration of the step, and silently running something else
  // would make the default path the hand-written loop again.
  if (rt->graph.nodes.empty())
    return nullptr;
  {
    mlir::OwningOpRef<mlir::ModuleOp> program_module(
        mlir::cast<mlir::ModuleOp>((*ir.module)->clone()));
    // The loop's shape follows SONICBOOM_SPEC: the plain greedy program by
    // default, the draft/verify/accept program when speculation is on. Both
    // lift the same decode graph; only the loop around it differs.
    const mlir::LogicalResult lifted =
        spec_enabled()
            ? compiler::addSpeculativeProgram(*program_module, "speculative",
                                              "prefill", {1}, {}, 0,
                                              kSpecMaxDrafts, "ngram")
            : compiler::addGreedyProgram(*program_module, "greedy", "prefill",
                                         {1}, {}, 0);
    if (mlir::failed(lifted)) {
      std::cerr << "[plan] lowering boom.program failed (diagnostic above)\n";
      return nullptr;
    }
    rt->program = read_greedy_program(*program_module);
    if (!rt->program.valid) {
      std::cerr << "[plan] boom.program did not read back as a greedy loop\n";
      return nullptr;
    }
  }
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
  if (cuda_graph_decode_enabled()) {
    // The k-step capture (§10): step j reads the token step j-1's argmax wrote
    // and its own staged position. The warm-up runs the same k steps so every
    // pooled buffer the capture records is already allocated.
    rt->cuda->enable_decode_chunking(decode_chunk_size());
    const uint32_t k = rt->cuda->decode_chunk();
    rt->cuda->seed_token(0);
    rt->cuda->set_pos(0);
    auto run_steps = [&]() {
      for (uint32_t j = 0; j < k; ++j) {
        if (!rt->cuda->set_step_index(j)) return false;
        std::vector<sx::Bytes> in;
        in.push_back(encode_i64(0));
        in.push_back(encode_i64(0));
        auto res = rt->exec->execute(rt->plan, rt->snap, in);
        if (!res) return false;
      }
      return true;
    };
    if (run_steps()) {
      rt->cuda->synchronize();
      rt->cuda->seed_token(0);
      rt->prog = rt->cuda->record_step([&]() { return run_steps(); });
      if (rt->prog) rt->prog_cache = rt->cuda->cache_identity();
    }
    // No re-playable graph (capture failed, e.g. a plan whose decode step has a
    // host task): the executor loop stages the single pos/token slots itself,
    // so the ring slots must not stay selected.
    if (!rt->prog) rt->cuda->enable_decode_chunking(1);
  }

  DecodeRuntime* raw = rt.get();
  g_decode_runtimes.emplace(key, std::move(rt));
  return raw;
}

// ---------------------------------------------------------------------------
// B3: the row-invariant verify step. One runtime per (model, n_ctx, rows):
// the decode graph emitted with `rows` query positions, run through the verify
// phase's passes (decode fusions, KV append left on attention), planned and
// captured as its own graph. Sharing the decode runtime's cache is the point:
// the verify appends rows at pos..pos+rows-1 exactly as `rows` plain decode
// steps would, so the cache after a verify is the cache a chained decode would
// have written.
// ---------------------------------------------------------------------------
struct VerifyRuntime {
  planner::Graph graph;
  planner::ExecutionPlan plan;
  planner::ResourceSnapshot snap;
  std::unique_ptr<planner::SonicBackend> cpu;
  std::unique_ptr<planner::SonicCudaBackend> cuda;
  std::unique_ptr<planner::RuntimeExecutor> exec;
  std::unique_ptr<hal::StepProgram> prog;  // captured verify step
  // The session cache the runtime (and therefore the capture) is bound to. The
  // backend adopts it after the warm-up (which must not write the session) and
  // before the capture, so the captured kernels carry its device pointers; the
  // shared_ptr keeps those buffers alive for as long as
  // the capture can be replayed, and a different session cannot alias it while
  // this entry exists (same reason as cache_identity()).
  std::shared_ptr<void> cache;
};

static std::map<std::tuple<const Gemma4Model*, uint64_t, int>,
                std::unique_ptr<VerifyRuntime>>
    g_verify_runtimes;

static std::unique_ptr<VerifyRuntime>
build_verify_runtime(const Gemma4Model& m, uint64_t n_ctx, int rows,
                     const planner::KvCacheSession& session) {
  auto rt = std::make_unique<VerifyRuntime>();
  GraphIr ir = emit_decode_ir(m, /*greedy=*/true, uint64_t(rows));
  rt->graph = lower_graph(ir, planner::IrPassPhase::Verify,
                          quant::cuda::q8_1_enabled());
  if (rt->graph.nodes.empty()) return nullptr;

  auto snap = planner::CpuResourceProvider::snapshot();
  if (!snap) return nullptr;
  rt->snap = std::move(*snap);
  {
    planner::AnalyticalCpuCostModel cost(rt->snap);
    auto plan = planner::plan_execution(rt->graph, rt->snap, cost);
    if (!plan) return nullptr;
    rt->plan = std::move(*plan);
  }
  rt->cpu = std::make_unique<planner::SonicBackend>(m, rt->graph);
  rt->cuda = std::make_unique<planner::SonicCudaBackend>(
      kv_cache_spec(m.config), rt->graph, n_ctx);
  rt->cuda->bind_model(m);
  // The capture records the session cache's device pointers, but the warm-up
  // that precedes it *writes* the cache: it runs one verify step at positions
  // 0..rows-1, which would overwrite the prompt's KV rows 0..rows-1 in the
  // session before the real verify replays at start_pos. So the warm-up runs on
  // the backend's own (freshly allocated, identically sized) cache and
  // get_verify_runtime adopts the session cache after it, before the capture.
  rt->cache = session.caches;
  rt->exec = std::make_unique<planner::RuntimeExecutor>(
      std::map<planner::BackendTag, planner::Backend*>{
          {planner::BackendTag::Sonic, rt->cuda.get()}});
  route_host_tasks(*rt->exec, rt->plan, *rt->cpu);
  return rt;
}

// Build (or fetch) the cached verify runtime. The capture reads the token ids
// from `tokens_buf_` and the base position from `pos_buf_`, both restaged before
// every replay, so one capture serves every call at any depth. Keyed by the
// session cache as well as the shape: a capture is only replayable against the
// vector it was recorded with. Returns nullptr on a build/plan failure.
static VerifyRuntime* get_verify_runtime(const Gemma4Model& m, uint64_t n_ctx,
                                         int rows,
                                         const planner::KvCacheSession& session) {
  if (!session.caches) return nullptr;
  auto key = std::make_tuple(&m, n_ctx, rows);
  auto it = g_verify_runtimes.find(key);
  if (it != g_verify_runtimes.end()) {
    if (it->second->cache.get() == session.caches.get())
      return it->second.get();
    // A different session on the same shape: the old capture cannot be replayed
    // against the new cache, and holding both would be two full verify graphs.
    g_verify_runtimes.erase(it);
  }

  auto rt = build_verify_runtime(m, n_ctx, rows, session);
  if (!rt) return nullptr;

  if (cuda_graph_decode_enabled()) {
    // Warm-up run: allocates every pooled buffer and uploads the weights the
    // capture pass then only records. The ids and position are placeholders;
    // every real call stages its own before replay.
    const std::vector<uint64_t> warm(size_t(rows), 0);
    const std::vector<sx::Bytes> in{encode_i64(0), encode_i64(0)};
    auto run = [&]() {
      auto res = rt->exec->execute(rt->plan, rt->snap, in);
      return bool(res);
    };
    if (rt->cuda->set_prefill_tokens(warm) &&
        rt->cuda->set_verify_pos(0, rows) && run()) {
      rt->cuda->synchronize();
      // Adopt the session cache only after the warm-up and before the capture.
      // Capture records the launches without executing them, so the graph bakes
      // the session's pointers while no kernel writes a row of it.
      rt->cuda->share_caches(session);
      if (rt->cuda->set_prefill_tokens(warm) &&
          rt->cuda->set_verify_pos(0, rows))
        rt->prog = rt->cuda->record_step(run);
    }
  }

  VerifyRuntime* raw = rt.get();
  g_verify_runtimes.emplace(key, std::move(rt));
  return raw;
}

// V2: the startup probe behind an unset SONICBOOM_SPEC. One plain decode step
// and one k=1 verify step are timed on this GPU (after untimed warm replays);
// their ratio decides the generation loop, because it is the price of a verify
// row relative to a plain step. A ratio below kSpecProbeAutoRatio means the
// k=1 iteration costs less than that many plain steps, i.e. the acceptance the
// length policy needs to come out ahead is under 50% (E(1,a) = 1 + a against
// c1/c0 = ratio: profitable when a >= ratio - 1).
//
// The result is cached per SM in the Q1 autotune file under
// ("spec_verify_ratio", "decode"), so only the first process on a GPU pays the
// probe. SONICBOOM_SPEC_PROBE_MS="<plain_ms>,<verify_ms>" replaces the measured
// pair (the test seam; measurement only, no kernel changes).
constexpr double kSpecProbeAutoRatio = 1.5;

// The probe measures whole replays: the captured plain step (position staging,
// graph replay, token readback) against the captured 2-row verify step (token
// + position staging, replay, sync). Both include the same host round-trip, so
// the ratio compares the device work, not the launch overhead.
static bool spec_probe_measure(const Gemma4Model& m, DecodeRuntime* rt,
                               uint64_t n_ctx, uint64_t pos, double& plain_ms,
                               double& verify_ms) {
  if (!cuda_graph_decode_enabled() || !rt || !rt->prog) return false;
  if (pos + 2 > n_ctx) return false;  // the k=1 verify appends 2 rows
  planner::KvCacheSession session;
  rt->cuda->publish_cache(session);
  VerifyRuntime* vrt = get_verify_runtime(m, n_ctx, 2, session);
  if (!vrt || !vrt->prog) return false;

  auto ms_since = [](std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
  };
  auto plain = [&](uint64_t p) -> double {
    const auto t0 = std::chrono::steady_clock::now();
    float tok = -1.0f;
    if (!rt->cuda->set_pos(p) || !rt->prog->replay() ||
        !rt->cuda->read_token(&tok))
      return -1.0;
    return ms_since(t0);
  };
  const uint64_t ids[2] = {1, 1};
  auto verify = [&](uint64_t p) -> double {
    const auto t0 = std::chrono::steady_clock::now();
    if (!vrt->cuda->stage_prefill_tokens(
            *vrt->prog, std::span<const uint64_t>(ids, 2)) ||
        !vrt->cuda->stage_verify_pos(*vrt->prog, p, 2) ||
        !vrt->prog->replay() || !vrt->prog->sync())
      return -1.0;
    return ms_since(t0);
  };
  // Warm both replays untimed (the verify capture's first replay included),
  // then time one of each at the same position. The rows the probe writes are
  // rows the caller is about to overwrite: prefill/decode writes every
  // position before any attention extent can read it.
  if (plain(pos) < 0.0 || verify(pos) < 0.0) return false;
  const double p_ms = plain(pos);
  const double v_ms = verify(pos);
  if (!(p_ms > 0.0) || !(v_ms > 0.0)) return false;
  plain_ms = p_ms;
  verify_ms = v_ms;
  return true;
}

static bool spec_probe_injected(double& plain_ms, double& verify_ms) {
  const char* e = std::getenv("SONICBOOM_SPEC_PROBE_MS");
  if (!e || !*e) return false;
  char* end = nullptr;
  const double p = std::strtod(e, &end);
  if (end == e || *end != ',') return false;
  const char* v = end + 1;
  char* end2 = nullptr;
  const double q = std::strtod(v, &end2);
  if (end2 == v || *end2 != '\0' || !(p > 0.0) || !(q > 0.0)) return false;
  plain_ms = p;
  verify_ms = q;
  return true;
}

// Resolve the mode when SONICBOOM_SPEC was unset: load the cached ratio, or
// measure it once and store it. Returns the runtime to decode with — `rt`
// itself, unless auto was selected and this runtime was built (while the mode
// was still undecided) with the greedy program, in which case it is rebuilt.
// The rebuild is safe at both call sites: the probe runs before the runtime's
// KV cache is published into a session, so no capture is bound to it yet.
static DecodeRuntime* spec_resolve_probe(const Gemma4Model& m, DecodeRuntime* rt,
                                         uint64_t n_ctx, uint64_t pos) {
  if (!rt || env_spec_mode() != SpecMode::Undecided ||
      g_spec_mode != SpecMode::Undecided)
    return rt;
  double ratio = 0.0;
  bool measured = false;
  if (const auto cached =
          mlir_tune::lookup_ratio("spec_verify_ratio", "decode")) {
    ratio = *cached;
  } else {
    double plain_ms = 0.0, verify_ms = 0.0;
    if (spec_probe_injected(plain_ms, verify_ms)) {
      ratio = verify_ms / plain_ms;
    } else {
      if (!spec_probe_measure(m, rt, n_ctx, pos, plain_ms, verify_ms))
        return rt;  // no captured step (or no room): stay greedy, retry later
      ratio = verify_ms / plain_ms;
      measured = true;
    }
    (void)mlir_tune::store_ratio("spec_verify_ratio", "decode", ratio);
  }
  const SpecMode mode = ratio < kSpecProbeAutoRatio ? SpecMode::Auto
                                                    : SpecMode::Off;
  g_spec_mode = mode;
  char line[80];
  std::snprintf(line, sizeof(line), "spec probe: ratio=%.2f → %s\n", ratio,
                mode == SpecMode::Auto ? "auto" : "off");
  std::cerr << line;
  if (measured) {
    // The probe's rows=2 verify runtime is only a measurement; drop it so a
    // decision for off does not keep a second set of device weights alive.
    g_verify_runtimes.erase(std::make_tuple(&m, n_ctx, 2));
  }
  if (mode == SpecMode::Auto && !rt->program.speculative) {
    g_decode_runtimes.erase(std::make_pair(&m, n_ctx));
    return get_decode_runtime(m, n_ctx);
  }
  return rt;
}

bool kv_session_allocate(const Gemma4Model& m, planner::KvCacheSession& session,
                         uint64_t n_ctx) {
  if (m.weights.blocks.empty() || n_ctx == 0) return false;
  session.caches =
      planner::SonicCudaBackend::allocate_cache_set(kv_cache_spec(m.config), n_ctx);
  if (!session.caches) return false;
  session.n_ctx = n_ctx;
  session.filled_tokens.clear();
  return true;
}

bool decode_steps_spine_cuda(const Gemma4Model& m,
                             std::span<const uint64_t> tokens,
                             uint64_t start_pos,
                             planner::KvCacheSession& session,
                             std::span<int64_t> out) {
  auto fail = [](const char* why) {
    std::cerr << "[spec] decode_steps: " << why << "\n";
    return false;
  };
  if (tokens.empty() || out.size() < tokens.size())
    return fail("empty input");
  if (session.n_ctx == 0 || !session.caches) return fail("session has no cache");
  if (start_pos + uint64_t(tokens.size()) > session.n_ctx)
    return fail("positions exceed the session cache");

  // Deliberately not the captured runtime: a capture bakes the cache it was
  // recorded against, and this seam is fed a session cache it did not choose.
  // The uncaptured runtime runs the very same decode graph through the very same
  // executor dispatch; only the graph launch is absent, and it has no numerical
  // effect. It is built once per (model, n_ctx) and bound to the session's cache
  // before its first execution.
  struct FedRuntime {
    std::unique_ptr<DecodeRuntime> rt;
    std::shared_ptr<void> cache;
  };
  static std::map<std::pair<const Gemma4Model*, uint64_t>,
                  std::unique_ptr<FedRuntime>>
      fed;
  auto key = std::make_pair(&m, session.n_ctx);
  auto it = fed.find(key);
  FedRuntime* fr = nullptr;
  if (it != fed.end() && it->second->cache.get() == session.caches.get()) {
    fr = it->second.get();
  } else {
    auto rt = build_decode_runtime(m, session.n_ctx);
    if (!rt) return fail("decode runtime build failed");
    rt->cuda->share_caches(session);
    it = fed.insert_or_assign(
        key, std::make_unique<FedRuntime>(FedRuntime{
                 std::move(rt), std::shared_ptr<void>(session.caches)})).first;
    fr = it->second.get();
  }

  for (size_t i = 0; i < tokens.size(); ++i) {
    // Feed the given token rather than the previous step's argmax: reseed the
    // device token slot and stage the row's position, then dispatch the step.
    if (!fr->rt->cuda->seed_token(tokens[i]) ||
        !fr->rt->cuda->set_pos(start_pos + i))
      return fail("device slot staging failed");
    const std::vector<sx::Bytes> in{encode_i64(tokens[i]),
                                    encode_i64(start_pos + i)};
    auto res = fr->rt->exec->execute(fr->rt->plan, fr->rt->snap, in);
    if (!res) {
      std::cerr << "[spec] decode_steps: step " << i
                << " execute failed: " << res.error().message << "\n";
      return false;
    }
    if (res->outputs.empty()) return fail("step produced no output");
    const sx::Bytes& arg = res->outputs[0];
    if (arg.size() < sizeof(float)) return fail("argmax output too small");
    out[i] = int64_t(*reinterpret_cast<const float*>(arg.data()));
  }
  return true;
}

bool verify_step_spine_cuda(const Gemma4Model& m,
                            std::span<const uint64_t> tokens,
                            uint64_t start_pos,
                            planner::KvCacheSession& session,
                            std::span<int64_t> out) {
  auto fail = [](const char* why) {
    std::cerr << "[spec] verify: " << why << "\n";
    return false;
  };
  const int rows = int(tokens.size());
  if (rows <= 0 || rows > planner::kMaxVerifyRows || out.size() < tokens.size())
    return fail("bad row count");
  if (session.n_ctx == 0 || !session.caches) return fail("session has no cache");
  if (start_pos + uint64_t(rows) > session.n_ctx)
    return fail("positions exceed the session cache");
  VerifyRuntime* rt = get_verify_runtime(m, session.n_ctx, rows, session);
  if (!rt) return fail("verify runtime build failed");
  rt->cuda->share_caches(session);  // identity: the capture is bound to it

  if (rt->prog) {
    // Stage the row token ids and the base position on the captured graph's
    // stream, then one replay: the whole verify step is one graph launch.
    if (!rt->cuda->stage_prefill_tokens(*rt->prog, tokens) ||
        !rt->cuda->stage_verify_pos(*rt->prog, start_pos, rows))
      return fail("staging failed");
    // The replay is async on the graph stream; the token readback must see the
    // completed step (the readback copy orders against the graph's stream).
    if (!rt->prog->replay() || !rt->prog->sync())
      return fail("graph replay failed");
  } else {
    if (!rt->cuda->set_prefill_tokens(tokens) ||
        !rt->cuda->set_verify_pos(start_pos, rows))
      return fail("staging failed");
    const std::vector<sx::Bytes> in{encode_i64(0), encode_i64(start_pos)};
    auto res = rt->exec->execute(rt->plan, rt->snap, in);
    if (!res) {
      std::cerr << "[spec] verify: execute failed: "
                << res.error().message << "\n";
      return false;
    }
  }
  float host[planner::kMaxVerifyRows];
  if (!rt->cuda->read_verify_tokens(host, rows))
    return fail("token readback failed");
  for (int i = 0; i < rows; ++i) out[i] = int64_t(host[i]);
  return true;
}

bool spec_step_spine_cuda(const Gemma4Model& m, uint64_t last_token,
                          std::span<const uint64_t> drafts,
                          planner::KvCacheSession& session,
                          std::span<int64_t> out, uint64_t* accepted_out) {
  auto fail = [](const char* why) {
    std::cerr << "[spec] spec_step: " << why << "\n";
    return false;
  };
  if (drafts.empty() || drafts.size() + 1 > size_t(planner::kMaxVerifyRows))
    return fail("bad draft length");
  if (out.size() < drafts.size() + 1) return fail("output too small");
  if (session.n_ctx == 0 || !session.caches) return fail("session has no cache");
  // The valid length: the history is exactly the tokens whose KV is resident,
  // so the token the previous step sampled (last_token) goes at start_pos and
  // the drafts at start_pos+1 .. start_pos+k.
  const uint64_t start_pos = uint64_t(session.filled_tokens.size());
  const size_t rows = drafts.size() + 1;
  if (start_pos + rows > session.n_ctx)
    return fail("positions exceed the session cache");

  // Row i's context ends at drafts[i-1] (row 0 at last_token), so its argmax is
  // the plain-decode greedy token at position start_pos+i+1.
  std::vector<uint64_t> feed(rows);
  feed[0] = last_token;
  for (size_t i = 0; i < drafts.size(); ++i) feed[i + 1] = drafts[i];
  std::vector<int64_t> row_argmax(rows, -1);
  if (!verify_step_spine_cuda(m, feed, start_pos, session, row_argmax))
    return fail("verify step failed");

  uint64_t accepted = 0;
  while (accepted < drafts.size() &&
         row_argmax[size_t(accepted)] == int64_t(drafts[size_t(accepted)]))
    ++accepted;
  for (uint64_t i = 0; i < accepted; ++i) out[i] = int64_t(drafts[size_t(i)]);
  out[accepted] = row_argmax[size_t(accepted)];
  if (accepted_out) *accepted_out = accepted;

  // Rollback: the verify wrote all rows, but only start_pos .. start_pos+accepted
  // belong to the accepted run. The rest are dead entries past the valid length:
  // no later attention extent reaches them (an extent at position p reads only
  // positions <= p), and the next steps overwrite them in order. The session
  // history records the accepted tokens only, so a later prefix match cannot
  // reuse a KV row computed from a rejected draft.
  session.filled_tokens.push_back(last_token);
  for (uint64_t i = 0; i < accepted; ++i)
    session.filled_tokens.push_back(drafts[size_t(i)]);
  return true;
}

bool kv_read_rows_cuda(const planner::KvCacheSession& session, uint32_t layer,
                       bool is_k, uint64_t first, uint64_t count,
                       std::span<float> out) {
  if (!session.caches || layer >= session.caches->size() || count == 0)
    return false;
  const planner::KvCacheDev& c = (*session.caches)[layer];
  const uint64_t row = c.n_heads_kv * c.head_dim;
  if (row == 0 || out.size() < count * row) return false;
  for (uint64_t i = 0; i < count; ++i) {
    if (!planner::SonicCudaBackend::read_cache_row(
            c, is_k, first + i,
            out.subspan(size_t(i * row), size_t(row))))
      return false;
  }
  return true;
}

bool warmup_spine_cuda(const Gemma4Model& m, uint64_t n_ctx) {
  if (m.weights.blocks.empty() || n_ctx == 0) return false;
  // Ablatable warm-up: with the feature off nothing is built here, so the
  // first generate call pays the plan/capture cost inside its own timed
  // region. Reported as success: skipping the warm-up is not a failure.
  if (!runtime::feature_enabled(runtime::Feature::Warmup)) return true;
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
  rt = spec_resolve_probe(m, rt, n_ctx, start_pos);
  if (!rt) return 0;

  const auto t_loop0 = std::chrono::steady_clock::now();
  auto t_first = t_loop0;
  uint64_t token = start_token;
  uint64_t pos = start_pos;
  uint64_t step = 0;

  // Seed the device-resident token-id slot (Phase 3.2 B4). The cached capture was
  // warmed with a placeholder token, so every call re-seeds the real start token.
  rt->cuda->seed_token(start_token);

  const uint32_t chunk = rt->cuda->decode_chunk();
  if (ir_program_enabled() && rt->program.valid && rt->program.speculative &&
      rt->prog) {
    // The speculative program drives the loop. There is no captured prefill
    // here, so the drafter starts from an empty history: its proposals are
    // verified, never trusted, so the token stream is the plain loop's.
    planner::KvCacheSession session;
    rt->cuda->publish_cache(session);
    step = run_spec_program_steps(m, *rt, rt->program, token, pos, tokens, step,
                                  session, {}, &t_first);
  } else if (ir_program_enabled() && rt->program.valid) {
    // The lowered boom.program drives the loop; each step node is the captured
    // StepProgram replay (or the executor dispatch when nothing was captured).
    step = run_program_steps(*rt, rt->program, token, pos, tokens, step,
                             &t_first);
  } else if (rt->prog && chunk > 1) {
    // Replay one captured k-step chunk: stage the carry token and the k
    // positions on the graph's stream, launch once, then read the k tokens in
    // one D2H. The device feeds each step's token to the next, so the host
    // touches the device once per chunk instead of once per step.
    float carry = float(start_token);
    std::vector<uint64_t> positions(chunk);
    std::vector<float> out(chunk);
    while (step < tokens.size()) {
      for (uint32_t j = 0; j < chunk; ++j) positions[j] = pos + j;
      const std::string name = "decode_chunk_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(name.c_str());
      if (!rt->cuda->stage_chunk(*rt->prog, carry, positions) ||
          !rt->prog->replay() ||
          !rt->cuda->read_chunk_tokens(out.data(), chunk)) {
        if (nvtx_enabled()) nvtxRangePop();
        std::cerr << "[spine] chunk at step " << step << " failed\n";
        break;
      }
      if (nvtx_enabled()) nvtxRangePop();
      bool stopped = false;
      for (uint32_t j = 0; j < chunk && step < tokens.size(); ++j) {
        const int64_t next = int64_t(out[j]);
        if (next < 0) { stopped = true; break; }
        tokens[step] = next;
        carry = out[j];
        ++pos;
        ++step;
        if (step == 1) t_first = std::chrono::steady_clock::now();
      }
      if (stopped) break;
    }
  } else if (rt->prog) {
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

uint64_t spec_generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                                  uint64_t start_pos, std::span<int64_t> tokens,
                                  uint64_t n_ctx, const char* draft_strategy) {
  if (m.weights.blocks.empty() || start_token >= m.config.vocab_size) return 0;
  if (n_ctx == 0) n_ctx = start_pos + tokens.size();
  DecodeRuntime* rt = get_decode_runtime(m, n_ctx);
  if (!rt || !rt->prog || !rt->program.valid || !rt->program.speculative)
    return 0;
  rt->cuda->seed_token(start_token);
  planner::KvCacheSession session;
  rt->cuda->publish_cache(session);
  return run_spec_program_steps(m, *rt, rt->program, start_token, start_pos,
                                tokens, 0, session, {}, nullptr,
                                draft_strategy);
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

uint64_t spec_generate_spine_cuda(const Gemma4Model&, uint64_t, uint64_t,
                                  std::span<int64_t>, uint64_t, const char*) {
  return 0;  // built without CUDA: no speculative loop to run
}

bool warmup_spine_cuda(const Gemma4Model&, uint64_t) {
  return false;  // built without CUDA: there is no runtime to build
}

bool kv_session_allocate(const Gemma4Model&, planner::KvCacheSession&,
                         uint64_t) {
  return false;  // built without CUDA: no device cache to allocate
}

bool decode_steps_spine_cuda(const Gemma4Model&, std::span<const uint64_t>,
                             uint64_t, planner::KvCacheSession&,
                             std::span<int64_t>) {
  return false;  // built without CUDA: no device spine to run
}

bool verify_step_spine_cuda(const Gemma4Model&, std::span<const uint64_t>,
                            uint64_t, planner::KvCacheSession&,
                            std::span<int64_t>) {
  return false;  // built without CUDA: no device spine to run
}

bool spec_step_spine_cuda(const Gemma4Model&, uint64_t,
                          std::span<const uint64_t>, planner::KvCacheSession&,
                          std::span<int64_t>, uint64_t*) {
  return false;  // built without CUDA: no device spine to run
}

bool kv_read_rows_cuda(const planner::KvCacheSession&, uint32_t, bool,
                       uint64_t, uint64_t, std::span<float>) {
  return false;  // built without CUDA: there is no device cache to read
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

// A non-greedy CUDA decode runtime: the full [vocab] logits reach the host each
// step, where the greedy graph reduces them to an argmax on the device. Built
// per call — a throwaway runtime, not the cached greedy one. `raw_logits` keeps
// the model's logit softcap off the output path: the logit dump wants the
// softcapped distribution, the perplexity scorer wants the raw one.
static std::unique_ptr<DecodeRuntime> build_eval_runtime(const Gemma4Model& m,
                                                         uint64_t n_ctx,
                                                         bool raw_logits) {
  auto rt = std::make_unique<DecodeRuntime>();
  GraphIr ir = emit_decode_ir(m, /*greedy=*/false, /*rows=*/1,
                              /*softcap_output=*/!raw_logits);
  rt->graph = lower_graph(ir, planner::IrPassPhase::Decode,
                          quant::cuda::q8_1_enabled());
  auto s = planner::CpuResourceProvider::snapshot();
  if (!s) return nullptr;
  rt->snap = std::move(*s);
  planner::AnalyticalCpuCostModel cost(rt->snap);
  auto p = planner::plan_execution(rt->graph, rt->snap, cost);
  if (!p) return nullptr;
  rt->plan = std::move(*p);
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
                                         planner::KvCacheSession* session,
                                         std::span<const uint64_t> history) {
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
    // V2: with SONICBOOM_SPEC unset, settle the loop shape (probe or cached
    // ratio) before the prefill/loop below reads rt->program.
    Phase _pp("prefill:spec_probe", phase_timing);
    rt = spec_resolve_probe(m, rt, n_ctx, start_pos);
    if (!rt) return 0;
  } else {
    Phase _p("prefill:build_decode_graph", phase_timing);
    dump_rt = build_eval_runtime(m, n_ctx, /*raw_logits=*/false);
    if (!dump_rt) return 0;
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

    // One symbolic plan per shape class (the full chunk; the final segment is
    // usually shorter but stays in the chunk's class and binds its own length
    // at execute time). Each is bound to its own graph but shares the decode
    // backend's KV cache, so prior segments' context is visible. The full-chunk
    // segment is captured into a CUDA Graph (prog) and replayed per non-final
    // segment; the final (possibly shorter) segment runs uncaptured so its
    // hidden column reaches the host through the executor's output path.
    //
    // The segment build (graph + rewrites + plan + backends) is a pure function
    // of (model, n_ctx, shape class, planner environment) and costs ~115 ms of
    // planning per 512-token Gemma segment, so it is cached across calls in the
    // decode runtime (see the DecodeRuntime note). The offset a segment is executed at
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
    // Shape classes. A plan is a function of the kernel-selection decisions
    // that depend on the segment length, not of the length itself. The only
    // such decision in the prefill passes is whether the mmq folds run at all
    // (the segment must support the q8_1 activation footprint, i.e.
    // length >= mmq_min_n), so every segment at or above that threshold shares
    // one plan sized at the full chunk and every shorter segment (a one-token
    // final segment) shares the plan sized at min_n - 1. The plan is lowered
    // from symbolic IR and executed with the actual segment length bound at
    // execute time; the cache is keyed by the class's sizing extent.
    const bool symbolic_plans =
        runtime::feature_enabled(runtime::Feature::SymbolicPrefill);
    const uint64_t mmq_min_n = std::max<uint64_t>(
        uint64_t(quant::cuda::mmq_min_n()), 1);
    auto plan_extent_for = [&](uint64_t nseg) -> uint64_t {
      if (!symbolic_plans) return nseg;  // ablation: concrete plan per length
      return nseg >= mmq_min_n ? chunk : mmq_min_n - 1;
    };
    // Per-call owners: segments when the cache is off. A captured program is
    // owned by its segment (which the replay loop reads through `progs`), so
    // it outlives the call with the segment that recorded it.
    std::map<uint64_t, std::unique_ptr<PrefillSegment>> call_segments;
    std::map<uint64_t, hal::StepProgram*> progs;
    auto segment_for = [&](uint64_t nseg) -> PrefillSegment* {
      const uint64_t extent = plan_extent_for(nseg);
      if (cache_segments) {
        auto it = std::find_if(
            rt->prefills.begin(), rt->prefills.end(),
            [extent](const auto& e) { return e.first == extent; });
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
        GraphIr seg_ir = emit_prefill_ir(m, extent,
                                         /*dynamic_extent=*/symbolic_plans);
        // C.4: fold the mmq activation quantize into the norm/gelu kernels when
        // the mmq tensor-core path is active for this segment (same conditions
        // the batched matmul uses, so fused and unfused positions never disagree
        // about which kernel a position runs). A class's extent decides it on
        // the class's behalf: all members share the decision.
        seg->graph = lower_graph(
            seg_ir, planner::IrPassPhase::Prefill,
            quant::cuda::q8_1_enabled() && quant::cuda::mmq_enabled() &&
                extent >= mmq_min_n,
            /*dynamic_extent=*/symbolic_plans ? extent : 0);
      }
      if (seg->graph.nodes.empty()) {
        std::cerr << "[prefill-cuda] n=" << nseg << " graph lowering failed\n";
        return nullptr;
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
        rt->prefills.emplace_front(extent, std::move(seg));
        while (rt->prefills.size() > kMaxCachedSegments) rt->prefills.pop_back();
      } else {
        // Uncached: keep it alive for the rest of the call (a later segment of
        // the same class must reuse it, not rebuild it).
        call_segments.emplace(extent, std::move(seg));
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
    const bool use_graphs = cuda_graph_prefill_enabled();
    if (use_graphs && n > chunk && start_pos == 0) {
      Phase _cap("prefill:capture_graph", phase_timing);
      PrefillSegment* full = segment_for(chunk);
      if (full) {
        const std::shared_ptr<void> cache_id = full->cuda->cache_identity();
        if (capture_reuse_enabled() && full->chunk_prog &&
            full->chunk_prog_cache.get() == cache_id.get()) {
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
            res = full->exec->execute(full->plan, snap, in, /*segment_extent=*/chunk);
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
                auto cres = full->exec->execute(full->plan, snap, cin,
                                                /*segment_extent=*/chunk);
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
      // Align the segment boundary to an absolute multiple of `chunk`. The SWA
      // ring is sized `chunk + kMaxVerifyRows` (the verify headroom), so a
      // segment is a run of at most `chunk` consecutive positions, never a full
      // ring. The alignment also makes a prefix-reuse suffix (start_pos not a
      // multiple of `chunk`) split at the same boundaries a full prefill would,
      // so the two paths write the same per-position KV.
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
      auto res = seg->exec->execute(seg->plan, snap, in,
                                    /*segment_extent=*/nseg);
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
  // needs the full softcapped vector rather than the argmax scalar), replay it:
  // k steps per launch in chunk mode (carry + k positions in, k tokens out),
  // else one step per launch staging the position and reading the winning token
  // back from the persistent device slot (the synchronous D2H also drains the
  // graph stream). Otherwise fall back to per-step executor dispatch.
  const uint32_t chunk = rt->cuda->decode_chunk();
  if (!dump_logits && ir_program_enabled() && rt->program.valid &&
      rt->program.speculative && rt->prog) {
    // The speculative program drives the loop over the same cache the prefill
    // filled: the session's when the caller passed one, otherwise a session
    // published from this runtime's own cache (the one the step capture was
    // recorded against).
    planner::KvCacheSession local;
    planner::KvCacheSession* kv = session;
    if (!kv) {
      rt->cuda->publish_cache(local);
      kv = &local;
    }
    // The drafter must see the session's accepted token history, not the
    // prefill suffix: on a prefix-reuse call the suffix can be a single token,
    // which leaves the n-gram tables with nothing to match. `history` is the
    // caller's full accepted sequence (see the prefix wrapper).
    step = run_spec_program_steps(m, *rt, rt->program, token, pos, tokens, step,
                                  *kv, history.empty() ? prompt : history);
  } else if (!dump_logits && ir_program_enabled() && rt->program.valid) {
    step = run_program_steps(*rt, rt->program, token, pos, tokens, step);
  } else if (!dump_logits && rt->prog && chunk > 1) {
    float carry = float(token);
    std::vector<uint64_t> positions(chunk);
    std::vector<float> out(chunk);
    while (step < tokens.size()) {
      for (uint32_t j = 0; j < chunk; ++j) positions[j] = pos + j;
      const std::string name = "decode_chunk_" + std::to_string(step);
      if (nvtx_enabled()) nvtxRangePushA(name.c_str());
      const bool ok = rt->cuda->stage_chunk(*rt->prog, carry, positions) &&
                      rt->prog->replay() &&
                      rt->cuda->read_chunk_tokens(out.data(), chunk);
      if (nvtx_enabled()) nvtxRangePop();
      if (!ok) {
        std::cerr << "[prefill-spine-cuda] chunk at step " << step
                  << " failed\n";
        break;
      }
      bool stopped = false;
      for (uint32_t j = 0; j < chunk && step < tokens.size(); ++j) {
        const int64_t next = int64_t(out[j]);
        if (next < 0) { stopped = true; break; }
        tokens[step] = next;
        carry = out[j];
        ++pos;
        ++step;
      }
      if (stopped) break;
    }
  } else if (!dump_logits && rt->prog) {
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
                                  use_prefill, /*session=*/nullptr, prompt);
}

bool score_spine_cuda(const Gemma4Model& m, std::span<const uint64_t> tokens,
                      uint64_t n_ctx, double& sum_nll, uint64_t& n_scored) {
  sum_nll = 0;
  n_scored = 0;
  if (m.weights.blocks.empty() || tokens.size() < 2) {
    std::cerr << "[score] empty model (" << m.weights.blocks.size()
              << " blocks) or short input (" << tokens.size() << ")\n";
    return false;
  }
  const uint64_t vocab = m.config.vocab_size;
  for (uint64_t t : tokens)
    if (t >= vocab) {
      std::cerr << "[score] token " << t << " outside vocab " << vocab << "\n";
      return false;
    }
  if (n_ctx < tokens.size()) n_ctx = tokens.size();
  // Teacher forcing on the non-greedy decode graph: token p is seeded and fed
  // at position p, and the graph's full logits predict token p+1. This is the
  // same per-token graph the greedy spine runs, so the score reflects exactly
  // the profile's kernels (KV storage, reductions, dense precision).
  std::unique_ptr<DecodeRuntime> rt =
      build_eval_runtime(m, n_ctx, /*raw_logits=*/true);
  if (!rt) {
    std::cerr << "[score] eval runtime build failed\n";
    return false;
  }
  std::vector<float> lf(vocab);
  double sum = 0;
  uint64_t scored = 0;
  for (uint64_t p = 0; p + 1 < tokens.size(); ++p) {
    rt->cuda->seed_token(tokens[p]);
    rt->cuda->set_pos(p);
    std::vector<sx::Bytes> in;
    in.push_back(encode_i64(tokens[p]));
    in.push_back(encode_i64(p));
    auto res = rt->exec->execute(rt->plan, rt->snap, in);
    if (!res) {
      std::cerr << "[score] step " << p
                << " execute failed: " << res.error().message << "\n";
      return false;
    }
    if (res->outputs.empty()) {
      std::cerr << "[score] step " << p << " no outputs\n";
      return false;
    }
    const sx::Bytes& lo = res->outputs[0];
    if (lo.size() < vocab * sizeof(float)) {
      std::cerr << "[score] step " << p << " logits too small: " << lo.size()
                << " < " << vocab * sizeof(float) << "\n";
      return false;
    }
    std::memcpy(lf.data(), lo.data(), vocab * sizeof(float));
    float mx = -INFINITY;
    for (uint64_t v = 0; v < vocab; ++v) mx = std::max(mx, lf[v]);
    double se = 0.0;
    for (uint64_t v = 0; v < vocab; ++v)
      se += std::exp(double(lf[v]) - double(mx));
    sum += (double(mx) + std::log(se)) - double(lf[tokens[p + 1]]);
    ++scored;
  }
  sum_nll = sum;
  n_scored = scored;
  return scored > 0;
}

namespace {

// Whether the reuse boundary is held back onto a prefill chunk boundary when a
// seeded prefix would otherwise end with a one-token final segment.
// SONICBOOM_PREFIX_ALIGN=0 restores the old boundary (divergence analysis).
bool prefix_align_enabled() {
  static const bool v = [] {
    const char* e = std::getenv("SONICBOOM_PREFIX_ALIGN");
    return !(e && std::string_view(e) == "0");
  }();
  return v;
}

} // namespace

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

  // Prefill lays its segments on absolute multiples of the chunk and a segment
  // shorter than the mmq threshold (i.e. a single token) takes the
  // single-column kernels instead of the batched ones. A full prefill of a
  // longer prompt computes the same position inside a multi-token segment, and
  // the two disagree in the KV cache (measured: the layer-0 V row differs by
  // ~3e-2 at a one-token position, the K row is bit-identical). Two boundaries
  // can put such a segment next to the reuse point:
  //   - the seed's final segment: prefix_len % chunk is nonzero but shorter
  //     than the mmq threshold. Hold those tokens back so the seed ends on a
  //     chunk boundary (the full prefill's segmentation).
  //   - the suffix's first segment: the suffix starts less than threshold-many
  //     tokens before a chunk boundary, so the boundary truncation would give
  //     it a one-token first segment while the full prefill computes those
  //     positions in a multi-token one. Hold back just enough tokens to lift
  //     the first segment to the threshold.
  // Both costs are at most threshold-1 recomputed tokens.
  const uint64_t chunk = (m.config.sliding_window > 0)
                             ? m.config.sliding_window
                             : kDefaultPrefillChunk;
  const uint64_t mmq_min_n = std::max<uint64_t>(
      uint64_t(quant::cuda::mmq_min_n()), 1);
  if (prefix_align_enabled() && prefix_len > 0) {
    const uint64_t rem = prefix_len % chunk;
    if (rem != 0 && rem < mmq_min_n) {
      prefix_len -= rem;
    } else if (rem != 0 && chunk - rem < mmq_min_n) {
      prefix_len -= mmq_min_n - (chunk - rem);
    }
  }
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

  // `prompt` is the full accepted sequence this request extends; `suffix` is
  // only what the cache still needs prefilled. The spec drafter is seeded with
  // the former so its n-gram tables cover the whole session, not just the
  // reuse tail.
  const uint64_t done = prefill_decode_cuda_impl(m, suffix, prefill_start,
                                                 tokens, n_ctx, use_prefill,
                                                 &session, prompt);
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
bool score_spine_cuda(const Gemma4Model&, std::span<const uint64_t>, uint64_t,
                      double& sum_nll, uint64_t& n_scored) {
  sum_nll = 0;
  n_scored = 0;
  return false;  // built without CUDA: no device scoring
}
#endif

} // namespace sonicboom::model
