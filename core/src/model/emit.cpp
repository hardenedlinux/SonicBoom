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

// The recipe emitter (emit.h).
//
// A phase decides how one position's activation is laid out and what each
// node carries for the kernels of that phase; a layer kind decides which nodes
// it is made of. The kinds are written once, against the phase's primitives
// (`Phase` below), and the two phases — a decode step and a prefill segment —
// supply the primitives. What differs structurally between the phases is the
// small set of virtual hooks: how a mixer reaches its cache, where the
// per-layer input is computed, and what opens and closes the graph.
//
// The graphs are the ones the two hand-written emitters produced: the same
// nodes in the same order with the same names and attributes. Node order and
// attribute order are part of that — tensor and node ids follow emission
// order — so the order of the calls below is deliberate.

#include "emit.h"

#include <sonicboom/compiler/BoomBuilder.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/numerics.h>
#include <sonicboom/planner/op_kind.h>
#include <sonicboom/sx/ir.h>

#include "mlir/IR/BuiltinTypes.h"

#ifdef SONICBOOM_USE_CUDA
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include "../nn/cuda_resident.h"     // attn_decode_plan_chunk (split-KV schedule)
#include "../quant/cuda_resident.h"  // decode_family_mmq (per-shape family)
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace sonicboom::model {

bool decode_mmq_available() {
#ifdef SONICBOOM_USE_CUDA
  return quant::cuda::q8_1_enabled() && quant::cuda::mmq_enabled();
#else
  return false;  // CPU-only build: no device matmul path exists.
#endif
}

namespace {

using planner::OpKind;
using Attrs = std::vector<sx::Attribute>;

sx::Attribute attr_i(const char* name, int64_t v) { return sx::Attribute{name, v}; }
sx::Attribute attr_f(const char* name, double v) { return sx::Attribute{name, v}; }
sx::Attribute attr_s(const char* name, std::string v) {
  return sx::Attribute{name, std::move(v)};
}
sx::Attribute slot_a(WeightSlot s) { return attr_i("slot", int64_t(s)); }
sx::Attribute layer_a(uint32_t l) { return attr_i("layer", int64_t(l)); }

// The activation scale layout the mmq quantize takes for a weight type: 1 for
// the types with a fused mmq quantize of that layout, -1 for a type with none
// (its matmul quantizes the activation inside the mmq call), 0 otherwise.
int64_t mmq_scale_layout(quant::QuantType qt) {
  return (qt == quant::QuantType::Q4_K || qt == quant::QuantType::Q5_K) ? 1
         : (qt == quant::QuantType::Q6_K)                               ? -1
                                                                        : 0;
}

// The load-time family decision for one projection. A CPU-only build has no
// family measurement and no device matmul, so it never selects mmq.
bool decode_family_mmq(quant::QuantType qt, uint64_t in_w, uint64_t out_w) {
#ifdef SONICBOOM_USE_CUDA
  return quant::cuda::decode_family_mmq(qt, int(out_w), int(in_w));
#else
  (void)qt;
  (void)in_w;
  (void)out_w;
  return false;
#endif
}

// ---------------------------------------------------------------------------
// A phase: the primitives the layer kinds are written against, and the kinds.
// ---------------------------------------------------------------------------
class Phase {
public:
  Phase(compiler::BoomBuilder& builder, const Recipe& recipe)
      : b(builder), r(recipe) {}
  virtual ~Phase() = default;

  // The whole stack: the embedding, then every block. Returns the residual
  // stream after the last block.
  mlir::Value stack() {
    mlir::Value residual = embed();
    for (uint32_t l = 0; l < r.blocks.size(); ++l)
      residual = block(r.blocks[l], l, residual);
    return residual;
  }

protected:
  compiler::BoomBuilder& b;
  const Recipe& r;
  std::vector<mlir::Value> kv_;     // the cache handles, by cache index
  std::vector<mlir::Value> state_;  // the recurrent state handles, by index
  mlir::Value inp_{};               // the (scaled) token embedding

  // ---- what a phase supplies -----------------------------------------------

  // The shape of an activation that holds `width` values per position.
  virtual std::vector<int64_t> shape(uint64_t width) const = 0;
  // The position input rotary embeddings and the cache are addressed by.
  virtual mlir::Value position() const = 0;
  // What a node carries to say how many positions it covers.
  virtual void rows(Attrs& a) const = 0;
  // What a full-width norm carries beyond `rows`.
  virtual void norm_extent(Attrs& a, uint64_t width) const = 0;
  // What a weight matmul carries to select its kernel.
  virtual void lowering(Attrs& a, const Projection& p) const = 0;
  // The token embedding lookup input.
  virtual mlir::Value tokens() const = 0;

  // Reads (and, for a layer that owns its cache, first extends) the cache.
  // `k` and `v` are null for a layer that does not own it.
  virtual mlir::Value attend(const SelfAttention& at, uint32_t layer,
                             mlir::Value q, mlir::Value k, mlir::Value v) = 0;
  // The recurrent scan of a gated DeltaNet block: the phase supplies the
  // position its rows start at and the row marking.
  virtual mlir::Value recur(const GatedDeltaNet& gdn, uint32_t layer,
                            mlir::Value qkv, mlir::Value bl, mlir::Value al) = 0;
  // Whatever the phase computes from the token alone ahead of the blocks.
  virtual void per_layer_preamble() = 0;
  // Block `layer`'s slice of the per-layer input.
  virtual mlir::Value per_layer_slice(uint32_t layer) = 0;

  // ---- primitives ----------------------------------------------------------

  mlir::Value node(OpKind kind, std::vector<mlir::Value> inputs,
                   const std::string& name, uint64_t width, Attrs attrs) {
    return b.node(kind, std::move(inputs), name, shape(width), std::move(attrs));
  }

  // RMSNorm over the full width, with learned weights.
  mlir::Value norm(mlir::Value x, const std::string& name, uint64_t width,
                   uint32_t layer, WeightSlot weight) {
    Attrs a = {layer_a(layer), slot_a(weight)};
    norm_extent(a, width);
    return node(OpKind::RmsNorm, {x}, name, width, std::move(a));
  }

  // RMSNorm per head; unit weight when `weight` is None. `segment` > 0 splits
  // the width into segments of that size and writes it on the node; 0 is the
  // layer's head dim and writes nothing, so the graph of a model that uses it
  // is unchanged.
  mlir::Value norm_heads(mlir::Value x, const std::string& name, uint64_t width,
                         uint32_t layer, WeightSlot weight,
                         int64_t segment = 0) {
    Attrs a = {layer_a(layer)};
    if (weight != WeightSlot::None)
      a.push_back(slot_a(weight));
    rows(a);
    if (segment != 0)
      a.push_back(attr_i("dim", segment));
    return node(OpKind::RmsNormHeads, {x}, name, width, std::move(a));
  }

  // The projection's matmul, without its bias.
  mlir::Value matmul(mlir::Value x, const std::string& name,
                     const Projection& p, uint32_t layer) {
    Attrs a = {layer_a(layer), slot_a(p.weight)};
    lowering(a, p);
    return node(OpKind::QuantizedMatmul, {x}, name, p.out, std::move(a));
  }

  // The projection's bias; `x` itself when it has none.
  mlir::Value bias(mlir::Value x, const std::string& name, const Projection& p,
                   uint32_t layer) {
    if (p.bias == WeightSlot::None)
      return x;
    Attrs a = {layer_a(layer), slot_a(p.bias)};
    rows(a);
    return node(OpKind::AddBias, {x}, name, p.out, std::move(a));
  }

  // `rot_dims` > 0 rotates only that many leading values of each head, and is
  // written on the node; 0 is the whole head and writes nothing, so the graph
  // of a model with full rope is unchanged.
  mlir::Value rope(mlir::Value x, const std::string& name, uint64_t width,
                   uint32_t layer, uint32_t rot_dims) {
    Attrs a = {layer_a(layer)};
    rows(a);
    if (rot_dims != 0) a.push_back(attr_i("rot_dims", int64_t(rot_dims)));
    return node(OpKind::Rope, {x, position()}, name, width, std::move(a));
  }

  // An elementwise op that carries only its layer.
  mlir::Value pointwise(OpKind kind, std::vector<mlir::Value> inputs,
                        const std::string& name, uint64_t width,
                        uint32_t layer) {
    return node(kind, std::move(inputs), name, width, {layer_a(layer)});
  }

  mlir::Value scale(mlir::Value x, const std::string& name, uint64_t width,
                    double by) {
    return node(OpKind::Scale, {x}, name, width, {attr_f("scale", by)});
  }

  // ---- the embedding -------------------------------------------------------

  mlir::Value embed() {
    Attrs a = {slot_a(WeightSlot::TokenEmbd)};
    rows(a);
    const mlir::Value embd =
        node(OpKind::Embedding, {tokens()}, "embd", r.width, std::move(a));
    inp_ = r.scale_embedding
               ? scale(embd, "inp", r.width, std::sqrt(float(r.width)))
               : embd;
    if (r.per_layer_input)
      per_layer_preamble();
    return inp_;
  }

  // ---- mixers --------------------------------------------------------------

  // Q is projected for every layer. A layer that owns its cache also projects
  // K and V, in the order: both matmuls, both biases, the K norm, the V norm,
  // V to fp16, K rotated, K to fp16. The bias precedes the rotation.
  mlir::Value mix(const SelfAttention& at, uint32_t l, mlir::Value h) {
    const std::string s = std::to_string(l);
    const uint32_t qw = at.q_width();
    const uint32_t kvw = at.kv_width();

    mlir::Value q = matmul(h, "q" + s, at.q, l);
    q = bias(q, "qb" + s, at.q, l);
    // Gated attention: the Q projection is [head_dim query | head_dim gate] per
    // head, so the gate slice takes the second half of each head and the query
    // keeps the first. Both are strided slices of the same projection.
    mlir::Value qgate;
    if (at.output_gate) {
      const int64_t hd = int64_t(at.head_dim);
      const int64_t stride = 2 * hd;
      const mlir::Value qfull = q;
      q = node(OpKind::Slice, {qfull}, "qq" + s, qw,
               {attr_i("offset", int64_t(0)), attr_i("length", hd),
                attr_i("stride", stride)});
      qgate = node(OpKind::Slice, {qfull}, "qg" + s, qw,
                   {attr_i("offset", hd), attr_i("length", hd),
                    attr_i("stride", stride)});
    }
    if (at.qk_norm)
      q = norm_heads(q, "qn" + s, qw, l, WeightSlot::AttnQNorm);
    q = rope(q, "qr" + s, qw, l, at.rotary_dims);

    mlir::Value attn;
    if (at.owns_cache) {
      mlir::Value k = matmul(h, "k" + s, at.k, l);
      mlir::Value v = matmul(h, "v" + s, at.v, l);
      k = bias(k, "kb" + s, at.k, l);
      v = bias(v, "vb" + s, at.v, l);
      if (at.qk_norm)
        k = norm_heads(k, "kn" + s, kvw, l, WeightSlot::AttnKNorm);
      if (at.v_norm)
        v = norm_heads(v, "vn" + s, kvw, l, WeightSlot::None);
      const mlir::Value vc = pointwise(OpKind::CastFp16, {v}, "vc" + s, kvw, l);
      k = rope(k, "kr" + s, kvw, l, at.rotary_dims);
      const mlir::Value kc = pointwise(OpKind::CastFp16, {k}, "kc" + s, kvw, l);
      attn = attend(at, l, q, kc, vc);
    } else {
      attn = attend(at, l, q, {}, {});
    }

    // The gate multiplies the attention result before the output projection.
    if (at.output_gate) {
      const mlir::Value g =
          pointwise(OpKind::Sigmoid, {qgate}, "gs" + s, qw, l);
      attn = pointwise(OpKind::Mul, {attn, g}, "gm" + s, qw, l);
    }

    mlir::Value out = matmul(attn, "ap" + s, at.out, l);
    return bias(out, "apb" + s, at.out, l);
  }

  // A projection of the recurrent mixer: the quantized matmul, or, for a
  // weight the file stores f32, the f32 matvec (the node the shared expert's
  // gate uses), which is what the backend dispatches on the projection's type.
  mlir::Value mixer_projection(mlir::Value x, const std::string& name,
                               const Projection& p, uint32_t layer) {
    if (!p.f32)
      return matmul(x, name, p, layer);
    Attrs a = {layer_a(layer), slot_a(p.weight)};
    rows(a);
    return node(OpKind::MatvecF32, {x}, name, p.out, std::move(a));
  }

  // The gated DeltaNet mixer: the four projections the scan consumes, the scan
  // over the block's state, the per-key-head norm of its output, the gate, and
  // the output projection. Node names and order are fixed.
  mlir::Value mix(const GatedDeltaNet& gdn, uint32_t l, mlir::Value h) {
    const std::string s = std::to_string(l);
    const uint64_t vd = gdn.value_dim();
    const mlir::Value dq = mixer_projection(h, "dq" + s, gdn.qkv, l);
    const mlir::Value dz = mixer_projection(h, "dz" + s, gdn.gate, l);
    const mlir::Value db = mixer_projection(h, "db" + s, gdn.beta, l);
    const mlir::Value da = mixer_projection(h, "da" + s, gdn.alpha, l);
    const mlir::Value ds = recur(gdn, l, dq, db, da);
    const mlir::Value dno = norm_heads(ds, "dno" + s, vd, l, WeightSlot::GdnNorm,
                                       int64_t(gdn.head_dim));
    const mlir::Value dzs = pointwise(OpKind::Silu, {dz}, "dzs" + s, vd, l);
    const mlir::Value dm = pointwise(OpKind::Mul, {dno, dzs}, "dm" + s, vd, l);
    return mixer_projection(dm, "dp" + s, gdn.out, l);
  }

  // The scan's attributes: the block's layer, its kernel weights, the
  // recurrence's geometry, its state, and the phase's row marking.
  Attrs scan_attrs(const GatedDeltaNet& gdn, uint32_t l) {
    Attrs a = {layer_a(l),
               slot_a(WeightSlot::GdnConv),
               attr_i("slot_dt", int64_t(WeightSlot::GdnDt)),
               attr_i("slot_a", int64_t(WeightSlot::GdnA)),
               attr_i("heads_k", int64_t(gdn.heads_k)),
               attr_i("heads_v", int64_t(gdn.heads_v)),
               attr_i("dim", int64_t(gdn.head_dim)),
               attr_i("kernel", int64_t(gdn.conv_kernel)),
               attr_i("state", int64_t(gdn.state))};
    rows(a);
    return a;
  }

  mlir::Value scan(const GatedDeltaNet& gdn, uint32_t l, mlir::Value qkv,
                   mlir::Value bl, mlir::Value al) {
    return node(OpKind::GdnScan, {qkv, bl, al, position(), state_[gdn.state]},
                "ds" + std::to_string(l), gdn.value_dim(), scan_attrs(gdn, l));
  }

  // ---- feed-forwards -------------------------------------------------------

  mlir::Value feed(const GatedFfn& ffn, uint32_t l, mlir::Value f) {
    const std::string s = std::to_string(l);
    const uint32_t ff = ffn.up.out;
    const mlir::Value up = matmul(f, "up" + s, ffn.up, l);
    mlir::Value gate = matmul(f, "gate" + s, ffn.gate, l);
    gate = ffn.activation == Activation::Silu
               ? pointwise(OpKind::Silu, {gate}, "silu" + s, ff, l)
               : pointwise(OpKind::GeluFp16, {gate}, "gelu" + s, ff, l);
    const mlir::Value fh = pointwise(OpKind::Mul, {gate, up}, "fh" + s, ff, l);
    return matmul(fh, "fo" + s, ffn.down, l);
  }

  // The mixture-of-experts feed-forward. The router node produces the selected
  // expert ids and their weights as its two results; the expert node consumes
  // them and sums the routed experts. A shared expert, when the model has one,
  // adds its gated projection scaled by the sigmoid of the shared gate.
  mlir::Value feed(const MoeFfn& ffn, uint32_t l, mlir::Value f) {
    const std::string s = std::to_string(l);
    const uint32_t d = r.width;
    const Activation act = ffn.activation;

    Attrs route_a = {layer_a(l), slot_a(ffn.router.weight),
                     attr_i("experts", int64_t(ffn.experts)),
                     attr_i("used", int64_t(ffn.used))};
    if (ffn.weight_scale != 1.0)
      route_a.push_back(attr_f("scale", ffn.weight_scale));
    rows(route_a);
    // The ids and weights are `used` values per position, so their shape is the
    // phase's, like every other node's (a prefill position is a row).
    const std::vector<mlir::Value> routed =
        b.nodes(OpKind::MoeRoute, {f}, {"moe_ids" + s, "moe_w" + s},
                {shape(ffn.used), shape(ffn.used)},
                {sx::DType::Int64, sx::DType::Float32}, route_a);
    if (routed.size() != 2) return mlir::Value{};

    Attrs expert_a = {
        layer_a(l), slot_a(ffn.routed.gate.weight),
        attr_i("slot_up", int64_t(ffn.routed.up.weight)),
        attr_i("slot_down", int64_t(ffn.routed.down.weight)),
        attr_i("experts", int64_t(ffn.experts)),
        attr_i("used", int64_t(ffn.used)),
        attr_s("activation", act == Activation::Silu ? "silu" : "gelu")};
    // The phase's row marker: the executor groups a prefill's rows by expert on
    // it, and runs one decode gemv per (row, expert) without it.
    rows(expert_a);
    const mlir::Value moe = node(OpKind::MoeExperts, {f, routed[0], routed[1]},
                                 "moe" + s, d, std::move(expert_a));
    if (!ffn.shared || ffn.shared_gate == WeightSlot::None) return moe;

    const Expert& shared = *ffn.shared;
    const uint32_t ff = shared.up.out;
    const mlir::Value sup = matmul(f, "sup" + s, shared.up, l);
    mlir::Value sgate = matmul(f, "sgate" + s, shared.gate, l);
    sgate = act == Activation::Silu
                ? pointwise(OpKind::Silu, {sgate}, "silu" + s, ff, l)
                : pointwise(OpKind::GeluFp16, {sgate}, "gelu" + s, ff, l);
    const mlir::Value sfh =
        pointwise(OpKind::Mul, {sgate, sup}, "sfh" + s, ff, l);
    const mlir::Value sfo = matmul(sfh, "sfo" + s, shared.down, l);
    Attrs gate_a = {layer_a(l), slot_a(ffn.shared_gate)};
    rows(gate_a);
    const mlir::Value sgi = node(OpKind::MatvecF32, {f}, "sgi" + s, 1,
                                 std::move(gate_a));
    const mlir::Value sgs =
        pointwise(OpKind::Sigmoid, {sgi}, "sgs" + s, 1, l);
    const mlir::Value sg =
        pointwise(OpKind::ScaleBy, {sfo, sgs}, "sg" + s, d, l);
    return pointwise(OpKind::Add, {moe, sg}, "ffo" + s, d, l);
  }

  // ---- the block -----------------------------------------------------------

  mlir::Value block(const Block& blk, uint32_t l, mlir::Value residual) {
    const std::string s = std::to_string(l);
    const uint32_t d = r.width;

    const mlir::Value h = norm(residual, "h" + s, d, l, WeightSlot::AttnNorm);
    const mlir::Value mixed = std::visit(
        [&](const auto& kind) { return mix(kind, l, h); }, blk.mixer);
    const mlir::Value o =
        blk.post_mixer_norm
            ? norm(mixed, "o" + s, d, l, WeightSlot::PostAttentionNorm)
            : mixed;
    const mlir::Value x1 = pointwise(OpKind::Add, {o, residual}, "x1" + s, d, l);

    const mlir::Value f = norm(x1, "f" + s, d, l, WeightSlot::FfnNorm);
    const mlir::Value fed = std::visit(
        [&](const auto& kind) { return feed(kind, l, f); }, blk.ffn);
    const mlir::Value f2 =
        blk.post_ffn_norm ? norm(fed, "f2" + s, d, l, WeightSlot::PostFfwNorm)
                          : fed;
    const mlir::Value x2 = pointwise(OpKind::Add, {f2, x1}, "x2" + s, d, l);

    const mlir::Value x3 = blk.per_layer_gate ? per_layer_gate(l, x2) : x2;
    return blk.output_scale ? scale(x3, "out" + s, d, *blk.output_scale) : x3;
  }

  // The per-layer gate: the block's output gates its slice of the per-layer
  // input, and the result is projected back and added to the stream.
  mlir::Value per_layer_gate(uint32_t l, mlir::Value x2) {
    const std::string s = std::to_string(l);
    const uint32_t d = r.width;
    const uint32_t per = r.per_layer_input->width;

    const mlir::Value ple = per_layer_slice(l);
    Attrs g0_a = {layer_a(l), slot_a(WeightSlot::InpGate)};
    rows(g0_a);
    mlir::Value g0 = node(OpKind::MatvecF32, {x2}, "g0" + s, per, std::move(g0_a));
    g0 = pointwise(OpKind::GeluFp16, {g0}, "g0g" + s, per, l);
    g0 = pointwise(OpKind::Mul, {g0, ple}, "g0m" + s, per, l);
    Attrs g1_a = {layer_a(l), slot_a(WeightSlot::Proj)};
    rows(g1_a);
    const mlir::Value g1 =
        node(OpKind::MatvecF32, {g0}, "g1" + s, d, std::move(g1_a));
    const mlir::Value g2 = norm(g1, "g2" + s, d, l, WeightSlot::PostNorm);
    return pointwise(OpKind::Add, {x2, g2}, "x3" + s, d, l);
  }

  uint32_t per_layer_total() const {
    return uint32_t(r.blocks.size()) * r.per_layer_input->width;
  }
};

// ---------------------------------------------------------------------------
// A decode step: one position (or `rows` of them, dealt by the builder), an
// attention that appends to its cache as it reads it, the per-layer input
// computed once ahead of the blocks, and a head.
// ---------------------------------------------------------------------------
class DecodePhase final : public Phase {
public:
  DecodePhase(compiler::BoomBuilder& builder, const Recipe& recipe,
              uint64_t rows)
      : Phase(builder, recipe), rows_(rows),
        mmq_available_(decode_mmq_available()) {
    // The graph's free variables, then the cache handles: block arguments, in
    // this order.
    token_id_ = b.input("token_id", {}, sx::DType::Int64);
    pos_ = b.input("pos", {}, sx::DType::Int64);
    kv_.resize(r.caches);
    for (uint32_t c = 0; c < r.caches; ++c)
      kv_[c] = b.kv(c);
    state_.resize(r.states);
    for (uint32_t i = 0; i < r.states; ++i)
      state_[i] = b.state(i);
  }

  // The model's logits: the final norm, the output projection, the softcap
  // when the model caps them.
  mlir::Value head(mlir::Value residual) {
    const mlir::Value hn =
        b.node(OpKind::RmsNorm, {residual}, "result_norm", {int64_t(r.width)},
               {slot_a(WeightSlot::OutputNorm)});
    Attrs a = {slot_a(r.head.weight)};
    family(a, r.head.type, r.width, r.vocab);
    mlir::Value out = b.node(OpKind::QuantizedMatmul, {hn}, "logits",
                             {int64_t(r.vocab)}, std::move(a));
    if (r.head.softcap != 0.0)
      out = b.node(OpKind::Softcap, {out}, "logits_sc", {int64_t(r.vocab)},
                   {attr_f("scale", r.head.softcap)});
    return out;
  }

protected:
  std::vector<int64_t> shape(uint64_t width) const override {
    return {int64_t(width)};
  }
  mlir::Value position() const override { return pos_; }
  mlir::Value tokens() const override { return token_id_; }
  void rows(Attrs&) const override {}
  void norm_extent(Attrs&, uint64_t) const override {}
  void lowering(Attrs& a, const Projection& p) const override {
    family(a, p.type, p.in, p.out);
  }

  // The split-KV decode schedule is carried by the op: the chunk size (keys
  // per split) and the partial-combine order. Chunk boundaries are absolute
  // (chunk index = key position / C) and partials merge in ascending chunk
  // index, so a query's reduction tree depends only on the key positions —
  // never on capacity, the current depth or prefix reuse. The value is the
  // class default or the per-device tuned pick; `context_length` only bounds
  // the depths the tuner measures at.
  mlir::Value attend(const SelfAttention& at, uint32_t l, mlir::Value q,
                     mlir::Value k, mlir::Value v) override {
    Attrs a = {layer_a(l), attr_i("kv_source_layer", int64_t(at.cache))};
    if (at.scale != 1.0)
      a.push_back(attr_f("scale", at.scale));
#ifdef SONICBOOM_USE_CUDA
    const int n_kv = int(at.heads_kv);
    const int n_q = int(at.heads_q);
    a.push_back(attr_i("decode_chunk",
                       int64_t(nn::cuda::attn_decode_plan_chunk(
                           int(at.head_dim), n_q / std::max(1, n_kv), n_q, n_kv,
                           at.window, r.context_length,
                           numerics_kv_storage() == KvStorage::F16,
                           numerics_kv_storage() == KvStorage::Q8))));
    a.push_back(attr_s("decode_combine", "chunk_index_asc"));
#endif
    const std::string name = "attn" + std::to_string(l);
    if (at.owns_cache)
      return b.node(OpKind::Attention, {q, k, v, pos_, kv_[at.cache]}, name,
                    {int64_t(at.q_width())}, std::move(a));
    return b.node(OpKind::Attention, {q, pos_, kv_[at.cache]}, name,
                  {int64_t(at.q_width())}, std::move(a));
  }

  // The scan starts at the decode position (the verify row scale deals the
  // rows out; the node needs no batch).
  mlir::Value recur(const GatedDeltaNet& gdn, uint32_t l, mlir::Value qkv,
                    mlir::Value bl, mlir::Value al) override {
    return scan(gdn, l, qkv, bl, al);
  }

  // One full-width projection of the embedding and one per-layer row per
  // token, combined once over every layer's slice:
  //   (norm(proj(inp) / sqrt(d)) + row * sqrt(per)) / sqrt(2).
  // Each block then slices its own `per` values out.
  void per_layer_preamble() override {
    const uint32_t d = r.width;
    const uint32_t per = r.per_layer_input->width;
    const uint32_t total = per_layer_total();
    const mlir::Value row =
        b.node(OpKind::Embedding, {token_id_}, "ple_row", {int64_t(total)},
               {slot_a(WeightSlot::PerLayerTokenEmbd)});
    const float proj_scale = 1.0f / std::sqrt(float(d));
    const float row_scale = std::sqrt(float(per));
    const float combine_scale = 1.0f / std::sqrt(2.0f);
    mlir::Value all =
        b.node(OpKind::MatvecBf16, {inp_}, "ple_proj", {int64_t(total)},
               {attr_i("offset", int64_t(0)), attr_i("rows", int64_t(total))});
    all = b.node(OpKind::Scale, {all}, "ple_projs", {int64_t(total)},
                 {attr_f("scale", double(proj_scale))});
    all = b.node(OpKind::RmsNormHeads, {all}, "ple_projn", {int64_t(total)},
                 {layer_a(0), slot_a(WeightSlot::PerLayerProjNorm),
                  attr_i("dim", int64_t(per))});
    const mlir::Value row_s =
        b.node(OpKind::Scale, {row}, "ple_s", {int64_t(total)},
               {attr_f("scale", double(row_scale))});
    const mlir::Value sum =
        b.node(OpKind::Add, {all, row_s}, "ple_add", {int64_t(total)}, {});
    per_layer_all_ = b.node(OpKind::Scale, {sum}, "ple_all", {int64_t(total)},
                            {attr_f("scale", double(combine_scale))});
  }

  mlir::Value per_layer_slice(uint32_t l) override {
    const uint32_t per = r.per_layer_input->width;
    return b.node(OpKind::Slice, {per_layer_all_}, "ple" + std::to_string(l),
                  {int64_t(per)},
                  {attr_i("offset", int64_t(l) * per),
                   attr_i("length", int64_t(per))});
  }

private:
  // A projection whose (type, rows, cols) shape was measured faster on the mmq
  // matmul carries the prefill lowering attributes — the extent, the input
  // width, the activation scale layout of the weight type — and `mmq_rows`,
  // which selects that matmul for an activation with no fused producer. A
  // shape on the gemv carries nothing and takes the ordinary decode lowering.
  void family(Attrs& a, quant::QuantType qt, uint64_t in_w,
              uint64_t out_w) const {
    if (!mmq_available_ || !decode_family_mmq(qt, in_w, out_w))
      return;
    a.push_back(attr_i("batch", int64_t(rows_)));
    a.push_back(attr_i("dim", int64_t(in_w)));
    a.push_back(attr_i("ds", mmq_scale_layout(qt)));
    a.push_back(attr_i("mmq_rows", int64_t(1)));
  }

  uint64_t rows_;
  bool mmq_available_;
  mlir::Value token_id_{};
  mlir::Value pos_{};
  mlir::Value per_layer_all_{};
};

// ---------------------------------------------------------------------------
// A prefill segment: `n` positions at once ([n, width] activations, or a
// dynamic extent), a flash attention that reads the segment's own K and V as
// values, the per-layer input computed inside each block, and the cache
// appends after the last block.
// ---------------------------------------------------------------------------
class PrefillPhase final : public Phase {
public:
  PrefillPhase(compiler::BoomBuilder& builder, const Recipe& recipe, uint64_t n,
               bool dynamic_extent)
      : Phase(builder, recipe), n_(n), dynamic_(dynamic_extent) {
    tokens_ = b.input(
        "tokens", {dynamic_ ? mlir::ShapedType::kDynamic : int64_t(n_)},
        sx::DType::Int64);
    start_pos_ = b.input("start_pos", {}, sx::DType::Int64);
    kv_.resize(r.caches);
    for (uint32_t c = 0; c < r.caches; ++c)
      kv_[c] = b.kv(c);
    state_.resize(r.states);
    for (uint32_t i = 0; i < r.states; ++i)
      state_[i] = b.state(i);
    k_.resize(r.caches);
    v_.resize(r.caches);
  }

  // Every owned cache takes the segment's K and V, in cache order. A cache no
  // block owns (the blocks of a recurrent model own no cache) has no segment K
  // or V to append and is left alone.
  void append_caches() {
    for (uint32_t c = 0; c < r.caches; ++c) {
      if (!k_[c]) continue;
      b.node(OpKind::KvAppend, {k_[c], v_[c], start_pos_, kv_[c]},
             "kv_append" + std::to_string(c), {},
             {layer_a(c), batch_a()});
    }
  }

protected:
  std::vector<int64_t> shape(uint64_t width) const override {
    return {dynamic_ ? mlir::ShapedType::kDynamic : int64_t(n_),
            int64_t(width)};
  }
  mlir::Value position() const override { return start_pos_; }
  mlir::Value tokens() const override { return tokens_; }
  void rows(Attrs& a) const override { a.push_back(batch_a()); }
  void norm_extent(Attrs& a, uint64_t width) const override {
    a.push_back(batch_a());
    a.push_back(attr_i("dim", int64_t(width)));
  }
  void lowering(Attrs& a, const Projection& p) const override {
    a.push_back(batch_a());
    a.push_back(attr_i("dim", int64_t(p.in)));
    a.push_back(attr_i("ds", mmq_scale_layout(p.type)));
    a.push_back(decomposition(p.out, p.in, p.type));
  }

  // A layer that owns its cache attends over its own K and V and leaves them
  // for the append; one that does not attends over its owner's.
  mlir::Value attend(const SelfAttention& at, uint32_t l, mlir::Value q,
                     mlir::Value k, mlir::Value v) override {
    Attrs a = {layer_a(l), batch_a(),
               attr_i("kv_source_layer", int64_t(at.cache))};
    if (at.scale != 1.0)
      a.push_back(attr_f("scale", at.scale));
    if (at.owns_cache) {
      k_[at.cache] = k;
      v_[at.cache] = v;
    }
    return node(OpKind::FlashAttention,
                {q, k_[at.cache], v_[at.cache], start_pos_, kv_[at.cache]},
                "attn" + std::to_string(l), at.q_width(), std::move(a));
  }

  // The scan covers this segment's rows, starting at its start position.
  mlir::Value recur(const GatedDeltaNet& gdn, uint32_t l, mlir::Value qkv,
                    mlir::Value bl, mlir::Value al) override {
    return scan(gdn, l, qkv, bl, al);
  }

  void per_layer_preamble() override {
    per_layer_row_ =
        node(OpKind::Embedding, {tokens_}, "ple_row", per_layer_total(),
             {slot_a(WeightSlot::PerLayerTokenEmbd), batch_a()});
  }

  // The same combination as the decode step's, over this block's slice only:
  //   (norm(proj_l(inp) / sqrt(d)) + row_l * sqrt(per)) / sqrt(2).
  mlir::Value per_layer_slice(uint32_t l) override {
    const std::string s = std::to_string(l);
    const uint32_t d = r.width;
    const uint32_t per = r.per_layer_input->width;
    const float proj_scale = 1.0f / std::sqrt(float(d));

    mlir::Value proj =
        node(OpKind::MatvecBf16, {inp_}, "proj" + s, per,
             {layer_a(l), batch_a(), attr_i("offset", int64_t(l) * per),
              attr_i("rows", int64_t(per))});
    proj = scale(proj, "projs" + s, per, proj_scale);
    proj = norm(proj, "projn" + s, per, l, WeightSlot::PerLayerProjNorm);
    const mlir::Value col =
        node(OpKind::Slice, {per_layer_row_}, "ple_col" + s, per,
             {attr_i("offset", int64_t(l) * per), attr_i("length", int64_t(per)),
              attr_i("stride", int64_t(per_layer_total()))});
    const mlir::Value col_s =
        scale(col, "ple_scaled" + s, per, double(std::sqrt(float(per))));
    const mlir::Value sum =
        node(OpKind::Add, {proj, col_s}, "ple_sum" + s, per, {});
    return scale(sum, "ple" + s, per, 1.0 / std::sqrt(2.0));
  }

private:
  sx::Attribute batch_a() const { return attr_i("batch", int64_t(n_)); }

  // How the mmq matmul of this shape is decomposed on the device.
  static sx::Attribute decomposition(uint64_t rows, uint64_t cols,
                                     quant::QuantType type) {
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
  }

  uint64_t n_;
  bool dynamic_;
  mlir::Value tokens_{};
  mlir::Value start_pos_{};
  mlir::Value per_layer_row_{};
  std::vector<mlir::Value> k_, v_;  // the segment's K and V, by cache index
};

// An empty recipe is a model whose layer kinds the recipe cannot express
// (recipe_for leaves it empty for one): there is no graph to write, and it is
// said so instead of building one from no blocks.
bool recipe_expressible(const Recipe& recipe) {
  if (!recipe.blocks.empty()) return true;
  std::fprintf(stderr,
               "sonicboom: model not supported yet (recurrent mixer layers)\n");
  return false;
}

const char* consumer_name(DecodeConsumer consumer) {
  switch (consumer) {
    case DecodeConsumer::Logits: return "logits";
    case DecodeConsumer::PrecapLogits: return "precap_logits";
    case DecodeConsumer::Argmax: return "argmax";
    case DecodeConsumer::SampleSelect: return "sample_select";
  }
  return "logits";
}

} // namespace

GraphIr emit_decode_ir(const Recipe& recipe, const DecodeGraphSpec& spec) {
  GraphIr ir;
  ir.context = std::make_unique<mlir::MLIRContext>();
  if (!recipe_expressible(recipe)) return ir;
  compiler::BoomBuilder b(ir.context.get(),
                          spec.rows > 1 ? "gemma4_verify" : "gemma4_decode");
  if (spec.rows > 1)
    b.set_row_scale(int64_t(spec.rows), "verify");

  DecodePhase phase(b, recipe, spec.rows);
  const mlir::Value out = phase.head(phase.stack());

  // The graph ends at the model's logits. What reads them — and so whether the
  // softcap is computed, how much of the head is, and what leaves the device —
  // is declared, not written.
  b.set_output(out);
  b.set_consumer(consumer_name(spec.consumer),
                 spec.allowed_list ? int64_t(nn::kGrammarGatherCap) : 0);

  ir.module = b.finish();
  return ir;
}

GraphIr emit_prefill_ir(const Recipe& recipe, uint64_t n, bool dynamic_extent) {
  GraphIr ir;
  ir.context = std::make_unique<mlir::MLIRContext>();
  if (!recipe_expressible(recipe)) return ir;
  compiler::BoomBuilder b(ir.context.get(), "gemma4_prefill", n);

  PrefillPhase phase(b, recipe, n, dynamic_extent);
  const mlir::Value residual = phase.stack();
  phase.append_caches();
  b.set_output(residual);

  ir.module = b.finish();
  return ir;
}

} // namespace sonicboom::model
