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

#pragma once

// A model as a recipe: what a decoder is made of, as data.
//
// A recipe says which layer kinds a model stacks and with which weights; it
// says nothing about how a graph is laid out for a phase. The emitter
// (src/model/emit.h) walks one recipe to write the decode, verify and prefill
// graphs, so a model is described once and every phase reads the same
// description.
//
// The layer kinds are a vocabulary. A block is a mixer (how positions exchange
// information) followed by a feed-forward (how a position transforms what it
// holds), each a variant over the kinds the engine knows. A new architecture
// is a new recipe over the existing kinds; a new kind is one alternative added
// to a variant, and the emitter then does not compile until every phase says
// how to write it.
//
// Nothing here is an MLIR type or a device type: a recipe can be built,
// inspected and compared with no compiler and no device.

#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <cstdint>
#include <optional>
#include <variant>
#include <vector>

namespace sonicboom::model {

// A linear map by a quantized weight, with its bias when the model has one.
// `in` and `out` are the activation widths on either side; `type` is the
// weight's quantization, which the emitter needs to choose a lowering.
struct Projection {
  WeightSlot weight = WeightSlot::None;
  WeightSlot bias = WeightSlot::None;  // None: no bias
  quant::QuantType type{};
  uint32_t in = 0;
  uint32_t out = 0;
  // True when the file stores the weight as f32 rather than in a quantized
  // format (some models keep their recurrent mixers' projections that way):
  // the emitter writes the f32 matvec and `type` is not used.
  bool f32 = false;
};

// ---- mixers ----------------------------------------------------------------

// Causal self-attention over a KV cache: grouped-query when `heads_kv` is less
// than `heads_q`, windowed when `window` is not zero.
struct SelfAttention {
  uint32_t heads_q = 0;
  uint32_t heads_kv = 0;
  uint32_t head_dim = 0;
  uint64_t window = 0;  // keys a query sees; 0 = all of them

  Projection q, k, v, out;

  bool qk_norm = false;  // per-head RMSNorm on Q and K, with learned weights
  bool v_norm = false;   // per-head RMSNorm on V, unit weight
  // Values per head that are rotated; 0 = the whole head. A model whose rope
  // covers only a prefix of each head (Qwen3.6) sets this, and the emitter
  // writes it on the block's Rope node.
  uint32_t rotary_dims = 0;
  // A gated attention output: the Q projection is twice as wide and the second
  // half of each head gates the attention result through a sigmoid, so `q.out`
  // is 2*q_width() and the attention's output stays q_width().
  bool output_gate = false;
  // Multiplies the attention scores; 1.0 when the model applies none (unit-
  // normed Q and K make the dot product a cosine already).
  double scale = 1.0;

  // The cache the layer attends over. A layer that owns it appends its own K
  // and V first; one that does not reads what the owner appended and has no
  // use for its `k` and `v` projections.
  uint32_t cache = 0;
  bool owns_cache = true;

  uint32_t q_width() const noexcept { return heads_q * head_dim; }
  uint32_t kv_width() const noexcept { return heads_kv * head_dim; }
};

// A gated DeltaNet: one recurrence over the positions, with the state the
// model carries per sequence. The output is gate * norm(scan(qkv)) projected
// back; the scan's inputs are projected by `qkv`, `beta` and `alpha`, and
// `gate` produces the second factor of the output.
struct GatedDeltaNet {
  uint32_t heads_k = 0;    // key heads (H_k)
  uint32_t heads_v = 0;    // value heads (H_v), a multiple of heads_k
  uint32_t head_dim = 0;   // the state size S, keys and values alike
  uint32_t conv_kernel = 0;  // convolution taps K

  Projection qkv, gate, beta, alpha, out;

  // Which state it owns, numbered from 0 over the recurrent blocks in the
  // order they appear in the stack.
  uint32_t state = 0;

  uint32_t key_dim() const noexcept { return heads_k * head_dim; }
  uint32_t value_dim() const noexcept { return heads_v * head_dim; }
  uint32_t channels() const noexcept { return 2 * key_dim() + value_dim(); }
};

using Mixer = std::variant<SelfAttention, GatedDeltaNet>;

// ---- feed-forwards ---------------------------------------------------------

enum class Activation : uint8_t {
  GeluFp16,  // GELU evaluated in fp16
  Silu,
};

// act(gate(x)) * up(x), projected back down.
struct GatedFfn {
  Projection up, gate, down;
  Activation activation = Activation::GeluFp16;
};

// One routed expert: act(gate(x)) * up(x), projected back down, the same three
// projections a GatedFfn has. The roles are indexed by the expert's number, so
// one record describes every expert of a block.
struct Expert {
  Projection gate, up, down;
};

// A mixture of experts: per position the router projects the activation onto
// `experts` logits, a softmax over them picks `used` experts, and the position's
// feed-forward is the weighted sum of those experts. `weight_scale` multiplies
// the routed weights (1.0 when the router's weights are used as they are).
//
// A shared expert, when the model has one, is always on: its gated projection
// runs alongside the routed sum and the result is gated by the sigmoid of a
// dense projection onto one value per position (`shared_gate`, an f32 [width]).
struct MoeFfn {
  Projection router;           // width -> experts
  uint32_t experts = 0;
  uint32_t used = 0;
  Expert routed;
  double weight_scale = 1.0;
  std::optional<Expert> shared;
  WeightSlot shared_gate = WeightSlot::None;  // None: no shared expert
  Activation activation = Activation::Silu;
};

using FeedForward = std::variant<GatedFfn, MoeFfn>;

// ---- the stack -------------------------------------------------------------

// One block: norm, mixer, residual; norm, feed-forward, residual.
struct Block {
  Mixer mixer;
  bool post_mixer_norm = false;  // a norm between the mixer and its residual
  FeedForward ffn;
  bool post_ffn_norm = false;    // a norm between the ffn and its residual
  // The block ends by mixing in its slice of the model's per-layer input
  // (Recipe::per_layer_input).
  bool per_layer_gate = false;
  // The whole block's output is multiplied by this.
  std::optional<double> output_scale;
};

// A second, per-layer token embedding: one row of `layers * width` values per
// token, of which each block gates in its own `width`.
struct PerLayerInput {
  uint32_t width = 0;
};

// The output projection onto the vocabulary.
struct Head {
  WeightSlot weight = WeightSlot::None;  // the token embedding when tied
  quant::QuantType type{};
  double softcap = 0.0;  // logits pass through softcap * tanh(x / softcap); 0 = none
};

struct Recipe {
  uint32_t width = 0;  // the residual stream
  uint64_t vocab = 0;
  uint64_t context_length = 0;

  bool scale_embedding = false;  // token embedding times sqrt(width)
  std::optional<PerLayerInput> per_layer_input;

  uint32_t caches = 0;  // KV caches; a block's mixer names the one it uses
  // Recurrent-block states; a GatedDeltaNet mixer names the one it owns.
  uint32_t states = 0;
  std::vector<Block> blocks;

  Head head;
};

// Whether any block routes to experts. A recipe whose routed matrices come
// from a device pool needs the host between the router and the experts, which
// a captured step cannot carry (§10).
inline bool has_routed_experts(const Recipe& recipe) noexcept {
  for (const Block& b : recipe.blocks)
    if (std::holds_alternative<MoeFfn>(b.ffn)) return true;
  return false;
}

// Whether the recipe carries recurrent state across steps. Every executed step
// advances it, so a speculative or measurement run on a live sequence would
// change the sequence it measures (§10).
inline bool has_recurrent_state(const Recipe& recipe) noexcept {
  return recipe.states > 0;
}

// The recipe of a loaded model: its architecture's layer kinds, with the
// widths of its config and the quantization of its weights.
Recipe recipe_for(const Gemma4Model& m);

// The graphs of a recipe. These are what the model entry points
// (build_decode_graph, build_gemma4_prefill_graph in model/plan.h) build for
// `recipe_for(m)`; a recipe written by hand goes through the same emitter, the
// same consumer resolution and the same passes.
planner::Graph build_decode_graph(const Recipe& recipe,
                                  const DecodeGraphSpec& spec);
// One prefill segment of `n` positions, lowered with no passes; over a dynamic
// extent bound to `n` when `dynamic_extent`. `steer_layer` / `stop_layer` follow
// emit_prefill_ir: >= 0 steers after / ends at that block.
planner::Graph build_prefill_graph(const Recipe& recipe, uint64_t n,
                                   bool dynamic_extent = false,
                                   int64_t steer_layer = -1,
                                   int64_t stop_layer = -1);

} // namespace sonicboom::model
