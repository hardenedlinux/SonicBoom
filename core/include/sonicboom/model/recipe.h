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

using Mixer = std::variant<SelfAttention>;

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

using FeedForward = std::variant<GatedFfn>;

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
  std::vector<Block> blocks;

  Head head;
};

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
// extent bound to `n` when `dynamic_extent`.
planner::Graph build_prefill_graph(const Recipe& recipe, uint64_t n,
                                   bool dynamic_extent = false);

} // namespace sonicboom::model
