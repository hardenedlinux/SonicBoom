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

#include <cstdint>
#include <span>
#include <vector>

#include <sonicboom/dtype.h>
#include <sonicboom/quant/quantized_tensor.h>

// Bound Gemma 4 weights (Phase 5A). A `Gemma4Weights` is a set of views over
// the GGUF reader's backing buffer: quantized tensors keep their packed bytes
// (QuantizedTensor borrows the byte span), f32 tensors are exposed as
// std::span<const float>, and the single bf16 tensor is exposed as raw halves.
//
// Alignment note: GGUF aligns every tensor's data offset to `general.alignment`
// (32 in the target model), so reinterpret_cast-ing the byte span to float (4)
// or uint16_t (2) is well-defined. The reader owns the buffer and outlives the
// model, so all spans are borrowed, never owned.

namespace sonicboom::model {

// bf16 <-> f32 conversions are shared with the nn kernels (whose bf16 matvec
// widens the same raw halves) and re-exported here for the model layer.
using sonicboom::bf16_to_f32;
using sonicboom::f32_to_bf16;

// One transformer block's weights (block_count of these, layers 0..41).
//
// All 17 per-block tensor groups are bound. The 7 added in Phase 5C-1
// (attn_q_norm/attn_k_norm/post_attention_norm/post_ffw_norm/post_norm/
// inp_gate/proj) are bound + shape/type-validated but NOT yet consumed by
// run_block; their exact arithmetic (q/k norm convention, per-layer gating,
// post-norm residual order) is unresolved pending an oracle (see
// design/gemma4-semantics.md). Bound-but-not-consumed spans are empty() only
// when the loader failed to bind them, which the loader already reports.
struct BlockWeights {
  std::span<const float> attn_norm;   // [embedding_length]
  std::span<const float> ffn_norm;    // [embedding_length]
  std::span<const float> attn_q_norm; // [head_dim] (256 SWA / 512 global); scalar broadcast in this GGUF
  std::span<const float> attn_k_norm; // [head_dim]; scalar broadcast
  std::span<const float> post_attention_norm; // [embedding_length]
  std::span<const float> post_ffw_norm;       // [embedding_length]
  std::span<const float> post_norm;           // [embedding_length]
  quant::QuantizedTensor attn_q;      // [embedding_length, n_heads_q*head_dim]
  quant::QuantizedTensor attn_k;      // [embedding_length, n_heads_kv*head_dim]
  quant::QuantizedTensor attn_v;      // [embedding_length, n_heads_kv*head_dim]
  quant::QuantizedTensor attn_output; // [n_heads_q*head_dim, embedding_length]
  quant::QuantizedTensor ffn_gate;    // [embedding_length, ff]
  quant::QuantizedTensor ffn_up;      // [embedding_length, ff]
  quant::QuantizedTensor ffn_down;    // [ff, embedding_length]
  std::span<const float> inp_gate;    // [embedding_length, per_layer_input] (f32)
  std::span<const float> proj;        // [per_layer_input, embedding_length] (f32)
  std::span<const float> layer_output_scale;  // [1]
};

struct Gemma4Weights {
  quant::QuantizedTensor token_embd;            // [embedding_length, vocab]
  quant::QuantizedTensor per_layer_token_embd;  // [block_count*per_layer_input, vocab]
  std::span<const float> per_layer_proj_norm;   // [per_layer_input]
  std::span<const uint16_t> per_layer_model_proj;  // bf16 [embedding_length, block_count*per_layer_input]
  std::span<const float> output_norm;           // [embedding_length]
  std::span<const float> rope_freqs;            // [256] (precomputed RoPE table)

  std::vector<BlockWeights> blocks;  // size == block_count

  bool empty() const noexcept { return blocks.empty(); }
};

} // namespace sonicboom::model
