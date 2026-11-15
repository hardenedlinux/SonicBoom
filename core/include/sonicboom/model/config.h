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
#include <vector>

// Gemma 4 model configuration and per-layer execution plan (Phase 5A).
//
// These are the frozen model facts read from the target GGUF
// (models/gemma-4-E4B-it-Q5_K_M.gguf). The values are validated by the loader
// against the metadata and the tensor
// directory, never taken on faith.

namespace sonicboom::model {

enum class AttentionKind : uint8_t { SlidingWindow, Global };

// The transformer architecture the config describes (Section C.6 / Qwen
// generality). Distinguishes Gemma 4's specific block (per-layer embedding
// gate, Q/K per-head norm, post-attn/post-ffn norms, GELU, attention scale 1.0)
// from the standard decoder block (SwiGLU SiLU, no Q/K norm, no post-norms,
// attention scale 1/sqrt(head_dim)). Both are carried by the same config/weight
// containers; the plan emitter branches on this value.
enum class ModelArch : uint8_t { Gemma4, Qwen2, Qwen35Moe };

// The per-layer configuration that differs between the two attention kinds.
// n_heads_q / n_heads_kv are constant (8 / 2); head_dim and rope_base flip
// between the sliding-window (256, 10000) and global (512, 1e6) layers.
struct LayerConfig {
  uint32_t layer = 0;
  AttentionKind attention = AttentionKind::SlidingWindow;
  uint32_t n_heads_q = 0;
  uint32_t n_heads_kv = 0;
  uint32_t head_dim = 0;        // shared q/k/v head dim for this layer
  float rope_base = 0.0f;       // freq_base or freq_base_swa
  uint64_t sliding_window = 0;  // 0 for global (no window)
};

struct Gemma4Config {
  ModelArch arch = ModelArch::Gemma4;
  uint32_t block_count = 0;
  uint64_t vocab_size = 0;
  uint32_t embedding_length = 0;      // 2560
  uint32_t feed_forward_length = 0;   // 10240
  uint32_t head_count = 0;            // 8
  uint32_t head_count_kv = 0;         // 2
  uint32_t head_dim_global = 0;       // key_length = 512
  uint32_t head_dim_swa = 0;          // key_length_swa = 256
  uint32_t per_layer_input = 0;       // embedding_length_per_layer_input = 256
  uint64_t context_length = 0;        // 131072
  uint64_t sliding_window = 0;        // 512
  float rms_epsilon = 0.0f;           // 1e-6
  float final_logit_softcapping = 0.0f;  // 30.0
  float rope_base = 0.0f;             // freq_base = 1e6
  float rope_base_swa = 0.0f;         // freq_base_swa = 10000
  uint32_t shared_kv_layers = 0;      // attention.shared_kv_layers = 18

  // Layer 0..41: true = sliding-window, false = global.
  std::vector<bool> sliding_window_pattern;

  // Qwen3.6-MoE (qwen35moe). Every field is defaulted, so the configurations
  // read for the other architectures are unaffected.
  //
  // The mixture of experts in every block: n_expert routed experts, of which
  // n_expert_used are selected per token; expert_ff is one expert's
  // feed-forward length and shared_expert_ff the always-on shared expert's
  // (0 = none). Routed gate/up/down weights are selected by the router, whose
  // output is scaled by expert_weights_scale before the expert sums.
  uint32_t n_expert = 0;
  uint32_t n_expert_used = 0;
  uint32_t expert_ff = 0;
  uint32_t shared_expert_ff = 0;
  float expert_weights_scale = 1.0f;

  // Layer kinds: every full_attention_interval-th layer is full attention and
  // the others are gated DeltaNet (linear attention). recurrent_layers has one
  // entry per block (true = DeltaNet); it is read from the file's explicit
  // array when present and derived from full_attention_interval otherwise.
  uint32_t full_attention_interval = 4;
  std::vector<bool> recurrent_layers;

  // Gated DeltaNet hyperparameters. Read and shape-validated; the recurrent
  // path does not run yet.
  uint32_t ssm_conv_kernel = 0;
  uint32_t ssm_inner_size = 0;
  uint32_t ssm_state_size = 0;
  uint32_t ssm_time_step_rank = 0;
  uint32_t ssm_group_count = 0;

  // Partial rotary embedding: only the leading rope_dimension_count values of
  // each head are rotated, and rope_dimension_sections (4 entries) splits that
  // range into the sections the reference rotates. 0 = the whole head.
  uint32_t rope_dimension_count = 0;
  std::vector<uint32_t> rope_dimension_sections;

  // The trailing multi-token-prediction blocks. Read only: those blocks are
  // not loaded or executed.
  uint32_t n_layer_nextn = 0;

  // Structural sanity: non-zero head dims, a consistent block count, an
  // embedding length that matches head_count * head_dim_swa, etc.
  bool valid() const noexcept;

  // The per-layer config for `layer` (must be < block_count and have a
  // sliding_window_pattern entry).
  LayerConfig layer_config(uint32_t layer) const;

  // The first `block_count - shared_kv_layers` layers own a K/V projection and
  // KV cache; the remaining `shared_kv_layers` layers reuse one (Gemma 4
  // attention is "shared KV": layers 24..41 read layers 22/23's KV). Returns 0
  // when the model declares no sharing (shared_kv_layers == 0), in which case
  // every layer has its own KV.
  uint32_t n_layer_kv() const noexcept;

  // true if `layer` has its own K/V (a KV layer, not a shared layer).
  bool has_kv(uint32_t layer) const noexcept;

  // The layer whose KV cache `layer` reads. Identity for a KV layer; for a
  // shared layer, llama.cpp maps it to `n_layer_kv - (is_swa ? 2 : 1)` — i.e.
  // the last sliding-window KV layer (22) for SWA shared layers and the last
  // global KV layer (23) for global shared layers.
  uint32_t kv_donor_layer(uint32_t layer) const noexcept;
};

} // namespace sonicboom::model
