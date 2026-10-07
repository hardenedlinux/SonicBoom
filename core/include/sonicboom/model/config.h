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
// (models/gemma-4-E4B-it-Q3_K_M.gguf); see design/gemma4-model-map.md. The
// values are validated by the loader against the metadata and the tensor
// directory, never taken on faith.

namespace sonicboom::model {

enum class AttentionKind : uint8_t { SlidingWindow, Global };

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
