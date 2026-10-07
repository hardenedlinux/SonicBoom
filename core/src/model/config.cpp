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

#include <sonicboom/model/config.h>

namespace sonicboom::model {

bool Gemma4Config::valid() const noexcept {
  if (block_count == 0) return false;
  if (embedding_length == 0 || feed_forward_length == 0) return false;
  if (head_count == 0 || head_count_kv == 0) return false;
  if (head_dim_global == 0 || head_dim_swa == 0) return false;
  if (head_count % head_count_kv != 0) return false;  // GQA grouping must be exact
  // The contiguous query width of a sliding-window layer is
  // head_count * head_dim_swa, which must match the embedding length's attention
  // projection only structurally here (shapes are checked by the loader).
  if (sliding_window_pattern.size() != block_count) return false;
  if (rms_epsilon < 0.0f) return false;
  if (rope_base <= 0.0f || rope_base_swa <= 0.0f) return false;
  if (shared_kv_layers >= block_count) return false;  // need >= 1 KV layer
  return true;
}

LayerConfig Gemma4Config::layer_config(uint32_t layer) const {
  LayerConfig c;
  c.layer = layer;
  c.n_heads_q = head_count;
  c.n_heads_kv = head_count_kv;

  const bool is_swa =
      layer < sliding_window_pattern.size() && sliding_window_pattern[layer];
  if (is_swa) {
    c.attention = AttentionKind::SlidingWindow;
    c.head_dim = head_dim_swa;
    c.rope_base = rope_base_swa;
    c.sliding_window = sliding_window;
  } else {
    c.attention = AttentionKind::Global;
    c.head_dim = head_dim_global;
    c.rope_base = rope_base;
    c.sliding_window = 0;
  }
  return c;
}

uint32_t Gemma4Config::n_layer_kv() const noexcept {
  if (shared_kv_layers >= block_count) return 0;
  return block_count - shared_kv_layers;
}

bool Gemma4Config::has_kv(uint32_t layer) const noexcept {
  return layer < n_layer_kv();
}

uint32_t Gemma4Config::kv_donor_layer(uint32_t layer) const noexcept {
  if (has_kv(layer)) return layer;
  const uint32_t n_kv = n_layer_kv();
  const bool is_swa =
      layer < sliding_window_pattern.size() && sliding_window_pattern[layer];
  return n_kv - (is_swa ? 2u : 1u);
}

} // namespace sonicboom::model
