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

// Generic KV-cache session (C.4 prefix reuse): a persistent, model-agnostic
// handle to the device-resident KV cache plus the host-side token history used
// to match a shared prompt prefix on the next request. It knows nothing about
// any model's layer structure — the backend that allocates the cache sizes each
// layer's slots from its own model config, and prefix matching is a pure token
// comparison, so it works for any KV-cache model, not just Gemma 4.

#include <sonicboom/hal/buffer.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace sonicboom::planner {

// One KV layer's persistent device-resident cache (moved here from
// SonicCudaBackend so a session can own it). K is RoPE'd + per-head-normed then
// fp16-rounded; V per-head-normed then fp16-rounded. Flat
// [n_slots, n_heads_kv, head_dim] in f32; slot for absolute position j is
// `j % n_slots`.
struct KvCacheDev {
  hal::Buffer k_buf;
  hal::Buffer v_buf;
  uint64_t bytes = 0;  // per-buffer byte size (k and v equal)
  uint64_t n_slots = 0;
  uint64_t n_heads_kv = 0;
  uint64_t head_dim = 0;
};

// Persistent KV-cache session for prefix reuse. `caches` owns the per-layer
// device buffers (shared across the prefill/decode backends through the
// shared_ptr); `filled_tokens` records the token id at each prefilled position
// 0 .. filled_tokens.size()-1, so the next request can match a shared prefix.
// Empty by default (null caches, no history); the first prefill through it
// allocates the cache and records the tokens.
struct KvCacheSession {
  std::shared_ptr<std::vector<KvCacheDev>> caches;
  std::vector<uint64_t> filled_tokens;  // token at each prefilled position
  uint64_t n_ctx = 0;

  // Longest prefix of `prompt` that matches the filled token history. 0 when
  // there is no history (or the first token already differs).
  uint64_t common_prefix_len(std::span<const uint64_t> prompt) const noexcept {
    const uint64_t n =
        std::min<uint64_t>(prompt.size(), filled_tokens.size());
    uint64_t i = 0;
    while (i < n && prompt[i] == filled_tokens[i]) ++i;
    return i;
  }
};

} // namespace sonicboom::planner
