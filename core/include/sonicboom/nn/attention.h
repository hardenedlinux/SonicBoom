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

// Reference scaled dot-product attention for the native transformer path
// (float32). This is the correctness kernel Phase 5 builds the decode loop on;
// Phase 6 swaps the raw dot products for the quantized matmul kernels.

namespace sonicboom::nn {

// Multi-query scaled dot-product attention with grouped-query attention (GQA),
// causal masking, sliding-window masking, and Gemma's logit softcapping.
//
// Layouts (row-major, inner dim contiguous):
//   q   [n_q, n_heads_q, head_dim]     — query vectors (RoPE already applied)
//   k   [n_k, n_heads_kv, head_dim]    — key vectors   (RoPE already applied)
//   v   [n_k, n_heads_kv, head_dim]    — value vectors
//   out [n_q, n_heads_q, head_dim]     — attention output
//
// Query head h attends to KV head `h * n_heads_kv / n_heads_q` (the GQA
// grouping: n_heads_q is a multiple of n_heads_kv).
//
// Per (query position i, query head h):
//   score_j = scale * sum_d q[i,h,d] * k[j, kv_h, d]
//   masked  if (causal && j > i) or (sliding_window > 0 && i - j >= sliding_window)
//   score_j = softcap * tanh(score_j / softcap)   if softcap > 0
//   prob    = softmax(score over unmasked j)
//   out[i,h,:] = sum_j prob_j * v[j, kv_h, :]
//
// `scale` is the logit temperature. Gemma 4 uses 1.0 (no pre-attn scaling):
// Q/K are RMSNormed to unit norm, so the dot product is a cosine similarity and
// needs no 1/sqrt(head_dim) temperature. (Other models pass 1/sqrt(head_dim).)
//
// Returns false (writing nothing) when any span size disagrees with the stated
// dims, or when head_dim == 0 or n_heads_q is not a positive multiple of
// n_heads_kv.
bool scaled_dot_product_attention(std::span<const float> q, uint64_t n_q,
                                  std::span<const float> k,
                                  std::span<const float> v, uint64_t n_k,
                                  uint64_t n_heads_q, uint64_t n_heads_kv,
                                  uint64_t head_dim, bool causal,
                                  uint64_t sliding_window, float softcap,
                                  float scale, std::span<float> out);

// GQA broadcast for single-token (n_k == 1) attention: the SDPA degenerates to
// copying the single value vector to every query head, one head_dim slice each:
//   dst[hq*head_dim + i] = src[(hq * n_heads_kv / n_heads_q) * head_dim + i]
// `src` holds n_heads_kv value heads (n_heads_kv * head_dim floats), `dst` holds
// n_heads_q heads. This is the decode block's shared-layer donor-V spread, hoisted
// to a named kernel (the K vector is irrelevant when the softmax is over one logit).
//
// Returns false (writing nothing) when head_dim == 0, n_heads_q/n_heads_kv == 0,
// n_heads_q is not a multiple of n_heads_kv, or a span is too small.
bool gqa_broadcast(std::span<const float> src, std::span<float> dst,
                   uint64_t n_heads_q, uint64_t n_heads_kv, uint64_t head_dim);

// Incremental single-token decode attention over a per-layer KV cache. The query
// is one token (`n_heads_q` heads of `head_dim` floats) at absolute position
// `pos`; it attends over the cached keys/values at positions
// [max(0, pos - sliding_window + 1), pos] — causal, further restricted by the
// sliding window when `sliding_window > 0` (Gemma SWA layers). `k_cache` and
// `v_cache` are flat [n_slots, n_heads_kv, head_dim] buffers of fp16-rounded
// f32 (the llama.cpp flash-attention KV representation); the slot for absolute
// position j is `j % n_slots`, so a global layer uses n_slots == n_ctx (no wrap)
// and a sliding-window layer uses n_slots == sliding_window (ring buffer).
//
// Per query head h (attending KV head `h * n_heads_kv / n_heads_q`):
//   score_j = scale * sum_d q[h,d] * k_cache[(j % n_slots), kv_h, d]
//   prob    = softmax(score over valid j)       // Gemma softcap == 0, scale 1.0
//   out[h,:] = sum_j prob_j * v_cache[(j % n_slots), kv_h, :]
//
// Returns false (writing nothing) when head_dim == 0, n_heads_q is not a
// positive multiple of n_heads_kv, or a span is too small.
bool decode_attention(std::span<const float> q, std::span<const float> k_cache,
                      std::span<const float> v_cache, uint64_t pos,
                      uint64_t n_slots, uint64_t n_heads_q, uint64_t n_heads_kv,
                      uint64_t head_dim, uint64_t sliding_window, float scale,
                      std::span<float> out);

} // namespace sonicboom::nn
