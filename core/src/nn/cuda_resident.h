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

// Internal device-resident elementwise / norm / RoPE / dense-matvec primitives
// (Phase 6a Option A). These mirror the public <sonicboom/nn/cuda_elementwise.h>
// entry points but take/return DEVICE pointers for the activation operands and
// perform no H2D/D2H: the caller owns the device buffers and chains these
// back-to-back so activations stay resident and the host only synchronizes at a
// block boundary. Weight operands are host spans, uploaded once and cached by
// (pointer, byte size) via the same cache the public path uses.
//
// NOT part of the public Layer 2 surface: this header lives under core/src/
// (not core/include/), so it may use raw device pointers. It is included only
// from .cu translation units (the caller in core/src/model/cuda_block.cu and
// the definitions in core/src/nn/cuda_elementwise.cu).

namespace sonicboom::nn::cuda {

// y[i] = (x[i] / sqrt(mean_j(x[j]^2) + eps)) * weight[i]; weight empty => 1.0.
// x, y are device buffers of n floats; weight is a host span of n floats (or
// empty). Mirrors nn::rms_norm / nn::cuda::rms_norm.
bool rms_norm_dev(std::span<const float> weight, const float* x, float eps,
                  float* y, int n);

// Per-head RMSNorm over `heads` heads of `head_dim` floats each:
//   y[h*head_dim + i] = (x[h*head_dim + i] / sqrt(mean_i(x[h*head_dim+i]^2) + eps)) * weight[i]
// `weight` is a single head_dim-length vector shared across heads (empty =>
// unit). Mirrors exec.cpp's per-head nn::rms_norm loop.
bool rms_norm_heads_dev(std::span<const float> weight, const float* x, float eps,
                        float* y, int head_dim, int heads);

// GGML_GELU_FP16 tanh-GELU (fp16 round-trip), elementwise. x/y may alias.
bool gelu_fp16_dev(const float* x, float* y, int n);

bool mul_dev(const float* a, const float* b, float* y, int n);
bool add_dev(const float* a, const float* b, float* y, int n);
bool scale_dev(const float* x, float s, float* y, int n);

// In-place f32 -> fp16 -> f32 round-trip.
bool cast_fp16_dev(float* v, int n);

// In-place f32 -> bf16 -> f32 round-trip (matches bf16_to_f32(f32_to_bf16(v))).
bool cast_bf16_dev(float* v, int n);

// In-place multi-head NEOX RoPE over a concatenated buffer of `heads` heads of
// `head_dim` floats each (mirrors nn::cuda::rope_neox_heads). freq_factors is a
// host span of head_dim/2 floats (empty => 1.0 per pair).
bool rope_neox_heads_dev(float* x, int head_dim, int heads, uint64_t pos,
                         float base, float freq_scale,
                         std::span<const float> freq_factors);

// Dense (non-K-quant) matvec, device in/out: y[j] = sum_i W[j*cols + i] * x[i].
// W is row-major with `cols` contiguous (GGUF order); rows = y length. W is a
// host span uploaded+cached. Mirrors nn::cuda::matvec_f32 / matvec_bf16.
bool matvec_f32_dev(std::span<const float> W, int cols, const float* x,
                    float* y, int rows);
bool matvec_bf16_dev(std::span<const uint16_t> W, int cols, const float* x,
                     float* y, int rows);

// GQA broadcast for n_k == 1 attention (the SDPA degenerates to copying the
// single value vector, GQA-broadcast): dst[hq*head_dim + i] =
// src[(hq * n_kv / n_q) * head_dim + i]. Mirrors exec.cpp's shared-layer donor
// broadcast and the KV-layer SDPA at n_k == 1.
bool gqa_broadcast_dev(const float* src, float* dst, int n_q, int n_kv, int head_dim);

// Incremental single-token decode attention over a device-resident KV cache
// (mirrors nn::decode_attention, one thread per query head). q is the one-token
// RoPE'd query [n_heads_q, head_dim]; k_cache/v_cache are flat device buffers
// [n_slots, n_heads_kv, head_dim] (fp16-rounded f32), slot for absolute position
// j == j % n_slots. out is [n_heads_q, head_dim].
bool decode_attention_dev(const float* q, const float* k_cache,
                          const float* v_cache, uint64_t pos, uint64_t n_slots,
                          int n_heads_q, int n_heads_kv, int head_dim,
                          uint64_t sliding_window, float scale, float* out);

// Prefill flash attention (mirrors nn::scaled_dot_product_attention for n_q > 1):
// tiled online-softmax over the KV sequence so the [n_q, n_kv] scores matrix is
// never materialized (O(n_q + n_kv) memory). q is [n_q, n_heads_q, head_dim] f32
// (RoPE'd); k/v are [n_kv, n_heads_kv, head_dim] (fp16-rounded f32, same
// representation as the KV cache). The causal + sliding-window mask is computed
// in-kernel from (iq, j) — Gemma 4 has no data-dependent mask, so this is exactly
// equivalent to a precomputed mask. out is [n_q, n_heads_q, head_dim] f32.
// scale == 1.0 for Gemma 4; no logit softcap. One warp per (query position, query
// head); requires head_dim % 32 == 0, head_dim <= 512, n_heads_q <= 32.
bool flash_attention_dev(const float* q, const float* k, const float* v,
                         int n_q, int n_kv, int n_heads_q, int n_heads_kv,
                         int head_dim, uint64_t sliding_window, float scale,
                         float* out);

// Per-layer embedding combine: y[i] = (proj[i] + ple[i] * ple_scale) * combine_scale.
// `ple` is the caller-offset device pointer into the full per-layer embedding
// row (i.e. already advanced by `layer * per_layer_input`). Mirrors
// nn::layer_combine.
bool layer_combine_dev(const float* proj, const float* ple, float ple_scale,
                       float combine_scale, float* y, int n);

} // namespace sonicboom::nn::cuda
