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

// CUDA backend for the native-transformer elementwise / norm / RoPE kernels and
// the dense (non-K-quant) gate matvecs (Phase 6a target 3, first slice). This is
// a SonicBoom-owned kernel set: each entry point mirrors a CPU reference in
// <sonicboom/nn/*.h> with the same signature, uploads its inputs, launches a
// device kernel, and copies the result back. It reproduces the CPU arithmetic to
// fp32 tolerance, not bit-exact — the CPU reference accumulates in double (ggml's
// ggml_float) while these kernels accumulate in fp32, the same relaxation the
// K-quant CUDA gemv in <sonicboom/quant/quantized_matmul_cuda.h> already makes.
//
// The declarations use only Layer 2 types (span, uint16_t, float) — no CUDA, c10
// or native-torch types appear in this public header. The CUDA runtime and
// kernels live entirely in core/src/nn/cuda_elementwise.cu. These entry points
// are only linked when the build is configured with SONICBOOM_USE_CUDA=ON.

namespace sonicboom::nn::cuda {

// True when a usable CUDA device is present (cached after the first call).
bool available() noexcept;

// y[i] = (x[i] / sqrt(mean_j(x[j]^2) + eps)) * weight[i]  (weight empty => 1.0).
bool rms_norm(std::span<const float> x, std::span<const float> weight, float eps,
              std::span<float> y);

// GGML_GELU_FP16 tanh-GELU: input rounded to fp16, tanh formula, result rounded
// back to fp16, with x <= -10 -> 0 and x >= 10 -> x. Mirrors nn::gelu_fp16.
bool gelu_fp16(std::span<const float> x, std::span<float> y);

// Elementwise a[i]*b[i], a[i]+b[i], x[i]*s (in y). Mirrors nn::mul/add/scale.
bool mul(std::span<const float> a, std::span<const float> b, std::span<float> y);
bool add(std::span<const float> a, std::span<const float> b, std::span<float> y);
bool scale(std::span<const float> x, float s, std::span<float> y);

// In-place f32 -> fp16 -> f32 round-trip (mirrors nn::cast_fp16).
bool cast_fp16(std::span<float> v);

// In-place GPT-NeoX rotary embedding (mirrors nn::rope_neox), freq_factors
// optional (proportional RoPE; empty => 1.0 per pair).
bool rope_neox(std::span<float> x, uint64_t pos, float theta_base,
               float freq_scale, std::span<const float> freq_factors);

// Same rotation over a concatenated multi-head buffer: `x` is n_q (or n_kv)
// heads of `head_dim` floats laid back-to-back, and head `g = i / head_dim`
// pairs offset `o = i % head_dim` with `o + head_dim/2` inside the same head
// (the per-head loop of exec.cpp run_block_core, but a single kernel launch
// over the whole Q/K span). freq_factors (if non-empty, length head_dim/2) is
// shared across heads — it is indexed by the intra-head offset, not by the flat
// index. Requires head_dim > 0, even, and dividing x.size() exactly.
bool rope_neox_heads(std::span<float> x, uint64_t head_dim, uint64_t pos,
                     float theta_base, float freq_scale,
                     std::span<const float> freq_factors);

// Dense (non-K-quant) matvec, for the per-layer gate weights: the f32
// inp_gate/proj matrices and the bf16 per_layer_model_proj. W is row-major in
// GGUF order ([cols, rows]): `cols` is the contiguous (inner) dim, rows = y.size().
// `x` has length `cols`. Weights are uploaded on first use and cached by
// (pointer, byte size). The bf16 variant takes W as raw uint16_t and requires x
// to already be bf16-rounded (matching the CPU's per-product rounding).
bool matvec_f32(std::span<const float> W, uint64_t cols, std::span<const float> x,
                std::span<float> y);
bool matvec_bf16(std::span<const uint16_t> W, uint64_t cols,
                 std::span<const float> x, std::span<float> y);

// Prefill flash attention (mirrors nn::scaled_dot_product_attention for n_q > 1):
// tiled online-softmax over the KV sequence so the [n_q, n_kv] scores matrix is
// never materialized. q is [n_q, n_heads_q, head_dim]; k/v are [n_kv, n_heads_kv,
// head_dim] (fp16-rounded f32); out is [n_q, n_heads_q, head_dim]. The causal +
// sliding-window mask is computed in-kernel from (iq, j) — Gemma 4 has no
// data-dependent mask. scale == 1.0 for Gemma 4, no logit softcap. Requires
// n_heads_q a multiple of n_heads_kv, head_dim % 32 == 0 and <= 512, n_heads_q
// <= 32. fp32 accumulation, ~1e-4 relative error vs the CPU reference (same
// relaxation as the other kernels here).
bool flash_attention(std::span<const float> q, uint64_t n_q,
                     std::span<const float> k, std::span<const float> v,
                     uint64_t n_kv, uint64_t n_heads_q, uint64_t n_heads_kv,
                     uint64_t head_dim, uint64_t sliding_window, float scale,
                     std::span<float> out);

// Drop all cached device weight copies (call when a model is reloaded at a new
// address so stale device buffers are not reused).
void clear_cache();

} // namespace sonicboom::nn::cuda
