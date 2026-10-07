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

#include <sonicboom/model/loader.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <functional>

// Gemma 4 reference execution (Phase 5C-2): the token / per-layer embedding
// paths and a single-block forward pass, in float32. The block arithmetic
// follows the llama.cpp baseline graph recorded in design/gemma4-semantics.md §4
// (token-embedding sqrt(n_embd) scale, Q/K per-head RMSNorm, V RMSNorm without
// weight, NEOX RoPE with global-layer freq_factors, attention scale 1.0,
// GELU-tanh FFN, post-norms, per-layer gate, whole-block layer_output_scale).
//
// All functions borrow the model (whose spans point into the GGUF reader's
// buffer) and write into caller-provided output spans. None allocate beyond a
// small per-call scratch; none throw.

namespace sonicboom::model {

// Token embedding (the residual-stream input to every block), scaled by
// sqrt(embedding_length) as the reference does:
//
//   out = token_embd[token_id] * sqrt(embedding_length)
//
// out.size() >= embedding_length.
bool embed_token(const Gemma4Model& m, uint64_t token_id, std::span<float> out);

// The per-layer embedding contribution for block `layer` — the reference
// `inp_per_layer[l]` vector of length `per_layer_input` that run_block injects
// via the per-layer gate (NOT added at block input):
//
//   inpL = token_embd[token] * sqrt(embedding_length)
//   proj = per_layer_model_proj[:, layer*per .. +per) @ inpL
//          * (1/sqrt(embedding_length))
//   proj = rms_norm(proj, eps) * per_layer_proj_norm
//   ple  = per_layer_token_embd[token][layer*per .. +per) * sqrt(per_layer_input)
//   out  = (proj + ple) * (1/sqrt(2))
//
// out.size() >= per_layer_input. `use_cuda` routes the bf16 projection
// (per_layer_model_proj @ inpL) to the device when the Cuda backend is active;
// pass false for the CPU double-accumulation reference.
bool embed_per_layer(const Gemma4Model& m, uint32_t layer, uint64_t token_id,
                     std::span<float> out, bool use_cuda);

// One transformer block (single-token decode, n_k = 1):
//
//   inpL = token_embd[token] * sqrt(d)
//   h    = rms_norm(inpL, eps) * attn_norm
//   q,k,v = attn_{q,k,v} @ h
//   q,k  = per-head rms_norm * attn_{q,k}_norm;  v = rms_norm(v, eps)   (no weight)
//   q,k  = RoPE(NEOX, pos, rope_base, freq_factors?)   (factors on global only)
//   a    = softmax(q·k * 1.0, causal) @ v   (== v when n_k == 1)
//   o    = rms_norm(attn_output @ a, eps) * post_attention_norm
//   x1   = o + inpL
//   f    = rms_norm(x1, eps) * ffn_norm
//   f    = rms_norm(ffn_down @ (gelu(ffn_gate@f) * (ffn_up@f)), eps) * post_ffw_norm
//   x2   = f + x1
//   g    = rms_norm(proj @ (gelu(inp_gate@x2) * inp_per_layer[l]), eps) * post_norm
//   out  = (x2 + g) * layer_output_scale
//
// `pos` is the absolute token position for RoPE. Single-token attention is
// causal with one key/value, so the softmax is over a single logit and the
// attention output equals V (multi-token KV attention is Phase 5B/6).
//
// Returns false (writing nothing) when `layer >= block_count`, when the layer
// is a shared-KV layer (24..41 — those need the donor KV, so use forward()), or
// when out.size() < embedding_length.
// `backend` selects the quantized-matmul backend (default CpuQ8K preserves the
// bit-exact oracle reference; Cuda offloads the K-quant matmuls to the device).
bool run_block(const Gemma4Model& m, uint32_t layer, uint64_t token_id,
               uint64_t pos, std::span<float> out,
               quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// Dev-time differential-validation hook (Phase 5C-2): like run_block, but
// invokes `trace(name, values)` for every named intermediate the llama.cpp
// Gemma 4 graph marks with cb(), so the arithmetic can be compared stage by
// stage against the baseline dump. Names use llama.cpp's convention — a per-layer
// suffix for block tensors ("Qcur_normed-0") and a bare name for the token
// embedding ("inp_scaled") is emitted by embed_token, not here. `values` is a
// view into an internal buffer and is valid only for the duration of the call;
// `trace` must not retain it.
//
// This is a source-level Layer 2 helper, not part of the stable runtime
// contract. It exists so the dump tool (tools/baseline) and tests can pin the
// exact arithmetic of each stage rather than only the final block output.
using BlockTraceFn = std::function<void(const char* name, std::span<const float> values)>;

bool run_block_traced(const Gemma4Model& m, uint32_t layer, uint64_t token_id,
                      uint64_t pos, std::span<float> out, const BlockTraceFn& trace,
                      quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// Full single-token forward: run all block_count blocks in sequence, feeding
// each block's l_out into the next as the residual stream. Gemma 4's shared-KV
// pattern is honoured — layers 24..41 do not compute K/V and instead read the
// cached V of their donor layer (22 for sliding-window, 23 for global; see
// Gemma4Config::kv_donor_layer). `out` (length embedding_length) receives the
// last block's l_out.
//
// This is still single-token decode (n_k == 1 everywhere), so the shared layers'
// attention output is the GQA-broadcast donor V. Multi-token KV accumulation and
// incremental decoding are Phase 5E/6.
bool forward(const Gemma4Model& m, uint64_t token_id, uint64_t pos,
             std::span<float> out,
             quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// forward + per-stage trace: emits every named intermediate of every block
// (llama.cpp's "<name>-<layer>" convention) via the same BlockTraceFn used by
// run_block_traced. "inp_scaled" (the token embedding) is NOT emitted here —
// call embed_token yourself if the dump needs it.
//
// Phase 5E: after the block loop this also computes and traces the lm-head
// intermediates with llama.cpp's bare names (no layer suffix):
//   "h_nextn"     = rms_norm(final_hidden, output_norm)   (== "result_norm")
//   "result_norm" = the post-output-norm hidden state
//   "result_output" = softcapped logits (30*tanh(logits/30))
// The returned `out` is still the final hidden state (length embedding_length),
// not the logits — use lm_head() for the logits.
bool forward_traced(const Gemma4Model& m, uint64_t token_id, uint64_t pos,
                    std::span<float> out, const BlockTraceFn& trace,
                    quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// The lm-head (Phase 5E): given a final hidden state (e.g. forward()'s output),
// produce the softcapped logits:
//
//   h     = rms_norm(hidden, output_norm, eps)      // "result_norm"
//   logit = token_embd @ h                           // weight-tied output (q3_K)
//   out   = 30 * tanh(logit / 30)                    // final_logit_softcapping
//
// `logits.size() >= vocab_size`.
bool lm_head(const Gemma4Model& m, std::span<const float> hidden,
             std::span<float> logits,
             quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// Autoregressive greedy decode through the Phase 6 spine (the runtime path).
// Run up to `tokens.size()` steps, each = forward + lm_head + argmax, starting
// from `start_token` at absolute position `start_pos`. Writes each step's
// sampled token id into `tokens` and returns the number of steps actually run
// (stops early on an empty model, `start_token >= vocab_size`, or a plan /
// execute failure). The decode is dispatched by `backend`:
//
//   CpuQ8K -> generate_spine      (CPU spine: planner + SonicBackend, q8_K)
//   Cuda   -> generate_spine_cuda (GPU spine; falls back to the CPU spine when
//                                   no device is present)
//   CpuF32 -> generate_reference  (f32 scalar reference: no spine variant)
//
// No tokenizer is involved: `start_token` is a raw vocab id (dev-test knob).
uint64_t generate(const Gemma4Model& m, uint64_t start_token, uint64_t start_pos,
                  std::span<int64_t> tokens,
                  quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// Prompt prefill + decode (Phase 7 prefill): process the whole non-empty `prompt`
// in one batched pass (flash attention + batched matmul) then autoregressively
// decode `tokens`. `tokens[0]` is sampled from the prompt's final lm-head logits;
// the rest decode from it. Dispatched by `backend`:
//
//   Cuda   -> generate_prefill_spine_cuda (flash attention on device; falls
//             back to the CPU prefill when no device is present)
//   CpuQ8K -> generate_prefill_spine      (CPU prefill: batched q8_K + SDPA)
//   CpuF32 -> no f32 prefill variant; falls back to generate_prefill_spine
//
// Writes each sampled token id into `tokens` and returns the number written
// (stops early on an empty model, an empty prompt, or a plan/execute failure).
// No tokenizer is involved: `prompt` is raw vocab ids (dev-test knob).
uint64_t generate_prompt(const Gemma4Model& m, std::span<const uint64_t> prompt,
                         uint64_t start_pos, std::span<int64_t> tokens,
                         quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

// The bespoke forward + lm_head + argmax decode loop, retained as an ORACLE for
// differential validation against the spine (generate()). This is the pre-Phase-6
// runtime path — run_block_core on the CPU and cuda_forward_resident on the
// device underneath — demoted to reference-only. `backend` selects the
// quantized-matmul backend exactly as generate() did before the spine landed
// (CpuQ8K bit-exact oracle, f32 reference, or the CUDA device). tools/baseline and
// tests use this to pin the arithmetic the spine must reproduce.
uint64_t generate_reference(const Gemma4Model& m, uint64_t start_token,
                            uint64_t start_pos, std::span<int64_t> tokens,
                            quant::MatmulBackend backend = quant::MatmulBackend::CpuQ8K);

} // namespace sonicboom::model
