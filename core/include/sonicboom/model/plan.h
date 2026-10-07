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

// Gemma 4 plan emitter (Phase 6). Expands the single-token decode forward pass
// into a static, acyclic planner::Graph of transformer OpKind nodes, so model
// inference runs on the shared spine (planner + SonicBackend) rather than the
// bespoke loop in exec.cpp. Weights are NOT graph tensors: the emitter bakes a
// (layer, slot) attribute pair onto each node and the SonicBackend reads the
// bound weights straight from the model. This keeps the ~500 MB of quantized
// weights out of the planner's tensor/buffer model (which is sized for host
// float32 activations).

#include <cstdint>
#include <span>

#include <sonicboom/model/loader.h>
#include <sonicboom/planner/graph.h>

namespace sonicboom::model {

// Names the specific Gemma 4 weight tensor (or None) a decode-graph node reads.
// The emitter bakes this as an Int attribute; the SonicBackend maps (slot,
// layer) back to the bound weight. Block slots (AttnNorm .. LayerOutputScale)
// index `weights.blocks[layer]`; the rest index `weights.<field>` directly.
enum class WeightSlot : uint8_t {
  None = 0,
  TokenEmbd,
  PerLayerTokenEmbd,
  PerLayerModelProj,
  PerLayerProjNorm,
  OutputNorm,
  RopeFreqs,
  AttnNorm,
  FfnNorm,
  AttnQNorm,
  AttnKNorm,
  PostAttentionNorm,
  PostFfwNorm,
  PostNorm,
  AttnQ,
  AttnK,
  AttnV,
  AttnOutput,
  FfnGate,
  FfnUp,
  FfnDown,
  InpGate,
  Proj,
  LayerOutputScale,
};

// Expand the 42-block Gemma 4 single-token decode (mirrors forward()/lm_head in
// exec.cpp, whose arithmetic is bit-exact against the llama.cpp baseline) into a
// static, acyclic planner::Graph. Every node is pinned to BackendTag::Sonic
// (including the shared `Add` residual nodes, which route_op would otherwise
// send to Mlir). Graph inputs are `token_id` and `pos` (Int64); the graph output
// is the softcapped logits (vocab float32). Argmax is left to the caller's
// generate loop, not encoded as a graph node.
planner::Graph build_gemma4_graph(const Gemma4Model& m);

// Expand the whole-prompt prefill (mirrors build_gemma4_graph per block) into a
// static, acyclic planner::Graph over [dim, n] / [n, dim] batched tensors, where
// n is the prompt length. Every node is pinned to BackendTag::Sonic; the batched
// OpKinds (EmbeddingBatched .. LayerCombineBatched) are dispatched by the
// SonicBackend looping the same per-token kernels the decode graph uses, so the
// prefill is bit-identical to n single-token decode steps. Graph inputs are
// `tokens` (Int64 [n]) and `start_pos` (Int64); the graph output is the final
// residual [dim, n] (the caller extracts column n-1 and runs lm_head).
planner::Graph build_gemma4_prefill_graph(const Gemma4Model& m, uint64_t n);

// Autoregressive greedy decode through the spine: build the graph once, plan it
// once (StaticPlanner + MemoryPlanner + TransferScheduler + PlanValidator), then
// run each step through the RuntimeExecutor with a SonicBackend (CpuQ8K, the
// bit-exact oracle backend). The decode is multi-token: the backend holds a
// per-KV-layer cache and each step attends over the accumulated keys (the same
// KV-cache semantics as llama.cpp's incremental decode). Same token-stream
// contract as generate(): writes each step's sampled token id into `tokens` and
// returns the number of steps actually run (stops early on model-empty,
// out-of-vocab seed, or a plan/execute failure).
//
// `n_ctx` sizes the KV cache (global layers allocate `n_ctx` slots; sliding-window
// layers a fixed `sliding_window` ring). 0 defaults to `start_pos + tokens.size()`
// (the exact decode length). The cache is cleared at session start.
uint64_t generate_spine(const Gemma4Model& m, uint64_t start_token,
                        uint64_t start_pos, std::span<int64_t> tokens,
                        uint64_t n_ctx = 0);

// CUDA spine: the same greedy decode as generate_spine, but planned against a
// CUDA-aware snapshot so Sonic nodes with *_dev kernels run on the GPU and only
// Embedding/Softcap stay on the CPU (mirrors cuda_forward_resident's placement).
// The RuntimeExecutor runs both the SonicCudaBackend (device-resident) and the
// CPU SonicBackend (host boundary) with explicit H2D/D2H transfer tasks between
// them. Returns the number of steps actually run; 0 when the build has no CUDA
// support (SONICBOOM_USE_CUDA off), so callers gate on quant::cuda_available().
uint64_t generate_spine_cuda(const Gemma4Model& m, uint64_t start_token,
                             uint64_t start_pos, std::span<int64_t> tokens,
                             uint64_t n_ctx = 0);

// Prefill + decode through the CPU spine. Processes the whole `prompt` in one
// batched graph pass (build_gemma4_prefill_graph → plan_execution →
// RuntimeExecutor with the batched OpKinds), samples `tokens[0]` from the final
// hidden column's lm-head logits, then decodes the remaining tokens with the
// same planner/backend/cache as generate_spine. `use_prefill` is a dev-test
// knob: false feeds the prompt one token at a time through the per-token spine
// (the pre-Phase-7 reference), so tests can assert the two paths produce the
// identical token stream.
//
// `prompt` must be non-empty. `n_ctx` sizes the KV cache (0 defaults to
// `start_pos + prompt.size() + tokens.size()`); the prompt must fit each KV
// layer without ring wrap (n <= sliding_window for SWA layers, start_pos + n <=
// n_ctx for global layers). Returns the number of tokens written.
uint64_t generate_prefill_spine(const Gemma4Model& m,
                                std::span<const uint64_t> prompt,
                                uint64_t start_pos, std::span<int64_t> tokens,
                                uint64_t n_ctx = 0, bool use_prefill = true);

// CUDA prefill + decode: the same prompt-prefill contract as
// generate_prefill_spine, but on the device — SonicCudaBackend::prefill (batched
// f32-activation matmul + flash attention) fills the device KV caches, lm_head
// samples tokens[0] on the device, and the remaining tokens decode through the
// CUDA spine (generate_spine_cuda's per-token loop over the same backend/cache).
// `use_prefill` is a dev-test knob: false feeds the prompt one token at a time
// through the CUDA spine (the per-token reference), so tests can assert the two
// paths produce the identical token stream. Returns 0 when the build has no CUDA
// support (SONICBOOM_USE_CUDA off), so callers gate on quant::cuda_available().
uint64_t generate_prefill_spine_cuda(const Gemma4Model& m,
                                     std::span<const uint64_t> prompt,
                                     uint64_t start_pos, std::span<int64_t> tokens,
                                     uint64_t n_ctx = 0, bool use_prefill = true);

} // namespace sonicboom::model
