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

#include <sonicboom/nn/sampling.h>

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

// y[i] = (x[i] / sqrt(mean_j(x[j]^2) + eps)) * weight[i] [+ add[i]]; weight
// empty => 1.0. x, y are device buffers of n floats; weight is a host span of
// n floats (or empty). `add` is the optional residual epilogue of
// fuse_residual_epilogue_pass (null => none), applied in the same launch.
// Mirrors nn::rms_norm / nn::cuda::rms_norm plus the following nn::add.
bool rms_norm_dev(std::span<const float> weight, const float* x, float eps,
                  float* y, int n, const float* add = nullptr);

// Per-head RMSNorm over `heads` heads of `head_dim` floats each:
//   y[h*head_dim + i] = (x[h*head_dim + i] / sqrt(mean_i(x[h*head_dim+i]^2) + eps)) * weight[i]
// [+ add[h*head_dim + i]]
// `weight` is a single head_dim-length vector shared across heads (empty =>
// unit). Mirrors exec.cpp's per-head nn::rms_norm loop.
bool rms_norm_heads_dev(std::span<const float> weight, const float* x, float eps,
                        float* y, int head_dim, int heads,
                        const float* add = nullptr);

// GGML_GELU_FP16 tanh-GELU (fp16 round-trip), elementwise. x/y may alias.
bool gelu_fp16_dev(const float* x, float* y, int n);

// SiLU (SwiGLU) elementwise: y[i] = x[i] / (1 + exp(-x[i])). x/y may alias.
bool silu_dev(const float* x, float* y, int n);

bool mul_dev(const float* a, const float* b, float* y, int n);
bool add_dev(const float* a, const float* b, float* y, int n);
// y[i] = x[i] + bias[i % bias.size()]: a [width] bias broadcast over the rows of
// a [rows, width] buffer (mirrors nn::add_bias). The bias is a host span, a
// bound model weight, uploaded once and cached.
bool add_bias_dev(std::span<const float> bias, const float* x, float* y, int n);
bool scale_dev(const float* x, float s, float* y, int n);

// y[i] = cap * tanh(x[i] / cap), the final-logit softcap (mirrors nn::softcap's
// expression cap * tanh(x * (1 / cap))). Keeps the sampled decode graph's
// logits on the device ahead of sample_select_dev; x and y may alias.
bool softcap_dev(const float* x, float cap, float* y, int n);

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

// Device-position variant of rope_neox_heads_dev: `pos` is a DEVICE uint64_t
// read in-kernel, so the whole decode step is capturable into a CUDA Graph (the
// position is an input to the graph, not a value baked into a node at capture).
bool rope_neox_heads_dev_pos(float* x, int head_dim, int heads,
                             const uint64_t* pos, float base, float freq_scale,
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
//
// CUDA-Graph-capturable form: `pos` is a DEVICE uint64_t and `window` is the
// fixed max key count the launch config is sized for (sliding_window for SWA
// layers, n_ctx for global layers). The split kernel computes the actual
// start/n_keys from *pos at runtime and early-exits chunks beyond n_keys, so a
// single fixed grid/smem/chunk replays correctly for every position 0..window-1
// (see decode_attention_dev in cuda_elementwise.cu). Requires window >= n_keys
// at every position, i.e. window == sliding_window (SWA) or window == n_ctx
// (global, since n_keys == pos+1 <= n_ctx).
// `pos_offset` selects a row of a verify step (B3): the position read is
// pos[pos_offset], so one call per query row executes exactly the n == 1 call
// of that row with the same split/chunk constants. 0 is the decode call.
// `kv_f16` selects the standard profile's f16 KV storage: the cache buffers
// hold __half instead of float. The values in them were fp16-rounded before
// either storage existed, so reading f16 and widening is numerically exact.
// `kv_q8` selects q8 KV storage instead: each cache row is head_dim int8
// values followed by that row's f32 scale (max|x| / 127) at byte offset
// head_dim, so a row is head_dim + 4 bytes. Loads dequantize (int8 * scale).
// `chunk_attr` is the plan's split-KV chunk size (the Boom IR `decode_chunk`
// attribute). 0 means "use the shared class default". Chunk boundaries are
// absolute (key position / chunk) and partials merge in ascending chunk order,
// so the reduction tree depends only on key positions, never on the window
// capacity or the current depth.
// `tier` is the capture-time decode depth tier (-1 = off, the default): when
// multi-tier decode graphs are active the host captures one graph per tier and
// passes that tier here, so the launch geometry (split-KV chunk, query heads per
// block, grid window) is baked into the graph for that tier. The tier's grid
// window is the tier capacity, not the full cache capacity, so a captured graph
// covers exactly the key counts it is selected for; replay selection stays a
// host-side choice of graph, never a kernel-node parameter update.
bool decode_attention_dev(const float* q, const float* k_cache,
                          const float* v_cache, const uint64_t* pos,
                          uint64_t window, uint64_t n_slots, int n_heads_q,
                          int n_heads_kv, int head_dim, uint64_t sliding_window,
                          float scale, float* out, int pos_offset = 0,
                          bool kv_f16 = false, uint64_t chunk_attr = 0,
                          bool kv_q8 = false, int tier = -1);

// --- multi-tier decode graphs ----------------------------------------------
//
// Depth tiers: fixed key-count boundaries, the last the runtime capacity. A
// greedy decode runtime captures one graph per tier so the per-node launch
// geometry can follow the tier (W: split-KV chunk; P3: query heads per block)
// while a replay stays one graph launch.
constexpr int kDecodeTierCount = 4;

// Capture one decode graph per depth tier. Opt-in (SONICBOOM_DECODE_MULTIGRAPH=1)
// because it changes the number of captured graphs; with it off the runtime
// captures the single graph it always has.
bool decode_multigraph_enabled();

// The tier capacities for a cache whose maximum key count is `keys_capacity`
// (n_ctx for a causal layer, the sliding window for a windowed one). Writes an
// ascending list, last == keys_capacity, at most kDecodeTierCount entries, and
// returns its length (>= 1).
int decode_tier_caps(uint64_t keys_capacity,
                     uint64_t caps[kDecodeTierCount]);

// Resolve the split-KV chunk size for one decode attention class (head_dim,
// GQA ratio, head counts, causal vs windowed, KV element type). Returns the
// class default or a per-device tuned pick; the pick is recorded in the
// decision registry and is a pure function of the class (never of n_ctx, the
// current position or the prefix). `capacity` only bounds the depths the tuner
// measures at; it is not part of the class key.
uint64_t attn_decode_plan_chunk(int head_dim, int gqa, int n_heads_q,
                                int n_heads_kv, uint64_t sliding_window,
                                uint64_t capacity, bool kv_f16,
                                bool kv_q8 = false);

// Append one token's K and V rows (row_elems floats each) into a device-resident
// KV cache at slot (pos % n_slots). src/dst are device buffers; `pos` is a DEVICE
// uint64_t, so the append is capturable into a CUDA Graph (the slot offset is
// computed in-kernel, not baked into a launch argument). dst_k/dst_v are the
// cache rows for this layer, src_k/src_v the freshly projected K/V.
// `pos_offset` is the verify step's row index (pos[pos_offset]; 0 = decode).
bool kv_append_dev(float* dst_k, float* dst_v, const float* src_k,
                   const float* src_v, const uint64_t* pos, uint64_t n_slots,
                   int row_elems, int pos_offset = 0, bool dst_f16 = false,
                   bool dst_q8 = false, int head_dim = 0);

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

// Dense full attention (non-causal, no window, no KV cache): every query
// attends to every key in [0, n). The class is keyed on those attributes
// (causal 0, kv_cache absent, prefix_reuse impossible, window 0) plus
// head_dim, the GQA ratio (1 here) and the head count, and resolved through
// the decision registry (SONICBOOM_ATTN_CLASS / its SONICBOOM_DIT_ATTN alias
// override). Two regimes: a batched cuBLAS f16 score-GEMM path while the
// per-head [n, n] planes fit the scratch budget (n <= 2048), and an online-
// softmax tiled class above it -- the generated FA-2 warp-head kernel, with
// the hand-written tiled kernel as an autotune candidate. q/k/v are
// [n, n_heads, head_dim] f32 (K/V are rounded to f16 in both regimes); out is
// [n, n_heads, head_dim] f32. Returns false when no class of this op can run
// (e.g. SONICBOOM_ATTN_CLASS=scalar, or an unsupported head width), and the
// caller keeps its own fallback.
bool flash_attention_full_dev(const float* q, const float* k, const float* v,
                              int n, int n_heads, int head_dim, float scale,
                              float* out);

// Release the score-GEMM regime's scratch planes (the largest per-call
// allocation of the full-attention class). Safe to call at any time.
void flash_attention_full_release_scratch();

// Chunked-prefill flash attention: n queries at absolute positions
// [start, start+n) each attend over [start_key, p] where start_key = max(0,
// p - window + 1). Keys in [start_key, start) are prior context read from the
// KV cache (slot j % n_slots); keys in [start, p] are the current chunk's own
// K/V read from the batch buffer (batch index j - start). One online-softmax
// pass covers both sources; the caller defers the chunk's cache append until
// after every layer (including shared donors) has read it, so the cache still
// holds the prior context here. For start == 0 the prior range is empty, so the
// kernel reads the current chunk only. q is [n, n_heads_q, head_dim];
// k_batch/v_batch are [n, n_heads_kv, head_dim] (fp16-rounded f32); k_cache/
// v_cache are [n_slots, n_heads_kv, head_dim]. out is [n, n_heads_q, head_dim].
// Device-position variant: `start` is a DEVICE uint64_t read in-kernel, so the
// chunk's absolute start position is a graph input rather than a value baked
// into the node at capture time (CUDA Graphs for chunked prefill). The launch
// config is fixed by n/head shapes, never by start. boom_fattn is the prefill
// attention implementation for every supported shape class.
bool flash_attention_chunk_dev_pos(const float* q, const float* k_batch,
                                   const float* v_batch, const float* k_cache,
                                   const float* v_cache, const uint64_t* start,
                                   uint64_t n_slots, int n, int n_heads_q,
                                   int n_heads_kv, int head_dim,
                                   uint64_t sliding_window, float scale,
                                   float* out, bool kv_f16 = false,
                                   bool kv_q8 = false);

// The prefill attention implementation (core/src/nn/boom_fattn.cu): f16 KV
// span materialization (f32/f16/q8 cache rows are dequantized into the f16
// workspace), tiled MMA with an online softmax, GQA query grouping and
// cp.async staging, with the causal/window mask derived in registers. Returns
// false without launching anything for shapes it has no instantiation for.
bool flash_attention_boom_fattn_chunk_dev_pos(
    const float* q, const float* k_batch, const float* v_batch,
    const float* k_cache, const float* v_cache, const uint64_t* start,
    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* out, bool kv_f16 = false,
    bool kv_q8 = false);

// boom_fattn's strict-order path for the d128/GQA-6 class
// (core/src/nn/boom_fattn_strict.cu). Same contract; false for any other
// shape. The exact kernel is the default and SONICBOOM_ATTN_D128_EXACT=0
// selects the scalar arm it is byte-identical to.
bool flash_attention_boom_fattn_strict_chunk_dev_pos(
    const float* q, const float* k_batch, const float* v_batch,
    const float* k_cache, const float* v_cache, const uint64_t* start,
    uint64_t n_slots, int n, int n_heads_q, int n_heads_kv, int head_dim,
    uint64_t sliding_window, float scale, float* out, bool kv_f16 = false,
    bool kv_q8 = false);

// Launch geometry the boom_fattn backend would use for this shape class
// (query tile, key tile, block threads, dynamic smem), for the attention trace
// and the planner's cost model. False when the backend has no instantiation.
bool flash_attention_boom_fattn_config(int head_dim, int gqa, int* br, int* bk,
                                       int* block, size_t* smem);

// Attention work trace (SONICBOOM_ATTN_TRACE=1). The backend sets the layer
// before the call; the attention entry points then print one line per call to
// stderr with the shape, the implementation/config chosen, the KV extent the
// kernel iterates and the number of key tiles visited versus the minimum the
// causal+window mask requires. Reading the device `start` value needs a D2H
// copy, so the trace is off by default and must never be enabled in a timing
// run. The layer is thread-local scratch: it does not affect the launch.
void attention_trace_set_layer(int layer);

// Per-layer embedding combine: y[i] = (proj[i] + ple[i] * ple_scale) * combine_scale.
// `ple` is the caller-offset device pointer into the full per-layer embedding
// row (i.e. already advanced by `layer * per_layer_input`). Mirrors
// nn::layer_combine.
bool layer_combine_dev(const float* proj, const float* ple, float ple_scale,
                       float combine_scale, float* y, int n);

// Batched (position-major) per-layer embedding combine. proj/y are [n, per];
// `ple` is the full [n, total_per] per-layer embedding row (NOT caller-offset —
// the layer offset is `col`), read with a total_per stride:
//   y[c*per+k] = (proj[c*per+k] + ple[c*total_per + col + k] * ple_scale) * combine_scale
// The decode path's contiguous combine is layer_combine_dev (the n == 1 case).
bool layer_combine_batched_dev(const float* proj, const float* ple, float ple_scale,
                               float combine_scale, float* y, int per, int n,
                               int col, int total_per);

// Strided row slice (Section C.5): y[r*length + k] = x[offset + r*stride + k]
// for r in [0, rows), k in [0, length). Extracts a layer's per-layer-embedding
// column from each position of a [n, total_per] row. One thread per output
// element.
bool slice_strided_dev(const float* x, float* y, int rows, int length, int offset,
                       int stride);

// Greedy argmax over a device f32 vector: writes the index (as a float) to
// out[0]. Ties are broken toward the lowest index, exactly mirroring
// nn::argmax. Requires n > 0 and a non-null x/out.
bool argmax_dev(const float* x, int n, float* out);

// Masked argmax: the same selection over the ids whose bit is set in `mask`
// (a device bitmap of ceil(n / 8) bytes, bit i of byte i / 8, LSB first; the
// caller owns it). Ids the mask disallows never win, and an all-disallowed row
// writes -1 like an all-NaN one. Mirrors nn::argmax_masked. Devices that
// cannot run it return false, and the caller falls back to the host.
bool argmax_masked_dev(const float* x, int n, const uint8_t* mask, float* out);

// Deterministic sampling's candidate selection on the device:
//   z[i] = (penalized logit i) / temperature        (float division)
// for the unmasked ids (mask may be null = everything allowed; a bitmap of
// ceil(vocab / 8) bytes, LSB first), then the kSampleCandidates highest z
// (ties toward the lower id) are written to out[0..*out_count) in that order
// (z descending, id ascending). *out_count = min(256, unmasked count).
//
// Penalties mirror nn::select_candidates_host exactly: `window` holds the last
// window_len history tokens (device int32, at most kSampleSelectMaxWindow
// entries; window may be null when window_len is 0); a token occurring c times
// there has its logit adjusted once, with the same single-rounding operations
// as the host: repeat penalty by division (positive) or multiplication
// (non-positive), then - (frequency * c + presence). The logits buffer is not
// modified. `out` and `out_count` are device buffers (kSampleCandidates
// entries and one int).
//
// The result is bitwise identical to nn::select_candidates_host for the same
// inputs. One block, deterministic; the device key scratch is retained and
// never freed because a captured decode graph may embed it.
inline constexpr int kSampleSelectMaxWindow = 1024;

// The per-launch parameters that a captured decode graph stages on the device
// before replay (the values that change between decode calls). Mirrors the
// scalar arguments of sample_select_dev.
struct SampleSelectParams {
  int32_t window_len;
  float temperature;
  float repeat_penalty;
  float presence_penalty;
  float frequency_penalty;
};

bool sample_select_dev(const float* logits, int vocab, const uint8_t* mask,
                       const int32_t* window, int window_len,
                       float temperature, float repeat_penalty, float presence,
                       float frequency, Candidate* out, int* out_count);

// The captured-graph form: the scalars (and window length) come from
// `params_dev` (device memory) instead of the launch arguments, so one
// recorded launch can be replayed with new sampling parameters. `window` must
// be non-null whenever params_dev->window_len > 0. Returns false if the device
// is unavailable; the caller validates the staged values.
bool sample_select_dev_dyn(const float* logits, int vocab, const uint8_t* mask,
                           const int32_t* window,
                           const SampleSelectParams* params_dev,
                           Candidate* out, int* out_count);

// --- batched prefill primitives (the decode *_dev kernels generalized to an
// n dimension; decode is n == 1). Activations are [n, dim] position-major
// (dim innermost). Each mirrors the CPU batched prefill op (SonicBackend) and,
// for n == 1, the corresponding single-token *_dev kernel above. -------------

// Multi-head NEOX RoPE over a batched [n_pos, heads_per_pos*head_dim] buffer
// (position-major, flat), with per-position absolute position
// pos = start_pos + (i / (heads_per_pos*head_dim)). Mirrors rope_neox_heads_dev
// applied per position; the decode path is n_pos == 1.
bool rope_neox_heads_batched_dev(float* x, int head_dim, int heads_per_pos,
                                 int n_pos, uint64_t start_pos, float base,
                                 float freq_scale,
                                 std::span<const float> freq_factors);

// Device-position variant of rope_neox_heads_batched_dev: `start_pos` is a DEVICE
// uint64_t read in-kernel (CUDA Graphs for chunked prefill).
bool rope_neox_heads_batched_dev_pos(float* x, int head_dim, int heads_per_pos,
                                     int n_pos, const uint64_t* start_pos,
                                     float base, float freq_scale,
                                     std::span<const float> freq_factors);

// Dense batched matvec (position-major): y[c*rows+r] = sum_i W[r*cols+i] *
// x[c*cols+i]. x is [ncols, cols], y is [ncols, rows] (dim innermost); W
// row-major. The decode matvec_f32_dev / matvec_bf16_dev are the ncols == 1
// case. Every ncols (including 1) routes to cuBLAS via fixed-width 128-position
// blocks, so a position's numerics are identical across batch sizes and offsets.
//
// strided: the cuBLAS path reads the position-major activation block and writes
// the position-major result directly (no transposing pack/copyback) for every
// full 128-position block; the tail keeps the packed path. For bf16 it also
// replaces the caller's bf16 staging copy with one contiguous rounding pass.
// Falls back to the plain path when cuBLAS is unavailable.
bool matvec_f32_batched_dev(std::span<const float> W, int cols, const float* x,
                            float* y, int rows, int ncols, bool strided = false);
// `alpha` is the folded scalar-scale epilogue (the planner folds a
// single-reader `scale` into the projection); it is applied by the generated
// GEMM's store when that kernel is selected and by a scale pass otherwise.
bool matvec_bf16_batched_dev(std::span<const uint16_t> W, int cols, const float* x,
                             float* y, int rows, int ncols, bool strided = false,
                             float alpha = 1.0f);

// Append n consecutive token rows (row_elems floats each) into the flat KV cache
// at slots (start_pos + p) % n_slots. src_k/src_v are [n, row_elems]
// position-major. Mirrors kv_append_dev (the n == 1 case); prefill guarantees a
// wrap-free run so the slots are consecutive.
bool kv_append_batched_dev(float* dst_k, float* dst_v, const float* src_k,
                           const float* src_v, uint64_t start_pos,
                           uint64_t n_slots, int row_elems, int n,
                           bool dst_f16 = false, bool dst_q8 = false,
                           int head_dim = 0);

// Device-position variant of kv_append_batched_dev: `start_pos` is a DEVICE
// uint64_t read in-kernel (CUDA Graphs for chunked prefill).
bool kv_append_batched_dev_pos(float* dst_k, float* dst_v, const float* src_k,
                               const float* src_v, const uint64_t* start_pos,
                               uint64_t n_slots, int row_elems, int n,
                               bool dst_f16 = false, bool dst_q8 = false,
                               int head_dim = 0);

} // namespace sonicboom::nn::cuda
