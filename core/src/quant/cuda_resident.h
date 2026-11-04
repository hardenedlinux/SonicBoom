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

#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <cstddef>

// Internal device-resident K-quant matvec (Phase 6a Option A). Mirrors
// quant::cuda::matvec_f32 but takes a device input and writes a device output
// with no H2D/D2H, so the caller can keep the activation resident on the
// device. W's packed bytes are uploaded once and cached by (data pointer, byte
// size) via the same cache the public path uses.
//
// NOT part of the public Layer 2 surface (lives under core/src/, may use raw
// device pointers). Included only from .cu translation units.

namespace sonicboom::quant::cuda {

// y = W @ x (single-column), device in/out. W is a 2-D QuantizedTensor
// ([cols, rows], cols contiguous). x device (cols floats), y device (rows
// floats, zero-initialized before accumulation). No host copies.
bool matvec_f32_dev(const QuantizedTensor& W, const float* x, float* y);

// Multi-row matvec (the verify step's positions): x is [n, cols], y is
// [n, rows]. Quantizes every row with one launch and runs the row-invariant
// batched gemv, so row i is bit-identical to matvec_f32_dev on that row.
// Declines (false) when the q8_1 path is disabled.
bool matvec_f32_rows_dev(const QuantizedTensor& W, const float* x, float* y,
                         int n);

// One q8_1 activation block: an fp32 scale `d` and 32 int8 quants (amax/127
// quantization, == llama.cpp's quantize_q8_1). Layout-identical to the .cu
// kernel's block struct; a pure POD so it needs no CUDA includes.
struct BlockQ8_1 {
  float d;
  int8_t qs[32];
};

// Quantize-once (T1): x -> q8_1 into `yq` (nblocks blocks, nblocks == n / 32).
// Separated from the gemv so the planner can hoist the activation quantization
// into an explicit Quantize node and CSE identical quantizations (h feeding
// q/k/v, f feeding up/gate). Device in/out, no host copies.
bool quantize_q8_1_dev(const float* x, BlockQ8_1* yq, int nblocks);

// rms_norm + q8_1 quantize fused (T3): yq = quantize_q8_1(rms_norm(x, weight)).
// One kernel computes the rms_norm sum-of-squares reduction (double, matching
// nn::cuda::rms_norm_dev), derives inv = 1/sqrtf(float(sum/n) + eps), then
// applies the normalization and quantizes the result to q8_1 in the same pass —
// the f32 normalized activation is never materialized. `weight` is a host span
// of n floats (empty => unit). Bit-identical to rms_norm_dev followed by
// quantize_q8_1_dev. Device in/out, no host copies. n must be a multiple of 32.
// `rows` > 1 is the verify step's multi-position form: x is [rows, n] and yq
// [rows, n/32]; every row runs the single-row block unchanged.
bool rms_norm_quantize_q8_1_dev(std::span<const float> weight, const float* x,
                                float eps, BlockQ8_1* yq, int n, int rows = 1);

// Upload a host byte range to the device once and cache it by (pointer, size).
// This is the cache the *_dev weight spans above use; it is exposed so the glue
// JIT can hand a generated kernel the same resident norm-weight pointer, rather
// than keeping a second copy. Returns nullptr on failure.
const void* upload_bytes_cached(const void* host, std::size_t bytes);

// gelu×up -> q8_1 fused (T5): yq = quantize_q8_1(gelu_fp16(gate) * up). One
// kernel per 32-element block computes the fp16-gelu gate, multiplies by up, and
// amax/127-quantizes — the f32 gelu and product intermediates are never
// materialized. Device in/out, no host copies. n must be a multiple of 32.
bool gelu_mul_quantize_q8_1_dev(const float* gate, const float* up, BlockQ8_1* yq,
                                int n, int rows = 1);

// silu×up -> q8_1 fused (T5, SwiGLU): yq = quantize_q8_1(silu(gate) * up). The
// Silu counterpart of gelu_mul_quantize_q8_1_dev (same fusion, same contract);
// the two differ only in the activation. Device in/out, no host copies. n must
// be a multiple of 32.
bool silu_mul_quantize_q8_1_dev(const float* gate, const float* up, BlockQ8_1* yq,
                                int n, int rows = 1);

// The K-quant gemv consuming a pre-quantized q8_1 activation (`xq`), no internal
// quantize. W's packed bytes are uploaded+cached exactly as matvec_f32_dev.
// blocks_per_row / num_rows are derived from W, so the caller passes only the
// weight and the q8_1 pointer.
//
// `bias` is an optional [num_rows] epilogue operand (a host weight span, empty =>
// none) — the same [width] projection bias add_bias_dev would add over the gemv
// output, added here to the row's reduced sum instead so the projection and its
// bias cost one launch. Only valid for the types gemv_bias_epilogue_supported()
// accepts; asking for a bias on any other type fails the call (it never silently
// drops the bias).
//
// `add` is an optional [num_rows] device epilogue operand (nullptr => none) — the
// residual of an `Add(proj, residual)`, added after the bias in the same fp32
// epilogue so the residual costs no separate launch. Same type restriction as
// `bias`; a non-null `add` on an unsupported type fails the call.
bool gemv_q8_1_dev(const QuantizedTensor& W, const BlockQ8_1* xq, float* y,
                   std::span<const float> bias = {}, const float* add = nullptr);

// Row-invariant verify shape (B3): n positions of pre-quantized q8_1
// activations against one weight. `xq` is [n, cols/32] blocks contiguous and
// `y` is [n, rows]. The batched gemv's per-position arithmetic is the same lane
// decomposition and summation order as gemv_q8_1_dev (single vs batched mmvq,
// B2 audit), so row i of the result is bit-identical to the single-column call
// on row i. `bias`/`add` are the optional epilogue operands of the fused gemv
// epilogue (applied in that order to each row's reduced sum, exactly as
// add_bias_dev then add_dev would): a type gemv_bias_epilogue_supported()
// rejects with a non-empty bias (or a non-null add) fails the call.
bool gemv_q8_1_batched_dev(const QuantizedTensor& W, const BlockQ8_1* xq,
                           float* y, int n, std::span<const float> bias = {},
                           const float* add = nullptr);

// The warps-per-row split gemv_q8_1_batched_dev uses for this projection and
// row count: the warm-up's measured pick for (type, rows, cols, n), or 1 (one
// warp per row, the reference kernel) when nothing has been measured. Every
// split is bit-identical, so this only ever selects a speed.
int verify_gemv_wp_for(QuantType qt, int rows, int cols, int n);

// Measure and cache that split for the model's quantized projections (distinct
// (type, rows, cols) shapes) at row counts 3..16, once per process. Synthetic
// zeroed weights and activations, sized to the real shapes; the pick is stored
// in the tune file keyed by the build fingerprint `build_id`, and a bounded
// time budget (~1 s) either measures every row count or falls back to
// {3, 8, 16} and fills the rest from those. A no-op when the split is pinned by
// SONICBOOM_VERIFY_GEMV_WP, when q8_1 is unavailable, or after the first call.
void warm_verify_gemv_wp(const QuantizedTensor* const* weights, size_t count,
                         const char* build_id);

// The decode/verify kernel family for one quantized projection shape: the
// single-column gemv (32-element q8_1 activation per position) or the mmq int8
// tensor-core matmul (128-element block_q8_1_mmq activation). True when the
// shape takes mmq; false (the gemv) when it has not been measured or the mmq
// cannot serve it. SONICBOOM_DECODE_FAMILY=gemv|mmq overrides the pick.
bool decode_family_mmq(QuantType qt, int rows, int cols);

// Measure (or load from the tune file) that family for the model's distinct
// quantized projection shapes, largest first, under a warm-up load budget.
// Synthetic zeroed weights and activations at the real shapes; the picks are
// stored keyed by the build fingerprint `build_id`. A no-op when q8_1 or mmq is
// unavailable, when the debug pin is set, or after the first call.
void warm_decode_family(const QuantizedTensor* const* weights, size_t count,
                        const char* build_id);

// One member of a sibling launch: a quantized projection reading the shared
// q8_1 activation, with its optional [rows] bias and its [rows] output.
struct GemvSibling {
  const QuantizedTensor* weight = nullptr;
  std::span<const float> bias;  // empty => none
  float* y = nullptr;
};

// Two or three projections over one pre-quantized q8_1 activation, in a single
// launch (the decode q/k/v trio, the gate/up pair). Members may be Q5_0, Q8_0
// or any supported K-quant, mixed within the group (Gemma's q3_K/q3_K/q5_K
// attention); all must share the activation's column count, and each member's
// bias is either empty or exactly [rows]. Every output value is bit-identical
// to gemv_q8_1_dev on that member (same row decomposition, summation order and
// bias epilogue); only the launch grouping differs. Returns false — having set
// no output — when the set does not fit the merged kernel, so a caller can fall
// back to the separate launches.
bool gemv_q8_1_siblings_dev(std::span<const GemvSibling> siblings,
                            const BlockQ8_1* xq);

// Multi-position form of gemv_q8_1_siblings_dev (the verify step): `xq` is
// [n, cols/32] q8_1 blocks and every member's `y` is [n, rows]. Row i of
// position p's output is bit-identical to gemv_q8_1_siblings_dev on position p
// (same per-row decomposition and bias epilogue); only the position loop is
// folded into the launch.
bool gemv_q8_1_siblings_batched_dev(std::span<const GemvSibling> siblings,
                                    const BlockQ8_1* xq, int n);

// Whether gemv_q8_1_dev can fold a bias for this quant type (i.e. whether the
// gemv kernel carries the epilogue). Callers that must add a bias regardless use
// this to choose between the epilogue and a separate add_bias_dev.
bool gemv_bias_epilogue_supported(QuantType qt);

// Y = W @ X (batched, the prefill shape), device in/out. X is device
// [cols, n] row-major (n innermost), Y is device [rows, n] (n innermost).
// n == 1 falls back to matvec_f32_dev. No host copies.
bool matmul_f32_dev(const QuantizedTensor& W, const float* X, float* Y, int n);

// The mmq matmul over an f32 activation: quantizes X to block_q8_1_mmq, then
// runs the mmq kernel with the shape's pinned tile config and split-K. Used at
// every n, including n == 1 (the decode step and the verify rows share the
// batched arithmetic). X is device [cols, n], Y device [n, rows]; no host
// copies.
//
// `bias`/`add` are the optional write-back epilogue: two f32 additions applied
// to each output value in that order (bias first, then the residual), at the
// same rounding as add_bias_dev followed by add_dev, so the fused node stores
// the values the separate kernels would. `bias` is a host span of `rows`
// floats (empty => none); `add` is a device [n, rows] buffer (nullptr => none,
// residual indexed by row).
bool matmul_mmq_dev(const QuantizedTensor& W, const float* X, float* Y, int n,
                    std::span<const float> bias = {}, const float* add = nullptr);

// --- MMQ int8-tensor-core batched/fused prefill ops (C.2/C.4) ---------------

// block_q8_1_mmq activation block — the mmq int8-tensor-core matmul's input. A
// 128-element block: four 32-element scales (float d4 for Q3_K weights / half2
// ds4 for Q4_K,Q5_K, both 16 bytes) + 128 int8 quants; 144 bytes total. POD
// mirror of ggml_derived/mmq.cuh's block_q8_1_mmq so host-side callers can size
// the buffer without including the .cuh.
struct BlockQ8_1Mmq {
  float d4[4];     // scale union (ds4 occupies the same 16 bytes for Q4/Q5)
  int8_t qs[128];
};
static_assert(sizeof(BlockQ8_1Mmq) == 144, "BlockQ8_1Mmq must be 144 bytes");

// Activation scale layout for the mmq quantize: D4 = float d4 (Q3_K weights),
// DS4 = half2 ds4 (Q4_K/Q5_K weights), matching the consuming matmul's weight
// type. Passed as an int so the header stays CUDA-free.
inline constexpr int MmqDsD4 = 0;
inline constexpr int MmqDsDS4 = 1;

// rms_norm + block_q8_1_mmq quantize fused (C.4, position-major): yq =
// q8_1_mmq(rms_norm(X)). X is [n, dim] position-major (dim innermost); yq is
// [dim/128][n] block_q8_1_mmq (index blk*n + token). `weight` is the per-element
// norm weight (host span of dim floats, empty => unit). Device in/out, no host
// copies. dim must be a multiple of 128.
bool rms_norm_quantize_mmq_q8_1_dev(std::span<const float> weight, const float* x,
                                    float eps, BlockQ8_1Mmq* yq, int dim, int n,
                                    int ds);

// gelu_fp16(gate)*up + block_q8_1_mmq quantize fused (C.4, position-major).
// gate/up are [n, dim] position-major; yq is [dim/128][n] block_q8_1_mmq. The
// f32 gelu/product intermediates are never materialized. Device in/out. dim must
// be a multiple of 128.
bool gelu_mul_quantize_mmq_q8_1_dev(const float* gate, const float* up,
                                    BlockQ8_1Mmq* yq, int dim, int n, int ds);

// silu(gate)*up + block_q8_1_mmq quantize fused (C.4, position-major, SwiGLU).
// The Silu counterpart of the gelu fusion above: identical contract, identical
// block mapping and quantizer, only the activation differs. dim must be a
// multiple of 128.
bool silu_mul_quantize_mmq_q8_1_dev(const float* gate, const float* up,
                                    BlockQ8_1Mmq* yq, int dim, int n, int ds);

// K-quant matmul consuming a pre-quantized block_q8_1_mmq activation (no internal
// quantize). W is the Q*_K weight [cols, rows]; yq is [cols/128][n] block_q8_1_mmq
// (index blk*n + token); Y is [n, rows] position-major. The caller sized yq with
// the J_max tail guard (see mmq_q8_blocks). Device in/out, no host copies.
// `decomp` is the plan's decomposition for this shape class (the node's `decomp`
// attribute); Auto lets the runtime choose (class policy / autotuner).
//
// `bias`/`add` are matmul_mmq_dev's optional write-back epilogue (bias first,
// then the residual, one f32 add each): `bias` a host span of `rows` floats,
// `add` a device [n, rows] buffer, row-indexed like the separate
// add_bias_dev/add_dev they replace.
bool matmul_mmq_q8_dev(const QuantizedTensor& W, const BlockQ8_1Mmq* yq, float* Y,
                       int n, MmqDecomp decomp = MmqDecomp::Auto,
                       std::span<const float> bias = {},
                       const float* add = nullptr);

// MMQ1: Y = W @ X on the compiler-generated int8 tensor-core kernel
// (compiler/MmqTcCodegen.h), over a per-weight q8_1 copy instead of the packed
// K-quant weight. On first use W is dequantized (quant::dequantize_row_f32,
// row by row), re-quantized to int8 with one f32 scale per 32 values, and the
// copy is uploaded and cached by (data pointer, byte size) — the same key
// discipline as upload_weight, in a cache of its own. `yq` and Y are the
// matmul_mmq_q8_dev operands ([cols/128][n] block_q8_1_mmq, [n, rows]); `ds`
// names the activation scale layout (MmqDsD4/MmqDsDS4). Returns false — having
// launched nothing — when the shape is outside the generator (cols not a
// multiple of 128, empty rows) or the copy/launch fails, so the caller can run
// the ggml mmq kernel for that node.
bool matmul_mmq_tc_dev(const QuantizedTensor& W, const BlockQ8_1Mmq* yq, float* Y,
                       int n, int ds);

// Dequantize a single row (token `token`) of a 2-D quantized embedding matrix
// to device f32: out[k] = W[k][token] for k in [0, width). W has GGUF shape
// [width, vocab]; `token` selects the row. `out` must hold `width` floats (the
// caller sizes it from W.dims()[0]). Only the token's packed row (blocks_per_row
// blocks) is uploaded to a scratch buffer each call — the full tensor can be
// ~1.5 GiB (per_layer_token_embd) and is not kept resident. Mirrors
// nn::embedding_f32 / quant::dequantize_row_f32.
bool gather_f32_dev(const QuantizedTensor& W, int64_t token, float* out);

// Like gather_f32_dev but reads the token's packed row directly from the cached
// resident device weight (upload_weight) instead of re-uploading the row. Only
// for weights whose full tensor is already resident — token_embd (the tied
// lm_head), which the per-token logits matvec keeps on the device.
bool gather_f32_resident_dev(const QuantizedTensor& W, int64_t token, float* out);

// Device-row variants (Phase 3.2 B4): read the token id from a device buffer
// (`token_dev`, one float index) at kernel time instead of a host value, so the
// next token id stays on the device between decode steps (argmax -> token_dev ->
// gather, no host round-trip).
bool gather_f32_resident_dev_row(const QuantizedTensor& W, const float* token_dev,
                                 float* out);

// Zero-copy single-row gather: pins+maps W in place (cudaHostRegisterMapped)
// and reads the token's row straight out of host memory via the mapped device
// pointer — no per-token copy. For weights too big to keep resident (the
// ~1.5 GiB per_layer_token_embd).
bool gather_f32_mapped_dev(const QuantizedTensor& W, int64_t token, float* out);
bool gather_f32_mapped_dev_row(const QuantizedTensor& W, const float* token_dev,
                               float* out);

// Batched (prefill) embedding gather: out[k*n + c] = W[k][tokens[c]] for
// k in [0, width), c in [0, n). `out` is [width, n] column-major (n innermost).
// The resident variant reads rows straight from the cached device weight (for
// token_embd, already resident via the logits matvec); the mapped variant
// pin+maps W in place (for the ~1.5 GiB per_layer_token_embd). Both upload the
// n token ids once and gather every column in one launch; the single-row
// gather_*_dev / _row entries above are the n == 1 case.
bool gather_f32_resident_dev_batched(const QuantizedTensor& W,
                                     std::span<const uint64_t> tokens, float* out);
bool gather_f32_mapped_dev_batched(const QuantizedTensor& W,
                                   std::span<const uint64_t> tokens, float* out);

// Device-token variants (CUDA Graphs for chunked prefill): the n token ids are
// read from a persistent DEVICE buffer (`token_ids_dev`, [n] uint64) at kernel
// time instead of being uploaded from a host span via a synchronous cudaMemcpy
// (which cannot be recorded into a graph). out is [width, n] column-major.
bool gather_f32_resident_dev_batched_ids(const QuantizedTensor& W,
                                         const uint64_t* token_ids_dev, int n,
                                         float* out);
bool gather_f32_mapped_dev_batched_ids(const QuantizedTensor& W,
                                       const uint64_t* token_ids_dev, int n,
                                       float* out);

} // namespace sonicboom::quant::cuda
