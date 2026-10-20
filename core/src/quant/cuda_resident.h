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
bool rms_norm_quantize_q8_1_dev(std::span<const float> weight, const float* x,
                                float eps, BlockQ8_1* yq, int n);

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
                                int n);

// silu×up -> q8_1 fused (T5, SwiGLU): yq = quantize_q8_1(silu(gate) * up). The
// Silu counterpart of gelu_mul_quantize_q8_1_dev (same fusion, same contract);
// the two differ only in the activation. Device in/out, no host copies. n must
// be a multiple of 32.
bool silu_mul_quantize_q8_1_dev(const float* gate, const float* up, BlockQ8_1* yq,
                                int n);

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
// on row i. No bias/add epilogue: the verify step adds them with elementwise
// kernels, which store the same f32 values as the fused epilogue.
bool gemv_q8_1_batched_dev(const QuantizedTensor& W, const BlockQ8_1* xq,
                           float* y, int n);

// One member of a sibling launch: a legacy (Q5_0/Q8_0) projection reading the
// shared q8_1 activation, with its optional [rows] bias and its [rows] output.
struct GemvSibling {
  const QuantizedTensor* weight = nullptr;
  std::span<const float> bias;  // empty => none
  float* y = nullptr;
};

// Two or three legacy projections over one pre-quantized q8_1 activation, in a
// single launch (the decode q/k/v trio, the gate/up pair). All members must
// share the activation's column count and be Q5_0 or Q8_0; each member's bias is
// either empty or exactly [rows]. Every output value is bit-identical to
// gemv_q8_1_dev on that member (same row decomposition, summation order and
// bias epilogue); only the launch grouping differs. Returns false — having set
// no output — when the set does not fit the merged kernel, so a caller can fall
// back to the separate launches.
bool gemv_q8_1_siblings_dev(std::span<const GemvSibling> siblings,
                            const BlockQ8_1* xq);

// Whether gemv_q8_1_dev can fold a bias for this quant type (i.e. whether the
// gemv kernel carries the epilogue). Callers that must add a bias regardless use
// this to choose between the epilogue and a separate add_bias_dev.
bool gemv_bias_epilogue_supported(QuantType qt);

// Y = W @ X (batched, the prefill shape), device in/out. X is device
// [cols, n] row-major (n innermost), Y is device [rows, n] (n innermost).
// n == 1 falls back to matvec_f32_dev. No host copies.
bool matmul_f32_dev(const QuantizedTensor& W, const float* X, float* Y, int n);

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
bool matmul_mmq_q8_dev(const QuantizedTensor& W, const BlockQ8_1Mmq* yq, float* Y,
                       int n, MmqDecomp decomp = MmqDecomp::Auto);

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
