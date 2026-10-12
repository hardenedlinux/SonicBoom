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

#include <sonicboom/quant/quantized_tensor.h>

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

// gelu×up -> q8_1 fused (T5): yq = quantize_q8_1(gelu_fp16(gate) * up). One
// kernel per 32-element block computes the fp16-gelu gate, multiplies by up, and
// amax/127-quantizes — the f32 gelu and product intermediates are never
// materialized. Device in/out, no host copies. n must be a multiple of 32.
bool gelu_mul_quantize_q8_1_dev(const float* gate, const float* up, BlockQ8_1* yq,
                                int n);

// The K-quant gemv consuming a pre-quantized q8_1 activation (`xq`), no internal
// quantize. W's packed bytes are uploaded+cached exactly as matvec_f32_dev.
// blocks_per_row / num_rows are derived from W, so the caller passes only the
// weight and the q8_1 pointer.
bool gemv_q8_1_dev(const QuantizedTensor& W, const BlockQ8_1* xq, float* y);

// Y = W @ X (batched, the prefill shape), device in/out. X is device
// [cols, n] row-major (n innermost), Y is device [rows, n] (n innermost).
// n == 1 falls back to matvec_f32_dev. No host copies.
bool matmul_f32_dev(const QuantizedTensor& W, const float* X, float* Y, int n);

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
