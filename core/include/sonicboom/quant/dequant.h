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

#include <cstddef>
#include <cstdint>
#include <span>

#include <sonicboom/quant/quantized_tensor.h>

// Reference dequantization of block-quantized weights to float32.
//
// These are the "numerical oracle" kernels: scalar, portable, and independent
// of any device backend. CPU kernels, CUDA kernels and future MLIR lowering all
// reproduce this arithmetic. The reconstruction formulas and block layouts are
// confirmed against ggml's reference implementation (see core/src/quant/
// dequant.cpp and the differential tests in tests/core/test_dequant.cpp).
//
// Destination dtype is float32. BF16 is deliberately NOT a dequant target here:
// the only bf16 tensor in the target model (`per_layer_model_proj.weight`) is
// stored as plain bf16, not a K-quant, and is handled by the tensor layer.

namespace sonicboom::quant {

// Dequantize `t` to float32, writing `t.numel()` floats into `out`. Returns
// false (writing nothing) when the tensor is invalid for its type (wrong byte
// count) or `out_capacity < t.numel()`. Never reads past `t.data()`.
bool dequantize_f32(const QuantizedTensor& t, float* out, uint64_t out_capacity);

// Dequantize a single row of a 2-D quantized tensor to float32. The row spans
// the contiguous dim `dims()[0]` (a whole number of blocks), so this writes
// `dims()[0]` floats. Returns false (writing nothing) when `t` is invalid, is
// not 2-D, `row >= dims()[1]`, or `out_capacity < dims()[0]`.
//
// This is the token-embedding primitive: for `token_embd.weight` of shape
// [embd_dim, vocab], `row == token_id` yields that token's embedding.
bool dequantize_row_f32(const QuantizedTensor& t, uint64_t row, float* out,
                        uint64_t out_capacity);

// Lower-level block-stream dequantizers. Each consumes exactly
// `n_blocks * bytes_per_block()` bytes and emits `n_blocks * block_size()`
// floats (256 per block). The caller guarantees the span is large enough and
// `n_blocks` is the exact ceil(numel/block_size) — the tensor-level
// `dequantize_f32` enforces this; kernels using these directly must too.
void dequantize_q3_K(std::span<const std::byte> blocks, uint64_t n_blocks, float* out);
void dequantize_q4_K(std::span<const std::byte> blocks, uint64_t n_blocks, float* out);
void dequantize_q5_K(std::span<const std::byte> blocks, uint64_t n_blocks, float* out);

} // namespace sonicboom::quant
