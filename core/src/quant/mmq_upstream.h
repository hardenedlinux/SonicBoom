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
//
// The implementation adapts the upstream ggml MMQ kernels (MIT); those
// imported files carry their own MIT copyright and permission notice, and the
// attribution is recorded in NOTICE.

#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// SonicBoom-owned glue around the imported upstream MMQ kernels.  No ggml
// type or API appears here: the entry points take raw pointers, integer
// dimensions and a CUDA stream only.
//
// This is a new, currently-unused path (M1).  It does not replace or route any
// existing production matmul; cuda_quant_matmul.cu's dispatch is untouched.

namespace sonicboom::quant::cuda {

// The six GGUF quant formats SonicBoom uses.
enum class MmqUpType : int {
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    Q5_0,
    Q8_0,
};

// Compute dst = weights @ dequant(y_q8).
//
//   weights: raw GGUF blocks, M rows x K columns (row stride K/qk blocks).
//   y_q8:    activation blocks in the upstream block_q8_1_mmq layout,
//            [K/128][n] blocks, followed by room for J_max (<= 128) trailing
//            blocks that the tiled kernel may read past the end of a row.
//   dst:     [M][n] float32, column-major: dst[i + j*M].
//
// Returns false for an unsupported type, malformed dimensions, or when the
// selected configuration uses the stream-K decomposition (not validated in
// this step).
bool mmq_up_matmul(const void * weights, MmqUpType type, const int32_t * y_q8,
                   float * dst, int64_t M, int64_t K, int64_t n,
                   cudaStream_t stream);

// Quantize x [n][K] (row-contiguous, row stride K) into the upstream
// block_q8_1_mmq layout [K/128][n] blocks.  y_q8 must have room for
// (K/128)*n + 128 blocks.  K must be a multiple of 128.
bool mmq_up_quantize(const float * x, int32_t * y_q8, int64_t K, int64_t n,
                     MmqUpType type, cudaStream_t stream);

}  // namespace sonicboom::quant::cuda
