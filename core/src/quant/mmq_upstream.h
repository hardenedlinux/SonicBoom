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

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

// SonicBoom-owned glue around the imported upstream MMQ kernels.  No ggml
// type or API appears here: the entry points take raw pointers, integer
// dimensions and a CUDA stream only.
//
// This is the batch (n >= 2) int8 tensor-core matmul for every quant format
// cuda_quant_matmul.cu routes through MMQ, and the quantizer that produces its
// block_q8_1_mmq activation.

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
// selected configuration uses the stream-K decomposition (the imported
// configuration table disables it, so this is a guard only).
// cfg selects a small-n tile configuration (0 is the imported table; see
// mmq_up_cfg_count/mmq_up_cfg_allowed).  It only takes effect for tile widths
// 8 and 16.
bool mmq_up_matmul(const void * weights, MmqUpType type, const int32_t * y_q8,
                   float * dst, int64_t M, int64_t K, int64_t n,
                   cudaStream_t stream, int cfg = 0);

// Quantize x [n][K] (row-contiguous, row stride K) into the upstream
// block_q8_1_mmq layout [K/128][n] blocks.  y_q8 must have room for
// (K/128)*n + 128 blocks.  K must be a multiple of 128.  The bytes written are
// the upstream quantize_mmq_q8_1 output for the same input.
bool mmq_up_quantize(const float * x, int32_t * y_q8, int64_t K, int64_t n,
                     MmqUpType type, cudaStream_t stream);

// Test-only helpers.  These query the imported configuration table; they do
// not launch anything.
//
// True when the imported kernel has a configuration for this tile width that
// fits the current device's shared-memory limit (the same test the launcher
// applies before choosing J).
bool mmq_up_j_allowed(MmqUpType type, int64_t M, int J);

// The tile width the launcher would choose for an n-column call, or 0 when no
// configuration fits.
int mmq_up_selected_j(MmqUpType type, int64_t M, int64_t n);

// The number of selectable tile configurations, including the table (index 0).
int mmq_up_cfg_count();

// True when the configuration reproduces the imported table's output bit for
// bit for this type at every tested shape and n <= 16.  A configuration that
// does not is never selected.
bool mmq_up_cfg_neutral(MmqUpType type, int cfg);

// True when the kernel has a configuration for this tile width, the selected
// configuration is neutral, and it fits the device shared-memory limit.
bool mmq_up_cfg_allowed(MmqUpType type, int64_t M, int J, int cfg);

// The (I, nthreads) pair a configuration selects.
void mmq_up_cfg_dims(MmqUpType type, int64_t M, int J, int cfg,
                     int * I, int * nthreads);

}  // namespace sonicboom::quant::cuda
