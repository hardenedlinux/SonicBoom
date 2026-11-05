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

#include <sonicboom/quant/quantized_tensor.h>

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
//
// split_k > 1 splits the K tiles into that many contiguous slices, runs the
// tile body per slice and sums the float partials in slice order.  The split
// is capped at the number of K tiles; 1 is the single-pass path.  The partial
// buffer is owned by this translation unit, reused across calls and sized for
// the largest call seen; it is not allocated on the single-pass path.
//
// bias/add are the optional write-back epilogue, applied with the output: the
// kernel writes dst = sum + bias[row], then + add[token*M + row], with one f32
// add each, in that order — the values the separate elementwise kernels would
// store, without their launches.  bias is a device [M] buffer (one value per
// output row) and add is a device [n][M] buffer in dst's layout.  They apply to
// the final output whether the shape runs one pass (added in the tile
// write-back) or split-K (added once in the partial-combine kernel).
bool mmq_up_matmul(const void * weights, MmqUpType type, const int32_t * y_q8,
                   float * dst, int64_t M, int64_t K, int64_t n,
                   cudaStream_t stream, int cfg = 0, int split_k = 1,
                   const float * bias = nullptr, const float * add = nullptr);

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

// The pinned small-n choice for one (type, rows, cols) shape: the tile
// configuration for tile width 8 (n <= 8) and 16 (n = 9..16), and the split-K
// count.  The split count comes from the warm-up's measured table (1, 2 or 4);
// the defaults 1/0/0 mean "not tuned".
struct MmqUpPin {
    int split_k = 1;
    int cfg_j8  = 0;
    int cfg_j16 = 0;
};
MmqUpPin mmq_up_pin_for(MmqUpType type, int64_t M, int64_t K);

// Measure (or load from the tune file) the pinned choice for every distinct
// quantized projection shape, largest first.  Every shape is resolved: the
// plan reads the pins before generation, so none may be left for a decode
// step to decide.  A time budget switches the shapes it has not reached to a
// reduced candidate set; each still stores a record.  Synthetic zeroed weights
// of the real shapes.  build_id keys the stored records.
void mmq_up_warm(const QuantizedTensor * const * weights, size_t count,
                 const char * build_id);

// One projection of a sibling group (see mmq_up_group_matmul).
struct MmqGroupMember {
    const void *  x;              // weight blocks, M rows x K columns
    const int32_t * y;            // the shared activation's blocks
    float *       dst;            // [M][n] float32, column-major
    const float * bias;           // optional [M] write-back epilogue
    const float * add;            // optional [n][M] residual epilogue
    int nrows;                    // M
    int stride_col_dst;           // nrows for the dense layout
    int stride_row_x;             // K / qk
    int blocks_per_row;           // K / qk
};

// A group of sibling projections launched as one kernel.  All members read the
// same activation; each member's output rows are a run of blocks in the grid's
// x dimension, in member order.  ncols_dst is n and ncols_y is the activation's
// block-row stride (also n for a decode/verify step).  tile_prefix, rows_prefix,
// total_rows and part are completed by mmq_up_group_matmul; the caller need not
// fill them.  nsplit is the group's split count (1 is the single-pass path) and
// selects the [nsplit][n][total_rows] partial buffer the launcher supplies.
struct MmqGroupArgs {
    int            n_members;      // 2 or 3
    MmqGroupMember m[3];
    int            tile_prefix[4];
    int            rows_prefix[4];
    int            total_rows;
    int            nsplit;
    float *        part;
    int            ncols_dst;
    int            ncols_y;
};

// dst[k] = weights[k] @ dequant(y_q8) for every member, in one launch: each
// member runs the same tile body as mmq_up_matmul with the tile configuration
// the imported table selects for the shared column count.  The members must
// use formats and shapes this device serves (rows a multiple of 128, K a
// multiple of 128) and their configurations must agree on the block shape.
// Returns false when the group is not launchable; the caller then runs the
// members' separate matmuls.  The epilogue values are the ones mmq_up_matmul
// produces.
bool mmq_up_group_matmul(const MmqGroupArgs & args, const MmqUpType * types,
                         cudaStream_t stream);

// Test-only: launch a group with an explicit split count, bypassing the
// warm-up pin.  The private tests use it to check the grouped launch against
// the members' separate matmuls at each split count.
bool mmq_up_group_matmul_split(const MmqGroupArgs & args,
                               const MmqUpType * types, int split_k,
                               cudaStream_t stream);

// Raise the group kernels' dynamic shared-memory limit for every tile width
// they can be launched with, for the closed set of format tuples this build
// instantiates.  Called once at warm-up; returns false for a tuple with no
// instantiation.
bool mmq_up_group_warm(const MmqUpType * types, int n_members);

// The group's split-K count, as measured (or loaded) by
// mmq_up_group_warm_split: 1 is the single-pass path.  mmq_up_group_matmul
// applies it itself.
int mmq_up_group_split_for(const MmqUpType * types, const int64_t * rows,
                           int n_members, int64_t cols);

// Measure (or load from the tune file) one sibling group's split-K count: the
// same S in {1, 2, 4} table, cost rule and row-count probes the per-shape
// choice uses, timed over the group launch.  The members must share one
// column count and be launchable as a group.  build_id keys the record.
void mmq_up_group_warm_split(const MmqUpType * types, const int64_t * rows,
                             int n_members, int64_t cols, const char * build_id);

}  // namespace sonicboom::quant::cuda
