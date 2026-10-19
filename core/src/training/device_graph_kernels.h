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

// Internal device API of the training graph executor. Buffers are opaque
// device handles (uint64_t), so this header carries no CUDA type; the
// implementation lives in device_graph_kernels.cu and is compiled only when
// SONICBOOM_USE_CUDA is on.
//
// Every kernel runs on the current SonicBoom stream. Shape/stride arrays are
// host memory: they are small, per-node, and read at launch, not in-kernel.

#include <cstdint>

namespace sonicboom::training::detail {

using DevBuf = std::uint64_t;

// --- buffers ----------------------------------------------------------------

DevBuf dev_alloc(std::uint64_t bytes);
void dev_free(DevBuf buf);
bool dev_upload(DevBuf dst, const void* src, std::uint64_t bytes);
bool dev_download(DevBuf src, void* dst, std::uint64_t bytes);
bool dev_zero(DevBuf dst, std::uint64_t bytes);
bool dev_copy(DevBuf dst, DevBuf src, std::int64_t n);
// Wait for every kernel submitted on the current stream.
bool dev_sync();

// --- forward ----------------------------------------------------------------

// out[r, :] = table[ids[r], :]; ids is int32, rows == number of ids.
bool graph_index_select(DevBuf out, DevBuf table, DevBuf ids, std::int64_t rows,
                        std::int64_t dim);
bool graph_matmul(DevBuf c, DevBuf a, DevBuf b, std::int64_t m,
                  std::int64_t k, std::int64_t n);
bool graph_bmm(DevBuf c, DevBuf a, DevBuf b, std::int64_t batch,
               std::int64_t m, std::int64_t k, std::int64_t n);
// `src_strides[d]` is the source's stride along output axis d; out_dims is
// the output shape (rank entries).
bool graph_permute(DevBuf out, DevBuf in, const std::int64_t* src_strides,
                   std::int64_t rank, std::int64_t out_numel,
                   const std::int64_t* out_dims);
bool graph_softmax(DevBuf out, DevBuf in, std::int64_t rows,
                   std::int64_t cols);
// y = x * rsqrt(mean(x^2) + eps) * w, with w a device buffer of `dim` floats.
bool graph_rms_norm(DevBuf out, DevBuf x, DevBuf w, std::int64_t rows,
                    std::int64_t dim, float eps);
// In-place GPT-NeoX rotation of [rows, heads * head_dim]; row r sits at
// position pos0 + r.
bool graph_rope_inplace(DevBuf x, std::int64_t rows, std::int64_t heads,
                        std::int64_t head_dim, std::uint64_t pos0,
                        float theta_base, float freq_scale);
bool graph_silu_mul(DevBuf out, DevBuf g, DevBuf u, std::int64_t n);
// Mean negative log-likelihood over rows; targets is int32, loss one float.
bool graph_cross_entropy(DevBuf loss, DevBuf logits, DevBuf targets,
                         std::int64_t rows, std::int64_t cols);

// out = a + b and out = a * b with numpy broadcasting. `a_strides`/`b_strides`
// are aligned to the output rank, 0 on broadcast axes.
bool graph_add(DevBuf out, DevBuf a, DevBuf b, const std::int64_t* a_strides,
               const std::int64_t* b_strides, const std::int64_t* out_dims,
               std::int64_t rank, std::int64_t out_numel);
bool graph_mul(DevBuf out, DevBuf a, DevBuf b, const std::int64_t* a_strides,
               const std::int64_t* b_strides, const std::int64_t* out_dims,
               std::int64_t rank, std::int64_t out_numel);
// Sum over the axes whose bit is set in `axes_mask`; out is the tensor
// table's result shape (rank-lowering when keepdims == 0).
bool graph_reduce_sum(DevBuf out, DevBuf in, const std::int64_t* in_dims,
                      const std::int64_t* in_strides,
                      const std::int64_t* out_dims, std::int64_t rank,
                      std::uint64_t axes_mask, std::int64_t out_numel);

// One AdamW update written as a pure function: new param/m1/m2 into the three
// output buffers, `step` is a one-element int32 device buffer (1-based).
bool graph_adamw_step(DevBuf param_out, DevBuf m1_out, DevBuf m2_out,
                      DevBuf param, DevBuf grad, DevBuf m1, DevBuf m2,
                      std::int64_t n, DevBuf step, float lr, float beta1,
                      float beta2, float eps, float weight_decay);

// Global-norm clipping. graph_sumsq_acc adds sum(x^2) over N float32 elements
// into ACC, a one-element double device buffer the caller has zeroed; the
// block sums are combined with double atomics, so the accumulation is one pass
// over the data. graph_scale_copy writes out = in * scale; with scale == 1.0
// it copies the values exactly.
bool graph_sumsq_acc(DevBuf acc, DevBuf x, std::int64_t n);
bool graph_scale_copy(DevBuf out, DevBuf in, std::int64_t n, double scale);

} // namespace sonicboom::training::detail
