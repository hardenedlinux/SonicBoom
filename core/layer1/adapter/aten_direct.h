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

// Direct calls into the native-torch CUDA kernels from the training executor.
//
// The executor hands over opaque device buffers (uint64_t, the same handle as
// device_graph_kernels.h) and dimensions; the adapter wraps them as
// non-owning tensors (at::from_blob) and calls the structured `_out` kernel
// directly — no dispatcher, no allocation, no host/device copy. The stream is
// the executor's current stream, installed as ATen's current CUDA stream for
// the duration of the call.
//
// This header is ATen-free: the executor (core/src/training) includes it.
// Only core/layer1/adapter/aten_direct.cpp sees native-torch types.
//
// SONICBOOM_TRAIN_ATEN selects which executor ops are served natively:
// unset, empty or "0" → none (the SonicBoom kernels, the default); "1" or
// "all" → every implemented op; otherwise a comma-separated list of op names
// (softmax, mul, index_select). The switch exists per op so each one is
// measured on its own.

#include <cstdint>

namespace sonicboom::training::detail {

using AtenDevBuf = std::uint64_t;

// True when SONICBOOM_TRAIN_ATEN names OP.
bool aten_direct_enabled(const char* op);

// out[rows, cols] = softmax(in[rows, cols], dim = -1) in float32.
// The reduction order differs from the SonicBoom kernel, so the result is
// within tolerance, not bitwise, of it.
bool aten_direct_softmax(AtenDevBuf out, AtenDevBuf in, std::int64_t rows,
                         std::int64_t cols, AtenDevBuf stream);

// out[dims] = a[dims with a_strides] * b[dims with b_strides], float32.
// dims/a_strides/b_strides are `rank` host entries (a zero stride broadcasts).
bool aten_direct_mul(AtenDevBuf out, AtenDevBuf a, AtenDevBuf b,
                     const std::int64_t* a_strides,
                     const std::int64_t* b_strides, const std::int64_t* dims,
                     std::int64_t rank, std::int64_t n, AtenDevBuf stream);

// out[rows, dim] = table[ids[r], :]; ids is int32 and table is
// [table_rows, dim]. Native handles at most 16 indices on its direct path
// (beyond that it falls back to the dispatcher's gather), so the caller must
// pass rows <= 16.
bool aten_direct_index_select(AtenDevBuf out, AtenDevBuf table, AtenDevBuf ids,
                              std::int64_t rows, std::int64_t table_rows,
                              std::int64_t dim, AtenDevBuf stream);

} // namespace sonicboom::training::detail
