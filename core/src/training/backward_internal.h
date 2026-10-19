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

// Internal, handle-based device entry points for the backward ops. Like
// training_internal.h, the handles are raw CUDA device pointers and this
// header carries no CUDA type; the implementation lives in backward_ops.cu.
// The training graph executor calls these so a backward pass costs kernel
// launches only, with every activation already resident on the device.

#include <cstdint>

namespace sonicboom::training::detail {

bool matmul_bwd_dev(std::uint64_t dA, std::uint64_t dB, std::uint64_t A,
                    std::uint64_t B, std::uint64_t dY, std::int64_t M,
                    std::int64_t K, std::int64_t N);

bool softmax_bwd_dev(std::uint64_t dX, std::uint64_t Y, std::uint64_t dY,
                     std::int64_t rows, std::int64_t cols);

// `targets` is an int32 device buffer of `rows` elements; `loss` is a
// one-element f32 device buffer.
bool cross_entropy_bwd_dev(std::uint64_t loss, std::uint64_t dlogits,
                           std::uint64_t logits, std::uint64_t targets,
                           std::int64_t rows, std::int64_t cols);

bool rmsnorm_bwd_dev(std::uint64_t dX, std::uint64_t dW, std::uint64_t x,
                     std::uint64_t w, std::uint64_t dY, std::int64_t rows,
                     std::int64_t dim, float eps);

bool rope_bwd_dev(std::uint64_t dX, std::uint64_t dY, std::int64_t rows,
                  std::int64_t heads, std::int64_t head_dim,
                  std::uint64_t pos0, float theta_base, float freq_scale);

bool silu_mul_bwd_dev(std::uint64_t dG, std::uint64_t dU, std::uint64_t g,
                      std::uint64_t u, std::uint64_t dZ, std::int64_t n);

// `tokens` is an int32 device buffer of `rows` elements; dW [vocab, dim] is
// written in full (no zero-fill needed).
bool embedding_bwd_dev(std::uint64_t dW, std::uint64_t tokens,
                       std::uint64_t dY, std::int64_t rows,
                       std::int64_t dim, std::int64_t vocab);

} // namespace sonicboom::training::detail
