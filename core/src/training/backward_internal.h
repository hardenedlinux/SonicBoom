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

} // namespace sonicboom::training::detail
