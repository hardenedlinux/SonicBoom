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

// Global-norm gradient clipping over a set of host float32 tensors. This is
// the CPU reference of the clip_grad_norm training op; the CUDA device path
// lives in device_graph_kernels.cu (one accumulation pass, then one scale
// kernel per gradient).
//
// The norm is accumulated in double over every element of every tensor, as
// CS336's clip_grad_norm_ does (torch uses a float64 accumulator for the
// global norm), so a large parameter set cannot lose precision to float32
// rounding before the comparison with max_norm.

#include <cstddef>
#include <span>

namespace sonicboom::training {

// Added to the norm before the scale division, so max_norm = 0 scales to zero
// without dividing by zero.
inline constexpr double kClipGradNormEps = 1e-6;

// L2 norm of the concatenation of GRADS (each SIZES[i] float32 elements).
// SIZES and GRADS must have the same length; a null pointer contributes
// nothing.
double grad_global_norm(std::span<float* const> grads,
                        std::span<const std::size_t> sizes) noexcept;

// Clip every gradient in place when the global norm exceeds MAX_NORM:
// g <- g * MAX_NORM / (norm + kClipGradNormEps). When the norm is at most
// MAX_NORM the buffers are not written at all, so they stay bit-for-bit
// unchanged. Returns the global norm measured before clipping.
double clip_grad_norm_inplace(std::span<float* const> grads,
                              std::span<const std::size_t> sizes,
                              double max_norm) noexcept;

} // namespace sonicboom::training
