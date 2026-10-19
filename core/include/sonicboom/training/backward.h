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

// Training backward kernels (R3): the VJPs of matmul, last-axis softmax and
// mean cross-entropy.
//
// The host entry points are the double-precision reference the device kernels
// are checked against and the CPU fallback for the training runtime. The CUDA
// entry points take host spans, move them to the device, launch one kernel per
// op, and copy the result back — the same shape as <sonicboom/nn/cuda_*.h>.
// The device-resident graph executor uses the internal handle-based entry
// points instead (core/src/training/backward_internal.h).
//
// Layouts are row-major throughout, matching the forward operators:
//   matmul:        Y  = A  @ B      A [M,K], B [K,N], Y [M,N]
//   matmul_bwd:    dA = dY @ B^T    dB = A^T @ dY
//   softmax_bwd:   dX = Y * (dY - sum_col(dY * Y)),  Y last-axis softmax
//   cross_entropy: loss = mean_r(-log softmax(logits)[r][targets[r]])
//                  dlogits = (softmax(logits) - onehot(targets)) / rows
//
// No CUDA type appears here; the declarations are only linked when the build
// has SONICBOOM_USE_CUDA=ON.

#include <cstdint>
#include <span>

namespace sonicboom::training {

// Host reference: A, B and dY are [M,K], [K,N], [M,N]; dA and dB receive the
// same shapes as A and B. Accumulates in double, stores float.
void matmul_bwd_f32(const float* A, const float* B, const float* dY, int64_t M,
                    int64_t K, int64_t N, float* dA, float* dB);

// Host reference: Y is a [rows, cols] last-axis softmax output. Accumulates in
// double, stores float.
void softmax_bwd_f32(const float* Y, const float* dY, int64_t rows,
                     int64_t cols, float* dX);

// Host reference: logits [rows, cols], targets [rows] int32. Writes dlogits
// [rows, cols] and returns the mean loss. Accumulates the loss and the
// normalization in double, stores float.
float cross_entropy_bwd_f32(const float* logits, const int32_t* targets,
                            int64_t rows, int64_t cols, float* dlogits);

namespace cuda {

// True when a usable CUDA device is present (cached after the first call).
bool available() noexcept;

// Device versions of the three backward ops. They return false (writing
// nothing) when CUDA is unavailable, a shape is degenerate, a span has the
// wrong size, or a device operation fails.
bool matmul_bwd(std::span<const float> A, std::span<const float> B,
                std::span<const float> dY, int64_t M, int64_t K, int64_t N,
                std::span<float> dA, std::span<float> dB);

bool softmax_bwd(std::span<const float> Y, std::span<const float> dY,
                 int64_t rows, int64_t cols, std::span<float> dX);

// `loss_out` receives the scalar mean loss on success.
bool cross_entropy_bwd(std::span<const float> logits,
                       std::span<const int32_t> targets, int64_t rows,
                       int64_t cols, std::span<float> dlogits,
                       float* loss_out);

} // namespace cuda

} // namespace sonicboom::training
