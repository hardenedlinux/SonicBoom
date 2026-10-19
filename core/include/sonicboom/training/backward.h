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

// --- R4: RMSNorm / RoPE / SwiGLU / embedding --------------------------------
//
// rmsnorm:  y[r,d] = x[r,d] * r * w[d],  r = 1/sqrt(mean_d(x[r,d]^2) + eps).
//   Backward of a [rows, dim] input with per-column weight w [dim]:
//   dX[r,d] = r * w[d] * dY[r,d] - r * xhat[r,d] * mean_d(dY[r,d]*w[d]*xhat[r,d])
//   dW[d]   = sum_r dY[r,d] * xhat[r,d],      xhat[r,d] = x[r,d] * r.
// rope: GPT-NeoX rotation, head-major [rows, heads*head_dim] layout, position
//   `pos0 + row` for row `row`, pair (i, i + head_dim/2) rotated by
//   angle = pos * freq_scale * theta_base^(-2i/head_dim). The rotation is
//   orthogonal, so the backward is the rotation by the negated angle.
// silu_mul (SwiGLU): z = silu(g) * u, elementwise, same shape for g, u, dZ.
//   dG = dZ * u * silu'(g),  dU = dZ * silu(g),
//   silu'(g) = sigmoid(g) * (1 + g * (1 - sigmoid(g))).
// embedding: y[r,:] = W[tokens[r],:] with W [vocab, dim], tokens [rows] int32.
//   Backward accumulates dY's rows into dW [vocab, dim] (no weights needed;
//   dW is written, not accumulated into a pre-existing buffer).

// Host references (double accumulation, float storage).
void rmsnorm_bwd_f32(const float* x, const float* w, const float* dY,
                     int64_t rows, int64_t dim, float eps, float* dX,
                     float* dW);

// The rotation reads no forward activation: the position of a row is
// `pos0 + row`, so only the upstream gradient is needed.
void rope_bwd_f32(const float* dY, int64_t rows, int64_t heads,
                  int64_t head_dim, uint64_t pos0, float theta_base,
                  float freq_scale, float* dX);

void silu_mul_bwd_f32(const float* g, const float* u, const float* dZ,
                      int64_t n, float* dG, float* dU);

void embedding_bwd_f32(const int32_t* tokens, const float* dY, int64_t rows,
                       int64_t dim, int64_t vocab, float* dW);

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

bool rmsnorm_bwd(std::span<const float> x, std::span<const float> w,
                 std::span<const float> dY, int64_t rows, int64_t dim,
                 float eps, std::span<float> dX, std::span<float> dW);

bool rope_bwd(std::span<const float> dY, int64_t rows, int64_t heads,
              int64_t head_dim, uint64_t pos0, float theta_base,
              float freq_scale, std::span<float> dX);

bool silu_mul_bwd(std::span<const float> g, std::span<const float> u,
                  std::span<const float> dZ, std::span<float> dG,
                  std::span<float> dU);

bool embedding_bwd(std::span<const int32_t> tokens, std::span<const float> dY,
                   int64_t rows, int64_t dim, int64_t vocab,
                   std::span<float> dW);

} // namespace cuda

} // namespace sonicboom::training
