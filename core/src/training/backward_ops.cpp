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

// Host reference kernels for the training backward ops. Every accumulation is
// in double and only the stored result is float, so the reference is the
// fp64 formula the device kernels are gated against.

#include <sonicboom/training/backward.h>

#include <cmath>

namespace sonicboom::training {

void matmul_bwd_f32(const float* A, const float* B, const float* dY, int64_t M,
                    int64_t K, int64_t N, float* dA, float* dB) {
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t k = 0; k < K; ++k) {
      double s = 0.0;
      for (int64_t n = 0; n < N; ++n)
        s += static_cast<double>(dY[m * N + n]) *
             static_cast<double>(B[k * N + n]);
      dA[m * K + k] = static_cast<float>(s);
    }
  }
  for (int64_t k = 0; k < K; ++k) {
    for (int64_t n = 0; n < N; ++n) {
      double s = 0.0;
      for (int64_t m = 0; m < M; ++m)
        s += static_cast<double>(A[m * K + k]) *
             static_cast<double>(dY[m * N + n]);
      dB[k * N + n] = static_cast<float>(s);
    }
  }
}

void softmax_bwd_f32(const float* Y, const float* dY, int64_t rows,
                     int64_t cols, float* dX) {
  for (int64_t r = 0; r < rows; ++r) {
    const float* y = Y + r * cols;
    const float* dy = dY + r * cols;
    float* dx = dX + r * cols;
    double s = 0.0;
    for (int64_t c = 0; c < cols; ++c)
      s += static_cast<double>(dy[c]) * static_cast<double>(y[c]);
    for (int64_t c = 0; c < cols; ++c)
      dx[c] = static_cast<float>(static_cast<double>(y[c]) *
                                 (static_cast<double>(dy[c]) - s));
  }
}

float cross_entropy_bwd_f32(const float* logits, const int32_t* targets,
                            int64_t rows, int64_t cols, float* dlogits) {
  double total = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    const float* x = logits + r * cols;
    float* dx = dlogits + r * cols;
    double max = -INFINITY;
    for (int64_t c = 0; c < cols; ++c)
      max = std::fmax(max, static_cast<double>(x[c]));
    double sum = 0.0;
    for (int64_t c = 0; c < cols; ++c)
      sum += std::exp(static_cast<double>(x[c]) - max);
    const int64_t t = targets[r];
    total += -(static_cast<double>(x[t]) - max - std::log(sum));
    for (int64_t c = 0; c < cols; ++c) {
      double p = std::exp(static_cast<double>(x[c]) - max) / sum;
      if (c == t)
        p -= 1.0;
      dx[c] = static_cast<float>(p / static_cast<double>(rows));
    }
  }
  return static_cast<float>(total / static_cast<double>(rows));
}

} // namespace sonicboom::training
