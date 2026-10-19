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
#include <vector>

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

void rmsnorm_bwd_f32(const float* x, const float* w, const float* dY,
                     int64_t rows, int64_t dim, float eps, float* dX,
                     float* dW) {
  std::vector<double> inv(static_cast<std::size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int64_t d = 0; d < dim; ++d) {
      const double v = x[r * dim + d];
      ss += v * v;
    }
    inv[r] = 1.0 / std::sqrt(ss / static_cast<double>(dim) + eps);

    double s = 0.0;
    for (int64_t d = 0; d < dim; ++d) {
      const double xhat = static_cast<double>(x[r * dim + d]) * inv[r];
      s += static_cast<double>(w[d]) * dY[r * dim + d] * xhat;
    }
    s /= static_cast<double>(dim);
    for (int64_t d = 0; d < dim; ++d) {
      const double xhat = static_cast<double>(x[r * dim + d]) * inv[r];
      const double u = static_cast<double>(w[d]) * dY[r * dim + d];
      dX[r * dim + d] = static_cast<float>(inv[r] * u - inv[r] * xhat * s);
    }
  }
  for (int64_t d = 0; d < dim; ++d) {
    double sum = 0.0;
    for (int64_t r = 0; r < rows; ++r)
      sum += static_cast<double>(dY[r * dim + d]) * x[r * dim + d] * inv[r];
    dW[d] = static_cast<float>(sum);
  }
}

void rope_bwd_f32(const float* dY, int64_t rows, int64_t heads,
                  int64_t head_dim, uint64_t pos0, float theta_base,
                  float freq_scale, float* dX) {
  const double theta = theta_base;
  const int64_t half = head_dim / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(pos0 + static_cast<uint64_t>(r));
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t base = (r * heads + h) * head_dim;
      for (int64_t i = 0; i < half; ++i) {
        const double angle =
            pos * freq_scale *
            std::exp(-2.0 * static_cast<double>(i) / head_dim * std::log(theta));
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double ga = dY[base + i];
        const double gb = dY[base + i + half];
        dX[base + i] = static_cast<float>(ga * c + gb * s);
        dX[base + i + half] = static_cast<float>(-ga * s + gb * c);
      }
    }
  }
}

void silu_mul_bwd_f32(const float* g, const float* u, const float* dZ,
                      int64_t n, float* dG, float* dU) {
  for (int64_t i = 0; i < n; ++i) {
    const double gi = g[i];
    const double sig = 1.0 / (1.0 + std::exp(-gi));
    const double silu = gi * sig;
    const double dz = dZ[i];
    dU[i] = static_cast<float>(dz * silu);
    dG[i] = static_cast<float>(dz * static_cast<double>(u[i]) * sig *
                               (1.0 + gi * (1.0 - sig)));
  }
}

void embedding_bwd_f32(const int32_t* tokens, const float* dY, int64_t rows,
                       int64_t dim, int64_t vocab, float* dW) {
  for (int64_t v = 0; v < vocab; ++v) {
    for (int64_t d = 0; d < dim; ++d) {
      double sum = 0.0;
      for (int64_t r = 0; r < rows; ++r)
        if (tokens[r] == v)
          sum += dY[r * dim + d];
      dW[v * dim + d] = static_cast<float>(sum);
    }
  }
}

} // namespace sonicboom::training
