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

// Fused AdamW device kernel: one pass over the parameter updating the two
// moment buffers and the parameter itself. Grid-stride so any parameter size
// uses one launch; the buffers are device-resident, so a training step costs
// one kernel launch and no host round-trip.

#include "training_internal.h"

#include "../nn/cuda_stream.h"

#include <cuda_runtime.h>

#include <cmath>

namespace sonicboom::training::detail {

namespace {

__global__ void adamw_kernel(float* param, const float* grad, float* m1,
                             float* m2, long n, float lr, float beta1,
                             float beta2, float bc1_scale, float bc2_scale,
                             float eps, float weight_decay) {
  const long stride = static_cast<long>(gridDim.x) * blockDim.x;
  for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += stride) {
    const float g = grad[i];
    const float mi = beta1 * m1[i] + (1.0f - beta1) * g;
    const float vi = beta2 * m2[i] + (1.0f - beta2) * g * g;
    m1[i] = mi;
    m2[i] = vi;

    const float mh = mi * bc1_scale;  // m_hat: 1/(1-b1^t) folded in
    const float vh = vi * bc2_scale;  // v_hat
    param[i] = fmaf(-lr, mh / (sqrtf(vh) + eps) + weight_decay * param[i],
                    param[i]);
  }
}

} // namespace

bool adamw_dev(std::uint64_t param, std::uint64_t grad, std::uint64_t m1,
               std::uint64_t m2, std::size_t n, const AdamWConfig& cfg,
               std::uint64_t step) {
  if (n == 0 || param == 0 || grad == 0 || m1 == 0 || m2 == 0)
    return false;

  const double b1 = cfg.beta1;
  const double b2 = cfg.beta2;
  const float bc1_scale =
      static_cast<float>(1.0 / (1.0 - std::pow(b1, static_cast<double>(step))));
  const float bc2_scale =
      static_cast<float>(1.0 / (1.0 - std::pow(b2, static_cast<double>(step))));

  constexpr int kThreads = 256;
  const long total = static_cast<long>(n);
  int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (blocks > 4096) blocks = 4096;

  adamw_kernel<<<blocks, kThreads, 0, ::sonicboom::cuda::sb_current_stream()>>>(
      reinterpret_cast<float*>(param), reinterpret_cast<const float*>(grad),
      reinterpret_cast<float*>(m1), reinterpret_cast<float*>(m2), total,
      cfg.lr, cfg.beta1, cfg.beta2, bc1_scale, bc2_scale, cfg.eps,
      cfg.weight_decay);
  return cudaGetLastError() == cudaSuccess;
}

} // namespace sonicboom::training::detail
