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

#include <sonicboom/nn/matmul.h>

#include "cpu_reduce.h"

#include <sonicboom/dtype.h>
#include <sonicboom/thread_pool.h>

namespace sonicboom::nn {

bool matvec_f32(std::span<const float> W, uint64_t cols, std::span<const float> x,
                std::span<float> y) {
  if (W.size() < cols * y.size() || x.size() < cols) return false;
  // Each output row accumulates independently into its own y[j], so the row loop
  // is split across the pool deterministically (identical to the serial path).
  ThreadPool::instance().parallel_for(0, y.size(), [&](uint64_t jb, uint64_t je) {
    for (uint64_t j = jb; j < je; ++j) {
      const uint64_t base = j * cols;
      y[j] = cpu_dot(W.subspan(base, cols), x.first(cols));
    }
  });
  return true;
}

bool matvec_bf16(std::span<const uint16_t> W, uint64_t cols,
                 std::span<const float> x, std::span<float> y) {
  if (W.size() < cols * y.size() || x.size() < cols) return false;
  // Same row-parallel structure as matvec_f32: each output row accumulates
  // independently into its own y[j]. Each raw bf16 weight half is widened to f32
  // before the (double-accumulated) product, reproducing ggml_vec_dot_bf16 with
  // the caller's already-rounded activation.
  ThreadPool::instance().parallel_for(0, y.size(), [&](uint64_t jb, uint64_t je) {
    // bf16 weights are widened to f32 before the product; the accumulator is
    // the process-wide reduction mode (f64 by default, matching ggml).
    for (uint64_t j = jb; j < je; ++j) {
      const uint64_t base = j * cols;
      y[j] = cpu_dot_bf16(W.subspan(base, cols), x.first(cols));
    }
  });
  return true;
}

} // namespace sonicboom::nn
