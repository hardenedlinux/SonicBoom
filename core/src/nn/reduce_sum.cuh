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

// The block-wide sum-of-squares reduction shared by the norm kernels, for each
// ReduceMode. The tree shape is identical in all three modes: each thread
// accumulates a strided slice, a shuffle-down warp tree reduces the warp, the
// lane-0 partials go through shared memory, and warp 0 runs the same
// shuffle-down tree over them. Only the per-add arithmetic differs.

#include <sonicboom/compiler/ReduceMode.h>

#include <cuda_runtime.h>

namespace sonicboom::nn::cuda {

using sonicboom::compiler::Df32;

// F64: the historical double accumulator (bit-identical reference).
struct SumF64 {
  using acc_t = double;
  using part_t = double;
  using out_t = double;
  __device__ static acc_t zero() { return 0.0; }
  __device__ static acc_t add_term(acc_t a, float sq) {
    return __dadd_rn(a, double(sq));
  }
  __device__ static acc_t add(acc_t a, acc_t b) { return __dadd_rn(a, b); }
  __device__ static acc_t shfl(acc_t a, int o) {
    return __shfl_down_sync(0xffffffffu, a, o);
  }
  __device__ static part_t to_part(acc_t a) { return a; }
  __device__ static acc_t from_part(part_t p) { return p; }
  __device__ static out_t to_out(acc_t a) { return a; }
  __device__ static float mean_to_float(acc_t a, int n) {
    return float(__ddiv_rn(a, double(n)));
  }
};

// DF32: a double-float pair accumulated with TwoSum; the tree carries both
// limbs through the shuffles and the shared-memory partials.
struct SumDf32 {
  using acc_t = Df32;
  using part_t = float2;
  using out_t = float2;
  __device__ static acc_t zero() { return {0.0f, 0.0f}; }
  __device__ static acc_t add_term(acc_t a, float sq) {
    return df32_add_float(a, sq);
  }
  __device__ static acc_t add(acc_t a, acc_t b) { return df32_add(a, b); }
  __device__ static acc_t shfl(acc_t a, int o) {
    return {__shfl_down_sync(0xffffffffu, a.hi, o),
            __shfl_down_sync(0xffffffffu, a.lo, o)};
  }
  __device__ static part_t to_part(acc_t a) { return make_float2(a.hi, a.lo); }
  __device__ static acc_t from_part(part_t p) { return {p.x, p.y}; }
  __device__ static out_t to_out(acc_t a) { return make_float2(a.hi, a.lo); }
  __device__ static float mean_to_float(acc_t a, int n) {
    const float nf = float(n);
    return __fadd_rn(__fdiv_rn(a.hi, nf), __fdiv_rn(a.lo, nf));
  }
};

// F32: plain float accumulator (performance reference / manual override).
struct SumF32 {
  using acc_t = float;
  using part_t = float;
  using out_t = float;
  __device__ static acc_t zero() { return 0.0f; }
  __device__ static acc_t add_term(acc_t a, float sq) { return a + sq; }
  __device__ static acc_t add(acc_t a, acc_t b) { return a + b; }
  __device__ static acc_t shfl(acc_t a, int o) {
    return __shfl_down_sync(0xffffffffu, a, o);
  }
  __device__ static part_t to_part(acc_t a) { return a; }
  __device__ static acc_t from_part(part_t p) { return p; }
  __device__ static out_t to_out(acc_t a) { return a; }
  __device__ static float mean_to_float(acc_t a, int n) { return a / float(n); }
};

// Block-wide sum of squares of x[0..n), one strided slice per thread. `wp` is
// the caller's 32-entry shared partial array. The result is valid on the lane-0
// thread of warp 0 (zero elsewhere).
template <typename R>
__device__ __forceinline__ typename R::acc_t block_sum_sq(
    const float* __restrict__ x, int n, typename R::part_t (&wp)[32]) {
  const int tid = threadIdx.x;
  typename R::acc_t s = R::zero();
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = x[i];
    // __fmul_rn keeps the square as a separate mul.rn (nvcc would otherwise be
    // free to contract it into an fma with the accumulator, which the
    // generated regions' plain arith.mulf/arith.addf do not).
    s = R::add_term(s, __fmul_rn(v, v));
  }
  for (int o = 16; o > 0; o >>= 1) s = R::add(s, R::shfl(s, o));
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (lane == 0) wp[warp] = R::to_part(s);
  __syncthreads();
  if (warp == 0) {
    const int nwarps = (blockDim.x + 31) >> 5;
    s = (lane < nwarps) ? R::from_part(wp[lane]) : R::zero();
    for (int o = 16; o > 0; o >>= 1) s = R::add(s, R::shfl(s, o));
  }
  return s;
}

} // namespace sonicboom::nn::cuda
