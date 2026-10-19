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

// Device backward kernels (R3): matmul VJP, last-axis softmax VJP, and the
// fused mean cross-entropy loss + gradient.
//
// Every reduction accumulates in double and stores float, so the kernels track
// the host reference to fp32 rounding only — the numerics gate compares them
// against an independent fp64 reference. The matmul and softmax kernels use a
// fixed accumulation order (ascending index within one thread's slice; a fixed
// shared-memory tree across threads), so repeated runs are bit-identical.

#include <sonicboom/training/backward.h>

#include "backward_internal.h"

#include "../nn/cuda_stream.h"
#include "../planner/device_pool.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace sonicboom::training {

namespace cu = ::sonicboom::cuda;

namespace {

inline cudaStream_t cur_stream() { return cu::sb_current_stream(); }

constexpr int kBlock = 256;

// A fixed-shape (kBlock) shared-memory reduction. `op` folds an element.
template <typename Op>
__device__ inline double block_reduce(double v, double* sh, Op op) {
  const int tid = threadIdx.x;
  sh[tid] = v;
  __syncthreads();
  for (int off = kBlock / 2; off > 0; off >>= 1) {
    if (tid < off)
      sh[tid] = op(sh[tid], sh[tid + off]);
    __syncthreads();
  }
  return sh[0];
}

struct AddOp {
  __device__ double operator()(double a, double b) const { return a + b; }
};
struct MaxOp {
  __device__ double operator()(double a, double b) const { return fmax(a, b); }
};

// --- matmul backward --------------------------------------------------------

// dA[m,k] = sum_n dY[m,n] * B[k,n].
__global__ void matmul_bwd_da_kernel(const float* __restrict__ dY,
                                     const float* __restrict__ B,
                                     float* __restrict__ dA, int64_t M,
                                     int64_t K, int64_t N) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < M * K; i += stride) {
    const int64_t m = i / K;
    const int64_t k = i % K;
    double s = 0.0;
    for (int64_t n = 0; n < N; ++n)
      s += static_cast<double>(dY[m * N + n]) *
           static_cast<double>(B[k * N + n]);
    dA[i] = static_cast<float>(s);
  }
}

// dB[k,n] = sum_m A[m,k] * dY[m,n].
__global__ void matmul_bwd_db_kernel(const float* __restrict__ A,
                                     const float* __restrict__ dY,
                                     float* __restrict__ dB, int64_t M,
                                     int64_t K, int64_t N) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < K * N; i += stride) {
    const int64_t k = i / N;
    const int64_t n = i % N;
    double s = 0.0;
    for (int64_t m = 0; m < M; ++m)
      s += static_cast<double>(A[m * K + k]) *
           static_cast<double>(dY[m * N + n]);
    dB[i] = static_cast<float>(s);
  }
}

// --- softmax backward -------------------------------------------------------

// One block per row: s = sum_c dY*Y, then dX = Y * (dY - s).
__global__ void softmax_bwd_kernel(const float* __restrict__ Y,
                                   const float* __restrict__ dY,
                                   float* __restrict__ dX, int64_t cols) {
  __shared__ double sh[kBlock];
  const int64_t r = blockIdx.x;
  const float* y = Y + r * cols;
  const float* dy = dY + r * cols;
  float* dx = dX + r * cols;
  double s = 0.0;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    s += static_cast<double>(dy[c]) * static_cast<double>(y[c]);
  const double total = block_reduce(s, sh, AddOp{});
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    dx[c] = static_cast<float>(static_cast<double>(y[c]) *
                               (static_cast<double>(dy[c]) - total));
}

// --- cross-entropy ----------------------------------------------------------

// One block per row: the row's loss goes to `row_loss`, its gradient to
// dlogits (already divided by the number of rows).
__global__ void ce_row_kernel(const float* __restrict__ logits,
                              const int32_t* __restrict__ targets,
                              float* __restrict__ dlogits,
                              float* __restrict__ row_loss, int64_t rows,
                              int64_t cols) {
  __shared__ double sh_max[kBlock];
  __shared__ double sh_sum[kBlock];
  const int64_t r = blockIdx.x;
  const float* x = logits + r * cols;
  float* dx = dlogits + r * cols;
  const int32_t t = targets[r];

  double m = -INFINITY;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    m = fmax(m, static_cast<double>(x[c]));
  m = block_reduce(m, sh_max, MaxOp{});
  __syncthreads();

  double s = 0.0;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    s += exp(static_cast<double>(x[c]) - m);
  const double sum = block_reduce(s, sh_sum, AddOp{});
  __syncthreads();

  if (threadIdx.x == 0)
    row_loss[r] =
        static_cast<float>(-(static_cast<double>(x[t]) - m - log(sum)));

  const double inv_rows = 1.0 / static_cast<double>(rows);
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    double p = exp(static_cast<double>(x[c]) - m) / sum;
    if (c == t)
      p -= 1.0;
    dx[c] = static_cast<float>(p * inv_rows);
  }
}

// Deterministic reduction of the per-row losses: one block, ascending strided
// slices, fixed shared-memory tree.
__global__ void ce_sum_kernel(const float* __restrict__ row_loss,
                              float* __restrict__ loss, int64_t rows) {
  __shared__ double sh[kBlock];
  double s = 0.0;
  for (int64_t r = threadIdx.x; r < rows; r += blockDim.x)
    s += static_cast<double>(row_loss[r]);
  const double total = block_reduce(s, sh, AddOp{});
  if (threadIdx.x == 0)
    loss[0] = static_cast<float>(total / static_cast<double>(rows));
}

int grid_for(int64_t total) {
  const int64_t blocks = (total + kBlock - 1) / kBlock;
  return static_cast<int>(blocks > 4096 ? 4096 : blocks);
}

bool ok() { return cudaGetLastError() == cudaSuccess; }

} // namespace

namespace detail {

bool matmul_bwd_dev(std::uint64_t dA, std::uint64_t dB, std::uint64_t A,
                    std::uint64_t B, std::uint64_t dY, std::int64_t M,
                    std::int64_t K, std::int64_t N) {
  if (M <= 0 || K <= 0 || N <= 0 || dA == 0 || dB == 0 || A == 0 || B == 0 ||
      dY == 0)
    return false;
  const cudaStream_t s = cur_stream();
  matmul_bwd_da_kernel<<<grid_for(M * K), kBlock, 0, s>>>(
      reinterpret_cast<const float*>(dY), reinterpret_cast<const float*>(B),
      reinterpret_cast<float*>(dA), M, K, N);
  matmul_bwd_db_kernel<<<grid_for(K * N), kBlock, 0, s>>>(
      reinterpret_cast<const float*>(A), reinterpret_cast<const float*>(dY),
      reinterpret_cast<float*>(dB), M, K, N);
  return ok();
}

bool softmax_bwd_dev(std::uint64_t dX, std::uint64_t Y, std::uint64_t dY,
                     std::int64_t rows, std::int64_t cols) {
  if (rows <= 0 || cols <= 0 || dX == 0 || Y == 0 || dY == 0)
    return false;
  softmax_bwd_kernel<<<rows, kBlock, 0, cur_stream()>>>(
      reinterpret_cast<const float*>(Y), reinterpret_cast<const float*>(dY),
      reinterpret_cast<float*>(dX), cols);
  return ok();
}

bool cross_entropy_bwd_dev(std::uint64_t loss, std::uint64_t dlogits,
                           std::uint64_t logits, std::uint64_t targets,
                           std::int64_t rows, std::int64_t cols) {
  if (rows <= 0 || cols <= 0 || loss == 0 || dlogits == 0 || logits == 0 ||
      targets == 0)
    return false;
  const std::uint64_t scratch_bytes =
      static_cast<std::uint64_t>(rows) * sizeof(float);
  const std::uint64_t scratch = planner::device::acquire(scratch_bytes);
  if (!scratch)
    return false;
  const cudaStream_t s = cur_stream();
  ce_row_kernel<<<rows, kBlock, 0, s>>>(
      reinterpret_cast<const float*>(logits),
      reinterpret_cast<const int32_t*>(targets),
      reinterpret_cast<float*>(dlogits), reinterpret_cast<float*>(scratch),
      rows, cols);
  ce_sum_kernel<<<1, kBlock, 0, s>>>(reinterpret_cast<const float*>(scratch),
                                     reinterpret_cast<float*>(loss), rows);
  const bool good = ok();
  // The scratch returns to the pool after the reduction has finished; the
  // training loop is step-synchronous, so this is not on a hot path.
  cudaStreamSynchronize(s);
  planner::device::release(scratch, scratch_bytes);
  return good;
}

} // namespace detail

namespace cuda {

bool available() noexcept { return planner::device::available(); }

namespace {

// A host<->device staging allocation for the span-based entry points.
struct DevBuf {
  void* p = nullptr;
  explicit DevBuf(std::uint64_t bytes) {
    if (bytes)
      cudaMalloc(&p, bytes);
  }
  ~DevBuf() {
    if (p)
      cudaFree(p);
  }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  bool ok() const { return p != nullptr; }
};

bool upload(const void* src, void* dst, std::uint64_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
}

bool download(const void* src, void* dst, std::uint64_t bytes) {
  return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
}

std::uint64_t bytes_of(std::size_t n, std::size_t elem) {
  return static_cast<std::uint64_t>(n) * elem;
}

} // namespace

bool matmul_bwd(std::span<const float> A, std::span<const float> B,
                std::span<const float> dY, int64_t M, int64_t K, int64_t N,
                std::span<float> dA, std::span<float> dB) {
  if (!available() || M <= 0 || K <= 0 || N <= 0)
    return false;
  const auto mk = static_cast<std::size_t>(M * K);
  const auto kn = static_cast<std::size_t>(K * N);
  const auto mn = static_cast<std::size_t>(M * N);
  if (A.size() != mk || B.size() != kn || dY.size() != mn || dA.size() != mk ||
      dB.size() != kn)
    return false;

  DevBuf da(bytes_of(mk, sizeof(float))), db(bytes_of(kn, sizeof(float))),
      a(bytes_of(mk, sizeof(float))), b(bytes_of(kn, sizeof(float))),
      dy(bytes_of(mn, sizeof(float)));
  if (!da.ok() || !db.ok() || !a.ok() || !b.ok() || !dy.ok())
    return false;
  if (!upload(A.data(), a.p, bytes_of(mk, sizeof(float))) ||
      !upload(B.data(), b.p, bytes_of(kn, sizeof(float))) ||
      !upload(dY.data(), dy.p, bytes_of(mn, sizeof(float))))
    return false;
  if (!detail::matmul_bwd_dev(reinterpret_cast<std::uint64_t>(da.p),
                              reinterpret_cast<std::uint64_t>(db.p),
                              reinterpret_cast<std::uint64_t>(a.p),
                              reinterpret_cast<std::uint64_t>(b.p),
                              reinterpret_cast<std::uint64_t>(dy.p), M, K, N))
    return false;
  if (cudaStreamSynchronize(cur_stream()) != cudaSuccess)
    return false;
  return download(da.p, dA.data(), bytes_of(mk, sizeof(float))) &&
         download(db.p, dB.data(), bytes_of(kn, sizeof(float)));
}

bool softmax_bwd(std::span<const float> Y, std::span<const float> dY,
                 int64_t rows, int64_t cols, std::span<float> dX) {
  if (!available() || rows <= 0 || cols <= 0)
    return false;
  const auto n = static_cast<std::size_t>(rows * cols);
  if (Y.size() != n || dY.size() != n || dX.size() != n)
    return false;

  DevBuf y(bytes_of(n, sizeof(float))), dy(bytes_of(n, sizeof(float))),
      dx(bytes_of(n, sizeof(float)));
  if (!y.ok() || !dy.ok() || !dx.ok())
    return false;
  if (!upload(Y.data(), y.p, bytes_of(n, sizeof(float))) ||
      !upload(dY.data(), dy.p, bytes_of(n, sizeof(float))))
    return false;
  if (!detail::softmax_bwd_dev(reinterpret_cast<std::uint64_t>(dx.p),
                               reinterpret_cast<std::uint64_t>(y.p),
                               reinterpret_cast<std::uint64_t>(dy.p), rows,
                               cols))
    return false;
  if (cudaStreamSynchronize(cur_stream()) != cudaSuccess)
    return false;
  return download(dx.p, dX.data(), bytes_of(n, sizeof(float)));
}

bool cross_entropy_bwd(std::span<const float> logits,
                       std::span<const int32_t> targets, int64_t rows,
                       int64_t cols, std::span<float> dlogits,
                       float* loss_out) {
  if (!available() || rows <= 0 || cols <= 0 || !loss_out)
    return false;
  const auto n = static_cast<std::size_t>(rows * cols);
  if (logits.size() != n || targets.size() != static_cast<std::size_t>(rows) ||
      dlogits.size() != n)
    return false;
  for (int32_t t : targets)
    if (t < 0 || t >= cols)
      return false;

  DevBuf x(bytes_of(n, sizeof(float))), t(bytes_of(targets.size(), sizeof(int32_t))),
      dx(bytes_of(n, sizeof(float))), loss(sizeof(float));
  if (!x.ok() || !t.ok() || !dx.ok() || !loss.ok())
    return false;
  if (!upload(logits.data(), x.p, bytes_of(n, sizeof(float))) ||
      !upload(targets.data(), t.p, bytes_of(targets.size(), sizeof(int32_t))))
    return false;
  if (!detail::cross_entropy_bwd_dev(
          reinterpret_cast<std::uint64_t>(loss.p),
          reinterpret_cast<std::uint64_t>(dx.p),
          reinterpret_cast<std::uint64_t>(x.p),
          reinterpret_cast<std::uint64_t>(t.p), rows, cols))
    return false;
  if (cudaStreamSynchronize(cur_stream()) != cudaSuccess)
    return false;
  if (!download(dx.p, dlogits.data(), bytes_of(n, sizeof(float))) ||
      !download(loss.p, loss_out, sizeof(float)))
    return false;
  return std::isfinite(*loss_out);
}

} // namespace cuda

} // namespace sonicboom::training
