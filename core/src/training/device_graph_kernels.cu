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

// CUDA kernels of the training graph executor.
//
// The graphs are small (a ≤4-layer transformer over a 32-token sequence in the
// R6 milestone), so these are straightforward correctness-first kernels: one
// thread per output element, double accumulation where a reduction is involved,
// and no tiling. The hot fused adjoints are not here — they are the training
// module's own kernels (backward_internal.h), which this executor calls
// directly.
//
// Every buffer argument is a device pointer carried as the opaque uint64_t the
// training module already uses (training_internal.h), so no CUDA type leaves
// this translation unit.

#include "device_graph_kernels.h"

#include "../nn/cuda_resident.h"
#include "../nn/cuda_stream.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace sonicboom::training::detail {

namespace {

inline cudaStream_t cur_stream() { return ::sonicboom::cuda::sb_current_stream(); }

template <typename T>
inline T* ptr(std::uint64_t h) { return reinterpret_cast<T*>(h); }

constexpr int kThreads = 256;
constexpr int kMaxRank = 8;

// The small per-node shape/stride arrays travel by value as kernel arguments
// (copied into the launch's parameter space), not as host pointers: a kernel
// cannot dereference host memory.
struct LayoutRef {
  std::int64_t dims[kMaxRank];
  std::int64_t strides_a[kMaxRank];
  std::int64_t strides_b[kMaxRank];
  std::int64_t rank;
  std::uint64_t mask;
};

LayoutRef make_layout(const std::int64_t* dims, const std::int64_t* a,
                      const std::int64_t* b, std::int64_t rank,
                      std::uint64_t mask = 0) {
  LayoutRef l{};
  l.rank = rank;
  l.mask = mask;
  for (std::int64_t i = 0; i < rank && i < kMaxRank; ++i) {
    l.dims[i] = dims[i];
    l.strides_a[i] = a != nullptr ? a[i] : 0;
    l.strides_b[i] = b != nullptr ? b[i] : 0;
  }
  return l;
}

inline int grid_for(std::int64_t n) {
  const std::int64_t blocks = (n + kThreads - 1) / kThreads;
  return static_cast<int>(blocks > 65535 ? 65535 : (blocks < 1 ? 1 : blocks));
}

template <typename Op>
__device__ inline double block_reduce(double v, double* sh, Op op) {
  const int tid = threadIdx.x;
  sh[tid] = v;
  __syncthreads();
  for (int off = kThreads / 2; off > 0; off >>= 1) {
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

// --- forward kernels --------------------------------------------------------

__global__ void index_select_kernel(float* __restrict__ out,
                                    const float* __restrict__ table,
                                    const int32_t* __restrict__ ids,
                                    std::int64_t rows, std::int64_t dim) {
  const std::int64_t total = rows * dim;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < total; i += stride) {
    const std::int64_t r = i / dim;
    const std::int64_t c = i % dim;
    out[i] = table[static_cast<std::int64_t>(ids[r]) * dim + c];
  }
}

__global__ void matmul_kernel(float* __restrict__ c,
                              const float* __restrict__ a,
                              const float* __restrict__ b, std::int64_t m,
                              std::int64_t k, std::int64_t n) {
  const std::int64_t total = m * n;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < total; i += stride) {
    const std::int64_t r = i / n;
    const std::int64_t col = i % n;
    double s = 0.0;
    for (std::int64_t j = 0; j < k; ++j)
      s += static_cast<double>(a[r * k + j]) *
           static_cast<double>(b[j * n + col]);
    c[i] = static_cast<float>(s);
  }
}

__global__ void bmm_kernel(float* __restrict__ c, const float* __restrict__ a,
                           const float* __restrict__ b, std::int64_t batch,
                           std::int64_t m, std::int64_t k, std::int64_t n) {
  const std::int64_t per = m * n;
  const std::int64_t total = batch * per;
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < total; i += stride) {
    const std::int64_t bi = i / per;
    const std::int64_t r = (i % per) / n;
    const std::int64_t col = i % n;
    const float* ab = a + bi * m * k;
    const float* bb = b + bi * k * n;
    double s = 0.0;
    for (std::int64_t j = 0; j < k; ++j)
      s += static_cast<double>(ab[r * k + j]) *
           static_cast<double>(bb[j * n + col]);
    c[i] = static_cast<float>(s);
  }
}

// out[o] = in[sum_d unravel(o, dims)[d] * strides_a[d]], where strides_a[d] is
// the source stride of output axis d (0 on an axis the source broadcasts).
__global__ void permute_kernel(float* __restrict__ out,
                               const float* __restrict__ in, LayoutRef l,
                               std::int64_t out_numel) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t o = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       o < out_numel; o += stride) {
    std::int64_t rem = o;
    std::int64_t off = 0;
    for (std::int64_t d = l.rank - 1; d >= 0; --d) {
      const std::int64_t dim = l.dims[d];
      const std::int64_t idx = rem % dim;
      rem /= dim;
      off += idx * l.strides_a[d];
    }
    out[o] = in[off];
  }
}

__global__ void softmax_kernel(float* __restrict__ out,
                               const float* __restrict__ in, std::int64_t cols) {
  __shared__ double sh[kThreads];
  const std::int64_t r = blockIdx.x;
  const float* x = in + r * cols;
  float* y = out + r * cols;

  double m = -INFINITY;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    m = fmax(m, static_cast<double>(x[c]));
  m = block_reduce(m, sh, MaxOp{});
  __syncthreads();

  double s = 0.0;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    s += exp(static_cast<double>(x[c]) - m);
  const double sum = block_reduce(s, sh, AddOp{});

  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    y[c] = static_cast<float>(exp(static_cast<double>(x[c]) - m) / sum);
}

__global__ void rms_norm_kernel(float* __restrict__ out,
                                const float* __restrict__ x,
                                const float* __restrict__ w,
                                std::int64_t dim, float eps) {
  __shared__ double sh[kThreads];
  const std::int64_t r = blockIdx.x;
  const float* xr = x + r * dim;
  float* yr = out + r * dim;

  double ss = 0.0;
  for (std::int64_t d = threadIdx.x; d < dim; d += blockDim.x)
    ss += static_cast<double>(xr[d]) * static_cast<double>(xr[d]);
  ss = block_reduce(ss, sh, AddOp{});
  const double rinv =
      1.0 / std::sqrt(ss / static_cast<double>(dim) + static_cast<double>(eps));
  for (std::int64_t d = threadIdx.x; d < dim; d += blockDim.x)
    yr[d] = static_cast<float>(static_cast<double>(xr[d]) * rinv *
                               static_cast<double>(w[d]));
}

__global__ void silu_mul_kernel(float* __restrict__ out,
                                const float* __restrict__ g,
                                const float* __restrict__ u,
                                std::int64_t n) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < n; i += stride) {
    const double gi = static_cast<double>(g[i]);
    const double sig = 1.0 / (1.0 + exp(-gi));
    out[i] = static_cast<float>(gi * sig * static_cast<double>(u[i]));
  }
}

__global__ void ce_row_kernel(const float* __restrict__ logits,
                              const int32_t* __restrict__ targets,
                              float* __restrict__ row_loss, std::int64_t cols) {
  __shared__ double sh_max[kThreads];
  __shared__ double sh_sum[kThreads];
  const std::int64_t r = blockIdx.x;
  const float* x = logits + r * cols;

  double m = -INFINITY;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    m = fmax(m, static_cast<double>(x[c]));
  m = block_reduce(m, sh_max, MaxOp{});
  __syncthreads();

  double s = 0.0;
  for (std::int64_t c = threadIdx.x; c < cols; c += blockDim.x)
    s += exp(static_cast<double>(x[c]) - m);
  const double sum = block_reduce(s, sh_sum, AddOp{});

  if (threadIdx.x == 0) {
    const std::int32_t t = targets[r];
    row_loss[r] = static_cast<float>(m + log(sum) - static_cast<double>(x[t]));
  }
}

__global__ void ce_mean_kernel(const float* __restrict__ row_loss,
                               float* __restrict__ loss, std::int64_t rows) {
  __shared__ double sh[kThreads];
  double s = 0.0;
  for (std::int64_t r = threadIdx.x; r < rows; r += blockDim.x)
    s += static_cast<double>(row_loss[r]);
  const double total = block_reduce(s, sh, AddOp{});
  if (threadIdx.x == 0) loss[0] = static_cast<float>(total / rows);
}

template <bool kMul>
__global__ void ewise_bcast_kernel(float* __restrict__ out,
                                   const float* __restrict__ a,
                                   const float* __restrict__ b, LayoutRef l,
                                   std::int64_t out_numel) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t o = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       o < out_numel; o += stride) {
    std::int64_t rem = o;
    std::int64_t oa = 0;
    std::int64_t ob = 0;
    for (std::int64_t d = l.rank - 1; d >= 0; --d) {
      const std::int64_t dim = l.dims[d];
      const std::int64_t idx = rem % dim;
      rem /= dim;
      oa += idx * l.strides_a[d];
      ob += idx * l.strides_b[d];
    }
    out[o] = kMul ? a[oa] * b[ob] : a[oa] + b[ob];
  }
}

// One thread per output element: walks the reduced subspace in input order.
__global__ void reduce_sum_kernel(float* __restrict__ out,
                                  const float* __restrict__ in, LayoutRef l,
                                  std::int64_t out_numel) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t o = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       o < out_numel; o += stride) {
    std::int64_t idx[kMaxRank];
    std::int64_t red[kMaxRank];
    std::int64_t nred = 0;
    std::int64_t rem = o;
    for (std::int64_t d = l.rank - 1; d >= 0; --d) {
      if ((l.mask >> d) & 1u) {
        idx[d] = 0;
        red[nred++] = d;
      } else {
        idx[d] = rem % l.dims[d];
        rem /= l.dims[d];
      }
    }
    std::int64_t red_total = 1;
    for (std::int64_t k = 0; k < nred; ++k)
      red_total *= l.dims[red[k]];

    double s = 0.0;
    for (std::int64_t t = 0; t < red_total; ++t) {
      std::int64_t tt = t;
      for (std::int64_t k = 0; k < nred; ++k) {
        idx[red[k]] = tt % l.dims[red[k]];
        tt /= l.dims[red[k]];
      }
      std::int64_t off = 0;
      for (std::int64_t d = 0; d < l.rank; ++d)
        off += idx[d] * l.strides_a[d];
      s += static_cast<double>(in[off]);
    }
    out[o] = static_cast<float>(s);
  }
}

__global__ void adamw_kernel(float* __restrict__ p_out, float* __restrict__ m1_out,
                             float* __restrict__ m2_out,
                             const float* __restrict__ p,
                             const float* __restrict__ g,
                             const float* __restrict__ m1,
                             const float* __restrict__ m2,
                             const int32_t* __restrict__ step, std::int64_t n,
                             const float* __restrict__ lr_dev, double lr,
                             double beta1, double beta2, double eps,
                             double weight_decay) {
  // lr_dev, when non-null, is the lr-input form: the rate is read on the
  // device and the host scalar is unused.
  const double rate =
      lr_dev ? static_cast<double>(lr_dev[0]) : lr;
  const double t = static_cast<double>(step[0]);
  const double bc1 = 1.0 / (1.0 - std::pow(beta1, t));
  const double bc2 = 1.0 / (1.0 - std::pow(beta2, t));
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < n; i += stride) {
    const double gi = static_cast<double>(g[i]);
    const double mi = beta1 * static_cast<double>(m1[i]) + (1.0 - beta1) * gi;
    const double vi =
        beta2 * static_cast<double>(m2[i]) + (1.0 - beta2) * gi * gi;
    m1_out[i] = static_cast<float>(mi);
    m2_out[i] = static_cast<float>(vi);
    const double mh = mi * bc1;
    const double vh = vi * bc2;
    const double update = rate * (mh / (std::sqrt(vh) + eps) +
                                  weight_decay * static_cast<double>(p[i]));
    p_out[i] = static_cast<float>(static_cast<double>(p[i]) - update);
  }
}

// sum(x^2): one pass over the tensor, a block reduction in double, then one
// atomic per block into the caller's double accumulator.
__global__ void sumsq_kernel(const float* __restrict__ x, std::int64_t n,
                             double* __restrict__ acc) {
  __shared__ double sh[kThreads];
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  double s = 0.0;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < n; i += stride) {
    const double v = static_cast<double>(x[i]);
    s += v * v;
  }
  s = block_reduce(s, sh, AddOp{});
  if (threadIdx.x == 0) atomicAdd(acc, s);
}

__global__ void scale_copy_kernel(float* __restrict__ out,
                                  const float* __restrict__ in,
                                  std::int64_t n, double scale) {
  const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x +
                        threadIdx.x;
       i < n; i += stride) {
    // scale == 1.0 is the no-clip case: copy so the values stay bit-exact.
    out[i] = scale == 1.0 ? in[i]
                          : static_cast<float>(static_cast<double>(in[i]) * scale);
  }
}

bool ok() { return cudaGetLastError() == cudaSuccess; }

} // namespace

// --- buffers ----------------------------------------------------------------

DevBuf dev_alloc(std::uint64_t bytes) {
  void* p = nullptr;
  if (bytes == 0) bytes = 1;
  if (cudaMalloc(&p, bytes) != cudaSuccess) return 0;
  return reinterpret_cast<DevBuf>(p);
}

void dev_free(DevBuf buf) {
  if (buf != 0) cudaFree(reinterpret_cast<void*>(buf));
}

bool dev_upload(DevBuf dst, const void* src, std::uint64_t bytes) {
  if (bytes == 0) return true;
  if (dst == 0 || src == nullptr) return false;
  return cudaMemcpyAsync(reinterpret_cast<void*>(dst), src, bytes,
                         cudaMemcpyHostToDevice,
                         cur_stream()) == cudaSuccess;
}

bool dev_download(DevBuf src, void* dst, std::uint64_t bytes) {
  if (bytes == 0) return true;
  if (src == 0 || dst == nullptr) return false;
  return cudaMemcpyAsync(dst, reinterpret_cast<const void*>(src), bytes,
                         cudaMemcpyDeviceToHost, cur_stream()) == cudaSuccess;
}

bool dev_zero(DevBuf dst, std::uint64_t bytes) {
  if (bytes == 0) return true;
  if (dst == 0) return false;
  return cudaMemsetAsync(reinterpret_cast<void*>(dst), 0, bytes,
                         cur_stream()) == cudaSuccess;
}

bool dev_copy(DevBuf dst, DevBuf src, std::int64_t n) {
  if (n < 0) return false;
  if (n == 0) return true;
  return cudaMemcpyAsync(reinterpret_cast<void*>(dst),
                         reinterpret_cast<const void*>(src),
                         static_cast<std::size_t>(n) * sizeof(float),
                         cudaMemcpyDeviceToDevice, cur_stream()) == cudaSuccess;
}

bool dev_sync() { return cudaStreamSynchronize(cur_stream()) == cudaSuccess; }

// --- forward ----------------------------------------------------------------

bool graph_index_select(DevBuf out, DevBuf table, DevBuf ids, std::int64_t rows,
                        std::int64_t dim) {
  if (rows <= 0 || dim <= 0) return rows == 0 || dim == 0;
  index_select_kernel<<<grid_for(rows * dim), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(table), ptr<const int32_t>(ids), rows,
      dim);
  return ok();
}

bool graph_matmul(DevBuf c, DevBuf a, DevBuf b, std::int64_t m,
                  std::int64_t k, std::int64_t n) {
  if (m <= 0 || k <= 0 || n <= 0) return false;
  matmul_kernel<<<grid_for(m * n), kThreads, 0, cur_stream()>>>(
      ptr<float>(c), ptr<const float>(a), ptr<const float>(b), m, k, n);
  return ok();
}

bool graph_bmm(DevBuf c, DevBuf a, DevBuf b, std::int64_t batch,
               std::int64_t m, std::int64_t k, std::int64_t n) {
  if (batch <= 0 || m <= 0 || k <= 0 || n <= 0) return false;
  bmm_kernel<<<grid_for(batch * m * n), kThreads, 0, cur_stream()>>>(
      ptr<float>(c), ptr<const float>(a), ptr<const float>(b), batch, m, k, n);
  return ok();
}

bool graph_permute(DevBuf out, DevBuf in, const std::int64_t* src_strides,
                   std::int64_t rank, std::int64_t out_numel,
                   const std::int64_t* out_dims) {
  if (rank < 0 || rank > kMaxRank || out_numel < 0) return false;
  if (out_numel == 0) return true;
  const LayoutRef l = make_layout(out_dims, src_strides, nullptr, rank);
  permute_kernel<<<grid_for(out_numel), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(in), l, out_numel);
  return ok();
}

bool graph_softmax(DevBuf out, DevBuf in, std::int64_t rows,
                   std::int64_t cols) {
  if (rows <= 0 || cols <= 0) return false;
  softmax_kernel<<<static_cast<unsigned>(rows), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(in), cols);
  return ok();
}

bool graph_rms_norm(DevBuf out, DevBuf x, DevBuf w, std::int64_t rows,
                    std::int64_t dim, float eps) {
  if (rows <= 0 || dim <= 0) return false;
  rms_norm_kernel<<<static_cast<unsigned>(rows), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(x), ptr<const float>(w), dim, eps);
  return ok();
}

bool graph_rope_inplace(DevBuf x, std::int64_t rows, std::int64_t heads,
                        std::int64_t head_dim, std::uint64_t pos0,
                        float theta_base, float freq_scale) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0) return false;
  return ::sonicboom::nn::cuda::rope_neox_heads_batched_dev(
      ptr<float>(x), static_cast<int>(head_dim), static_cast<int>(heads),
      static_cast<int>(rows), pos0, theta_base, freq_scale, {});
}

bool graph_silu_mul(DevBuf out, DevBuf g, DevBuf u, std::int64_t n) {
  if (n < 0) return false;
  if (n == 0) return true;
  silu_mul_kernel<<<grid_for(n), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(g), ptr<const float>(u), n);
  return ok();
}

bool graph_cross_entropy(DevBuf loss, DevBuf logits, DevBuf targets,
                         std::int64_t rows, std::int64_t cols) {
  if (rows <= 0 || cols <= 0) return false;
  DevBuf scratch = dev_alloc(static_cast<std::uint64_t>(rows) * sizeof(float));
  if (scratch == 0) return false;
  ce_row_kernel<<<static_cast<unsigned>(rows), kThreads, 0, cur_stream()>>>(
      ptr<const float>(logits), ptr<const int32_t>(targets), ptr<float>(scratch),
      cols);
  ce_mean_kernel<<<1, kThreads, 0, cur_stream()>>>(ptr<const float>(scratch),
                                                   ptr<float>(loss), rows);
  const bool good = ok();
  dev_free(scratch);
  return good;
}

bool graph_add(DevBuf out, DevBuf a, DevBuf b, const std::int64_t* a_strides,
               const std::int64_t* b_strides, const std::int64_t* out_dims,
               std::int64_t rank, std::int64_t out_numel) {
  if (out_numel < 0 || rank < 0 || rank > kMaxRank) return false;
  if (out_numel == 0) return true;
  const LayoutRef l = make_layout(out_dims, a_strides, b_strides, rank);
  ewise_bcast_kernel<false>
      <<<grid_for(out_numel), kThreads, 0, cur_stream()>>>(
          ptr<float>(out), ptr<const float>(a), ptr<const float>(b), l,
          out_numel);
  return ok();
}

bool graph_mul(DevBuf out, DevBuf a, DevBuf b, const std::int64_t* a_strides,
               const std::int64_t* b_strides, const std::int64_t* out_dims,
               std::int64_t rank, std::int64_t out_numel) {
  if (out_numel < 0 || rank < 0 || rank > kMaxRank) return false;
  if (out_numel == 0) return true;
  const LayoutRef l = make_layout(out_dims, a_strides, b_strides, rank);
  ewise_bcast_kernel<true>
      <<<grid_for(out_numel), kThreads, 0, cur_stream()>>>(
          ptr<float>(out), ptr<const float>(a), ptr<const float>(b), l,
          out_numel);
  return ok();
}

bool graph_reduce_sum(DevBuf out, DevBuf in, const std::int64_t* in_dims,
                      const std::int64_t* in_strides,
                      const std::int64_t* out_dims, std::int64_t rank,
                      std::uint64_t axes_mask, std::int64_t out_numel) {
  if (out_numel < 0 || rank < 0 || rank > kMaxRank) return false;
  if (out_numel == 0) return true;
  const LayoutRef l =
      make_layout(in_dims, in_strides, nullptr, rank, axes_mask);
  reduce_sum_kernel<<<grid_for(out_numel), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(in), l, out_numel);
  return ok();
}

bool graph_adamw_step(DevBuf param_out, DevBuf m1_out, DevBuf m2_out,
                      DevBuf param, DevBuf grad, DevBuf m1, DevBuf m2,
                      std::int64_t n, DevBuf step, float lr, float beta1,
                      float beta2, float eps, float weight_decay) {
  if (n <= 0) return false;
  adamw_kernel<<<grid_for(n), kThreads, 0, cur_stream()>>>(
      ptr<float>(param_out), ptr<float>(m1_out), ptr<float>(m2_out),
      ptr<const float>(param), ptr<const float>(grad), ptr<const float>(m1),
      ptr<const float>(m2), ptr<const int32_t>(step), n, nullptr,
      static_cast<double>(lr), static_cast<double>(beta1),
      static_cast<double>(beta2), static_cast<double>(eps),
      static_cast<double>(weight_decay));
  return ok();
}

bool graph_adamw_step_lr(DevBuf param_out, DevBuf m1_out, DevBuf m2_out,
                         DevBuf param, DevBuf grad, DevBuf m1, DevBuf m2,
                         std::int64_t n, DevBuf step, DevBuf lr,
                         float beta1, float beta2, float eps,
                         float weight_decay) {
  if (n <= 0) return false;
  adamw_kernel<<<grid_for(n), kThreads, 0, cur_stream()>>>(
      ptr<float>(param_out), ptr<float>(m1_out), ptr<float>(m2_out),
      ptr<const float>(param), ptr<const float>(grad), ptr<const float>(m1),
      ptr<const float>(m2), ptr<const int32_t>(step), n,
      ptr<const float>(lr), 0.0, static_cast<double>(beta1),
      static_cast<double>(beta2), static_cast<double>(eps),
      static_cast<double>(weight_decay));
  return ok();
}

bool graph_sumsq_acc(DevBuf acc, DevBuf x, std::int64_t n) {
  if (n <= 0) return true; // an empty gradient adds nothing
  sumsq_kernel<<<grid_for(n), kThreads, 0, cur_stream()>>>(
      ptr<const float>(x), n, ptr<double>(acc));
  return ok();
}

bool graph_scale_copy(DevBuf out, DevBuf in, std::int64_t n, double scale) {
  if (n <= 0) return true;
  scale_copy_kernel<<<grid_for(n), kThreads, 0, cur_stream()>>>(
      ptr<float>(out), ptr<const float>(in), n, scale);
  return ok();
}

} // namespace sonicboom::training::detail
