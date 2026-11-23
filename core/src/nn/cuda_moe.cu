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

#include <sonicboom/nn/cuda_moe.h>

#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "cuda_resident.h"
#include "cuda_scratch_epoch.h"
#include "cuda_stream.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace sonicboom::nn::cuda {

namespace {

// The lower bound on the routed weight denominator (2^-14); see nn::moe_route.
constexpr float kMoeWeightSumFloor = 6.103515625e-5f;

// One row's routing, the same arithmetic as nn::moe_route after the logits:
// stable softmax over all experts in ascending index (f32), top-k with a strict
// comparison (a tie keeps the lower index), then the ascending renormalisation.
bool route_row(std::span<const float> x, const quant::QuantizedTensor* router,
               std::span<const float> router_f32, uint32_t experts,
               uint32_t used, float weight_scale, std::span<int64_t> ids,
               std::span<float> weights) {
  std::vector<float> logits(experts, 0.0f);
  const uint64_t d = x.size();
  if (!router_f32.empty()) {
    if (router_f32.size() < uint64_t(experts) * d) return false;
    if (!matvec_f32(router_f32.first(uint64_t(experts) * d), d, x, logits))
      return false;
  } else if (router && router->valid()) {
    if (!quant::cuda::matvec_f32(*router, x, logits)) return false;
  } else {
    return false;
  }

  float mx = logits[0];
  for (float l : logits) mx = std::max(mx, l);
  float sum = 0.0f;
  std::vector<float> probs(experts, 0.0f);
  for (uint32_t e = 0; e < experts; ++e) {
    probs[e] = std::exp(logits[e] - mx);
    sum += probs[e];
  }
  if (!(sum > 0.0f)) return false;
  const float inv = 1.0f / sum;
  for (uint32_t e = 0; e < experts; ++e) probs[e] *= inv;

  std::vector<bool> taken(experts, false);
  std::vector<uint32_t> selected;
  selected.reserve(used);
  for (uint32_t k = 0; k < used; ++k) {
    uint32_t best = experts;  // none yet
    for (uint32_t e = 0; e < experts; ++e) {
      if (taken[e]) continue;
      if (best == experts || probs[e] > probs[best]) best = e;
    }
    if (best == experts) return false;
    taken[best] = true;
    selected.push_back(best);
  }

  std::sort(selected.begin(), selected.end());
  float denom = 0.0f;
  for (uint32_t e : selected) denom += probs[e];
  denom = std::max(denom, kMoeWeightSumFloor);
  for (uint32_t k = 0; k < used; ++k) {
    ids[k] = int64_t(selected[k]);
    weights[k] = probs[selected[k]] / denom * weight_scale;
  }
  return true;
}

// A pooled matrix's address: the role's array plus the slot's stride. Only the
// address differs from the uploading path.
const void* pooled(const void* bytes, uint64_t stride, int32_t slot) {
  return static_cast<const char*>(bytes) + uint64_t(slot) * stride;
}

// --- device-resident form ----------------------------------------------------

int moe_grid(int64_t n) { return int((n + 255) / 256); }

// Process-global workspace for the device-resident experts. Buffers are grown
// as the model's geometry requires and kept, like the other kernel scratch (see
// cuda_scratch_epoch.h: a live capture may still hold an older pointer).
enum ScratchSlot {
  kRouteLogits = 0,  // [rows, experts]
  kDecodeOut,        // [rows, k, d]
  kA,                // decode gate; prefill X
  kB,                // decode up; prefill G
  kC,                // prefill U
  kD,                // prefill down result
  kRowOf,            // prefill int32 row per (expert, row)
  kKwOf,             // prefill int32 r*k+j per entry (weights_dev index)
  kOffsets,          // prefill int32 per-expert start (e_count + 1)
  kWeightOf,         // prefill f32 weight per entry
  kScratchSlots
};

struct Scratch {
  void* ptr = nullptr;
  size_t bytes = 0;
};
Scratch g_scratch[kScratchSlots];

// Old buffers a growth replaced while a capture was live. They cannot be freed
// then (a recorded graph may launch against them) and must not be dropped
// either; moe_release_scratch() frees them at model teardown.
std::vector<std::pair<void*, size_t>> g_retired;

void* scratch(int slot, size_t need) {
  Scratch& s = g_scratch[slot];
  if (s.bytes >= need) return s.ptr;
  sonicboom::cuda::scratch_epoch_bump();
  void* old = s.ptr;
  if (old && sonicboom::cuda::scratch_may_free()) {
    cudaFree(old);
    old = nullptr;
  }
  void* p = nullptr;
  if (cudaMalloc(&p, need) != cudaSuccess) {
    s.ptr = old;
    return nullptr;
  }
  if (old) g_retired.emplace_back(old, s.bytes);
  s.ptr = p;
  s.bytes = need;
  return p;
}

void moe_release_scratch_impl() {
  if (!sonicboom::cuda::scratch_may_free()) return;
  for (auto& s : g_scratch) {
    if (s.ptr) cudaFree(s.ptr);
    s.ptr = nullptr;
    s.bytes = 0;
  }
  for (auto& r : g_retired) cudaFree(r.first);
  g_retired.clear();
}

__device__ inline float moe_half_roundtrip(float v) {
  return __half2float(__float2half_rn(v));
}

// ggml's ggml_gelu_f32, the same constants and order as the CPU's
// gelu_tanh_f32 and the elementwise kernel's device copy.
__device__ inline float moe_gelu_tanh(float x) {
  constexpr float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
  constexpr float gelu_coef_a = 0.044715f;
  return 0.5f * x *
         (1.0f + tanhf(sqrt_2_over_pi * x * (1.0f + gelu_coef_a * x * x)));
}

// h[i] = act(gate[i]) * up[i], the host form's silu/gelu_fp16 followed by the
// elementwise multiply, with the same per-element operations (expf, fp16
// round-trip around the tanh) so the result matches the host step bit for bit.
__global__ void moe_act_mul_kernel(const float* __restrict__ gate,
                                   const float* __restrict__ up,
                                   float* __restrict__ h, int64_t n,
                                   int gelu) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = gate[i];
  float a;
  if (gelu) {
    if (v <= -10.0f)
      a = 0.0f;
    else if (v >= 10.0f)
      a = v;
    else
      a = moe_half_roundtrip(moe_gelu_tanh(moe_half_roundtrip(v)));
  } else {
    a = v / (1.0f + expf(-v));
  }
  h[i] = a * up[i];
}

// One row per block, thread 0: the post-logit half of route_row. logits is the
// device scratch buffer, overwritten with the probabilities in place.
__global__ void moe_route_topk_kernel(float* __restrict__ logits,
                                      uint32_t experts, uint32_t used,
                                      float weight_scale,
                                      int64_t* __restrict__ ids,
                                      float* __restrict__ weights) {
  extern __shared__ unsigned char smem[];
  if (threadIdx.x != 0) return;
  const uint64_t r = blockIdx.x;
  float* p = logits + r * experts;
  unsigned char* taken = smem;
  uint32_t* selected = reinterpret_cast<uint32_t*>(
      smem + ((uint64_t(experts) + 3u) & ~uint64_t(3)));
  int64_t* idrow = ids + r * used;
  float* wrow = weights + r * used;

  float mx = p[0];
  for (uint32_t e = 1; e < experts; ++e) mx = fmaxf(mx, p[e]);
  float sum = 0.0f;
  for (uint32_t e = 0; e < experts; ++e) {
    p[e] = expf(p[e] - mx);
    sum += p[e];
  }
  if (!(sum > 0.0f)) {
    for (uint32_t j = 0; j < used; ++j) {
      idrow[j] = -1;
      wrow[j] = 0.0f;
    }
    return;
  }
  const float inv = 1.0f / sum;
  for (uint32_t e = 0; e < experts; ++e) {
    p[e] *= inv;
    taken[e] = 0;
  }
  // Top-k with a strict comparison, so a tie keeps the lower index.
  for (uint32_t j = 0; j < used; ++j) {
    uint32_t best = experts;
    for (uint32_t e = 0; e < experts; ++e) {
      if (taken[e]) continue;
      if (best == experts || p[e] > p[best]) best = e;
    }
    if (best == experts) {
      for (uint32_t q = 0; q < used; ++q) {
        idrow[q] = -1;
        wrow[q] = 0.0f;
      }
      return;
    }
    taken[best] = 1;
    selected[j] = best;
  }
  // Ascending order, as std::sort gives on the host (ties cannot occur here —
  // the selected indices are distinct).
  for (uint32_t a = 1; a < used; ++a) {
    const uint32_t v = selected[a];
    int32_t b = int32_t(a) - 1;
    while (b >= 0 && selected[b] > v) {
      selected[b + 1] = selected[b];
      --b;
    }
    selected[b + 1] = v;
  }
  float denom = 0.0f;
  for (uint32_t j = 0; j < used; ++j) denom += p[selected[j]];
  denom = fmaxf(denom, kMoeWeightSumFloor);
  for (uint32_t j = 0; j < used; ++j) {
    idrow[j] = int64_t(selected[j]);
    wrow[j] = p[selected[j]] / denom * weight_scale;
  }
}

// Decode / verify: y[r, i] = sum over j of w[r, j] * out[r, j, i], j ascending.
// The explicit intrinsic mul/add keep the compiler from contracting the pair
// into an FMA — the host form's rounding is a rounded multiply then a rounded
// add.
__global__ void moe_weighted_sum_kernel(const float* __restrict__ out,
                                        const float* __restrict__ w,
                                        float* __restrict__ y, int64_t rows,
                                        int64_t k, int64_t d) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= rows * d) return;
  const int64_t r = idx / d;
  const int64_t i = idx - r * d;
  float acc = 0.0f;
  for (int64_t j = 0; j < k; ++j)
    acc = __fadd_rn(acc,
                    __fmul_rn(w[r * k + j], out[(r * k + j) * d + i]));
  y[idx] = acc;
}

// Prefill: gather the selected rows of one expert into a position-major
// [nnz, d] activation.
__global__ void moe_gather_kernel(const float* __restrict__ x,
                                  const int32_t* __restrict__ row_of,
                                  float* __restrict__ X, int64_t nnz,
                                  int64_t d) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nnz * d) return;
  const int64_t c = idx / d;
  const int64_t i = idx - c * d;
  X[idx] = x[int64_t(row_of[c]) * d + i];
}

// Prefill: the weight of each gathered entry, read from the routing node's
// device weights by the flat (row, selected) index.
__global__ void moe_gather_weights_kernel(const float* __restrict__ weights_dev,
                                          const int32_t* __restrict__ kw_of,
                                          float* __restrict__ w_of,
                                          int64_t nnz) {
  const int64_t c = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (c >= nnz) return;
  w_of[c] = weights_dev[kw_of[c]];
}

// Prefill: y[row, i] += w * D[c, i] for one expert, with the explicit
// multiply/add pair of the host loop.
__global__ void moe_scatter_kernel(const float* __restrict__ D,
                                   const int32_t* __restrict__ row_of,
                                   const float* __restrict__ w_of,
                                   float* __restrict__ y, int64_t nnz,
                                   int64_t d) {
  const int64_t idx = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= nnz * d) return;
  const int64_t c = idx / d;
  const int64_t i = idx - c * d;
  const int64_t o = int64_t(row_of[c]) * d + i;
  y[o] = __fadd_rn(y[o], __fmul_rn(w_of[c], D[idx]));
}

// One expert's routed FFN for one row: gate and up through the decode gemv,
// the activation, then down. `out` receives the expert's d values, unwritten
// on failure. With `pool` set the matrices come from the pool's slot `slot`;
// the type and shape still come from the block's views.
bool expert_row(std::span<const float> x, const MoeExpertWeights& experts,
                uint64_t e, MoeActivation activation, std::span<float> out,
                std::vector<float>& gate, std::vector<float>& up,
                std::vector<float>& hidden, const MoeExpertPool* pool,
                int32_t slot) {
  const quant::QuantizedTensor& gw = experts.gate[e];
  const quant::QuantizedTensor& uw = experts.up[e];
  const quant::QuantizedTensor& dw = experts.down[e];
  if (!gw.valid() || !uw.valid() || !dw.valid()) return false;
  const uint64_t ff = gw.dims().size() > 1 ? gw.dims()[1] : 0;
  if (ff == 0 || out.size() < dw.dims()[1]) return false;
  gate.assign(ff, 0.0f);
  up.assign(ff, 0.0f);
  hidden.assign(ff, 0.0f);
  bool ok = false;
  if (pool)
    ok = quant::cuda::matvec_f32_device_weight(
             gw, pooled(pool->gate_bytes, pool->gate_stride, slot), x, gate) &&
         quant::cuda::matvec_f32_device_weight(
             uw, pooled(pool->up_bytes, pool->up_stride, slot), x, up);
  else
    ok = quant::cuda::matvec_f32(gw, x, gate) &&
         quant::cuda::matvec_f32(uw, x, up);
  if (!ok) return false;
  const bool act_ok =
      activation == MoeActivation::Silu ? silu(gate, gate) : gelu_fp16(gate, gate);
  if (!act_ok) return false;
  for (uint64_t i = 0; i < ff; ++i) hidden[i] = gate[i] * up[i];
  std::span<float> down = out.first(dw.dims()[1]);
  if (pool)
    return quant::cuda::matvec_f32_device_weight(
        dw, pooled(pool->down_bytes, pool->down_stride, slot), hidden, down);
  return quant::cuda::matvec_f32(dw, hidden, down);
}

} // namespace

bool moe_route(std::span<const float> x, const quant::QuantizedTensor* router,
               std::span<const float> router_f32, uint32_t experts,
               uint32_t used, float weight_scale, std::span<int64_t> ids,
               std::span<float> weights, uint64_t rows) {
  if (rows == 0 || experts == 0 || used == 0 || used > experts || x.empty() ||
      x.size() % rows != 0)
    return false;
  const uint64_t d = x.size() / rows;
  if (ids.size() < rows * used || weights.size() < rows * used) return false;
  for (uint64_t r = 0; r < rows; ++r) {
    if (!route_row(x.subspan(r * d, d), router, router_f32, experts, used,
                   weight_scale, ids.subspan(r * used, used),
                   weights.subspan(r * used, used)))
      return false;
  }
  return true;
}

bool moe_route_dev(const float* x, const quant::QuantizedTensor* router,
                   std::span<const float> router_f32, uint32_t experts,
                   uint32_t used, float weight_scale, int64_t* ids,
                   float* weights, uint64_t rows) {
  if (!quant::cuda::available() || rows == 0 || experts == 0 || used == 0 ||
      used > experts || !x || !ids || !weights)
    return false;
  uint64_t d = 0;
  if (!router_f32.empty()) {
    if (router_f32.size() % experts != 0) return false;
    d = router_f32.size() / experts;
  } else if (router && router->valid()) {
    if (router->dims().size() != 2 || router->dims()[1] != experts) return false;
    d = router->dims()[0];
  } else {
    return false;
  }
  if (d == 0) return false;

  float* logits = static_cast<float*>(
      scratch(kRouteLogits, rows * experts * sizeof(float)));
  if (!logits) return false;
  for (uint64_t r = 0; r < rows; ++r) {
    if (!router_f32.empty()) {
      // The dense f32 router's device gemv; the weight is uploaded and cached
      // by the host pointer, as in matvec_f32.
      if (!nn::cuda::matvec_f32_dev(router_f32, int(d), x + r * d,
                                    logits + r * experts, int(experts), 1))
        return false;
    } else if (!quant::cuda::matvec_f32_dev(*router, x + r * d,
                                            logits + r * experts)) {
      return false;
    }
  }

  const size_t smem =
      ((size_t(experts) + 3u) & ~size_t(3)) + size_t(used) * sizeof(uint32_t);
  if (smem > 48u * 1024u) return false;
  moe_route_topk_kernel<<<rows, 1, smem, sonicboom::cuda::boom_current_stream()>>>(
      logits, experts, used, weight_scale, ids, weights);
  return cudaGetLastError() == cudaSuccess;
}

bool moe_experts(std::span<const float> x, std::span<const int64_t> ids,
                 std::span<const float> weights,
                 const MoeExpertWeights& experts, MoeActivation activation,
                 std::span<float> y, uint64_t rows, bool group_by_expert,
                 const MoeExpertPool* pool) {
  if (rows == 0 || x.empty() || x.size() % rows != 0 ||
      ids.size() != weights.size() || ids.size() % rows != 0 ||
      experts.gate.size() != experts.up.size() ||
      experts.gate.size() != experts.down.size())
    return false;
  const uint64_t d = x.size() / rows;
  const uint64_t k = ids.size() / rows;
  if (y.size() < x.size()) return false;
  const uint64_t e_count = experts.gate.size();
  for (int64_t id : ids)
    if (id < 0 || uint64_t(id) >= e_count) return false;
  if (k == 0) return true;

  // The pool wrote one slot per id; read it once (the backend has already
  // synchronised on the ids themselves).
  std::vector<int32_t> slots;
  if (pool) {
    if (!pool->slots) return false;
    slots.resize(ids.size());
    if (cudaMemcpy(slots.data(), pool->slots,
                   ids.size() * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
  }

  if (!group_by_expert) {
    // Decode / verify: one row at a time, each selected expert through the
    // decode gemv in the order the ids give them (ascending expert index).
    std::vector<float> gate, up, hidden, one(d);
    for (uint64_t r = 0; r < rows; ++r) {
      std::span<float> yrow = y.subspan(r * d, d);
      for (uint64_t j = 0; j < k; ++j) {
        const uint64_t e = uint64_t(ids[r * k + j]);
        const int32_t slot = pool ? slots[r * k + j] : 0;
        if (!expert_row(x.subspan(r * d, d), experts, e, activation, one, gate,
                        up, hidden, pool, slot))
          return false;
        const float w = weights[r * k + j];
        for (uint64_t i = 0; i < d; ++i) yrow[i] += w * one[i];
      }
    }
    return true;
  }

  // Prefill: group the rows by expert (ascending), one batched projection per
  // expert over the rows that selected it, scattered back into the rows'
  // accumulators. Experts are visited in ascending index, so each row still
  // accumulates in that order.
  std::vector<uint64_t> rows_of;
  for (uint64_t e = 0; e < e_count; ++e) {
    rows_of.clear();
    std::vector<float> w_row;  // this expert's weight for each gathered row
    int32_t slot = 0;          // the pool slot of this expert
    for (uint64_t r = 0; r < rows; ++r) {
      for (uint64_t j = 0; j < k; ++j) {
        if (uint64_t(ids[r * k + j]) == e) {
          rows_of.push_back(r);
          w_row.push_back(weights[r * k + j]);
          if (pool) slot = slots[r * k + j];
          break;
        }
      }
    }
    if (rows_of.empty()) continue;
    const uint64_t nnz = rows_of.size();
    const quant::QuantizedTensor& gw = experts.gate[e];
    const quant::QuantizedTensor& uw = experts.up[e];
    const quant::QuantizedTensor& dw = experts.down[e];
    if (!gw.valid() || !uw.valid() || !dw.valid()) return false;
    const uint64_t ff = gw.dims().size() > 1 ? gw.dims()[1] : 0;
    if (ff == 0) return false;
    // The batched matmul takes a position-major activation: X is [n, cols] and
    // Y is [n, rows], each position's row contiguous. Gather the selected rows
    // and scatter the results back in that layout.
    std::vector<float> X(d * nnz);
    for (uint64_t c = 0; c < nnz; ++c)
      for (uint64_t i = 0; i < d; ++i) X[c * d + i] = x[rows_of[c] * d + i];
    std::vector<float> G(ff * nnz), U(ff * nnz), H(ff * nnz), D(d * nnz);
    if (pool) {
      if (!quant::cuda::matmul_f32_device_weight(
              gw, pooled(pool->gate_bytes, pool->gate_stride, slot), X, G,
              nnz))
        return false;
      if (!quant::cuda::matmul_f32_device_weight(
              uw, pooled(pool->up_bytes, pool->up_stride, slot), X, U, nnz))
        return false;
    } else {
      if (!quant::cuda::matmul_f32(gw, X, G, nnz)) return false;
      if (!quant::cuda::matmul_f32(uw, X, U, nnz)) return false;
    }
    const bool act_ok = activation == MoeActivation::Silu
                            ? silu(G, G)
                            : gelu_fp16(G, G);
    if (!act_ok) return false;
    for (uint64_t i = 0; i < ff * nnz; ++i) H[i] = G[i] * U[i];
    if (pool) {
      if (!quant::cuda::matmul_f32_device_weight(
              dw, pooled(pool->down_bytes, pool->down_stride, slot), H, D,
              nnz))
        return false;
    } else if (!quant::cuda::matmul_f32(dw, H, D, nnz)) {
      return false;
    }
    for (uint64_t c = 0; c < nnz; ++c) {
      std::span<float> yrow = y.subspan(rows_of[c] * d, d);
      const float w = w_row[c];
      for (uint64_t i = 0; i < d; ++i) yrow[i] += w * D[c * d + i];
    }
  }
  return true;
}

bool moe_experts_dev(const float* x, const int64_t* ids_host,
                     const float* weights_dev, const int32_t* slots_dev,
                     const int32_t* slots_host,
                     const MoeExpertWeights& experts, MoeActivation activation,
                     float* y, uint64_t rows, uint64_t k, uint64_t d,
                     bool group_by_expert, const MoeExpertPool& pool,
                     void* stream) {
  if (!quant::cuda::available() || rows == 0 || k == 0 || d == 0 || !x ||
      !ids_host || !weights_dev || !y || !slots_host)
    return false;
  if (experts.gate.size() != experts.up.size() ||
      experts.gate.size() != experts.down.size() ||
      experts.gate.empty())
    return false;
  if (!pool.gate_bytes || !pool.up_bytes || !pool.down_bytes) return false;
  (void)slots_dev;  // the pool's device buffer; the host copy drives the launches

  const uint64_t e_count = experts.gate.size();
  uint64_t ff_max = 0;
  for (uint64_t e = 0; e < e_count; ++e) {
    const quant::QuantizedTensor& gw = experts.gate[e];
    if (!gw.valid() || gw.dims().size() != 2) return false;
    ff_max = std::max(ff_max, gw.dims()[1]);
  }
  if (ff_max == 0) return false;
  for (uint64_t i = 0; i < rows * k; ++i)
    if (ids_host[i] < 0 || uint64_t(ids_host[i]) >= e_count) return false;

  const cudaStream_t s =
      stream ? static_cast<cudaStream_t>(stream) : sonicboom::cuda::boom_current_stream();
  sonicboom::cuda::StreamScope scope(s);
  const int gelu = activation == MoeActivation::GeluFp16 ? 1 : 0;

  if (!group_by_expert) {
    // Decode / verify: one (row, expert) at a time, the decode gemv per
    // projection, the activation and the multiply fused into one kernel whose
    // per-element arithmetic is the host silu/gelu_fp16 + multiply.
    float* out = static_cast<float*>(
        scratch(kDecodeOut, rows * k * d * sizeof(float)));
    float* gate =
        static_cast<float*>(scratch(kA, ff_max * sizeof(float)));
    float* up = static_cast<float*>(scratch(kB, ff_max * sizeof(float)));
    if (!out || !gate || !up) return false;
    for (uint64_t r = 0; r < rows; ++r) {
      for (uint64_t j = 0; j < k; ++j) {
        const uint64_t e = uint64_t(ids_host[r * k + j]);
        const int32_t slot = slots_host[r * k + j];
        float* out_slice = out + (r * k + j) * d;
        if (slot < 0) {
          // SONICBOOM_MOE_CPU_MISSES: the caller computes this expert on the
          // host. Leave a zero here so the weighted sum contributes nothing
          // for it; the caller adds the host result to the sum afterwards.
          if (cudaMemsetAsync(out_slice, 0, d * sizeof(float), s) !=
              cudaSuccess)
            return false;
          continue;
        }
        const quant::QuantizedTensor& gw = experts.gate[e];
        const quant::QuantizedTensor& uw = experts.up[e];
        const quant::QuantizedTensor& dw = experts.down[e];
        if (!uw.valid() || !dw.valid()) return false;
        const uint64_t ff = gw.dims()[1];
        if (ff == 0 || ff > ff_max) return false;
        const float* xr = x + r * d;
        if (!quant::cuda::matvec_f32_dev_device_weight(
                gw, pooled(pool.gate_bytes, pool.gate_stride, slot), xr, gate) ||
            !quant::cuda::matvec_f32_dev_device_weight(
                uw, pooled(pool.up_bytes, pool.up_stride, slot), xr, up))
          return false;
        const int ff_i = int(ff);
        moe_act_mul_kernel<<<moe_grid(ff_i), 256, 0, s>>>(
            gate, up, gate, ff_i, gelu);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (!quant::cuda::matvec_f32_dev_device_weight(
                dw, pooled(pool.down_bytes, pool.down_stride, slot), gate,
                out_slice))
          return false;
      }
    }
    if (rows * d > uint64_t(INT32_MAX)) return false;
    moe_weighted_sum_kernel<<<moe_grid(int64_t(rows * d)), 256, 0, s>>>(
        out, weights_dev, y, int64_t(rows), int64_t(k), int64_t(d));
    return cudaGetLastError() == cudaSuccess;
  }

  // Prefill: group the rows by expert in ascending expert order, exactly as the
  // host form builds its lists (row ascending, the first matching selection).
  std::vector<int32_t> row_of;
  std::vector<int32_t> kw_of;
  std::vector<int32_t> offsets(e_count + 1, 0);
  row_of.reserve(rows * k);
  kw_of.reserve(rows * k);
  for (uint64_t e = 0; e < e_count; ++e) {
    offsets[e] = int32_t(row_of.size());
    for (uint64_t r = 0; r < rows; ++r) {
      for (uint64_t j = 0; j < k; ++j) {
        if (uint64_t(ids_host[r * k + j]) != e) continue;
        row_of.push_back(int32_t(r));
        kw_of.push_back(int32_t(r * k + j));
        break;
      }
    }
  }
  offsets[e_count] = int32_t(row_of.size());
  const uint64_t nnz = row_of.size();
  if (nnz == 0) return true;
  if (nnz > uint64_t(INT32_MAX) || rows * k > uint64_t(INT32_MAX))
    return false;

  int32_t* d_row = static_cast<int32_t*>(
      scratch(kRowOf, nnz * sizeof(int32_t)));
  int32_t* d_kw = static_cast<int32_t*>(
      scratch(kKwOf, nnz * sizeof(int32_t)));
  int32_t* d_off = static_cast<int32_t*>(
      scratch(kOffsets, (e_count + 1) * sizeof(int32_t)));
  float* d_w = static_cast<float*>(
      scratch(kWeightOf, nnz * sizeof(float)));
  float* X = static_cast<float*>(scratch(kA, rows * d * sizeof(float)));
  float* G = static_cast<float*>(scratch(kB, rows * ff_max * sizeof(float)));
  float* U = static_cast<float*>(scratch(kC, rows * ff_max * sizeof(float)));
  float* D = static_cast<float*>(scratch(kD, rows * d * sizeof(float)));
  if (!d_row || !d_kw || !d_off || !d_w || !X || !G || !U || !D) return false;
  if (cudaMemcpy(d_row, row_of.data(), nnz * sizeof(int32_t),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_kw, kw_of.data(), nnz * sizeof(int32_t),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(d_off, offsets.data(), (e_count + 1) * sizeof(int32_t),
                 cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  moe_gather_weights_kernel<<<moe_grid(int64_t(nnz)), 256, 0, s>>>(
      weights_dev, d_kw, d_w, int64_t(nnz));
  if (cudaGetLastError() != cudaSuccess) return false;

  // One expert at a time on this stream, ascending: a row's accumulation order
  // is the expert order, so no atomics are needed.
  for (uint64_t e = 0; e < e_count; ++e) {
    const uint64_t begin = uint64_t(offsets[e]);
    const uint64_t n_e = uint64_t(offsets[e + 1]) - begin;
    if (n_e == 0) continue;
    const quant::QuantizedTensor& gw = experts.gate[e];
    const quant::QuantizedTensor& uw = experts.up[e];
    const quant::QuantizedTensor& dw = experts.down[e];
    if (!uw.valid() || !dw.valid()) return false;
    const uint64_t ff = gw.dims()[1];
    if (ff == 0 || ff > ff_max) return false;
    // Prefill uploads every miss: a negative slot here is a caller bug, not a
    // host-computed expert.
    const int32_t slot = slots_host[kw_of[begin]];
    if (slot < 0) return false;
    const int n_i = int(n_e);
    moe_gather_kernel<<<moe_grid(int64_t(n_e * d)), 256, 0, s>>>(
        x, d_row + begin, X, int64_t(n_e), int64_t(d));
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!quant::cuda::matmul_f32_dev_device_weight(
            gw, pooled(pool.gate_bytes, pool.gate_stride, slot), X, G, n_i) ||
        !quant::cuda::matmul_f32_dev_device_weight(
            uw, pooled(pool.up_bytes, pool.up_stride, slot), X, U, n_i))
      return false;
    const int ff_i = int(ff * n_e);
    moe_act_mul_kernel<<<moe_grid(ff_i), 256, 0, s>>>(
        G, U, G, ff_i, gelu);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!quant::cuda::matmul_f32_dev_device_weight(
            dw, pooled(pool.down_bytes, pool.down_stride, slot), G, D, n_i))
      return false;
    moe_scatter_kernel<<<moe_grid(int64_t(n_e * d)), 256, 0, s>>>(
        D, d_row + begin, d_w + begin, y, int64_t(n_e), int64_t(d));
    if (cudaGetLastError() != cudaSuccess) return false;
  }
  return true;
}

// Free the MoE scratch (live slots and buffers retired for a capture). The
// caller has established that the model is going away; the function itself is
// a no-op while any capture is live.
void moe_release_scratch() { moe_release_scratch_impl(); }

} // namespace sonicboom::nn::cuda
