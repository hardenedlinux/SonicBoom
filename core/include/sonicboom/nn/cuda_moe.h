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

// CUDA mixture-of-experts kernels: the device twins of <sonicboom/nn/moe.h>.
// Every expert projection is a K-quant gemv / batched matmul on the device;
// the packed weights are uploaded on first use and stay resident in the weight
// cache, so a step only ever moves activations.
//
// Two shapes are distinct, as they are on the CPU:
//   - decode (one row) and verify (M rows) run the decode gemv once per
//     (row, selected expert). A row's work does not depend on the other rows,
//     so verify row i is bit-identical to that row decoded alone.
//   - prefill groups the rows by selected expert and runs one batched matmul
//     per expert, scattering the weighted result back. Rows still accumulate
//     in ascending expert index; prefill is not required to match decode
//     bit for bit.
//
// The declarations use only Layer 2 types: no CUDA type appears here. The
// runtime and kernels live in core/src/nn/cuda_moe.cu and are only linked when
// the build is configured with SONICBOOM_USE_CUDA=ON.

#include <sonicboom/nn/moe.h>

#include <cstdint>
#include <span>

namespace sonicboom::nn::cuda {

// The routing of `rows` positions: x is [rows, d], ids and weights are
// [rows, used]. Each row is routed with the same arithmetic as nn::moe_route —
// the router projection is a device gemv (or the dense matvec for an f32
// router), and the stable softmax, the top-k with ties to the lower index and
// the ascending renormalisation run on the host, so a row's ids and weights do
// not depend on the row count. Returns false (writing nothing) on a bad shape,
// a missing router, or a failed device kernel.
bool moe_route(std::span<const float> x, const quant::QuantizedTensor* router,
               std::span<const float> router_f32, uint32_t experts,
               uint32_t used, float weight_scale, std::span<int64_t> ids,
               std::span<float> weights, uint64_t rows);

// The device-resident form: x is a [rows, d] device buffer, and ids and weights
// are [rows, used] device buffers that the router fills in place. The router
// projection is the same device gemv as above; one thread block per row then
// runs the softmax, the top-k (ties to the lower index) and the ascending
// renormalisation in one thread, in the same operation order as nn::moe_route.
// The host never sees an intermediate. The result may differ from the host
// form by a few ulp (device expf for std::exp); see the M2 gate in the round-7
// notes. `router_f32` is a host weight span, uploaded and cached by the f32
// gemv exactly as in moe_route.
bool moe_route_dev(const float* x, const quant::QuantizedTensor* router,
                   std::span<const float> router_f32, uint32_t experts,
                   uint32_t used, float weight_scale, int64_t* ids,
                   float* weights, uint64_t rows);

// A routed expert's matrices already resident on the device, in the planner's
// bounded expert pool (design §9): one slot per (block, expert), slot s of role
// r at role_bytes + s * role_stride. `slots` holds one int32 slot per
// (row, selected expert) entry in `ids` order, as the pool wrote it. With the
// pool null the kernels take the uploading path (each expert's device copy
// cached on first use). Residency changes only where a matrix is found, never
// the arithmetic.
struct MoeExpertPool {
  const void* gate_bytes = nullptr;
  const void* up_bytes = nullptr;
  const void* down_bytes = nullptr;
  uint64_t gate_stride = 0;
  uint64_t up_stride = 0;
  uint64_t down_stride = 0;
  const int32_t* slots = nullptr;
};

// y = sum over the rows' selected experts of w_e * down_e(act(gate_e x) * up_e x).
// x is [rows, d], ids and weights are [rows, used], y is [rows, d]. `y` is
// accumulated into, exactly as nn::moe_experts accumulates: the caller supplies
// a zeroed buffer.
//
// `group_by_expert` selects the prefill form (one batched matmul per expert over
// the rows that selected it). With it false, each (row, expert) pair runs the
// decode gemv — the decode and verify form.
bool moe_experts(std::span<const float> x, std::span<const int64_t> ids,
                 std::span<const float> weights,
                 const MoeExpertWeights& experts, MoeActivation activation,
                 std::span<float> y, uint64_t rows, bool group_by_expert,
                 const MoeExpertPool* pool = nullptr);

// The device-resident form of the same node. x, weights_dev, y and slots_dev
// are device buffers, x [rows, d], weights_dev [rows, k], y [rows, d] and
// slots_dev [rows, k]; y must be zeroed by the caller (the backend issues a
// cudaMemsetAsync) and every weight stays resident. ids_host and slots_host are
// the host copies of the two routing results (the pool selection reads ids
// back; slots are the pool's assignment), so the node downloads only the ids.
//
// The arithmetic is the host form's: the decode/verify path runs one gemv per
// (row, expert) and a final j-ascending weighted sum; the prefill path groups
// the rows by expert and visits the experts in ascending index on one stream,
// scattering with an explicit __fadd_rn/__fmul_rn. `stream` is an opaque CUDA
// stream handle (nullptr = the current stream); the kernels run there.
//
// `entry_events` (optional, one per (row, expert) in the order of ids/slots) is
// the pool's completion mark for that selection entry: a nonzero value makes
// the stream wait for it before the entry's expert is used, so an expert's
// upload can finish while the experts resident before it compute. The marks
// only order the launches; no arithmetic changes.
bool moe_experts_dev(const float* x, const int64_t* ids_host,
                     const float* weights_dev, const int32_t* slots_dev,
                     const int32_t* slots_host,
                     const MoeExpertWeights& experts, MoeActivation activation,
                     float* y, uint64_t rows, uint64_t k, uint64_t d,
                     bool group_by_expert, const MoeExpertPool& pool,
                     void* stream = nullptr,
                     const uint64_t* entry_events = nullptr);

// Drop the MoE kernel scratch buffers (live slots and any retired because a CUDA
// graph capture had their pointers baked in). A no-op while a capture is live;
// called when the last model goes away.
void moe_release_scratch();

} // namespace sonicboom::nn::cuda
