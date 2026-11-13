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
                 std::span<float> y, uint64_t rows, bool group_by_expert);

} // namespace sonicboom::nn::cuda
