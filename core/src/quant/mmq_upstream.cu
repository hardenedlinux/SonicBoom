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
//
// This translation unit instantiates the imported ggml MMQ kernels (MIT,
// see the files under ggml_derived/mmq_upstream/ and NOTICE).  It is a new,
// currently-unused entry point; it does not alter any production dispatch.

#include "mmq_upstream.h"

#include "ggml_derived/mmq_upstream/mmq.cuh"
#include "ggml_derived/mmq_upstream/quantize_mmq_upstream.cuh"

#include <climits>
#include <cstdint>

namespace sonicboom::quant::cuda {
namespace {

// SonicBoom: map the public enum onto the upstream ggml type values.
ggml_type to_ggml_type(MmqUpType type) {
    switch (type) {
        case MmqUpType::Q3_K: return GGML_TYPE_Q3_K;
        case MmqUpType::Q4_K: return GGML_TYPE_Q4_K;
        case MmqUpType::Q5_K: return GGML_TYPE_Q5_K;
        case MmqUpType::Q6_K: return GGML_TYPE_Q6_K;
        case MmqUpType::Q5_0: return GGML_TYPE_Q5_0;
        case MmqUpType::Q8_0: return GGML_TYPE_Q8_0;
    }
    return GGML_TYPE_COUNT;
}

// SonicBoom: plain replacement for ggml_cuda_info().devices[id].
sb_cuda_props get_props() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    sb_cuda_props props;
    props.cc        = prop.major * 10 + prop.minor;
    props.nsm       = prop.multiProcessorCount;
    props.warp_size = prop.warpSize;
    props.smpbo     = static_cast<int>(prop.sharedMemPerBlockOptin);
    return props;
}

// SonicBoom: the same J selection as mul_mat_q_switch_J(), used here only to
// inspect the selected configuration before launching.  Returns 0 when no
// configuration fits.
int pick_J(ggml_type type, bool fallback, int64_t ncols_opt, const sb_cuda_props & props) {
    int J_best        = 0;
    int ntiles_J_best = INT_MAX;

    for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
        const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(type, J, fallback, props.cc);
        if (config.type == GGML_TYPE_COUNT) {
            continue;
        }
        if (mmq_get_nbytes_shared(config, props.cc) > props.smpbo) {
            continue;
        }
        const int ntiles_x = (int)((ncols_opt + config.J - 1) / config.J);
        if (ntiles_x < ntiles_J_best) {
            J_best = J;
            ntiles_J_best = ntiles_x;
        }
    }
    return J_best;
}

// SonicBoom: build the 2-D, non-MoE mmq_args mapping used by upstream
// ggml_cuda_mul_mat_q()'s non-ids branch:
//   nchannels_x = nchannels_y = nsamples_x = nsamples_y = 1,
//   ncols_x = K, nrows_x = M, ncols_dst = n, ncols_y = n, nrows_dst = M,
//   stride_row_x = K/qk, ncols_max = ncols_opt = n.
template <ggml_type type>
bool run_mmq(const void * weights, const int32_t * y_q8, float * dst,
             int64_t M, int64_t K, int64_t n, cudaStream_t stream) {
    const sb_cuda_props props = get_props();

    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const bool fallback = M % 128 != 0;

    const ggml_type gtype = type;

    // SonicBoom: the stream-K decomposition is not validated in this step.
    const int J_best = pick_J(gtype, fallback, n, props);
    if (J_best == 0) {
        return false;
    }
    if (ggml_cuda_mmq_get_config(gtype, J_best, fallback, props.cc).stream_k) {
        return false;
    }

    const mmq_args args = {
        (const char *) weights, gtype, (const int *) y_q8, nullptr, nullptr, dst, nullptr,
        K, M, n, K / qk, n, M,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        n, n};

    mul_mat_q_case<type>(props, args, stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

}  // namespace

bool mmq_up_matmul(const void * weights, MmqUpType type, const int32_t * y_q8,
                   float * dst, int64_t M, int64_t K, int64_t n,
                   cudaStream_t stream) {
    if (weights == nullptr || y_q8 == nullptr || dst == nullptr) {
        return false;
    }
    if (M <= 0 || K <= 0 || n <= 0 || K % 128 != 0) {
        return false;
    }

    const ggml_type gtype = to_ggml_type(type);
    switch (gtype) {
        case GGML_TYPE_Q3_K: return run_mmq<GGML_TYPE_Q3_K>(weights, y_q8, dst, M, K, n, stream);
        case GGML_TYPE_Q4_K: return run_mmq<GGML_TYPE_Q4_K>(weights, y_q8, dst, M, K, n, stream);
        case GGML_TYPE_Q5_K: return run_mmq<GGML_TYPE_Q5_K>(weights, y_q8, dst, M, K, n, stream);
        case GGML_TYPE_Q6_K: return run_mmq<GGML_TYPE_Q6_K>(weights, y_q8, dst, M, K, n, stream);
        case GGML_TYPE_Q5_0: return run_mmq<GGML_TYPE_Q5_0>(weights, y_q8, dst, M, K, n, stream);
        case GGML_TYPE_Q8_0: return run_mmq<GGML_TYPE_Q8_0>(weights, y_q8, dst, M, K, n, stream);
        default:             return false;
    }
}

bool mmq_up_quantize(const float * x, int32_t * y_q8, int64_t K, int64_t n,
                     MmqUpType type, cudaStream_t stream) {
    if (x == nullptr || y_q8 == nullptr) {
        return false;
    }
    if (K <= 0 || n <= 0 || K % 128 != 0) {
        return false;
    }

    const ggml_type gtype = to_ggml_type(type);
    if (gtype == GGML_TYPE_COUNT) {
        return false;
    }

    // x is [n][K] (row stride K); the output is [K/128][n] block_q8_1_mmq.
    quantize_mmq_q8_1_upstream(x, y_q8, gtype, K, /*s01=*/K, /*ne0=*/K,
                               /*ne1=*/static_cast<int>(n), stream);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

}  // namespace sonicboom::quant::cuda
