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
// This translation unit instantiates the imported ggml MMQ kernels (MIT, see
// the files under ggml_derived/mmq_upstream/ and NOTICE) and exposes them
// through plain-pointer entry points.  cuda_quant_matmul.cu runs the batch
// matmul and its activation quantize through these entry points.

#include "mmq_upstream.h"

#include "ggml_derived/mmq_upstream/mmq.cuh"
#include "ggml_derived/mmq_upstream/quantize_mmq_upstream.cuh"

#include <atomic>
#include <climits>
#include <cstdint>
#include <mutex>

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

// SonicBoom: plain replacement for ggml_cuda_info().devices[id].  The
// properties are queried once per device: cudaGetDeviceProperties costs
// hundreds of microseconds, and a prefill calls the matmul thousands of times.
sb_cuda_props get_props() {
    static std::mutex mutex;
    static std::atomic<int> cached_device{-1};
    static sb_cuda_props cached{};

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (cached_device.load(std::memory_order_acquire) != device) {
        std::lock_guard<std::mutex> lock(mutex);
        if (cached_device.load(std::memory_order_relaxed) != device) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

            sb_cuda_props props;
            // Upstream encodes the compute capability as 100*major + 10*minor
            // (sm_86 -> 860); every GGML_CUDA_CC_* predicate expects that form.
            props.cc        = 100 * prop.major + 10 * prop.minor;
            props.nsm       = prop.multiProcessorCount;
            props.warp_size = prop.warpSize;
            props.smpbo     = static_cast<int>(prop.sharedMemPerBlockOptin);
            cached = props;
            cached_device.store(device, std::memory_order_release);
        }
    }
    return cached;
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

// SonicBoom: device buffer for split-K partials, [split_k][n][M] float.
// Grown on demand and reused; the largest call seen determines the size, so
// the first call for a shape must run before CUDA graph capture.
float * split_scratch_get(size_t bytes) {
    static std::mutex mutex;
    static float * ptr = nullptr;
    static size_t capacity = 0;

    std::lock_guard<std::mutex> lock(mutex);
    if (bytes > capacity) {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        capacity = bytes;
    }
    return ptr;
}

// SonicBoom: build the 2-D, non-MoE mmq_args mapping used by upstream
// ggml_cuda_mul_mat_q()'s non-ids branch:
//   nchannels_x = nchannels_y = nsamples_x = nsamples_y = 1,
//   ncols_x = K, nrows_x = M, ncols_dst = n, ncols_y = n, nrows_dst = M,
//   stride_row_x = K/qk, ncols_max = ncols_opt = n.
//
// split_k > 1 runs the K tiles in split_k slices through the partial buffer.
// Columns beyond 16 are processed 16 at a time with the same partial buffer,
// which keeps every column's split identical to the n = 16 call.
template <ggml_type type>
bool run_mmq(const void * weights, const int32_t * y_q8, float * dst,
             int64_t M, int64_t K, int64_t n, cudaStream_t stream, int cfg,
             int split_k) {
    const sb_cuda_props props = get_props();

    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    constexpr int y_block_ints = sizeof(block_q8_1_mmq) / sizeof(int);
    const bool fallback = M % 128 != 0;

    const ggml_type gtype = type;

    // Launch columns [n_lo, n_lo + n_cnt) of the output.  ncols_y stays the
    // full column count: it is the row stride of the y block layout.
    const auto launch_range = [&](int64_t n_lo, int64_t n_cnt) -> bool {
        // SonicBoom: the configuration table disables stream-K (it splits K by
        // the total tile count, which depends on n); reject the launch if a
        // config ever selects it.
        const int J_best = pick_J(gtype, fallback, n_cnt, props);
        if (J_best == 0) {
            return false;
        }
        if (ggml_cuda_mmq_get_config(gtype, J_best, fallback, props.cc).stream_k) {
            return false;
        }

        float * scratch = nullptr;
        if (split_k > 1) {
            scratch = split_scratch_get(size_t(split_k) * size_t(M) * size_t(n_cnt) *
                                        sizeof(float));
        }

        const mmq_args args = {
            (const char *) weights, gtype, (const int *) y_q8 + n_lo * y_block_ints,
            nullptr, nullptr, dst + n_lo * M, nullptr,
            K, M, n_cnt, K / qk, n, M,
            1, 1, 0, 0, 0,
            1, 1, 0, 0, 0,
            n_cnt, n_cnt};

        mul_mat_q_case<type>(props, args, stream, cfg, split_k, scratch);
        CUDA_CHECK(cudaGetLastError());
        return true;
    };

    if (split_k > 1 && n > 16) {
        for (int64_t c0 = 0; c0 < n; c0 += 16) {
            const int64_t n_cnt = n - c0 < 16 ? n - c0 : 16;
            if (!launch_range(c0, n_cnt)) {
                return false;
            }
        }
        return true;
    }
    return launch_range(0, n);
}

}  // namespace

bool mmq_up_matmul(const void * weights, MmqUpType type, const int32_t * y_q8,
                   float * dst, int64_t M, int64_t K, int64_t n,
                   cudaStream_t stream, int cfg, int split_k) {
    if (weights == nullptr || y_q8 == nullptr || dst == nullptr) {
        return false;
    }
    if (M <= 0 || K <= 0 || n <= 0 || split_k <= 0 || K % 128 != 0) {
        return false;
    }

    const ggml_type gtype = to_ggml_type(type);
    switch (gtype) {
        case GGML_TYPE_Q3_K: return run_mmq<GGML_TYPE_Q3_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k);
        case GGML_TYPE_Q4_K: return run_mmq<GGML_TYPE_Q4_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k);
        case GGML_TYPE_Q5_K: return run_mmq<GGML_TYPE_Q5_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k);
        case GGML_TYPE_Q6_K: return run_mmq<GGML_TYPE_Q6_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k);
        case GGML_TYPE_Q5_0: return run_mmq<GGML_TYPE_Q5_0>(weights, y_q8, dst, M, K, n, stream, cfg, split_k);
        case GGML_TYPE_Q8_0: return run_mmq<GGML_TYPE_Q8_0>(weights, y_q8, dst, M, K, n, stream, cfg, split_k);
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

bool mmq_up_j_allowed(MmqUpType type, int64_t M, int J) {
    const ggml_type gtype = to_ggml_type(type);
    if (gtype == GGML_TYPE_COUNT || M <= 0) {
        return false;
    }
    if (J < 8 || J > 128 || J % 8 != 0) {
        return false;
    }
    const sb_cuda_props props = get_props();
    const bool fallback = M % 128 != 0;
    const ggml_cuda_mmq_config config = ggml_cuda_mmq_get_config(gtype, J, fallback, props.cc);
    return config.type != GGML_TYPE_COUNT &&
           mmq_get_nbytes_shared(config, props.cc) <= size_t(props.smpbo);
}

int mmq_up_selected_j(MmqUpType type, int64_t M, int64_t n) {
    const ggml_type gtype = to_ggml_type(type);
    if (gtype == GGML_TYPE_COUNT || M <= 0 || n <= 0) {
        return 0;
    }
    return pick_J(gtype, M % 128 != 0, n, get_props());
}

int mmq_up_cfg_count() {
    return MMQ_CFG_COUNT;
}

bool mmq_up_cfg_neutral(MmqUpType type, int cfg) {
    // Filled from test_mmq_cfg_neutral: a candidate is listed only when it
    // reproduces the table's output bit for bit for this type at every tested
    // shape and n <= 16.  Config 2 (I = 64, 128 threads) passed for every
    // type; configs 1, 3, 4 and 5 do not match the MMA warp row mapping
    // (mmq_cfg_rows_fit), so they were never launched and are not listed.
    static const bool neutral[6][MMQ_CFG_COUNT] = {
        /* Q3_K */ {true, false, true, false, false, false, true},
        /* Q4_K */ {true, false, true, false, false, false, true},
        /* Q5_K */ {true, false, true, false, false, false, true},
        /* Q6_K */ {true, false, true, false, false, false, true},
        /* Q5_0 */ {true, false, true, false, false, false, true},
        /* Q8_0 */ {true, false, true, false, false, false, true},
    };
    const int row = static_cast<int>(type);
    if (row < 0 || row >= 6 || cfg < 0 || cfg >= MMQ_CFG_COUNT) {
        return false;
    }
    return neutral[row][cfg];
}

// The MMA path maps one rows_per_warp-row minitile to each warp; with
// occupancy 1 the tile width must be exactly nthreads/warp_size minitiles,
// otherwise part of the tile is never computed (or the tile overruns the
// shared-memory buffer).  The same test the launcher applies.
static bool mmq_cfg_rows_fit(int I, int nthreads, int J, int warp_size) {
    if (warp_size <= 0) return false;
    const int rows_per_warp = (J >= 48 && J % 16 == 0) ? 32 : 16;
    return I * warp_size == nthreads * rows_per_warp;
}

bool mmq_up_cfg_allowed(MmqUpType type, int64_t M, int J, int cfg) {
    const ggml_type gtype = to_ggml_type(type);
    if (gtype == GGML_TYPE_COUNT || M <= 0 || J < 8 || J > 128 || J % 8 != 0) {
        return false;
    }
    if (cfg < 0 || cfg >= MMQ_CFG_COUNT || (cfg != 0 && J != 8 && J != 16)) {
        return false;
    }
    if (!mmq_up_cfg_neutral(type, cfg)) {
        return false;
    }
    const sb_cuda_props props = get_props();
    const bool fallback = M % 128 != 0;
    const ggml_cuda_mmq_config base = ggml_cuda_mmq_get_config_base(gtype, J, fallback, props.cc);
    if (base.type == GGML_TYPE_COUNT) {
        return false;
    }
    const ggml_cuda_mmq_config config = mmq_apply_cfg(cfg, base);
    return mmq_get_nbytes_shared(config, props.cc) <= size_t(props.smpbo) &&
           (cfg == 0 || mmq_cfg_rows_fit(config.I, config.nthreads, J,
                                         props.warp_size));
}

void mmq_up_cfg_dims(MmqUpType type, int64_t M, int J, int cfg, int * I, int * nthreads) {
    const ggml_type gtype = to_ggml_type(type);
    const sb_cuda_props props = get_props();
    const bool fallback = M % 128 != 0;
    const ggml_cuda_mmq_config config = mmq_apply_cfg(cfg, ggml_cuda_mmq_get_config_base(gtype, J, fallback, props.cc));
    if (I) *I = config.I;
    if (nthreads) *nthreads = config.nthreads;
}

}  // namespace sonicboom::quant::cuda
