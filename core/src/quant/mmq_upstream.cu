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

#include "../nn/cuda_scratch_epoch.h"
#include "../planner/autotune.h"

#include "ggml_derived/mmq_upstream/mmq.cuh"
#include "ggml_derived/mmq_upstream/quantize_mmq_upstream.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
// Grown on demand and reused.  The warm-up measures the split counts, so the
// buffer reaches its high-water size before any CUDA graph capture is built; a
// growth after a capture exists retains the old allocation (the capture keeps
// its baked pointer) and bumps the scratch epoch, as the shared scratch
// contract requires.
float * split_scratch_get(size_t bytes) {
    static std::mutex mutex;
    static float * ptr = nullptr;
    static size_t capacity = 0;
    static std::vector<float *> retired;

    std::lock_guard<std::mutex> lock(mutex);
    if (bytes > capacity) {
        if (ptr != nullptr) {
            if (sonicboom::cuda::scratch_may_free()) {
                CUDA_CHECK(cudaFree(ptr));
            } else {
                retired.push_back(ptr);
            }
            ptr = nullptr;
        }
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        capacity = bytes;
        sonicboom::cuda::scratch_epoch_bump();
    }
    return ptr;
}

// SonicBoom: build the 2-D, non-MoE mmq_args mapping used by upstream
// ggml_cuda_mul_mat_q()'s non-ids branch:
//   nchannels_x = nchannels_y = nsamples_x = nsamples_y = 1,
//   ncols_x = K, nrows_x = M, ncols_dst = n, ncols_y = n, nrows_dst = M,
//   stride_row_x = K/qk, ncols_max = ncols_opt = n.
//
// split_k > 1 runs the K tiles in split_k contiguous slices over all n
// columns: one slice pass into the [split_k][n][M] partial buffer, then one
// reduce over its entries in slice order.  n does not change the decomposition.
template <ggml_type type>
bool run_mmq(const void * weights, const int32_t * y_q8, float * dst,
             int64_t M, int64_t K, int64_t n, cudaStream_t stream, int cfg,
             int split_k, const float * bias, const float * add) {
    const sb_cuda_props props = get_props();

    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const bool fallback = M % 128 != 0;

    const ggml_type gtype = type;

    // SonicBoom: the configuration table disables stream-K (it splits K by the
    // total tile count, which depends on n); reject the launch if a config ever
    // selects it.
    const int J_best = pick_J(gtype, fallback, n, props);
    if (J_best == 0) {
        return false;
    }
    if (ggml_cuda_mmq_get_config(gtype, J_best, fallback, props.cc).stream_k) {
        return false;
    }

    float * scratch = nullptr;
    if (split_k > 1) {
        scratch = split_scratch_get(size_t(split_k) * size_t(M) * size_t(n) *
                                    sizeof(float));
    }

    const mmq_args args = {
        (const char *) weights, gtype, (const int *) y_q8,
        nullptr, nullptr, dst, nullptr,
        K, M, n, K / qk, n, M,
        1, 1, 0, 0, 0,
        1, 1, 0, 0, 0,
        n, n,
        bias, add};

    mul_mat_q_case<type>(props, args, stream, cfg, split_k, scratch);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

}  // namespace

bool mmq_up_matmul(const void * weights, MmqUpType type, const int32_t * y_q8,
                   float * dst, int64_t M, int64_t K, int64_t n,
                   cudaStream_t stream, int cfg, int split_k,
                   const float * bias, const float * add) {
    if (weights == nullptr || y_q8 == nullptr || dst == nullptr) {
        return false;
    }
    if (M <= 0 || K <= 0 || n <= 0 || split_k <= 0 || K % 128 != 0) {
        return false;
    }
    const ggml_type gtype = to_ggml_type(type);
    switch (gtype) {
        case GGML_TYPE_Q3_K: return run_mmq<GGML_TYPE_Q3_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k, bias, add);
        case GGML_TYPE_Q4_K: return run_mmq<GGML_TYPE_Q4_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k, bias, add);
        case GGML_TYPE_Q5_K: return run_mmq<GGML_TYPE_Q5_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k, bias, add);
        case GGML_TYPE_Q6_K: return run_mmq<GGML_TYPE_Q6_K>(weights, y_q8, dst, M, K, n, stream, cfg, split_k, bias, add);
        case GGML_TYPE_Q5_0: return run_mmq<GGML_TYPE_Q5_0>(weights, y_q8, dst, M, K, n, stream, cfg, split_k, bias, add);
        case GGML_TYPE_Q8_0: return run_mmq<GGML_TYPE_Q8_0>(weights, y_q8, dst, M, K, n, stream, cfg, split_k, bias, add);
        default:             return false;
    }
}

// ---------------------------------------------------------------------------
// SonicBoom: sibling-group matmul.  One launch carries every member's row
// tiles; see mul_mat_q_group in the imported kernel header.  The tuple of
// member formats is a template parameter, so only the tuples this build
// instantiates can be launched — every other group runs its members'
// separate matmuls.
// ---------------------------------------------------------------------------

namespace {

// Number of K tiles one member's tile body visits at this tile width.  K_vram
// comes from the member's host-side configuration; the block count divides it
// the same way the kernel's ITER_K does.
template <ggml_type type>
int group_k_tiles(int K_vram, int blocks_per_row) {
    constexpr int qk = ggml_cuda_type_traits<type>::qk;
    const int blocks_per_iter = K_vram / qk;
    return (blocks_per_row + blocks_per_iter - 1) / blocks_per_iter;
}

// Launch the group at one tile width.  split_k > 1 runs the group's K tiles in
// that many slices (capped at the smallest member tile count) and then reduces
// the shared partial buffer once over all members.
template <ggml_type T0, ggml_type T1, ggml_type T2, int J>
bool group_launch_j(const sb_cuda_props & props,
                    const sonicboom::quant::cuda::MmqGroupArgs & args,
                    const ggml_cuda_mmq_config & c0,
                    const ggml_cuda_mmq_config & c1,
                    const ggml_cuda_mmq_config & c2,
                    size_t nbs, int split_k, cudaStream_t stream) {
    int ntiles = group_k_tiles<T0>(c0.K_vram, args.m[0].blocks_per_row);
    ntiles = std::min(ntiles, group_k_tiles<T1>(c1.K_vram, args.m[0].blocks_per_row));
    if (args.n_members > 2) {
        ntiles = std::min(ntiles, group_k_tiles<T2>(c2.K_vram, args.m[0].blocks_per_row));
    }
    int split_eff = split_k > 1 ? std::min(split_k, ntiles) : 1;
    if (split_eff < 2) {
        split_eff = 1;
    }

    sonicboom::quant::cuda::MmqGroupArgs a = args;
    a.nsplit = split_eff;
    a.part = nullptr;
    if (split_eff > 1) {
        a.part = split_scratch_get(size_t(split_eff) * size_t(args.ncols_dst) *
                                   size_t(args.total_rows) * sizeof(float));
    }

    const dim3 block(props.warp_size, c0.nthreads / props.warp_size, 1);
    const dim3 grid(unsigned(args.tile_prefix[args.n_members]),
                    unsigned((args.ncols_dst + J - 1) / J), unsigned(split_eff));
    mul_mat_q_group<T0, T1, T2, J><<<grid, block, nbs, stream>>>(a);
    CUDA_CHECK(cudaGetLastError());

    if (split_eff > 1) {
        const int64_t n_elem = int64_t(args.ncols_dst) * int64_t(args.total_rows);
        const int     reduce_threads = 256;
        const int64_t reduce_blocks  = (n_elem + reduce_threads - 1) / reduce_threads;
        SB_NOTE_LAUNCH(mul_mat_q_group_split_k_reduce,
                       dim3((unsigned)reduce_blocks), dim3(reduce_threads), 0,
                       stream);
        mul_mat_q_group_split_k_reduce<<<(unsigned)reduce_blocks, reduce_threads,
                                         0, stream>>>(a.part, a);
        CUDA_CHECK(cudaGetLastError());
    }
    return true;
}

// The group's shared column count fixes the tile width: the same pick_J the
// imported launcher makes, which depends only on n.  A member whose table has
// no configuration at that width makes the group unlaunchable.
template <ggml_type T0, ggml_type T1, ggml_type T2>
bool group_launch_typed(const sb_cuda_props & props,
                        const sonicboom::quant::cuda::MmqGroupArgs & args,
                        int split_k, cudaStream_t stream) {
    const int n = args.ncols_dst;
    if (n <= 0 || args.ncols_y <= 0) {
        return false;
    }
    const int J = pick_J(T0, false, n, props);
    if (J == 0 || pick_J(T1, false, n, props) != J ||
        (args.n_members > 2 && pick_J(T2, false, n, props) != J)) {
        return false;
    }
    const ggml_cuda_mmq_config c0 = ggml_cuda_mmq_get_config<0>(T0, J, false, props.cc);
    const ggml_cuda_mmq_config c1 = ggml_cuda_mmq_get_config<0>(T1, J, false, props.cc);
    const ggml_cuda_mmq_config c2 = ggml_cuda_mmq_get_config<0>(T2, J, false, props.cc);
    if (c0.type == GGML_TYPE_COUNT || c1.type == GGML_TYPE_COUNT ||
        (args.n_members > 2 && c2.type == GGML_TYPE_COUNT)) {
        return false;
    }
    if (c0.nthreads != c1.nthreads ||
        (args.n_members > 2 && c0.nthreads != c2.nthreads) ||
        c0.I != c1.I || (args.n_members > 2 && c0.I != c2.I)) {
        return false;
    }
    const size_t nbs = std::max({mmq_get_nbytes_shared(c0, props.cc),
                                 mmq_get_nbytes_shared(c1, props.cc),
                                 mmq_get_nbytes_shared(c2, props.cc)});
    if (nbs > size_t(props.smpbo)) {
        return false;
    }

    sonicboom::quant::cuda::MmqGroupArgs a = args;
    int prefix = 0;
    int rows = 0;
    for (int k = 0; k < args.n_members; ++k) {
        auto & m = a.m[k];
        const ggml_cuda_mmq_config & ck = k == 0 ? c0 : (k == 1 ? c1 : c2);
        // The group's tile body runs without the out-of-bounds check and
        // writes every row of its last tile: the rows must fill whole tiles.
        if (m.nrows <= 0 || m.nrows % ck.I != 0 || m.blocks_per_row <= 0 ||
            m.stride_row_x != m.blocks_per_row ||
            m.stride_col_dst != m.nrows) {
            return false;
        }
        a.tile_prefix[k] = prefix;
        prefix += (m.nrows + ck.I - 1) / ck.I;
        a.rows_prefix[k] = rows;
        rows += m.nrows;
    }
    a.tile_prefix[args.n_members] = prefix;
    a.rows_prefix[args.n_members] = rows;
    a.total_rows = rows;

    switch (J) {
        case   8: return group_launch_j<T0, T1, T2,   8>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  16: return group_launch_j<T0, T1, T2,  16>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  24: return group_launch_j<T0, T1, T2,  24>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  32: return group_launch_j<T0, T1, T2,  32>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  40: return group_launch_j<T0, T1, T2,  40>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  48: return group_launch_j<T0, T1, T2,  48>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  56: return group_launch_j<T0, T1, T2,  56>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  64: return group_launch_j<T0, T1, T2,  64>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  72: return group_launch_j<T0, T1, T2,  72>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  80: return group_launch_j<T0, T1, T2,  80>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  88: return group_launch_j<T0, T1, T2,  88>(props, a, c0, c1, c2, nbs, split_k, stream);
        case  96: return group_launch_j<T0, T1, T2,  96>(props, a, c0, c1, c2, nbs, split_k, stream);
        case 104: return group_launch_j<T0, T1, T2, 104>(props, a, c0, c1, c2, nbs, split_k, stream);
        case 112: return group_launch_j<T0, T1, T2, 112>(props, a, c0, c1, c2, nbs, split_k, stream);
        case 120: return group_launch_j<T0, T1, T2, 120>(props, a, c0, c1, c2, nbs, split_k, stream);
        case 128: return group_launch_j<T0, T1, T2, 128>(props, a, c0, c1, c2, nbs, split_k, stream);
        default:  return false;
    }
}

// Raise the shared-memory limit for every tile width the group kernel can be
// launched with.  Nothing is launched here.
template <ggml_type T0, ggml_type T1, ggml_type T2>
void group_warm_typed(const sb_cuda_props & props) {
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c0 = ggml_cuda_mmq_get_config<0>(T0, J, false, props.cc);
        const ggml_cuda_mmq_config c1 = ggml_cuda_mmq_get_config<0>(T1, J, false, props.cc);
        const ggml_cuda_mmq_config c2 = ggml_cuda_mmq_get_config<0>(T2, J, false, props.cc);
        if (c0.type == GGML_TYPE_COUNT || c1.type == GGML_TYPE_COUNT ||
            c2.type == GGML_TYPE_COUNT) {
            continue;
        }
        const size_t nbs = std::max({mmq_get_nbytes_shared(c0, props.cc),
                                     mmq_get_nbytes_shared(c1, props.cc),
                                     mmq_get_nbytes_shared(c2, props.cc)});
        if (nbs > size_t(props.smpbo)) {
            continue;
        }
        switch (J) {
            case   8: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,   8>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  16: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  16>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  24: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  24>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  32: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  32>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  40: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  40>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  48: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  48>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  56: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  56>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  64: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  64>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  72: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  72>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  80: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  80>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  88: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  88>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case  96: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2,  96>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case 104: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2, 104>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case 112: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2, 112>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case 120: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2, 120>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            case 128: CUDA_CHECK(cudaFuncSetAttribute(mul_mat_q_group<T0, T1, T2, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(nbs))); break;
            default:  break;
        }
    }
}

// The format tuples whose group kernels this build instantiates: the sibling
// projections the four reference models' decode and verify graphs carry.
bool group_dispatch(const sb_cuda_props & props,
                    const sonicboom::quant::cuda::MmqGroupArgs & args,
                    const MmqUpType * types, int split_k, cudaStream_t stream,
                    bool warm) {
    const ggml_type t0 = to_ggml_type(types[0]);
    const ggml_type t1 = to_ggml_type(types[1]);
    const ggml_type t2 = args.n_members > 2 ? to_ggml_type(types[2]) : t1;
    if (t0 == GGML_TYPE_Q3_K && t1 == GGML_TYPE_Q3_K && t2 == GGML_TYPE_Q3_K) {
        if (warm) group_warm_typed<GGML_TYPE_Q3_K, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K>(props);
        else return group_launch_typed<GGML_TYPE_Q3_K, GGML_TYPE_Q3_K, GGML_TYPE_Q3_K>(props, args, split_k, stream);
        return true;
    }
    if (t0 == GGML_TYPE_Q4_K && t1 == GGML_TYPE_Q4_K && t2 == GGML_TYPE_Q4_K) {
        if (warm) group_warm_typed<GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K>(props);
        else return group_launch_typed<GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q4_K>(props, args, split_k, stream);
        return true;
    }
    if (t0 == GGML_TYPE_Q5_0 && t1 == GGML_TYPE_Q5_0 && t2 == GGML_TYPE_Q5_0) {
        if (warm) group_warm_typed<GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0>(props);
        else return group_launch_typed<GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0>(props, args, split_k, stream);
        return true;
    }
    if (t0 == GGML_TYPE_Q5_0 && t1 == GGML_TYPE_Q5_0 && t2 == GGML_TYPE_Q8_0) {
        if (warm) group_warm_typed<GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0>(props);
        else return group_launch_typed<GGML_TYPE_Q5_0, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0>(props, args, split_k, stream);
        return true;
    }
    return false;
}

}  // namespace

bool mmq_up_group_matmul(const sonicboom::quant::cuda::MmqGroupArgs & args,
                         const MmqUpType * types, cudaStream_t stream) {
    if (types == nullptr || args.n_members < 2 || args.n_members > 3) {
        return false;
    }
    int64_t rows[3] = {};
    for (int k = 0; k < args.n_members; ++k) {
        const auto & m = args.m[k];
        if (m.x == nullptr || m.y == nullptr || m.dst == nullptr ||
            m.nrows <= 0 || m.blocks_per_row <= 0) {
            return false;
        }
        rows[k] = m.nrows;
    }
    // The members' shared column count, from the first member's block row
    // stride and its format's element count per block.
    const int qk = (types[0] == MmqUpType::Q5_0 || types[0] == MmqUpType::Q8_0)
                       ? 32
                       : 256;
    const int64_t cols = int64_t(args.m[0].blocks_per_row) * qk;
    const int split_k = mmq_up_group_split_for(types, rows, args.n_members, cols);
    return group_dispatch(get_props(), args, types, split_k, stream, /*warm=*/false);
}

bool mmq_up_group_matmul_split(const sonicboom::quant::cuda::MmqGroupArgs & args,
                               const MmqUpType * types, int split_k,
                               cudaStream_t stream) {
    if (types == nullptr || args.n_members < 2 || args.n_members > 3 ||
        split_k < 1) {
        return false;
    }
    for (int k = 0; k < args.n_members; ++k) {
        const auto & m = args.m[k];
        if (m.x == nullptr || m.y == nullptr || m.dst == nullptr ||
            m.nrows <= 0 || m.blocks_per_row <= 0) {
            return false;
        }
    }
    return group_dispatch(get_props(), args, types, split_k, stream,
                          /*warm=*/false);
}

bool mmq_up_group_warm(const MmqUpType * types, int n_members) {
    if (types == nullptr || n_members < 2 || n_members > 3) {
        return false;
    }
    sonicboom::quant::cuda::MmqGroupArgs args{};
    args.n_members = n_members;
    return group_dispatch(get_props(), args, types, 1, nullptr, /*warm=*/true);
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

namespace {

// ---------------------------------------------------------------------------
// Warm-up measurement of the small-n split-K count and tile configuration.
// ---------------------------------------------------------------------------

// The row count a verify step (n = k+1 <= 16) can carry that runs the wide
// tile: the narrow tile (J = 8) is measured at the decode row count, n = 1.
constexpr int kMmqTuneNarrowN = 1;
constexpr int kMmqTuneWideN   = 16;
// The split-count measurement probes n = 1, 8 and 512 on the same synthetic
// buffers, so they are sized for the widest probe.
constexpr int kMmqTuneMaxN = 512;
// (I, nthreads) = (64, 256) and (128, 256).
constexpr int kMmqTuneCfgsReduced[]   = {5, 6};
constexpr double kMmqTuneBudgetS = 1.5;
constexpr int kMmqTuneTimed = 7;
constexpr int kMmqTuneWarm  = 3;

// Byte size of one GGUF block and the element count it covers.
void mmq_block_meta(MmqUpType type, int * qk, int * bytes) {
    switch (type) {
        case MmqUpType::Q3_K: *qk = 256; *bytes = 110; break;
        case MmqUpType::Q4_K: *qk = 256; *bytes = 144; break;
        case MmqUpType::Q5_K: *qk = 256; *bytes = 176; break;
        case MmqUpType::Q6_K: *qk = 256; *bytes = 210; break;
        case MmqUpType::Q5_0: *qk = 32;  *bytes = 22;  break;
        case MmqUpType::Q8_0: *qk = 32;  *bytes = 34;  break;
    }
}

const char * mmq_type_name(MmqUpType type) {
    switch (type) {
        case MmqUpType::Q3_K: return "q3_K";
        case MmqUpType::Q4_K: return "q4_K";
        case MmqUpType::Q5_K: return "q5_K";
        case MmqUpType::Q6_K: return "q6_K";
        case MmqUpType::Q5_0: return "q5_0";
        case MmqUpType::Q8_0: return "q8_0";
    }
    return "unknown";
}

std::string mmq_pin_key(MmqUpType type, int64_t M, int64_t K) {
    return std::string(mmq_type_name(type)) + "|" + std::to_string(M) + "x" +
           std::to_string(K);
}

int mmq_build_hash(std::string_view build_id) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (unsigned char c : build_id) {
        h ^= c;
        h *= 0x100000001b3ull;
    }
    return int(static_cast<uint32_t>(h ^ (h >> 32)));
}

std::mutex g_mmq_pin_mutex;
std::map<std::string, MmqUpPin> g_mmq_pin;

struct MmqTuneShape {
    MmqUpType type;
    int64_t rows;
    int64_t cols;
};

struct MmqTuneBuffers {
    void * dw = nullptr;
    float * dx = nullptr;
    int32_t * dyq8 = nullptr;
    float * dout = nullptr;
};

void mmq_tune_free(MmqTuneBuffers & b) {
    if (b.dout) cudaFree(b.dout);
    if (b.dyq8) cudaFree(b.dyq8);
    if (b.dx) cudaFree(b.dx);
    if (b.dw) cudaFree(b.dw);
    b = {};
}

bool mmq_tune_alloc(const MmqTuneShape & s, MmqTuneBuffers & b) {
    int qk = 0, bytes = 0;
    mmq_block_meta(s.type, &qk, &bytes);
    const size_t wbytes = size_t(s.rows) * size_t(s.cols / qk) * size_t(bytes);
    const size_t xbytes = size_t(kMmqTuneMaxN) * size_t(s.cols) * sizeof(float);
    const size_t yblocks = size_t(s.cols / 128) * size_t(kMmqTuneMaxN) + 128;
    const size_t ybytes = yblocks * sizeof(int) * 36;  // block_q8_1_mmq
    const size_t obytes = size_t(s.rows) * size_t(kMmqTuneMaxN) * sizeof(float);
    if (cudaMalloc(&b.dw, wbytes) != cudaSuccess) return false;
    if (cudaMalloc(&b.dx, xbytes) != cudaSuccess) { mmq_tune_free(b); return false; }
    if (cudaMalloc(&b.dyq8, ybytes) != cudaSuccess) { mmq_tune_free(b); return false; }
    if (cudaMalloc(&b.dout, obytes) != cudaSuccess) { mmq_tune_free(b); return false; }
    cudaMemset(b.dw, 0, wbytes);
    cudaMemset(b.dx, 0, xbytes);
    cudaMemset(b.dyq8, 0, ybytes);
    return true;
}

// Median of the timed launches, after the untimed warm-ups.
double mmq_tune_time(const MmqTuneShape & s, const MmqTuneBuffers & b, int n,
                     int cfg, int split_k) {
    for (int i = 0; i < kMmqTuneWarm; ++i) {
        mmq_up_matmul(b.dw, s.type, b.dyq8, b.dout, s.rows, s.cols, n,
                      nullptr, cfg, split_k);
    }
    cudaEvent_t begin = nullptr, end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess) return -1.0;
    if (cudaEventCreate(&end) != cudaSuccess) {
        cudaEventDestroy(begin);
        return -1.0;
    }
    double times[kMmqTuneTimed];
    int valid = 0;
    for (int i = 0; i < kMmqTuneTimed; ++i) {
        times[i] = 0.0;
        if (cudaEventRecord(begin, nullptr) != cudaSuccess) continue;
        const bool ok = mmq_up_matmul(b.dw, s.type, b.dyq8, b.dout, s.rows,
                                      s.cols, n, nullptr, cfg, split_k);
        if (cudaEventRecord(end, nullptr) != cudaSuccess) continue;
        if (cudaEventSynchronize(end) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        float ms = 0.0f;
        if (!ok || cudaEventElapsedTime(&ms, begin, end) != cudaSuccess ||
            cudaGetLastError() != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        times[valid++] = double(ms);
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (valid == 0) return -1.0;
    std::sort(times, times + valid);
    return times[valid / 2];
}

// One shape's sweep: the tile configuration priced at the decode row count
// (the narrow tile) and at the widest verify row count (the wide tile). The
// split count is pinned to 1 for every shape.
struct MmqTuneResult {
    double narrow[MMQ_CFG_COUNT] = {};
    double wide[MMQ_CFG_COUNT] = {};
    bool have_narrow[MMQ_CFG_COUNT] = {};
    bool have_wide[MMQ_CFG_COUNT] = {};
};

// The configurations worth timing for this shape: any that is neutral, fits
// the device, and has a table entry at the tile width it would be launched
// with.  The reduced set keeps the two (I, nthreads) pairs of the budget
// fallback.
int mmq_tune_cfgs(const MmqTuneShape & s, int J, bool reduced, int * out) {
    int count = 0;
    for (int c = 0; c < MMQ_CFG_COUNT; ++c) {
        if (reduced) {
            bool listed = false;
            for (int rc : kMmqTuneCfgsReduced) listed = listed || rc == c;
            if (!listed) continue;
        }
        if (mmq_up_cfg_allowed(s.type, s.rows, J, c)) out[count++] = c;
    }
    return count;
}

MmqTuneResult mmq_tune_sweep(const MmqTuneShape & s, const MmqTuneBuffers & b,
                             bool reduced) {
    MmqTuneResult r;
    int cfgs8[MMQ_CFG_COUNT], cfgs16[MMQ_CFG_COUNT];
    const int n8 = mmq_tune_cfgs(s, 8, reduced, cfgs8);
    const int n16 = mmq_tune_cfgs(s, 16, reduced, cfgs16);
    for (int ci = 0; ci < n8; ++ci) {
        const int cfg = cfgs8[ci];
        const double ms = mmq_tune_time(s, b, kMmqTuneNarrowN, cfg, 1);
        if (ms < 0.0) continue;
        r.narrow[cfg] = ms;
        r.have_narrow[cfg] = true;
    }
    for (int ci = 0; ci < n16; ++ci) {
        const int cfg = cfgs16[ci];
        const double ms = mmq_tune_time(s, b, kMmqTuneWideN, cfg, 1);
        if (ms < 0.0) continue;
        r.wide[cfg] = ms;
        r.have_wide[cfg] = true;
    }
    return r;
}

int mmq_pick_cfg(const MmqTuneResult & r, int J, const int * cfgs, int ncfgs) {
    int best = -1;
    double best_ms = 0.0;
    for (int ci = 0; ci < ncfgs; ++ci) {
        const int cfg = cfgs[ci];
        const bool have = J == 8 ? r.have_narrow[cfg] : r.have_wide[cfg];
        if (!have) continue;
        const double ms = J == 8 ? r.narrow[cfg] : r.wide[cfg];
        if (best < 0 || ms < best_ms) {
            best = cfg;
            best_ms = ms;
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Split-K counts.  The candidates are 1 (single pass), 2 and 4; a candidate
// whose partial buffer at the widest measured column count would exceed 32 MB
// is not a candidate for that shape.  The choice minimizes the sum of the cost
// ratios at n = 1, 8 and 512; a candidate within 2% of the minimum keeps the
// smaller count.
// ---------------------------------------------------------------------------

constexpr int kMmqSplitCandidates[] = {1, 2, 4};
constexpr int kMmqSplitCandidateCount = 3;
constexpr int kMmqSplitProbeN[] = {1, 8, 512};
constexpr int kMmqSplitProbeCount = 3;
constexpr int64_t kMmqSplitMaxScratch = int64_t(32) << 20;
// A larger candidate must beat the incumbent by more than 2% to displace it.
constexpr double kMmqSplitMinGain = 0.98;

bool mmq_split_candidate(int64_t rows, int split_k) {
    return split_k <= 1 ||
           double(split_k) * double(kMmqSplitProbeN[2]) * double(rows) *
                   sizeof(float) <=
               double(kMmqSplitMaxScratch);
}

const char * mmq_split_name(int split_k) {
    return split_k == 1 ? "1" : (split_k == 2 ? "2" : (split_k == 4 ? "4" : "?"));
}

// The smallest candidate whose cost is within the tie window of the minimum.
int mmq_split_choose(const double * cost, const bool * have) {
    int chosen = 0;
    if (!have[0]) {
        return chosen;
    }
    double best = cost[0];
    for (int ci = 1; ci < kMmqSplitCandidateCount; ++ci) {
        if (!have[ci]) continue;
        if (cost[ci] < best * kMmqSplitMinGain) {
            best = cost[ci];
            chosen = ci;
        }
    }
    return chosen;
}

// The pinned choice for one shape: the tile configuration at n = 1 (J = 8) and
// at n = 16 (J = 16) from the configuration sweep, then the split count whose
// t(1) + t(8) + t(512) cost ratio sum is the minimum.  A candidate within 2% of
// the minimum keeps the smaller split count; candidates whose partial buffer at
// n = 512 would exceed 32 MB are not considered.
MmqUpPin mmq_pin_measure(const MmqTuneShape & s, const MmqTuneBuffers & b,
                         bool reduced) {
    int cfgs8[MMQ_CFG_COUNT], cfgs16[MMQ_CFG_COUNT];
    const int n8 = mmq_tune_cfgs(s, 8, reduced, cfgs8);
    const int n16 = mmq_tune_cfgs(s, 16, reduced, cfgs16);
    const MmqTuneResult r = mmq_tune_sweep(s, b, reduced);
    MmqUpPin pin;
    const int c8 = mmq_pick_cfg(r, 8, cfgs8, n8);
    if (c8 < 0) return pin;  // nothing measured: keep the defaults
    pin.cfg_j8 = c8;
    const int c16 = mmq_pick_cfg(r, 16, cfgs16, n16);
    pin.cfg_j16 = c16 >= 0 ? c16 : pin.cfg_j8;

    double t[kMmqSplitCandidateCount][kMmqSplitProbeCount] = {};
    bool have[kMmqSplitCandidateCount] = {};
    double cost[kMmqSplitCandidateCount] = {};
    for (int ci = 0; ci < kMmqSplitCandidateCount; ++ci) {
        const int S = kMmqSplitCandidates[ci];
        if (!mmq_split_candidate(s.rows, S)) continue;
        bool ok = true;
        for (int pi = 0; pi < kMmqSplitProbeCount && ok; ++pi) {
            const int nprobe = kMmqSplitProbeN[pi];
            const int cfg = nprobe <= 8 ? pin.cfg_j8 : pin.cfg_j16;
            t[ci][pi] = mmq_tune_time(s, b, nprobe, cfg, S);
            ok = t[ci][pi] >= 0.0;
        }
        have[ci] = ok;
        if (ok) {
            for (int pi = 0; pi < kMmqSplitProbeCount; ++pi) {
                if (t[0][pi] <= 0.0) {
                    have[ci] = false;
                    break;
                }
                cost[ci] += t[ci][pi] / t[0][pi];
            }
        }
    }
    pin.split_k = kMmqSplitCandidates[mmq_split_choose(cost, have)];
    if (std::getenv("SONICBOOM_MMQ_TUNE_VERBOSE") != nullptr) {
        std::fprintf(stderr,
                     "mmq-pick %s %lldx%lld S=%d cfg8=%d cfg16=%d "
                     "t1=%.4f t16=%.4f",
                     mmq_type_name(s.type), (long long)s.rows,
                     (long long)s.cols, pin.split_k, pin.cfg_j8, pin.cfg_j16,
                     r.narrow[pin.cfg_j8], r.wide[pin.cfg_j16]);
        for (int ci = 0; ci < kMmqSplitCandidateCount; ++ci) {
            if (!have[ci]) continue;
            std::fprintf(stderr, " S=%s:%.3f/%.3f/%.3f=>%.3f",
                         mmq_split_name(kMmqSplitCandidates[ci]), t[ci][0],
                         t[ci][1], t[ci][2], cost[ci]);
        }
        std::fprintf(stderr, "\n");
    }
    return pin;
}

std::string mmq_splitk_key(const MmqTuneShape & s) {
    return "mmq-splitk|" + mmq_pin_key(s.type, s.rows, s.cols);
}

std::string mmq_cfg_key(const MmqTuneShape & s, int J) {
    return "mmq-cfg|" + mmq_pin_key(s.type, s.rows, s.cols) + "|J=" +
           std::to_string(J);
}

// A stored choice is usable only when all three records are present, carry
// this build's fingerprint, and still pass the neutrality/device filters.
bool mmq_pin_from_cache(const MmqTuneShape & s, int build_key, MmqUpPin * pin) {
    if (!tune::enabled() || build_key == 0) return false;
    const auto rec = tune::lookup("mmq-splitk", mmq_splitk_key(s));
    const auto si = rec.find("S");
    const auto ki = rec.find("k");
    if (si == rec.end() || ki == rec.end() || ki->second != build_key) {
        return false;
    }
    // Only the three candidates are valid split counts.
    const int S = si->second;
    if (S != 1 && S != 2 && S != 4) return false;
    int cfg[2] = {0, 0};
    for (int jj = 0; jj < 2; ++jj) {
        const int J = jj == 0 ? 8 : 16;
        const auto crec = tune::lookup("mmq-cfg", mmq_cfg_key(s, J));
        const auto ci = crec.find("cfg");
        const auto ck = crec.find("k");
        if (ci == crec.end() || ck == crec.end() || ck->second != build_key) {
            return false;
        }
        if (!mmq_up_cfg_allowed(s.type, s.rows, J, ci->second)) return false;
        cfg[jj] = ci->second;
    }
    pin->split_k = S;
    pin->cfg_j8 = cfg[0];
    pin->cfg_j16 = cfg[1];
    return true;
}

void mmq_pin_store(const MmqTuneShape & s, const MmqUpPin & pin,
                   int build_key) {
    if (!tune::enabled() || build_key == 0) return;
    tune::store("mmq-splitk", mmq_splitk_key(s),
                {{"S", pin.split_k}, {"k", build_key}});
    tune::store("mmq-cfg", mmq_cfg_key(s, 8),
                {{"cfg", pin.cfg_j8}, {"k", build_key}});
    tune::store("mmq-cfg", mmq_cfg_key(s, 16),
                {{"cfg", pin.cfg_j16}, {"k", build_key}});
}

// --- sibling groups --------------------------------------------------------

std::mutex g_mmq_group_pin_mutex;
std::map<std::string, int> g_mmq_group_pin;

std::string mmq_group_key(const MmqUpType * types, const int64_t * rows,
                          int n_members, int64_t cols) {
    std::string k;
    for (int i = 0; i < n_members; ++i) {
        if (i) k += ",";
        k += mmq_type_name(types[i]);
    }
    k += "|";
    for (int i = 0; i < n_members; ++i) {
        if (i) k += ",";
        k += std::to_string(rows[i]);
    }
    k += "x" + std::to_string(cols);
    return k;
}

std::string mmq_group_splitk_key(const MmqUpType * types, const int64_t * rows,
                                 int n_members, int64_t cols) {
    return "mmq-splitk|" + mmq_group_key(types, rows, n_members, cols);
}

struct MmqGroupTuneBuffers {
    void *  dw[3] = {};
    float * dout[3] = {};
    int32_t * dyq8 = nullptr;
};

void mmq_group_tune_free(MmqGroupTuneBuffers & b) {
    for (int k = 0; k < 3; ++k) {
        if (b.dout[k]) cudaFree(b.dout[k]);
        if (b.dw[k]) cudaFree(b.dw[k]);
    }
    if (b.dyq8) cudaFree(b.dyq8);
    b = {};
}

bool mmq_group_tune_alloc(const MmqUpType * types, const int64_t * rows,
                          int n_members, int64_t cols,
                          MmqGroupTuneBuffers & b) {
    const int64_t planes = cols / 128 + ((cols / 128) & 1);
    const size_t yblocks = size_t(planes) * size_t(kMmqSplitProbeN[2]) + 128;
    const size_t ybytes = yblocks * sizeof(int) * 36;  // block_q8_1_mmq
    if (cudaMalloc(&b.dyq8, ybytes) != cudaSuccess) {
        mmq_group_tune_free(b);
        return false;
    }
    cudaMemset(b.dyq8, 0, ybytes);
    for (int k = 0; k < n_members; ++k) {
        int qk = 0, bytes = 0;
        mmq_block_meta(types[k], &qk, &bytes);
        const size_t wbytes = size_t(rows[k]) * size_t(cols / qk) * size_t(bytes);
        const size_t obytes = size_t(rows[k]) * size_t(kMmqSplitProbeN[2]) *
                              sizeof(float);
        if (cudaMalloc(&b.dw[k], wbytes) != cudaSuccess ||
            cudaMalloc(&b.dout[k], obytes) != cudaSuccess) {
            mmq_group_tune_free(b);
            return false;
        }
        cudaMemset(b.dw[k], 0, wbytes);
    }
    return true;
}

sonicboom::quant::cuda::MmqGroupArgs mmq_group_tune_args(
        const MmqGroupTuneBuffers & b, const MmqUpType * types,
        const int64_t * rows, int n_members, int64_t cols, int n) {
    sonicboom::quant::cuda::MmqGroupArgs a{};
    a.n_members = n_members;
    a.ncols_dst = n;
    a.ncols_y = n;
    for (int k = 0; k < n_members; ++k) {
        int qk = 0, bytes = 0;
        mmq_block_meta(types[k], &qk, &bytes);
        auto & m = a.m[k];
        m.x = b.dw[k];
        m.y = b.dyq8;
        m.dst = b.dout[k];
        m.bias = nullptr;
        m.add = nullptr;
        m.nrows = int(rows[k]);
        m.stride_col_dst = int(rows[k]);
        m.blocks_per_row = int(cols / qk);
        m.stride_row_x = m.blocks_per_row;
    }
    return a;
}

// Median of the timed group launches (3 warm-ups, 7 timings).
double mmq_group_tune_time(const sb_cuda_props & props, const MmqUpType * types,
                           const MmqGroupTuneBuffers & b, const int64_t * rows,
                           int n_members, int64_t cols, int n, int split_k) {
    const auto args = mmq_group_tune_args(b, types, rows, n_members, cols, n);
    const auto launch = [&] {
        return group_dispatch(props, args, types, split_k, nullptr,
                              /*warm=*/false);
    };
    for (int i = 0; i < kMmqTuneWarm; ++i) {
        (void)launch();
    }
    cudaEvent_t begin = nullptr, end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess) return -1.0;
    if (cudaEventCreate(&end) != cudaSuccess) {
        cudaEventDestroy(begin);
        return -1.0;
    }
    double times[kMmqTuneTimed];
    int valid = 0;
    for (int i = 0; i < kMmqTuneTimed; ++i) {
        times[i] = 0.0;
        if (cudaEventRecord(begin, nullptr) != cudaSuccess) continue;
        const bool ok = launch();
        if (cudaEventRecord(end, nullptr) != cudaSuccess) continue;
        if (cudaEventSynchronize(end) != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        float ms = 0.0f;
        if (!ok || cudaEventElapsedTime(&ms, begin, end) != cudaSuccess ||
            cudaGetLastError() != cudaSuccess) {
            (void)cudaGetLastError();
            continue;
        }
        times[valid++] = double(ms);
    }
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (valid == 0) return -1.0;
    std::sort(times, times + valid);
    return times[valid / 2];
}

bool g_mmq_warmed = false;

}  // namespace

MmqUpPin mmq_up_pin_for(MmqUpType type, int64_t M, int64_t K) {
    std::lock_guard<std::mutex> lock(g_mmq_pin_mutex);
    const auto it = g_mmq_pin.find(mmq_pin_key(type, M, K));
    return it != g_mmq_pin.end() ? it->second : MmqUpPin{};
}

void mmq_up_warm(const QuantizedTensor * const * weights, size_t count,
                 const char * build_id) {
    if (!weights || count == 0) return;
    // The debug pin fixes the configuration, so there is nothing to choose.
    if (std::getenv("SONICBOOM_MMQ_CFG") != nullptr) return;
    {
        std::lock_guard<std::mutex> lock(g_mmq_pin_mutex);
        if (g_mmq_warmed) return;
        g_mmq_warmed = true;
    }
    const int build_key =
        (build_id && *build_id) ? mmq_build_hash(build_id) : 0;

    // Distinct projection shapes, largest first: when the load budget runs
    // out, the shapes that dominate a decode step are the ones already
    // priced.  Shapes the matmul cannot take (K not a multiple of 128) are
    // not candidates.
    std::vector<MmqTuneShape> shapes;
    for (size_t i = 0; i < count; ++i) {
        const QuantizedTensor * w = weights[i];
        if (!w || !w->valid() || w->dims().size() != 2) continue;
        const int64_t cols = int64_t(w->dims()[0]);
        const int64_t rows = int64_t(w->dims()[1]);
        if (cols % 128 != 0 || rows <= 0) continue;
        const std::optional<MmqUpType> type = [&]() -> std::optional<MmqUpType> {
            switch (w->type()) {
                case QuantType::Q3_K: return MmqUpType::Q3_K;
                case QuantType::Q4_K: return MmqUpType::Q4_K;
                case QuantType::Q5_K: return MmqUpType::Q5_K;
                case QuantType::Q6_K: return MmqUpType::Q6_K;
                case QuantType::Q5_0: return MmqUpType::Q5_0;
                case QuantType::Q8_0: return MmqUpType::Q8_0;
            }
            return std::nullopt;
        }();
        if (!type) continue;
        bool seen = false;
        for (const MmqTuneShape & s : shapes) {
            if (s.type == *type && s.rows == rows && s.cols == cols) {
                seen = true;
                break;
            }
        }
        if (!seen) shapes.push_back({*type, rows, cols});
    }
    std::sort(shapes.begin(), shapes.end(),
              [](const MmqTuneShape & a, const MmqTuneShape & b) {
                  return a.rows * a.cols > b.rows * b.cols;
              });

    const auto t0 = std::chrono::steady_clock::now();
    const auto elapsed_s = [&t0] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t0)
            .count();
    };
    bool reduced_mode = false;
    for (const MmqTuneShape & s : shapes) {
        MmqUpPin pin;
        const bool cached = mmq_pin_from_cache(s, build_key, &pin);
        if (!cached) {
            if (!reduced_mode && elapsed_s() > kMmqTuneBudgetS) {
                reduced_mode = true;
                std::fprintf(stderr,
                             "mmq_up_warm: %.2fs load budget reached; "
                             "measuring the remaining shapes with the reduced "
                             "set (cfgs {5,6})\n",
                             elapsed_s());
            }
            MmqTuneBuffers b;
            if (mmq_tune_alloc(s, b)) {
                pin = mmq_pin_measure(s, b, reduced_mode);
                mmq_tune_free(b);
                mmq_pin_store(s, pin, build_key);
            }
        }
        std::lock_guard<std::mutex> lock(g_mmq_pin_mutex);
        g_mmq_pin[mmq_pin_key(s.type, s.rows, s.cols)] = pin;
    }
}

int mmq_up_group_split_for(const MmqUpType * types, const int64_t * rows,
                           int n_members, int64_t cols) {
    if (types == nullptr || rows == nullptr || n_members < 2 || n_members > 3) {
        return 1;
    }
    std::lock_guard<std::mutex> lock(g_mmq_group_pin_mutex);
    const auto it = g_mmq_group_pin.find(mmq_group_key(types, rows, n_members, cols));
    return it != g_mmq_group_pin.end() ? it->second : 1;
}

void mmq_up_group_warm_split(const MmqUpType * types, const int64_t * rows,
                             int n_members, int64_t cols, const char * build_id) {
    if (types == nullptr || rows == nullptr || n_members < 2 || n_members > 3) {
        return;
    }
    if (std::getenv("SONICBOOM_MMQ_CFG") != nullptr) return;
    const std::string key = mmq_group_key(types, rows, n_members, cols);
    {
        std::lock_guard<std::mutex> lock(g_mmq_group_pin_mutex);
        if (g_mmq_group_pin.count(key) != 0) return;  // already warmed
    }

    const int build_key =
        (build_id && *build_id) ? mmq_build_hash(build_id) : 0;
    int split = 1;
    bool cached = false;
    if (tune::enabled() && build_key != 0) {
        const auto rec = tune::lookup(
            "mmq-splitk", mmq_group_splitk_key(types, rows, n_members, cols));
        const auto si = rec.find("S");
        const auto ki = rec.find("k");
        if (si != rec.end() && ki != rec.end() && ki->second == build_key &&
            (si->second == 1 || si->second == 2 || si->second == 4)) {
            split = si->second;
            cached = true;
        }
    }

    if (!cached) {
        const sb_cuda_props props = get_props();
        // Raise the group kernels' shared-memory limits before the first timed
        // launch; only the attribute calls matter here.
        {
            sonicboom::quant::cuda::MmqGroupArgs warm{};
            warm.n_members = n_members;
            (void)group_dispatch(props, warm, types, 1, nullptr, /*warm=*/true);
        }
        int64_t total_rows = 0;
        for (int k = 0; k < n_members; ++k) total_rows += rows[k];

        MmqGroupTuneBuffers b;
        double t[kMmqSplitCandidateCount][kMmqSplitProbeCount] = {};
        bool have[kMmqSplitCandidateCount] = {};
        double cost[kMmqSplitCandidateCount] = {};
        if (mmq_group_tune_alloc(types, rows, n_members, cols, b)) {
            for (int ci = 0; ci < kMmqSplitCandidateCount; ++ci) {
                const int S = kMmqSplitCandidates[ci];
                if (!mmq_split_candidate(total_rows, S)) continue;
                bool ok = true;
                for (int pi = 0; pi < kMmqSplitProbeCount && ok; ++pi) {
                    t[ci][pi] = mmq_group_tune_time(props, types, b, rows,
                                                    n_members, cols,
                                                    kMmqSplitProbeN[pi], S);
                    ok = t[ci][pi] >= 0.0;
                }
                have[ci] = ok;
                if (ok) {
                    for (int pi = 0; pi < kMmqSplitProbeCount; ++pi) {
                        if (t[0][pi] <= 0.0) {
                            have[ci] = false;
                            break;
                        }
                        cost[ci] += t[ci][pi] / t[0][pi];
                    }
                }
            }
            mmq_group_tune_free(b);
        }
        split = kMmqSplitCandidates[mmq_split_choose(cost, have)];
        if (tune::enabled() && build_key != 0) {
            tune::store(
                "mmq-splitk", mmq_group_splitk_key(types, rows, n_members, cols),
                {{"S", split}, {"k", build_key}});
        }
        if (std::getenv("SONICBOOM_MMQ_TUNE_VERBOSE") != nullptr) {
            std::fprintf(stderr, "mmq-group-pick %s S=%d",
                         mmq_group_key(types, rows, n_members, cols).c_str(),
                         split);
            for (int ci = 0; ci < kMmqSplitCandidateCount; ++ci) {
                if (!have[ci]) continue;
                std::fprintf(stderr, " S=%s:%.3f/%.3f/%.3f=>%.3f",
                             mmq_split_name(kMmqSplitCandidates[ci]),
                             t[ci][0], t[ci][1], t[ci][2], cost[ci]);
            }
            std::fprintf(stderr, "\n");
        }
    }

    std::lock_guard<std::mutex> lock(g_mmq_group_pin_mutex);
    g_mmq_group_pin[key] = split;
}

}  // namespace sonicboom::quant::cuda
