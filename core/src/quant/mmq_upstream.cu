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
             int split_k, const float * bias, const float * add) {
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
            n_cnt, n_cnt,
            bias, add == nullptr ? nullptr : add + n_lo * M};

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

// The row counts a decode step (n = 1) and a verify step (n = k+1 <= 16) can
// carry, and the large-n width whose cost split-K may not disturb.
constexpr int kMmqTuneNs[]     = {1, 8, 16};
constexpr int kMmqTuneBigN     = 512;
constexpr int kMmqTuneSplits[] = {1, 2, 4, 8};
constexpr int kMmqTuneSplitsReduced[] = {1, 4};
// (I, nthreads) = (64, 256) and (128, 256).
constexpr int kMmqTuneCfgsReduced[]   = {5, 6};
constexpr double kMmqTuneBudgetS = 1.5;
constexpr double kMmqTuneBigSlack = 1.03;
constexpr double kMmqTuneTieSlack = 1.02;
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
    const size_t xbytes = size_t(kMmqTuneBigN) * size_t(s.cols) * sizeof(float);
    const size_t yblocks = size_t(s.cols / 128) * size_t(kMmqTuneBigN) + 128;
    const size_t ybytes = yblocks * sizeof(int) * 36;  // block_q8_1_mmq
    const size_t obytes = size_t(s.rows) * size_t(kMmqTuneBigN) * sizeof(float);
    if (cudaMalloc(&b.dw, wbytes) != cudaSuccess) return false;
    if (cudaMalloc(&b.dx, xbytes) != cudaSuccess) { mmq_tune_free(b); return false; }
    if (cudaMalloc(&b.dyq8, ybytes) != cudaSuccess) { mmq_tune_free(b); return false; }
    if (cudaMalloc(&b.dout, obytes) != cudaSuccess) { mmq_tune_free(b); return false; }
    cudaMemset(b.dw, 0, wbytes);
    cudaMemset(b.dx, 0, xbytes);
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

// Position of a split count in the fixed candidate array.
int mmq_split_index(int S) {
    return S == 1 ? 0 : (S == 2 ? 1 : (S == 4 ? 2 : 3));
}

// One shape's sweep; fills t[n][S][cfg] for the small n and big[S] at the
// default configuration.
struct MmqTuneResult {
    double t[3][4][MMQ_CFG_COUNT] = {};
    double big[4] = {};
    bool have[3][4][MMQ_CFG_COUNT] = {};
    bool have_big[4] = {};
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
    const int * splits = reduced ? kMmqTuneSplitsReduced : kMmqTuneSplits;
    const int nsplit = reduced ? 2 : 4;
    for (int si = 0; si < nsplit; ++si) {
        const int S = splits[si];
        const int idx = mmq_split_index(S);
        const int ni[3] = {0, 1, 2};
        const int * lists[3] = {cfgs8, cfgs8, cfgs16};
        const int counts[3] = {n8, n8, n16};
        for (int k = 0; k < 3; ++k) {
            for (int ci = 0; ci < counts[k]; ++ci) {
                const int cfg = lists[k][ci];
                const double ms =
                    mmq_tune_time(s, b, kMmqTuneNs[ni[k]], cfg, S);
                if (ms < 0.0) continue;
                r.t[k][idx][cfg] = ms;
                r.have[k][idx][cfg] = true;
            }
        }
        const double big = mmq_tune_time(s, b, kMmqTuneBigN, 0, S);
        if (big >= 0.0) {
            r.big[idx] = big;
            r.have_big[idx] = true;
            std::fprintf(stderr, "mmq-tune %s %lldx%lld S=%d t512=%.3fms\n",
                         mmq_type_name(s.type), (long long)s.rows,
                         (long long)s.cols, S, big);
        }
    }
    return r;
}

int mmq_pick_cfg(const MmqTuneResult & r, int ni, int idx,
                 const int * cfgs, int ncfgs) {
    int best = -1;
    double best_ms = 0.0;
    for (int ci = 0; ci < ncfgs; ++ci) {
        const int cfg = cfgs[ci];
        if (!r.have[ni][idx][cfg]) continue;
        if (best < 0 || r.t[ni][idx][cfg] < best_ms) {
            best = cfg;
            best_ms = r.t[ni][idx][cfg];
        }
    }
    return best;
}

// The pinned choice for one shape: the split count that minimizes t(1) + t(8)
// at its best tile configuration, among the split counts that do not slow the
// n = 512 call by more than 3%; a candidate within 2% of the incumbent keeps
// the smaller split count.  The tile configuration at each split count is the
// fastest at n = 1 (J = 8) and at n = 16 (J = 16).
MmqUpPin mmq_pin_measure(const MmqTuneShape & s, const MmqTuneBuffers & b,
                         bool reduced) {
    int cfgs8[MMQ_CFG_COUNT], cfgs16[MMQ_CFG_COUNT];
    const int n8 = mmq_tune_cfgs(s, 8, reduced, cfgs8);
    const int n16 = mmq_tune_cfgs(s, 16, reduced, cfgs16);
    const int * splits = reduced ? kMmqTuneSplitsReduced : kMmqTuneSplits;
    const int nsplit = reduced ? 2 : 4;
    const MmqTuneResult r = mmq_tune_sweep(s, b, reduced);
    MmqUpPin pin;
    int best_idx = -1;
    double best_score = 0.0;
    for (int si = 0; si < nsplit; ++si) {
        const int idx = mmq_split_index(splits[si]);
        if (!r.have_big[idx] || !r.have_big[0]) continue;
        if (r.big[idx] > r.big[0] * kMmqTuneBigSlack) continue;
        const int c8 = mmq_pick_cfg(r, 0, idx, cfgs8, n8);
        if (c8 < 0 || !r.have[1][idx][c8]) continue;
        const double score = r.t[0][idx][c8] + r.t[1][idx][c8];
        // Ascending split counts: a larger split replaces the incumbent only
        // when it is more than 2% faster, so a tie keeps the smaller one.
        if (best_idx < 0 || score * kMmqTuneTieSlack < best_score) {
            best_idx = idx;
            best_score = score;
            pin.cfg_j8 = c8;
        }
    }
    if (best_idx < 0) return pin;  // nothing measured: keep the defaults
    pin.split_k = kMmqTuneSplits[best_idx];
    const int c16 = mmq_pick_cfg(r, 2, best_idx, cfgs16, n16);
    pin.cfg_j16 = c16 >= 0 ? c16 : pin.cfg_j8;
    if (std::getenv("SONICBOOM_MMQ_TUNE_VERBOSE") != nullptr) {
        const double t1  = r.have[0][best_idx][pin.cfg_j8] ? r.t[0][best_idx][pin.cfg_j8] : -1.0;
        const double t8  = r.have[1][best_idx][pin.cfg_j8] ? r.t[1][best_idx][pin.cfg_j8] : -1.0;
        const double t16 = r.have[2][best_idx][pin.cfg_j16] ? r.t[2][best_idx][pin.cfg_j16] : -1.0;
        std::fprintf(stderr,
                     "mmq-pick %s %lldx%lld S=%d cfg8=%d cfg16=%d "
                     "t1=%.4f t8=%.4f t16=%.4f t512=%.4f t512_s1=%.4f\n",
                     mmq_type_name(s.type), (long long)s.rows,
                     (long long)s.cols, pin.split_k, pin.cfg_j8, pin.cfg_j16,
                     t1, t8, t16, r.big[best_idx], r.big[0]);
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
    const int S = si->second;
    if (S != 1 && S != 2 && S != 4 && S != 8) return false;
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

bool g_mmq_warmed = false;

}  // namespace

bool mmq_up_decode_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("SONICBOOM_DECODE_MMQ");
        return v != nullptr && (std::strcmp(v, "1") == 0 ||
                                std::strcmp(v, "on") == 0 ||
                                std::strcmp(v, "true") == 0);
    }();
    return enabled;
}

MmqUpPin mmq_up_pin_for(MmqUpType type, int64_t M, int64_t K) {
    std::lock_guard<std::mutex> lock(g_mmq_pin_mutex);
    const auto it = g_mmq_pin.find(mmq_pin_key(type, M, K));
    return it != g_mmq_pin.end() ? it->second : MmqUpPin{};
}

void mmq_up_warm(const QuantizedTensor * const * weights, size_t count,
                 const char * build_id) {
    if (!mmq_up_decode_enabled() || !weights || count == 0) return;
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
                             "set (S in {1,4}, cfgs {5,6})\n",
                             elapsed_s());
            }
            MmqTuneBuffers b;
            if (mmq_tune_alloc(s, b)) {
                pin = mmq_pin_measure(s, b, reduced_mode);
                mmq_tune_free(b);
                mmq_pin_store(s, pin, build_key);
            }
        }
        // Reach the split-K scratch high-water mark for this shape now, so no
        // generation step ever grows the buffer (and no capture allocates).
        if (pin.split_k > 1) {
            (void)split_scratch_get(size_t(pin.split_k) * 16 *
                                    size_t(s.rows) * sizeof(float));
        }
        std::lock_guard<std::mutex> lock(g_mmq_pin_mutex);
        g_mmq_pin[mmq_pin_key(s.type, s.rows, s.cols)] = pin;
    }
}

}  // namespace sonicboom::quant::cuda
