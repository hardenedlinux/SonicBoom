// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Standalone K-quant gemv microbenchmark (dev-time only; not part of the CMake
// build). Measures the internal quant::cuda::matvec_f32_dev kernel in isolation
// — NOT end-to-end tok/s — across the Gemma 4 E4B-it matmul shapes, so we can
// see how much of the decode-time cost is the dequantize-on-the-fly gemv and how
// it scales with (cols, rows, blocks_per_row). Each tensor is timed at its
// ACTUAL GGUF quant type (Unsloth Q3_K_M: q3_K for q/k/gate/up, q4_K for
// o/v/down), and every line reports effective bandwidth (GB/s = weight bytes /
// kernel time).
//
// Per-shape it (1) times the device-resident kernel with CUDA events using an
// adaptive iteration count (the first call's one-time weight upload is excluded
// by warm-up), and (2) diffs the CUDA output against the scalar CPU reference
// quant::matvec_f32 (f32-activation semantics) within an fp32 tolerance.
//
// It links only libsonicboom.so (the SHARED core) + libcudart, and includes the
// internal core/src/quant/cuda_resident.h via relative path (matvec_f32_dev is
// not on the public Layer 2 surface). The synthetic weights use the same
// construction as tests/core/test_cuda_quant.cpp (random-but-finite block bytes,
// finite half d/dmin scales), so dequant does the same work as real GGUF bytes.

#include <cuda_runtime.h>

#include <sonicboom/dtype.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/quant/quantized_matmul.h>       // matvec_f32 (CPU ref)
#include <sonicboom/quant/quantized_matmul_cuda.h>  // available()
#include <sonicboom/quant/quantized_tensor.h>

#include "../../core/src/nn/cuda_resident.h"
#include "../../core/src/quant/cuda_resident.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace sbquant = sonicboom::quant;
namespace sbcuda = sonicboom::quant::cuda;
namespace sbnn = sonicboom::nn;
namespace sbnncuda = sonicboom::nn::cuda;

namespace {

uint16_t rand_finite_half(std::mt19937& rng) {
  uint16_t h;
  do { h = uint16_t(rng()); } while (((h >> 10) & 0x1F) == 0x1F);
  return h;
}

// Same construction as tests/core/test_cuda_quant.cpp: random block bytes with
// valid finite d (Q3_K) / d + dmin (Q4_K/Q5_K) scales so the packed weight is a
// well-formed QuantizedTensor that dequantizes to finite f32.
std::vector<std::byte> build_weight(sbquant::QuantType type, uint64_t rows,
                                    uint64_t cols, std::mt19937& rng) {
  const sbquant::QuantTypeInfo* info = sbquant::quant_type_info(type);
  const uint64_t blocks_per_row = (cols + info->block_size - 1) / info->block_size;
  const uint64_t n_blocks = rows * blocks_per_row;
  const uint64_t total = n_blocks * info->bytes_per_block;

  std::vector<std::byte> bytes(total);
  for (auto& b : bytes) b = std::byte(uint8_t(rng()));

  for (uint64_t i = 0; i < n_blocks; ++i) {
    const size_t off = i * info->bytes_per_block;
    if (type == sbquant::QuantType::Q3_K) {
      const uint16_t d = rand_finite_half(rng);
      bytes[off + 108] = std::byte(d & 0xFF);
      bytes[off + 109] = std::byte(d >> 8);
    } else {
      const uint16_t d = rand_finite_half(rng);
      const uint16_t dmin = rand_finite_half(rng);
      bytes[off + 0] = std::byte(d & 0xFF);
      bytes[off + 1] = std::byte(d >> 8);
      bytes[off + 2] = std::byte(dmin & 0xFF);
      bytes[off + 3] = std::byte(dmin >> 8);
    }
  }
  return bytes;
}

double compare(std::span<const float> got, std::span<const float> ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double denom = std::max(1.0, double(std::fabs(ref[i])));
    worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])) / denom);
  }
  return worst;
}

struct Shape {
  const char* name;
  uint64_t cols;              // contiguous / inner dim (W.dims()[0])
  uint64_t rows;              // output dim (W.dims()[1])
  sbquant::QuantType type;    // actual GGUF quant type (dominant block type)
  int count;                  // occurrences per decoder block
  bool mixed;                 // true when blocks 0-1 are q5_K (Unsloth Q3_K_M)
};

// Gemma 4 E4B-it matmul shapes (d=2560, ff=10240, n_heads_q=8, n_heads_kv=2,
// head_dim=256 for the 24 SWA layers 0-23) with the ACTUAL per-tensor GGUF
// quant type read from models/gemma-4-E4B-it-Q3_K_M.gguf (Unsloth Q3_K_M):
//   token_embd / attn_q / attn_k / ffn_gate / ffn_up = q3_K
//   per_layer_token_embd / attn_output / attn_v / ffn_down = q4_K
//   attn_v / ffn_down are q5_K in blocks 0-1 (mixed = true), q4_K elsewhere.
const Shape kShapes[] = {
    {"ffn_gate", 2560, 10240, sbquant::QuantType::Q3_K, 1, false},
    {"ffn_up", 2560, 10240, sbquant::QuantType::Q3_K, 1, false},
    {"ffn_down", 10240, 2560, sbquant::QuantType::Q4_K, 1, true},
    {"attn_q", 2560, 2048, sbquant::QuantType::Q3_K, 1, false},
    {"attn_k", 2560, 512, sbquant::QuantType::Q3_K, 1, false},
    {"attn_v", 2560, 512, sbquant::QuantType::Q4_K, 1, true},
    {"attn_output", 2048, 2560, sbquant::QuantType::Q4_K, 1, false},
};

// Timing methodology (fixes the idle-clock measurement artifact). The previous
// version warmed up with only 5 calls, then timed a *single* call to size an
// adaptive iteration count. A single small kernel launched right after a sync
// runs at the GPU's idle clock (~210 MHz vs ~2.1 GHz boost) and pays SM
// power-gating wake-up, so its "one call" time was ~10-30x the true kernel
// time; the adaptive N then shrank accordingly, and the batch stayed at the
// low clock.
//
// The fix: a fixed large warmup and a fixed large timed batch (2000 calls
// each), with no sync in between, so the clock is warm throughout the measured
// region. 2000 calls is enough to ramp the clock even for the smallest
// (~15 us) kernels (2000 x 15 us = 30 ms of sustained load), and short enough
// that the largest kernels (~1.2 ms fp32 ffn_down) don't overheat/throttle.
constexpr int kWarmupCalls = 2000;
constexpr int kTimedCalls = 2000;

// Returns steady-state kernel time in µs/call (0.0 on failure).
double run(const Shape& s, sbquant::QuantType type, std::mt19937& rng) {
  const char* tname =
      type == sbquant::QuantType::Q3_K ? "Q3_K" :
      type == sbquant::QuantType::Q4_K ? "Q4_K" : "Q5_K";

  // The device weight cache (cuda_quant_matmul.cu upload_weight) is keyed by the
  // host data pointer. If this shape's `bytes` is freed at run()'s return and the
  // next shape's malloc reuses the same address at the same size (e.g. ffn_gate
  // and ffn_up are both Q3_K [2560,10240] = 11.26 MB), the cache would return the
  // previous shape's stale device copy — corrupting both the diff (err) and, for a
  // size-matched reuse, the timing. Keep every weight alive for the whole run so
  // each shape's host pointer stays unique and stable.
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(type, s.rows, s.cols, rng));
  const std::vector<std::byte>& bytes = keepalive.back();
  sbquant::QuantizedTensor W(type, {s.cols, s.rows}, bytes);
  if (!W.valid()) {
    std::printf("  %-14s [%6llu, %6llu] %s  INVALID\n", s.name,
                (unsigned long long)s.cols, (unsigned long long)s.rows, tname);
    return 0.0;
  }

  std::vector<float> x(s.cols), y(s.rows), y_ref(s.rows);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& v : x) v = ud(rng);

  if (!sbquant::matvec_f32(W, x, y_ref)) {
    std::printf("  %-14s [%6llu, %6llu] %s  CPU ref FAILED\n", s.name,
                (unsigned long long)s.cols, (unsigned long long)s.rows, tname);
    return 0.0;
  }

  float *dx = nullptr, *dy = nullptr;
  cudaMalloc(&dx, s.cols * sizeof(float));
  cudaMalloc(&dy, s.rows * sizeof(float));
  cudaMemcpy(dx, x.data(), s.cols * sizeof(float), cudaMemcpyHostToDevice);

  auto call = [&]() { sbcuda::matvec_f32_dev(W, dx, dy); };

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  // Warm up (the first call's one-time device weight upload is absorbed here,
  // not the timed batch). No sync after — a sync drains the GPU and drops the
  // clock back to idle, which is the whole artifact.
  for (int i = 0; i < kWarmupCalls; ++i) call();

  // Time the batch immediately after warmup (no sync in between), so the clock
  // stays warm across the measured region.
  cudaEventRecord(e0);
  for (int i = 0; i < kTimedCalls; ++i) call();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, e0, e1);
  const double us_per_call = double(total_ms) * 1000.0 / double(kTimedCalls);

  cudaMemcpy(y.data(), dy, s.rows * sizeof(float), cudaMemcpyDeviceToHost);
  const double err = compare(y, y_ref);

  // Effective bandwidth: the packed weight bytes this gemv actually reads
  // (num_blocks × bytes_per_block), over the measured kernel time. This is the
  // number to compare against the card's ~224 GB/s DRAM ceiling (or ~half of it
  // for a single-pass read-modify), not the unquantized f32 traffic.
  const double weight_bytes =
      double(W.num_blocks()) * double(sbquant::quant_type_info(type)->bytes_per_block);
  const double gbs = weight_bytes / (us_per_call * 1e-6) / 1e9;

  std::printf("  %-14s [%6llu, %6llu] %5s %7llu blk  %10.2f us  %8.1f GB/s  err %10.2e\n",
              s.name, (unsigned long long)s.cols, (unsigned long long)s.rows,
              tname, (unsigned long long)W.num_blocks(), us_per_call, gbs, err);

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(dx);
  cudaFree(dy);

  return us_per_call;
}

// --- dense (non-K-quant) gemv benchmark (Phase 2.3b) ------------------------
// The three dense f32/bf16 gate shapes the decode path exercises, timed via
// nn::cuda::matvec_*_dev (device-resident, no H2D/D2H). Weight bytes = the raw
// f32 (4B) / bf16 (2B) matrix this gemv streams; GB/s = weight bytes / time.

struct DenseShape {
  const char* name;
  uint64_t cols;  // contiguous / inner dim
  uint64_t rows;  // output dim
  bool bf16;      // true => bf16 weights, else f32
};

const DenseShape kDenseShapes[] = {
    {"per_layer_model_proj", 2560, 10752, true},  // 2560→10752 bf16
    {"inp_gate", 2560, 256, false},               // 2560→256 f32
    {"proj", 256, 2560, false},                   // 256→2560 f32
};

double run_dense(const DenseShape& s, std::mt19937& rng) {
  const size_t n = (size_t)s.cols * s.rows;
  std::vector<float> x(s.cols), y(s.rows), y_ref(s.rows);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& v : x) v = ud(rng);

  // Keep the synthetic weights alive for the whole run (upload_bytes caches by
  // host pointer; freeing + malloc reuse would return a stale device copy).
  static std::vector<std::vector<std::byte>> keep_bf16;
  static std::vector<std::vector<float>> keep_f32;

  float* dx = nullptr;
  float* dy = nullptr;
  cudaMalloc(&dx, s.cols * sizeof(float));
  cudaMalloc(&dy, s.rows * sizeof(float));
  cudaMemcpy(dx, x.data(), s.cols * sizeof(float), cudaMemcpyHostToDevice);

  double weight_bytes = 0.0;
  double us_per_call = 0.0;
  const char* tname = s.bf16 ? "bf16" : "f32";

  auto time_it = [&](auto&& call) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < kWarmupCalls; ++i) call();
    cudaEventRecord(e0);
    for (int i = 0; i < kTimedCalls; ++i) call();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, e0, e1);
    us_per_call = double(total_ms) * 1000.0 / double(kTimedCalls);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
  };

  if (s.bf16) {
    std::vector<std::byte> bytes(n * sizeof(uint16_t));
    auto* h = reinterpret_cast<uint16_t*>(bytes.data());
    for (size_t i = 0; i < n; ++i) h[i] = sonicboom::f32_to_bf16(ud(rng));
    // CPU reference: double-accumulation bf16 dot (matches nn::matvec_bf16).
    for (uint64_t j = 0; j < s.rows; ++j) {
      double acc = 0.0;
      for (uint64_t i = 0; i < s.cols; ++i)
        acc += double(sonicboom::bf16_to_f32(h[j * s.cols + i]) * x[i]);
      y_ref[j] = float(acc);
    }
    keep_bf16.push_back(std::move(bytes));
    const uint16_t* wp = reinterpret_cast<const uint16_t*>(keep_bf16.back().data());
    std::span<const uint16_t> W(wp, n);
    weight_bytes = double(n) * 2.0;
    time_it([&]() {
      sbnncuda::matvec_bf16_dev(W, int(s.cols), dx, dy, int(s.rows));
    });
  } else {
    std::vector<float> w(n);
    for (auto& v : w) v = ud(rng);
    if (!sbnn::matvec_f32(w, s.cols, x, y_ref)) {
      std::printf("  %-22s [%6llu, %6llu] f32  CPU ref FAILED\n", s.name,
                  (unsigned long long)s.cols, (unsigned long long)s.rows);
      cudaFree(dx);
      cudaFree(dy);
      return 0.0;
    }
    keep_f32.push_back(std::move(w));
    std::span<const float> W(keep_f32.back().data(), n);
    weight_bytes = double(n) * 4.0;
    time_it([&]() {
      sbnncuda::matvec_f32_dev(W, int(s.cols), dx, dy, int(s.rows));
    });
  }

  cudaMemcpy(y.data(), dy, s.rows * sizeof(float), cudaMemcpyDeviceToHost);
  const double err = compare(y, y_ref);
  cudaFree(dx);
  cudaFree(dy);

  const double gbs = weight_bytes / (us_per_call * 1e-6) / 1e9;
  std::printf("  %-22s [%6llu, %6llu] %5s  %10.2f us  %8.1f GB/s  err %10.2e\n",
              s.name, (unsigned long long)s.cols, (unsigned long long)s.rows,
              tname, us_per_call, gbs, err);
  return us_per_call;
}

} // namespace

int main() {
  if (!sbcuda::available()) {
    std::printf("gemv_bench SKIP (no CUDA device)\n");
    return 0;
  }

  std::mt19937 rng(0xC0FFEE);

  const int N = int(sizeof(kShapes) / sizeof(kShapes[0]));

  std::printf("== per-tensor quant types (read from the GGUF) ==\n");
  std::vector<double> us(N, 0.0);
  std::vector<double> q5_us(N, -1.0);
  for (int i = 0; i < N; ++i) us[i] = run(kShapes[i], kShapes[i].type, rng);

  // q5_K reference for the two mixed-precision tensors (their blocks 0-1 are
  // q5_K in Unsloth Q3_K_M, q4_K elsewhere).
  std::printf("== q5_K reference (attn_v / ffn_down blocks 0-1 variant) ==\n");
  for (int i = 0; i < N; ++i)
    if (kShapes[i].mixed) q5_us[i] = run(kShapes[i], sbquant::QuantType::Q5_K, rng);

  // Per-token K-quant gemv time = 42 decoder blocks × Σ(count_i × t_i), with the
  // two mixed tensors (attn_v, ffn_down) at q5_K in blocks 0-1 and q4_K in
  // blocks 2-41 (40×q4_K + 2×q5_K per tensor). This is the steady-state gemv
  // time one decode step spends in K-quant matmuls (before lm_head / attention
  // / norm overheads).
  double per_token_us = 0.0;
  bool all_timed = true;
  for (int i = 0; i < N; ++i) {
    if (us[i] <= 0.0) all_timed = false;
    double per_block_us = double(kShapes[i].count) * us[i];
    if (kShapes[i].mixed && q5_us[i] > 0.0)
      per_block_us = double(kShapes[i].count) * (40.0 * us[i] + 2.0 * q5_us[i]) / 42.0;
    per_token_us += per_block_us;
  }
  per_token_us *= 42.0;
  if (all_timed) {
    std::printf("\nK-quant gemv total per token: %.0f us (%.2f ms)\n", per_token_us,
                per_token_us / 1000.0);
  } else {
    std::printf("\n(some shapes failed to time; per-token total omitted)\n");
  }

  std::printf("\n== dense f32/bf16 gate gemvs (Phase 2.3b) ==\n");
  for (int i = 0; i < int(sizeof(kDenseShapes) / sizeof(kDenseShapes[0])); ++i)
    run_dense(kDenseShapes[i], rng);

  return 0;
}
