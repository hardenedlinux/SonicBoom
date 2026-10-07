// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Standalone K-quant gemv microbenchmark (dev-time only; not part of the CMake
// build). Measures the internal quant::cuda::matvec_f32_dev kernel in isolation
// — NOT end-to-end tok/s — across the Gemma 4 E4B-it matmul shapes, so we can
// see how much of the decode-time cost is the dequantize-on-the-fly gemv and how
// it scales with (cols, rows, blocks_per_row). The model is Q3_K_M, so Q3_K is
// the headline; Q4_K/Q5_K are swept at the FFN shapes for reference only.
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

#include <sonicboom/quant/quantized_matmul.h>       // matvec_f32 (CPU ref)
#include <sonicboom/quant/quantized_matmul_cuda.h>  // available()
#include <sonicboom/quant/quantized_tensor.h>

#include "../../core/src/quant/cuda_resident.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace sbquant = sonicboom::quant;
namespace sbcuda = sonicboom::quant::cuda;

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
  uint64_t cols;   // contiguous / inner dim (W.dims()[0])
  uint64_t rows;   // output dim (W.dims()[1])
  int count;       // occurrences per decoder block
};

// Gemma 4 E4B-it matmul shapes (d=2560, ff=10240, n_heads_q=8, n_heads_kv=2).
const Shape kShapes[] = {
    {"ffn_gate_up", 2560, 10240, 2},  // gate + up  (ff=10240)
    {"ffn_down", 10240, 2560, 1},     // down       (2560→10240→2560)
    {"attn_q", 2560, 2048, 1},        // q proj     (8 heads × 256)
    {"attn_o", 2048, 2560, 1},        // o proj
    {"attn_kv", 2560, 512, 2},        // k + v proj (2 kv heads × 256)
};

void run(const Shape& s, sbquant::QuantType type, std::mt19937& rng,
         std::vector<double>& us_out) {
  const char* tname =
      type == sbquant::QuantType::Q3_K ? "Q3_K" :
      type == sbquant::QuantType::Q4_K ? "Q4_K" : "Q5_K";

  auto bytes = build_weight(type, s.rows, s.cols, rng);
  sbquant::QuantizedTensor W(type, {s.cols, s.rows}, bytes);
  if (!W.valid()) {
    std::printf("  %-14s [%6llu, %6llu] %s  INVALID\n", s.name,
                (unsigned long long)s.cols, (unsigned long long)s.rows, tname);
    us_out.push_back(0.0);
    return;
  }

  std::vector<float> x(s.cols), y(s.rows), y_ref(s.rows);
  std::uniform_real_distribution<float> ud(-1.0f, 1.0f);
  for (auto& v : x) v = ud(rng);

  if (!sbquant::matvec_f32(W, x, y_ref)) {
    std::printf("  %-14s [%6llu, %6llu] %s  CPU ref FAILED\n", s.name,
                (unsigned long long)s.cols, (unsigned long long)s.rows, tname);
    us_out.push_back(0.0);
    return;
  }

  float *dx = nullptr, *dy = nullptr;
  cudaMalloc(&dx, s.cols * sizeof(float));
  cudaMalloc(&dy, s.rows * sizeof(float));
  cudaMemcpy(dx, x.data(), s.cols * sizeof(float), cudaMemcpyHostToDevice);

  auto call = [&]() { sbcuda::matvec_f32_dev(W, dx, dy); };

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  // Warm up (also triggers the one-time device weight upload, excluded below).
  for (int i = 0; i < 5; ++i) call();
  cudaDeviceSynchronize();

  cudaEventRecord(e0);
  call();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float one_ms = 0.0f;
  cudaEventElapsedTime(&one_ms, e0, e1);
  const double one_us = double(one_ms) * 1000.0;
  const int N = int(std::clamp(20000.0 / std::max(one_us, 0.5), 20.0, 5000.0));

  cudaEventRecord(e0);
  for (int i = 0; i < N; ++i) call();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float total_ms = 0.0f;
  cudaEventElapsedTime(&total_ms, e0, e1);
  const double us_per_call = double(total_ms) * 1000.0 / double(N);

  cudaMemcpy(y.data(), dy, s.rows * sizeof(float), cudaMemcpyDeviceToHost);
  const double err = compare(y, y_ref);

  std::printf("  %-14s [%6llu, %6llu] %5s %7llu blk  %10.2f us  err %10.2e\n",
              s.name, (unsigned long long)s.cols, (unsigned long long)s.rows,
              tname, (unsigned long long)W.num_blocks(), us_per_call, err);

  us_out.push_back(us_per_call);

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  cudaFree(dx);
  cudaFree(dy);
}

} // namespace

int main() {
  if (!sbcuda::available()) {
    std::printf("gemv_bench SKIP (no CUDA device)\n");
    return 0;
  }

  std::mt19937 rng(0xC0FFEE);

  std::printf("== Q3_K (model format) ==\n");
  std::vector<double> q3;
  q3.reserve(5);
  for (int i = 0; i < 5; ++i) run(kShapes[i], sbquant::QuantType::Q3_K, rng, q3);

  std::printf("== Q4_K / Q5_K (reference, FFN shapes only) ==\n");
  std::vector<double> q4, q5;
  q4.reserve(2);
  q5.reserve(2);
  for (int i = 0; i < 2; ++i) {  // ffn_gate_up, ffn_down
    run(kShapes[i], sbquant::QuantType::Q4_K, rng, q4);
    run(kShapes[i], sbquant::QuantType::Q5_K, rng, q5);
  }

  // Per-token cost: 42 decoder blocks × Σ(count_i × t_i). If every shape was
  // timed, this is the steady-state gemv time a single decode step spends in
  // K-quant matmuls (before the lm_head / attention kernel / norm overheads).
  double per_token_us = 0.0;
  bool all_timed = true;
  for (int i = 0; i < 5; ++i) {
    if (q3[i] <= 0.0) all_timed = false;
    per_token_us += double(kShapes[i].count) * q3[i];
  }
  per_token_us *= 42.0;
  if (all_timed) {
    std::printf("\nQ3_K gemv total per token: %.0f us (%.2f ms)\n", per_token_us,
                per_token_us / 1000.0);
  } else {
    std::printf("\n(some shapes failed to time; per-token total omitted)\n");
  }

  return 0;
}
