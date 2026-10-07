// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Standalone decode-attention microbenchmark (dev-time only; not part of the
// CMake build). Measures the internal nn::cuda::decode_attention_dev kernel in
// isolation — NOT end-to-end tok/s — across context lengths, for both Gemma 4
// attention shapes:
//   - SWA   (local) layers: head_dim=256, sliding_window=512 (ring cache)
//   - global        layers: head_dim=512, sliding_window=0   (full decode)
// n_heads_q=8, n_heads_kv=2 throughout (Gemma 4 E4B-it).
//
// For each (shape, pos) it (1) times the kernel with CUDA events using an
// adaptive iteration count, and (2) diffs the CUDA output against the CPU
// reference nn::decode_attention (the oracle) within an fp32 tolerance — the
// CPU accumulates in the same scalar order but uses std::exp vs the kernel's
// expf, and the warp rewrite changes the dot/sum reduction order, so agreement
// is to ~1e-3 relative, not bit-exact.
//
// It links only libsonicboom.so (the SHARED core) + libcudart, and includes the
// internal core/src/nn/cuda_resident.h via relative path (decode_attention_dev
// is not on the public Layer 2 surface).

#include <cuda_runtime.h>

#include <sonicboom/nn/attention.h>
#include <sonicboom/nn/cuda_elementwise.h>  // available()

#include "../../core/src/nn/cuda_resident.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace sbnn = sonicboom::nn;
namespace sbcuda = sonicboom::nn::cuda;

namespace {

// LCG fill in [-1, 1].
void fill(std::vector<float>& v, uint32_t& s) {
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = float(int32_t(s >> 8) % 2000) / 1000.0f - 1.0f;
  }
}

// L2-normalize each row so q·k dots are cosine similarities in [-1, 1],
// matching the RMS-normed q/k the real decode path feeds in.
void normalize_rows(std::vector<float>& v, uint64_t rows, uint64_t head_dim) {
  for (uint64_t r = 0; r < rows; ++r) {
    float* row = v.data() + r * head_dim;
    double s = 0.0;
    for (uint64_t i = 0; i < head_dim; ++i) s += double(row[i]) * row[i];
    const float inv = 1.0f / std::sqrt(float(s));
    for (uint64_t i = 0; i < head_dim; ++i) row[i] *= inv;
  }
}

double compare(const std::vector<float>& got, const std::vector<float>& ref) {
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double denom = std::max(1.0, double(std::fabs(ref[i])));
    worst = std::max(worst, std::fabs(double(got[i]) - double(ref[i])) / denom);
  }
  return worst;
}

struct Config {
  const char* name;
  int head_dim;
  uint64_t sliding_window;
};

void run(const Config& cfg) {
  const int n_heads_q = 8, n_heads_kv = 2;
  const uint64_t pos_list[] = {32, 64, 128, 256, 512, 1024};

  std::printf("== %s (head_dim=%d, sliding_window=%llu) ==\n", cfg.name, cfg.head_dim,
              (unsigned long long)cfg.sliding_window);
  std::printf("  %6s | %8s | %10s | %13s\n", "pos", "n_keys", "us/call", "worst_rel_err");

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  for (uint64_t pos : pos_list) {
    const uint64_t start =
        (cfg.sliding_window > 0 && pos >= cfg.sliding_window) ? pos - cfg.sliding_window + 1 : 0;
    const uint64_t n_keys = pos - start + 1;
    const uint64_t n_slots = (cfg.sliding_window > 0) ? cfg.sliding_window : (pos + 1);

    const uint64_t q_n = uint64_t(n_heads_q) * cfg.head_dim;
    const uint64_t kv_rows = n_slots * n_heads_kv;
    const uint64_t kv_n = kv_rows * cfg.head_dim;

    std::vector<float> q(q_n), k(kv_n), v(kv_n), out(q_n), ref(q_n);
    uint32_t s = 0x12345678u;
    fill(q, s);
    fill(k, s);
    fill(v, s);
    normalize_rows(q, n_heads_q, cfg.head_dim);
    normalize_rows(k, kv_rows, cfg.head_dim);
    normalize_rows(v, kv_rows, cfg.head_dim);

    // CPU oracle.
    sbnn::decode_attention(q, k, v, pos, n_slots, n_heads_q, n_heads_kv,
                           cfg.head_dim, cfg.sliding_window, 1.0f, ref);

    float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dout = nullptr;
    cudaMalloc(&dq, q_n * sizeof(float));
    cudaMalloc(&dk, kv_n * sizeof(float));
    cudaMalloc(&dv, kv_n * sizeof(float));
    cudaMalloc(&dout, q_n * sizeof(float));
    cudaMemcpy(dq, q.data(), q_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dk, k.data(), kv_n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dv, v.data(), kv_n * sizeof(float), cudaMemcpyHostToDevice);

    auto call = [&]() {
      sbcuda::decode_attention_dev(dq, dk, dv, pos, n_slots, n_heads_q, n_heads_kv,
                                   cfg.head_dim, cfg.sliding_window, 1.0f, dout);
    };

    // Warm up, then time one call to size the batch.
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

    cudaMemcpy(out.data(), dout, q_n * sizeof(float), cudaMemcpyDeviceToHost);
    const double err = compare(out, ref);

    std::printf("  %6llu | %8llu | %10.2f | %13.2e\n", (unsigned long long)pos,
                (unsigned long long)n_keys, us_per_call, err);

    cudaFree(dq);
    cudaFree(dk);
    cudaFree(dv);
    cudaFree(dout);
  }

  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
}

} // namespace

int main() {
  if (!sbcuda::available()) {
    std::printf("attention_bench SKIP (no CUDA device)\n");
    return 0;
  }
  run({"SWA", 256, 512});
  run({"global", 512, 0});
  return 0;
}
