// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// prefill_bench — SonicBoom CUDA prefill throughput (pp tokens/sec) at a fixed
// prompt length, for comparison against llama.cpp's `llama-bench -p <n> -n 0
// -ngl 99`. Calls generate_prefill_spine_cuda with a prompt of N tokens and a
// single decode token, so the timed region is prefill + lm-head sample (no
// autoregressive decode).
//
// The prefill path requires the prompt to land in a straight, wrap-free run of
// each KV cache (SWA ring n_slots == sliding_window == 512, global n_slots ==
// n_ctx), so N <= 512. N > 512 returns 0 (prefill rejected) — pp2048 is not
// supported by the current one-shot prefill; llama.cpp handles it via chunked
// (ubatch) prefill.
//
// Plan overhead isolation: generate_prefill_spine_cuda rebuilds the decode graph
// + plan and re-allocates the CUDA backend's KV caches on every call, all of
// which is ~constant in N. It is subtracted by also timing a 1-token prompt
// (same per-call overhead) and reporting the marginal pp:
//   pp_marginal = (N-1) / (time(N) - time(1)).
//
// Usage: prefill_bench [--prompt N] [--reps R]
// Build: tools/baseline/build_prefill_bench.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;

int main(int argc, char** argv) {
  uint64_t n = 512;
  uint64_t reps = 3;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--prompt")) n = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--reps")) reps = atoll(argv[++i]);
  }
  if (n == 0 || reps == 0) { fprintf(stderr, "usage: prefill_bench [--prompt N] [--reps R]\n"); return 2; }

  std::string path;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF")) path = env;
  if (path.empty()) {
    const char* root = std::getenv("SONICBOOM_ROOT");
    std::string base = root ? std::string(root) : std::string(".");
    path = base + "/models/gemma-4-E4B-it-Q3_K_M.gguf";
  }
  if (!std::filesystem::exists(path)) {
    fprintf(stderr, "model not found: %s\n", path.c_str());
    return 2;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) { fprintf(stderr, "failed to load GGUF\n"); return 1; }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) { fprintf(stderr, "load_gemma4 error\n"); return 1; }

  // Deterministic prompt (token values don't affect matmul/flash-attention cost).
  auto make_prompt = [](uint64_t len) {
    std::vector<uint64_t> p(len);
    for (uint64_t i = 0; i < len; ++i) p[i] = 2 + (i % 500);
    return p;
  };

  auto run = [&](const std::vector<uint64_t>& prompt) -> uint64_t {
    std::vector<int64_t> toks(1, -1);
    // n_ctx = prompt + 1 decode token; global layers need start_pos+n <= n_ctx.
    return sbmodel::generate_prefill_spine_cuda(*m, prompt, 0, toks,
                                                prompt.size() + 1);
  };

  auto best = [&](const std::vector<uint64_t>& prompt, bool* ok) -> double {
    double b = 1e18;
    for (uint64_t rep = 0; rep < reps; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      const uint64_t done = run(prompt);
      const auto t1 = std::chrono::steady_clock::now();
      if (done == 0) { *ok = false; return 0.0; }
      const double secs = std::chrono::duration<double>(t1 - t0).count();
      if (secs < b) b = secs;
    }
    *ok = true;
    return b;
  };

  const std::vector<uint64_t> prompt = make_prompt(n);
  const std::vector<uint64_t> one = make_prompt(1);

  // Warm up (builds graph/plan caches, allocates buffers, warms the device).
  bool ok = true;
  best(prompt, &ok);
  if (!ok) {
    fprintf(stderr, "prefill rejected at --prompt %llu (SWA wrap-free constraint: N>512)\n",
            (unsigned long long)n);
    return 1;
  }
  best(one, &ok);

  const double t_n = best(prompt, &ok);
  if (!ok) return 1;
  const double t_1 = best(one, &ok);

  const double pp_wall = t_n > 0.0 ? double(n) / t_n : 0.0;
  const double dt = t_n - t_1;
  const double pp_marginal = dt > 0.0 ? double(n - 1) / dt : 0.0;

  printf("prefill --prompt %llu: %.3f ms (N=%llu) vs %.3f ms (N=1) | "
         "pp_wall %.1f tok/s | pp_marginal %.1f tok/s (best of %llu, CUDA)\n",
         (unsigned long long)n, t_n * 1000.0, (unsigned long long)n,
         t_1 * 1000.0, pp_wall, pp_marginal, (unsigned long long)reps);
  return 0;
}
