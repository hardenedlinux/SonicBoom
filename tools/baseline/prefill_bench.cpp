// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// prefill_bench — SonicBoom CUDA prefill throughput (pp tokens/sec) at a fixed
// prompt length, for comparison against llama.cpp's `llama-bench -p <n> -n 0
// -ngl 99`. Calls generate_prefill_spine_cuda with a prompt of N tokens and a
// single decode token, so the timed region is prefill + lm-head sample (no
// autoregressive decode).
//
// Uses the persistent prefix-session entry point, which chunks prompts larger
// than the 512-token SWA window and supports pp512/pp2048/pp4100.
//
// Plan overhead isolation: generate_prefill_spine_cuda_prefix rebuilds the
// decode graph + plan and re-allocates the CUDA backend's KV caches on every
// call, all of which is ~constant in N. It is subtracted by also timing a
// 1-token prompt
// (same per-call overhead) and reporting the marginal pp:
//   pp_marginal = (N-1) / (time(N) - time(1)).
//
// Usage: prefill_bench [--prompt N] [--reps R]
// Build: tools/baseline/build_prefill_bench.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>

#include <nvtx3/nvToolsExt.h>  // header-only NVTX (no libnvtx link)

#include <algorithm>
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
namespace sbplanner = sonicboom::planner;

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
    sbplanner::KvCacheSession session;
    session.n_ctx = prompt.size() + toks.size();
    return sbmodel::generate_prefill_spine_cuda_prefix(
        *m, prompt, 0, toks, session, true);
  };

  // NVTX call markers (opt-in via SONICBOOM_NVTX, the runtime's own switch):
  // nsys can then isolate exactly one measured prefill call. The range wraps
  // the timed region, so the host duration of the range is the call latency.
  const bool nvtx_on = std::getenv("SONICBOOM_NVTX") != nullptr;
  struct Stats {
    std::vector<double> reps;  // seconds, in execution order
    double best = 1e18;
    double median = 0.0;
  };
  auto best = [&](const std::vector<uint64_t>& prompt, const char* label,
                  bool* ok) -> Stats {
    Stats s;
    s.reps.reserve(reps);
    for (uint64_t rep = 0; rep < reps; ++rep) {
      if (nvtx_on) nvtxRangePushA(label);
      const auto t0 = std::chrono::steady_clock::now();
      const uint64_t done = run(prompt);
      const auto t1 = std::chrono::steady_clock::now();
      if (nvtx_on) nvtxRangePop();
      if (done == 0) { *ok = false; return s; }
      const double secs = std::chrono::duration<double>(t1 - t0).count();
      s.reps.push_back(secs);
      if (secs < s.best) s.best = secs;
    }
    std::vector<double> sorted = s.reps;
    std::sort(sorted.begin(), sorted.end());
    const size_t m = sorted.size();
    s.median = (m % 2) ? sorted[m / 2]
                       : 0.5 * (sorted[m / 2 - 1] + sorted[m / 2]);
    *ok = true;
    return s;
  };
  auto print_reps = [](const char* what, const Stats& s) {
    fprintf(stderr, "[reps] %s ms=", what);
    for (size_t i = 0; i < s.reps.size(); ++i)
      fprintf(stderr, "%s%.3f", i ? "," : "", s.reps[i] * 1000.0);
    fprintf(stderr, " best=%.3f median=%.3f\n", s.best * 1000.0,
            s.median * 1000.0);
  };

  const std::vector<uint64_t> prompt = make_prompt(n);
  const std::vector<uint64_t> one = make_prompt(1);

  // Warm up (builds graph/plan caches, allocates buffers, warms the device).
  bool ok = true;
  best(prompt, "bench:warmup_N", &ok);
  if (!ok) {
    fprintf(stderr, "prefill rejected at --prompt %llu\n",
            (unsigned long long)n);
    return 1;
  }
  best(one, "bench:warmup_1", &ok);

  const Stats sn = best(prompt, "bench:measure_N", &ok);
  if (!ok) return 1;
  const Stats s1 = best(one, "bench:measure_1", &ok);
  const double t_n = sn.best;
  const double t_1 = s1.best;
  print_reps("N", sn);
  print_reps("1", s1);

  const double pp_wall = t_n > 0.0 ? double(n) / t_n : 0.0;
  const double dt = t_n - t_1;
  const double pp_marginal = dt > 0.0 ? double(n - 1) / dt : 0.0;
  // Median-of-reps view (the same marginal definition on median latencies).
  const double dt_med = sn.median - s1.median;
  const double pp_marginal_med = dt_med > 0.0 ? double(n - 1) / dt_med : 0.0;

  printf("prefill --prompt %llu: %.3f ms (N=%llu) vs %.3f ms (N=1) | "
         "pp_marginal %.1f tok/s (CANONICAL) | pp_wall %.1f tok/s "
         "(legacy: includes the per-call graph/plan rebuild + KV alloc) "
         "(best of %llu, CUDA)\n",
         (unsigned long long)n, t_n * 1000.0, (unsigned long long)n,
         t_1 * 1000.0, pp_marginal, pp_wall, (unsigned long long)reps);
  // Canonical = pp_marginal: the marginal prefill cost per token, matching
  // llama-bench `-p <n> -n 0` (which amortizes context/KV setup outside the
  // timed region). pp_wall is reported for reference only.
  printf("RESULT prefill prompt=%llu pp_marginal_tok_s=%.1f pp_wall_tok_s=%.1f\n",
         (unsigned long long)n, pp_marginal, pp_wall);
  // Median-of-reps view, alongside the canonical best-of line above.
  printf("RESULT_MEDIAN prefill prompt=%llu t_n_ms=%.3f t_1_ms=%.3f "
         "pp_marginal_tok_s=%.1f pp_wall_tok_s=%.1f\n",
         (unsigned long long)n, sn.median * 1000.0, s1.median * 1000.0,
         pp_marginal_med,
         sn.median > 0.0 ? double(n) / sn.median : 0.0);
  return 0;
}
