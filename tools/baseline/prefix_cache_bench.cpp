// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// prefix_cache_bench — SonicBoom CUDA KV prefix-cache reuse (C.4). Measures the
// first-token latency of a fixed `--prefix`-token prompt followed by `--new`
// freshly-added tokens, comparing the reuse path (shared prefix's KV is reused,
// only the new suffix is prefilled) against a full re-prefill of the whole
// prompt. It also checks correctness: the greedy continuation after reuse must
// match a full prefill's continuation (or report the first divergent step).
//
// Usage: prefix_cache_bench [--prefix N] [--new N] [--steps S] [--reps R]
// Build: tools/baseline/build_prefix_cache_bench.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

// Load a space-separated token-id file (real-text prompt produced by
// tools/baseline/make_real_prompt.sh). Fails loudly on empty/parse error so a
// synthetic fallback can never silently creep back in.
static std::vector<uint64_t> load_tokens(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    fprintf(stderr, "tokens file not found: %s\n", path.c_str());
    exit(2);
  }
  std::vector<uint64_t> t;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    uint64_t tok;
    while (ss >> tok) t.push_back(tok);
  }
  if (t.empty()) {
    fprintf(stderr, "tokens file empty: %s\n", path.c_str());
    exit(2);
  }
  return t;
}

static std::string toks(const std::vector<int64_t>& v) {
  std::string s = "[";
  const size_t kMax = 16;
  for (size_t i = 0; i < v.size() && i < kMax; ++i) {
    if (i) s += ",";
    s += std::to_string(v[i]);
  }
  if (v.size() > kMax) s += ",...";
  s += "]";
  return s;
}

int main(int argc, char** argv) {
  uint64_t prefix_n = 4000;
  uint64_t new_n = 100;
  uint64_t steps = 16;
  uint64_t reps = 3;
  std::string tokens_file = "real_prompt.tokens";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--prefix")) prefix_n = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--new")) new_n = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--steps")) steps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--reps")) reps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--tokens-file")) tokens_file = argv[++i];
  }
  if (prefix_n == 0 || new_n == 0 || steps == 0 || reps == 0) {
    fprintf(stderr,
            "usage: prefix_cache_bench [--prefix N] [--new N] [--steps S] [--reps R] [--tokens-file PATH]\n");
    return 2;
  }
  const std::vector<uint64_t> all_tokens = load_tokens(tokens_file);
  if (prefix_n + new_n > all_tokens.size()) {
    fprintf(stderr,
            "tokens file has %zu tokens, need at least %llu (prefix+new)\n",
            all_tokens.size(), (unsigned long long)(prefix_n + new_n));
    return 2;
  }

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

  const std::vector<uint64_t> prefix(all_tokens.begin(),
                                     all_tokens.begin() + prefix_n);
  const std::vector<uint64_t> full(all_tokens.begin(),
                                   all_tokens.begin() + prefix_n + new_n);

  // --- correctness: reuse continuation == full-prefill continuation ----------
  {
    std::vector<int64_t> full_toks(steps, -1);
    const uint64_t full_done = sbmodel::generate_prefill_spine_cuda(
        *m, full, 0, full_toks, full.size() + steps);
    full_toks.resize(full_done);

    sbplanner::KvCacheSession session;
    session.n_ctx = full.size() + steps;

    // Seed the cache with the prefix only (one decode token = no decode steps),
    // so the reuse call below is the sole source of the continuation.
    std::vector<int64_t> setup_toks(1, -1);
    if (sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, setup_toks,
                                                    session) == 0) {
      fprintf(stderr, "prefix setup prefill failed\n");
      return 1;
    }

    std::vector<int64_t> reuse_toks(steps, -1);
    const uint64_t reuse_done = sbmodel::generate_prefill_spine_cuda_prefix(
        *m, full, 0, reuse_toks, session);
    reuse_toks.resize(reuse_done);

    if (full_toks == reuse_toks) {
      printf("correctness: reuse continuation identical to full prefill (%zu tokens)\n",
             full_toks.size());
    } else {
      size_t div = 0;
      while (div < full_toks.size() && div < reuse_toks.size() &&
             full_toks[div] == reuse_toks[div])
        ++div;
      printf("correctness: DIVERGE at step %zu (full %s vs reuse %s)\n", div,
             toks(full_toks).c_str(), toks(reuse_toks).c_str());
    }
    printf("  full : %s\n  reuse: %s\n", toks(full_toks).c_str(),
           toks(reuse_toks).c_str());
  }

  // --- first-token latency: reuse vs no-reuse -------------------------------
  auto best = [&](std::function<uint64_t()> run, bool* ok) -> double {
    double b = 1e18;
    for (uint64_t rep = 0; rep < reps; ++rep) {
      const auto t0 = std::chrono::steady_clock::now();
      const uint64_t done = run();
      const auto t1 = std::chrono::steady_clock::now();
      if (done == 0) { *ok = false; return 0.0; }
      const double secs = std::chrono::duration<double>(t1 - t0).count();
      if (secs < b) b = secs;
    }
    *ok = true;
    return b;
  };

  // Warm up (graph/plan caches, device allocation).
  {
    std::vector<int64_t> warm(1, -1);
    sbmodel::generate_prefill_spine_cuda(*m, full, 0, warm, full.size() + 1);
    sbplanner::KvCacheSession s;
    s.n_ctx = full.size() + 1;
    sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, warm, s);
  }

  // No-reuse: full prefill of the whole prompt, one decode token.
  bool ok = true;
  const double t_noreuse = best(
      [&]() {
        std::vector<int64_t> toks(1, -1);
        return sbmodel::generate_prefill_spine_cuda(*m, full, 0, toks,
                                                    full.size() + 1);
      },
      &ok);
  if (!ok) { fprintf(stderr, "no-reuse prefill failed\n"); return 1; }

  // Reuse: seed the prefix once, then per rep prefill only the new suffix
  // (vary the suffix so each rep matches only the shared prefix).
  sbplanner::KvCacheSession session;
  session.n_ctx = full.size() + 1;
  {
    std::vector<int64_t> setup_toks(1, -1);
    if (sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, setup_toks,
                                                    session) == 0) {
      fprintf(stderr, "reuse prefix setup failed\n");
      return 1;
    }
  }
  const double t_reuse = best(
      [&]() {
        // Reset the token history so every rep reuses only the shared prefix
        // (the device cache's suffix slots are simply overwritten by the same
        // `full` suffix each rep).
        session.filled_tokens = prefix;
        std::vector<int64_t> toks(1, -1);
        return sbmodel::generate_prefill_spine_cuda_prefix(*m, full, 0, toks,
                                                           session);
      },
      &ok);
  if (!ok) { fprintf(stderr, "reuse prefill failed\n"); return 1; }

  printf("first-token latency (prefix %llu + new %llu):\n",
         (unsigned long long)prefix_n, (unsigned long long)new_n);
  printf("  no-reuse: %.3f ms (full %llu-token prefill)\n",
         t_noreuse * 1000.0, (unsigned long long)(prefix_n + new_n));
  printf("  reuse   : %.3f ms (%llu new tokens prefilled, prefix reused)\n",
         t_reuse * 1000.0, (unsigned long long)new_n);
  printf("  speedup : %.1fx\n", t_reuse > 0.0 ? t_noreuse / t_reuse : 0.0);
  // Machine-readable canonical result: first-token latency of a `new_n`-token
  // continuation after a `prefix_n`-token shared prefix (reuse) vs the same
  // continuation after a full re-prefill (no reuse). llama-bench has no
  // prefix-reuse equivalent.
  printf("RESULT prefix_cache prefix=%llu new=%llu reuse_ms=%.3f "
         "noreuse_ms=%.3f speedup=%.2f\n",
         (unsigned long long)prefix_n, (unsigned long long)new_n,
         t_reuse * 1000.0, t_noreuse * 1000.0,
         t_reuse > 0.0 ? t_noreuse / t_reuse : 0.0);
  return 0;
}
