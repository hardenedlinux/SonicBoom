// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// longctx_bench — decode tokens/sec at a fixed KV-cache depth, via the CUDA
// decode spine. The one-shot prefill path rejects prompts longer than the SWA
// window (its cache write requires a wrap-free run), so instead this measures
// depth by decoding incrementally and subtracting: it decodes `depth + steps`
// tokens from position 0 (building the cache), then decodes `depth` tokens from
// position 0 again, and reports `steps / (t_full - t_warm)` tok/s — the last
// `steps` steps, whose KV cache is ~`depth` keys deep. Deterministic greedy
// decode means the first `depth` steps are bit-identical between the two runs,
// so their time cancels exactly.
//
// Usage: longctx_bench [--depth N] [--steps N] [--token id]
// Build: tools/baseline/build_longctx_bench.sh

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
  uint64_t depth = 512;
  uint64_t steps = 64;
  uint64_t seed = 2;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth")) depth = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--steps")) steps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--token")) seed = atoll(argv[++i]);
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

  // Warm up the spine once (builds graph/plan caches, allocates device buffers).
  {
    std::vector<int64_t> warm(1, -1);
    sbmodel::generate_spine_cuda(*m, seed, 0, warm, 0);
  }

  auto run_decode = [&](uint64_t n, double* secs) -> uint64_t {
    std::vector<int64_t> toks(n, -1);
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t done = sbmodel::generate_spine_cuda(*m, seed, 0, toks, 0);
    const auto t1 = std::chrono::steady_clock::now();
    *secs = std::chrono::duration<double>(t1 - t0).count();
    return done;
  };

  double t_full = 0.0, t_warm = 0.0;
  const uint64_t done_full = run_decode(depth + steps, &t_full);
  const uint64_t done_warm = run_decode(depth, &t_warm);

  const double tail_secs = t_full - t_warm;      // first `depth` steps cancel
  const uint64_t tail_toks = done_full > done_warm ? done_full - done_warm : 0;
  const double tok_per_sec = tail_secs > 0.0 ? double(tail_toks) / tail_secs : 0.0;

  printf("longctx depth=%llu: %llu decode tokens in %.3f ms = %.3f tok/s "
         "(CUDA, seed %llu)\n",
         (unsigned long long)depth, (unsigned long long)tail_toks,
         tail_secs * 1000.0, tok_per_sec, (unsigned long long)seed);
  return 0;
}
