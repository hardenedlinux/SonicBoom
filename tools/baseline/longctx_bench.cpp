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
// CAVEAT — subtraction is thermal-fragile at large depth. At depth 2048 the two
// decodes are ~47 s each, and the ~1.3 s tail is ~2.7% of each total; a ~0.4%
// GPU-thermal drift between the second run (hotter) and the first inflates the
// tail by ~12% (measured 50.4 tok/s here vs 44.0 tok/s steady). Prefer the
// runtime's own single-run steady metric instead:
//   SONICBOOM_GEN_TIMING=1 ./tools/baseline/longctx_bench --depth 2048 --steps 1
// and read the `steady … tok/s` line (the `depth`-step decode averaged, first
// step excluded). That is what benchmark.md §1.2 uses.
//
// --prefill reaches the depth with the chunked prefill path instead of decoding
// to it, which is what makes the 4K/16K/32K tiers affordable: two prefix-session
// calls (prompt = depth tokens, one with 1 decode token and one with `steps`)
// are subtracted, so the chunked prefill and the per-call graph/plan rebuild
// cancel and only `steps-1` decode steps at depth remain. This matches
// llama-bench `-d <depth>`, which also prefills the depth.
//
// Usage: longctx_bench [--depth N] [--steps N] [--token id] [--n-ctx N]
//                      [--prefill]
// Build: tools/baseline/build_longctx_bench.sh

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
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

int main(int argc, char** argv) {
  uint64_t depth = 512;
  uint64_t steps = 64;
  uint64_t seed = 2;
  uint64_t n_ctx = 0;  // 0 = default (depth + steps)
  bool prefill_mode = false;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth")) depth = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--steps")) steps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--token")) seed = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--n-ctx")) n_ctx = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--prefill")) prefill_mode = true;
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

  // Pin n_ctx to the full decode length for BOTH runs (or an explicit --n-ctx).
  // Otherwise the warm run (n tokens) and the full run (n+steps tokens) size the
  // global-layer KV cache and the split-KV chunk (= n_ctx/40) differently, so
  // the first `depth` steps are not timing-identical and the subtraction
  // over/under-counts the tail. --n-ctx overrides to probe the chunk-size
  // dependence at a fixed depth (the n_ctx sensitivity sweep).
  const uint64_t ctx = n_ctx ? n_ctx : (depth + steps);

  if (prefill_mode) {
    // Chunked prefill to `depth`, then subtraction against a 1-decode-token
    // call. Fresh session per call: both calls perform the identical chunked
    // prefill of `depth` tokens, so their difference is exactly `steps-1`
    // steady decode steps at depth (the first sampled token comes from the
    // prefill itself).
    std::vector<uint64_t> prompt(depth);
    for (uint64_t i = 0; i < depth; ++i) prompt[i] = seed + (i % 500);
    // The warm call must be the exact timed call. A short prompt at the same
    // n_ctx does not absorb the one-time decode CUDA-graph capture: the capture
    // lands in the first call that reaches this depth, leaving it inside the
    // 1-token timing and making the subtraction negative (observed: -309 ms at
    // depth 512). Warm with `prompt` itself and discard the result.
    const std::vector<uint64_t>& warm_prompt = prompt;
    auto run = [&](const std::vector<uint64_t>& p, uint64_t ntok,
                   double* secs) -> uint64_t {
      std::vector<int64_t> toks(ntok, -1);
      sbplanner::KvCacheSession s;
      s.n_ctx = ctx;
      const auto t0 = std::chrono::steady_clock::now();
      const uint64_t done =
          sbmodel::generate_prefill_spine_cuda_prefix(*m, p, 0, toks, s,
                                                      true);
      const auto t1 = std::chrono::steady_clock::now();
      *secs = std::chrono::duration<double>(t1 - t0).count();
      return done;
    };
    double t_warm = 0.0, t_one = 0.0, t_all = 0.0;
    if (run(warm_prompt, 1, &t_warm) == 0 || run(prompt, 1, &t_one) == 0) {
      fprintf(stderr, "prefill-to-depth failed at depth %llu\n",
              (unsigned long long)depth);
      return 1;
    }
    const uint64_t done = run(prompt, steps, &t_all);
    const double tail = t_all - t_one;
    const uint64_t tail_toks = done > 1 ? done - 1 : 0;
    const double tok_per_sec = tail > 0.0 ? double(tail_toks) / tail : 0.0;
    printf("longctx depth=%llu: prefill-to-depth, %llu decode tokens in %.3f ms "
           "= %.3f tok/s (CUDA, seed %llu, prefill %.3f ms)\n",
           (unsigned long long)depth, (unsigned long long)tail_toks,
           tail * 1000.0, tok_per_sec, (unsigned long long)seed, t_one * 1000.0);
    printf("RESULT longctx depth=%llu steps=%llu tok_s=%.3f mode=prefill\n",
           (unsigned long long)depth, (unsigned long long)steps, tok_per_sec);
    return 0;
  }

  // Warm up at the EXACT n_ctx the timed runs use. The decode runtime (greedy
  // graph + plan + CUDA-graph capture) is cached per (model, n_ctx); the
  // subtraction below cancels the shared `depth` decode steps but NOT a one-time
  // plan/capture triggered inside the timed region. Warming at n_ctx = 1 (a
  // different cache key) used to leave that ~130 ms cost in the tail, which is
  // exactly the depth-0 gap 52.6 -> 47.5 tok/s measured against the out-of-tree
  // test_generate.
  {
    std::vector<int64_t> warm(1, -1);
    sbmodel::generate_spine_cuda(*m, seed, 0, warm, ctx);
  }

  auto run_decode = [&](uint64_t n, double* secs) -> uint64_t {
    std::vector<int64_t> toks(n, -1);
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t done = sbmodel::generate_spine_cuda(*m, seed, 0, toks, ctx);
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
  // Machine-readable canonical result. `tok_s` is the steady-state decode rate
  // at `depth` (llama-bench `-p 0 -n <steps> -d <depth>` semantics).
  printf("RESULT longctx depth=%llu steps=%llu tok_s=%.3f\n",
         (unsigned long long)depth, (unsigned long long)steps, tok_per_sec);
  return 0;
}
