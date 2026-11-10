// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// longctx_bench — decode tokens/sec at a fixed KV-cache depth, via the CUDA
// decode spine.
//
// The measured segment is timed directly, inside one run: a session cache is
// brought to `depth` (by decoding to it, or with --prefill by prefilling to it),
// warmed for a few steps, and then a cudaEvent is recorded before the first
// measured step and after the last. `RESULT` is `steps / elapsed`, so the
// prefill/decode that builds the depth never enters the number and nothing is
// subtracted across runs.
//
// `RESULT_legacy` keeps the previous estimate, `steps / (t_full - t_warm)`, from
// two separate runs that cancel `depth` decode steps in between. The subtraction
// is thermal-fragile at large depth (a ~0.4% drift between the two runs inflated
// a 2048-deep tail by ~12%), which is why the direct number is the one to use;
// the legacy line remains for continuity with older measurements. `--no-legacy`
// skips the two extra runs.
//
// --prefill reaches the depth with the chunked prefill path instead of decoding
// to it, which is what makes the 4K/16K/32K tiers affordable. This matches
// llama-bench `-d <depth>`, which also prefills the depth.
//
// Usage: longctx_bench [--depth N] [--steps N] [--token id] [--n-ctx N]
//                      [--prefill] [--no-legacy]
// Build: tools/baseline/build_longctx_bench.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>

#include <cuda_runtime.h>

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
  bool legacy = true;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--depth")) depth = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--steps")) steps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--token")) seed = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--n-ctx")) n_ctx = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--prefill")) prefill_mode = true;
    else if (!strcmp(argv[i], "--no-legacy")) legacy = false;
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

  // Direct measurement: bring a session cache to `depth`, warm a few steps, then
  // time exactly `steps` decode steps with events on the stream boundary. The
  // depth-building work is outside the event pair, so it drops out without
  // subtraction.
  {
    sbplanner::KvCacheSession session;
    session.n_ctx = ctx;
    if (!sbmodel::kv_session_allocate(*m, session, ctx)) {
      fprintf(stderr, "session allocate failed at n_ctx %llu\n",
              (unsigned long long)ctx);
      return 1;
    }
    // Depth 0 is an empty context: the first measured step consumes the seed
    // token at position 0, so there is nothing to build and nothing to take
    // from a build vector.
    uint64_t pos = 0;
    uint64_t token = seed;
    if (prefill_mode && depth > 0) {
      std::vector<uint64_t> prompt(depth);
      for (uint64_t i = 0; i < depth; ++i) prompt[i] = seed + (i % 500);
      std::vector<int64_t> first(1, -1);
      if (sbmodel::generate_prefill_spine_cuda_prefix(*m, prompt, 0, first,
                                                      session, true) == 0) {
        fprintf(stderr, "prefill-to-depth failed at depth %llu\n",
                (unsigned long long)depth);
        return 1;
      }
      pos = depth + 1;
      token = uint64_t(first[0]);
    } else if (depth > 0) {
      std::vector<int64_t> build(depth, -1);
      const uint64_t built =
          sbmodel::generate_spine_cuda(*m, seed, 0, build, ctx, {}, &session);
      if (built != depth) {
        fprintf(stderr, "decode-to-depth failed at depth %llu (%llu steps)\n",
                (unsigned long long)depth, (unsigned long long)built);
        return 1;
      }
      pos = depth;
      token = uint64_t(build.back());
    }
    constexpr uint64_t kWarmSteps = 8;
    {
      std::vector<int64_t> warm(kWarmSteps, -1);
      const uint64_t done = sbmodel::generate_spine_cuda(*m, token, pos, warm,
                                                         ctx, {}, &session);
      if (done != kWarmSteps) {
        fprintf(stderr, "warm-up decode failed (%llu steps)\n",
                (unsigned long long)done);
        return 1;
      }
      pos += done;
      token = uint64_t(warm.back());
    }
    cudaEvent_t begin = nullptr, end = nullptr;
    if (cudaEventCreate(&begin) != cudaSuccess ||
        cudaEventCreate(&end) != cudaSuccess) {
      fprintf(stderr, "cudaEventCreate failed\n");
      return 1;
    }
    std::vector<int64_t> out(steps, -1);
    cudaEventRecord(begin);
    const uint64_t done = sbmodel::generate_spine_cuda(*m, token, pos, out, ctx,
                                                       {}, &session);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (done == 0) {
      fprintf(stderr, "measured decode failed\n");
      return 1;
    }
    const double secs = double(elapsed_ms) / 1000.0;
    const double rate = secs > 0.0 ? double(done) / secs : 0.0;
    printf("longctx depth=%llu: %llu decode tokens in %.3f ms = %.3f tok/s "
           "(CUDA, seed %llu, one-run%s)\n",
           (unsigned long long)depth, (unsigned long long)done,
           double(elapsed_ms), rate, (unsigned long long)seed,
           prefill_mode ? ", prefill-to-depth" : "");
    printf("RESULT longctx depth=%llu steps=%llu tok_s=%.3f%s\n",
           (unsigned long long)depth, (unsigned long long)done, rate,
           prefill_mode ? " mode=prefill" : "");
  }

  if (!legacy) return 0;

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
    printf("RESULT_legacy longctx depth=%llu steps=%llu tok_s=%.3f "
           "mode=prefill\n",
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
  // The previous estimate, kept for continuity: the tail of a full decode minus
  // a shorter one.
  printf("RESULT_legacy longctx depth=%llu steps=%llu tok_s=%.3f\n",
         (unsigned long long)depth, (unsigned long long)steps, tok_per_sec);
  return 0;
}
