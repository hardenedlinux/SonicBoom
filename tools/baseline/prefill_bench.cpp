// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// prefill_bench — SonicBoom CUDA prefill throughput (pp tokens/sec) at a fixed
// prompt length, for comparison against llama.cpp's `llama-bench -p <n> -n 0
// -ngl 99`. Calls generate_prefill_spine_cuda_prefix with a prompt of N tokens
// and a single decode token, so the timed region is prefill + lm-head sample
// (no autoregressive decode). Chunks prompts larger than the model's prefill
// chunk (Gemma's 512-token SWA window) and supports pp512/pp2048/pp4100.
//
// ---------------------------------------------------------------------------
// Measurement methodology
// ---------------------------------------------------------------------------
//
// A session owns its KV cache; every execution runtime is bound to that cache
// and keyed by its identity. So *how many sessions a run creates* and *where
// the session's one-time build is charged* decide what the number means. The
// tool measures two lifecycle scenarios and never mixes them:
//
//   reused (canonical throughput)
//     One session is created before measurement. Its KV cache, the decode
//     runtime and the prefill segment plans/captures are built once (the
//     "first prefill" below), then every repetition runs the same full prefill
//     on that session. Between repetitions only the logical prefill state is
//     reset (filled_tokens is cleared, so the next call prefills the prompt
//     from position 0 instead of reusing it as a prefix); the KV allocation,
//     the runtime and the captured segment are kept. Each repetition therefore
//     measures the same work: N prompt tokens through the steady-state path.
//
//       warm_tok_s        = N / t_warm          steady-state prefill throughput
//       pp_marginal_tok_s = (N - n0) / (t_N - t_n0)
//
//     `pp_marginal_tok_s` is the per-token prefill cost with the per-call
//     constant expense removed, and it is measured on the same session as the
//     warm number. The cancellation baseline is a *same-execution-path*
//     prefill of n0 tokens, not a one-token call:
//       - n <= chunk: n0 = max(2, n/2). Both are one uncaptured batched
//         segment, so the delta is pure per-token work.
//       - n > chunk:  n0 = max(2, n - chunk). Both are steady-state prefill
//         runs (the capture is built during the first prefill, outside the
//         measured reps); the delta is exactly one full chunk.
//     A one-token call is never used: it takes the single-column kernels and
//     never reaches the prefill segment/capture path, so subtracting it
//     cancels costs that were never added (the misleading B0.2 regression).
//
//   fresh (session lifecycle cost, not throughput)
//     Every repetition creates a new session, exactly as
//     sb_llm_session_new does: the KV cache is allocated for this session and
//     the session-bound runtimes are warmed (the C API pays both in session
//     creation). The first prefill is then timed separately; its runtime
//     registries and captured prefill segment are released afterwards
//     (release_session_runtimes), so repetitions do not accumulate sessions or
//     device memory. Reported per repetition and as a median:
//
//       session_create_ms   kv_session_allocate + session-bound runtime warm-up
//       first_prefill_ms    the session's first prefill: prefill segment build,
//                           CUDA-graph capture (when N > chunk) and execution
//       e2e_ms              session_create_ms + first_prefill_ms (what a server
//                           that creates one session per request actually pays)
//
// The canonical `RESULT prefill ... pp_marginal_tok_s=` line is the reused
// number; `pp_wall_tok_s` is the same steady-state throughput (no session build
// in the timed region). Session creation is never folded into it.
//
// Before either section an untimed warm-up session/prefill initializes
// process-wide first-use state (CUDA module loading, autotune selection
// records, the device pool). "First prefill latency" is therefore the session's
// own first prefill — its segment build and capture — not one process's
// one-time kernel setup, and it means the same thing in both sections.
//
// Both scenarios are run by default. `--session reused` is the quick canonical
// path (tools/bench.sh uses it); `--session fresh` reports the lifecycle cost.
//
// Usage: prefill_bench [--prompt N] [--reps R] [--session reused|fresh|both]
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
#include <span>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

namespace {

// The same tool must build against the B0.0 baseline library and the current
// one. B0.2 added the session-keyed runtime registries (release_session_runtimes)
// and KvCacheSession::cache_id; before that the runtimes were keyed by
// (model, n_ctx) and nothing had to be released, and the warm-up took no
// session. These helpers compile the new calls only where they exist, using
// ADL at instantiation time, so the measurement semantics stay identical on
// both libraries.
template <class Model>
void release_session_runtimes_if_present(const Model& m, uint64_t n_ctx,
                                         uint64_t cache_id) {
  if constexpr (requires { release_session_runtimes(m, n_ctx, cache_id); }) {
    release_session_runtimes(m, n_ctx, cache_id);
  } else {
    (void)m, (void)n_ctx, (void)cache_id;
  }
}

template <class Model>
bool warmup_session_runtimes(const Model& m, uint64_t n_ctx,
                             sbplanner::KvCacheSession& session) {
  if constexpr (requires { warmup_spine_cuda(m, n_ctx, &session); })
    return warmup_spine_cuda(m, n_ctx, &session);
  else if constexpr (requires { warmup_spine_cuda(m, n_ctx); })
    return warmup_spine_cuda(m, n_ctx);
  else
    return true;
}

template <class Session>
uint64_t session_cache_id(const Session& s) {
  if constexpr (requires { s.cache_id; }) return s.cache_id;
  else return 0;
}

}  // namespace

int main(int argc, char** argv) {
  uint64_t n = 512;
  uint64_t reps = 3;
  enum class Mode { Reused, Fresh, Both } mode = Mode::Both;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--prompt")) n = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--reps")) reps = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--session")) {
      const char* m = argv[++i];
      if (!strcmp(m, "reused")) mode = Mode::Reused;
      else if (!strcmp(m, "fresh")) mode = Mode::Fresh;
      else if (!strcmp(m, "both")) mode = Mode::Both;
      else {
        fprintf(stderr, "--session expects reused|fresh|both (got '%s')\n", m);
        return 2;
      }
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      return 2;
    }
  }
  if (n == 0 || reps == 0) {
    fprintf(stderr,
            "usage: prefill_bench [--prompt N] [--reps R] "
            "[--session reused|fresh|both]\n");
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

  // Deterministic prompt (token values don't affect matmul/flash-attention cost).
  auto make_prompt = [](uint64_t len) {
    std::vector<uint64_t> p(len);
    for (uint64_t i = 0; i < len; ++i) p[i] = 2 + (i % 500);
    return p;
  };
  const std::vector<uint64_t> prompt = make_prompt(n);

  // The prefill chunk the runtime uses (the SWA window where the model has one,
  // the fixed default segment otherwise). It decides which lengths take the
  // same execution path, which is what the marginal baseline has to respect.
  const uint64_t chunk = std::max<uint64_t>(
      sbmodel::kv_cache_spec(m->config).chunk, 1);
  // Same-execution-path cancellation baseline. Never 1: that path is the
  // single-column kernels and skips the segment/capture machinery entirely.
  uint64_t n0 = 0;
  if (n > chunk) n0 = n - chunk;
  else n0 = n / 2;
  n0 = std::max<uint64_t>(n0, 2);
  const bool have_marginal = n0 < n;
  const std::vector<uint64_t> baseline_prompt = make_prompt(n0);

  // NVTX call markers (opt-in via SONICBOOM_NVTX, the runtime's own switch):
  // nsys can then isolate exactly one measured prefill call. The range wraps
  // the timed region, so the host duration of the range is the call latency.
  const bool nvtx_on = std::getenv("SONICBOOM_NVTX") != nullptr;

  // One full prefill on `session`. filled_tokens is cleared first: the session
  // keeps its KV allocation and runtimes, but the next call prefills the prompt
  // from position 0 (no prefix reuse), so repeated reps do identical work.
  auto prefill = [&](sbplanner::KvCacheSession& session,
                     std::span<const uint64_t> p, const char* label,
                     double* ms_out) -> bool {
    session.filled_tokens.clear();
    std::vector<int64_t> toks(1, -1);
    if (nvtx_on) nvtxRangePushA(label);
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t done = sbmodel::generate_prefill_spine_cuda_prefix(
        *m, p, 0, toks, session, /*use_prefill=*/true);
    const auto t1 = std::chrono::steady_clock::now();
    if (nvtx_on) nvtxRangePop();
    *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return done != 0;
  };

  struct Stats {
    std::vector<double> reps;  // ms, in execution order
    double best = 1e18;
    double median = 0.0;
    double mean = 0.0;
    bool ok = true;
  };
  auto measure = [&](sbplanner::KvCacheSession& session,
                     std::span<const uint64_t> p, uint64_t count,
                     const char* label) -> Stats {
    Stats s;
    s.reps.reserve(count);
    for (uint64_t rep = 0; rep < count; ++rep) {
      double ms = 0.0;
      if (!prefill(session, p, label, &ms)) { s.ok = false; return s; }
      s.reps.push_back(ms);
      s.best = std::min(s.best, ms);
      s.mean += ms;
    }
    s.mean /= double(s.reps.size());
    std::vector<double> sorted = s.reps;
    std::sort(sorted.begin(), sorted.end());
    const size_t k = sorted.size();
    s.median = (k % 2) ? sorted[k / 2]
                       : 0.5 * (sorted[k / 2 - 1] + sorted[k / 2]);
    return s;
  };
  auto print_reps = [](const char* what, const Stats& s) {
    fprintf(stderr, "[reps] %s ms=", what);
    for (size_t i = 0; i < s.reps.size(); ++i)
      fprintf(stderr, "%s%.3f", i ? "," : "", s.reps[i]);
    fprintf(stderr, " best=%.3f median=%.3f\n", s.best, s.median);
  };

  // Session creation, exactly what sb_llm_session_new charges to the session:
  // allocate the session's own KV cache, then build and warm the runtimes bound
  // to it (with SONICBOOM_WARMUP=0 the build moves into the first prefill).
  auto create_session = [&](uint64_t n_ctx, sbplanner::KvCacheSession& session,
                            double* ms_out) -> bool {
    session.n_ctx = n_ctx;
    const auto t0 = std::chrono::steady_clock::now();
    bool ok = sbmodel::kv_session_allocate(*m, session, n_ctx);
    if (ok) ok = warmup_session_runtimes(*m, n_ctx, session);
    const auto t1 = std::chrono::steady_clock::now();
    *ms_out = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ok;
  };
  // Mirror sb_llm_session_free: release the runtimes bound to the session's
  // cache identity before dropping the set. On a library without session-keyed
  // registries there is nothing to release.
  auto destroy_session = [&](sbplanner::KvCacheSession& session) {
    release_session_runtimes_if_present(*m, session.n_ctx,
                                        session_cache_id(session));
    session.caches.reset();
  };

  // Untimed process warm-up. It initializes process-wide state the first use
  // of a shape pays (CUDA module loading, autotune selection records, the
  // device pool) on a throwaway session. Without it the first measured session
  // would carry those costs in its "first prefill" (732 ms vs 294 ms on Gemma
  // pp512) and the fresh and reused sections would disagree depending on which
  // ran first. Session-scoped work is not warmed here: a prefill segment plan
  // is built per cache identity, so every measured first prefill still pays its
  // own segment build.
  {
    sbplanner::KvCacheSession warm;
    double create_ms = 0.0, prefill_ms = 0.0;
    if (!create_session(n + 1, warm, &create_ms) ||
        !prefill(warm, prompt, "bench:process_warmup", &prefill_ms)) {
      fprintf(stderr, "process warm-up failed\n");
      return 1;
    }
    fprintf(stderr,
            "[warmup] process first use (excluded): session create %.3f ms, "
            "first prefill %.3f ms\n",
            create_ms, prefill_ms);
    destroy_session(warm);
  }

  double canonical_pp_warm = 0.0;
  double first_prefill_ms = 0.0;
  double reused_create_ms = 0.0;

  // --- reused: one session, one KV set, one runtime, one segment capture ----
  if (mode != Mode::Fresh) {
    sbplanner::KvCacheSession session;
    if (!create_session(n + 1, session, &reused_create_ms)) {
      fprintf(stderr, "session creation failed (reused mode)\n");
      return 1;
    }
    // The first prefill on this session builds whatever the warm-up did not:
    // the prefill segments and, for n > chunk, the captured full-chunk graph.
    // It is reported separately and excluded from the warm repetitions.
    if (!prefill(session, prompt, "bench:reused_first", &first_prefill_ms)) {
      fprintf(stderr, "prefill rejected at --prompt %llu\n",
              (unsigned long long)n);
      return 1;
    }
    const Stats sn = measure(session, prompt, reps, "bench:reused_N");
    if (!sn.ok) {
      fprintf(stderr, "reused prefill failed\n");
      return 1;
    }
    print_reps("reused_N", sn);

    // Marginal baseline on the same session: one untimed warm call (so a
    // non-symbolic plan for this length is built outside the measurement),
    // then the reps.
    Stats s0;
    if (have_marginal) {
      double warm_ms = 0.0;
      if (!prefill(session, baseline_prompt, "bench:reused_baseline_warm",
                   &warm_ms)) {
        fprintf(stderr, "marginal baseline prefill failed\n");
        return 1;
      }
      s0 = measure(session, baseline_prompt, reps, "bench:reused_baseline");
      if (!s0.ok) {
        fprintf(stderr, "marginal baseline prefill failed\n");
        return 1;
      }
      print_reps("reused_baseline", s0);
    }

    const double t_warm = sn.best;  // ms
    canonical_pp_warm = t_warm > 0.0 ? 1000.0 * double(n) / t_warm : 0.0;
    double pp_marginal = 0.0;
    if (have_marginal) {
      const double dt = t_warm - s0.best;  // ms
      pp_marginal = dt > 0.0 ? 1000.0 * double(n - n0) / dt : 0.0;
    }
    printf("prefill --prompt %llu: reused session, %.3f ms warm "
           "(median %.3f, best of %llu) | pp_marginal %.1f tok/s "
           "(CANONICAL: full prefill, no session build in the timed region) | "
           "pp_warm %.1f tok/s (N/t_warm) | first prefill %.3f ms | "
           "session create %.3f ms\n",
           (unsigned long long)n, t_warm, sn.median,
           (unsigned long long)reps, pp_marginal, canonical_pp_warm,
           first_prefill_ms, reused_create_ms);
    printf("RESULT prefill prompt=%llu pp_marginal_tok_s=%.1f "
           "pp_wall_tok_s=%.1f session=reused t_warm_ms=%.3f "
           "t_baseline_ms=%.3f baseline=%llu first_prefill_ms=%.3f "
           "session_create_ms=%.3f\n",
           (unsigned long long)n, pp_marginal, canonical_pp_warm, t_warm,
           have_marginal ? s0.best : 0.0, (unsigned long long)n0,
           first_prefill_ms, reused_create_ms);
    // Median-of-reps view, alongside the canonical best-of line above.
    printf("RESULT_MEDIAN prefill prompt=%llu t_warm_ms=%.3f "
           "t_baseline_ms=%.3f pp_marginal_tok_s=%.1f pp_warm_tok_s=%.1f\n",
           (unsigned long long)n, sn.median, have_marginal ? s0.median : 0.0,
           (have_marginal && sn.median > s0.median)
               ? 1000.0 * double(n - n0) / (sn.median - s0.median)
               : 0.0,
           sn.median > 0.0 ? 1000.0 * double(n) / sn.median : 0.0);

    destroy_session(session);
  }

  // --- fresh: a new session (and its runtimes) per repetition --------------
  if (mode != Mode::Reused) {
    Stats create, first, e2e;
    create.reps.reserve(reps);
    first.reps.reserve(reps);
    e2e.reps.reserve(reps);
    for (uint64_t rep = 0; rep < reps; ++rep) {
      sbplanner::KvCacheSession session;
      double create_ms = 0.0, first_ms = 0.0;
      if (!create_session(n + 1, session, &create_ms)) {
        fprintf(stderr, "session creation failed (fresh rep %llu)\n",
                (unsigned long long)rep);
        return 1;
      }
      if (!prefill(session, prompt, "bench:fresh_first", &first_ms)) {
        fprintf(stderr, "fresh prefill failed at rep %llu\n",
                (unsigned long long)rep);
        return 1;
      }
      const double total_ms = create_ms + first_ms;
      create.reps.push_back(create_ms);
      first.reps.push_back(first_ms);
      e2e.reps.push_back(total_ms);
      printf("RESULT prefill_fresh prompt=%llu rep=%llu "
             "session_create_ms=%.3f first_prefill_ms=%.3f e2e_ms=%.3f "
             "e2e_tok_s=%.1f first_prefill_tok_s=%.1f\n",
             (unsigned long long)n, (unsigned long long)rep, create_ms,
             first_ms, total_ms, total_ms > 0.0 ? 1000.0 * double(n) / total_ms : 0.0,
             first_ms > 0.0 ? 1000.0 * double(n) / first_ms : 0.0);
      // Drop this session's runtimes and KV before the next repetition; a
      // fresh-session loop must not accumulate registries or device memory.
      destroy_session(session);
    }
    auto finish = [](Stats& s) {
      std::vector<double> sorted = s.reps;
      std::sort(sorted.begin(), sorted.end());
      const size_t k = sorted.size();
      s.median = (k % 2) ? sorted[k / 2]
                         : 0.5 * (sorted[k / 2 - 1] + sorted[k / 2]);
      s.best = sorted.empty() ? 0.0 : sorted.front();
      s.mean = 0.0;
      for (double v : s.reps) s.mean += v;
      if (!s.reps.empty()) s.mean /= double(s.reps.size());
    };
    finish(create);
    finish(first);
    finish(e2e);
    print_reps("fresh_session_create", create);
    print_reps("fresh_first_prefill", first);
    print_reps("fresh_e2e", e2e);
    printf("RESULT prefill_fresh_stats prompt=%llu reps=%llu "
           "session_create_ms=%.3f first_prefill_ms=%.3f e2e_ms=%.3f "
           "e2e_tok_s=%.1f\n",
           (unsigned long long)n, (unsigned long long)reps, create.median,
           first.median, e2e.median,
           e2e.median > 0.0 ? 1000.0 * double(n) / e2e.median : 0.0);
  }

  return 0;
}
