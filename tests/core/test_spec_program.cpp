// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// B6 speculative generation program (CUDA): the accept/emit decision's
// truncation rules, and the loop's bit-identity with plain greedy decode.
//
// The speculative contract is lossless: the accepted run is exactly the tokens
// greedy decoding would have emitted. That has two halves:
//
//   1. spec_accept_run, the pure accept/emit decision: the longest matching
//      draft prefix, then the verifier's token, cut at a stop token inside the
//      run exactly as the plain loop cuts (the stop token is emitted, and the
//      valid KV length stops at its position) or at the caller's bound;
//   2. the loop: with the forced "repeat" drafter (every iteration proposes,
//      so every iteration runs a verify) its token stream must equal the stream
//      of chained plain decode steps from the same starting state, for Gemma 4
//      E4B and Qwen2.5 0.5B at depth 0 and 512.
//
// Part 3 covers B7's SONICBOOM_SPEC=auto: the draft-length policy's warm-up and
// cost cache, with the same bit-identity contract.
//
// SONICBOOM_SPEC=on is set before the first call: the mode is parsed once per
// process. The reference stream uses decode_steps_spine_cuda, which feeds given
// tokens through the uncaptured decode dispatch — the plain per-step decode.
//
// Each CUDA case runs in its own forked child for the same reason
// test_spec_invariance does (device pools retained per case). Skips cleanly when
// the models are absent or there is no CUDA device.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;
namespace sbquant = sonicboom::quant;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::string find_file(const char* env, const std::string& rel) {
  std::vector<std::string> candidates;
  if (const char* e = std::getenv(env)) candidates.push_back(e);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) + "/" + rel);
#endif
  candidates.push_back(rel);
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

std::vector<uint64_t> synth_tokens(uint64_t vocab, size_t n) {
  std::vector<uint64_t> t(n);
  uint64_t x = 12345;
  for (size_t i = 0; i < n; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;  // PCG constants
    t[i] = ((x >> 33) % vocab);
  }
  return t;
}

// --- part 1: the accept/emit decision --------------------------------------

struct AcceptCase {
  std::vector<uint64_t> drafts;
  std::vector<int64_t> rows;
  std::vector<int64_t> eos;
  uint64_t before;
  uint64_t limit;
  uint64_t expected_accepted;
  std::vector<uint64_t> expected_run;
  bool expected_stop;
};

void run_accept_case(const AcceptCase& c, int index) {
  std::vector<uint64_t> out(c.drafts.size() + 1, ~uint64_t(0));
  const sbmodel::SpecAccept r = sbmodel::spec_accept_run(
      c.drafts, c.rows, c.eos, c.before, c.limit, out);
  const bool ok = r.accepted == c.expected_accepted &&
                  r.emitted == c.expected_run.size() &&
                  r.stopped == c.expected_stop &&
                  std::equal(c.expected_run.begin(), c.expected_run.end(),
                             out.begin());
  if (!ok) {
    std::cerr << "  FAIL: accept case " << index << ": accepted=" << r.accepted
              << " emitted=" << r.emitted << " stopped=" << r.stopped
              << " run=[";
    for (size_t i = 0; i < r.emitted; ++i)
      std::cerr << (i ? "," : "") << out[i];
    std::cerr << "]\n";
    ++g_failures;
  }
}

void accept_unit_tests() {
  const std::vector<int64_t> none;
  // Every draft matches: the run is the drafts plus the verifier's token.
  run_accept_case({{10, 11, 12}, {10, 11, 12, 13}, none, 0, 100, 3,
                   {10, 11, 12, 13}, false},
                  0);
  // The first rejection keeps the drafts before it plus that row's token.
  run_accept_case({{10, 11, 12}, {10, 99, 12, 13}, none, 0, 100, 1,
                   {10, 99}, false},
                  1);
  run_accept_case({{10, 11, 12}, {99, 11, 12, 13}, none, 0, 100, 0,
                   {99}, false},
                  2);
  // A stop token inside the accepted run is emitted, and emission stops there.
  run_accept_case({{10, 11, 12}, {10, 11, 12, 13}, {11}, 0, 100, 3,
                   {10, 11}, true},
                  3);
  // The verifier's own token can be the stop token.
  run_accept_case({{10, 11, 12}, {10, 11, 12, 13}, {13}, 0, 100, 3,
                   {10, 11, 12, 13}, true},
                  4);
  // The caller's bound cuts the run without a stop.
  run_accept_case({{10, 11, 12}, {10, 11, 12, 13}, none, 0, 2, 3,
                   {10, 11}, false},
                  5);
  // The bound already reached: nothing is emitted.
  run_accept_case({{10, 11, 12}, {10, 11, 12, 13}, none, 7, 7, 3, {}, false},
                  6);
  run_accept_case({{10, 11, 12}, {10, 11, 12, 13}, none, 5, 7, 3,
                   {10, 11}, false},
                  7);
  // A row span too short for the drafts emits nothing (contract violation).
  run_accept_case({{10, 11, 12}, {10, 11}, none, 0, 100, 0, {}, false}, 8);
  if (g_failures == 0) std::cout << "  accept/emit truncation rules: OK\n";
}

// --- part 1b: the B7 draft-length policy ------------------------------------

void run_choose_case(double alpha, std::vector<double> costs, int64_t max_k,
                     int64_t expected, int index) {
  const int64_t got = sbmodel::spec_choose_drafts(alpha, costs, max_k);
  if (got != expected) {
    std::cerr << "  FAIL: choose case " << index << ": alpha=" << alpha
              << " max_k=" << max_k << " got k=" << got << " expected "
              << expected << "\n";
    ++g_failures;
  }
}

void policy_unit_tests() {
  // alpha0 = 0.5 with rising costs: one draft pays, the longer runs do not.
  run_choose_case(0.5, {2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5}, 7, 1, 0);
  // Flat costs: the expected geometric run keeps growing, so the cap wins.
  run_choose_case(0.9, {1, 1, 1, 1, 1, 1, 1, 1}, 7, 7, 1);
  // A weak drafter: the plain step is the best value.
  run_choose_case(0.1, {1, 10, 10, 10}, 3, 0, 2);
  // Nothing measured (autotune disabled): the fixed cap, `on`'s behavior.
  run_choose_case(0.5, {0, 0, 0, 0}, 3, 3, 3);
  // alpha == 1: E(k) = k + 1, still linear in the measured cost.
  run_choose_case(1.0, {2.0, 3.0, 4.0}, 2, 2, 4);
  // max_k past the measured span only considers the span.
  run_choose_case(0.5, {2.0, 1.0, 1.0}, 7, 2, 5);
  // k = 0 is a legitimate choice (no draft at all).
  run_choose_case(0.0, {1.0, 5.0}, 1, 0, 6);

  // The EMA: alpha <- 0.5*alpha + 0.5*(accepted/drafted).
  const double a1 = sbmodel::spec_alpha_update(0.5, 3, 7);
  check(std::abs(a1 - (0.25 + 0.5 * 3.0 / 7.0)) < 1e-12, "alpha EMA value");
  check(sbmodel::spec_alpha_update(0.4, 0, 0) == 0.4,
        "alpha EMA ignores an empty draft");
  if (g_failures == 0) std::cout << "  draft-length policy: OK\n";
}

// --- part 2: the speculative loop vs chained plain decode -------------------

constexpr uint64_t kOut = 16;

int run_case(const sbmodel::Gemma4Model& m, const std::vector<uint64_t>& all,
             uint64_t depth) {
  g_failures = 0;
  const uint64_t n_ctx = depth + 64;
  const std::span<const uint64_t> prompt(all.data(), size_t(depth));
  // The loop's starting token. For depth 0 the synthetic first token stands in
  // for a prefill's sample; for depth > 0 both sessions sample it from the same
  // prefill below.
  uint64_t seed = all[depth];

  // Session B: the reference. Cache-loss at first use would publish the decode
  // runtime's cache, so allocate it explicitly to keep A and B independent.
  sbplanner::KvCacheSession b;
  if (!sbmodel::kv_session_allocate(m, b, n_ctx)) {
    check(false, "kv_session_allocate failed");
    return g_failures;
  }

  if (depth > 0) {
    // Session A: the speculative loop's cache. Left cache-less so its prefill
    // publishes the decode runtime's cache, which spec_generate_spine_cuda then
    // reuses.
    sbplanner::KvCacheSession a;
    a.n_ctx = n_ctx;
    std::vector<int64_t> seed_a(1, -1);
    check(sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, seed_a, a,
                                                      true) == 1,
          "speculative session prefill failed");
    std::vector<int64_t> seed_b(1, -1);
    check(sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, seed_b, b,
                                                      true) == 1,
          "reference session prefill failed");
    check(seed_a == seed_b, "both prefills sampled the same token");
    if (seed_a[0] < 0) return g_failures;
    seed = uint64_t(seed_a[0]);
  }

  // The speculative loop, with the always-proposing drafter: every iteration
  // verifies, so the verify/accept/emit path is exercised every step.
  sbmodel::spec_counters_reset();
  std::vector<int64_t> spec(kOut, -1);
  const uint64_t spec_done = sbmodel::spec_generate_spine_cuda(
      m, seed, depth, spec, n_ctx, "repeat");
  const sbmodel::SpecCounters counters = sbmodel::spec_counters();
  check(spec_done == kOut, "speculative loop produced all tokens");
  check(counters.iterations > 0 && counters.verify_steps == counters.iterations,
        "every iteration ran a verify (repeat drafter)");
  check(counters.plain_steps == 0, "no iteration fell back to the plain step");
  check(counters.emitted_tokens == kOut, "the counters saw every emitted token");

  // The reference: feed each sampled token back through the plain decode step.
  std::vector<int64_t> ref(kOut, -1);
  uint64_t token = seed;
  for (uint64_t i = 0; i < kOut; ++i) {
    std::vector<int64_t> one(1, -1);
    if (!sbmodel::decode_steps_spine_cuda(m, std::span<const uint64_t>(&token, 1),
                                          depth + i, b, one))
      break;
    ref[i] = one[0];
    if (one[0] < 0) break;
    token = uint64_t(one[0]);
  }

  size_t diverge = 0;
  while (diverge < ref.size() && diverge < spec.size() &&
         ref[diverge] == spec[diverge])
    ++diverge;
  check(diverge == ref.size(), "speculative stream is bit-identical to plain");
  std::cout << "  depth=" << depth << ": " << kOut << " tokens, "
            << counters.verify_steps << " verifies, "
            << counters.draft_tokens << " drafts, "
            << counters.accepted_tokens << " accepted"
            << (diverge == ref.size() ? "" : " (DIVERGED)")
            << "\n";
  return g_failures;
}

// --- part 3: SONICBOOM_SPEC=auto (B7) --------------------------------------

// The adaptive policy's contract is still the plain greedy stream. A 32-token
// generation covers the warm-up (k = 0..7, two passes each: the first absorbs
// that row count's one-time runtime construction, the second is timed) and
// leaves policy iterations after it. The forced drafts of the warm-up are
// proposals only, so they must not leak into the emitted stream, and the
// measured costs must be persisted under the child's private tune directory.
// The spec mode and the tune directory are parsed once per process, so this
// case runs in its own child with the environment set before the first call.
int run_auto_case(const sbmodel::Gemma4Model& m, const std::string& tune_dir) {
  g_failures = 0;
  setenv("SONICBOOM_SPEC", "auto", 1);
  setenv("SONICBOOM_TUNE_DIR", tune_dir.c_str(), 1);
  // The warm-up is 16 iterations (two per k = 0..7); with every forced draft
  // accepted one iteration can emit k+1 tokens, so the bounds must fit
  // 2 * (1 + ... + 8) = 72 emitted tokens before the policy gets its turn.
  constexpr uint64_t kAutoOut = 128;
  const uint64_t n_ctx = 256;
  const uint64_t seed = m.config.vocab_size / 3;

  // The reference's session is allocated first: the auto run publishes the
  // decode runtime's cache, and the reference decodes through its own session.
  sbplanner::KvCacheSession b;
  if (!sbmodel::kv_session_allocate(m, b, n_ctx)) {
    check(false, "kv_session_allocate failed");
    return g_failures;
  }

  sbmodel::spec_counters_reset();
  std::vector<int64_t> got(kAutoOut, -1);
  const uint64_t done = sbmodel::generate_spine_cuda(m, seed, 0, got, n_ctx);
  const sbmodel::SpecCounters counters = sbmodel::spec_counters();
  check(done == kAutoOut, "auto loop produced all tokens");
  check(counters.iterations >= 16, "the warm-up ran (2 passes x k = 0..7)");
  check(counters.plain_steps >= 2, "the warm-up primed and timed c(0)");

  std::vector<int64_t> ref(kAutoOut, -1);
  uint64_t token = seed;
  for (uint64_t i = 0; i < kAutoOut; ++i) {
    std::vector<int64_t> one(1, -1);
    if (!sbmodel::decode_steps_spine_cuda(m, std::span<const uint64_t>(&token, 1),
                                          i, b, one))
      break;
    ref[i] = one[0];
    if (one[0] < 0) break;
    token = uint64_t(one[0]);
  }
  size_t diverge = 0;
  while (diverge < ref.size() && ref[diverge] == got[diverge]) ++diverge;
  check(diverge == ref.size(), "auto stream is bit-identical to plain");

  bool cached = false;
  std::error_code ec;
  for (const auto& e : std::filesystem::recursive_directory_iterator(tune_dir, ec)) {
    if (e.path().filename() != "spec-length.json") continue;
    std::ifstream f(e.path());
    const std::string s((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    cached = s.find("\"c0\"") != std::string::npos &&
             s.find("\"c1\"") != std::string::npos;
  }
  check(cached, "the warm-up cached c(0) and c(1)");
  std::cout << "  " << kAutoOut << " tokens, " << counters.iterations
            << " iterations, " << counters.verify_steps << " verifies, "
            << counters.accepted_tokens << "/" << counters.draft_tokens
            << " drafts accepted"
            << (diverge == ref.size() ? "" : " (DIVERGED)") << "\n";
  return g_failures;
}

template <class F>
int run_in_child(F&& fn) {
  std::cout.flush();
  std::cerr.flush();
  const pid_t pid = fork();
  if (pid < 0) {
    std::perror("  fork");
    return 1;
  }
  if (pid == 0) {
    int rc = 1;
    try {
      rc = fn();
    } catch (...) {
      rc = 1;
    }
    std::cout.flush();
    std::cerr.flush();
    _exit(rc);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) < 0) return 1;
  if (WIFEXITED(status)) return WEXITSTATUS(status);
  std::cerr << "  (child terminated by signal " << WTERMSIG(status) << ")\n";
  return 1;
}

} // namespace

int main() {
  accept_unit_tests();
  policy_unit_tests();

  const std::string gemma_path =
      find_file("SONICBOOM_GEMMA_GGUF", "models/gemma-4-E4B-it-Q3_K_M.gguf");
  const std::string qwen_path = find_file(
      "SONICBOOM_QWEN_GGUF", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  if (gemma_path.empty() || qwen_path.empty()) {
    std::cout << "  (CUDA cases skipped: model GGUF not found)\n";
    if (g_failures == 0) {
      std::cout << "test_spec_program OK\n";
      return 0;
    }
    std::cerr << "test_spec_program FAILED: " << g_failures << " check(s)\n";
    return 1;
  }

  // The mode is parsed once per process, so it is set before any call.
  setenv("SONICBOOM_SPEC", "on", 1);

  // Host-only loads: the parent never touches CUDA.
  auto r_gemma = sbgguf::Reader::load(gemma_path);
  auto r_qwen = sbgguf::Reader::load(qwen_path);
  if (!r_gemma || !r_qwen) {
    std::cerr << "test_spec_program FAILED: GGUF load\n";
    return 1;
  }
  auto lg = sbmodel::load_gemma4(*r_gemma);
  auto lq = sbmodel::load_gemma4(*r_qwen);
  if (!lg || !lq) {
    std::cerr << "  load error: " << (lg ? lq.error() : lg.error()) << "\n";
    std::cerr << "test_spec_program FAILED: model load\n";
    return 1;
  }
  const sbmodel::Gemma4Model gemma = std::move(*lg);
  const sbmodel::Gemma4Model qwen = std::move(*lq);

  if (run_in_child([] { return sbquant::cuda_available() ? 0 : 2; }) == 2) {
    std::cout << "  (CUDA cases skipped: no CUDA device)\n";
    if (g_failures == 0) {
      std::cout << "test_spec_program OK\n";
      return 0;
    }
    std::cerr << "test_spec_program FAILED: " << g_failures << " check(s)\n";
    return 1;
  }

  int failures = 0;
  for (const auto* model : {&gemma, &qwen}) {
    const char* name = model->config.arch == sbmodel::ModelArch::Gemma4
                           ? "gemma4"
                           : "qwen2";
    // A repeated synthetic history, so the n-gram drafter has material (the
    // forced "repeat" strategy does not need it; the prefill's own loop may).
    std::vector<uint64_t> all = synth_tokens(model->config.vocab_size, 512 + 8);
    for (size_t i = 8; i < all.size(); ++i) all[i] = all[i % 8];
    for (uint64_t depth : {uint64_t(0), uint64_t(512)}) {
      std::cout << "model=" << name << " depth=" << depth << "\n";
      failures += run_in_child([&] { return run_case(*model, all, depth); });
    }
    const std::string dir = "/tmp/sb-spec-auto-" + std::to_string(getpid()) +
                            "-" + name;
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::cout << "model=" << name << " spec=auto\n";
    failures += run_in_child([&] { return run_auto_case(*model, dir); });
    std::filesystem::remove_all(dir, ec);
  }

  if (failures == 0) {
    std::cout << "test_spec_program OK\n";
    return 0;
  }
  std::cerr << "test_spec_program FAILED: " << failures << " check(s)\n";
  return 1;
}
