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

// S2: the adaptive draft-length policy's cold-start bootstrap
// (SONICBOOM_SPEC=auto).
//
// With alpha at its 0.5 prior and nearly-linear measured costs, k = 0 wins the
// cost model, no draft is ever proposed, and alpha is never updated: the policy
// deadlocks at k = 0. The bootstrap forces k = 3 for the first
// SONICBOOM_SPEC_BOOTSTRAP adaptive iterations after the cost warm-up, so the
// acceptance-rate EMA sees real drafts; the cost model then takes over.
//
// The contract checked here:
//   - exactly SONICBOOM_SPEC_BOOTSTRAP (50) iterations are forced to k = 3;
//   - the policy decides at least one later iteration (the bootstrap ends);
//   - the emitted stream is still the plain greedy stream, bit for bit — the
//     forced drafts are proposals only and the verifier decides.
//
// SONICBOOM_SPEC and SONICBOOM_SPEC_BOOTSTRAP are parsed once per process, so
// the case runs in a forked child with the environment set before its first
// call. Skips cleanly when the Qwen GGUF is absent or there is no CUDA device.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <span>
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

// Forced iterations after the warm-up. The warm-up itself (2 passes x k = 0..7)
// only measures costs and is not counted by the bootstrap.
constexpr uint64_t kBootstrap = 50;
// Enough emitted tokens that the warm-up and 50 four-token bootstrap iterations
// both complete before the output bound is reached (worst case 72 + 200).
constexpr uint64_t kOut = 320;
constexpr uint64_t kNctx = kOut + 64;

int run_case(const sbmodel::Gemma4Model& m, const std::string& tune_dir) {
  g_failures = 0;
  setenv("SONICBOOM_SPEC", "auto", 1);
  setenv("SONICBOOM_SPEC_BOOTSTRAP", "50", 1);
  setenv("SONICBOOM_TUNE_DIR", tune_dir.c_str(), 1);

  // The reference session is allocated first: the speculative run publishes the
  // decode runtime's cache, and the reference decodes through its own session.
  sbplanner::KvCacheSession b;
  if (!sbmodel::kv_session_allocate(m, b, kNctx)) {
    check(false, "kv_session_allocate failed");
    return g_failures;
  }

  const uint64_t seed = m.config.vocab_size / 3;
  sbmodel::spec_counters_reset();
  std::vector<int64_t> got(kOut, -1);
  const uint64_t done = sbmodel::generate_spine_cuda(m, seed, 0, got, kNctx);
  const sbmodel::SpecCounters counters = sbmodel::spec_counters();
  check(done == kOut, "bootstrap run produced all tokens");
  check(counters.bootstrap_steps == kBootstrap,
        "the first 50 adaptive iterations were forced to k = 3");
  check(counters.policy_steps > 0, "the policy took over after the bootstrap");

  // The plain per-step decode over the same start state is the lossless
  // reference: forced drafts are proposals, the verifier decides.
  std::vector<int64_t> ref(kOut, -1);
  uint64_t token = seed;
  for (uint64_t i = 0; i < kOut; ++i) {
    std::vector<int64_t> one(1, -1);
    if (!sbmodel::decode_steps_spine_cuda(
            m, std::span<const uint64_t>(&token, 1), i, b, one))
      break;
    ref[i] = one[0];
    if (one[0] < 0) break;
    token = uint64_t(one[0]);
  }
  check(ref == got, "bootstrap stream is bit-identical to plain decode");

  std::cout << "  " << kOut << " tokens, " << counters.iterations
            << " iterations, " << counters.bootstrap_steps
            << " bootstrap (k=3), " << counters.policy_steps
            << " policy, " << counters.accepted_tokens << "/"
            << counters.draft_tokens << " drafts accepted\n";
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
  const std::string path = find_file(
      "SONICBOOM_QWEN_GGUF", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  if (path.empty()) {
    std::cout << "  (skipped: GGUF not found)\n";
    std::cout << "test_spec_bootstrap OK\n";
    return 0;
  }
  // The device probe must run in a child. CUDA is not fork-safe: probing in
  // the parent initializes a context that the forked child cannot allocate in
  // (the case below would fail with "device buffer allocation failed").
  if (run_in_child([] { return sbquant::cuda_available() ? 0 : 2; }) == 2) {
    std::cout << "  (skipped: no CUDA device)\n";
    std::cout << "test_spec_bootstrap OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    std::cerr << "test_spec_bootstrap FAILED: GGUF load\n";
    return 1;
  }
  auto l = sbmodel::load_gemma4(*r);
  if (!l) {
    std::cerr << "  load error: " << l.error() << "\n";
    std::cerr << "test_spec_bootstrap FAILED: model load\n";
    return 1;
  }

  const std::string tune_dir =
      "/tmp/sb-spec-bootstrap-" + std::to_string(getpid());
  std::error_code ec;
  std::filesystem::remove_all(tune_dir, ec);
  const int failures = run_in_child([&] { return run_case(*l, tune_dir); });
  std::filesystem::remove_all(tune_dir, ec);

  if (failures != 0) {
    std::cerr << "test_spec_bootstrap FAILED: " << failures << " check(s)\n";
    return 1;
  }
  std::cout << "test_spec_bootstrap OK\n";
  return 0;
}
