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

// B3 speculative verify-step invariance (CUDA). The verify step runs one decode
// graph over k+1 query positions, each row using the decode path's kernel
// arithmetic at pos+i. Lossless speculative decoding depends on that being
// *bit-identical* to k+1 plain decode steps, not close: the accepted tokens must
// be exactly the ones greedy decoding would have emitted.
//
// For each model (Gemma 4 E4B, Qwen2.5 0.5B), depth in {0, 512} and k in 1..7
// (token ids are deterministic and in-vocabulary for each model):
//   1. build two sessions with the same starting KV state — both prefilled with
//      the same prompt, or both freshly allocated at depth 0;
//   2. session A: k+1 plain decode steps, each fed the draft token at pos+i
//      (decode_steps_spine_cuda, the uncaptured per-step decode dispatch);
//   3. session B: one verify step over the same k+1 tokens
//      (verify_step_spine_cuda);
//   4. the argmax of every row must be identical.
//
// Every case runs in its own forked child: each case builds a verify graph for
// its row count plus two KV caches and the prefill's segment state, and on an
// 8 GiB device the retained device pools of earlier cases would push a later
// case's peak over the limit (the same reason test_prefix_cache forks). The
// parent loads the models (host only) before forking; it never touches CUDA.
//
// Skips cleanly when either model is absent, and when there is no CUDA device.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
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

// In-vocabulary synthetic token ids for one model. The shared real_prompt
// fixture is tokenized for Gemma and its ids exceed Qwen's vocabulary, which
// the embedding gather rejects; the invariance contract only needs valid ids,
// so both models get a deterministic sequence from their own vocab size.
std::vector<uint64_t> synth_tokens(uint64_t vocab, size_t n) {
  std::vector<uint64_t> t(n);
  uint64_t x = 12345;
  for (size_t i = 0; i < n; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;  // PCG constants
    t[i] = ((x >> 33) % vocab);
  }
  return t;
}

} // namespace

// One case: reference decode vs verify step from identical KV state.
int run_case(const sbmodel::Gemma4Model& m, const std::vector<uint64_t>& all,
             uint64_t depth, int k) {
  g_failures = 0;
  const int rows = k + 1;
  const uint64_t n_ctx = depth + 64;
  const std::span<const uint64_t> prompt(all.data(), size_t(depth));
  const std::vector<uint64_t> feed(all.begin() + depth,
                                   all.begin() + depth + rows);

  // Session A: the reference. Left cache-less so the first prefill publishes the
  // decode runtime's own cache into it (the prefix path's first-use contract);
  // n_ctx is set so that cache is sized for the verify positions too.
  sbplanner::KvCacheSession a;
  a.n_ctx = n_ctx;
  // Session B: the verify. Pre-allocated, so it is independent of A's cache.
  sbplanner::KvCacheSession b;
  if (!sbmodel::kv_session_allocate(m, b, n_ctx)) {
    check(false, "kv_session_allocate failed");
    return g_failures;
  }

  if (depth > 0) {
    // A first: on an empty session the prefill publishes the runtime's cache.
    std::vector<int64_t> seed_a(1, -1);
    check(sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, seed_a, a,
                                                      true) == 1,
          "reference session prefill failed");
    std::vector<int64_t> seed_b(1, -1);
    check(sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, seed_b, b,
                                                      true) == 1,
          "verify session prefill failed");
    check(seed_a == seed_b, "both prefills sampled the same token");
  } else {
    // Depth 0: no prefill, so the reference gets its own allocated cache; both
    // sessions start empty and write rows 0..k inside the compared steps.
    check(sbmodel::kv_session_allocate(m, a, n_ctx),
          "reference session allocation failed");
  }

  std::vector<int64_t> ref(rows, -1);
  check(sbmodel::decode_steps_spine_cuda(m, feed, depth, a, ref),
        "reference decode steps failed");
  std::vector<int64_t> ver(rows, -1);
  check(sbmodel::verify_step_spine_cuda(m, feed, depth, b, ver),
        "verify step failed");

  if (ref == ver) {
    std::cout << "  depth=" << depth << " k=" << k
              << ": verify identical to " << rows << " decode steps"
              << " (tokens " << ref.front() << ".." << ref.back() << ")\n";
    check(true, "verify rows identical to plain decode");
  } else {
    size_t div = 0;
    while (div < ref.size() && ref[div] == ver[div]) ++div;
    std::cout << "  depth=" << depth << " k=" << k << ": DIVERGE at row " << div
              << " (decode " << ref[div] << " vs verify " << ver[div] << ")\n";
    check(false, "verify rows identical to plain decode");
  }
  return g_failures;
}

// Run fn in a forked child; see the header note on why each case is isolated.
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

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const std::string gemma_path =
      find_file("SONICBOOM_GEMMA_GGUF", "models/gemma-4-E4B-it-Q3_K_M.gguf");
  const std::string qwen_path = find_file(
      "SONICBOOM_QWEN_GGUF", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  if (gemma_path.empty() || qwen_path.empty()) {
    std::cout << "  (skipped: model GGUF not found)\n";
    std::cout << "test_spec_invariance OK\n";
    return 0;
  }

  // Host-only loads: the parent never touches CUDA (a context is not inherited
  // across fork). The readers own the weight bytes the models reference, so
  // they must outlive every case (a destroyed reader leaves the quantized
  // tensors pointing at freed memory).
  auto r_gemma = sbgguf::Reader::load(gemma_path);
  auto r_qwen = sbgguf::Reader::load(qwen_path);
  if (!r_gemma || !r_qwen) {
    std::cerr << "test_spec_invariance FAILED: GGUF load\n";
    return 1;
  }
  auto lg = sbmodel::load_gemma4(*r_gemma);
  auto lq = sbmodel::load_gemma4(*r_qwen);
  if (!lg || !lq) {
    std::cerr << "  load error: " << (lg ? lq.error() : lg.error()) << "\n";
    std::cerr << "test_spec_invariance FAILED: model load\n";
    return 1;
  }
  const sbmodel::Gemma4Model gemma = std::move(*lg);
  const sbmodel::Gemma4Model qwen = std::move(*lq);

  if (run_in_child([] { return sbquant::cuda_available() ? 0 : 2; }) == 2) {
    std::cout << "  (skipped: no CUDA device)\n";
    std::cout << "test_spec_invariance OK\n";
    return 0;
  }

  int failures = 0;
  // Test-harness filters (not runtime switches): one depth, one model, one k,
  // an explicit depth list (the dense enumeration without an ONLY depth), and
  // an in-process (no fork) mode for running a case under a debugger.
  const char* only = std::getenv("SONICBOOM_SPEC_INVARIANCE_ONLY");  // depth
  const char* only_model = std::getenv("SONICBOOM_SPEC_INVARIANCE_MODEL");
  const char* only_k = std::getenv("SONICBOOM_SPEC_INVARIANCE_K");
  const bool nofork = std::getenv("SONICBOOM_SPEC_INVARIANCE_NOFORK") != nullptr;
  for (const auto* model : {&gemma, &qwen}) {
    const char* name = model->config.arch == sbmodel::ModelArch::Gemma4
                           ? "gemma4"
                           : "qwen2";
    if (only_model && std::string(only_model) != name) continue;
    const std::vector<uint64_t> all =
        synth_tokens(model->config.vocab_size, 512 + 8);
    std::vector<uint64_t> depths{0, 512};
    if (const char* dl = std::getenv("SONICBOOM_SPEC_INVARIANCE_DEPTHS")) {
      depths.clear();
      std::string s(dl);
      size_t p = 0;
      while (p <= s.size()) {
        const size_t q = s.find(',', p);
        depths.push_back(uint64_t(std::atoi(s.substr(p, q - p).c_str())));
        if (q == std::string::npos) break;
        p = q + 1;
      }
    }
    for (uint64_t depth : depths) {
      if (only && std::atoi(only) >= 0 && uint64_t(std::atoi(only)) != depth)
        continue;
      for (int k = 1; k <= 7; ++k) {
        if (only_k && std::atoi(only_k) != k) continue;
        std::cout << "model=" << name << " depth=" << depth << " k=" << k
                  << "\n";
        if (nofork) {
          failures += run_case(*model, all, depth, k);
        } else {
          failures +=
              run_in_child([&] { return run_case(*model, all, depth, k); });
        }
      }
    }
  }

  if (failures == 0) {
    std::cout << "test_spec_invariance OK\n";
    return 0;
  }
  std::cerr << "test_spec_invariance FAILED: " << failures << " check(s)\n";
  return 1;
}
