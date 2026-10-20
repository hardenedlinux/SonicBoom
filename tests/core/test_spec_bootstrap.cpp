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

// S2/S3: the adaptive draft-length policy's cold-start bootstrap
// (SONICBOOM_SPEC=auto).
//
// With alpha at its 0.5 prior and nearly-linear measured costs, k = 0 wins the
// cost model, no draft is ever proposed, and alpha is never updated: the policy
// deadlocks at k = 0. The bootstrap forces k = 1 for the first
// SONICBOOM_SPEC_BOOTSTRAP adaptive iterations after the cost warm-up, so the
// acceptance-rate EMA sees real drafts; the cost model then takes over.
//
// S3 adds an early abort: a run of SONICBOOM_SPEC_BOOTSTRAP_ABORT (5)
// consecutive bootstrap verify steps that accept nothing means the workload
// has no drafter matches, and the remaining forced iterations are pure verify
// cost. The contract checked here:
//   - SONICBOOM_SPEC_BOOTSTRAP_ABORT=0 restores the fixed 50-step ablation:
//     exactly SONICBOOM_SPEC_BOOTSTRAP (50) iterations are forced, and the
//     policy decides at least one later iteration;
//   - a copy-heavy trace (one token repeated) accepts the bootstrap drafts, so
//     the bootstrap runs to its 50-step end;
//   - a random-token trace accepts none, so the bootstrap ends within 10
//     iterations (5 consecutive zero-accept verify steps);
//   - every emitted stream is still the plain greedy stream, bit for bit — the
//     forced drafts are proposals only and the verifier decides.
//
// SONICBOOM_SPEC and the bootstrap knobs are parsed once per process, so each
// case runs in a forked child with the environment set before its first call.
// Skips cleanly when the Qwen GGUF is absent or there is no CUDA device.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/vocab/vocab.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;
namespace sbquant = sonicboom::quant;
namespace sbvocab = sonicboom::vocab;

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
constexpr uint64_t kAbort = 5;
// Enough emitted tokens that the warm-up and 50 two-token bootstrap iterations
// both complete before the output bound is reached (worst case 72 + 200).
constexpr uint64_t kOut = 320;
constexpr uint64_t kNctx = kOut + 64;
// The trace prompt length for the S3 cases (copy-heavy / random).
constexpr size_t kPromptLen = 64;

bool write_tokens(const std::string& path,
                  std::span<const int64_t> tokens) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(tokens.data()),
          std::streamsize(tokens.size() * sizeof(int64_t)));
  return bool(f);
}

std::vector<int64_t> read_tokens(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  const std::vector<char> bytes((std::istreambuf_iterator<char>(f)), {});
  std::vector<int64_t> out(bytes.size() / sizeof(int64_t));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(int64_t));
  return out;
}

bool write_meta(const std::string& path, const sbmodel::SpecCounters& c) {
  std::ofstream f(path);
  if (!f) return false;
  f << c.iterations << " " << c.bootstrap_steps << " " << c.policy_steps << " "
    << c.verify_steps << " " << c.accepted_tokens << " " << c.draft_tokens
    << "\n";
  return bool(f);
}

// The S2 ablation: early abort disabled, so the bootstrap owes exactly
// SONICBOOM_SPEC_BOOTSTRAP forced iterations. No prefill: the drafter starts
// from an empty history (its proposals are verified, never trusted).
int run_ablation_case(const sbmodel::Gemma4Model& m,
                      const std::string& tune_dir) {
  g_failures = 0;
  setenv("SONICBOOM_SPEC", "auto", 1);
  setenv("SONICBOOM_SPEC_BOOTSTRAP", "50", 1);
  setenv("SONICBOOM_SPEC_BOOTSTRAP_ABORT", "0", 1);
  setenv("SONICBOOM_TUNE_DIR", tune_dir.c_str(), 1);

  const uint64_t seed = m.config.vocab_size / 3;
  sbmodel::spec_counters_reset();
  std::vector<int64_t> got(kOut, -1);
  const uint64_t done = sbmodel::generate_spine_cuda(m, seed, 0, got, kNctx);
  const sbmodel::SpecCounters counters = sbmodel::spec_counters();
  check(done == kOut, "bootstrap run produced all tokens");
  check(counters.bootstrap_steps == kBootstrap,
        "ABORT=0: the first 50 adaptive iterations were forced");
  check(counters.policy_steps > 0, "the policy took over after the bootstrap");

  // The reference session is allocated first: the speculative run publishes
  // the decode runtime's cache, and the reference decodes through its own.
  sbplanner::KvCacheSession b;
  if (!sbmodel::kv_session_allocate(m, b, kNctx)) {
    check(false, "kv_session_allocate failed");
    return g_failures;
  }
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

  std::cout << "  ablation: " << kOut << " tokens, " << counters.iterations
            << " iterations, " << counters.bootstrap_steps
            << " bootstrap (k=1), " << counters.policy_steps << " policy, "
            << counters.accepted_tokens << "/" << counters.draft_tokens
            << " drafts accepted\n";
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

// One arm of a trace: prefill + decode through the canonical no-session path
// (the decode runtime publishes its own cache, so the captured step and the
// prefill see the same buffers). The token stream goes to `tokens_path`; the
// auto arm also writes its counters to `meta_path`.
int run_trace_arm(const sbmodel::Gemma4Model& m, const std::string& tune_dir,
                  const std::vector<uint64_t>& prompt, bool auto_mode,
                  const std::string& tokens_path,
                  const std::string& meta_path) {
  g_failures = 0;
  setenv("SONICBOOM_SPEC", auto_mode ? "auto" : "off", 1);
  setenv("SONICBOOM_SPEC_BOOTSTRAP", "50", 1);
  setenv("SONICBOOM_SPEC_BOOTSTRAP_ABORT", "5", 1);
  setenv("SONICBOOM_TUNE_DIR", tune_dir.c_str(), 1);

  const uint64_t n_ctx = prompt.size() + kOut + 64;
  sbmodel::spec_counters_reset();
  std::vector<int64_t> out(kOut, -1);
  const uint64_t done = sbmodel::generate_prefill_spine_cuda(
      m, std::span<const uint64_t>(prompt), 0, out, n_ctx);
  const sbmodel::SpecCounters counters = sbmodel::spec_counters();
  check(done == kOut, "trace produced all tokens");
  check(write_tokens(tokens_path, out), "trace token stream written");
  if (auto_mode && !write_meta(meta_path, counters))
    check(false, "trace counters written");
  return g_failures;
}

// A prefill + decode trace on a prompt that either repeats one token
// (copy-heavy: the verifier's repeat proposals are likely accepted, so the
// bootstrap runs its full length) or uses unrelated tokens (no match: every
// bootstrap verify rejects, so the abort fires). The reference is the real
// SONICBOOM_SPEC=off loop, which must run in its own process (the mode is
// parsed once).
int run_trace_case(const sbmodel::Gemma4Model& m, const std::string& base,
                   const std::vector<uint64_t>& prompt, bool copy_heavy) {
  g_failures = 0;
  const std::string name = copy_heavy ? "copy" : "random";
  const std::string auto_tokens = base + "-" + name + "-auto.tok";
  const std::string off_tokens = base + "-" + name + "-off.tok";
  const std::string meta_path = base + "-" + name + "-auto.meta";

  int failures = 0;
  failures += run_in_child([&] {
    return run_trace_arm(m, base + "-" + name + "-auto-tune", prompt, true,
                         auto_tokens, meta_path);
  });
  failures += run_in_child([&] {
    return run_trace_arm(m, base + "-" + name + "-off-tune", prompt, false,
                         off_tokens, {});
  });

  const std::vector<int64_t> got = read_tokens(auto_tokens);
  const std::vector<int64_t> ref = read_tokens(off_tokens);
  check(got.size() == kOut, "auto trace stream read back");
  check(ref.size() == kOut, "off trace stream read back");
  size_t diverge = 0;
  while (diverge < got.size() && diverge < ref.size() &&
         got[diverge] == ref[diverge])
    ++diverge;
  if (diverge < got.size()) {
    std::cerr << "  divergence at " << diverge << ": auto=" << got[diverge]
              << " off=" << ref[diverge] << "\n";
  }
  check(got == ref, "trace stream is bit-identical to SONICBOOM_SPEC=off");

  sbmodel::SpecCounters counters;
  {
    std::ifstream f(meta_path);
    if (!(f >> counters.iterations >> counters.bootstrap_steps >>
          counters.policy_steps >> counters.verify_steps >>
          counters.accepted_tokens >> counters.draft_tokens))
      check(false, "trace counters read back");
  }
  check(counters.bootstrap_steps > 0, "the bootstrap ran");
  check(counters.policy_steps > 0, "the policy took over");
  if (copy_heavy) {
    check(counters.bootstrap_steps >= kBootstrap - 5,
          "copy-heavy trace: the bootstrap ran to (near) its 50-step end");
  } else {
    check(counters.bootstrap_steps <= 2 * kAbort,
          "no-match trace: the bootstrap aborted within 10 iterations");
  }

  std::cout << "  " << name << ": prompt=" << prompt.size() << " " << kOut
            << " tokens, " << counters.bootstrap_steps
            << " bootstrap (k=1), " << counters.policy_steps << " policy, "
            << counters.verify_steps << " verifies, "
            << counters.accepted_tokens << "/" << counters.draft_tokens
            << " drafts accepted, stream bit-identical to off\n";
  return failures + g_failures;
}

// A prompt of `kPromptLen` copies of one real token: a long run of the Qwen2
// space token. The model's greedy continuation of that run is the same token
// (measured on the GPTQ q4_k_m checkpoint at the test's n_ctx), so the
// verifier's repeat proposals match and the bootstrap has something to accept.
std::vector<uint64_t> copy_heavy_prompt(const sbvocab::Vocab& v) {
  const std::vector<int32_t> ids = sbvocab::tokenize(v, " ");
  if (ids.size() != 1) return {};
  return std::vector<uint64_t>(kPromptLen, uint64_t(ids[0]));
}

// Unrelated ids over the ordinary-token range: no structure for the drafter.
std::vector<uint64_t> no_match_prompt(const sbmodel::Gemma4Model& m) {
  std::vector<uint64_t> prompt;
  prompt.reserve(kPromptLen);
  uint64_t x = 0x2545F4914F6CDD1Dull;
  const uint64_t span = uint64_t(m.config.vocab_size) / 2;
  for (size_t i = 0; i < kPromptLen; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    prompt.push_back(128 + (x >> 17) % (span - 128));
  }
  return prompt;
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
  // (the cases below would fail with "device buffer allocation failed").
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
  auto vocab = sbvocab::load_vocab(*r);
  if (!vocab) {
    std::cerr << "  vocab error: " << vocab.error() << "\n";
    std::cerr << "test_spec_bootstrap FAILED: vocab load\n";
    return 1;
  }
  const std::vector<uint64_t> copy_prompt = copy_heavy_prompt(*vocab);
  if (copy_prompt.empty()) {
    std::cerr << "test_spec_bootstrap FAILED: the copy-heavy prompt did not "
                 "tokenize to one repeated id\n";
    return 1;
  }
  const std::vector<uint64_t> random_prompt = no_match_prompt(*l);

  const std::string base = "/tmp/sb-spec-bootstrap-" + std::to_string(getpid());
  std::error_code ec;
  std::filesystem::remove_all(base, ec);
  int failures = 0;
  failures +=
      run_in_child([&] { return run_ablation_case(*l, base + "-ablation"); });
  // Each trace case forks its own auto/off arms; it must not run inside
  // another child (the arms also compare their outputs in the parent).
  failures += run_trace_case(*l, base, copy_prompt, true);
  failures += run_trace_case(*l, base, random_prompt, false);
  for (const char* suffix : {"-ablation", "-copy-auto.tok", "-copy-off.tok",
                             "-copy-auto.meta", "-copy-auto-tune",
                             "-copy-off-tune", "-random-auto.tok",
                             "-random-off.tok", "-random-auto.meta",
                             "-random-auto-tune", "-random-off-tune"})
    std::filesystem::remove_all(base + suffix, ec);

  if (failures != 0) {
    std::cerr << "test_spec_bootstrap FAILED: " << failures << " check(s)\n";
    return 1;
  }
  std::cout << "test_spec_bootstrap OK\n";
  return 0;
}
