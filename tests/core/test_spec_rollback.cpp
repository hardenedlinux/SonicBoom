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

// B4 speculative KV rollback (CUDA). A verify step writes KV for every query
// row it was fed, but only the rows of the accepted run remain valid: the rows
// after the first rejected draft were computed from tokens the verifier
// rejected. Those entries must never influence a later token, and the session's
// token history must record accepted tokens only, so prefix reuse cannot later
// match a KV row computed from a rejected draft.
//
// Each case forces a rejection at a chosen index: the drafts match the plain
// greedy continuation up to that index and then differ. From an identical
// prefill, one session runs the speculative step (verify + accept + rollback)
// and another runs the equivalent plain decode steps. Then both run the same 8
// continuation steps, which overwrite every dead row, and the test requires:
//   - the accepted count and emitted tokens are the greedy continuation;
//   - the session history is the accepted run only;
//   - the whole valid KV region (all layers, K and V, every valid position) is
//     bit-identical to the plain-decode session;
//   - a prefix-reuse prefill on a session whose history came from a speculative
//     step produces the same tokens as a full prefill of the same prompt.
//
// Runs for Gemma 4 E4B (sliding-window ring wraps at depth 600, shared-KV donor
// layers) and Qwen2.5 0.5B (global attention only), at depth 64 and 600, with
// forced rejections at 0, 3 and 7 of a k=7 draft. Every case runs in its own
// forked child (the same device-memory reason as test_spec_invariance).
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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

std::vector<uint64_t> synth_tokens(uint64_t vocab, size_t n) {
  std::vector<uint64_t> t(n);
  uint64_t x = 12345;
  for (size_t i = 0; i < n; ++i) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;  // PCG constants
    t[i] = ((x >> 33) % vocab);
  }
  return t;
}

// Compare every layer's K and V rows for positions [0, valid_len) bit-for-bit.
// Reads the device cache in blocks; on the first difference reports the layer,
// buffer, position and the two values.
bool compare_kv(const sbplanner::KvCacheSession& a,
                const sbplanner::KvCacheSession& b, uint64_t valid_len) {
  if (!a.caches || !b.caches || a.caches->size() != b.caches->size()) {
    std::cerr << "  KV compare: cache sets differ\n";
    return false;
  }
  constexpr uint64_t kBlock = 128;
  std::vector<float> va, vb;
  for (uint32_t l = 0; l < uint32_t(a.caches->size()); ++l) {
    const sbplanner::KvCacheDev& ca = (*a.caches)[l];
    const sbplanner::KvCacheDev& cb = (*b.caches)[l];
    const uint64_t row = ca.n_heads_kv * ca.head_dim;
    if (row == 0 || row != cb.n_heads_kv * cb.head_dim) {
      std::cerr << "  KV compare: layer " << l << " geometry differs\n";
      return false;
    }
    va.resize(size_t(kBlock * row));
    vb.resize(size_t(kBlock * row));
    for (bool is_k : {true, false}) {
      for (uint64_t first = 0; first < valid_len; first += kBlock) {
        const uint64_t n = std::min<uint64_t>(kBlock, valid_len - first);
        const size_t fl = size_t(n * row);
        const std::span<float> sa(va.data(), fl), sb(vb.data(), fl);
        if (!sbmodel::kv_read_rows_cuda(a, l, is_k, first, n, sa) ||
            !sbmodel::kv_read_rows_cuda(b, l, is_k, first, n, sb)) {
          std::cerr << "  KV compare: read failed (layer " << l
                    << ", is_k=" << is_k << ", first=" << first << ")\n";
          return false;
        }
        if (std::memcmp(va.data(), vb.data(), fl * sizeof(float)) == 0) continue;
        size_t i = 0;
        while (va[i] == vb[i]) ++i;
        const uint64_t pos = first + uint64_t(i / row);
        std::cerr << "  KV compare: layer " << l << (is_k ? " K" : " V")
                  << " pos " << pos << " elem " << (i % row) << ": spec "
                  << va[i] << " vs decode " << vb[i] << "\n";
        return false;
      }
    }
  }
  return true;
}

// One speculative step from a fresh prefill, compared against plain decode.
int run_kv_case(const sbmodel::Gemma4Model& m, const std::vector<uint64_t>& all,
                uint64_t depth, int reject_at) {
  g_failures = 0;
  constexpr int kDraft = 7;     // drafts per step (rows = kDraft + 1 = 8)
  constexpr int kCont = 24;     // greedy continuation probed ahead
  constexpr int kExtra = 8;     // plain steps after the spec step
  const uint64_t n_ctx = depth + 64;
  const uint64_t vocab = m.config.vocab_size;
  const std::span<const uint64_t> prompt(all.data(), size_t(depth));

  // Probe: the plain greedy continuation from the prompt.
  std::vector<int64_t> cont(kCont, -1);
  {
    sbplanner::KvCacheSession probe;
    probe.n_ctx = n_ctx;
    if (sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, cont, probe,
                                                    true) != kCont) {
      check(false, "probe prefill/generate failed");
      return g_failures;
    }
  }

  // A one-token generate budget prefills the prompt and samples exactly one
  // token: the KV holds positions 0..depth-1 and the history is the prompt.
  auto prefill_one = [&](sbplanner::KvCacheSession& s) -> int64_t {
    std::vector<int64_t> one(1, -1);
    if (sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, one, s,
                                                    true) != 1)
      return -1;
    return one[0];
  };

  sbplanner::KvCacheSession ref, spec;
  if (!sbmodel::kv_session_allocate(m, ref, n_ctx) ||
      !sbmodel::kv_session_allocate(m, spec, n_ctx)) {
    check(false, "session allocation failed");
    return g_failures;
  }
  check(prefill_one(ref) == cont[0], "reference prefill sampled the greedy token");
  check(prefill_one(spec) == cont[0], "spec prefill sampled the greedy token");
  check(ref.filled_tokens.size() == depth, "reference history after prefill");
  check(spec.filled_tokens.size() == depth, "spec history after prefill");

  // Drafts that match the greedy continuation up to `reject_at`, then differ.
  // At index reject_at the draft is (true + 1), so the row's argmax (the true
  // token) cannot match it and acceptance stops there exactly.
  std::vector<uint64_t> drafts(kDraft);
  for (int i = 0; i < kDraft; ++i) {
    drafts[size_t(i)] = uint64_t(cont[size_t(i + 1)]);
    if (i >= reject_at)
      drafts[size_t(i)] = (drafts[size_t(i)] + 1) % vocab;
  }

  // Speculative step on `spec`.
  std::vector<int64_t> emitted(kDraft + 1, -1);
  uint64_t accepted = ~uint64_t(0);
  check(sbmodel::spec_step_spine_cuda(m, uint64_t(cont[0]), drafts, spec,
                                      emitted, &accepted),
        "speculative step failed");
  check(accepted == uint64_t(reject_at), "accepted count is the forced index");
  for (int i = 0; i <= reject_at; ++i)
    check(emitted[size_t(i)] == cont[size_t(i + 1)],
          "emitted token is the greedy continuation");

  // History: the prompt, then the fed last token, then the accepted drafts.
  const uint64_t want_hist = depth + uint64_t(reject_at) + 1;
  check(spec.filled_tokens.size() == want_hist,
        "history length is the accepted run");
  bool hist_ok = spec.filled_tokens.size() == want_hist;
  for (uint64_t i = 0; hist_ok && i < depth; ++i)
    hist_ok = spec.filled_tokens[size_t(i)] == all[size_t(i)];
  if (hist_ok)
    hist_ok = spec.filled_tokens[size_t(depth)] == uint64_t(cont[0]);
  for (int i = 0; hist_ok && i < reject_at; ++i)
    hist_ok = spec.filled_tokens[size_t(depth + 1 + uint64_t(i))] ==
              uint64_t(cont[size_t(i + 1)]);
  check(hist_ok, "history is the accepted run only");

  // Reference: plain decode of the same tokens (last token + accepted drafts),
  // then both sessions take the same continuation steps. The continuation
  // overwrites every dead row (positions depth+reject_at+1 .. depth+kDraft).
  std::vector<uint64_t> fed(cont.begin(), cont.begin() + reject_at + 1);
  std::vector<int64_t> ref_out(fed.size(), -1);
  check(sbmodel::decode_steps_spine_cuda(m, fed, depth, ref, ref_out),
        "reference decode steps failed");
  for (int i = 0; i <= reject_at; ++i)
    check(ref_out[size_t(i)] == cont[size_t(i + 1)],
          "reference step is the greedy continuation");
  std::vector<uint64_t> next(cont.begin() + reject_at + 1,
                             cont.begin() + reject_at + 1 + kExtra);
  std::vector<int64_t> ref_next(next.size(), -1), spec_next(next.size(), -1);
  check(sbmodel::decode_steps_spine_cuda(m, next, depth + reject_at + 1, ref,
                                         ref_next),
        "reference continuation failed");
  check(sbmodel::decode_steps_spine_cuda(m, next, depth + reject_at + 1, spec,
                                         spec_next),
        "spec continuation failed");
  check(ref_next == spec_next, "continuation tokens identical");
  for (size_t i = 0; i < next.size(); ++i)
    check(spec_next[i] == cont[size_t(reject_at + 2) + i],
          "spec continuation is the greedy continuation");

  // The whole valid KV region must be the plain decode's, bit for bit. Dead
  // rows are past it; every dead row has by now been overwritten in order.
  const uint64_t valid = depth + uint64_t(reject_at) + 1 + kExtra;
  if (!compare_kv(spec, ref, valid))
    check(false, "valid KV bitwise equal to plain decode");
  else
    std::cout << "  valid KV identical over " << valid << " positions ("
              << spec.caches->size() << " layers, K and V)\n";
  return g_failures;
}

// Prefix reuse from a session whose history came out of a speculative step.
int run_prefix_case(const sbmodel::Gemma4Model& m,
                    const std::vector<uint64_t>& all, uint64_t depth) {
  g_failures = 0;
  constexpr int kDraft = 7;
  const uint64_t n_ctx = depth + 64;
  const uint64_t vocab = m.config.vocab_size;
  const std::span<const uint64_t> prompt(all.data(), size_t(depth));

  sbplanner::KvCacheSession spec;
  if (!sbmodel::kv_session_allocate(m, spec, n_ctx)) {
    check(false, "session allocation failed");
    return g_failures;
  }
  std::vector<int64_t> one(1, -1);
  if (sbmodel::generate_prefill_spine_cuda_prefix(m, prompt, 0, one, spec,
                                                  true) != 1) {
    check(false, "spec prefill failed");
    return g_failures;
  }

  // Arbitrary draft tokens: the acceptance count is whatever the verifier says
  // (typically 0), and the counterfactual rows the verify writes past it are
  // exactly the dead rows this case is about. Deterministic ids keep the case
  // reproducible across runs.
  std::vector<uint64_t> drafts(kDraft);
  for (int i = 0; i < kDraft; ++i)
    drafts[size_t(i)] = (uint64_t(one[0]) + 97 * uint64_t(i + 1)) % vocab;
  std::vector<int64_t> emitted(kDraft + 1, -1);
  uint64_t accepted = 0;
  check(sbmodel::spec_step_spine_cuda(m, uint64_t(one[0]), drafts, spec,
                                      emitted, &accepted),
        "speculative step failed");

  // A speculative step leaves history == KV valid positions (accepted run
  // only); prefill four new tokens after it, in a fresh full prefill and by
  // reuse from the spec session.
  std::vector<uint64_t> prompt2 = spec.filled_tokens;
  const uint64_t hist = prompt2.size();
  for (int i = 0; i < 4; ++i)
    prompt2.push_back((uint64_t(one[0]) + 13 * uint64_t(i + 1)) % vocab);

  sbplanner::KvCacheSession full;
  full.n_ctx = n_ctx;
  std::vector<int64_t> full_toks(3, -1);
  check(sbmodel::generate_prefill_spine_cuda_prefix(m, prompt2, 0, full_toks,
                                                    full, true) == 3,
        "full prefill of the same prompt failed");
  std::vector<int64_t> reuse_toks(3, -1);
  uint64_t reused = 0;
  check(sbmodel::generate_prefill_spine_cuda_prefix(m, prompt2, 0, reuse_toks,
                                                    spec, true, &reused) == 3,
        "prefix-reuse prefill failed");
  check(reused > 0, "a prefix was reused from the speculative session");
  check(reused <= hist, "reuse stays inside the accepted KV");
  if (full_toks == reuse_toks)
    std::cout << "  prefix reuse after spec (hist " << hist << ", reused "
              << reused << ") == full prefill\n";
  check(full_toks == reuse_toks,
        "prefix reuse after a speculative session == full prefill");
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

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  const std::string gemma_path =
      find_file("SONICBOOM_GEMMA_GGUF", "models/gemma-4-E4B-it-Q3_K_M.gguf");
  const std::string qwen_path = find_file(
      "SONICBOOM_QWEN_GGUF", "models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  if (gemma_path.empty() || qwen_path.empty()) {
    std::cout << "  (skipped: model GGUF not found)\n";
    std::cout << "test_spec_rollback OK\n";
    return 0;
  }

  auto r_gemma = sbgguf::Reader::load(gemma_path);
  auto r_qwen = sbgguf::Reader::load(qwen_path);
  if (!r_gemma || !r_qwen) {
    std::cerr << "test_spec_rollback FAILED: GGUF load\n";
    return 1;
  }
  auto lg = sbmodel::load_gemma4(*r_gemma);
  auto lq = sbmodel::load_gemma4(*r_qwen);
  if (!lg || !lq) {
    std::cerr << "  load error: " << (lg ? lq.error() : lg.error()) << "\n";
    std::cerr << "test_spec_rollback FAILED: model load\n";
    return 1;
  }
  const sbmodel::Gemma4Model gemma = std::move(*lg);
  const sbmodel::Gemma4Model qwen = std::move(*lq);

  if (run_in_child([] { return sbquant::cuda_available() ? 0 : 2; }) == 2) {
    std::cout << "  (skipped: no CUDA device)\n";
    std::cout << "test_spec_rollback OK\n";
    return 0;
  }

  int failures = 0;
  // Test-harness filters (not runtime switches): one model, one depth, one
  // rejection index, and the prefix sub-test only.
  const char* only_model = std::getenv("SONICBOOM_SPEC_ROLLBACK_MODEL");
  const char* only_depth = std::getenv("SONICBOOM_SPEC_ROLLBACK_ONLY");
  const char* only_reject = std::getenv("SONICBOOM_SPEC_ROLLBACK_REJECT");
  const bool prefix_only =
      std::getenv("SONICBOOM_SPEC_ROLLBACK_PREFIX") != nullptr;
  for (const auto* model : {&gemma, &qwen}) {
    const char* name = model->config.arch == sbmodel::ModelArch::Gemma4
                           ? "gemma4"
                           : "qwen2";
    if (only_model && std::string(only_model) != name) continue;
    const std::vector<uint64_t> all =
        synth_tokens(model->config.vocab_size, 600 + 8);
    for (uint64_t depth : {uint64_t(64), uint64_t(600)}) {
      if (only_depth && uint64_t(std::atoi(only_depth)) != depth) continue;
      for (int reject : {0, 3, 7}) {
        if (only_reject && std::atoi(only_reject) != reject) continue;
        if (!prefix_only) {
          std::cout << "model=" << name << " depth=" << depth << " reject_at="
                    << reject << " (KV)\n";
          failures += run_in_child(
              [&] { return run_kv_case(*model, all, depth, reject); });
        }
        std::cout << "model=" << name << " depth=" << depth << " (prefix)\n";
        failures +=
            run_in_child([&] { return run_prefix_case(*model, all, depth); });
        if (prefix_only) break;
      }
    }
  }

  if (failures == 0) {
    std::cout << "test_spec_rollback OK\n";
    return 0;
  }
  std::cerr << "test_spec_rollback FAILED: " << failures << " check(s)\n";
  return 1;
}
