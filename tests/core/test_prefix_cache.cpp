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

// C.4 prefix-cache KV reuse acceptance (CUDA). For a real-text prompt (encoded
// by tools/baseline/make_real_prompt.sh -> tools/baseline/real_prompt.tokens),
// reusing the shared prefix's KV slots and prefilling only the new suffix must
// reproduce the full prefill's greedy continuation token-for-token. The matrix
// covers the C.4 acceptance prefixes (512 / 600 / 1000 / 1040 / 3584 / 3600 /
// 3996 / 4000 / 4095 / 4097, each +100 new) plus a +600 new case whose suffix
// exceeds the sliding window (single segment over the SWA ring) and a
// fully-reused case (+0 new: the prompt is already resident, so it is reused up
// to len-1 and the last token is re-prefilled).
//
// Loads the Gemma 4 model and the token fixture (skips cleanly when either is
// absent) and gates on a real CUDA device via quant::cuda_available().

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

std::string find_model() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF"))
    candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/gemma-4-E4B-it-Q3_K_M.gguf");
#endif
  candidates.push_back("models/gemma-4-E4B-it-Q3_K_M.gguf");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

std::string find_tokens_file() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_PREFIX_CACHE_TOKENS"))
    candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/tools/baseline/real_prompt.tokens");
#endif
  candidates.push_back("real_prompt.tokens");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

uint64_t env_u64(const char* name, uint64_t dflt) {
  const char* s = std::getenv(name);
  if (!s || !*s) return dflt;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) return dflt;
  return static_cast<uint64_t>(v);
}

std::vector<uint64_t> load_tokens(const std::string& path) {
  std::ifstream in(path);
  std::vector<uint64_t> t;
  uint64_t tok;
  while (in >> tok) t.push_back(tok);
  return t;
}

} // namespace

int main() {
  const std::string path = find_model();
  if (path.empty()) {
    std::cout << "  (skipped: Gemma 4 model not found)\n";
    std::cout << "test_prefix_cache OK\n";
    return 0;
  }
  const std::string tokens_file = find_tokens_file();
  if (tokens_file.empty()) {
    std::cout << "  (skipped: real_prompt.tokens fixture not found)\n";
    std::cout << "test_prefix_cache OK\n";
    return 0;
  }
  if (!sbquant::cuda_available()) {
    std::cout << "  (skipped: no CUDA device)\n";
    std::cout << "test_prefix_cache OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    std::cerr << "  failed to load GGUF\n";
    std::cerr << "test_prefix_cache FAILED: 1 check(s)\n";
    return 1;
  }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) {
    std::cerr << "  load_gemma4 error: " << m.error() << "\n";
    std::cerr << "test_prefix_cache FAILED: 1 check(s)\n";
    return 1;
  }

  const std::vector<uint64_t> all = load_tokens(tokens_file);
  if (all.empty()) {
    std::cerr << "  tokens fixture empty: " << tokens_file << "\n";
    std::cerr << "test_prefix_cache FAILED: 1 check(s)\n";
    return 1;
  }

  const uint64_t steps = env_u64("SONICBOOM_PREFIX_CACHE_STEPS", 32);

  // new_n == 0 is the fully-reused case: the prompt is what the session already
  // holds, so the reuse is capped at len-1 (the last token is re-prefilled to
  // produce the next-token logits) rather than leaving an empty suffix.
  struct Case { uint64_t prefix; uint64_t new_n; };
  const Case kCases[] = {
      {512, 100}, {600, 100},   {1000, 100}, {1040, 100},
      {3584, 100}, {3600, 100}, {3996, 100}, {4000, 100},
      {4095, 100}, {4097, 100}, {4000, 600},  {512, 0},
  };

  for (const Case& c : kCases) {
    const uint64_t need = c.prefix + c.new_n;
    if (need > all.size()) {
      check(false, "tokens fixture too short");
      continue;
    }
    const std::vector<uint64_t> prefix(all.begin(),
                                       all.begin() + c.prefix);
    const std::vector<uint64_t> full(all.begin(), all.begin() + need);

    // Full prefill (fresh session -> no reuse).
    std::vector<int64_t> full_toks(steps, -1);
    const uint64_t full_done = sbmodel::generate_prefill_spine_cuda(
        *m, full, 0, full_toks, full.size() + steps);
    full_toks.resize(full_done);

    // Prefix reuse: seed the cache with the prefix, then reuse for `full`.
    sbplanner::KvCacheSession session;
    session.n_ctx = full.size() + steps;
    std::vector<int64_t> setup_toks(1, -1);
    if (sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, setup_toks,
                                                    session, true) == 0) {
      check(false, "prefix setup prefill failed");
      continue;
    }
    std::vector<int64_t> reuse_toks(steps, -1);
    uint64_t reused = 0;
    const uint64_t reuse_done = sbmodel::generate_prefill_spine_cuda_prefix(
        *m, full, 0, reuse_toks, session, true, &reused);
    reuse_toks.resize(reuse_done);

    check(full_done == reuse_done, "same step count as full prefill");
    check(reused == (c.new_n == 0 ? c.prefix - 1 : c.prefix),
          "reuse count is the shared prefix (len-1 when fully reused)");

    if (full_toks == reuse_toks) {
      std::cout << "  prefix=" << c.prefix << " new=" << c.new_n
                << ": reuse identical to full prefill (" << full_toks.size()
                << " tokens)\n";
      check(true, "reuse continuation identical to full prefill");
    } else {
      size_t div = 0;
      while (div < full_toks.size() && div < reuse_toks.size() &&
             full_toks[div] == reuse_toks[div])
        ++div;
      std::cout << "  prefix=" << c.prefix << " new=" << c.new_n
                << ": DIVERGE at step " << div << " (full "
                << (div < full_toks.size() ? full_toks[div] : -1) << " vs reuse "
                << (div < reuse_toks.size() ? reuse_toks[div] : -1) << ")\n";
      check(false, "reuse continuation identical to full prefill");
    }
  }

  if (g_failures == 0) {
    std::cout << "test_prefix_cache OK\n";
    return 0;
  }
  std::cerr << "test_prefix_cache FAILED: " << g_failures << " check(s)\n";
  return 1;
}
