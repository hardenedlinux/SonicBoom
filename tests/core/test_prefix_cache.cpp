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
// 3996 / 4000, each +100 new), the segment-boundary set 4095 / 4096 / 4097 /
// 4098 and 8191 / 8192 / 8193 (the sliding-window chunk is 512, so these are
// the positions around 8x and 16x the chunk boundary; 4097 and 8193 also leave
// a one-token final seed segment), plus a +600 new case whose suffix exceeds
// the sliding window (single segment over the SWA ring) and a fully-reused case
// (+0 new: the prompt is already resident, so it is reused up to len-1 and the
// last token is re-prefilled).
//
// Every case runs in its own forked child: on an 8 GiB device the device
// allocations retained from earlier cases push a later case's peak over the
// limit (measured: the 8191 full prefill fails to instantiate its CUDA graph
// with "out of memory" in one combined run, and the 4098 setup prefill fails
// with "device buffer allocation failed" after the preceding cases), while the
// same cases fit when the process starts fresh. The 8K-context cases are also
// a separate enumeration (`--long`, ctest test_prefix_cache_long) so a default
// run stays inside a sane wall-clock budget.
//
// Loads the Gemma 4 model and the token fixture (skips cleanly when either is
// absent) and gates on a real CUDA device via quant::cuda_available().

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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
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

struct Case { uint64_t prefix; uint64_t new_n; };

// One case: a full prefill (fresh session) and a prefix-reuse prefill of the
// same tokens, compared token-for-token. Returns this case's failure count.
int run_case(const sbmodel::Gemma4Model& m, const std::vector<uint64_t>& all,
             const Case& c, uint64_t steps) {
  g_failures = 0;
  const uint64_t need = c.prefix + c.new_n;
  if (need > all.size()) {
    check(false, "tokens fixture too short");
    return g_failures;
  }
  const std::vector<uint64_t> prefix(all.begin(), all.begin() + c.prefix);
  const std::vector<uint64_t> full(all.begin(), all.begin() + need);

  // Full prefill (fresh session -> no reuse).
  std::vector<int64_t> full_toks(steps, -1);
  const uint64_t full_done = sbmodel::generate_prefill_spine_cuda(
      m, full, 0, full_toks, full.size() + steps);
  full_toks.resize(full_done);

  // Prefix reuse: seed the cache with the prefix, then reuse for `full`.
  sbplanner::KvCacheSession session;
  session.n_ctx = full.size() + steps;
  std::vector<int64_t> setup_toks(1, -1);
  if (sbmodel::generate_prefill_spine_cuda_prefix(m, prefix, 0, setup_toks,
                                                  session, true) == 0) {
    check(false, "prefix setup prefill failed");
    return g_failures;
  }
  std::vector<int64_t> reuse_toks(steps, -1);
  uint64_t reused = 0;
  const uint64_t reuse_done = sbmodel::generate_prefill_spine_cuda_prefix(
      m, full, 0, reuse_toks, session, true, &reused);
  reuse_toks.resize(reuse_done);

  check(full_done == reuse_done, "same step count as full prefill");
  // The reuse boundary is the shared prefix, minus one token when the prompt
  // is fully reused (its activation is needed for the next-token logits) and
  // minus one more when that boundary falls one token past a chunk boundary:
  // a seeded prefix ending there was computed as a one-token segment, while
  // the full prefill computes the same position in a multi-token segment
  // (see the prefix-align note in gemma4_plan.cpp). With
  // SONICBOOM_PREFIX_ALIGN=0 the plan keeps the literal boundary, so this
  // check must follow the same switch (the prefix-alignment repro depends on it).
  const uint64_t chunk =
      m.config.sliding_window > 0 ? m.config.sliding_window : 512;
  const char *align_env = std::getenv("SONICBOOM_PREFIX_ALIGN");
  const bool align =
      !(align_env != nullptr && std::string_view(align_env) == "0");
  uint64_t want_reused = (c.new_n == 0) ? c.prefix - 1 : c.prefix;
  if (align && want_reused > 0) {
    // Mirror the production holdback (see generate_prefill_spine_cuda_prefix):
    // a segment shorter than the mmq threshold runs the single-column kernels
    // and does not reproduce a multi-token segment's KV, so the boundary is
    // moved off any such segment, on either side of the reuse point.
    const uint64_t min_n =
        std::max<uint64_t>(uint64_t(sbquant::cuda::mmq_min_n()), 1);
    const uint64_t rem = want_reused % chunk;
    if (rem != 0 && rem < min_n)
      want_reused -= rem;
    else if (rem != 0 && chunk - rem < min_n)
      want_reused -= min_n - (chunk - rem);
  }
  check(reused == want_reused,
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
  return g_failures;
}

// Run fn in a forked child and return its exit status. Each case gets a fresh
// CUDA context and address space: the device allocations retained from earlier
// cases (device pools sized against free memory, captured segment graphs) push
// the later cases' peak over an 8 GiB device's limit — measured: the 4098 case
// fails its setup prefill with "device buffer allocation failed" after the
// preceding cases, and passes on its own. The child inherits the loaded model
// copy-on-write, so this costs no reload. The parent must not touch CUDA
// before forking (a CUDA context is not inherited across fork), which is why
// the availability probe is itself a child.
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
  const bool long_only = argc > 1 && std::string(argv[1]) == "--long";
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

  // CUDA availability probe, in a child (see run_in_child).
  if (run_in_child([] { return sbquant::cuda_available() ? 0 : 2; }) == 2) {
    std::cout << "  (skipped: no CUDA device)\n";
    std::cout << "test_prefix_cache OK\n";
    return 0;
  }

  const uint64_t steps = env_u64("SONICBOOM_PREFIX_CACHE_STEPS", 32);

  // new_n == 0 is the fully-reused case: the prompt is what the session already
  // holds, so the reuse is capped at len-1 (the last token is re-prefilled to
  // produce the next-token logits) rather than leaving an empty suffix.
  const Case kCases[] = {
      {512, 100},  {600, 100},   {1000, 100}, {1040, 100},
      {3584, 100}, {3600, 100},  {3996, 100}, {4000, 100},
      {4095, 100}, {4096, 100},  {4097, 100}, {4098, 100},
      {4000, 600}, {512, 0},
  };
  const Case kLongCases[] = {{8191, 100}, {8192, 100}, {8193, 100}};
  const Case* cases = long_only ? kLongCases : kCases;
  const size_t ncases = long_only ? sizeof(kLongCases) / sizeof(kLongCases[0])
                                  : sizeof(kCases) / sizeof(kCases[0]);

  // SONICBOOM_PREFIX_CACHE_ONLY=8191[,8192...] runs a subset (debug repro).
  std::vector<uint64_t> only;
  if (const char* e = std::getenv("SONICBOOM_PREFIX_CACHE_ONLY")) {
    const char* p = e;
    while (*p) {
      char* end = nullptr;
      const unsigned long long v = std::strtoull(p, &end, 10);
      if (end == p) break;
      only.push_back(v);
      p = (*end == ',') ? end + 1 : end;
    }
  }

  int failures = 0;
  for (size_t ci = 0; ci < ncases; ++ci) {
    const Case c = cases[ci];
    if (!only.empty() &&
        std::find(only.begin(), only.end(), c.prefix) == only.end())
      continue;
    failures += run_in_child([&] { return run_case(*m, all, c, steps); });
  }

  if (failures == 0) {
    std::cout << "test_prefix_cache OK\n";
    return 0;
  }
  std::cerr << "test_prefix_cache FAILED: " << failures << " check(s)\n";
  return 1;
}
