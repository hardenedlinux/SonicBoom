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

// B5 prompt-lookup drafter (CPU). Checks the n-gram lookup order (4, 3, 2),
// the most-recent-occurrence rule, the no-match case, max_drafts clipping and
// determinism, then measures the per-step cost (one push + one propose) against
// the < 2% of a decode step requirement.

#include <sonicboom/planner/drafter.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace sbplanner = sonicboom::planner;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

void expect(const sbplanner::NGramDrafter& d, std::span<const uint64_t> want,
            const char* what) {
  uint64_t out[16];
  const size_t n = d.propose(std::span<uint64_t>(out, sizeof(out) / sizeof(out[0])));
  bool ok = n == want.size();
  for (size_t i = 0; ok && i < n; ++i) ok = out[i] == want[i];
  if (!ok) {
    std::cerr << "  " << what << ": got [" << n << "]";
    for (size_t i = 0; i < n; ++i) std::cerr << " " << out[i];
    std::cerr << ", want [" << want.size() << "]";
    for (size_t i = 0; i < want.size(); ++i) std::cerr << " " << want[i];
    std::cerr << "\n";
  }
  check(ok, what);
}

} // namespace

int main() {
  using sbplanner::NGramDrafter;

  // A 3-gram hit: the history's suffix [1,2,3] last occurred ending at
  // position 2, and [4,5,1,2] followed it.
  {
    NGramDrafter d(4);
    d.push(std::vector<uint64_t>{1, 2, 3, 4, 5, 1, 2, 3});
    const uint64_t want[] = {4, 5, 1, 2};
    expect(d, want, "3-gram hit proposes the earlier continuation");
  }

  // No match: every suffix n-gram occurs only at the current end, so there is
  // no earlier occurrence to copy and the drafter proposes nothing.
  {
    NGramDrafter d(4);
    d.push(std::vector<uint64_t>{9, 8, 7});
    expect(d, std::span<const uint64_t>(), "no earlier occurrence proposes 0");
  }

  // 2-gram fallback: the 4- and 3-gram suffixes are new, but [8,9] occurred
  // earlier (positions 3..4), so the tokens after that occurrence are
  // proposed.
  {
    NGramDrafter d(3);
    d.push(std::vector<uint64_t>{5, 6, 7, 8, 9, 1, 2, 8, 9});
    const uint64_t want[] = {1, 2, 8};  // history[5..7]
    expect(d, want, "2-gram fallback proposes the earlier continuation");
  }

  // Most recent occurrence wins: [1,2,3] ended at positions 2 and 6; the later
  // occurrence's continuation [20,1,2] is the proposal, not [10,...].
  {
    NGramDrafter d(3);
    d.push(std::vector<uint64_t>{1, 2, 3, 10, 1, 2, 3, 20, 1, 2, 3});
    const uint64_t want[] = {20, 1, 2};
    expect(d, want, "most recent occurrence wins");
  }

  // max_drafts clips the proposal.
  {
    NGramDrafter d(2);
    d.push(std::vector<uint64_t>{1, 2, 3, 4, 5, 1, 2, 3});
    const uint64_t want[] = {4, 5};
    expect(d, want, "proposal clipped to max_drafts");
  }

  // A match proposes only the tokens that followed it in the history:
  // [1,2,3,4] occurred at positions 0..3 with five tokens after it, so asking
  // for eight returns those five and invents nothing.
  {
    NGramDrafter d(8);
    d.push(std::vector<uint64_t>{1, 2, 3, 4, 5, 1, 2, 3, 4});
    const uint64_t want[] = {5, 1, 2, 3, 4};
    expect(d, want, "the proposal is bounded by the matched run");
  }

  // S4's measured structural limit, on the degenerate stream the code-edit
  // trace collapses into (one token repeated): the most recent occurrence of
  // the 4-gram ends one token back, so the proposal is exactly one token even
  // though the repetition continues indefinitely. A chained proposal was tried
  // against this (and reverted): proposal length was not the binding
  // constraint — the verifier's per-row cost is (see the S4 report section).
  {
    NGramDrafter d(7);
    d.push(std::vector<uint64_t>{7, 7, 7, 7, 7, 7});
    uint64_t out[8];
    size_t n4 = 0;
    const size_t n = d.propose(std::span<uint64_t>(out, 7), &n4);
    check(n == 1 && n4 == 4, "a one-token cycle proposes one token (n=4)");
    check(out[0] == 7, "the proposed cycle token is the repeated one");
  }

  // Determinism: the same history gives the same proposal, and rebuilding the
  // drafter token by token or in one call gives the same state.
  {
    NGramDrafter a(4), b(4);
    const std::vector<uint64_t> h{3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
    a.push(h);
    for (uint64_t t : h) b.push(t);
    uint64_t oa[8], ob[8];
    const size_t na = a.propose(oa), nb = b.propose(ob);
    check(na == nb && std::equal(oa, oa + na, ob),
          "same history proposes the same tokens");
    a.reset();
    check(a.size() == 0 && a.propose(oa) == 0, "reset clears the history");
  }

  // Code-edit trace coverage (S1). The trace pastes 150-400 line repo source
  // files and asks for the file back, so its history is full of repeated
  // spans. Feeding that history one token at a time must give the drafter
  // something to propose: a seeded table that stays empty (the prefix-reuse
  // suffix bug) shows up here as a near-zero hit rate. The bytes are the
  // token ids — the real tokenizer is byte-level BPE and the drafter treats
  // ids as opaque, so this needs no model.
  {
#ifdef SONICBOOM_SRC_DIR
    static const char* kFiles[] = {
        "core/src/planner/cost_model.cpp",
        "core/src/planner/partition.cpp",
        "core/src/planner/device_pool.cpp",
        "core/src/planner/transfer_scheduler.cpp",
        "core/src/planner/native_torch_backend.cpp",
        "core/src/autograd/autograd.cpp",
    };
    size_t calls = 0, hits = 0, drafted = 0;
    for (const char* rel : kFiles) {
      std::ifstream in(std::string(SONICBOOM_SRC_DIR) + "/" + rel,
                       std::ios::binary);
      if (!in) continue;  // an out-of-tree source copy: the case is optional
      const std::string bytes((std::istreambuf_iterator<char>(in)),
                              std::istreambuf_iterator<char>());
      NGramDrafter d(7);
      uint64_t out[8];
      for (size_t i = 0; i < bytes.size(); ++i) {
        d.push(uint64_t(static_cast<unsigned char>(bytes[i])));
        if (i < 100) continue;  // the trace is warmed up after 100 tokens
        const size_t n = d.propose(out);
        ++calls;
        if (n > 0) {
          ++hits;
          drafted += n;
        }
      }
    }
    if (calls > 0) {
      std::printf("  code-edit history: %zu/%zu proposals after token 100 "
                  "(%.1f%%), mean %.2f tokens\n",
                  hits, calls, 100.0 * double(hits) / double(calls),
                  hits ? double(drafted) / double(hits) : 0.0);
      check(hits * 2 >= calls,
            "code-edit trace history proposes on >= 50% of calls after 100 "
            "tokens");
      check(drafted > hits,
            "code-edit trace history proposes runs longer than one token");
    }
#else
    std::cerr << "  (SONICBOOM_SRC_DIR undefined: code-edit history case "
                 "skipped)\n";
#endif
  }

  // Cost: one push (the token an accepted step emits) plus one propose per
  // speculative step. The requirement is < 2% of a decode step; the measured
  // decode steps in this tree are ~3 ms (Qwen) to ~19 ms (Gemma) on the RTX
  // 3050, so 2% is ~60-380 us. The bound asserted here is 50 us per step.
  {
    NGramDrafter d(7);
    // A 4096-token history over a 512-symbol alphabet: the maps are populated
    // and hits are frequent, the steady state the drafter runs in.
    std::vector<uint64_t> h(4096);
    uint64_t x = 99;
    for (auto& t : h) {
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      t = (x >> 33) % 512;
    }
    d.push(h);
    constexpr int kIters = 200000;
    uint64_t out[8];
    size_t hits = 0, proposed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kIters; ++i) {
      x = x * 6364136223846793005ull + 1442695040888963407ull;
      const uint64_t tok = (x >> 33) % 512;
      d.push(tok);
      const size_t n = d.propose(out);
      if (n > 0) {
        ++hits;
        proposed += n;
      }
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                      double(kIters);
    std::printf("  push+propose: %.3f us/step (%.0f%% hit, mean %.2f tokens)\n",
                us, 100.0 * double(hits) / kIters,
                hits ? double(proposed) / double(hits) : 0.0);
    check(us < 50.0, "drafter step cost < 50 us (< 2% of a decode step)");
  }

  if (g_failures == 0) {
    std::cout << "test_ngram_drafter OK\n";
    return 0;
  }
  std::cerr << "test_ngram_drafter FAILED: " << g_failures << " check(s)\n";
  return 1;
}
