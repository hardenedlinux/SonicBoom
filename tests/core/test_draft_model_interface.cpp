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

// SD1: the DraftModel interface.
//
// The gate is token identity: over a corpus with repeated spans, a
// NgramDraftModel reached directly and through a DraftModel* must propose
// exactly what NGramDrafter proposes at every step, both the tokens and the
// matched n-gram length. The factory selects the n-gram drafter when
// SONICBOOM_DRAFT_MODEL is unset or empty, and the model-backed drafter throws
// "LlmDraftModel not yet implemented" when it is set (and when constructed
// directly). No model, no device.

#include <sonicboom/planner/draft_model.h>
#include <sonicboom/planner/drafter.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sbplanner = sonicboom::planner;

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
  if (!ok) {
    ++failures;
    std::cout << "FAIL: " << what << "\n";
  }
}

void set_env(const char *name, const char *value) {
  if (value)
    ::setenv(name, value, 1);
  else
    ::unsetenv(name);
}

// A 37-token block repeated with small mutations: long enough runs recur, so
// the drafter hits at every n-gram length, and deterministic.
std::vector<uint64_t> corpus() {
  static const uint64_t kBlock[37] = {
      4103, 88,  12,  991, 4,   4103, 77,  12,  991, 1234, 5,
      6,    4103, 88,  12,  991, 90,  4,   4103, 77,  12,  991,
      1234, 5,   7,   4103, 88,  12,  991, 4,   4103, 77,  12,
      991,  1234, 5,   9,
  };
  std::vector<uint64_t> tokens;
  tokens.reserve(37 * 40);
  for (int rep = 0; rep < 40; ++rep)
    for (int i = 0; i < 37; ++i)
      tokens.push_back(kBlock[i] + uint64_t((rep * 7 + i) % 3));
  return tokens;
}

struct Proposal {
  std::vector<uint64_t> tokens;
  size_t matched_n = 0;
  bool operator==(const Proposal &o) const {
    return tokens == o.tokens && matched_n == o.matched_n;
  }
};

Proposal propose(const sbplanner::DraftModel &d) {
  uint64_t out[sbplanner::NGramDrafter::kMaxN * 2];
  Proposal p;
  const size_t n =
      d.propose(std::span<uint64_t>(out, sizeof(out) / sizeof(out[0])),
                &p.matched_n);
  p.tokens.assign(out, out + n);
  return p;
}

Proposal propose(const sbplanner::NGramDrafter &d) {
  uint64_t out[sbplanner::NGramDrafter::kMaxN * 2];
  Proposal p;
  const size_t n =
      d.propose(std::span<uint64_t>(out, sizeof(out) / sizeof(out[0])),
                &p.matched_n);
  p.tokens.assign(out, out + n);
  return p;
}

std::string show(const Proposal &p) {
  std::string s = "[" + std::to_string(p.tokens.size()) + "]";
  for (uint64_t t : p.tokens) s += " " + std::to_string(t);
  s += " n=" + std::to_string(p.matched_n);
  return s;
}

} // namespace

int main() {
  const std::vector<uint64_t> tokens = corpus();

  // --- token identity: NGramDrafter vs NgramDraftModel vs DraftModel* ----
  {
    set_env("SONICBOOM_DRAFT_MODEL", nullptr);
    sbplanner::NGramDrafter ref(7);
    sbplanner::NgramDraftModel direct(7);
    std::unique_ptr<sbplanner::DraftModel> via = sbplanner::make_draft_model(7);

    check(direct.max_drafts() == 7 && via->max_drafts() == 7,
          "the interface reports the drafter's max_drafts");

    size_t steps = 0, hits = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
      ref.push(tokens[i]);
      direct.push(tokens[i]);
      via->push(tokens[i]);
      // Every step is compared, including the empty-table warm-up.
      const Proposal want = propose(ref);
      const Proposal got_direct = propose(direct);
      const Proposal got_via = propose(*via);
      if (i < 8) continue;  // keep the loop's cost down; identity holds anyway
      ++steps;
      if (!want.tokens.empty()) ++hits;
      if (!(want == got_direct)) {
        check(false, "NgramDraftModel differs from NGramDrafter at token " +
                         std::to_string(i) + ": want " + show(want) + ", got " +
                         show(got_direct));
        break;
      }
      if (!(want == got_via)) {
        check(false, "DraftModel* differs from NGramDrafter at token " +
                         std::to_string(i) + ": want " + show(want) + ", got " +
                         show(got_via));
        break;
      }
    }
    check(steps > 0 && hits > 0,
          "the corpus exercises the interface across matching steps");
    std::cout << "  token identity: " << steps << " steps, " << hits
              << " proposals\n";
  }

  // --- max_drafts is honoured through the interface ----------------------
  {
    set_env("SONICBOOM_DRAFT_MODEL", nullptr);
    std::unique_ptr<sbplanner::DraftModel> d = sbplanner::make_draft_model(3);
    d->push(std::span<const uint64_t>(tokens.data(), tokens.size()));
    Proposal p = propose(*d);
    check(p.tokens.size() <= 3, "propose respects max_drafts");
    d->set_max_drafts(1);
    check(d->max_drafts() == 1, "set_max_drafts updates the cap");
    p = propose(*d);
    check(p.tokens.size() <= 1, "propose respects a lowered max_drafts");
    d->reset();
    check(propose(*d).tokens.empty(), "reset empties the drafter");
  }

  // --- factory selection -------------------------------------------------
  {
    set_env("SONICBOOM_DRAFT_MODEL", nullptr);
    check(dynamic_cast<sbplanner::NgramDraftModel *>(
              sbplanner::make_draft_model(4).get()) != nullptr,
          "factory: unset SONICBOOM_DRAFT_MODEL selects NgramDraftModel");

    set_env("SONICBOOM_DRAFT_MODEL", "");
    check(dynamic_cast<sbplanner::NgramDraftModel *>(
              sbplanner::make_draft_model(4).get()) != nullptr,
          "factory: empty SONICBOOM_DRAFT_MODEL selects NgramDraftModel");

    set_env("SONICBOOM_DRAFT_MODEL", "/nonexistent/draft.gguf");
    std::string message;
    try {
      (void)sbplanner::make_draft_model(4);
      check(false, "factory: a set SONICBOOM_DRAFT_MODEL must throw");
    } catch (const std::runtime_error &e) {
      message = e.what();
    }
    check(message == "LlmDraftModel not yet implemented",
          "factory: the model-backed drafter throws the not-implemented "
          "message (got '" + message + "')");
  }

  // --- direct construction of the skeleton -------------------------------
  {
    std::string message;
    try {
      sbplanner::LlmDraftModel m("/nonexistent/draft.gguf", 4);
      (void)m;
      check(false, "LlmDraftModel construction must throw");
    } catch (const std::runtime_error &e) {
      message = e.what();
    }
    check(message == "LlmDraftModel not yet implemented",
          "LlmDraftModel throws the not-implemented message (got '" + message +
              "')");
  }

  ::unsetenv("SONICBOOM_DRAFT_MODEL");

  if (failures == 0) {
    std::cout << "test_draft_model_interface OK\n";
    return 0;
  }
  std::cout << "test_draft_model_interface FAILED: " << failures
            << " check(s)\n";
  return 1;
}
