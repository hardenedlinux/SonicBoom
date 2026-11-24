// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Unit test for the Output Decay metric over hand-made fixtures:
//   identical/  two prompt outputs mirrored byte for byte (one non-empty, one
//               empty)
//   diverging/  one output that diverges mid-text, one that ends early
//   empty/      an empty reference against text and text against an empty
//               comparison
//   mismatched/ the two directories name different prompts (an error)
//
// The test drives the same compare_directories() the tool's main() uses, so
// file pairing, byte reads and the overall figures are covered. It checks
// with explicit failures rather than assert(): test targets compile with
// assertions on, but the tool must also fail loudly if that ever changes.

#include "output_decay.h"

#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

namespace od = sonicboom::tools::od;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (ok) return;
  std::fprintf(stderr, "FAIL: %s\n", what);
  ++failures;
}

void near(double got, double want, const char* what) {
  if (std::fabs(got - want) < 1e-9) return;
  std::fprintf(stderr, "FAIL: %s: got %.6f, want %.6f\n", what, got, want);
  ++failures;
}

od::Summary run(const char* fixture, const char* ref, const char* cmp) {
  const std::string base = OD_FIXTURES_DIR;
  return od::compare_directories(base + "/" + fixture + "/" + ref,
                                 base + "/" + fixture + "/" + cmp);
}

void test_common_prefix() {
  check(od::common_prefix_length("", "") == 0, "empty prefixes");
  check(od::common_prefix_length("abc", "abc") == 3, "equal strings");
  check(od::common_prefix_length("abc", "abd") == 2, "differing tail");
  check(od::common_prefix_length("ab", "abcd") == 2, "prefix of a longer string");
  check(od::common_prefix_length("xy", "ab") == 0, "nothing shared");
}

void test_identical() {
  const od::Summary s = run("identical", "reference", "compared");
  check(s.prompts.size() == 2, "identical: two prompts");
  for (const od::PromptResult& p : s.prompts) {
    check(p.identical, "identical: outputs match");
    near(p.od_percent, 0.0, "identical: per-prompt OD");
  }
  near(s.od_mean_percent, 0.0, "identical: mean OD");
  near(s.od_pooled_percent, 0.0, "identical: pooled OD");
}

void test_diverging() {
  const od::Summary s = run("diverging", "reference", "compared");
  check(s.prompts.size() == 2, "diverging: two prompts");
  if (s.prompts.size() != 2) return;
  const od::PromptResult& mid = s.prompts[0];
  check(mid.name == "00.txt", "diverging: name order");
  check(mid.reference_bytes == 10, "diverging: reference bytes");
  check(mid.prefix_bytes == 5, "diverging: common prefix");
  check(!mid.identical && mid.first_divergence == 5,
        "diverging: first divergence");
  near(mid.od_percent, 50.0, "diverging: mid-text OD");

  const od::PromptResult& early = s.prompts[1];
  check(early.reference_bytes == 18, "diverging: truncated reference bytes");
  check(early.prefix_bytes == 9, "diverging: truncated prefix");
  check(!early.identical && early.first_divergence == 9,
        "diverging: truncation offset");
  near(early.od_percent, 50.0, "diverging: truncated OD");

  near(s.od_mean_percent, 50.0, "diverging: mean OD");
  near(s.od_pooled_percent, 50.0, "diverging: pooled OD");
}

void test_empty() {
  const od::Summary s = run("empty", "reference", "compared");
  check(s.prompts.size() == 2, "empty: two prompts");
  if (s.prompts.size() != 2) return;
  check(s.prompts[0].reference_bytes == 0, "empty: empty reference");
  near(s.prompts[0].od_percent, 100.0, "empty: text against empty reference");
  check(s.prompts[1].reference_bytes == 1, "empty: non-empty reference");
  near(s.prompts[1].od_percent, 100.0, "empty: empty against text");
  near(s.od_mean_percent, 100.0, "empty: mean OD");
  near(s.od_pooled_percent, 100.0, "empty: pooled OD");
}

void test_mismatched() {
  bool threw = false;
  try {
    run("mismatched", "reference", "compared");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "mismatched directories are an error");
}

} // namespace

int main() {
  test_common_prefix();
  test_identical();
  test_diverging();
  test_empty();
  test_mismatched();
  if (failures != 0) {
    std::fprintf(stderr, "test_output_decay: %d check(s) failed\n", failures);
    return 1;
  }
  std::printf("test_output_decay: all checks passed\n");
  return 0;
}
