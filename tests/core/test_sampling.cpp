// Tests for sonicboom::nn sampling: the Philox4x32-10 known answer, greedy
// equivalence, the exact candidate sets of the top-k / min-p / top-p filters,
// the penalty formula, determinism by (seed, position) and the sampled
// distribution against the softmax.

#include <sonicboom/nn/sampling.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace sbnn = sonicboom::nn;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool same_set(std::vector<int64_t> got, std::vector<int64_t> want) {
  std::sort(got.begin(), got.end());
  std::sort(want.begin(), want.end());
  if (got == want) return true;
  std::cerr << "    got {";
  for (size_t i = 0; i < got.size(); ++i)
    std::cerr << (i ? "," : "") << got[i];
  std::cerr << "} want {";
  for (size_t i = 0; i < want.size(); ++i)
    std::cerr << (i ? "," : "") << want[i];
  std::cerr << "}\n";
  return false;
}

void test_philox_kat() {
  // Random123 paper, philox4x32-10, zero counter and key.
  const auto out = sbnn::philox4x32_10({0, 0, 0, 0}, 0);
  check(out[0] == 0x6627e8d5u, "philox kat word 0");
  check(out[1] == 0xe169c58du, "philox kat word 1");
  check(out[2] == 0xbc57ac4cu, "philox kat word 2");
  check(out[3] == 0x9b00dbd8u, "philox kat word 3");
}

void test_greedy() {
  const std::vector<float> logits = {0.2f, 0.2f, 0.9f, 0.1f};
  sbnn::SamplingParams p;  // temperature 0
  check(sbnn::sample_token(logits, p, 0, {}, {}) == 2,
        "temperature 0 equals argmax");
  const std::vector<float> tie = {0.5f, 0.5f, 0.1f};
  check(sbnn::sample_token(tie, p, 0, {}, {}) == 0,
        "temperature 0 tie goes to the lowest id");
}

void test_top_k_one_is_argmax() {
  const std::vector<float> logits = {0.5f, 2.5f, 1.5f, -1.0f};
  sbnn::SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 1;
  for (uint64_t pos = 0; pos < 32; ++pos)
    check(sbnn::sample_token(logits, p, pos, {}, {}) == 1,
          "top_k 1 equals argmax");
}

void test_top_k_set() {
  const std::vector<float> logits = {3.0f, 2.0f, 1.0f, 0.0f};
  sbnn::SamplingParams p;
  p.temperature = 1.0f;
  p.top_k = 2;
  check(same_set(sbnn::sample_candidates(logits, p, {}, {}), {0, 1}),
        "top_k keeps the two largest");
  // Ties order by id ascending.
  const std::vector<float> tie = {0.0f, 0.0f, 0.0f, 0.0f};
  p.top_k = 2;
  check(same_set(sbnn::sample_candidates(tie, p, {}, {}), {0, 1}),
        "top_k tie-break by id");
  // The mask is applied before the filters.
  std::vector<uint8_t> mask = {0x06};  // ids 1 and 2
  p.top_k = 1;
  check(same_set(sbnn::sample_candidates(tie, p, {}, mask), {1}),
        "top_k over masked candidates");
}

void test_min_p_set() {
  // z = {0, -1, -3}; min_p 0.1 excludes only -3 (ln 0.1 = -2.3026).
  const std::vector<float> logits = {0.0f, -1.0f, -3.0f};
  sbnn::SamplingParams p;
  p.temperature = 1.0f;
  p.min_p = 0.1f;
  check(same_set(sbnn::sample_candidates(logits, p, {}, {}), {0, 1}),
        "min_p keeps the candidates at or above max + ln(min_p)");
  p.min_p = 0.5f;  // cutoff ln 0.5 = -0.693
  check(same_set(sbnn::sample_candidates(logits, p, {}, {}), {0}),
        "min_p 0.5 keeps only the max");
}

void test_top_p_set() {
  // Four equal logits: nucleus order is ids 0,1,2,3.
  const std::vector<float> logits = {0.0f, 0.0f, 0.0f, 0.0f};
  sbnn::SamplingParams p;
  p.temperature = 1.0f;
  p.top_p = 0.6f;
  check(same_set(sbnn::sample_candidates(logits, p, {}, {}), {0, 1, 2}),
        "top_p keeps the shortest prefix reaching it");
  p.top_p = 0.5f;
  check(same_set(sbnn::sample_candidates(logits, p, {}, {}), {0, 1}),
        "top_p exactly reached at a prefix boundary");
  p.top_p = 0.24f;
  check(same_set(sbnn::sample_candidates(logits, p, {}, {}), {0}),
        "top_p keeps at least one candidate");
}

void test_penalties() {
  // repeat_penalty: 1.5 -> 0.75, so the untouched 1.0 wins.
  {
    const std::vector<float> logits = {1.5f, 1.0f};
    sbnn::SamplingParams p;  // temperature 0: argmax of the penalized logits
    p.repeat_penalty = 2.0f;
    const std::vector<int64_t> hist = {0};
    check(sbnn::sample_token(logits, p, 0, hist, {}) == 1,
          "repeat_penalty divides a positive logit");
  }
  // penalty applied to a negative logit multiplies it (moves further away).
  {
    const std::vector<float> logits = {-1.0f, -2.0f};
    sbnn::SamplingParams p;
    p.repeat_penalty = 2.0f;
    const std::vector<int64_t> hist = {1};  // -2 -> -4
    check(sbnn::sample_token(logits, p, 0, hist, {}) == 0,
          "repeat_penalty multiplies a negative logit");
  }
  // presence_penalty subtracts once.
  {
    const std::vector<float> logits = {1.2f, 1.0f};
    sbnn::SamplingParams p;
    p.presence_penalty = 0.4f;
    const std::vector<int64_t> hist = {0};
    check(sbnn::sample_token(logits, p, 0, hist, {}) == 1,
          "presence_penalty subtracted once");
  }
  // frequency_penalty subtracts per occurrence.
  {
    const std::vector<float> logits = {1.0f, 1.0f};
    sbnn::SamplingParams p;
    p.frequency_penalty = 0.2f;
    const std::vector<int64_t> hist = {0, 0};
    check(sbnn::sample_token(logits, p, 0, hist, {}) == 1,
          "frequency_penalty subtracted per occurrence");
  }
  // repeat_last_n restricts the window.
  {
    const std::vector<float> logits = {1.5f, 1.0f};
    sbnn::SamplingParams p;
    p.repeat_penalty = 2.0f;
    const std::vector<int64_t> hist = {1, 0};  // 1 then 0; window of 1 = {0}
    p.repeat_last_n = 1;
    check(sbnn::sample_token(logits, p, 0, hist, {}) == 1,
          "repeat_last_n window excludes older tokens");
    p.repeat_last_n = 2;
    check(sbnn::sample_token(logits, p, 0, hist, {}) == 0,
          "repeat_last_n covers both tokens");
  }
}

void test_determinism() {
  const std::vector<float> logits = {0.3f, 1.1f, 0.7f, 0.2f, 0.9f};
  sbnn::SamplingParams p;
  p.temperature = 0.9f;
  p.top_p = 0.95f;
  p.seed = 7;
  const std::vector<int64_t> hist = {1, 2};
  const int64_t a = sbnn::sample_token(logits, p, 42, hist, {});
  const int64_t b = sbnn::sample_token(logits, p, 42, hist, {});
  check(a == b, "same (seed, position) gives the same token");
  // A different seed changes at least one of many positions (it need not
  // change every one).
  bool differs = false;
  for (uint64_t pos = 0; pos < 64 && !differs; ++pos) {
    const int64_t t0 = sbnn::sample_token(logits, p, pos, hist, {});
    p.seed = 8;
    const int64_t t1 = sbnn::sample_token(logits, p, pos, hist, {});
    p.seed = 7;
    differs = t0 != t1;
  }
  check(differs, "a different seed changes the sampled stream");
}

void test_distribution() {
  const std::vector<float> logits = {0.0f, 1.0f, 2.0f, 3.0f};
  sbnn::SamplingParams p;
  p.temperature = 1.0f;
  p.seed = 12345;
  constexpr int kDraws = 200000;
  std::array<int, 4> count{};
  for (int i = 0; i < kDraws; ++i) {
    const int64_t t = sbnn::sample_token(logits, p, uint64_t(i), {}, {});
    check(t >= 0 && t < 4, "sample in range");
    if (t < 0 || t >= 4) return;
    ++count[size_t(t)];
  }
  double z[4];
  double total = 0;
  for (int i = 0; i < 4; ++i) {
    z[i] = std::exp(double(logits[size_t(i)]));
    total += z[i];
  }
  double chi2 = 0;
  for (int i = 0; i < 4; ++i) {
    const double e = total > 0 ? kDraws * z[i] / total : 0;
    chi2 += (count[size_t(i)] - e) * (count[size_t(i)] - e) / e;
  }
  // df = 3; the 0.001 critical value is 16.27.
  if (!(chi2 < 16.27)) {
    std::cerr << "    chi2 = " << chi2 << " counts {" << count[0] << ","
              << count[1] << "," << count[2] << "," << count[3] << "}\n";
  }
  check(chi2 < 16.27, "sampled distribution matches softmax (chi-square)");
}

}  // namespace

int main() {
  test_philox_kat();
  test_greedy();
  test_top_k_one_is_argmax();
  test_top_k_set();
  test_min_p_set();
  test_top_p_set();
  test_penalties();
  test_determinism();
  test_distribution();
  if (g_failures) {
    std::cerr << g_failures << " sampling test(s) failed\n";
    return 1;
  }
  std::cout << "sampling tests passed\n";
  return 0;
}
