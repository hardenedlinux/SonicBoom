// Tests for the Phase 4 transformer operator primitives (sonicboom::nn):
// SiLU, GELU, and RMSNorm. These are pure C++23 float32 reference kernels with
// no device backend, so correctness is anchored on known values and simple
// algebraic invariants (RMSNorm maps any input to unit RMS with unit weights).

#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/attention.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/embedding.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/nn/softmax.h>

#include <sonicboom/dtype.h>
#include <sonicboom/quant/dequant.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <span>
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

void test_silu() {
  std::vector<float> x = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
  std::vector<float> y(x.size());

  check(sbnn::silu(x, y), "silu returns true with a matching output span");
  check(y[0] == 0.0f, "silu(0) == 0");
  check(std::abs(y[1] - 0.7310585786f) < 1e-6f, "silu(1) ~= 0.7310586");
  check(y[2] < 0.0f, "silu(-1) < 0");
  check(y[1] > 0.0f && y[3] > y[1], "silu is increasing on the positive side");

  // A too-short output span is rejected and writes nothing.
  std::vector<float> short_y(x.size() - 1);
  check(!sbnn::silu(x, short_y), "silu rejects a too-short output span");
}

void test_gelu() {
  std::vector<float> x = {0.0f, 1.0f};
  std::vector<float> y(x.size());

  check(sbnn::gelu(x, y), "gelu returns true with a matching output span");
  check(y[0] == 0.0f, "gelu(0) == 0");
  // tanh-approximation GELU at x=1.
  check(std::abs(y[1] - 0.841192f) < 1e-4f, "gelu(1) ~= 0.841192");
}

// gelu_fp16 replicates llama.cpp's GGML_GELU_FP16 fp16-table GELU: the input is
// rounded to fp16 before the tanh-GELU and the result is rounded back to fp16,
// clamped to 0 / identity outside [-10, 10]. The clamp edges are exact; the
// fp16-rounded values are checked against their exact fp16 representations.
void test_gelu_fp16() {
  std::vector<float> x = {0.0f, 1.0f, -10.5f, 10.5f, 2.0f};
  std::vector<float> y(x.size());

  check(sbnn::gelu_fp16(x, y), "gelu_fp16 returns true with a matching output span");

  // Clamping outside [-10, 10] (llama.cpp's ggml_vec_gelu_f32 table path).
  check(y[0] == 0.0f, "gelu_fp16(0) == 0");
  check(y[2] == 0.0f, "gelu_fp16(-10.5) == 0 (clamped)");
  check(y[3] == 10.5f, "gelu_fp16(10.5) == 10.5 (identity clamp)");

  // fp16-table quantization: both the input and the result are fp16-rounded, so
  // gelu_fp16 differs from the precise gelu(). These are the exact fp16 values.
  check(std::abs(y[1] - 0.841308594f) < 1e-3f, "gelu_fp16(1) ~= 0.841309 (fp16)");
  check(std::abs(y[4] - 1.95507812f) < 1e-3f, "gelu_fp16(2) ~= 1.955078 (fp16)");

  // The fp16 quantization is what distinguishes gelu_fp16 from gelu: at x=1 the
  // precise tanh-GELU is 0.841192, not the fp16-rounded 0.841309.
  std::vector<float> precise(x.size());
  check(sbnn::gelu(x, precise), "gelu returns true for the comparison");
  check(std::abs(y[1] - precise[1]) > 5e-5f,
        "gelu_fp16(1) differs from gelu(1) (fp16 quantization)");

  // A too-short output span is rejected and writes nothing.
  std::vector<float> short_y(x.size() - 1);
  check(!sbnn::gelu_fp16(x, short_y), "gelu_fp16 rejects a too-short output span");
}

void test_rms_norm_known() {
  // x = [2,2,2,2]: mean(x^2) = 4, so unit RMS is 1 -> y all 1 (eps = 0).
  {
    std::vector<float> x = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> y(4);
    check(sbnn::rms_norm(x, {}, 0.0f, y), "rms_norm (unit weight) returns true");
    for (int i = 0; i < 4; ++i)
      check(std::abs(y[i] - 1.0f) < 1e-6f, "rms_norm([2,2,2,2], eps=0) == 1");
  }
  // A learned weight scales the normalized result.
  {
    std::vector<float> x = {2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> w = {3.0f, 3.0f, 3.0f, 3.0f};
    std::vector<float> y(4);
    check(sbnn::rms_norm(x, w, 0.0f, y), "rms_norm (weighted) returns true");
    for (int i = 0; i < 4; ++i)
      check(std::abs(y[i] - 3.0f) < 1e-6f, "rms_norm weighted == weight * 1");
  }
}

void test_rms_norm_invariant() {
  // For any input and unit weights, RMS(y) == 1 (eps = 0, and up to float error).
  std::mt19937 rng(0x5EED);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  for (int trial = 0; trial < 8; ++trial) {
    const size_t n = 64 + (trial % 3) * 137;  // varied, not power-of-two
    std::vector<float> x(n), y(n);
    for (auto& v : x) v = dist(rng);

    check(sbnn::rms_norm(x, {}, 0.0f, y), "rms_norm returns true (invariant)");

    double sum_sq = 0.0;
    for (auto v : y) sum_sq += double(v) * double(v);
    const float rms = std::sqrt(float(sum_sq / double(n)));
    check(std::abs(rms - 1.0f) < 1e-4f, "RMSNorm output has unit RMS");
  }
}

void test_rms_norm_eps() {
  // x = [1, 0, 0, 0]: mean(x^2) = 0.25. With eps = 0.75, the scale is
  // 1/sqrt(1) = 1, so y = [1, 0, 0, 0]. (Picks a round number so the result is
  // exact.)
  std::vector<float> x = {1.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> y(4);
  check(sbnn::rms_norm(x, {}, 0.75f, y), "rms_norm (eps) returns true");
  check(y[0] == 1.0f && y[1] == 0.0f && y[2] == 0.0f && y[3] == 0.0f,
        "rms_norm eps shifts the denominator correctly");
}

void test_rope() {
  // pos = 0 -> theta = 0 for every pair -> identity rotation.
  {
    std::vector<float> x = {0.5f, -1.25f, 2.0f, 3.5f};
    std::vector<float> before = x;
    check(sbnn::rope_neox(x, 0, 10000.0f), "rope_neox returns true at pos 0");
    check(x == before, "rope_neox(pos=0) is the identity");
  }

  // head_dim = 2: theta = freq_scale * pos * base^0 = freq_scale * pos. Pass
  // freq_scale = pi/2 with pos = 1 so the angle is exactly pi/2 -> cos = 0,
  // sin = 1, so [1, 0] rotates to [0, 1].
  {
    constexpr float kPiOver2 = 1.5707963267948966f;
    std::vector<float> x = {1.0f, 0.0f};
    check(sbnn::rope_neox(x, 1, 1.0f, kPiOver2),
          "rope_neox returns true (quarter turn)");
    check(std::abs(x[0]) < 1e-6f && std::abs(x[1] - 1.0f) < 1e-6f,
          "[1,0] rotated by pi/2 -> [0,1]");
  }

  // RoPE is an orthogonal transform: it preserves the L2 norm of the vector.
  {
    std::mt19937 rng(0xE7);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int trial = 0; trial < 8; ++trial) {
      const size_t n = 32 + (trial % 3) * 16;
      std::vector<float> x(n);
      double before = 0.0;
      for (auto& v : x) { v = dist(rng); before += double(v) * double(v); }

      check(sbnn::rope_neox(x, 7, 10000.0f), "rope_neox returns true (norm test)");
      double after = 0.0;
      for (auto v : x) after += double(v) * double(v);
      check(std::abs(after - before) < 1e-3 * std::max(1.0, before),
            "rope_neox preserves the L2 norm");
    }
  }

  // Odd-length or empty spans are rejected.
  {
    std::vector<float> odd = {1.0f, 2.0f, 3.0f};
    check(!sbnn::rope_neox(odd, 1, 10000.0f), "rope_neox rejects odd-length span");
    std::vector<float> empty;
    check(!sbnn::rope_neox(empty, 1, 10000.0f), "rope_neox rejects empty span");
  }

  // Proportional RoPE: a per-pair factor divides the angle; a 1e30 sentinel
  // drives the angle to ~0, disabling rotation for that pair (Gemma 4 global
  // layers). With base=1, freq_scale=pi/2, pos=1 every pair's angle is pi/2;
  // freq_factors=[1.0, 1e30] rotates pair 0 and freezes pair 1.
  {
    constexpr float kPiOver2 = 1.5707963267948966f;
    const float sentinel = 1e30f;
    std::vector<float> x = {1.0f, 0.0f, 2.0f, 3.0f};
    std::vector<float> f = {1.0f, sentinel};
    check(sbnn::rope_neox(x, 1, 1.0f, kPiOver2, f),
          "rope_neox returns true (freq_factors)");
    check(std::abs(x[0] - (-2.0f)) < 1e-5f && std::abs(x[2] - 1.0f) < 1e-5f,
          "freq_factors pair 0 rotates by pi/2: (1,2) -> (-2,1)");
    check(std::abs(x[1]) < 1e-5f && std::abs(x[3] - 3.0f) < 1e-5f,
          "freq_factors sentinel freezes pair 1: (0,3) unchanged");

    // A too-short freq_factors span is rejected and writes nothing.
    std::vector<float> short_f = {1.0f};
    std::vector<float> y = {1.0f, 0.0f, 2.0f, 3.0f};
    check(!sbnn::rope_neox(y, 1, 1.0f, kPiOver2, short_f),
          "rope_neox rejects too-short freq_factors");
  }
}

void test_softmax() {
  // Known values.
  {
    std::vector<float> x = {0.0f, 0.0f};
    std::vector<float> y(2);
    check(sbnn::softmax(x, y), "softmax returns true");
    check(std::abs(y[0] - 0.5f) < 1e-6f && std::abs(y[1] - 0.5f) < 1e-6f,
          "softmax([0,0]) == [0.5,0.5]");
  }
  {
    std::vector<float> x = {1.0f, 2.0f, 3.0f};
    std::vector<float> y(3);
    check(sbnn::softmax(x, y), "softmax([1,2,3]) returns true");
    check(std::abs(y[0] - 0.09003057f) < 1e-6f, "softmax([1,2,3])[0]");
    check(std::abs(y[2] - 0.66524096f) < 1e-6f, "softmax([1,2,3])[2]");
  }

  // Numerical stability: a large common offset must not produce NaN, and equal
  // inputs stay uniform.
  {
    std::vector<float> x = {1000.0f, 1000.0f, 1000.0f};
    std::vector<float> y(3);
    check(sbnn::softmax(x, y), "softmax(large) returns true");
    for (int i = 0; i < 3; ++i)
      check(std::isfinite(y[i]) && std::abs(y[i] - 1.0f / 3.0f) < 1e-6f,
            "softmax(large equal) is uniform and finite");
  }

  // Random rows sum to 1.
  {
    std::mt19937 rng(0x50F7);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<float> x(32), y(32);
    for (auto& v : x) v = dist(rng);
    check(sbnn::softmax(x, y), "softmax(random) returns true");
    float sum = 0.0f;
    for (auto v : y) sum += v;
    check(std::abs(sum - 1.0f) < 1e-5f, "softmax sums to 1");
  }

  // Causal mask.
  {
    std::vector<float> x = {0.0f, 0.0f, 0.0f};
    std::vector<float> y(3);
    check(sbnn::softmax_causal(x, 0, y), "softmax_causal(key_pos=0) returns true");
    check(y[0] == 1.0f && y[1] == 0.0f && y[2] == 0.0f,
          "softmax_causal(key_pos=0) attends only index 0");

    check(sbnn::softmax_causal(x, 1, y), "softmax_causal(key_pos=1) returns true");
    check(std::abs(y[0] - 0.5f) < 1e-6f && std::abs(y[1] - 0.5f) < 1e-6f &&
              y[2] == 0.0f,
          "softmax_causal(key_pos=1) attends indices 0..1");

    check(sbnn::softmax_causal(x, 2, y), "softmax_causal(key_pos=2) returns true");
    for (int i = 0; i < 3; ++i)
      check(std::abs(y[i] - 1.0f / 3.0f) < 1e-6f, "softmax_causal(key_pos=2) uniform");
  }

  // A query position beyond the key count leaves the mask empty.
  {
    std::vector<float> x = {1.0f, 1.0f};
    std::vector<float> y(2);
    check(sbnn::softmax_causal(x, 5, y), "softmax_causal(key_pos > n) returns true");
    check(std::abs(y[0] - 0.5f) < 1e-6f && std::abs(y[1] - 0.5f) < 1e-6f,
          "softmax_causal(key_pos > n) is unmasked");
  }

  // Short output span rejected.
  {
    std::vector<float> x(4, 0.0f), y(3);
    check(!sbnn::softmax(x, y), "softmax rejects short output");
    check(!sbnn::softmax_causal(x, 0, y), "softmax_causal rejects short output");
  }
}

void test_elementwise() {
  {
    std::vector<float> a = {1.0f, 2.0f, 3.0f}, b = {4.0f, 5.0f, 6.0f}, y(3);
    check(sbnn::mul(a, b, y), "mul returns true");
    check(y[0] == 4.0f && y[1] == 10.0f && y[2] == 18.0f, "mul elementwise");
  }
  {
    std::vector<float> x = {1.0f, 2.0f, 3.0f}, y(3);
    check(sbnn::scale(x, 2.0f, y), "scale returns true");
    check(y[0] == 2.0f && y[1] == 4.0f && y[2] == 6.0f, "scale elementwise");
  }
  {
    std::vector<float> a = {1.0f, 2.0f}, b = {3.0f, 4.0f}, y(2);
    check(sbnn::add(a, b, y), "add returns true");
    check(y[0] == 4.0f && y[1] == 6.0f, "add elementwise");
  }
  {
    std::vector<float> a = {3.0f, 4.0f}, y = {1.0f, 2.0f};
    check(sbnn::axpy(a, 2.0f, y), "axpy returns true");
    check(y[0] == 7.0f && y[1] == 10.0f, "axpy y += alpha*a");
  }

  // Size mismatches and short outputs are rejected.
  {
    std::vector<float> a = {1.0f, 2.0f}, b = {1.0f}, y(2);
    check(!sbnn::mul(a, b, y), "mul rejects mismatched sizes");
    check(!sbnn::add(a, b, y), "add rejects mismatched sizes");
  }
  {
    std::vector<float> a(4, 1.0f), y(3);
    check(!sbnn::scale(a, 1.0f, y), "scale rejects short output");
    check(!sbnn::axpy(a, 1.0f, y), "axpy rejects short output");
  }
}

void test_attention() {
  // Single head, head_dim = 1: q=[1], k=[1], v=[5] -> out=[5] (score scaled by 1).
  {
    std::vector<float> q = {1.0f}, k = {1.0f}, v = {5.0f}, out(1);
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 1, 1, 1, 1, false, 0,
                                             0.0f, 1.0f, out),
          "attention (1x1x1) returns true");
    check(std::abs(out[0] - 5.0f) < 1e-6f, "attention single-key copies value");
  }

  // head_dim = 4, one key: out = v regardless of score magnitude.
  {
    std::vector<float> q = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> k = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> out(4);
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 1, 1, 1, 4, false, 0,
                                             0.0f, 1.0f, out),
          "attention (head_dim=4) returns true");
    for (int d = 0; d < 4; ++d)
      check(std::abs(out[d] - v[d]) < 1e-6f, "attention single-key == value");
  }

  // Causal masking across two query positions.
  {
    std::vector<float> q = {1.0f, 1.0f};  // two queries, head_dim=1
    std::vector<float> k = {1.0f, 1.0f};
    std::vector<float> v = {2.0f, 4.0f};
    std::vector<float> out(2);
    check(sbnn::scaled_dot_product_attention(q, 2, k, v, 2, 1, 1, 1, true, 0,
                                             0.0f, 1.0f, out),
          "attention (causal) returns true");
    check(std::abs(out[0] - 2.0f) < 1e-6f, "causal: q0 attends only key0");
    check(std::abs(out[1] - 3.0f) < 1e-6f, "causal: q1 attends keys0..1");
  }

  // GQA: 4 query heads over 2 KV heads (group = 2).
  {
    std::vector<float> q(4, 1.0f);   // one query, 4 heads, head_dim=1
    std::vector<float> k(2, 1.0f);   // one key,   2 heads
    std::vector<float> v = {1.0f, 2.0f};
    std::vector<float> out(4);
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 1, 4, 2, 1, false, 0,
                                             0.0f, 1.0f, out),
          "attention (GQA) returns true");
    check(std::abs(out[0] - 1.0f) < 1e-6f && std::abs(out[1] - 1.0f) < 1e-6f,
          "GQA: query heads 0..1 -> kv head 0");
    check(std::abs(out[2] - 2.0f) < 1e-6f && std::abs(out[3] - 2.0f) < 1e-6f,
          "GQA: query heads 2..3 -> kv head 1");
  }

  // Sliding window: with window 2, query 2 drops key 0.
  {
    std::vector<float> q = {1.0f, 1.0f, 1.0f};
    std::vector<float> k = {1.0f, 1.0f, 1.0f};
    std::vector<float> v = {1.0f, 2.0f, 4.0f};
    std::vector<float> out(3);
    check(sbnn::scaled_dot_product_attention(q, 3, k, v, 3, 1, 1, 1, true, 2,
                                             0.0f, 1.0f, out),
          "attention (sliding window) returns true");
    check(std::abs(out[2] - 3.0f) < 1e-6f,
          "sliding window: q2 attends keys1..2 -> 3");
  }

  // Softcap: caps large logits, shifting mass toward the smaller-logit key.
  {
    std::vector<float> q = {2.0f}, k = {2.0f, 0.0f}, v = {1.0f, 3.0f};
    std::vector<float> out_plain(1), out_softcap(1);
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 2, 1, 1, 1, false, 0,
                                             0.0f, 1.0f, out_plain),
          "attention (no softcap) returns true");
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 2, 1, 1, 1, false, 0,
                                             1.0f, 1.0f, out_softcap),
          "attention (softcap) returns true");
    check(std::abs(out_plain[0] - 1.0360f) < 1e-3f, "attention no-softcap value");
    check(std::abs(out_softcap[0] - 1.538f) < 1e-3f, "attention softcap value");
    check(out_softcap[0] != out_plain[0], "softcap changes the output");
  }

  // Scale (logit temperature): scale < 1 flattens the softmax toward the
  // lower-logit key, lifting the output toward the mean of the values.
  {
    std::vector<float> q = {2.0f}, k = {2.0f, 0.0f}, v = {1.0f, 3.0f};
    std::vector<float> out_1(1), out_half(1);
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 2, 1, 1, 1, false, 0,
                                             0.0f, 1.0f, out_1),
          "attention (scale=1) returns true");
    check(sbnn::scaled_dot_product_attention(q, 1, k, v, 2, 1, 1, 1, false, 0,
                                             0.0f, 0.5f, out_half),
          "attention (scale=0.5) returns true");
    check(std::abs(out_1[0] - 1.0360f) < 1e-3f, "attention scale=1 value");
    check(std::abs(out_half[0] - 1.2384f) < 1e-3f, "attention scale=0.5 value");
    check(out_half[0] > out_1[0], "scale < 1 spreads attention toward the mean");
  }

  // Validation.
  {
    std::vector<float> q(4, 1.0f), k(2, 1.0f), v(2, 1.0f), out(4);
    check(!sbnn::scaled_dot_product_attention(q, 1, k, v, 1, 4, 2, 0, false, 0,
                                              0.0f, 1.0f, out),
          "attention rejects head_dim == 0");
    check(!sbnn::scaled_dot_product_attention(q, 1, k, v, 1, 3, 2, 1, false, 0,
                                              0.0f, 1.0f, out),
          "attention rejects non-multiple GQA heads");
    std::vector<float> short_out(3);
    check(!sbnn::scaled_dot_product_attention(q, 1, k, v, 1, 4, 2, 1, false, 0,
                                              0.0f, 1.0f, short_out),
          "attention rejects wrong-sized output");
  }
}

void test_sampling() {
  // softcap: y = cap * tanh(x/cap).
  {
    std::vector<float> x = {0.0f, 30.0f, 1000.0f};
    std::vector<float> y(3);
    check(sbnn::softcap(x, 30.0f, y), "softcap returns true");
    check(y[0] == 0.0f, "softcap(0) == 0");
    check(std::abs(y[1] - 30.0f * std::tanh(1.0f)) < 1e-5f, "softcap(30, cap=30)");
    check(std::abs(y[2] - 30.0f) < 1e-5f, "softcap saturates to cap");
    check(y[1] > y[0] && y[2] > y[1], "softcap is monotonic");
  }
  {
    std::vector<float> x = {1.0f}, y(1);
    check(!sbnn::softcap(x, 0.0f, y), "softcap rejects cap <= 0");
    check(!sbnn::softcap(x, -1.0f, y), "softcap rejects negative cap");
    std::vector<float> short_y(0);
    check(!sbnn::softcap(x, 30.0f, short_y), "softcap rejects short output");
  }

  // argmax (greedy selection, ties -> lowest index).
  {
    std::vector<float> x = {1.0f, 3.0f, 2.0f};
    check(sbnn::argmax(x) == 1, "argmax picks the max");
  }
  {
    std::vector<float> x = {5.0f, 5.0f, 5.0f};
    check(sbnn::argmax(x) == 0, "argmax ties -> lowest index");
  }
  {
    std::vector<float> empty;
    check(sbnn::argmax(empty) == -1, "argmax(empty) == -1");
  }
}

void test_embedding() {
  // Build a 2-row q4_K embedding [256, 2]: row 0 first element 1, row 1 first
  // element 2. d=1.0, dmin=0, scales -> sc=1/min=0.
  auto make_block = [](uint8_t q0) {
    std::vector<std::byte> b;
    b.reserve(144);
    auto push_u16 = [&](uint16_t v) {
      b.push_back(std::byte(v & 0xFF));
      b.push_back(std::byte(v >> 8));
    };
    push_u16(0x3C00);  // d = 1.0
    push_u16(0x0000);  // dmin = 0.0
    for (int i = 0; i < 12; ++i) b.push_back(std::byte(i < 4 ? 1 : 0));  // sc=1,min=0
    for (int i = 0; i < 128; ++i) b.push_back(std::byte(i == 0 ? q0 : 0));
    return b;
  };
  std::vector<std::byte> bytes;
  auto b0 = make_block(0x01), b1 = make_block(0x02);
  bytes.insert(bytes.end(), b0.begin(), b0.end());
  bytes.insert(bytes.end(), b1.begin(), b1.end());

  sonicboom::quant::QuantizedTensor W(sonicboom::quant::QuantType::Q4_K, {256, 2},
                                      bytes);
  check(W.valid(), "embedding: tensor valid");

  std::vector<float> e0(256), e1(256);
  check(sbnn::embedding_f32(W, 0, e0), "embedding_f32(token 0) returns true");
  check(sbnn::embedding_f32(W, 1, e1), "embedding_f32(token 1) returns true");
  check(e0[0] == 1.0f && e1[0] == 2.0f, "embedding_f32 returns distinct rows");

  std::vector<float> short_out(255);
  check(!sbnn::embedding_f32(W, 0, short_out), "embedding_f32 rejects short output");
  check(!sbnn::embedding_f32(W, 2, e0), "embedding_f32 rejects token out of range");
  {
    sonicboom::quant::QuantizedTensor flat(sonicboom::quant::QuantType::Q4_K,
                                           {256}, bytes);
    check(!sbnn::embedding_f32(flat, 0, e0), "embedding_f32 rejects 1-D weight");
  }
}

void test_validation() {
  std::vector<float> x(8, 1.0f);

  // Output span too short.
  {
    std::vector<float> y(7);
    check(!sbnn::rms_norm(x, {}, 0.0f, y), "rms_norm rejects short y");
  }
  // Weight span present but too short.
  {
    std::vector<float> w(7, 1.0f), y(8);
    check(!sbnn::rms_norm(x, w, 0.0f, y), "rms_norm rejects short weight");
  }
  // Empty weight is accepted as unit weights.
  {
    std::vector<float> y(8);
    check(sbnn::rms_norm(x, {}, 0.0f, y), "rms_norm accepts empty weight span");
  }
}

// The 5 kernels promoted from exec.cpp's inline loops (Phase 6 M1). Each is
// differentially checked against the pre-hoist inline reference; the kernels
// preserve the reference's per-row/per-head independence, so agreement is
// bit-exact (==), not fp32 tolerance.
void test_matvec_bf16() {
  const uint64_t cols = 64, rows = 16;
  std::mt19937 rng(0xB16);
  std::vector<uint16_t> W(cols * rows);
  std::vector<float> x(cols), y(rows), yr(rows);
  for (auto& v : W)
    v = sonicboom::f32_to_bf16(std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng));
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  check(sbnn::matvec_bf16(W, cols, x, y), "matvec_bf16 returns true");
  for (uint64_t j = 0; j < rows; ++j) {
    double acc = 0.0;
    const uint64_t base = j * cols;
    for (uint64_t i = 0; i < cols; ++i)
      acc += double(sonicboom::bf16_to_f32(W[base + i]) * x[i]);
    yr[j] = float(acc);
  }
  for (uint64_t j = 0; j < rows; ++j) check(y[j] == yr[j], "matvec_bf16 == reference");

  check(!sbnn::matvec_bf16(W, cols + 1, x, y), "matvec_bf16 rejects short W");
  std::vector<float> short_x(cols - 1);
  check(!sbnn::matvec_bf16(W, cols, short_x, y), "matvec_bf16 rejects short x");
}

void test_rms_norm_heads() {
  const uint64_t head_dim = 64, heads = 4;
  const size_t n = head_dim * heads;
  std::mt19937 rng(0x0B0E);
  std::vector<float> x(n), w(head_dim), xr, scratch(head_dim);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);

  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  xr = x;
  check(sbnn::rms_norm_heads(x, w, head_dim, 1e-6f), "rms_norm_heads (weighted) returns true");
  for (uint64_t h = 0; h < heads; ++h) {
    std::span<float> xh = std::span<float>(xr).subspan(h * head_dim, head_dim);
    check(sbnn::rms_norm(xh, w, 1e-6f, scratch), "ref rms_norm returns true");
    std::copy(scratch.begin(), scratch.end(), xh.begin());
  }
  for (size_t i = 0; i < n; ++i) check(x[i] == xr[i], "rms_norm_heads == per-head reference");

  // Unweighted (empty weight => unit).
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  xr = x;
  check(sbnn::rms_norm_heads(x, {}, head_dim, 1e-6f), "rms_norm_heads (unit) returns true");
  for (uint64_t h = 0; h < heads; ++h) {
    std::span<float> xh = std::span<float>(xr).subspan(h * head_dim, head_dim);
    check(sbnn::rms_norm(xh, {}, 1e-6f, scratch), "ref rms_norm (unit) returns true");
    std::copy(scratch.begin(), scratch.end(), xh.begin());
  }
  for (size_t i = 0; i < n; ++i) check(x[i] == xr[i], "rms_norm_heads (unit) == reference");

  check(!sbnn::rms_norm_heads(x, w, 0, 1e-6f), "rms_norm_heads rejects head_dim == 0");
  std::vector<float> short_w(head_dim - 1, 1.0f);
  check(!sbnn::rms_norm_heads(x, short_w, head_dim, 1e-6f),
        "rms_norm_heads rejects short weight");
  std::vector<float> not_mult(n - 1);
  check(!sbnn::rms_norm_heads(not_mult, w, head_dim, 1e-6f),
        "rms_norm_heads rejects non-multiple span");
}

void test_rope_heads() {
  const uint64_t head_dim = 64, heads = 4;
  const size_t n = head_dim * heads;
  const uint64_t pos = 7;
  std::mt19937 rng(0xE0F);
  std::vector<float> x(n), xr, ff(head_dim / 2, 1.0f);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (size_t i = 16; i < ff.size(); ++i) ff[i] = 1e30f;

  xr = x;
  check(sbnn::rope_neox_heads(x, head_dim, pos, 10000.0f, 1.0f, ff),
        "rope_neox_heads returns true");
  for (uint64_t h = 0; h < heads; ++h)
    sbnn::rope_neox(std::span<float>(xr).subspan(h * head_dim, head_dim), pos,
                    10000.0f, 1.0f, ff);
  for (size_t i = 0; i < n; ++i) check(x[i] == xr[i], "rope_neox_heads == per-head reference");

  check(!sbnn::rope_neox_heads(x, 0, pos, 10000.0f, 1.0f, {}),
        "rope_neox_heads rejects head_dim == 0");
  check(!sbnn::rope_neox_heads(x, 63, pos, 10000.0f, 1.0f, {}),
        "rope_neox_heads rejects odd head_dim");
  std::vector<float> not_mult(n - 1);
  check(!sbnn::rope_neox_heads(not_mult, head_dim, pos, 10000.0f, 1.0f, {}),
        "rope_neox_heads rejects non-multiple span");
}

void test_gqa_broadcast() {
  // n_q = 4, n_kv = 2, head_dim = 2: query heads 0,1 -> kv head 0; 2,3 -> kv 1.
  const uint64_t n_q = 4, n_kv = 2, head_dim = 2;
  std::vector<float> src(n_kv * head_dim), dst(n_q * head_dim);
  for (size_t i = 0; i < src.size(); ++i) src[i] = float(i + 1);  // [1,2 | 3,4]

  check(sbnn::gqa_broadcast(src, dst, n_q, n_kv, head_dim), "gqa_broadcast returns true");
  for (uint64_t hq = 0; hq < n_q; ++hq) {
    const uint64_t kv_h = hq * n_kv / n_q;
    for (uint64_t d = 0; d < head_dim; ++d)
      check(dst[hq * head_dim + d] == src[kv_h * head_dim + d],
            "gqa_broadcast maps head -> donor KV head");
  }

  check(!sbnn::gqa_broadcast(src, dst, n_q, n_kv, 0), "gqa_broadcast rejects head_dim == 0");
  check(!sbnn::gqa_broadcast(src, dst, 3, n_kv, head_dim),
        "gqa_broadcast rejects non-multiple n_heads_q");
  std::vector<float> short_src(src.size() - 1), short_dst(dst.size() - 1);
  check(!sbnn::gqa_broadcast(short_src, dst, n_q, n_kv, head_dim),
        "gqa_broadcast rejects short src");
  check(!sbnn::gqa_broadcast(src, short_dst, n_q, n_kv, head_dim),
        "gqa_broadcast rejects short dst");
}

void test_layer_combine() {
  std::vector<float> proj = {1.0f, 2.0f, 3.0f};
  std::vector<float> ple = {4.0f, 5.0f, 6.0f};
  std::vector<float> out(3);
  const float ple_scale = 2.0f, combine_scale = 0.5f;
  check(sbnn::layer_combine(proj, ple, ple_scale, combine_scale, out),
        "layer_combine returns true");
  for (size_t i = 0; i < 3; ++i)
    check(out[i] == (proj[i] + ple[i] * ple_scale) * combine_scale,
          "layer_combine arithmetic");

  std::vector<float> short_ple(2), short_out(2);
  check(!sbnn::layer_combine(proj, short_ple, ple_scale, combine_scale, out),
        "layer_combine rejects short ple");
  check(!sbnn::layer_combine(proj, ple, ple_scale, combine_scale, short_out),
        "layer_combine rejects short out");
}

} // namespace

int main() {
  test_silu();
  test_gelu();
  test_gelu_fp16();
  test_rms_norm_known();
  test_rms_norm_invariant();
  test_rms_norm_eps();
  test_rope();
  test_softmax();
  test_elementwise();
  test_attention();
  test_sampling();
  test_embedding();
  test_validation();
  test_matvec_bf16();
  test_rms_norm_heads();
  test_rope_heads();
  test_gqa_broadcast();
  test_layer_combine();

  if (g_failures == 0) {
    std::cout << "test_nn OK\n";
    return 0;
  }
  std::cerr << "test_nn FAILED: " << g_failures << " check(s)\n";
  return 1;
}
