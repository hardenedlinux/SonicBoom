// Tests for the Phase 4 transformer operator primitives (sonicboom::nn):
// SiLU, Sigmoid, GELU, and RMSNorm. These are pure C++23 float32 reference kernels with
// no device backend, so correctness is anchored on known values and simple
// algebraic invariants (RMSNorm maps any input to unit RMS with unit weights).

#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/attention.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/embedding.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/moe.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>
#include <sonicboom/nn/sampling.h>
#include <sonicboom/nn/softmax.h>

#include <sonicboom/dtype.h>
#include <sonicboom/quant/dequant.h>
#include <sonicboom/quant/quantized_matmul.h>

#ifdef SB_TEST_CUDA
#include <cuda_runtime.h>

#include <sonicboom/nn/cuda_moe.h>

#include "../../core/src/nn/cuda_resident.h"
#endif

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
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

// Sigmoid is the same arithmetic as the reference on the CPU: both compute
// 1 / (1 + exp(-x)), so the results agree bit for bit (not just numerically).
// The values cover the flat tails where the activated value is 1 or 0 to f32
// (|x| = 20 and 100), the near-linear region around zero (|x| = 1e-3) and 0
// itself.
void test_sigmoid() {
  std::vector<float> x = {0.0f, 1e-3f, -1e-3f, 20.0f, -20.0f, 100.0f, -100.0f};
  std::vector<float> y(x.size());

  check(sbnn::sigmoid(x, y),
        "sigmoid returns true with a matching output span");
  for (size_t i = 0; i < x.size(); ++i) {
    const float ref = 1.0f / (1.0f + std::exp(-x[i]));
    check(std::bit_cast<uint32_t>(y[i]) == std::bit_cast<uint32_t>(ref),
          "sigmoid agrees with 1 / (1 + exp(-x)) bit for bit");
  }
  check(y[0] == 0.5f, "sigmoid(0) == 0.5");
  check(y[3] == 1.0f, "sigmoid(20) rounds to 1");
  check(y[4] > 0.0f && y[4] < 1e-6f, "sigmoid(-20) ~= 0");
  check(y[5] == 1.0f, "sigmoid(100) rounds to 1");
  check(y[6] == 0.0f, "sigmoid(-100) == 0 (exp(100) overflows f32)");

  // A too-short output span is rejected and writes nothing.
  std::vector<float> short_y(x.size() - 1);
  check(!sbnn::sigmoid(x, short_y),
        "sigmoid rejects a too-short output span");
}

#ifdef SB_TEST_CUDA
// The device sigmoid against the CPU one on the same values. Device expf and
// host exp are not required to be bit-identical, so this is the comparison the
// tree's other device-kernel suites use (an independent reference within
// 1e-5): the arithmetic is the same 1 / (1 + exp(-x)) on both sides.
void test_sigmoid_cuda() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "sigmoid: no CUDA device, device case skipped\n";
    return;
  }
  const std::vector<float> x = {0.0f,  1e-3f, -1e-3f, 20.0f,
                                -20.0f, 100.0f, -100.0f};
  const std::size_t bytes = x.size() * sizeof(float);
  std::vector<float> cpu(x.size()), dev(x.size());

  float* dx = nullptr;
  float* dy = nullptr;
  check(cudaMalloc(&dx, bytes) == cudaSuccess, "sigmoid: allocate the input");
  check(cudaMalloc(&dy, bytes) == cudaSuccess, "sigmoid: allocate the output");
  check(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess,
        "sigmoid: upload the input");
  check(sbnn::sigmoid(x, cpu), "sigmoid: the CPU reference runs");
  check(sbnn::cuda::sigmoid_dev(dx, dy, int(x.size())),
        "sigmoid_dev returns true");
  check(cudaMemcpy(dev.data(), dy, bytes, cudaMemcpyDeviceToHost) == cudaSuccess,
        "sigmoid: download the result");
  for (std::size_t i = 0; i < x.size(); ++i)
    check(std::abs(dev[i] - cpu[i]) <= 1e-5f,
          "sigmoid_dev agrees with the CPU sigmoid");
  cudaFree(dx);
  cudaFree(dy);
}
#endif

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

  // Masked argmax (constrained decoding): the plain result whenever its id is
  // allowed, the best allowed id otherwise, and -1 when nothing is allowed.
  {
    std::vector<float> x = {1.0f, 3.0f, 2.0f, 4.0f, 0.5f};
    const std::vector<uint8_t> all{0xFF};
    check(sbnn::argmax_masked(x, all) == 3,
          "masked argmax: all-ones equals argmax");
    const std::vector<uint8_t> without{uint8_t(~(1u << 3))};
    check(sbnn::argmax_masked(x, without) == 1,
          "masked argmax: skips a disallowed max");
    const std::vector<uint8_t> only{uint8_t(1u << 4)};
    check(sbnn::argmax_masked(x, only) == 4,
          "masked argmax: a single allowed id");
    const std::vector<uint8_t> none{0x00};
    check(sbnn::argmax_masked(x, none) == -1,
          "masked argmax: nothing allowed -> -1");
    const std::vector<uint8_t> short_mask{};
    check(sbnn::argmax_masked(x, short_mask) == -1,
          "masked argmax: an empty mask allows nothing");
    const std::vector<uint8_t> first_two{0x03};
    check(sbnn::argmax_masked(x, first_two) == 1,
          "masked argmax: ids past the allowed bits lose");
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

void test_moe() {
  namespace sbquant = sonicboom::quant;
  // q4_K stores 256-element blocks and its contiguous dimension must be a whole
  // number of blocks (QuantizedTensor::valid), so the smallest expert pair is
  // 256 wide on both sides. The gate's d = 64 / width 32 cannot be a q4_K row.
  constexpr uint32_t d = 256, ff = 256, E = 8, used = 2;

  // q4_K weights assembled block by block (the format's own test block, as in
  // test_dequant): d = 1 and dmin = 0, and every sub-block's scale is 1 with a
  // zero minimum, so a dequantized weight is its stored 4-bit nibble. The
  // nibbles are a deterministic pseudo-random sequence keyed by the tensor and
  // the expert, so no two experts share a matrix.
  std::vector<std::shared_ptr<std::vector<std::byte>>> keep;
  auto make = [&](uint64_t cols, uint64_t rows, uint16_t scale, uint32_t seed) {
    auto bytes = std::make_shared<std::vector<std::byte>>();
    const uint64_t blocks = cols * rows / 256;
    uint32_t h = seed * 2654435761u + 1u;
    for (uint64_t b = 0; b < blocks; ++b) {
      bytes->push_back(std::byte(scale & 0xFF));  // d (fp16, LE)
      bytes->push_back(std::byte(scale >> 8));
      bytes->push_back(std::byte(0x00));          // dmin = 0
      bytes->push_back(std::byte(0x00));
      const uint8_t scales[12] = {1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
      for (uint8_t s : scales) bytes->push_back(std::byte(s));
      for (int i = 0; i < 128; ++i) {
        h ^= h << 13;
        h ^= h >> 17;
        h ^= h << 5;
        bytes->push_back(std::byte(uint8_t(h & 0xFF)));
      }
    }
    keep.push_back(bytes);
    return sbquant::QuantizedTensor(sbquant::QuantType::Q4_K, {cols, rows},
                                    *bytes);
  };
  constexpr uint16_t kF16_1 = 0x3C00;
  // 2^-5: with unit scales the 256-wide gate/up sums reach ~1e5 and the hidden
  // product ~1e8. The CUDA prefill batched matmul routes through the fp16
  // activation path, so a scale that large takes the example outside the range
  // that path handles; 1/32 keeps the intermediates small enough and leaves the
  // arithmetic identical.
  constexpr uint16_t kF16_1_16 = 0x2800;

  std::vector<sbquant::QuantizedTensor> gate, up, down;
  for (uint32_t e = 0; e < E; ++e) {
    gate.push_back(make(d, ff, kF16_1_16, 1000 + e));
    up.push_back(make(d, ff, kF16_1_16, 2000 + e));
    down.push_back(make(ff, d, kF16_1_16, 3000 + e));
  }
  check(gate[0].valid() && down[0].valid(), "moe: the expert views are valid");

  std::vector<float> x(d);
  for (uint32_t i = 0; i < d; ++i) x[i] = float(int(i % 7) - 3) * 0.25f + 0.5f;

  // The routing written out directly: the logits through the same CPU
  // quantized dot the kernel's router uses, then the stable f32 softmax over
  // all experts in ascending index, the top-k with a strict comparison and the
  // ascending sum of the selected probabilities.
  auto route_reference = [&](std::span<const float> logits,
                             std::vector<int64_t>& want_ids,
                             std::vector<float>& want_w) {
    std::vector<float> probs(E, 0.0f);
    float mx = logits[0];
    for (float l : logits) mx = std::max(mx, l);
    float sum = 0.0f;
    for (uint32_t e = 0; e < E; ++e) {
      probs[e] = std::exp(logits[e] - mx);
      sum += probs[e];
    }
    for (uint32_t e = 0; e < E; ++e) probs[e] /= sum;
    std::vector<bool> taken(E, false);
    std::vector<uint32_t> sel;
    for (uint32_t j = 0; j < used; ++j) {
      uint32_t best = E;
      for (uint32_t e = 0; e < E; ++e) {
        if (taken[e]) continue;
        if (best == E || probs[e] > probs[best]) best = e;
      }
      taken[best] = true;
      sel.push_back(best);
    }
    std::sort(sel.begin(), sel.end());
    float denom = 0.0f;
    for (uint32_t e : sel) denom += probs[e];
    denom = std::max(denom, 6.103515625e-5f);
    want_ids.assign(used, 0);
    want_w.assign(used, 0.0f);
    for (uint32_t j = 0; j < used; ++j) {
      want_ids[j] = int64_t(sel[j]);
      want_w[j] = probs[sel[j]] / denom;
    }
  };

  const sbquant::QuantizedTensor router = make(d, E, kF16_1, 4000);
  std::vector<int64_t> ids(used);
  std::vector<float> weights(used);
  check(sbnn::moe_route(x, &router, std::span<const float>{}, E, used, 1.0f,
                        ids, weights),
        "moe_route returns true");
  {
    std::vector<float> logits(E);
    check(sbquant::matvec(router, x, logits, sbquant::MatmulBackend::CpuQ8K),
          "moe_route: the reference router runs");
    std::vector<int64_t> want_ids;
    std::vector<float> want_w;
    route_reference(logits, want_ids, want_w);
    check(ids[0] == want_ids[0] && ids[1] == want_ids[1],
          "moe_route selects the same experts as the reference");
    check(std::bit_cast<uint32_t>(weights[0]) ==
                  std::bit_cast<uint32_t>(want_w[0]) &&
              std::bit_cast<uint32_t>(weights[1]) ==
                  std::bit_cast<uint32_t>(want_w[1]),
          "moe_route weights are bit-identical to the reference");
  }

  // The float32 router branch routes through the dense matvec.
  {
    std::vector<float> router_f32(size_t(d) * E);
    for (uint32_t k = 0; k < d; ++k)
      for (uint32_t e = 0; e < E; ++e)
        router_f32[size_t(e) * d + k] =
            float(e) * 0.05f + float(k % 3) * 0.01f;
    std::vector<int64_t> fids(used);
    std::vector<float> fw(used);
    check(sbnn::moe_route(x, nullptr, router_f32, E, used, 1.0f, fids, fw),
          "moe_route: a float32 router returns true");
    std::vector<float> logits(E);
    check(sbnn::matvec_f32(router_f32, d, x, logits),
          "moe_route: the float32 reference router runs");
    std::vector<int64_t> want_ids;
    std::vector<float> want_w;
    route_reference(logits, want_ids, want_w);
    check(fids[0] == want_ids[0] && fids[1] == want_ids[1],
          "moe_route: the float32 router selects the same experts");
  }

  // Tie rule: a zero router makes every expert equally likely, and equal
  // probabilities select the lowest indices with equal weights.
  {
    const sbquant::QuantizedTensor zero_router =
        make(d, E, 0x0000, 5000);  // d = 0 -> every weight is zero
    std::vector<int64_t> tids(used);
    std::vector<float> tw(used);
    check(sbnn::moe_route(x, &zero_router, std::span<const float>{}, E, used,
                          1.0f, tids, tw),
          "moe_route: a flat router returns true");
    check(tids[0] == 0 && tids[1] == 1, "moe_route: a tie takes lower indices");
    check(tw[0] == 0.5f && tw[1] == 0.5f, "moe_route: tied weights are equal");
  }
  // Validation: more experts used than exist, and a short output span.
  {
    std::vector<int64_t> bad_ids(E + 1);
    std::vector<float> bad_w(E + 1);
    check(!sbnn::moe_route(x, &router, std::span<const float>{}, E, E + 1,
                           1.0f, bad_ids, bad_w),
          "moe_route rejects used > experts");
    std::vector<int64_t> short_ids(used - 1);
    check(!sbnn::moe_route(x, &router, std::span<const float>{}, E, used, 1.0f,
                           short_ids, weights),
          "moe_route rejects a short id span");
    check(!sbnn::moe_route(x, nullptr, std::span<const float>{}, E, used, 1.0f,
                           std::span<int64_t>(ids), weights),
          "moe_route rejects a missing router");
    check(!sbnn::moe_route(std::span<const float>{}, &router,
                           std::span<const float>{}, E, used, 1.0f,
                           std::span<int64_t>(ids), weights),
          "moe_route rejects an empty activation");
  }

  // The routed experts: the same projections, each expert's weight applied to
  // its own output, accumulated in ascending expert index.
  sbnn::MoeExpertWeights views{gate, up, down};
  std::vector<float> y(d, 0.0f);
  check(sbnn::moe_experts(x, ids, weights, views, sbnn::MoeActivation::Silu, y),
        "moe_experts returns true");
  {
    std::vector<float> want(d, 0.0f);
    for (size_t k = 0; k < ids.size(); ++k) {
      std::vector<float> g(ff), u(ff), h(ff), out(d);
      sbquant::matvec(gate[size_t(ids[k])], x, g,
                      sbquant::MatmulBackend::CpuQ8K);
      sbquant::matvec(up[size_t(ids[k])], x, u, sbquant::MatmulBackend::CpuQ8K);
      sbnn::silu(g, g);
      for (uint32_t i = 0; i < ff; ++i) h[i] = g[i] * u[i];
      sbquant::matvec(down[size_t(ids[k])], h, out,
                      sbquant::MatmulBackend::CpuQ8K);
      for (uint32_t i = 0; i < d; ++i) want[i] += weights[k] * out[i];
    }
    bool exact = true;
    for (uint32_t i = 0; i < d; ++i)
      exact = exact && std::bit_cast<uint32_t>(y[i]) ==
                           std::bit_cast<uint32_t>(want[i]);
    check(exact, "moe_experts output is bit-identical to the reference");
  }
  // A zero-weight expert contributes nothing; an out-of-range id is refused.
  {
    std::vector<float> zero_w = {0.0f, 0.0f};
    std::vector<float> z(d);
    check(sbnn::moe_experts(x, ids, zero_w, views, sbnn::MoeActivation::Silu, z),
          "moe_experts: zero weights return true");
    bool all_zero = true;
    for (float v : z) all_zero = all_zero && v == 0.0f;
    check(all_zero, "moe_experts: zero weights add nothing");
    std::vector<int64_t> bad = {int64_t(E)};
    std::vector<float> w1 = {1.0f}, out(d);
    check(!sbnn::moe_experts(x, bad, w1, views, sbnn::MoeActivation::Silu, out),
          "moe_experts rejects an expert index past the end");
  }

  // The routed sum accumulates in ascending expert index. Contributions that
  // differ in magnitude by many binades make the ascending and the descending
  // sums differ in their low bits, so the reference can tell the two apart and
  // the kernel has to match the ascending one bit for bit.
  {
    constexpr uint32_t kUsed = 4;
    const uint16_t scales[kUsed] = {0x6400, 0x3C00, 0x1400, 0x0010};
    std::vector<sbquant::QuantizedTensor> g4, u4, d4;
    for (uint32_t e = 0; e < kUsed; ++e) {
      g4.push_back(make(d, ff, kF16_1, 6000 + e));
      u4.push_back(make(d, ff, kF16_1, 7000 + e));
      d4.push_back(make(ff, d, scales[e], 8000 + e));
    }
    sbnn::MoeExpertWeights v4{g4, u4, d4};
    const std::vector<int64_t> order = {0, 1, 2, 3};
    const std::vector<float> w4 = {0.4f, 0.3f, 0.2f, 0.1f};
    std::vector<std::vector<float>> outs(kUsed, std::vector<float>(d));
    for (uint32_t e = 0; e < kUsed; ++e) {
      std::vector<float> g(ff), u(ff), h(ff);
      sbquant::matvec(g4[e], x, g, sbquant::MatmulBackend::CpuQ8K);
      sbquant::matvec(u4[e], x, u, sbquant::MatmulBackend::CpuQ8K);
      sbnn::silu(g, g);
      for (uint32_t i = 0; i < ff; ++i) h[i] = g[i] * u[i];
      sbquant::matvec(d4[e], h, outs[e], sbquant::MatmulBackend::CpuQ8K);
    }
    std::vector<float> asc(d, 0.0f), desc(d, 0.0f);
    for (uint32_t i = 0; i < d; ++i) {
      for (uint32_t e = 0; e < kUsed; ++e) asc[i] += w4[e] * outs[e][i];
      for (uint32_t j = kUsed; j-- > 0;) desc[i] += w4[j] * outs[j][i];
    }
    bool order_matters = false;
    for (uint32_t i = 0; i < d; ++i)
      order_matters = order_matters || std::bit_cast<uint32_t>(asc[i]) !=
                                          std::bit_cast<uint32_t>(desc[i]);
    check(order_matters,
          "moe_experts: the example distinguishes the two summation orders");
    std::vector<float> y4(d, 0.0f);
    check(sbnn::moe_experts(x, order, w4, v4, sbnn::MoeActivation::Silu, y4),
          "moe_experts: the four-expert order case returns true");
    bool asc_exact = true;
    for (uint32_t i = 0; i < d; ++i)
      asc_exact = asc_exact && std::bit_cast<uint32_t>(y4[i]) ==
                                   std::bit_cast<uint32_t>(asc[i]);
    check(asc_exact, "moe_experts sums the routed experts in ascending order");
  }

#ifdef SB_TEST_CUDA
  // The CUDA twin on the same example: decode within the tree's fp32
  // tolerance of the CPU kernel, a four-row verify bit-identical row by row to
  // four single-row decodes, and a four-row prefill within tolerance of the
  // CPU.
  {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::cout << "moe: no CUDA device, device cases skipped\n";
    } else {
      std::vector<int64_t> cids(used);
      std::vector<float> cw(used);
      check(sbnn::cuda::moe_route(x, &router, std::span<const float>{}, E, used,
                                  1.0f, cids, cw, 1),
            "moe cuda: the route runs");
      check(cids[0] == ids[0] && cids[1] == ids[1],
            "moe cuda: the route selects the CPU's experts");
      std::vector<float> y_cuda(d, 0.0f), y_cpu(d, 0.0f);
      check(sbnn::cuda::moe_experts(x, cids, cw, views,
                                    sbnn::MoeActivation::Silu, y_cuda, 1, false),
            "moe cuda: the decode runs");
      check(sbnn::moe_experts(x, cids, cw, views, sbnn::MoeActivation::Silu,
                              y_cpu),
            "moe cuda: the CPU reference runs");
      bool close = true;
      for (uint32_t i = 0; i < d; ++i)
        close = close && std::abs(y_cuda[i] - y_cpu[i]) <=
                             2e-3f * std::max(1.0f, std::abs(y_cpu[i]));
      check(close, "moe cuda: decode agrees with the CPU within tolerance");

      // Verify: four rows through one call, then each row decoded alone.
      constexpr uint64_t rows = 4;
      std::vector<float> x4(rows * d);
      for (uint64_t r = 0; r < rows; ++r)
        for (uint32_t i = 0; i < d; ++i)
          x4[r * d + i] = x[i] * (0.5f + 0.25f * float(r));
      std::vector<int64_t> ids4(rows * used);
      std::vector<float> w4(rows * used);
      check(sbnn::cuda::moe_route(x4, &router, std::span<const float>{}, E,
                                  used, 1.0f, ids4, w4, rows),
            "moe cuda: the verify route runs");
      std::vector<float> yv(rows * d, 0.0f);
      check(sbnn::cuda::moe_experts(x4, ids4, w4, views,
                                    sbnn::MoeActivation::Silu, yv, rows, false),
            "moe cuda: the verify step runs");
      bool bitwise = true;
      for (uint64_t r = 0; r < rows && bitwise; ++r) {
        std::vector<float> one(d, 0.0f);
        if (!sbnn::cuda::moe_experts(
                std::span<const float>(x4).subspan(r * d, d),
                std::span<const int64_t>(ids4).subspan(r * used, used),
                std::span<const float>(w4).subspan(r * used, used), views,
                sbnn::MoeActivation::Silu, one, 1, false)) {
          bitwise = false;
          break;
        }
        for (uint32_t i = 0; i < d; ++i)
          bitwise = bitwise && std::bit_cast<uint32_t>(yv[r * d + i]) ==
                                   std::bit_cast<uint32_t>(one[i]);
      }
      check(bitwise,
            "moe cuda: every verify row matches one decode bit for bit");

      // Prefill: the grouped form against the CPU row by row.
      std::vector<float> yp(rows * d, 0.0f);
      check(sbnn::cuda::moe_experts(x4, ids4, w4, views,
                                    sbnn::MoeActivation::Silu, yp, rows, true),
            "moe cuda: the prefill runs");
      bool prefill_close = true;
      for (uint64_t r = 0; r < rows && prefill_close; ++r) {
        std::vector<float> want(d, 0.0f);
        if (!sbnn::moe_experts(
                std::span<const float>(x4).subspan(r * d, d),
                std::span<const int64_t>(ids4).subspan(r * used, used),
                std::span<const float>(w4).subspan(r * used, used), views,
                sbnn::MoeActivation::Silu, want)) {
          prefill_close = false;
          break;
        }
        for (uint32_t i = 0; i < d; ++i)
          prefill_close = prefill_close &&
                          std::abs(yp[r * d + i] - want[i]) <=
                              2e-3f * std::max(1.0f, std::abs(want[i]));
      }
      check(prefill_close,
            "moe cuda: prefill agrees with the CPU within tolerance");
    }
  }
#endif
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

// Partial rotary embedding: rot_dims (0 = the whole head) rotates only that many
// leading values of each head, with rot_dims as the angle denominator and the
// head's tail copied through unchanged. The reference is rope_neox applied to
// the leading rot_dims values with the same freq_factors prefix.
void test_rope_heads_partial() {
  const uint64_t head_dim = 64, heads = 4;
  const size_t n = head_dim * heads;
  const uint64_t pos = 7, rot = head_dim / 4;
  std::mt19937 rng(0xE0F1);
  std::vector<float> x(n), ff(head_dim / 2, 1.0f);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);
  for (size_t i = 4; i < ff.size(); ++i) ff[i] = 1e30f;

  // rot_dims == head_dim is the whole head: bit-identical to leaving it 0.
  std::vector<float> a = x, b = x;
  check(sbnn::rope_neox_heads(a, head_dim, pos, 10000.0f, 1.0f, ff),
        "rope_neox_heads (whole head) returns true");
  check(sbnn::rope_neox_heads(b, head_dim, pos, 10000.0f, 1.0f, ff, head_dim),
        "rope_neox_heads (rot_dims = head_dim) returns true");
  bool same = true;
  for (size_t i = 0; i < n; ++i)
    same = same && std::bit_cast<uint32_t>(a[i]) == std::bit_cast<uint32_t>(b[i]);
  check(same, "rot_dims == head_dim is bit-identical to the whole head");

  std::vector<float> p = x;
  check(sbnn::rope_neox_heads(p, head_dim, pos, 10000.0f, 1.0f, ff, rot),
        "rope_neox_heads (partial) returns true");
  for (uint64_t h = 0; h < heads; ++h) {
    std::vector<float> ref(x.begin() + h * head_dim,
                           x.begin() + h * head_dim + head_dim);
    check(sbnn::rope_neox(std::span<float>(ref).subspan(0, rot), pos,
                          10000.0f, 1.0f,
                          std::span<const float>(ff).subspan(0, rot / 2)),
          "partial-rope reference runs");
    for (uint64_t d = 0; d < head_dim; ++d)
      check(p[h * head_dim + d] == ref[d],
            "partial rope matches the reference of the leading dims");
    // The tail is copied through unchanged.
    for (uint64_t d = rot; d < head_dim; ++d)
      check(p[h * head_dim + d] == x[h * head_dim + d],
            "partial rope leaves the tail of each head untouched");
  }

  check(!sbnn::rope_neox_heads(p, head_dim, pos, 10000.0f, 1.0f, {}, 17),
        "rope_neox_heads rejects odd rot_dims");
  check(!sbnn::rope_neox_heads(p, head_dim, pos, 10000.0f, 1.0f, {},
                               head_dim + 2),
        "rope_neox_heads rejects rot_dims > head_dim");
  std::vector<float> short_ff(rot / 2 - 1, 1.0f);
  check(!sbnn::rope_neox_heads(p, head_dim, pos, 10000.0f, 1.0f, short_ff, rot),
        "rope_neox_heads rejects short freq_factors for rot_dims");
}

#ifdef SB_TEST_CUDA
// The device partial rope against the CPU reference on the same values. Device
// arithmetic is not required to be bit-identical, so this is a tolerance check.
void test_rope_heads_cuda() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::cout << "rope_heads: no CUDA device, device case skipped\n";
    return;
  }
  const int head_dim = 64, heads = 4;
  const std::size_t n = head_dim * heads;
  const uint64_t rot = head_dim / 4;
  std::mt19937 rng(0xE0F2);
  std::vector<float> x(n), cpu(n), dev(n), ff(head_dim / 2, 1.0f);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-2.0f, 2.0f)(rng);

  float* dx = nullptr;
  const std::size_t bytes = n * sizeof(float);
  check(cudaMalloc(&dx, bytes) == cudaSuccess, "rope_heads: allocate");
  check(cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess,
        "rope_heads: upload");
  check(sbnn::cuda::rope_neox_heads_dev(dx, head_dim, heads, 7, 10000.0f, 1.0f,
                                        ff, rot),
        "rope_neox_heads_dev (partial) returns true");
  check(cudaMemcpy(dev.data(), dx, bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess,
        "rope_heads: download");

  cpu = x;
  check(sbnn::rope_neox_heads(cpu, head_dim, 7, 10000.0f, 1.0f, ff, rot),
        "rope_heads: the CPU reference runs");
  for (std::size_t i = 0; i < n; ++i)
    check(std::abs(dev[i] - cpu[i]) <= 1e-4f,
          "rope_neox_heads_dev (partial) agrees with the CPU reference");
  cudaFree(dx);
}
#endif

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
  test_sigmoid();
#ifdef SB_TEST_CUDA
  test_sigmoid_cuda();
#endif
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
  test_rope_heads_partial();
#ifdef SB_TEST_CUDA
  test_rope_heads_cuda();
#endif
  test_gqa_broadcast();
  test_layer_combine();
  test_moe();

  if (g_failures == 0) {
    std::cout << "test_nn OK\n";
    return 0;
  }
  std::cerr << "test_nn FAILED: " << g_failures << " check(s)\n";
  return 1;
}
