// CUDA elementwise/norm/RoPE/dense-matvec differential test (Phase 6a target 3).
//
// Each nn::cuda::* entry point is compared against its CPU reference in
// core/src/nn/*.cpp (or, for the bf16 matvec, the hand-rolled double-accumulation
// loop the model uses). The CPU reference accumulates mean/matvec in double
// (ggml_float) while the CUDA kernels accumulate in fp32, so agreement is to an
// fp32 tolerance (1e-3 relative), not bit-exact — the same relaxation the K-quant
// gemv test (test_cuda_quant.cpp) makes.
//
// Skips cleanly (exit 0) when no CUDA device is present.

#include <sonicboom/nn/activation.h>
#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/nn/elementwise.h>
#include <sonicboom/nn/matmul.h>
#include <sonicboom/nn/norm.h>
#include <sonicboom/nn/rope.h>

#include <algorithm>
#include <bit>
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

// Compare CUDA output to CPU reference within an fp32 tolerance; worst relative
// error against max(1, |ref|) (avoids cancellation blowup).
double compare(std::span<const float> got, std::span<const float> ref, bool* ok) {
  *ok = true;
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double denom = std::max(1.0, double(std::fabs(ref[i])));
    const double err = std::fabs(double(got[i]) - double(ref[i])) / denom;
    worst = std::max(worst, err);
    if (err > 1e-3) *ok = false;
  }
  return worst;
}

void check_cmp(const char* name, std::span<const float> got,
               std::span<const float> ref) {
  bool ok = false;
  const double worst = compare(got, ref, &ok);
  if (!ok) std::cerr << "    " << name << " worst rel err = " << worst << "\n";
  check(ok, name);
}

float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(uint32_t(h) << 16); }

// Round a bounded float to bf16 (round-to-nearest-even on the dropped 16 bits).
// Real model weights are small; raw random uint16 patterns would carry an 8-bit
// exponent up to ~3e38 and overflow the kernel's fp32 accumulation.
uint16_t f32_to_bf16(float f) {
  uint32_t u = std::bit_cast<uint32_t>(f);
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return uint16_t(u >> 16);
}

void test_rms_norm(std::mt19937& rng) {
  const size_t n = 2560;
  std::vector<float> x(n), w(n), y(n), yr(n);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  for (auto& v : w) v = std::uniform_real_distribution<float>(0.5f, 1.5f)(rng);

  check(sbnn::cuda::rms_norm(x, w, 1e-6f, y), "cuda::rms_norm (weighted) returns true");
  check(sbnn::rms_norm(x, w, 1e-6f, yr), "CPU rms_norm (weighted) returns true");
  check_cmp("cuda::rms_norm (weighted) matches CPU", y, yr);

  check(sbnn::cuda::rms_norm(x, {}, 1e-6f, y), "cuda::rms_norm (unweighted) returns true");
  check(sbnn::rms_norm(x, {}, 1e-6f, yr), "CPU rms_norm (unweighted) returns true");
  check_cmp("cuda::rms_norm (unweighted) matches CPU", y, yr);
}

void test_gelu_fp16(std::mt19937& rng) {
  const size_t n = 10240;
  std::vector<float> x(n), y(n), yr(n);
  // Span the clamp (-10..10) and the fp16-table interior.
  for (auto& v : x) v = std::uniform_real_distribution<float>(-12.0f, 12.0f)(rng);

  check(sbnn::cuda::gelu_fp16(x, y), "cuda::gelu_fp16 returns true");
  check(sbnn::gelu_fp16(x, yr), "CPU gelu_fp16 returns true");
  check_cmp("cuda::gelu_fp16 matches CPU", y, yr);
}

void test_mul_add_scale(std::mt19937& rng) {
  const size_t n = 2560;
  std::vector<float> a(n), b(n), y(n), yr(n);
  for (auto& v : a) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  for (auto& v : b) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  check(sbnn::cuda::mul(a, b, y), "cuda::mul returns true");
  check(sbnn::mul(a, b, yr), "CPU mul returns true");
  check_cmp("cuda::mul matches CPU", y, yr);

  check(sbnn::cuda::add(a, b, y), "cuda::add returns true");
  check(sbnn::add(a, b, yr), "CPU add returns true");
  check_cmp("cuda::add matches CPU", y, yr);

  check(sbnn::cuda::scale(a, 0.75f, y), "cuda::scale returns true");
  check(sbnn::scale(a, 0.75f, yr), "CPU scale returns true");
  check_cmp("cuda::scale matches CPU", y, yr);
}

void test_cast_fp16(std::mt19937& rng) {
  const size_t n = 2560;
  std::vector<float> x(n), xr(n);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  xr = x;

  check(sbnn::cuda::cast_fp16(x), "cuda::cast_fp16 returns true");
  sbnn::cast_fp16(xr);
  check_cmp("cuda::cast_fp16 matches CPU", x, xr);
}

void test_rope(std::mt19937& rng) {
  const size_t n = 512;
  const uint64_t pos = 7;
  std::vector<float> x(n), xr(n), ff(n / 2, 1.0f);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  xr = x;
  check(sbnn::cuda::rope_neox(x, pos, 10000.0f, 1.0f, {}), "cuda::rope_neox returns true");
  check(sbnn::rope_neox(xr, pos, 10000.0f, 1.0f, {}), "CPU rope_neox returns true");
  check_cmp("cuda::rope_neox matches CPU", x, xr);

  // Proportional RoPE: tail pairs disabled by a 1e30 divisor (Gemma global).
  for (size_t i = 64; i < ff.size(); ++i) ff[i] = 1e30f;
  xr = x;
  check(sbnn::cuda::rope_neox(x, pos, 10000.0f, 1.0f, ff),
        "cuda::rope_neox (freq) returns true");
  check(sbnn::rope_neox(xr, pos, 10000.0f, 1.0f, ff), "CPU rope_neox (freq) returns true");
  check_cmp("cuda::rope_neox (freq) matches CPU", x, xr);
}

void test_rope_heads(std::mt19937& rng) {
  // 4 heads of head_dim=256, with proportional-RoPE factors (64x1.0, tail 1e30)
  // shared across heads. The CPU reference rotates each head independently.
  const uint64_t head_dim = 256, heads = 4;
  const size_t n = head_dim * heads;
  const uint64_t pos = 7;
  std::vector<float> x(n), xr(n), ff(head_dim / 2, 1.0f);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  for (size_t i = 64; i < ff.size(); ++i) ff[i] = 1e30f;

  xr = x;
  check(sbnn::cuda::rope_neox_heads(x, head_dim, pos, 10000.0f, 1.0f, ff),
        "cuda::rope_neox_heads returns true");
  for (uint64_t h = 0; h < heads; ++h)
    sbnn::rope_neox(std::span<float>(xr).subspan(h * head_dim, head_dim), pos,
                    10000.0f, 1.0f, ff);
  check_cmp("cuda::rope_neox_heads matches CPU", x, xr);
}

void test_matvec_f32(std::mt19937& rng) {
  const uint64_t cols = 2560, rows = 256;
  std::vector<float> W(cols * rows), x(cols), y(rows), yr(rows);
  for (auto& v : W) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  check(sbnn::cuda::matvec_f32(W, cols, x, y), "cuda::matvec_f32 returns true");
  check(sbnn::matvec_f32(W, cols, x, yr), "CPU matvec_f32 returns true");
  check_cmp("cuda::matvec_f32 matches CPU", y, yr);
}

void test_matvec_bf16(std::mt19937& rng) {
  const uint64_t cols = 2560, rows = 256;
  std::vector<uint16_t> W(cols * rows);
  std::vector<float> x(cols), y(rows), yr(rows);
  for (auto& v : W)
    v = f32_to_bf16(std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng));
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  check(sbnn::cuda::matvec_bf16(W, cols, x, y), "cuda::matvec_bf16 returns true");
  // CPU reference: the model's double-accumulation bf16 dot (float(acc)).
  for (uint64_t j = 0; j < rows; ++j) {
    double acc = 0.0;
    const uint64_t base = j * cols;
    for (uint64_t i = 0; i < cols; ++i) acc += double(bf16_to_f32(W[base + i]) * x[i]);
    yr[j] = float(acc);
  }
  check_cmp("cuda::matvec_bf16 matches CPU", y, yr);
}

} // namespace

int main() {
  if (!sbnn::cuda::available()) {
    std::cout << "test_cuda_elementwise SKIP (no CUDA device)\n";
    return 0;
  }

  std::mt19937 rng(0xCAFE);
  test_rms_norm(rng);
  test_gelu_fp16(rng);
  test_mul_add_scale(rng);
  test_cast_fp16(rng);
  test_rope(rng);
  test_rope_heads(rng);
  test_matvec_f32(rng);
  test_matvec_bf16(rng);

  if (g_failures == 0) {
    std::cout << "test_cuda_elementwise OK\n";
    return 0;
  }
  std::cerr << "test_cuda_elementwise FAILED: " << g_failures << " check(s)\n";
  return 1;
}
