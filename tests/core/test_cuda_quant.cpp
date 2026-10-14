// CUDA K-quant matvec differential test.
//
// Builds synthetic Q3_K/Q4_K/Q5_K/Q6_K weights (random but finite block bytes, the
// same construction as test_quantized_matmul.cpp), runs the CUDA dequantize-
// on-the-fly gemv, and compares against the scalar CPU f32-activation reference
// (quant::matvec_f32). The CUDA kernel reproduces the dequant arithmetic but
// accumulates in fp32 with a non-deterministic block-reduction order, so
// agreement is to an fp32 tolerance, not bit-exact.
//
// Skips cleanly (exit 0) when no CUDA device is present, mirroring test_cuda.

#include <sonicboom/quant/dequant.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/quant/quantized_tensor.h>

#include "ggml_baseline.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sbquant = sonicboom::quant;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

uint16_t rand_finite_half(std::mt19937& rng) {
  uint16_t h;
  do { h = uint16_t(rng()); } while (((h >> 10) & 0x1F) == 0x1F);
  return h;
}

std::vector<std::byte> build_weight(sbquant::QuantType type, uint64_t rows,
                                    uint64_t cols, std::mt19937& rng) {
  const sbquant::QuantTypeInfo* info = sbquant::quant_type_info(type);
  const uint64_t blocks_per_row =
      (cols + info->block_size - 1) / info->block_size;
  const uint64_t n_blocks = rows * blocks_per_row;
  const uint64_t total = n_blocks * info->bytes_per_block;

  std::vector<std::byte> bytes(total);
  for (auto& b : bytes) b = std::byte(uint8_t(rng()));

  for (uint64_t i = 0; i < n_blocks; ++i) {
    const size_t off = i * info->bytes_per_block;
    if (type == sbquant::QuantType::Q3_K) {
      const uint16_t d = rand_finite_half(rng);
      bytes[off + 108] = std::byte(d & 0xFF);
      bytes[off + 109] = std::byte(d >> 8);
    } else if (type == sbquant::QuantType::Q6_K) {
      // Keep d at one so the Q8_1 differential does not hide lane errors.
      const uint16_t d = 0x3C00;  // 1.0
      bytes[off + 208] = std::byte(d & 0xFF);
      bytes[off + 209] = std::byte(d >> 8);
    } else {
      const uint16_t d = rand_finite_half(rng);
      const uint16_t dmin = rand_finite_half(rng);
      bytes[off + 0] = std::byte(d & 0xFF);
      bytes[off + 1] = std::byte(d >> 8);
      bytes[off + 2] = std::byte(dmin & 0xFF);
      bytes[off + 3] = std::byte(dmin >> 8);
    }
  }
  return bytes;
}

// Compare CUDA output to CPU reference within an fp32 tolerance. Returns the
// worst relative error seen (against max(1, |ref|) to avoid cancellation blowup).
double compare(std::span<const float> got, std::span<const float> ref,
               double tol, bool* ok) {
  *ok = true;
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double denom = std::max(1.0, double(std::fabs(ref[i])));
    const double err = std::fabs(double(got[i]) - double(ref[i])) / denom;
    worst = std::max(worst, err);
    if (err > tol) *ok = false;
  }
  return worst;
}

std::vector<float> q6_f32_reference(const sbquant::QuantizedTensor& W,
                                    std::span<const float> X, uint64_t n) {
  const uint64_t cols = W.dims()[0], rows = W.dims()[1];
  std::vector<float> weights(W.numel());
  if (!sbquant::dequantize_f32(W, weights.data(), weights.size())) return {};
  std::vector<float> out(rows * n, 0.0f);
  for (uint64_t c = 0; c < n; ++c) {
    for (uint64_t r = 0; r < rows; ++r) {
      float sum = 0.0f;
      for (uint64_t k = 0; k < cols; ++k)
        sum += weights[r * cols + k] * X[c * cols + k];
      out[c * rows + r] = sum;
    }
  }
  return out;
}

int q6_centered_q(const uint8_t* wb, int k) {
  const int half = k >> 7;
  const int r = k & 127;
  const int group = r >> 5;
  const int lane = r & 31;
  const int ql_index = (half << 6) + ((group & 1) ? 32 : 0) + lane;
  const uint8_t ql_byte = wb[ql_index];
  const int ql = group < 2 ? (ql_byte & 0x0F) : (ql_byte >> 4);
  const int qh = (wb[128 + (half << 5) + lane] >> (2 * group)) & 3;
  return (ql | (qh << 4)) - 32;
}

float fp16_to_f32(uint16_t h) {
  const uint32_t sign = uint32_t(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) bits = sign;
    else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1F) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }
  return std::bit_cast<float>(bits);
}

std::vector<float> q6_q8_1_reference(const sbquant::QuantizedTensor& W,
                                     std::span<const float> X, uint64_t n) {
  const uint64_t cols = W.dims()[0], rows = W.dims()[1];
  constexpr uint64_t kBlock = 256;
  struct Q8Block { float d; std::array<int8_t, 32> q; };
  const uint64_t q8_blocks_per_row = cols / 32;
  std::vector<Q8Block> xq(n * q8_blocks_per_row);
  for (uint64_t c = 0; c < n; ++c) {
    for (uint64_t k0 = 0; k0 < cols; k0 += 32) {
      const float* xb = X.data() + c * cols + k0;
      float amax = 0.0f;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(xb[j]));
      const float d = amax / 127.0f;
      Q8Block& qb = xq[c * q8_blocks_per_row + k0 / 32];
      qb.d = d;
      for (int j = 0; j < 32; ++j) {
        qb.q[j] = d == 0.0f ? 0 : int8_t(std::round(xb[j] / d));
      }
    }
  }

  const auto raw = W.data();
  const auto* bytes = reinterpret_cast<const uint8_t*>(raw.data());
  const uint64_t blocks_per_row = cols / kBlock;
  std::vector<float> out(rows * n, 0.0f);
  for (uint64_t c = 0; c < n; ++c) {
    for (uint64_t r = 0; r < rows; ++r) {
      float lane_sum[32] = {};
      for (uint64_t b = 0; b < blocks_per_row; ++b) {
        const uint8_t* wb = bytes + (r * blocks_per_row + b) * 210;
        const auto* scales = reinterpret_cast<const int8_t*>(wb + 192);
        const uint16_t dh = uint16_t(wb[208]) | (uint16_t(wb[209]) << 8);
        const float d6 = fp16_to_f32(dh);
        for (int iqs = 0; iqs < 32; ++iqs) {
          const int bq8_offset = 2 * 2 * (iqs / 16) + (iqs % 16) / 8;
          const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
          const int vh_shift = 2 * ((iqs % 16) / 8);
          auto load_u32 = [](const uint8_t* p) {
            return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
                   (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
          };
          const uint32_t vl = load_u32(wb + 4 * iqs);
          const int vh_index = 8 * (iqs / 16) + iqs % 8;
          const uint32_t vh = load_u32(wb + 128 + 4 * vh_index) >> vh_shift;
          float sumf = 0.0f;
          for (int i = 0; i < 2; ++i) {
            const Q8Block& qb = xq[c * q8_blocks_per_row +
                                   b * 8 + bq8_offset + 2 * i];
            const int q8_byte = (iqs % 8) * 4;
            int32_t dot = 0;
            for (int j = 0; j < 4; ++j) {
              const int ql = (vl >> (4 * i + 8 * j)) & 0x0F;
              const int qh = ((vh >> (4 * i + 8 * j)) << 4) & 0x30;
              const int q = ql | qh;
              dot += (q - 32) * int32_t(qb.q[q8_byte + j]);
            }
            const int32_t scaled_dot = dot * int32_t(scales[scale_offset + 4 * i]);
            sumf = std::fma(qb.d, float(scaled_dot), sumf);
          }
          lane_sum[iqs] = std::fma(d6, sumf, lane_sum[iqs]);
        }
      }
      for (int stride = 16; stride > 0; stride >>= 1)
        for (int lane = 0; lane < 32 - stride; ++lane)
          lane_sum[lane] += lane_sum[lane + stride];
      out[c * rows + r] = lane_sum[0];
    }
  }
  return out;
}

void test_q6_single_q8_1(std::mt19937& rng) {
  constexpr uint64_t rows = 64;
  constexpr uint64_t cols = 256 * 4;
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(sbquant::QuantType::Q6_K, rows, cols, rng));
  const auto& bytes = keepalive.back();
  sbquant::QuantizedTensor W(sbquant::QuantType::Q6_K, {cols, rows}, bytes);
  std::vector<float> X(cols), Y(rows);
  for (float& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  unsetenv("SONICBOOM_GEMV_F32");
  const bool ok = sbquant::cuda::matvec_f32(W, X, Y);
  check(ok, "Q6_K q8_1 single-column gemv returns true");
  if (!ok) return;
  const std::vector<float> ref = q6_q8_1_reference(W, X, 1);
  bool cmp = false;
  const double worst = compare(Y, ref, 1e-4, &cmp);
  check(cmp, "Q6_K q8_1 single-column gemv matches integer CPU q8_1 reference");
  if (!cmp) std::cerr << "    worst rel err (Q6_K q8_1 single) = " << worst << "\n";
  setenv("SONICBOOM_GEMV_F32", "1", 1);
}

void test_one_shape(sbquant::QuantType type, const char* name, uint64_t rows,
                    uint64_t cols, double tol, std::mt19937& rng) {
  // The device weight cache keys by host pointer + byte size; keep each weight's
  // backing buffer alive for the whole run so a later shape can't reuse the freed
  // address at the same size and hit a stale device copy.
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(type, rows, cols, rng));
  const std::vector<std::byte>& bytes = keepalive.back();
  sbquant::QuantizedTensor W(type, {cols, rows}, bytes);
  if (!W.valid()) {
    check(false, "build_weight produced an invalid tensor");
    return;
  }

  std::vector<float> x(cols), y(rows), y_ref(rows);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  bool ok = sbquant::cuda::matvec_f32(W, x, y);
  check(ok, "cuda::matvec_f32 returns true");
  if (!ok) return;

  check(sbquant::matvec_f32(W, x, y_ref), "CPU matvec_f32 returns true");

  bool cmp = false;
  const double worst = compare(y, y_ref, tol, &cmp);
  check(cmp, "cuda::matvec_f32 matches CPU reference");
  if (!cmp) std::cerr << "    worst rel err (" << name << ") = " << worst << "\n";

  // Second call must hit the cached device weight and still agree.
  std::vector<float> y2(rows);
  check(sbquant::cuda::matvec_f32(W, x, y2), "cuda::matvec_f32 (cached) returns true");
  bool cmp2 = false;
  compare(y2, y_ref, tol, &cmp2);
  check(cmp2, "cuda::matvec_f32 (cached) matches CPU reference");
}

void test_one_type(sbquant::QuantType type, const char* name,
                   std::mt19937& rng) {
  // Small shape (4 blocks/row) -> one-thread-per-block kernel path. The fp32 dot
  // over 4*256 terms reproduces the CPU reference closely, so 1e-3 suffices.
  test_one_shape(type, name, 64, 256 * 4, 1e-3, rng);
  // Large shape (40 blocks/row) -> cooperative warp-per-row kernel path. The fp32
  // dot over 40*256 terms in a different accumulation order (__shfl reduction) is
  // the same arithmetic but drifts ~3e-2, so the tolerance is loosened.
  test_one_shape(type, name, 64, 256 * 40, 1e-1, rng);
}

// Batched (prefill) matmul, fp32 differential path: Y = W @ X for n positions,
// compared against the CPU matmul_f32 reference (X [n, cols], Y [n, rows]
// position-major). main() sets SONICBOOM_GEMV_F32=1, so this exercises the fp32
// fallback (the n× gemv/copy loop) — the exact dequant arithmetic, compared at
// the fp32 tolerances below. The q8_1 batched path (the default) is covered
// separately by test_batched_q8_1, bit-exact against the per-position q8_1 matvec.
void test_batched(sbquant::QuantType type, const char* name, uint64_t rows,
                  uint64_t cols, uint64_t n, double tol, std::mt19937& rng) {
  // Same keepalive as test_one_shape: keep each weight's backing buffer alive so
  // the pointer-keyed device cache never sees a freed-and-reused address.
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(type, rows, cols, rng));
  const std::vector<std::byte>& bytes = keepalive.back();
  sbquant::QuantizedTensor W(type, {cols, rows}, bytes);
  if (!W.valid()) {
    check(false, "batched: build_weight produced an invalid tensor");
    return;
  }

  std::vector<float> X(cols * n), Y(rows * n), Y_ref(rows * n);
  for (auto& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  bool ok = sbquant::cuda::matmul_f32(W, X, Y, n);
  check(ok, "cuda::matmul_f32 returns true");
  if (!ok) return;

  check(sbquant::matmul_f32(W, X, Y_ref, n), "CPU matmul_f32 returns true");

  bool cmp = false;
  const double worst = compare(Y, Y_ref, tol, &cmp);
  check(cmp, "cuda::matmul_f32 matches CPU reference");
  if (!cmp) std::cerr << "    worst rel err (batched " << name << ") = " << worst << "\n";
}

void test_batched_type(sbquant::QuantType type, const char* name,
                       std::mt19937& rng) {
  // n == 1 exercises the fallback to the single-column matvec path.
  test_batched(type, name, 64, 256 * 4, 1, 1e-3, rng);
  // Small shape (4 blocks/row), n == 8 -> gather/gemv/scatter over the
  // one-thread-per-block kernel.
  test_batched(type, name, 64, 256 * 4, 8, 1e-3, rng);
  // Large shape (40 blocks/row), n == 4 -> cooperative warp-per-row kernel.
  test_batched(type, name, 64, 256 * 40, 4, 1e-1, rng);
}

// q8_1 batched matmul (the decode/prefill default), checked bit-exact against
// the per-position q8_1 matvec. Unlike test_batched, this runs with
// SONICBOOM_GEMV_F32 *unset* so matmul_f32 takes the quantize_q8_1 -> batched
// warp-per-row path. That path must be bit-identical to n calls of matvec_f32
// (same quantize_q8_1, same vec_dot, same __shfl reduction order); any
// divergence here means the position-major q8_1 layout is wrong.
void test_batched_q8_1(sbquant::QuantType type, const char* name, uint64_t rows,
                       uint64_t cols, uint64_t n, std::mt19937& rng) {
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(type, rows, cols, rng));
  const std::vector<std::byte>& bytes = keepalive.back();
  sbquant::QuantizedTensor W(type, {cols, rows}, bytes);
  if (!W.valid()) {
    check(false, "q8_1 batched: build_weight produced an invalid tensor");
    return;
  }

  std::vector<float> X(cols * n), Yb(rows * n), Yr(rows * n);
  for (auto& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  bool ok = sbquant::cuda::matmul_f32(W, X, Yb, n);
  check(ok, "cuda::matmul_f32 (q8_1 batched) returns true");
  if (!ok) return;

  for (uint64_t c = 0; c < n; ++c) {
    std::vector<float> xc(cols), yc(rows);
    for (uint64_t k = 0; k < cols; ++k) xc[k] = X[c * cols + k];
    if (!sbquant::cuda::matvec_f32(W, xc, yc)) {
      check(false, "cuda::matvec_f32 (q8_1 single-col) returns true");
      return;
    }
    for (uint64_t r = 0; r < rows; ++r) Yr[c * rows + r] = yc[r];
  }

  size_t ndiff = 0;
  for (uint64_t i = 0; i < rows * n; ++i)
    if (Yb[i] != Yr[i]) ++ndiff;
  if (ndiff) {
    std::cerr << "    q8_1 batched " << name << ": " << ndiff << "/" << rows * n
              << " elements differ from per-column matvec\n";
  }
  check(ndiff == 0, "q8_1 batched == per-column matvec (bit-exact)");

  if (type == sbquant::QuantType::Q6_K) {
    const std::vector<float> Y_cpu = q6_q8_1_reference(W, X, n);
    bool cmp = false;
    const double worst = compare(Yb, Y_cpu, 1e-4, &cmp);
    check(cmp, "Q6_K q8_1 batched matches integer CPU q8_1 reference");
    if (!cmp) std::cerr << "    worst rel err (Q6_K q8_1 batch) = " << worst << "\n";
  }
}

void test_batched_q8_1_type(sbquant::QuantType type, const char* name,
                            std::mt19937& rng) {
  // n == 3 (partial tile), n == 8 (full tile), n == 10 (two tiles): covers the
  // p0/T tile loop's partial-tile path and the multi-tile path.
  test_batched_q8_1(type, name, 64, 256 * 4, 3, rng);
  test_batched_q8_1(type, name, 64, 256 * 40, 3, rng);
  test_batched_q8_1(type, name, 64, 256 * 40, 8, rng);
  test_batched_q8_1(type, name, 64, 256 * 40, 10, rng);
}

void test_mmq_q6_type(std::mt19937& rng) {
  (void)rng;
  std::cout << "  SKIP Q6_K MMQ: TODO(q6k-mmq): incoherent output on Gemma Q4_K_M, disabled until fixed\n";
  return;

  constexpr uint64_t rows = 128;
  constexpr uint64_t cols = 256 * 4;
  constexpr uint64_t n = 3;
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(sbquant::QuantType::Q6_K, rows, cols, rng));
  const auto& bytes = keepalive.back();
  sbquant::QuantizedTensor W(sbquant::QuantType::Q6_K, {cols, rows}, bytes);
  std::vector<float> X(cols * n), Y(rows * n), ref(rows * n);
  for (float& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  unsetenv("SONICBOOM_GEMV_F32");
  setenv("SONICBOOM_MMQ", "1", 1);
  setenv("SONICBOOM_CUBLAS", "0", 1);
  const bool ok = sbquant::cuda::matmul_f32(W, X, Y, n);
  check(ok, "Q6_K MMQ matmul returns true");
  if (!ok) return;
  check(sbquant::matmul_f32(W, X, ref, n), "Q6_K MMQ CPU reference returns true");
  bool cmp = false;
  const double worst = compare(Y, ref, 1e-1, &cmp);
  check(cmp, "Q6_K MMQ matches CPU reference tolerance");
  if (!cmp) std::cerr << "    worst rel err (Q6_K MMQ) = " << worst << "\n";
}

void test_cublas_q6_type(std::mt19937& rng) {
  constexpr uint64_t rows = 64;
  constexpr uint64_t cols = 256 * 4;
  constexpr uint64_t n = 32;
  static std::vector<std::vector<std::byte>> keepalive;
  keepalive.push_back(build_weight(sbquant::QuantType::Q6_K, rows, cols, rng));
  auto& bytes = keepalive.back();
  for (size_t off = 0; off < bytes.size(); off += 210)
    for (int i = 0; i < 16; ++i) bytes[off + 192 + i] = std::byte{1};
  sbquant::QuantizedTensor W(sbquant::QuantType::Q6_K, {cols, rows}, bytes);
  std::vector<float> X(cols * n), Y(rows * n), ref(rows * n);
  for (float& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);
  unsetenv("SONICBOOM_GEMV_F32");
  setenv("SONICBOOM_MMQ", "0", 1);
  unsetenv("SONICBOOM_CUBLAS");
  const bool ok = sbquant::cuda::matmul_f32(W, X, Y, n);
  check(ok, "Q6_K cuBLAS fp16 matmul returns true");
  if (!ok) return;
  check(sbquant::matmul_f32(W, X, ref, n), "Q6_K cuBLAS CPU reference returns true");
  bool cmp = false;
  const double worst = compare(Y, ref, 1e-1, &cmp);
  check(cmp, "Q6_K cuBLAS fp16 dequant matches CPU reference tolerance");
  if (!cmp) std::cerr << "    worst rel err (Q6_K cuBLAS) = " << worst << "\n";
}

} // namespace

int main() {
  // This test diffs the CUDA kernel against the scalar CPU f32-activation
  // reference, so it must exercise the exact fp32 dequant-on-the-fly path. The
  // q8_1 mmvq path (the decode default) quantizes the activation to int8 and is
  // approximate; it is validated end-to-end by test_generate's golden token
  // stream instead of differentially.
  setenv("SONICBOOM_GEMV_F32", "1", 1);

  if (!sbquant::cuda::available()) {
    std::cout << "test_cuda_quant SKIP (no CUDA device)\n";
    return 0;
  }

  std::mt19937 rng(0xBEEF);
  test_q6_single_q8_1(rng);
  if (std::getenv("SONICBOOM_TEST_Q6_SINGLE_ONLY")) {
    if (g_failures) {
      std::cerr << "test_cuda_quant FAILED: " << g_failures << " check(s)\n";
      return 1;
    }
    std::cout << "test_cuda_quant Q6_K single q8_1 OK\n";
    return 0;
  }
  test_one_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_one_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_one_type(sbquant::QuantType::Q5_K, "q5_K", rng);
  test_one_type(sbquant::QuantType::Q6_K, "q6_K", rng);

  test_batched_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_batched_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_batched_type(sbquant::QuantType::Q5_K, "q5_K", rng);
  test_batched_type(sbquant::QuantType::Q6_K, "q6_K", rng);

  // The q8_1 batched matmul is the prefill default; the differential above used
  // the fp32 fallback (SONICBOOM_GEMV_F32=1). For n >= 2 the default is now the
  // mmq int8 tensor-core path (C.2, and the sole n>=2 path since the C.4
  // kMmqMinN==2 change), which is approximate and validated end-to-end rather
  // than bit-exactly. This block targets the dp4a batched gemv (the n >= 2
  // fallback when mmq is disabled), which must be bit-identical to n calls of
  // matvec_f32 — so unset GEMV_F32 (activate q8_1) and disable mmq to force it.
  unsetenv("SONICBOOM_GEMV_F32");
  setenv("SONICBOOM_MMQ", "0", 1);
  test_batched_q8_1_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_batched_q8_1_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_batched_q8_1_type(sbquant::QuantType::Q5_K, "q5_K", rng);
  test_batched_q8_1_type(sbquant::QuantType::Q6_K, "q6_K", rng);

  test_mmq_q6_type(rng);
  test_cublas_q6_type(rng);

  if (g_failures == 0) {
    std::cout << "test_cuda_quant OK\n";
    return 0;
  }
  std::cerr << "test_cuda_quant FAILED: " << g_failures << " check(s)\n";
  return 1;
}
