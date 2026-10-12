// CUDA K-quant matvec differential test.
//
// Builds synthetic Q3_K/Q4_K/Q5_K weights (random but finite block bytes, the
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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <span>
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

// Batched (prefill) matmul, fp32 differential path: Y = W @ X for n columns,
// compared against the CPU matmul_f32 reference (X/Y row-major with n innermost).
// main() sets SONICBOOM_GEMV_F32=1, so this exercises the fp32 fallback (the n×
// gather/gemv/scatter loop) — the exact dequant arithmetic, compared at the fp32
// tolerances below. The q8_1 batched path (the default) is covered separately by
// test_batched_q8_1, bit-exact against the per-column q8_1 matvec.
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
// the per-column q8_1 matvec. Unlike test_batched, this runs with
// SONICBOOM_GEMV_F32 *unset* so matmul_f32 takes the transpose -> quantize_q8_1
// -> batched warp-per-row path. That path must be bit-identical to n calls of
// matvec_f32 (same quantize_q8_1, same vec_dot, same __shfl reduction order);
// any divergence here means the transpose or the position-major q8_1 layout is
// wrong (a swapped transpose grid silently produced garbage, so this is the
// guard that catches it).
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
    for (uint64_t k = 0; k < cols; ++k) xc[k] = X[k * n + c];
    if (!sbquant::cuda::matvec_f32(W, xc, yc)) {
      check(false, "cuda::matvec_f32 (q8_1 single-col) returns true");
      return;
    }
    for (uint64_t r = 0; r < rows; ++r) Yr[r * n + c] = yc[r];
  }

  size_t ndiff = 0;
  for (uint64_t i = 0; i < rows * n; ++i)
    if (Yb[i] != Yr[i]) ++ndiff;
  if (ndiff) {
    std::cerr << "    q8_1 batched " << name << ": " << ndiff << "/" << rows * n
              << " elements differ from per-column matvec\n";
  }
  check(ndiff == 0, "q8_1 batched == per-column matvec (bit-exact)");
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
  test_one_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_one_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_one_type(sbquant::QuantType::Q5_K, "q5_K", rng);

  test_batched_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_batched_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_batched_type(sbquant::QuantType::Q5_K, "q5_K", rng);

  // The q8_1 batched path is the decode/prefill default; the differential above
  // used the fp32 fallback (SONICBOOM_GEMV_F32=1). Unset it here so the q8_1
  // batched path is exercised bit-exact against the per-column q8_1 matvec.
  unsetenv("SONICBOOM_GEMV_F32");
  test_batched_q8_1_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_batched_q8_1_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_batched_q8_1_type(sbquant::QuantType::Q5_K, "q5_K", rng);

  if (g_failures == 0) {
    std::cout << "test_cuda_quant OK\n";
    return 0;
  }
  std::cerr << "test_cuda_quant FAILED: " << g_failures << " check(s)\n";
  return 1;
}
