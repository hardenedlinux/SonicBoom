// Differential test for the reference quantized matvec (Phase 3).
//
// Strategy: build a quantized weight with random-but-finite block bytes, run
// matvec_f32, and compare bit-exactly against a "dequantize the whole weight to
// f32, then naive dot product" reference. Both paths share the dequant kernels,
// so agreement here proves the block iteration, row/col mapping, and partial
// last-block handling — not the (already separately tested) dequant arithmetic.

#include <sonicboom/quant/dequant.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <cstdint>
#include <cstring>
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

void push_u16(std::vector<std::byte>& b, uint16_t v) {
  b.push_back(std::byte(v & 0xFF));
  b.push_back(std::byte(v >> 8));
}

// Build a random quantized weight of GGUF shape [cols, rows] with finite block
// scales. The returned byte buffer owns the payload; a QuantizedTensor borrows
// it. cols is the contiguous (inner) dimension.
std::vector<std::byte> build_weight(sbquant::QuantType type, uint64_t rows,
                                    uint64_t cols, std::mt19937& rng) {
  const sbquant::QuantTypeInfo* info = sbquant::quant_type_info(type);
  const uint64_t blocks_per_row =
      (cols + info->block_size - 1) / info->block_size;
  const uint64_t n_blocks = rows * blocks_per_row;
  const uint64_t total = n_blocks * info->bytes_per_block;

  std::vector<std::byte> bytes(total);
  for (auto& b : bytes) b = std::byte(uint8_t(rng()));

  // Overwrite the fp16 scale field(s) with finite values so no Inf/NaN sneaks
  // into the dot product.
  for (uint64_t i = 0; i < n_blocks; ++i) {
    const size_t off = i * info->bytes_per_block;
    if (type == sbquant::QuantType::Q3_K) {
      // d is the last 2 bytes of the 110-byte block.
      const uint16_t d = rand_finite_half(rng);
      bytes[off + 108] = std::byte(d & 0xFF);
      bytes[off + 109] = std::byte(d >> 8);
    } else {
      // q4_K/q5_K: d at 0, dmin at 2.
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

// Naive reference: dequantize the whole weight to f32, then plain matvec.
void naive_matvec(const sbquant::QuantizedTensor& W, std::span<const float> x,
                  std::span<float> y) {
  const uint64_t cols = W.dims()[0];
  const uint64_t rows = W.dims()[1];
  std::vector<float> wf(rows * cols);
  bool ok = sbquant::dequantize_f32(W, wf.data(), wf.size());
  if (!ok) return;
  for (uint64_t r = 0; r < rows; ++r) {
    float acc = 0.0f;
    for (uint64_t k = 0; k < cols; ++k) acc += wf[r * cols + k] * x[k];
    y[r] = acc;
  }
}

void test_matvec_one_type(sbquant::QuantType type, const char* name,
                          std::mt19937& rng) {
  const uint64_t rows = 3;
  // Multiple blocks per row and multiple rows. (The contiguous dim is a whole
  // number of blocks — the invariant ggml requires and valid() enforces.)
  const uint64_t cols = 256 * 3;  // 768 -> 3 blocks per row

  auto bytes = build_weight(type, rows, cols, rng);
  sbquant::QuantizedTensor W(type, {cols, rows}, bytes);
  if (!W.valid()) {
    check(false, "build_weight produced an invalid tensor");
    return;
  }

  std::vector<float> x(cols), y(rows), y_ref(rows);
  for (auto& v : x) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  bool ok = sbquant::matvec_f32(W, x, y);
  check(ok, "matvec_f32 returns true for a valid weight");
  naive_matvec(W, x, y_ref);

  bool exact = true;
  for (uint64_t r = 0; r < rows; ++r)
    if (y[r] != y_ref[r]) exact = false;
  check(exact, "matvec_f32 == naive reference");
}

void test_matmul_one_type(sbquant::QuantType type, std::mt19937& rng) {
  const uint64_t rows = 3;
  const uint64_t cols = 256 * 2;  // 2 blocks per row
  const uint64_t n = 4;           // batch columns (prefill)

  auto bytes = build_weight(type, rows, cols, rng);
  sbquant::QuantizedTensor W(type, {cols, rows}, bytes);

  std::vector<float> X(cols * n), Y(rows * n), Y_ref(rows * n);
  for (auto& v : X) v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  bool ok = sbquant::matmul_f32(W, X, Y, n);
  check(ok, "matmul_f32 returns true for a valid weight");

  // Naive reference: dequantize the whole weight, then plain matmul.
  const uint64_t total = rows * cols;
  std::vector<float> wf(total);
  sbquant::dequantize_f32(W, wf.data(), wf.size());
  for (uint64_t r = 0; r < rows; ++r) {
    for (uint64_t c = 0; c < n; ++c) {
      float acc = 0.0f;
      for (uint64_t k = 0; k < cols; ++k) acc += wf[r * cols + k] * X[k * n + c];
      Y_ref[r * n + c] = acc;
    }
  }

  bool exact = (Y == Y_ref);
  check(exact, "matmul_f32 == naive reference");
}

void test_validation() {
  // Non-2-D weight is rejected.
  {
    std::vector<std::byte> bytes(144);
    sbquant::QuantizedTensor W(sbquant::QuantType::Q4_K, {256}, bytes);
    float x[256]{}, y[1]{};
    check(!sbquant::matvec_f32(W, x, y), "matvec_f32 rejects 1-D weight");
  }
  // Truncated payload (invalid) is rejected.
  {
    std::vector<std::byte> bytes(100);  // too few for a 144-byte q4_K block
    sbquant::QuantizedTensor W(sbquant::QuantType::Q4_K, {256, 1}, bytes);
    float x[256]{}, y[1]{};
    check(!W.valid() && !sbquant::matvec_f32(W, x, y),
          "matvec_f32 rejects invalid weight");
  }
  // Too-short x/y spans are rejected.
  {
    std::vector<std::byte> bytes(144);
    sbquant::QuantizedTensor W(sbquant::QuantType::Q4_K, {256, 1}, bytes);
    float x[255]{}, y[1]{};
    check(!sbquant::matvec_f32(W, x, y), "matvec_f32 rejects short x");
  }
  // A contiguous dim that is not a whole number of blocks is invalid (the
  // ggml block-layout invariant), so matvec refuses it too.
  {
    std::vector<std::byte> bytes(144);  // one block's worth of bytes
    sbquant::QuantizedTensor W(sbquant::QuantType::Q4_K, {100, 1}, bytes);
    float x[100]{}, y[1]{};
    check(!W.valid() && !sbquant::matvec_f32(W, x, y),
          "matvec_f32 rejects non-multiple-of-256 contiguous dim");
  }
}

} // namespace

int main() {
  std::mt19937 rng(0xABCD);
  test_matvec_one_type(sbquant::QuantType::Q3_K, "q3_K", rng);
  test_matvec_one_type(sbquant::QuantType::Q4_K, "q4_K", rng);
  test_matvec_one_type(sbquant::QuantType::Q5_K, "q5_K", rng);
  test_matmul_one_type(sbquant::QuantType::Q3_K, rng);
  test_matmul_one_type(sbquant::QuantType::Q4_K, rng);
  test_matmul_one_type(sbquant::QuantType::Q5_K, rng);
  test_validation();

  if (g_failures == 0) {
    std::cout << "test_quantized_matmul OK\n";
    return 0;
  }
  std::cerr << "test_quantized_matmul FAILED: " << g_failures << " check(s)\n";
  return 1;
}
