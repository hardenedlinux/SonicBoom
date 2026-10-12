// mmq_minn_bench — microbenchmark of the quant matmul mmq-vs-dp4a crossover at
// small batch n (Phase C Step 4). kMmqMinN = 2 routes every n >= 2 through the
// int8 tensor-core mmq path so a prefill position's algorithm never depends on
// the segment size it lands in (prefix-cache batch-independence). This measures
// what that correctness-forced threshold costs: for each small n it times the
// mmq path against the dp4a batched gemv a higher threshold would use.
//
// Build: tools/baseline/build_mmq_minn_bench.sh
// Run:   tools/baseline/mmq_minn_bench

#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <span>
#include <vector>

namespace sbquant = sonicboom::quant;

namespace {

uint16_t rand_finite_half(std::mt19937& rng) {
  uint16_t h;
  do { h = uint16_t(rng()); } while (((h >> 10) & 0x1F) == 0x1F);
  return h;
}

// Random quantized weight [cols, rows] with finite block scales (the same
// construction as tests/core/test_quantized_matmul.cpp).
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

const char* type_name(sbquant::QuantType t) {
  switch (t) {
    case sbquant::QuantType::Q3_K: return "Q3_K";
    case sbquant::QuantType::Q4_K: return "Q4_K";
    case sbquant::QuantType::Q5_K: return "Q5_K";
  }
  return "?";
}

double time_one(const sbquant::QuantizedTensor& W, uint64_t cols, uint64_t rows,
                uint64_t n, int reps) {
  std::vector<float> X(cols * n), Y(rows * n);
  std::mt19937 rng(7);
  for (auto& v : X)
    v = std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng);

  sbquant::cuda::matmul_f32(W, X, Y, n);  // warmup (also uploads W)
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) sbquant::cuda::matmul_f32(W, X, Y, n);
  auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

} // namespace

int main() {
  if (!sbquant::cuda::available()) {
    std::cout << "mmq_minn_bench SKIP (no CUDA device)\n";
    return 0;
  }

  // Gemma-4 FFN gate shape: cols is the contiguous dim (d = 2560), rows = 10240.
  const uint64_t cols = 2560, rows = 10240;
  const int reps = 50;
  const uint64_t ns[] = {2, 3, 4, 6, 8, 16, 32, 64, 128};

  for (sbquant::QuantType type :
       {sbquant::QuantType::Q3_K, sbquant::QuantType::Q4_K,
        sbquant::QuantType::Q5_K}) {
    std::mt19937 rng(0xCAFE);
    auto bytes = build_weight(type, rows, cols, rng);
    sbquant::QuantizedTensor W(type, {cols, rows}, bytes);
    if (!W.valid()) {
      std::cerr << "build_weight produced an invalid tensor\n";
      return 1;
    }

    std::printf("== %s [%llu x %llu] ==\n", type_name(type),
                (unsigned long long)cols, (unsigned long long)rows);
    std::printf("   n    mmq_ms   dp4a_ms   ratio(dp4a/mmq)\n");
    for (uint64_t n : ns) {
      unsetenv("SONICBOOM_MMQ");
      unsetenv("SONICBOOM_CUBLAS");
      const double mmq = time_one(W, cols, rows, n, reps);

      setenv("SONICBOOM_MMQ", "0", 1);
      setenv("SONICBOOM_CUBLAS", "0", 1);  // force the dp4a batched gemv
      const double dp4a = time_one(W, cols, rows, n, reps);
      unsetenv("SONICBOOM_MMQ");
      unsetenv("SONICBOOM_CUBLAS");

      std::printf(" %4llu  %7.3f  %7.3f   %6.2fx\n", (unsigned long long)n, mmq,
                  dp4a, dp4a / mmq);
    }
  }
  return 0;
}
