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

// CUDA single-row gather differential test (Phase 3.2 B5).
//
// Builds synthetic Q3_K/Q4_K/Q5_K embedding weights (random-but-finite block
// bytes, the same construction as test_cuda_quant.cpp) and, for random rows,
// compares the device gather (quant::cuda::gather_f32, the host-in/out wrapper
// around gather_f32_dev) against the CPU reference quant::dequantize_row_f32.
// The gather kernels reproduce the dequant reconstruction element-for-element
// (one thread per element, no dot product), so agreement is bit-exact, not
// tolerance.
//
// Skips cleanly (exit 0) when no CUDA device is present, mirroring test_cuda.

#include <sonicboom/quant/dequant.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
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

// Build a synthetic 2-D weight ([cols, rows]) of random K-quant blocks with
// finite (d, dmin) halves, so subnormal/negative scales are exercised but Inf/
// NaN never are. Identical to test_cuda_quant.cpp's build_weight.
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

void test_gather_type(sbquant::QuantType type, const char* name,
                      uint64_t rows, uint64_t cols, std::mt19937& rng) {
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

  const uint64_t width = cols;
  std::vector<float> got(width), ref(width);

  // Boundary rows plus a spread of random rows.
  std::vector<uint64_t> rows_to_test;
  rows_to_test.push_back(0);
  rows_to_test.push_back(rows - 1);
  for (int i = 0; i < 8; ++i)
    rows_to_test.push_back(uint64_t(rng()) % rows);

  for (uint64_t token : rows_to_test) {
    bool cpu_ok = sbquant::dequantize_row_f32(W, token, ref.data(), width);
    check(cpu_ok, "dequantize_row_f32 returns true");
    if (!cpu_ok) continue;

    bool cuda_ok = sbquant::cuda::gather_f32(W, int64_t(token), got);
    check(cuda_ok, "cuda::gather_f32 returns true");
    if (!cuda_ok) continue;

    check(std::memcmp(got.data(), ref.data(), width * sizeof(float)) == 0,
          "cuda::gather_f32 bit-exact vs dequantize_row_f32");
  }
}

} // namespace

int main() {
  if (!sbquant::cuda::available()) {
    std::cout << "test_cuda_gather SKIP (no CUDA device)\n";
    return 0;
  }

  std::mt19937 rng(0x6A7E);
  // 5 blocks per row: enough to cross super-block boundaries and exercise the
  // scale/min unpacking in every block without being the full model width.
  test_gather_type(sbquant::QuantType::Q3_K, "q3_K", /*rows=*/64, /*cols=*/256 * 5, rng);
  test_gather_type(sbquant::QuantType::Q4_K, "q4_K", /*rows=*/64, /*cols=*/256 * 5, rng);
  test_gather_type(sbquant::QuantType::Q5_K, "q5_K", /*rows=*/64, /*cols=*/256 * 5, rng);

  // A larger width (40 blocks/row, matching the big FFN down-projection) to
  // confirm the bit-exactness holds independent of row span.
  test_gather_type(sbquant::QuantType::Q4_K, "q4_K@40", /*rows=*/32, /*cols=*/256 * 40, rng);

  if (g_failures == 0) {
    std::cout << "test_cuda_gather OK\n";
    return 0;
  }
  std::cerr << "test_cuda_gather FAILED: " << g_failures << " check(s)\n";
  return 1;
}
