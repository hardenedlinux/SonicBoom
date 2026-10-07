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

// Reference dequantization tests for the ggml Q3_K / Q4_K / Q5_K formats.
//
// Coverage:
//   - known-value reconstruction (hand-computed anchors)
//   - block-boundary / multi-block dequantization
//   - scale/min edge cases and signed-value reconstruction (q3_K hmask)
//   - invalid / truncated / oversized buffers (QuantizedTensor::valid(),
//     dequantize_f32 rejection)
//   - random differential against a verbatim ggml oracle (bit-exact)
//   - real-model differential against the target Gemma 4 E4B GGUF (skipped
//     cleanly when the model file is absent)

#include <sonicboom/quant/dequant.h>
#include <sonicboom/quant/quantized_tensor.h>

#include <sonicboom/gguf/reader.h>

#include "ggml_oracle.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace sbquant = sonicboom::quant;
namespace sbgguf = sonicboom::gguf;
namespace oracle = ggml_oracle;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

constexpr uint16_t kFp16_1p0 = 0x3C00;
constexpr uint16_t kFp16_0p5 = 0x3800;
constexpr uint16_t kFp16_0p0 = 0x0000;

void push_u16(std::vector<std::byte>& b, uint16_t v) {
  b.push_back(std::byte(v & 0xFF));
  b.push_back(std::byte(v >> 8));
}

// Build one serialized q4_K block (144 bytes).
std::vector<std::byte> make_q4k(uint16_t d, uint16_t dmin,
                                const std::array<uint8_t, 12>& scales,
                                const std::array<uint8_t, 128>& qs) {
  std::vector<std::byte> b;
  b.reserve(144);
  push_u16(b, d);
  push_u16(b, dmin);
  for (uint8_t s : scales) b.push_back(std::byte(s));
  for (uint8_t q : qs) b.push_back(std::byte(q));
  return b;
}

// Build one serialized q5_K block (176 bytes).
std::vector<std::byte> make_q5k(uint16_t d, uint16_t dmin,
                                const std::array<uint8_t, 12>& scales,
                                const std::array<uint8_t, 32>& qh,
                                const std::array<uint8_t, 128>& qs) {
  std::vector<std::byte> b;
  b.reserve(176);
  push_u16(b, d);
  push_u16(b, dmin);
  for (uint8_t s : scales) b.push_back(std::byte(s));
  for (uint8_t h : qh) b.push_back(std::byte(h));
  for (uint8_t q : qs) b.push_back(std::byte(q));
  return b;
}

// Build one serialized q3_K block (110 bytes): hmask[32] qs[64] scales[12] d[2].
std::vector<std::byte> make_q3k(uint16_t d,
                                const std::array<uint8_t, 32>& hmask,
                                const std::array<uint8_t, 64>& qs,
                                const std::array<uint8_t, 12>& scales) {
  std::vector<std::byte> b;
  b.reserve(110);
  for (uint8_t h : hmask) b.push_back(std::byte(h));
  for (uint8_t q : qs) b.push_back(std::byte(q));
  for (uint8_t s : scales) b.push_back(std::byte(s));
  push_u16(b, d);
  return b;
}

// --- known-value tests ------------------------------------------------------

void test_q4k_known_values() {
  // d=1.0, dmin=0.0, all scales -> sc=1 / min=0.
  std::array<uint8_t, 12> scales{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};

  // Case 1: qs all zero -> all 256 elements 0.0.
  {
    std::array<uint8_t, 128> qs{};
    auto block = make_q4k(kFp16_1p0, kFp16_0p0, scales, qs);
    std::vector<float> out(256, 123.0f);
    sbquant::dequantize_q4_K(block, 1, out.data());
    bool all_zero = true;
    for (float v : out)
      if (v != 0.0f) all_zero = false;
    check(all_zero, "q4_K: zero quants -> all 0.0");
  }

  // Case 2: qs[0]=0x3F -> element 0 = 15, element 32 = 3, rest 0.
  {
    std::array<uint8_t, 128> qs{};
    qs[0] = 0x3F;
    auto block = make_q4k(kFp16_1p0, kFp16_0p0, scales, qs);
    std::vector<float> out(256);
    sbquant::dequantize_q4_K(block, 1, out.data());
    check(out[0] == 15.0f && out[32] == 3.0f, "q4_K: nibble reconstruction");
    bool rest_zero = true;
    for (int i = 0; i < 256; ++i)
      if (i != 0 && i != 32 && out[i] != 0.0f) rest_zero = false;
    check(rest_zero, "q4_K: remaining elements zero");
  }

  // Case 3: half value 0.5 -> element 0 = 0.5 * 15 = 7.5.
  {
    std::array<uint8_t, 128> qs{};
    qs[0] = 0x0F;  // low nibble 15
    auto block = make_q4k(kFp16_0p5, kFp16_0p0, scales, qs);
    std::vector<float> out(256);
    sbquant::dequantize_q4_K(block, 1, out.data());
    check(out[0] == 7.5f, "q4_K: half scale 0.5 -> 7.5");
  }
}

void test_q5k_known_values() {
  std::array<uint8_t, 12> scales{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::array<uint8_t, 32> qh{};
  std::array<uint8_t, 128> qs{};

  // Case 1: all zero -> all 0.0.
  {
    auto block = make_q5k(kFp16_1p0, kFp16_0p0, scales, qh, qs);
    std::vector<float> out(256, 5.0f);
    sbquant::dequantize_q5_K(block, 1, out.data());
    bool all_zero = true;
    for (float v : out)
      if (v != 0.0f) all_zero = false;
    check(all_zero, "q5_K: zero quants -> all 0.0");
  }

  // Case 2: qs[0]=0x0F (low 15), qh[0]=0x01 -> element 0 = 31, element 32 = 0.
  {
    std::array<uint8_t, 32> qh2{};
    std::array<uint8_t, 128> qs2{};
    qs2[0] = 0x0F;
    qh2[0] = 0x01;
    auto block = make_q5k(kFp16_1p0, kFp16_0p0, scales, qh2, qs2);
    std::vector<float> out(256);
    sbquant::dequantize_q5_K(block, 1, out.data());
    check(out[0] == 31.0f && out[32] == 0.0f, "q5_K: high-bit reconstruction");
  }
}

void test_q3k_known_values() {
  // d=1.0, scales raw all zero -> unpacked scales all 0 -> dl = 1.0*(0-32) = -32.
  std::array<uint8_t, 12> scales{};  // all zero
  std::array<uint8_t, 32> hmask_lo{};  // all 0 -> subtract 4
  std::array<uint8_t, 32> hmask_hi{};  // all 0xFF -> no subtract
  hmask_hi.fill(0xFF);
  std::array<uint8_t, 64> qs_hi{};  // all 0xFF -> value 3
  qs_hi.fill(0xFF);
  std::array<uint8_t, 64> qs_lo{};  // all 0 -> value 0

  // Case 1: hmask=0, qs=0xFF -> dl=-32, value=3-4=-1 -> 32.0.
  {
    auto block = make_q3k(kFp16_1p0, hmask_lo, qs_hi, scales);
    std::vector<float> out(256);
    sbquant::dequantize_q3_K(block, 1, out.data());
    bool all_32 = true;
    for (float v : out)
      if (v != 32.0f) all_32 = false;
    check(all_32, "q3_K: hmask=0 signed reconstruction -> 32.0");
  }

  // Case 2: hmask=0xFF, qs=0xFF -> value=3 -> -32*3 = -96.0.
  {
    auto block = make_q3k(kFp16_1p0, hmask_hi, qs_hi, scales);
    std::vector<float> out(256);
    sbquant::dequantize_q3_K(block, 1, out.data());
    bool all_neg96 = true;
    for (float v : out)
      if (v != -96.0f) all_neg96 = false;
    check(all_neg96, "q3_K: hmask=1 unsigned reconstruction -> -96.0");
  }

  // Case 3: hmask=0xFF, qs=0x00 -> value 0 -> 0.0.
  {
    auto block = make_q3k(kFp16_1p0, hmask_hi, qs_lo, scales);
    std::vector<float> out(256, 7.0f);
    sbquant::dequantize_q3_K(block, 1, out.data());
    bool all_zero = true;
    for (float v : out)
      if (v != 0.0f) all_zero = false;
    check(all_zero, "q3_K: zero quants -> 0.0");
  }
}

// --- negative-subnormal scale regression -----------------------------------

// A block scale (d / dmin) stored as a negative *subnormal* half must keep its
// sign. The oracle's half->float once dropped the sign for subnormals, which
// silently flipped every dequantized element for such blocks (random data
// produces them; real GGUF scales are positive, so the model diff missed it).
void test_negative_subnormal_scale() {
  // 0x8278: sign=1, exp=0, mant=0x278 -> -632 * 2^-24 = -0x1.3cp-15.
  constexpr uint16_t kFp16_neg_sub = 0x8278;

  std::array<uint8_t, 12> scales{};  // all 0
  scales[0] = 1;                     // sub-block 0: sc=1, m=0
  std::array<uint8_t, 128> qs{};
  qs[0] = 0x01;  // low nibble 1, high nibble 0

  auto block = make_q4k(kFp16_neg_sub, kFp16_0p0, scales, qs);
  float out[256];
  sbquant::dequantize_q4_K(block, 1, out);

  // d * sc * (q&0xF) - dmin*m = -0x1.3cp-15 * 1 * 1 - 0.
  check(out[0] < 0.0f, "q4_K: negative subnormal d keeps sign");
  check(out[0] == -0x1.3cp-15f, "q4_K: negative subnormal d exact value");
}

// --- multi-block / boundary tests ------------------------------------------

void test_multi_block() {
  std::array<uint8_t, 12> scales{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::array<uint8_t, 128> qs{};
  qs[0] = 0x0F;  // low nibble 15 in block 0
  auto b0 = make_q4k(kFp16_1p0, kFp16_0p0, scales, qs);

  std::array<uint8_t, 128> qs2{};
  qs2[0] = 0x08;  // low nibble 8 in block 1
  auto b1 = make_q4k(kFp16_1p0, kFp16_0p0, scales, qs2);

  std::vector<std::byte> both;
  both.insert(both.end(), b0.begin(), b0.end());
  both.insert(both.end(), b1.begin(), b1.end());

  std::vector<float> out(512);
  sbquant::dequantize_q4_K(both, 2, out.data());
  check(out[0] == 15.0f, "q4_K multi-block: block 0 element 0");
  check(out[256] == 8.0f, "q4_K multi-block: block 1 element 0 (boundary)");
}

// --- single-row dequantization ---------------------------------------------

void test_dequantize_row() {
  std::array<uint8_t, 12> scales{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};

  // Four distinct q4_K blocks: rows 0 and 1 each span two blocks (width 512).
  auto block_with = [&](uint8_t q0) {
    std::array<uint8_t, 128> qs{};
    qs[0] = q0;
    return make_q4k(kFp16_1p0, kFp16_0p0, scales, qs);
  };
  auto b0 = block_with(0x01);  // element 0 = 1
  auto b1 = block_with(0x02);  // element 0 = 2
  auto b2 = block_with(0x03);  // element 0 = 3
  auto b3 = block_with(0x04);  // element 0 = 4

  std::vector<std::byte> bytes;
  for (auto* b : {&b0, &b1, &b2, &b3}) bytes.insert(bytes.end(), b->begin(), b->end());

  sbquant::QuantizedTensor W(sbquant::QuantType::Q4_K, {512, 2}, bytes);
  check(W.valid(), "row dequant: 2-D tensor valid");

  std::vector<float> full(1024);
  check(sbquant::dequantize_f32(W, full.data(), 1024), "row dequant: full dequant ok");

  std::vector<float> row(512, -1.0f);
  check(sbquant::dequantize_row_f32(W, 0, row.data(), 512), "row dequant: row 0 ok");
  check(row[0] == 1.0f && row[256] == 2.0f, "row dequant: row 0 block values");
  check(row == std::vector<float>(full.begin(), full.begin() + 512),
        "row dequant: row 0 == full[0..512)");

  check(sbquant::dequantize_row_f32(W, 1, row.data(), 512), "row dequant: row 1 ok");
  check(row[0] == 3.0f && row[256] == 4.0f, "row dequant: row 1 block values");
  check(row == std::vector<float>(full.begin() + 512, full.end()),
        "row dequant: row 1 == full[512..1024)");

  // Validation: out-of-range row, 1-D tensor, and short output all rejected.
  check(!sbquant::dequantize_row_f32(W, 2, row.data(), 512), "row dequant: rejects row >= n_rows");
  check(!sbquant::dequantize_row_f32(W, 0, row.data(), 511), "row dequant: rejects short output");
  {
    sbquant::QuantizedTensor flat(sbquant::QuantType::Q4_K, {512}, bytes);
    check(!sbquant::dequantize_row_f32(flat, 0, row.data(), 512),
          "row dequant: rejects 1-D tensor");
  }
}

// --- invalid / truncated / oversized ---------------------------------------

void test_validation() {
  std::array<uint8_t, 12> scales{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1};
  std::array<uint8_t, 128> qs{};
  auto good = make_q4k(kFp16_1p0, kFp16_0p0, scales, qs);  // 144 bytes

  // Exact-size tensor is valid.
  {
    auto qt = sbquant::quantized_tensor_from_gguf(12, {256}, good);
    check(qt.has_value() && qt->valid(), "q4_K: exact 144 bytes valid");
    std::vector<float> out(256);
    check(sbquant::dequantize_f32(*qt, out.data(), 256), "q4_K: dequantize_f32 ok");
  }

  // Truncated (143 bytes) -> invalid.
  {
    std::vector<std::byte> trunc(good.begin(), good.end() - 1);
    auto qt = sbquant::quantized_tensor_from_gguf(12, {256}, trunc);
    check(qt.has_value() && !qt->valid(), "q4_K: 143 bytes invalid (truncated)");
    std::vector<float> out(256);
    check(!sbquant::dequantize_f32(*qt, out.data(), 256),
          "q4_K: dequantize_f32 rejects truncated");
  }

  // Oversized (145 bytes) -> invalid.
  {
    auto over = good;
    over.push_back(std::byte(0));
    auto qt = sbquant::quantized_tensor_from_gguf(12, {256}, over);
    check(qt.has_value() && !qt->valid(), "q4_K: 145 bytes invalid (oversized)");
  }

  // Non-multiple element count still packs ceil() blocks; 257 elements -> 2 blocks.
  {
    auto qt = sbquant::quantized_tensor_from_gguf(12, {257}, std::span<const std::byte>());
    check(qt.has_value() && qt->num_blocks() == 2, "q4_K: 257 elements -> 2 blocks");
  }

  // Insufficient output capacity rejected.
  {
    auto qt = sbquant::quantized_tensor_from_gguf(12, {256}, good);
    std::vector<float> out(255);
    check(!sbquant::dequantize_f32(*qt, out.data(), 255),
          "q4_K: out capacity < numel rejected");
  }

  // Unknown ggml type id -> nullopt bridge.
  {
    auto qt = sbquant::quantized_tensor_from_gguf(0, {256}, good);  // id 0 = f32
    check(!qt.has_value(), "quantized_tensor_from_gguf: f32 id -> nullopt");
  }
}

// --- random differential vs oracle -----------------------------------------

void test_differential_random() {
  std::mt19937 rng(0x5EED);
  auto rand_finite_half = [&]() -> uint16_t {
    uint16_t h;
    do {
      h = uint16_t(rng());
    } while (((h >> 10) & 0x1F) == 0x1F);  // avoid Inf/NaN
    return h;
  };
  auto rand_byte = [&]() -> uint8_t { return uint8_t(rng()); };

  const int kBlocks = 400;
  int mismatches = 0;

  // q4_K
  for (int i = 0; i < kBlocks; ++i) {
    std::vector<std::byte> b(144);
    uint16_t d = rand_finite_half(), dmin = rand_finite_half();
    b[0] = std::byte(d & 0xFF); b[1] = std::byte(d >> 8);
    b[2] = std::byte(dmin & 0xFF); b[3] = std::byte(dmin >> 8);
    for (int j = 4; j < 144; ++j) b[j] = std::byte(rand_byte());

    std::vector<float> mine(256), ref(256);
    sbquant::dequantize_q4_K(b, 1, mine.data());
    oracle::dequantize_row_q4_K(reinterpret_cast<const oracle::block_q4_K*>(b.data()),
                                ref.data(), 256);
    for (int j = 0; j < 256; ++j)
      if (mine[j] != ref[j]) ++mismatches;
  }

  // q5_K
  for (int i = 0; i < kBlocks; ++i) {
    std::vector<std::byte> b(176);
    uint16_t d = rand_finite_half(), dmin = rand_finite_half();
    b[0] = std::byte(d & 0xFF); b[1] = std::byte(d >> 8);
    b[2] = std::byte(dmin & 0xFF); b[3] = std::byte(dmin >> 8);
    for (int j = 4; j < 176; ++j) b[j] = std::byte(rand_byte());

    std::vector<float> mine(256), ref(256);
    sbquant::dequantize_q5_K(b, 1, mine.data());
    oracle::dequantize_row_q5_K(reinterpret_cast<const oracle::block_q5_K*>(b.data()),
                                ref.data(), 256);
    for (int j = 0; j < 256; ++j)
      if (mine[j] != ref[j]) ++mismatches;
  }

  // q3_K
  for (int i = 0; i < kBlocks; ++i) {
    std::vector<std::byte> b(110);
    for (int j = 0; j < 108; ++j) b[j] = std::byte(rand_byte());
    uint16_t d = rand_finite_half();
    b[108] = std::byte(d & 0xFF); b[109] = std::byte(d >> 8);

    std::vector<float> mine(256), ref(256);
    sbquant::dequantize_q3_K(b, 1, mine.data());
    oracle::dequantize_row_q3_K(reinterpret_cast<const oracle::block_q3_K*>(b.data()),
                                ref.data(), 256);
    for (int j = 0; j < 256; ++j)
      if (mine[j] != ref[j]) ++mismatches;
  }

  check(mismatches == 0, "differential: bit-exact vs ggml oracle (random blocks)");
  if (mismatches != 0)
    std::cerr << "  differential mismatches: " << mismatches << " / "
              << (3 * kBlocks * 256) << "\n";
}

// --- real-model differential (skipped when model absent) -------------------

void test_differential_model() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF")) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/gemma-4-E4B-it-Q3_K_M.gguf");
#endif
  candidates.push_back("models/gemma-4-E4B-it-Q3_K_M.gguf");

  std::string path;
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) { path = c; break; }
  if (path.empty()) {
    std::cout << "  (skipped: Gemma 4 model not found)\n";
    return;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    check(false, "model differential: failed to load model");
    return;
  }

  int tensors_checked = 0;
  int blocks_checked = 0;
  int mismatches = 0;

  // Dequantize the first `n_blocks` of a tensor both ways and compare.
  auto compare_tensor = [&](const sbgguf::TensorInfo& t, uint64_t n_blocks) {
    auto data = r->tensor_data(t);
    if (!data || !t.byte_size) return;
    auto qt = sbquant::quantized_tensor_from_gguf(t.type, t.dims, *data);
    if (!qt || !qt->valid()) return;
    const uint32_t bpb = qt->bytes_per_block();
    if (qt->num_blocks() < n_blocks) n_blocks = qt->num_blocks();
    if (n_blocks == 0) return;

    std::span<const std::byte> sub = data->first(n_blocks * bpb);
    std::vector<float> mine(n_blocks * 256), ref(n_blocks * 256);

    switch (qt->type()) {
      case sbquant::QuantType::Q3_K:
        sbquant::dequantize_q3_K(sub, n_blocks, mine.data());
        oracle::dequantize_row_q3_K(
            reinterpret_cast<const oracle::block_q3_K*>(sub.data()), ref.data(),
            int64_t(n_blocks * 256));
        break;
      case sbquant::QuantType::Q4_K:
        sbquant::dequantize_q4_K(sub, n_blocks, mine.data());
        oracle::dequantize_row_q4_K(
            reinterpret_cast<const oracle::block_q4_K*>(sub.data()), ref.data(),
            int64_t(n_blocks * 256));
        break;
      case sbquant::QuantType::Q5_K:
        sbquant::dequantize_q5_K(sub, n_blocks, mine.data());
        oracle::dequantize_row_q5_K(
            reinterpret_cast<const oracle::block_q5_K*>(sub.data()), ref.data(),
            int64_t(n_blocks * 256));
        break;
    }

    for (uint64_t i = 0; i < n_blocks * 256; ++i)
      if (mine[i] != ref[i]) ++mismatches;
    tensors_checked += 1;
    blocks_checked += int(n_blocks);
  };

  // Representative tensors for each format (present in the target model).
  const char* q3[] = {"token_embd.weight", "blk.0.attn_q.weight",
                      "blk.0.ffn_gate.weight"};
  const char* q4[] = {"blk.0.attn_output.weight", "per_layer_token_embd.weight"};
  const char* q5[] = {"blk.0.attn_v.weight", "blk.0.ffn_down.weight"};

  for (const char* name : q3)
    if (const auto* t = r->tensor(name)) compare_tensor(*t, 16);
  for (const char* name : q4)
    if (const auto* t = r->tensor(name)) compare_tensor(*t, 16);
  for (const char* name : q5)
    if (const auto* t = r->tensor(name)) compare_tensor(*t, 16);

  std::cout << "  model differential: " << tensors_checked << " tensors, "
            << blocks_checked << " blocks compared\n";
  check(tensors_checked >= 3, "model differential: sampled >= 3 tensors");
  check(mismatches == 0, "model differential: bit-exact vs ggml oracle");
}

} // namespace

int main() {
  test_q4k_known_values();
  test_q5k_known_values();
  test_q3k_known_values();
  test_negative_subnormal_scale();
  test_multi_block();
  test_dequantize_row();
  test_validation();
  test_differential_random();
  test_differential_model();

  if (g_failures == 0) {
    std::cout << "test_dequant OK\n";
    return 0;
  }
  std::cerr << "test_dequant FAILED: " << g_failures << " check(s)\n";
  return 1;
}
