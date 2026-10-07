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

#include <sonicboom/quant/quantized_tensor.h>

#include <limits>

namespace sonicboom::quant {

// Supported block-quantized formats. Geometry mirrors the ggml type table:
// all K-quants use a 256-element super-block.
static constexpr QuantTypeInfo kTypeInfo[] = {
    {QuantType::Q3_K, "q3_K", 11, 256, 110},
    {QuantType::Q4_K, "q4_K", 12, 256, 144},
    {QuantType::Q5_K, "q5_K", 13, 256, 176},
};

const QuantTypeInfo* quant_type_info(QuantType type) {
  for (const auto& e : kTypeInfo)
    if (e.type == type) return &e;
  return nullptr;
}

std::optional<QuantType> quant_type_from_ggml(uint32_t ggml_type_id) {
  for (const auto& e : kTypeInfo)
    if (e.ggml_type_id == ggml_type_id) return e.type;
  return std::nullopt;
}

QuantizedTensor::QuantizedTensor(QuantType type, std::vector<uint64_t> dims,
                                 std::span<const std::byte> data)
    : type_(type), dims_(std::move(dims)), data_(data) {
  numel_ = 1;
  for (uint64_t d : dims_) {
    if (d != 0 && numel_ > std::numeric_limits<uint64_t>::max() / d) {
      // Overflow: saturate so valid() fails (byte count can never match).
      numel_ = std::numeric_limits<uint64_t>::max();
      return;
    }
    numel_ *= d;
  }
}

uint32_t QuantizedTensor::block_size() const noexcept {
  const QuantTypeInfo* info = quant_type_info(type_);
  return info ? info->block_size : 0;
}

uint32_t QuantizedTensor::bytes_per_block() const noexcept {
  const QuantTypeInfo* info = quant_type_info(type_);
  return info ? info->bytes_per_block : 0;
}

uint64_t QuantizedTensor::num_blocks() const noexcept {
  const uint32_t bs = block_size();
  if (bs == 0) return 0;
  return numel_ / bs + (numel_ % bs != 0 ? 1 : 0);
}

bool QuantizedTensor::valid() const noexcept {
  const QuantTypeInfo* info = quant_type_info(type_);
  if (!info) return false;
  if (numel_ == std::numeric_limits<uint64_t>::max()) return false;  // overflow
  // The packed block layout requires the contiguous (fastest-moving) dim to be
  // a whole number of blocks; ggml asserts the same when quantizing. This also
  // guarantees numel() is a multiple of block_size(), so the dequantizers emit
  // exactly numel() floats and the flat block count equals the per-row count.
  if (dims_.empty()) return false;
  if (dims_[0] % info->block_size != 0) return false;
  const uint64_t expected = num_blocks() * uint64_t(info->bytes_per_block);
  return data_.size() == expected;
}

std::optional<QuantizedTensor> quantized_tensor_from_gguf(
    uint32_t ggml_type_id, std::vector<uint64_t> dims,
    std::span<const std::byte> data) {
  const auto type = quant_type_from_ggml(ggml_type_id);
  if (!type) return std::nullopt;
  return QuantizedTensor(*type, std::move(dims), data);
}

} // namespace sonicboom::quant
