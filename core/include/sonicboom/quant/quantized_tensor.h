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

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// Quantized weight representation for SonicBoom's native inference path.
//
// This is the backend-neutral "storage semantics" layer: it describes a
// block-quantized weight tensor (type, shape, block geometry, and a borrowed
// view over the serialized bytes) without any file-format or kernel logic. The
// GGUF Reader stays a pure format parser; this layer carries what an execution
// kernel needs to iterate blocks. Dequantization and (later) quantized matmul
// live in <sonicboom/quant/dequant.h> and downstream, never here.
//
// Dependency boundary: no ggml, llama.cpp, native-torch, MLIR, or GGUF header.
// The ggml type ids are re-declared by reference to the format only.

namespace sonicboom::quant {

// Quantized weight formats SonicBoom can dequantize/execute. A SonicBoom-owned
// semantic enum; it does not alias the ggml type table even though the ids in
// `QuantTypeInfo::ggml_type_id` match it for the gguf→quant bridge.
enum class QuantType : uint8_t {
  Q3_K,
  Q4_K,
  Q5_K,
};

struct QuantTypeInfo {
  QuantType type;
  const char* name;          // "q3_K" etc.
  uint32_t ggml_type_id;     // id in the GGUF/ggml type table (11 / 12 / 13)
  uint32_t block_size;       // elements per super-block (256 for the K-quants)
  uint32_t bytes_per_block;  // serialized bytes per block (110 / 144 / 176)
};

// Size/geometry entry for a supported format, or nullptr for unknown ids.
const QuantTypeInfo* quant_type_info(QuantType type);

// Map a ggml type id to a supported quantized format, or nullopt when the id
// is not a supported quantized type (e.g. f32/f16/bf16 or an iq_* id).
std::optional<QuantType> quant_type_from_ggml(uint32_t ggml_type_id);

// A read-only view over a quantized weight tensor's serialized bytes.
//
// Owns nothing: `data()` is a borrowed span that must outlive this object (the
// GGUF Reader or a model loader owns the backing buffer). Carries the shape and
// block geometry needed to iterate blocks without any GGUF file-format code.
//
// The serialized payload is interpreted as a packed array of
// `num_blocks()` fixed-size blocks laid out exactly as ggml serializes them;
// see dequant.cpp for the per-format layout. The contiguous (fastest-moving)
// dimension `dims()[0]` must be a whole number of blocks — ggml quantizes the
// contiguous dim and asserts it is a multiple of QK_K, so every well-formed
// K-quant tensor satisfies `dims()[0] % block_size() == 0`.
class QuantizedTensor {
 public:
  QuantizedTensor() = default;

  // `dims` in GGUF order (fastest-moving first). Only the element count (their
  // product) matters for dequantization; shape semantics are applied later by
  // the matmul/op layer.
  QuantizedTensor(QuantType type, std::vector<uint64_t> dims,
                  std::span<const std::byte> data);

  QuantType type() const noexcept { return type_; }
  const std::vector<uint64_t>& dims() const noexcept { return dims_; }

  // Logical element count = product of dims.
  uint64_t numel() const noexcept { return numel_; }

  uint32_t block_size() const noexcept;        // elements per block
  uint32_t bytes_per_block() const noexcept;
  uint64_t num_blocks() const noexcept;        // ceil(numel / block_size)
  uint64_t byte_size() const noexcept { return data_.size(); }

  std::span<const std::byte> data() const noexcept { return data_; }

  // True when the type is supported, the contiguous dim is a whole number of
  // blocks (dims()[0] % block_size() == 0), and the serialized byte count equals
  // num_blocks() * bytes_per_block() (neither truncated nor oversized).
  // Dequantization and quantized matmul refuse to run unless this holds.
  bool valid() const noexcept;

 private:
  QuantType type_ = QuantType::Q4_K;
  std::vector<uint64_t> dims_;
  uint64_t numel_ = 0;
  std::span<const std::byte> data_;
};

// Build a QuantizedTensor from a GGUF tensor's type id, dims and raw data span.
// Returns nullopt when `ggml_type_id` is not a supported quantized format.
// (Takes primitives, not gguf types, so this layer stays gguf-independent.)
std::optional<QuantizedTensor> quantized_tensor_from_gguf(
    uint32_t ggml_type_id, std::vector<uint64_t> dims,
    std::span<const std::byte> data);

} // namespace sonicboom::quant
