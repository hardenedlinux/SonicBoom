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

// GGUF v3 file reader.
//
// This is a pure file-format reader: it understands the GGUF on-disk layout
// (header, key/value metadata, tensor directory, alignment, offsets) and the
// ggml type size table, and exposes each tensor's raw — possibly quantized —
// bytes by offset + computed size. It performs NO dequantization, builds NO
// compute graph, and does NO scheduling or device work.
//
// Dependency boundary: this header and its implementation have no dependency
// on ggml, llama.cpp, or any device backend. The format constants and type
// table below are re-declared by reference to the public GGUF v3 specification
// (ggml/docs/gguf.md); only the *format* is mirrored, never its execution code.

#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::gguf {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

enum class ErrorCategory {
  Io,         // could not open / read the backing file
  Format,     // bad magic, unsupported version, malformed structure
  Truncated,  // a read ran past the end of the buffer
};

struct Error {
  ErrorCategory category = ErrorCategory::Format;
  std::string message;
  // Byte offset (from the start of the backing buffer) where the failure was
  // detected, when known.
  std::optional<uint64_t> offset;
};

// ---------------------------------------------------------------------------
// ggml type table
// ---------------------------------------------------------------------------

struct GgmlTypeInfo {
  uint32_t id = 0;
  std::string_view name = "unknown";
  uint32_t block_size = 0;       // elements per block (1 for scalar types)
  uint64_t bytes_per_block = 0;  // bytes per block (== element size for scalars)
};

// Size table entry for a known ggml type id, or nullopt for unknown ids.
std::optional<GgmlTypeInfo> ggml_type_info(uint32_t id);

// Human-readable type name, or "unknown" for ids outside the table.
std::string_view ggml_type_name(uint32_t id);

// Byte size of `n_elements` stored with the given ggml type, or nullopt when
// the type is unknown. Quantized types round the element count up to the next
// block boundary.
std::optional<uint64_t> ggml_type_size(uint32_t id, uint64_t n_elements);

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

// GGUF metadata value type ids (uint32 on disk).
enum class ValueType : uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

// A metadata value. Scalars are widened losslessly (all integers into `u`/`i`,
// floats into `f`); arrays keep their element type and a vector of element
// values. String elements are supported (e.g. tokenizer vocabularies).
struct Value {
  ValueType type = ValueType::Int32;

  bool b = false;         // Bool
  int64_t i = 0;          // Int8/16/32/64
  uint64_t u = 0;         // UInt8/16/32/64
  double f = 0.0;         // Float32/64
  std::string s;          // String

  // Valid only when type == Array.
  ValueType elem_type = ValueType::Int32;
  std::vector<Value> elems;
};

// ---------------------------------------------------------------------------
// Tensor directory
// ---------------------------------------------------------------------------

struct TensorInfo {
  std::string name;
  uint32_t type = 0;                 // ggml type id
  std::vector<uint64_t> dims;        // GGUF order (fastest-moving first)
  uint64_t n_elements = 0;           // product of dims
  uint64_t offset = 0;               // byte offset relative to the data section
  std::optional<uint64_t> byte_size; // computed; nullopt for unknown types
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

class Reader {
public:
  // Load a GGUF file, reading the whole file into memory.
  static std::expected<Reader, Error> load(const std::filesystem::path& path);

  // Parse an in-memory buffer. The buffer is borrowed: it must outlive the
  // Reader (spans returned by tensor_data() point into it).
  static std::expected<Reader, Error> from_bytes(std::span<const std::byte> data);
  static std::expected<Reader, Error> from_bytes(const void* data, size_t size);

  Reader(Reader&&) noexcept = default;
  Reader& operator=(Reader&&) noexcept = default;
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  ~Reader() = default;

  // --- header --------------------------------------------------------------
  uint32_t version() const noexcept { return version_; }
  uint64_t tensor_count() const noexcept { return tensors_.size(); }
  uint64_t kv_count() const noexcept { return metadata_.size(); }
  uint64_t alignment() const noexcept { return alignment_; }

  // --- metadata ------------------------------------------------------------
  const std::vector<std::pair<std::string, Value>>& metadata() const noexcept {
    return metadata_;
  }
  // Lookup a metadata key; nullptr if absent.
  const Value* find(std::string_view key) const;

  // --- tensors -------------------------------------------------------------
  const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
  // Lookup a tensor by name; nullptr if absent.
  const TensorInfo* tensor(std::string_view name) const;

  // Raw tensor bytes as a span into the backing buffer. Returns nullopt when
  // the tensor's type size is unknown or its [offset, offset + byte_size)
  // range falls outside the data section.
  std::optional<std::span<const std::byte>> tensor_data(const TensorInfo& t) const;
  std::optional<std::span<const std::byte>> tensor_data(std::string_view name) const;

  // The data section as a single span (all tensor bytes live within it).
  std::span<const std::byte> data_section() const noexcept;

private:
  Reader() = default;

  // Shared parse entry point used by both load (file) and from_bytes (memory).
  static std::expected<Reader, Error> parse(std::span<const std::byte> data);

  std::vector<std::byte> owned_;        // file-backed storage (load path)
  std::span<const std::byte> data_;     // view into owned_ or the caller buffer
  uint64_t data_section_offset_ = 0;    // where tensor data begins (aligned)
  uint64_t alignment_ = 32;
  uint32_t version_ = 0;
  std::vector<std::pair<std::string, Value>> metadata_;
  std::vector<TensorInfo> tensors_;
};

} // namespace sonicboom::gguf
