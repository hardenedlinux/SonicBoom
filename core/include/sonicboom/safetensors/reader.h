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

// safetensors reader (https://github.com/huggingface/safetensors).
//
// This is a pure file-format reader: it parses the 8-byte little-endian header
// length, the JSON header (one entry per tensor: dtype, shape, byte range) and
// validates the directory against the data section. Tensor bytes are exposed
// as spans over a read-only mmap (load) or over a caller-owned buffer
// (from_memory). It performs NO dtype conversion, NO dequantization and NO
// device work.
//
// All tensor offsets in the file are relative to the start of the data section
// (8 + header length), and every entry's byte range must be disjoint and
// match shape * dtype size, per the format specification.
//
// ShardedReader follows a Hugging Face index file
// ("model.safetensors.index.json", weight_map: tensor -> shard filename) and
// opens each shard with Reader, so a multi-shard checkpoint is one lookup
// namespace.

#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::safetensors {

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

enum class ErrorCategory {
  Io,         // could not open / map / read the backing file
  Format,     // bad header length, malformed JSON, bad directory
  Truncated,  // a range runs past the end of the data section
};

struct Error {
  ErrorCategory category = ErrorCategory::Format;
  std::string message;
  std::optional<uint64_t> offset;  // byte offset in the file, when known
};

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

enum class DType {
  F64,
  F32,
  F16,
  BF16,
  I64,
  I32,
  I16,
  I8,
  U8,
  U16,
  U32,
  U64,
  BOOL,
  F8_E4M3,
  F8_E5M2,
  F8_E8M0,
};

// Bytes per element, or nullopt for an unrecognized dtype name.
std::optional<uint64_t> dtype_size(DType dtype);

// Map a safetensors dtype string ("F32", "BF16", …) to the enum; nullopt when
// the name is not in the table. Names outside the table are rejected rather
// than guessed, so a caller cannot silently mis-size them.
std::optional<DType> dtype_from_name(std::string_view name);

// The on-disk spelling, or "unknown".
std::string_view dtype_name(DType dtype);

// ---------------------------------------------------------------------------
// Tensor directory
// ---------------------------------------------------------------------------

struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<uint64_t> shape;  // row-major, as stored in the header
  uint64_t n_elements = 0;      // product of shape
  uint64_t begin = 0;           // byte offset relative to the data section
  uint64_t end = 0;             // exclusive
};

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

class Reader {
public:
  // Map the file read-only and parse its header. The mapping stays alive for
  // the lifetime of the Reader; tensor_data() spans point into it.
  static std::expected<Reader, Error> load(const std::filesystem::path& path);

  // Parse an in-memory safetensors image. The buffer is borrowed: it must
  // outlive the Reader.
  static std::expected<Reader, Error> from_memory(std::span<const std::byte> data);
  static std::expected<Reader, Error> from_memory(const void* data, size_t size);

  Reader(Reader&&) noexcept;
  Reader& operator=(Reader&&) noexcept;
  Reader(const Reader&) = delete;
  Reader& operator=(const Reader&) = delete;
  ~Reader();

  // --- directory -----------------------------------------------------------
  const std::vector<TensorInfo>& tensors() const noexcept { return tensors_; }
  const std::vector<std::pair<std::string, std::string>>& metadata() const noexcept {
    return metadata_;
  }
  // Lookup a tensor by name; nullptr if absent.
  const TensorInfo* tensor(std::string_view name) const;

  // Raw tensor bytes as a span into the data section.
  std::optional<std::span<const std::byte>> tensor_data(const TensorInfo& t) const;
  std::optional<std::span<const std::byte>> tensor_data(std::string_view name) const;

  // The whole data section (all tensor bytes live within it).
  std::span<const std::byte> data_section() const noexcept { return data_; }

  const std::filesystem::path& path() const noexcept { return path_; }

private:
  Reader() = default;

  static std::expected<Reader, Error> parse(std::span<const std::byte> file,
                                            const std::filesystem::path& path);
  void reset() noexcept;

  int fd_ = -1;
  const std::byte* map_ = nullptr;
  size_t map_size_ = 0;
  std::span<const std::byte> data_;  // data section: owned mapping or caller buffer
  std::filesystem::path path_;
  std::vector<std::pair<std::string, std::string>> metadata_;
  std::vector<TensorInfo> tensors_;
};

// ---------------------------------------------------------------------------
// ShardedReader
// ---------------------------------------------------------------------------

struct ShardedTensor {
  TensorInfo info;
  size_t shard = 0;  // index into shards()
};

class ShardedReader {
public:
  // Open every shard named by a Hugging Face `.safetensors.index.json` file.
  // Shard paths are resolved relative to the index file's directory.
  static std::expected<ShardedReader, Error> load(const std::filesystem::path& index_path);

  ShardedReader(ShardedReader&&) noexcept = default;
  ShardedReader& operator=(ShardedReader&&) noexcept = default;
  ShardedReader(const ShardedReader&) = delete;
  ShardedReader& operator=(const ShardedReader&) = delete;
  ~ShardedReader() = default;

  const std::vector<Reader>& shards() const noexcept { return shards_; }
  const std::vector<ShardedTensor>& tensors() const noexcept { return tensors_; }
  // Lookup across all shards; nullptr if absent.
  const ShardedTensor* tensor(std::string_view name) const;
  std::optional<std::span<const std::byte>> tensor_data(std::string_view name) const;

private:
  ShardedReader() = default;
  std::vector<Reader> shards_;
  std::vector<ShardedTensor> tensors_;
};

} // namespace sonicboom::safetensors
