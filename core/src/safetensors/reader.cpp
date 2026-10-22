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

#include <sonicboom/safetensors/reader.h>

#include "json.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <system_error>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace sonicboom::safetensors {
namespace {

constexpr uint64_t kHeaderLengthBytes = 8;
// A safetensors header is a small JSON directory; a gigabyte-sized one is a
// corrupt or hostile file, not a checkpoint.
constexpr uint64_t kMaxHeaderBytes = 1ull << 30;

std::unexpected<Error> make_error(ErrorCategory category, std::string message,
                                  std::optional<uint64_t> offset = std::nullopt) {
  return std::unexpected(Error{category, std::move(message), offset});
}

uint64_t load_le64(const std::byte* p) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | uint64_t(p[i]);
  return v;
}

// Multiplies with overflow detection.
bool mul_overflow(uint64_t a, uint64_t b, uint64_t& out) {
  if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
  out = a * b;
  return true;
}

std::optional<uint64_t> entry_offset(const detail::JsonValue& v) {
  if (v.type != detail::JsonValue::Type::Number || !v.is_integer || v.integer < 0)
    return std::nullopt;
  return uint64_t(v.integer);
}

} // namespace

// ---------------------------------------------------------------------------
// dtypes
// ---------------------------------------------------------------------------

std::optional<uint64_t> dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F64:
    case DType::I64:
    case DType::U64:
      return 8;
    case DType::F32:
    case DType::I32:
    case DType::U32:
      return 4;
    case DType::F16:
    case DType::BF16:
    case DType::I16:
    case DType::U16:
      return 2;
    case DType::I8:
    case DType::U8:
    case DType::BOOL:
    case DType::F8_E4M3:
    case DType::F8_E5M2:
    case DType::F8_E8M0:
      return 1;
  }
  return std::nullopt;
}

std::optional<DType> dtype_from_name(std::string_view name) {
  if (name == "F64") return DType::F64;
  if (name == "F32") return DType::F32;
  if (name == "F16") return DType::F16;
  if (name == "BF16") return DType::BF16;
  if (name == "I64") return DType::I64;
  if (name == "I32") return DType::I32;
  if (name == "I16") return DType::I16;
  if (name == "I8") return DType::I8;
  if (name == "U8") return DType::U8;
  if (name == "U16") return DType::U16;
  if (name == "U32") return DType::U32;
  if (name == "U64") return DType::U64;
  if (name == "BOOL") return DType::BOOL;
  if (name == "F8_E4M3") return DType::F8_E4M3;
  if (name == "F8_E5M2") return DType::F8_E5M2;
  if (name == "F8_E8M0") return DType::F8_E8M0;
  return std::nullopt;
}

std::string_view dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F64: return "F64";
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::I64: return "I64";
    case DType::I32: return "I32";
    case DType::I16: return "I16";
    case DType::I8: return "I8";
    case DType::U8: return "U8";
    case DType::U16: return "U16";
    case DType::U32: return "U32";
    case DType::U64: return "U64";
    case DType::BOOL: return "BOOL";
    case DType::F8_E4M3: return "F8_E4M3";
    case DType::F8_E5M2: return "F8_E5M2";
    case DType::F8_E8M0: return "F8_E8M0";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

Reader::Reader(Reader&& other) noexcept { *this = std::move(other); }

Reader& Reader::operator=(Reader&& other) noexcept {
  if (this == &other) return *this;
  reset();
  fd_ = other.fd_;
  map_ = other.map_;
  map_size_ = other.map_size_;
  data_ = other.data_;
  path_ = std::move(other.path_);
  metadata_ = std::move(other.metadata_);
  tensors_ = std::move(other.tensors_);
  other.fd_ = -1;
  other.map_ = nullptr;
  other.map_size_ = 0;
  other.data_ = {};
  return *this;
}

Reader::~Reader() { reset(); }

void Reader::reset() noexcept {
  if (map_ != nullptr) {
    ::munmap(const_cast<std::byte*>(map_), map_size_);
    map_ = nullptr;
    map_size_ = 0;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  data_ = {};
}

std::expected<Reader, Error> Reader::load(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0)
    return make_error(ErrorCategory::Io,
                      "cannot open " + path.string() + ": " + std::system_category().message(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    const std::string why = std::system_category().message(errno);
    ::close(fd);
    return make_error(ErrorCategory::Io, "cannot stat " + path.string() + ": " + why);
  }
  if (st.st_size <= 0) {
    ::close(fd);
    return make_error(ErrorCategory::Format, path.string() + ": empty file");
  }
  const size_t size = size_t(st.st_size);
  void* map = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    const std::string why = std::system_category().message(errno);
    ::close(fd);
    return make_error(ErrorCategory::Io, "cannot mmap " + path.string() + ": " + why);
  }
  // The mapping outlives the descriptor.
  ::close(fd);

  auto file = std::span<const std::byte>(static_cast<const std::byte*>(map), size);
  auto parsed = parse(file, path);
  if (!parsed) {
    ::munmap(map, size);
    return parsed;
  }
  parsed->map_ = static_cast<const std::byte*>(map);
  parsed->map_size_ = size;
  parsed->fd_ = -1;
  return parsed;
}

std::expected<Reader, Error> Reader::from_memory(std::span<const std::byte> data) {
  auto parsed = parse(data, {});
  if (!parsed) return parsed;
  parsed->fd_ = -1;
  parsed->map_ = nullptr;
  parsed->map_size_ = 0;
  // parse() leaves data_ pointing into `data`; it is borrowed, not owned.
  return parsed;
}

std::expected<Reader, Error> Reader::from_memory(const void* data, size_t size) {
  return from_memory(std::span<const std::byte>(static_cast<const std::byte*>(data), size));
}

std::expected<Reader, Error> Reader::parse(std::span<const std::byte> file,
                                           const std::filesystem::path& path) {
  Reader out;
  out.path_ = path;
  if (file.size() < kHeaderLengthBytes)
    return make_error(ErrorCategory::Truncated,
                      "file is shorter than the 8-byte header length", file.size());

  const uint64_t header_len = load_le64(file.data());
  if (header_len == 0)
    return make_error(ErrorCategory::Format, "zero-length JSON header", kHeaderLengthBytes);
  if (header_len > kMaxHeaderBytes)
    return make_error(ErrorCategory::Format, "JSON header is implausibly large",
                      kHeaderLengthBytes);
  if (header_len > file.size() - kHeaderLengthBytes)
    return make_error(ErrorCategory::Truncated,
                      "JSON header runs past the end of the file", kHeaderLengthBytes);

  const auto header_bytes = file.subspan(kHeaderLengthBytes, size_t(header_len));
  const std::string_view header(reinterpret_cast<const char*>(header_bytes.data()),
                                header_bytes.size());
  std::string json_error;
  auto root = detail::parse_json(header, &json_error);
  if (!root)
    return make_error(ErrorCategory::Format, "malformed JSON header: " + json_error,
                      kHeaderLengthBytes);
  if (root->type != detail::JsonValue::Type::Object)
    return make_error(ErrorCategory::Format, "JSON header is not an object",
                      kHeaderLengthBytes);

  const auto data_offset = kHeaderLengthBytes + header_len;
  const uint64_t data_size = file.size() - data_offset;
  out.data_ = file.subspan(size_t(data_offset));

  struct Range {
    uint64_t begin = 0;
    uint64_t end = 0;
    std::string name;
  };
  std::vector<Range> ranges;
  ranges.reserve(root->object.size());

  for (const auto& [key, entry] : root->object) {
    if (key == "__metadata__") {
      if (entry.type != detail::JsonValue::Type::Object)
        return make_error(ErrorCategory::Format, "__metadata__ is not an object");
      for (const auto& [mk, mv] : entry.object) {
        if (mv.type != detail::JsonValue::Type::String)
          return make_error(ErrorCategory::Format,
                            "__metadata__ value \"" + mk + "\" is not a string");
        out.metadata_.emplace_back(mk, mv.string);
      }
      continue;
    }

    if (entry.type != detail::JsonValue::Type::Object)
      return make_error(ErrorCategory::Format, "tensor entry \"" + key + "\" is not an object");

    const auto* dtype_value = entry.find("dtype");
    const auto* shape_value = entry.find("shape");
    const auto* offsets_value = entry.find("data_offsets");
    if (dtype_value == nullptr || dtype_value->type != detail::JsonValue::Type::String)
      return make_error(ErrorCategory::Format, "tensor \"" + key + "\" has no dtype string");
    if (shape_value == nullptr || shape_value->type != detail::JsonValue::Type::Array)
      return make_error(ErrorCategory::Format, "tensor \"" + key + "\" has no shape array");
    if (offsets_value == nullptr || offsets_value->type != detail::JsonValue::Type::Array ||
        offsets_value->array.size() != 2)
      return make_error(ErrorCategory::Format,
                        "tensor \"" + key + "\" has no two-element data_offsets");

    const auto dtype = dtype_from_name(dtype_value->string);
    if (!dtype)
      return make_error(ErrorCategory::Format,
                        "tensor \"" + key + "\" has unsupported dtype \"" +
                            dtype_value->string + "\"");

    TensorInfo info;
    info.name = key;
    info.dtype = *dtype;
    info.shape.reserve(shape_value->array.size());
    info.n_elements = 1;
    for (const auto& dim : shape_value->array) {
      const auto d = entry_offset(dim);
      if (!d)
        return make_error(ErrorCategory::Format,
                          "tensor \"" + key + "\" has a non-integral or negative shape");
      uint64_t product = 0;
      if (!mul_overflow(info.n_elements, *d, product))
        return make_error(ErrorCategory::Format, "tensor \"" + key + "\" element count overflows");
      info.n_elements = product;
      info.shape.push_back(*d);
    }

    const auto begin = entry_offset(offsets_value->array[0]);
    const auto end = entry_offset(offsets_value->array[1]);
    if (!begin || !end)
      return make_error(ErrorCategory::Format,
                        "tensor \"" + key + "\" has non-integral or negative data_offsets");
    if (*begin > *end)
      return make_error(ErrorCategory::Format,
                        "tensor \"" + key + "\" has data_offsets begin > end");
    info.begin = *begin;
    info.end = *end;

    const auto element_size = dtype_size(info.dtype);
    uint64_t byte_size = 0;
    if (!mul_overflow(info.n_elements, *element_size, byte_size))
      return make_error(ErrorCategory::Format, "tensor \"" + key + "\" byte size overflows");
    if (byte_size != info.end - info.begin)
      return make_error(ErrorCategory::Format,
                        "tensor \"" + key + "\": shape * dtype size (" +
                            std::to_string(byte_size) + " bytes) does not match data_offsets (" +
                            std::to_string(info.end - info.begin) + " bytes)");
    if (info.end > data_size)
      return make_error(ErrorCategory::Truncated,
                        "tensor \"" + key + "\" runs past the data section",
                        data_offset + info.end);

    ranges.push_back(Range{info.begin, info.end, info.name});
    out.tensors_.push_back(std::move(info));
  }

  // Ranges must be disjoint (empty tensors may share an offset).
  std::sort(ranges.begin(), ranges.end(),
            [](const Range& a, const Range& b) { return a.begin < b.begin; });
  for (size_t i = 1; i < ranges.size(); ++i) {
    const Range& prev = ranges[i - 1];
    const Range& cur = ranges[i];
    if (prev.begin != prev.end && cur.begin != cur.end && cur.begin < prev.end)
      return make_error(ErrorCategory::Format,
                        "tensors \"" + prev.name + "\" and \"" + cur.name +
                            "\" have overlapping data ranges");
  }

  return out;
}

const TensorInfo* Reader::tensor(std::string_view name) const {
  for (const auto& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

std::optional<std::span<const std::byte>> Reader::tensor_data(const TensorInfo& t) const {
  if (t.end > data_.size() || t.begin > t.end) return std::nullopt;
  return data_.subspan(size_t(t.begin), size_t(t.end - t.begin));
}

std::optional<std::span<const std::byte>> Reader::tensor_data(std::string_view name) const {
  const TensorInfo* t = tensor(name);
  if (t == nullptr) return std::nullopt;
  return tensor_data(*t);
}

uint64_t Reader::file_offset(const TensorInfo& t) const noexcept {
  if (map_ == nullptr) return t.begin;
  return uint64_t(data_.data() - map_) + t.begin;
}

// ---------------------------------------------------------------------------
// ShardedReader
// ---------------------------------------------------------------------------

std::expected<ShardedReader, Error> ShardedReader::load(
    const std::filesystem::path& index_path) {
  std::ifstream in(index_path, std::ios::binary);
  if (!in)
    return make_error(ErrorCategory::Io, "cannot open " + index_path.string());
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (!in.good() && !in.eof())
    return make_error(ErrorCategory::Io, "cannot read " + index_path.string());

  std::string json_error;
  auto root = detail::parse_json(text, &json_error);
  if (!root)
    return make_error(ErrorCategory::Format,
                      "malformed index JSON: " + json_error);
  const auto* weight_map = root->find("weight_map");
  if (weight_map == nullptr || weight_map->type != detail::JsonValue::Type::Object)
    return make_error(ErrorCategory::Format, "index has no weight_map object");

  const std::filesystem::path base = index_path.parent_path();
  ShardedReader out;
  std::vector<std::string> shard_names;
  std::vector<size_t> shard_of_entry;
  shard_of_entry.reserve(weight_map->object.size());

  for (const auto& [name, shard_value] : weight_map->object) {
    if (shard_value.type != detail::JsonValue::Type::String)
      return make_error(ErrorCategory::Format,
                        "weight_map entry \"" + name + "\" is not a string");
    size_t index = 0;
    while (index < shard_names.size() && shard_names[index] != shard_value.string) ++index;
    if (index == shard_names.size()) {
      auto reader = Reader::load(base / shard_value.string);
      if (!reader)
        return make_error(reader.error().category,
                          "shard \"" + shard_value.string + "\": " + reader.error().message,
                          reader.error().offset);
      shard_names.push_back(shard_value.string);
      out.shards_.push_back(std::move(*reader));
    }
    shard_of_entry.push_back(index);
  }

  size_t entry = 0;
  for (const auto& [name, shard_value] : weight_map->object) {
    (void)shard_value;
    const size_t shard = shard_of_entry[entry++];
    const TensorInfo* info = out.shards_[shard].tensor(name);
    if (info == nullptr)
      return make_error(ErrorCategory::Format,
                        "shard \"" + shard_names[shard] + "\" does not contain \"" + name + "\"");
    out.tensors_.push_back(ShardedTensor{*info, shard});
  }
  return out;
}

const ShardedTensor* ShardedReader::tensor(std::string_view name) const {
  for (const auto& t : tensors_)
    if (t.info.name == name) return &t;
  return nullptr;
}

std::optional<std::span<const std::byte>> ShardedReader::tensor_data(
    std::string_view name) const {
  const ShardedTensor* t = tensor(name);
  if (t == nullptr) return std::nullopt;
  return shards_[t->shard].tensor_data(t->info);
}

} // namespace sonicboom::safetensors
