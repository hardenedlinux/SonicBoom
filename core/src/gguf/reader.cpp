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

#include <sonicboom/gguf/reader.h>

#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <utility>

namespace sonicboom::gguf {

// ---------------------------------------------------------------------------
// ggml type table
// ---------------------------------------------------------------------------

std::optional<GgmlTypeInfo> ggml_type_info(uint32_t id) {
  // Format reference only (ggml/docs/gguf.md + ggml type traits): name and the
  // (block size, bytes per block) pair used to compute a tensor's byte size.
  // This mirrors the on-disk layout, never the dequantization kernels.
  struct Entry {
    const char* name;
    uint64_t bytes_per_block;
    uint32_t id;
    uint32_t block_size;
  };
  static constexpr Entry kTable[] = {
      {"f32", 4, 0, 1},          {"f16", 2, 1, 1},
      {"q4_0", 18, 2, 32},       {"q4_1", 20, 3, 32},
      {"q4_2", 16, 4, 32},       {"q4_3", 18, 5, 32},
      {"q5_0", 22, 6, 32},       {"q5_1", 24, 7, 32},
      {"q8_0", 34, 8, 32},       {"q8_1", 36, 9, 32},
      {"q2_K", 84, 10, 256},     {"q3_K", 110, 11, 256},
      {"q4_K", 144, 12, 256},    {"q5_K", 176, 13, 256},
      {"q6_K", 210, 14, 256},    {"q8_K", 292, 15, 256},
      {"iq2_xxs", 66, 16, 256},  {"iq2_xs", 74, 17, 256},
      {"iq3_xxs", 98, 18, 256},  {"iq1_s", 66, 19, 256},
      {"iq4_nl", 18, 20, 32},    {"iq3_s", 110, 21, 256},
      {"iq2_s", 66, 22, 256},    {"iq4_xs", 136, 23, 256},
      {"i8", 1, 24, 1},          {"i16", 2, 25, 1},
      {"i32", 4, 26, 1},         {"i64", 8, 27, 1},
      {"f64", 8, 28, 1},         {"iq1_m", 66, 29, 256},
      {"bf16", 2, 30, 1},
  };
  for (const auto& e : kTable)
    if (e.id == id)
      return GgmlTypeInfo{id, e.name, e.block_size, e.bytes_per_block};
  return std::nullopt;
}

std::string_view ggml_type_name(uint32_t id) {
  auto info = ggml_type_info(id);
  return info ? info->name : std::string_view("unknown");
}

std::optional<uint64_t> ggml_type_size(uint32_t id, uint64_t n_elements) {
  auto info = ggml_type_info(id);
  if (!info || info->block_size == 0)
    return std::nullopt;
  uint64_t blocks = n_elements / info->block_size +
                    (n_elements % info->block_size != 0 ? 1 : 0);
  if (info->bytes_per_block != 0 &&
      blocks > std::numeric_limits<uint64_t>::max() / info->bytes_per_block)
    return std::nullopt;  // byte size overflows uint64
  return blocks * info->bytes_per_block;
}

// ---------------------------------------------------------------------------
// Bounds-checked little-endian cursor
// ---------------------------------------------------------------------------

namespace {

struct Cursor {
  const std::byte* base;
  size_t size;
  size_t pos = 0;

  Error err(ErrorCategory c, std::string msg) const {
    return Error{c, std::move(msg), uint64_t(pos)};
  }

  std::expected<uint8_t, Error> u8() {
    if (pos + 1 > size) return std::unexpected(err(ErrorCategory::Truncated, "u8 past end"));
    uint8_t v;
    std::memcpy(&v, base + pos, 1);
    pos += 1;
    return v;
  }

  std::expected<uint16_t, Error> u16() {
    if (pos + 2 > size) return std::unexpected(err(ErrorCategory::Truncated, "u16 past end"));
    uint16_t v;
    std::memcpy(&v, base + pos, 2);
    pos += 2;
    return v;
  }

  std::expected<uint32_t, Error> u32() {
    if (pos + 4 > size) return std::unexpected(err(ErrorCategory::Truncated, "u32 past end"));
    uint32_t v;
    std::memcpy(&v, base + pos, 4);
    pos += 4;
    return v;
  }

  std::expected<uint64_t, Error> u64() {
    if (pos + 8 > size) return std::unexpected(err(ErrorCategory::Truncated, "u64 past end"));
    uint64_t v;
    std::memcpy(&v, base + pos, 8);
    pos += 8;
    return v;
  }

  std::expected<std::string, Error> str() {
    auto len = u64();
    if (!len) return std::unexpected(len.error());
    if (*len > size - pos) return std::unexpected(err(ErrorCategory::Truncated, "string past end"));
    std::string s(reinterpret_cast<const char*>(base + pos), size_t(*len));
    pos += size_t(*len);
    return s;
  }
};

// Read one metadata value (recursive for arrays).
std::expected<Value, Error> read_value(Cursor& c, uint32_t vtype) {
  Value v;
  v.type = static_cast<ValueType>(vtype);
  switch (vtype) {
    case 0: {  // uint8
      auto x = c.u8();
      if (!x) return std::unexpected(x.error());
      v.u = *x;
      break;
    }
    case 1: {  // int8
      auto x = c.u8();
      if (!x) return std::unexpected(x.error());
      // The on-disk byte is a two's-complement int8.
      v.i = static_cast<int8_t>(*x);  // NOLINT(bugprone-signed-char-misuse)
      break;
    }
    case 2: {  // uint16
      auto x = c.u16();
      if (!x) return std::unexpected(x.error());
      v.u = *x;
      break;
    }
    case 3: {  // int16
      auto x = c.u16();
      if (!x) return std::unexpected(x.error());
      v.i = static_cast<int16_t>(*x);
      break;
    }
    case 4: {  // uint32
      auto x = c.u32();
      if (!x) return std::unexpected(x.error());
      v.u = *x;
      break;
    }
    case 5: {  // int32
      auto x = c.u32();
      if (!x) return std::unexpected(x.error());
      v.i = static_cast<int32_t>(*x);
      break;
    }
    case 6: {  // float32
      auto x = c.u32();
      if (!x) return std::unexpected(x.error());
      float f;
      std::memcpy(&f, &*x, 4);
      v.f = f;
      break;
    }
    case 7: {  // bool
      auto x = c.u8();
      if (!x) return std::unexpected(x.error());
      v.b = *x != 0;
      break;
    }
    case 8: {  // string
      auto s = c.str();
      if (!s) return std::unexpected(s.error());
      v.s = std::move(*s);
      break;
    }
    case 9: {  // array
      auto et = c.u32();
      if (!et) return std::unexpected(et.error());
      auto len = c.u64();
      if (!len) return std::unexpected(len.error());
      // Every element is >= 1 byte (uint8/int8/bool), so a length larger than
      // the remaining bytes is guaranteed truncated — bounds the reserve below.
      if (*len > c.size - c.pos)
        return std::unexpected(c.err(ErrorCategory::Truncated, "array length past end"));
      v.elem_type = static_cast<ValueType>(*et);
      v.elems.reserve(size_t(*len));
      for (uint64_t i = 0; i < *len; ++i) {
        auto ev = read_value(c, *et);
        if (!ev) return std::unexpected(ev.error());
        v.elems.push_back(std::move(*ev));
      }
      break;
    }
    case 10: {  // uint64
      auto x = c.u64();
      if (!x) return std::unexpected(x.error());
      v.u = *x;
      break;
    }
    case 11: {  // int64
      auto x = c.u64();
      if (!x) return std::unexpected(x.error());
      v.i = static_cast<int64_t>(*x);
      break;
    }
    case 12: {  // float64
      auto x = c.u64();
      if (!x) return std::unexpected(x.error());
      double d;
      std::memcpy(&d, &*x, 8);
      v.f = d;
      break;
    }
    default:
      return std::unexpected(
          c.err(ErrorCategory::Format, "unknown metadata value type " + std::to_string(vtype)));
  }
  return v;
}

uint64_t align_up(uint64_t x, uint64_t a) {
  // a is non-zero by caller contract.
  uint64_t r = x % a;
  return r == 0 ? x : x + (a - r);
}

} // namespace

std::expected<Reader, Error> Reader::parse(std::span<const std::byte> data) {
  Reader r;
  r.data_ = data;
  Cursor c{data.data(), data.size()};

  // --- header: magic + version + n_tensors + n_kv --------------------------
  auto magic = c.u32();
  if (!magic) return std::unexpected(magic.error());
  if (*magic != 0x46554747u)  // "GGUF" little-endian
    return std::unexpected(Error{ErrorCategory::Format, "bad magic: not a GGUF file", 0});

  auto version = c.u32();
  if (!version) return std::unexpected(version.error());
  if (*version != 3)
    return std::unexpected(
        Error{ErrorCategory::Format, "unsupported GGUF version " + std::to_string(*version), 4});
  r.version_ = *version;

  auto n_tensors = c.u64();
  if (!n_tensors) return std::unexpected(n_tensors.error());
  auto n_kv = c.u64();
  if (!n_kv) return std::unexpected(n_kv.error());

  // --- key/value metadata ---------------------------------------------------
  for (uint64_t i = 0; i < *n_kv; ++i) {
    auto key = c.str();
    if (!key) return std::unexpected(key.error());
    auto vtype = c.u32();
    if (!vtype) return std::unexpected(vtype.error());
    auto val = read_value(c, *vtype);
    if (!val) return std::unexpected(val.error());
    r.metadata_.emplace_back(std::move(*key), std::move(*val));
  }

  // --- tensor directory -----------------------------------------------------
  for (uint64_t i = 0; i < *n_tensors; ++i) {
    auto name = c.str();
    if (!name) return std::unexpected(name.error());

    auto ndims = c.u32();
    if (!ndims) return std::unexpected(ndims.error());
    if (*ndims == 0)
      return std::unexpected(c.err(ErrorCategory::Format, "tensor with zero dimensions"));
    if (uint64_t(*ndims) > (c.size - c.pos) / 8)
      return std::unexpected(c.err(ErrorCategory::Truncated, "dimension list past end"));

    TensorInfo t;
    t.name = std::move(*name);
    t.dims.reserve(*ndims);
    uint64_t nelem = 1;
    for (uint32_t d = 0; d < *ndims; ++d) {
      auto dim = c.u64();
      if (!dim) return std::unexpected(dim.error());
      t.dims.push_back(*dim);
      if (*dim != 0 && nelem > std::numeric_limits<uint64_t>::max() / *dim)
        return std::unexpected(c.err(ErrorCategory::Format, "tensor element count overflow"));
      nelem *= *dim;
    }
    t.n_elements = nelem;

    auto type = c.u32();
    if (!type) return std::unexpected(type.error());
    t.type = *type;

    auto off = c.u64();
    if (!off) return std::unexpected(off.error());
    t.offset = *off;

    t.byte_size = ggml_type_size(*type, nelem);  // nullopt for unknown types
    r.tensors_.push_back(std::move(t));
  }

  // --- alignment + data section start ---------------------------------------
  r.alignment_ = 32;
  if (const Value* a = r.find("general.alignment")) {
    if (a->type == ValueType::UInt32 || a->type == ValueType::UInt64)
      r.alignment_ = a->u;
  }
  if (r.alignment_ == 0)
    r.alignment_ = 32;  // malformed → spec default

  r.data_section_offset_ = align_up(c.pos, r.alignment_);
  return r;
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

std::expected<Reader, Error> Reader::load(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return std::unexpected(Error{ErrorCategory::Io,
                                 "cannot open file: " + path.string(),
                                 std::nullopt});
  f.seekg(0, std::ios::end);
  std::streamoff sz = f.tellg();
  if (sz < 0)
    return std::unexpected(
        Error{ErrorCategory::Io,
              "cannot determine file size: " + path.string(), std::nullopt});
  f.seekg(0, std::ios::beg);

  std::vector<std::byte> buf(static_cast<size_t>(sz));
  if (sz > 0) {
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    if (!f)
      return std::unexpected(Error{ErrorCategory::Io,
                                   "read error: " + path.string(),
                                   std::nullopt});
  }

  auto r = parse(buf);
  if (!r) return std::unexpected(r.error());
  r->owned_ = std::move(buf);  // take ownership; data_ already points into buf's heap
  return r;
}

std::expected<Reader, Error> Reader::from_bytes(std::span<const std::byte> data) {
  return Reader::parse(data);
}

std::expected<Reader, Error> Reader::from_bytes(const void* data, size_t size) {
  return Reader::parse(std::span<const std::byte>(static_cast<const std::byte*>(data), size));
}

const Value* Reader::find(std::string_view key) const {
  for (const auto& [k, v] : metadata_)
    if (k == key) return &v;
  return nullptr;
}

const TensorInfo* Reader::tensor(std::string_view name) const {
  for (const auto& t : tensors_)
    if (t.name == name) return &t;
  return nullptr;
}

std::optional<std::span<const std::byte>> Reader::tensor_data(const TensorInfo& t) const {
  if (!t.byte_size.has_value()) return std::nullopt;
  uint64_t start = data_section_offset_ + t.offset;
  if (start > data_.size()) return std::nullopt;
  uint64_t remaining = data_.size() - start;
  if (*t.byte_size > remaining) return std::nullopt;
  return std::span<const std::byte>(data_.data() + start, size_t(*t.byte_size));
}

std::optional<std::span<const std::byte>> Reader::tensor_data(std::string_view name) const {
  const TensorInfo* t = tensor(name);
  return t ? tensor_data(*t) : std::nullopt;
}

std::span<const std::byte> Reader::data_section() const noexcept {
  if (data_section_offset_ >= data_.size()) return {};
  return std::span<const std::byte>(data_.data() + data_section_offset_,
                                    data_.size() - data_section_offset_);
}

} // namespace sonicboom::gguf
