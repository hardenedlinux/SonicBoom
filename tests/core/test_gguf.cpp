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

// GGUF v3 reader tests. Builds synthetic GGUF files in memory (no external
// model required) and exercises: header + metadata reading, every metadata
// value type (incl. string arrays), the tensor directory, alignment and
// offsets, raw byte access, boundary checks, unknown ggml types, and
// corrupted/truncated files. Real-model validation is a separate step.

#include <sonicboom/gguf/reader.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace sbgguf = sonicboom::gguf;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

// --- little-endian byte writer (host == little-endian, matching the reader) --
struct Writer {
  std::vector<std::byte> b;

  template <class T>
  void le(T v) {
    size_t off = b.size();
    b.resize(off + sizeof(T));
    std::memcpy(b.data() + off, &v, sizeof(T));
  }
  void u8(uint8_t v) { b.push_back(std::byte(v)); }
  void u16(uint16_t v) { le(v); }
  void u32(uint32_t v) { le(v); }
  void u64(uint64_t v) { le(v); }
  void f32(float v) { uint32_t u; std::memcpy(&u, &v, 4); u32(u); }
  void str(std::string_view s) {
    u64(s.size());
    for (char c : s) u8(uint8_t(c));
  }
  void raw(const void* p, size_t n) {
    const auto* c = static_cast<const std::byte*>(p);
    b.insert(b.end(), c, c + n);
  }
  void pad_to(size_t align) {
    while (b.size() % align != 0) u8(0);
  }
};

// Encode n floats as little-endian bytes.
std::vector<std::byte> floats(const std::vector<float>& v) {
  Writer w;
  for (float x : v) w.f32(x);
  return std::move(w.b);
}

void write_header(Writer& w, uint32_t version, uint64_t n_tensors, uint64_t n_kv) {
  w.u32(0x46554747u);  // "GGUF"
  w.u32(version);
  w.u64(n_tensors);
  w.u64(n_kv);
}

} // namespace

int main() {
  namespace fs = std::filesystem;

  // --- header + metadata + tensor read --------------------------------------
  {
    Writer w;
    // KV: general.architecture (string), general.alignment (u32), general.name
    // (string), a u64, an f32.
    write_header(w, 3, 1, 5);
    w.str("general.architecture"); w.u32(8); w.str("gemma4");
    w.str("general.alignment"); w.u32(4); w.u32(32);
    w.str("general.name"); w.u32(8); w.str("test-model");
    w.str("some.count"); w.u32(10); w.u64(9000000000000000000ull);
    w.str("some.scale"); w.u32(6); w.f32(1.5f);
    w.str("w"); w.u32(2); w.u64(4); w.u64(2); w.u32(0 /*f32*/); w.u64(0);
    w.pad_to(32);
    auto fbytes = floats({1, 2, 3, 4, 5, 6, 7, 8});
    for (auto b : fbytes) w.b.push_back(b);

    auto r = sbgguf::Reader::from_bytes(w.b.data(), w.b.size());
    check(r.has_value(), "header: parses");
    if (!r) {
      std::cerr << "    error: " << r.error().message << "\n";
      return 1;
    }

    check(r->version() == 3, "header: version 3");
    check(r->kv_count() == 5, "header: kv_count 5");
    check(r->tensor_count() == 1, "header: tensor_count 1");
    check(r->alignment() == 32, "header: alignment 32");

    const auto* arch = r->find("general.architecture");
    check(arch && arch->type == sbgguf::ValueType::String && arch->s == "gemma4",
          "metadata: architecture string");
    const auto* align = r->find("general.alignment");
    check(align && align->type == sbgguf::ValueType::UInt32 && align->u == 32,
          "metadata: alignment u32");
    const auto* cnt = r->find("some.count");
    check(cnt && cnt->type == sbgguf::ValueType::UInt64 &&
              cnt->u == 9000000000000000000ull,
          "metadata: uint64");
    const auto* scale = r->find("some.scale");
    check(scale && scale->type == sbgguf::ValueType::Float32 &&
              scale->f == 1.5f,
          "metadata: float32");
    check(r->find("absent.key") == nullptr, "metadata: absent key is null");

    const auto* t = r->tensor("w");
    check(t != nullptr, "tensor: found by name");
    if (t) {
      check(t->name == "w", "tensor: name");
      check(t->type == 0 && sbgguf::ggml_type_name(0) == "f32", "tensor: f32 type");
      check(t->dims == std::vector<uint64_t>({4, 2}), "tensor: dims");
      check(t->n_elements == 8, "tensor: n_elements");
      check(t->byte_size.has_value() && *t->byte_size == 32, "tensor: byte_size 32");
    }

    auto data = r->tensor_data("w");
    check(data.has_value() && data->size() == 32, "tensor_data: 32 bytes");
    if (data) {
      std::vector<float> got(8);
      std::memcpy(got.data(), data->data(), 32);
      for (int i = 0; i < 8; ++i)
        check(got[i] == float(i + 1), "tensor_data: byte contents match");
    }
  }

  // --- every metadata value type --------------------------------------------
  {
    Writer w;
    write_header(w, 3, 0, 14);
    w.str("a.u8");    w.u32(0);  w.u8(250);
    w.str("a.i8");    w.u32(1);  w.u8(uint8_t(int8_t(-5)));
    w.str("a.u16");   w.u32(2);  w.u16(65530);
    w.str("a.i16");   w.u32(3);  w.u16(uint16_t(int16_t(-300)));
    w.str("a.u32");   w.u32(4);  w.u32(4000000000u);
    w.str("a.i32");   w.u32(5);  w.u32(uint32_t(int32_t(-123456)));
    w.str("a.f32");   w.u32(6);  w.f32(1.5f);
    w.str("a.bool");  w.u32(7);  w.u8(1);
    w.str("a.str");   w.u32(8);  w.str("hello");
    w.str("a.u64");   w.u32(10); w.u64(123456789012345ull);
    w.str("a.i64");   w.u32(11); w.u64(uint64_t(int64_t(-7)));
    w.str("a.f64");   w.u32(12); { double d = 2.25; w.raw(&d, 8); }
    w.str("a.arr");   w.u32(9);  w.u32(5); w.u64(3); w.u32(1); w.u32(2); w.u32(3);
    w.str("a.strarr"); w.u32(9); w.u32(8); w.u64(3); w.str("a"); w.str("bb"); w.str("ccc");

    auto r = sbgguf::Reader::from_bytes(w.b.data(), w.b.size());
    check(r.has_value(), "values: parses");
    if (!r) {
      std::cerr << "    error: " << r.error().message << "\n";
      return 1;
    }
    auto get = [&](std::string_view k) { return r->find(k); };

    check(get("a.u8")->u == 250, "value: uint8");
    check(get("a.i8")->i == -5, "value: int8");
    check(get("a.u16")->u == 65530, "value: uint16");
    check(get("a.i16")->i == -300, "value: int16");
    check(get("a.u32")->u == 4000000000u, "value: uint32");
    check(get("a.i32")->i == -123456, "value: int32");
    check(get("a.f32")->f == 1.5f, "value: float32");
    check(get("a.bool")->b == true, "value: bool");
    check(get("a.str")->s == "hello", "value: string");
    check(get("a.u64")->u == 123456789012345ull, "value: uint64");
    check(get("a.i64")->i == -7, "value: int64");
    check(get("a.f64")->f == 2.25, "value: float64");

    const auto* arr = get("a.arr");
    check(arr && arr->type == sbgguf::ValueType::Array &&
              arr->elem_type == sbgguf::ValueType::Int32 && arr->elems.size() == 3 &&
              arr->elems[0].i == 1 && arr->elems[1].i == 2 && arr->elems[2].i == 3,
          "value: int32 array");

    const auto* sarr = get("a.strarr");
    check(sarr && sarr->type == sbgguf::ValueType::Array &&
              sarr->elem_type == sbgguf::ValueType::String && sarr->elems.size() == 3 &&
              sarr->elems[0].s == "a" && sarr->elems[1].s == "bb" &&
              sarr->elems[2].s == "ccc",
          "value: string array");
  }

  // --- alignment + offsets ---------------------------------------------------
  {
    // Two tensors, one f32[8] (32 bytes) at offset 0 and one f16[4] (8 bytes)
    // at offset 32, with a distinctive byte pattern so the span is verifiable.
    Writer w;
    write_header(w, 3, 2, 1);
    w.str("general.alignment"); w.u32(4); w.u32(32);

    w.str("a"); w.u32(1); w.u64(8);  w.u32(0 /*f32*/); w.u64(0);
    w.str("b"); w.u32(1); w.u64(4);  w.u32(1 /*f16*/); w.u64(32);

    size_t info_end = w.b.size();
    w.pad_to(32);
    size_t data_off = w.b.size();
    check(data_off % 32 == 0 && data_off >= info_end, "align: data section aligned");
    // tensor a: 32 bytes 0xAA; tensor b: 8 bytes 0xBB
    w.b.insert(w.b.end(), 32, std::byte{0xAA});
    w.b.insert(w.b.end(), 8, std::byte{0xBB});

    auto r = sbgguf::Reader::from_bytes(w.b.data(), w.b.size());
    check(r.has_value(), "align: parses");
    if (!r) {
      std::cerr << "    error: " << r.error().message << "\n";
      return 1;
    }
    check(r->data_section().size() == 40, "align: data section spans both tensors");

    auto da = r->tensor_data("a");
    check(da.has_value() && da->size() == 32 && da->front() == std::byte{0xAA},
          "align: tensor a bytes at offset 0");
    auto db = r->tensor_data("b");
    check(db.has_value() && db->size() == 8 && db->front() == std::byte{0xBB},
          "align: tensor b bytes at offset 32");
  }

  // --- boundary checks -------------------------------------------------------
  {
    // Tensor declares 16 bytes but the file has no data section bytes.
    Writer w;
    write_header(w, 3, 1, 0);
    w.str("x"); w.u32(1); w.u64(4); w.u32(0 /*f32*/); w.u64(0);  // 16 bytes
    auto r = sbgguf::Reader::from_bytes(w.b.data(), w.b.size());
    check(r.has_value(), "boundary: parses");
    if (r) {
      const auto* t = r->tensor("x");
      check(t && t->byte_size.has_value() && *t->byte_size == 16,
            "boundary: size computed");
      check(!r->tensor_data("x").has_value(),
            "boundary: out-of-range tensor_data is nullopt");
    }

    // Offset beyond the data section.
    Writer w2;
    write_header(w2, 3, 1, 0);
    w2.str("y"); w2.u32(1); w2.u64(4); w2.u32(0); w2.u64(1000000);
    w2.pad_to(32);
    w2.b.insert(w2.b.end(), 16, std::byte{0});
    auto r2 = sbgguf::Reader::from_bytes(w2.b.data(), w2.b.size());
    check(r2.has_value(), "boundary: offset-past-end parses");
    if (r2) check(!r2->tensor_data("y").has_value(),
                  "boundary: offset past end is nullopt");
  }

  // --- unknown ggml types ----------------------------------------------------
  {
    Writer w;
    write_header(w, 3, 1, 0);
    w.str("q"); w.u32(1); w.u64(32); w.u32(999 /*unknown*/); w.u64(0);
    auto r = sbgguf::Reader::from_bytes(w.b.data(), w.b.size());
    check(r.has_value(), "unknown type: parses (graceful)");
    if (r) {
      const auto* t = r->tensor("q");
      check(t && t->type == 999, "unknown type: id preserved");
      check(t && !t->byte_size.has_value(), "unknown type: byte_size nullopt");
      check(!r->tensor_data("q").has_value(), "unknown type: data nullopt");
    }
    check(sbgguf::ggml_type_info(999) == std::nullopt, "unknown type: no type info");
    check(sbgguf::ggml_type_name(999) == "unknown", "unknown type: name \"unknown\"");
    check(sbgguf::ggml_type_size(999, 32) == std::nullopt, "unknown type: size nullopt");
    // Known type size sanity: q4_0 rounds 33 elements up to 2 blocks = 36 bytes.
    auto sz = sbgguf::ggml_type_size(2 /*q4_0*/, 33);
    check(sz.has_value() && *sz == 36, "type size: q4_0 33 elems -> 36 bytes");
    check(sbgguf::ggml_type_name(30) == "bf16", "type name: bf16");
  }

  // --- corrupted / truncated files -------------------------------------------
  {
    // bad magic
    Writer w;
    write_header(w, 3, 0, 0);
    w.b[0] = std::byte{0x00};
    auto r = sbgguf::Reader::from_bytes(w.b.data(), w.b.size());
    check(!r.has_value() && r.error().category == sbgguf::ErrorCategory::Format,
          "corrupt: bad magic rejected");

    // unsupported version
    Writer w2;
    write_header(w2, 2, 0, 0);
    auto r2 = sbgguf::Reader::from_bytes(w2.b.data(), w2.b.size());
    check(!r2.has_value() && r2.error().category == sbgguf::ErrorCategory::Format,
          "corrupt: unsupported version rejected");

    // truncated header (fewer than 24 bytes)
    Writer w3;
    w3.u32(0x46554747u);
    w3.u32(3);
    auto r3 = sbgguf::Reader::from_bytes(w3.b.data(), w3.b.size());
    check(!r3.has_value() && r3.error().category == sbgguf::ErrorCategory::Truncated,
          "corrupt: truncated header rejected");

    // truncated string (declared length exceeds remaining bytes)
    Writer w4;
    write_header(w4, 3, 0, 1);
    w4.str("key");
    w4.u32(8);           // string value
    w4.u64(1000);        // length 1000, but no bytes follow
    auto r4 = sbgguf::Reader::from_bytes(w4.b.data(), w4.b.size());
    check(!r4.has_value() && r4.error().category == sbgguf::ErrorCategory::Truncated,
          "corrupt: truncated string rejected");

    // zero-dims tensor
    Writer w5;
    write_header(w5, 3, 1, 0);
    w5.str("z");
    w5.u32(0);           // n_dims == 0
    auto r5 = sbgguf::Reader::from_bytes(w5.b.data(), w5.b.size());
    check(!r5.has_value() && r5.error().category == sbgguf::ErrorCategory::Format,
          "corrupt: zero-dim tensor rejected");

    // truncated tensor info (n_dims claims more dims than present)
    Writer w6;
    write_header(w6, 3, 1, 0);
    w6.str("t");
    w6.u32(4);           // 4 dims
    w6.u64(2);           // but only one follows
    auto r6 = sbgguf::Reader::from_bytes(w6.b.data(), w6.b.size());
    check(!r6.has_value() && r6.error().category == sbgguf::ErrorCategory::Truncated,
          "corrupt: truncated tensor info rejected");
  }

  // --- file-backed load ------------------------------------------------------
  {
    Writer w;
    write_header(w, 3, 1, 1);
    w.str("general.name"); w.u32(8); w.str("disk");
    w.str("w"); w.u32(1); w.u64(2); w.u32(0 /*f32*/); w.u64(0);
    w.pad_to(32);
    auto fbytes = floats({3.5f, -1.25f});
    for (auto b : fbytes) w.b.push_back(b);

    fs::path dir = fs::temp_directory_path() / "sonicboom_gguf_test";
    std::error_code ec;
    fs::create_directories(dir, ec);
    fs::path file = dir / "synthetic.gguf";
    {
      std::ofstream out(file, std::ios::binary);
      out.write(reinterpret_cast<const char*>(w.b.data()), std::streamsize(w.b.size()));
    }
    auto r = sbgguf::Reader::load(file);
    check(r.has_value(), "file: load succeeds");
    if (r) {
      check(r->find("general.name")->s == "disk", "file: metadata read");
      auto d = r->tensor_data("w");
      check(d.has_value() && d->size() == 8, "file: tensor data read");
      if (d) {
        std::vector<float> got(2);
        std::memcpy(got.data(), d->data(), 8);
        check(got[0] == 3.5f && got[1] == -1.25f, "file: tensor bytes match");
      }
    }
    // missing file
    auto missing = sbgguf::Reader::load(dir / "nope.gguf");
    check(!missing.has_value() && missing.error().category == sbgguf::ErrorCategory::Io,
          "file: missing file is Io error");
    fs::remove_all(dir, ec);
  }

  if (g_failures == 0) {
    std::cout << "test_gguf OK\n";
    return 0;
  }
  std::cerr << "test_gguf FAILED: " << g_failures << " check(s)\n";
  return 1;
}
