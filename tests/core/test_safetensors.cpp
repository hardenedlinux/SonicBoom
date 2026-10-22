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

// safetensors reader tests. Builds synthetic safetensors images in memory (no
// external model required) and exercises: the header length + JSON directory,
// dtype sizes, shapes and byte ranges, metadata, raw byte access through both
// the memory and mmap (file) paths, the dtype table, a two-shard index, and
// corrupted / inconsistent files (bad header length, malformed JSON, unknown
// dtype, byte-size mismatch, overlap, out-of-range offsets, duplicates).

#include <sonicboom/safetensors/reader.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace sbst = sonicboom::safetensors;
namespace fs = std::filesystem;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

using Bytes = std::vector<std::byte>;

void append_le64(Bytes& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(std::byte((v >> (8 * i)) & 0xff));
}

void append(Bytes& out, std::string_view s) {
  for (char c : s) out.push_back(std::byte(c));
}

void append(Bytes& out, std::span<const std::byte> s) {
  out.insert(out.end(), s.begin(), s.end());
}

// One safetensors image: `header` is the JSON directory, `data` the data
// section. Offsets inside `header` are relative to `data`.
Bytes make_image(std::string_view header, std::span<const std::byte> data) {
  Bytes out;
  append_le64(out, uint64_t(header.size()));
  append(out, header);
  append(out, data);
  return out;
}

std::span<const std::byte> as_bytes(const Bytes& b) {
  return std::span<const std::byte>(b.data(), b.size());
}

Bytes as_bytes_vec(std::string_view s) {
  Bytes out;
  append(out, s);
  return out;
}

void write_file(const fs::path& p, std::string_view bytes) {
  std::ofstream out(p, std::ios::binary | std::ios::trunc);
  out.write(bytes.data(), std::streamsize(bytes.size()));
}

const char kMetadataHeader[] =
    R"({"__metadata__":{"format":"pt","note":"a\"bé"},)"
    R"("w":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}})";
const char kMetadataData[] = "abcdefghijklmnopqrstuvwx";  // 24 bytes

} // namespace

int main() {
  const fs::path tmp = fs::temp_directory_path() /
                       ("sb_safetensors_test_" + std::to_string(::getpid()));
  fs::remove_all(tmp);
  fs::create_directories(tmp);

  // --- dtype table ---------------------------------------------------------
  {
    check(sbst::dtype_size(sbst::DType::F64) == 8, "F64 is 8 bytes");
    check(sbst::dtype_size(sbst::DType::F32) == 4, "F32 is 4 bytes");
    check(sbst::dtype_size(sbst::DType::BF16) == 2, "BF16 is 2 bytes");
    check(sbst::dtype_size(sbst::DType::F16) == 2, "F16 is 2 bytes");
    check(sbst::dtype_size(sbst::DType::I8) == 1, "I8 is 1 byte");
    check(sbst::dtype_size(sbst::DType::BOOL) == 1, "BOOL is 1 byte");
    check(sbst::dtype_from_name("BF16") == sbst::DType::BF16, "BF16 name roundtrip");
    check(!sbst::dtype_from_name("FP8"), "unknown dtype is rejected");
    check(sbst::dtype_name(sbst::DType::F16) == "F16", "F16 name");
  }

  // --- single tensor + metadata, memory path -------------------------------
  {
    Bytes data = as_bytes_vec(kMetadataData);
    Bytes img = make_image(kMetadataHeader, as_bytes(data));
    auto r = sbst::Reader::from_memory(img.data(), img.size());
    check(bool(r), "metadata image parses");
    if (r) {
      check(r->tensors().size() == 1, "one tensor");
      check(r->metadata().size() == 2, "two metadata entries");
      check(r->metadata()[1].second == "a\"b\xc3\xa9", "string escapes decode");
      const auto* w = r->tensor("w");
      check(w != nullptr, "tensor lookup");
      if (w) {
        check(w->dtype == sbst::DType::F32, "dtype");
        check(w->shape.size() == 2 && w->shape[0] == 2 && w->shape[1] == 3, "shape");
        check(w->n_elements == 6, "element count");
        check(w->begin == 0 && w->end == 24, "byte range");
        auto bytes = r->tensor_data(*w);
        check(bytes.has_value() && bytes->size() == 24, "byte size");
        if (bytes)
          check(std::memcmp(bytes->data(), kMetadataData, 24) == 0, "byte contents");
      }
      check(!r->tensor("missing"), "absent tensor");
    }
  }

  // --- mmap path -----------------------------------------------------------
  {
    const fs::path p = tmp / "one.safetensors";
    std::string bytes;
    {
      Bytes data = as_bytes_vec(kMetadataData);
      Bytes img = make_image(kMetadataHeader, as_bytes(data));
      bytes.assign(reinterpret_cast<const char*>(img.data()), img.size());
    }
    write_file(p, bytes);
    auto r = sbst::Reader::load(p);
    check(bool(r), "mmap image loads");
    if (r) {
      check(r->path() == p, "path recorded");
      auto d = r->tensor_data("w");
      check(d.has_value() && std::memcmp(d->data(), kMetadataData, 24) == 0,
            "mmap byte contents");
      check(r->data_section().size() == 24, "data section size");
    }
  }

  // --- several dtypes in one file ------------------------------------------
  {
    // bf16 [3,4] = 24 bytes at 0, f16 [2] = 4 bytes at 24, i8 [5] at 28,
    // empty f32 [0] at 33, bool [3] at 33.
    const std::string header =
        R"({"a":{"dtype":"BF16","shape":[3,4],"data_offsets":[0,24]},)"
        R"("b":{"dtype":"F16","shape":[2],"data_offsets":[24,28]},)"
        R"("c":{"dtype":"I8","shape":[5],"data_offsets":[28,33]},)"
        R"("e":{"dtype":"F32","shape":[0],"data_offsets":[33,33]},)"
        R"("d":{"dtype":"BOOL","shape":[3],"data_offsets":[33,36]}})";
    Bytes data(36, std::byte(0x5a));
    Bytes img = make_image(header, as_bytes(data));
    auto r = sbst::Reader::from_memory(img.data(), img.size());
    check(bool(r), "multi-dtype image parses");
    if (r) {
      check(r->tensors().size() == 5, "five tensors");
      check(r->tensor("a") && r->tensor("a")->n_elements == 12, "bf16 element count");
      check(r->tensor("e") && r->tensor("e")->begin == r->tensor("e")->end,
            "empty tensor range");
      check(r->tensor_data("e") && r->tensor_data("e")->empty(), "empty tensor data");
      check(r->tensor_data("d") && r->tensor_data("d")->size() == 3, "bool bytes");
    }
  }

  // --- error cases ---------------------------------------------------------
  auto expect_error = [&](const Bytes& img, sbst::ErrorCategory category, const char* what) {
    auto r = sbst::Reader::from_memory(img.data(), img.size());
    check(!r && r.error().category == category, what);
  };

  {
    Bytes img(4, std::byte(0));
    expect_error(img, sbst::ErrorCategory::Truncated, "short file is truncated");

    Bytes img2;
    append_le64(img2, 1000);
    expect_error(img2, sbst::ErrorCategory::Truncated, "header length past EOF");

    Bytes img3;
    append_le64(img3, 0);
    expect_error(img3, sbst::ErrorCategory::Format, "zero header length");

    Bytes img4;
    const std::string bad = R"({"a":{"dtype":"F32","shape":[1],)"
                            R"("data_offsets":[0,4]}})";
    append_le64(img4, bad.size());
    append(img4, bad);
    img4.resize(img4.size() + 4, std::byte(0));  // range is valid
    auto ok = sbst::Reader::from_memory(img4.data(), img4.size());
    check(bool(ok), "small valid file parses");

    Bytes img5;
    const std::string malformed = R"({"a":)";
    append_le64(img5, malformed.size());
    append(img5, malformed);
    expect_error(img5, sbst::ErrorCategory::Format, "malformed JSON");

    Bytes img6;
    const std::string unsupported =
        R"({"a":{"dtype":"FP8","shape":[1],"data_offsets":[0,1]}})";
    append_le64(img6, unsupported.size());
    append(img6, unsupported);
    img6.push_back(std::byte(0));
    expect_error(img6, sbst::ErrorCategory::Format, "unsupported dtype");

    Bytes img7;
    const std::string mismatch =
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}})";
    append_le64(img7, mismatch.size());
    append(img7, mismatch);
    Bytes four(4, std::byte(0));
    append(img7, as_bytes(four));
    expect_error(img7, sbst::ErrorCategory::Format, "shape/size mismatch");

    Bytes img8;
    const std::string overlap =
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},)"
        R"("b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}})";
    append_le64(img8, overlap.size());
    append(img8, overlap);
    Bytes twelve(12, std::byte(0));
    append(img8, as_bytes(twelve));
    expect_error(img8, sbst::ErrorCategory::Format, "overlapping ranges");

    Bytes img9;
    const std::string past =
        R"({"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    append_le64(img9, past.size());
    append(img9, past);
    Bytes only_four(4, std::byte(0));
    append(img9, as_bytes(only_four));
    expect_error(img9, sbst::ErrorCategory::Truncated, "range past the data section");

    Bytes img10;
    const std::string negative =
        R"({"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,0]}})";
    append_le64(img10, negative.size());
    append(img10, negative);
    expect_error(img10, sbst::ErrorCategory::Format, "negative shape");

    Bytes img11;
    const std::string dup =
        R"({"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},)"
        R"("a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";
    append_le64(img11, dup.size());
    append(img11, dup);
    Bytes four2(4, std::byte(0));
    append(img11, as_bytes(four2));
    expect_error(img11, sbst::ErrorCategory::Format, "duplicate tensor name");
  }

  // --- sharded index -------------------------------------------------------
  {
    const std::string s0_header =
        R"({"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
    const std::string s1_header =
        R"({"y":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}})";
    {
      Bytes d0(8, std::byte(0x11));
      Bytes i0 = make_image(s0_header, as_bytes(d0));
      write_file(tmp / "model-00001-of-00002.safetensors",
                 std::string_view(reinterpret_cast<const char*>(i0.data()), i0.size()));
      Bytes d1(8, std::byte(0x22));
      Bytes i1 = make_image(s1_header, as_bytes(d1));
      write_file(tmp / "model-00002-of-00002.safetensors",
                 std::string_view(reinterpret_cast<const char*>(i1.data()), i1.size()));
    }
    const std::string index =
        R"({"metadata":{"total_size":16},)"
        R"("weight_map":{"x":"model-00001-of-00002.safetensors",)"
        R"("y":"model-00002-of-00002.safetensors"}})";
    write_file(tmp / "model.safetensors.index.json", index);

    auto sr = sbst::ShardedReader::load(tmp / "model.safetensors.index.json");
    check(bool(sr), "sharded index parses");
    if (sr) {
      check(sr->shards().size() == 2, "two shards");
      check(sr->tensors().size() == 2, "two tensors");
      check(sr->tensor("x") && sr->tensor("x")->shard == 0, "x lives in shard 0");
      check(sr->tensor("y") && sr->tensor("y")->shard == 1, "y lives in shard 1");
      auto x = sr->tensor_data("x");
      auto y = sr->tensor_data("y");
      check(x && x->size() == 8 && (*x)[0] == std::byte(0x11), "x bytes");
      check(y && y->size() == 8 && (*y)[0] == std::byte(0x22), "y bytes");
      check(!sr->tensor("z"), "absent tensor in a sharded index");
    }

    // Missing shard file.
    const std::string broken_index =
        R"({"weight_map":{"x":"nope.safetensors"}})";
    write_file(tmp / "broken.index.json", broken_index);
    auto br = sbst::ShardedReader::load(tmp / "broken.index.json");
    check(!br && br.error().category == sbst::ErrorCategory::Io, "missing shard is an I/O error");

    // Shard that does not hold the named tensor.
    const std::string wrong_index =
        R"({"weight_map":{"zzz":"model-00001-of-00002.safetensors"}})";
    write_file(tmp / "wrong.index.json", wrong_index);
    auto wr = sbst::ShardedReader::load(tmp / "wrong.index.json");
    check(!wr && wr.error().category == sbst::ErrorCategory::Format,
          "tensor missing from its shard");
  }

  fs::remove_all(tmp);

  if (g_failures == 0) std::cout << "test_safetensors OK\n";
  else std::cout << "test_safetensors FAILED: " << g_failures << " check(s)\n";
  return g_failures == 0 ? 0 : 1;
}
