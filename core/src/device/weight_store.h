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

// The inference weight store: a safetensors or GGUF file, mapped once, whose
// named tensors are uploaded to the device on first use and shared by every
// executable bound to the store.
//
// The store owns its device buffers. An executable that binds a weight gets
// the buffer handle and never frees it; the C API keeps the store alive for as
// long as any executable references it, so a caller that frees its store
// handle first cannot dangle one.
//
// Dtype: a graph reads float32 weights. A file entry may be stored f32 (used
// as is) or f16/bf16/f64, which `convert_to_f32` converts on the host at
// upload. Quantized GGUF entries are not converted here — the kernels read f32
// — and are reported by name and type.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/safetensors/reader.h>
#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <expected>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::device {

enum class WeightFormat { Auto, Safetensors, Gguf };

struct WeightStoreOptions {
  WeightFormat format = WeightFormat::Auto;
  bool convert_to_f32 = false;
};

class WeightStore {
public:
  // Map and parse PATH. Fails with a diagnostic naming the file and, when the
  // format is known, the offending tensor.
  static std::expected<std::shared_ptr<WeightStore>, std::string>
  open(const std::filesystem::path& path, const WeightStoreOptions& options = {});

  ~WeightStore();
  WeightStore(const WeightStore&) = delete;
  WeightStore& operator=(const WeightStore&) = delete;

  bool has(std::string_view name) const;

  // Host dtype/shape of a named entry (the shape in row-major order, after
  // GGUF's fastest-first dims are reversed). For a caller checking a name
  // mapping before compiling against it.
  std::optional<sx::DType> dtype_of(std::string_view name) const;
  std::optional<std::vector<std::int64_t>> shape_of(std::string_view name) const;

  // The stored type as the file names it ("F32", "F16", "Q4_K", ...), the
  // stored shape and the stored byte count, for a caller reporting a mismatch
  // the way the store reports it on upload. The pointers stay valid until the
  // store is destroyed; nullopt for an absent name.
  struct EntryInfo {
    const std::string* type_name = nullptr;
    const std::vector<std::int64_t>* shape = nullptr;
    std::uint64_t num_bytes = 0;
  };
  std::optional<EntryInfo> entry_info(std::string_view name) const;

  // The device buffer holding NAME as float32 of WANT_SHAPE, uploaded (and
  // converted, when enabled) once. The buffer stays owned by this store.
  // Returns 0 with a diagnostic when the name is absent, the shape differs
  // (naming both), or the dtype cannot be supplied.
  std::uint64_t device_buffer(std::string_view name, sx::DType want_dtype,
                              const std::vector<std::int64_t>& want_shape,
                              std::string* error);

  // A borrowed host copy of NAME as float32: the file's own bytes when the
  // entry is f32, a conversion cached in this store otherwise. The pointer is
  // valid until the store is destroyed. Used by the CPU fallback, which binds
  // weights to the interpreter from the same file the device would upload.
  bool host_f32(std::string_view name, const void** data, std::uint64_t* bytes,
                std::string* error);

  std::size_t resident_count() const noexcept { return device_.size(); }
  std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }
  const std::filesystem::path& path() const noexcept { return path_; }
  WeightFormat format() const noexcept { return format_; }

private:
  WeightStore() = default;

  struct Entry {
    // nullopt for a stored type the f32 store cannot supply (a quantized GGUF
    // entry); type_name then names it in the diagnostic.
    std::optional<sx::DType> dtype;
    std::string type_name;
    std::vector<std::int64_t> shape;
    const std::byte* data = nullptr;  // into the mapped file / owned bytes
    std::uint64_t num_bytes = 0;
  };

  // ENTRY (named NAME) as host float32 for N values: the file's bytes when it
  // is f32, a conversion into CACHE otherwise. Null on a dtype or size the f32
  // form cannot represent, naming the reason.
  const void* f32_bytes_of(std::string_view name, const Entry& e,
                           std::uint64_t n, bool convert,
                           std::vector<float>* cache, std::string* error) const;

  std::filesystem::path path_;
  WeightFormat format_ = WeightFormat::Safetensors;
  WeightStoreOptions options_;
  // Exactly one is engaged; Entry::data points into its mapping/buffer.
  std::optional<safetensors::Reader> st_;
  std::optional<gguf::Reader> gguf_;
  std::map<std::string, Entry, std::less<>> entries_;
  std::map<std::string, std::uint64_t, std::less<>> device_;
  // Converted host copies, one per entry asked for as float32 (host_f32 or a
  // converting device upload); f32 entries are not copied.
  std::map<std::string, std::vector<float>, std::less<>> f32_cache_;
  std::uint64_t resident_bytes_ = 0;
};

} // namespace sonicboom::device
