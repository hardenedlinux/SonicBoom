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

// The inference weight store (weight_store.h).

#include "weight_store.h"

#include "../training/device_graph_kernels.h"

#include <cstring>
#include <fstream>
#include <string>

namespace sonicboom::device {

namespace {

namespace fs = std::filesystem;

std::string join_shape(const std::vector<std::int64_t>& shape) {
  std::string s = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  return s + ")";
}

std::uint64_t numel_of(const std::vector<std::int64_t>& shape) {
  std::uint64_t n = 1;
  for (std::int64_t d : shape) n *= static_cast<std::uint64_t>(d);
  return n;
}

std::optional<sx::DType> map_safetensors_dtype(safetensors::DType d) {
  switch (d) {
  case safetensors::DType::F64: return sx::DType::Float64;
  case safetensors::DType::F32: return sx::DType::Float32;
  case safetensors::DType::F16: return sx::DType::Float16;
  case safetensors::DType::BF16: return sx::DType::BFloat16;
  case safetensors::DType::I64: return sx::DType::Int64;
  case safetensors::DType::I32: return sx::DType::Int32;
  case safetensors::DType::I16: return sx::DType::Int16;
  case safetensors::DType::I8: return sx::DType::Int8;
  case safetensors::DType::U8: return sx::DType::UInt8;
  case safetensors::DType::BOOL: return sx::DType::Bool;
  default: return std::nullopt;  // f8 and the unsigned widths outside the set
  }
}

std::optional<sx::DType> map_ggml_dtype(std::uint32_t type) {
  switch (type) {
  case 0: return sx::DType::Float32;
  case 1: return sx::DType::Float16;
  case 24: return sx::DType::Int8;
  case 25: return sx::DType::Int16;
  case 26: return sx::DType::Int32;
  case 27: return sx::DType::Int64;
  case 28: return sx::DType::Float64;
  case 30: return sx::DType::BFloat16;
  default: return std::nullopt;  // quantized types stay in the file
  }
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1u;
  const std::uint32_t exp = (h >> 10) & 0x1fu;
  const std::uint32_t mant = h & 0x3ffu;
  std::uint32_t bits;
  if (exp == 0) {
    // Subnormal (or zero): rebias 2^-14 into the float exponent.
    if (mant == 0) {
      bits = sign << 31;
    } else {
      std::uint32_t e = 127 - 15 + 1;
      std::uint32_t m = mant;
      while ((m & 0x400u) == 0) {
        m <<= 1;
        --e;
      }
      bits = (sign << 31) | (e << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (exp == 0x1f) {
    bits = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

} // namespace

// --- open -------------------------------------------------------------------

std::expected<std::shared_ptr<WeightStore>, std::string>
WeightStore::open(const fs::path& path, const WeightStoreOptions& options) {
  std::error_code ec;
  if (!fs::is_regular_file(path, ec))
    return std::unexpected("weight file '" + path.string() + "' does not exist");
  std::ifstream probe(path, std::ios::binary);
  if (!probe)
    return std::unexpected("cannot open weight file '" + path.string() + "'");
  char magic[4] = {};
  probe.read(magic, sizeof magic);
  const bool is_gguf = probe.gcount() == 4 && std::memcmp(magic, "GGUF", 4) == 0;
  probe.close();

  WeightFormat format = options.format;
  if (format == WeightFormat::Auto)
    format = is_gguf ? WeightFormat::Gguf : WeightFormat::Safetensors;
  if ((format == WeightFormat::Gguf) != is_gguf)
    return std::unexpected("weight file '" + path.string() + "' is not " +
                           (format == WeightFormat::Gguf ? "GGUF" : "safetensors") +
                           " (format requested explicitly)");

  auto store = std::shared_ptr<WeightStore>(new WeightStore());
  store->path_ = path;
  store->format_ = format;
  store->options_ = options;

  if (format == WeightFormat::Safetensors) {
    auto reader = safetensors::Reader::load(path);
    if (!reader)
      return std::unexpected("safetensors '" + path.string() + "': " +
                             reader.error().message);
    safetensors::Reader& st = store->st_.emplace(std::move(*reader));
    for (const safetensors::TensorInfo& t : st.tensors()) {
      auto data = st.tensor_data(t);
      if (!data)
        return std::unexpected("safetensors '" + path.string() +
                               "': tensor '" + t.name +
                               "' has no readable data range");
      Entry e;
      e.dtype = map_safetensors_dtype(t.dtype);
      e.type_name = std::string(safetensors::dtype_name(t.dtype));
      e.shape.assign(t.shape.begin(), t.shape.end());
      for (std::int64_t& d : e.shape) d = static_cast<std::int64_t>(d);
      e.data = data->data();
      e.num_bytes = static_cast<std::uint64_t>(data->size());
      store->entries_.emplace(t.name, std::move(e));
    }
  } else {
    auto reader = gguf::Reader::load(path);
    if (!reader)
      return std::unexpected("GGUF '" + path.string() + "': " +
                             reader.error().message);
    gguf::Reader& gr = store->gguf_.emplace(std::move(*reader));
    for (const gguf::TensorInfo& t : gr.tensors()) {
      auto data = gr.tensor_data(t);
      if (!data)
        return std::unexpected("GGUF '" + path.string() + "': tensor '" +
                               t.name + "' has no readable data range");
      Entry e;
      e.dtype = map_ggml_dtype(t.type);
      e.type_name = std::string(gguf::ggml_type_name(t.type));
      // GGUF stores dims fastest-first; the graph names tensors row-major.
      e.shape.assign(t.dims.rbegin(), t.dims.rend());
      for (std::int64_t& d : e.shape) d = static_cast<std::int64_t>(d);
      e.data = data->data();
      e.num_bytes = static_cast<std::uint64_t>(data->size());
      store->entries_.emplace(t.name, std::move(e));
    }
  }
  return store;
}

WeightStore::~WeightStore() {
#ifdef SONICBOOM_USE_CUDA
  for (const auto& [name, handle] : device_) {
    (void)name;
    training::detail::dev_free(handle);
  }
#endif
}

// --- directory --------------------------------------------------------------

bool WeightStore::has(std::string_view name) const {
  return entries_.find(name) != entries_.end();
}

std::optional<sx::DType> WeightStore::dtype_of(std::string_view name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  return it->second.dtype;
}

std::optional<std::vector<std::int64_t>> WeightStore::shape_of(
    std::string_view name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  return it->second.shape;
}

std::optional<WeightStore::EntryInfo> WeightStore::entry_info(
    std::string_view name) const {
  auto it = entries_.find(name);
  if (it == entries_.end()) return std::nullopt;
  return EntryInfo{&it->second.type_name, &it->second.shape, it->second.num_bytes};
}

// --- upload -----------------------------------------------------------------

std::uint64_t WeightStore::device_buffer(
    std::string_view name, sx::DType want_dtype,
    const std::vector<std::int64_t>& want_shape, std::string* error) {
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    if (error)
      *error = "weight '" + std::string(name) + "' is not in '" +
               path_.string() + "'";
    return 0;
  }
  const Entry& e = it->second;
  if (e.shape != want_shape) {
    if (error)
      *error = "weight '" + std::string(name) + "' is " +
               join_shape(e.shape) + " in '" + path_.string() +
               "' but the graph expects " + join_shape(want_shape);
    return 0;
  }
  if (want_dtype != sx::DType::Float32) {
    if (error)
      *error = "weight '" + std::string(name) + "' is requested as " +
               sx::dtype_name(want_dtype) +
               "; the store supplies float32";
    return 0;
  }
  auto cached = device_.find(name);
  if (cached != device_.end()) return cached->second;

  if (!e.dtype) {
    if (error)
      *error = "weight '" + std::string(name) + "' is stored as " + e.type_name +
               ", which the float32 weight store cannot supply";
    return 0;
  }
  const std::uint64_t n = numel_of(want_shape);
  std::vector<float> scratch;
  std::vector<float>* convert_into = &scratch;
  bool cached_convert = false;
  if (*e.dtype != sx::DType::Float32) {
    auto itc = f32_cache_.try_emplace(std::string(name));
    cached_convert = itc.second;
    convert_into = &itc.first->second;
  }
  std::string msg;
  const void* src = f32_bytes_of(name, e, n, options_.convert_to_f32,
                                 convert_into, &msg);
  if (src == nullptr) {
    if (cached_convert) f32_cache_.erase(std::string(name));
    if (error) *error = msg;
    return 0;
  }

#ifndef SONICBOOM_USE_CUDA
  (void)src;
  if (error)
    *error = "SonicBoom was built without CUDA; the weight store uploads to "
             "a device buffer";
  return 0;
#else
  const std::uint64_t bytes = static_cast<std::uint64_t>(n) * sizeof(float);
  const std::uint64_t buf = training::detail::dev_alloc(bytes);
  if (buf == 0) {
    if (error)
      *error = "cannot allocate " + std::to_string(bytes) +
               " device bytes for weight '" + std::string(name) + "'";
    return 0;
  }
  if (!training::detail::dev_upload(buf, src, bytes) ||
      !training::detail::dev_sync()) {
    training::detail::dev_free(buf);
    if (error)
      *error = "cannot upload weight '" + std::string(name) + "' to the device";
    return 0;
  }
  device_.emplace(std::string(name), buf);
  resident_bytes_ += bytes;
  return buf;
#endif
}

const void* WeightStore::f32_bytes_of(std::string_view name, const Entry& e,
                                      std::uint64_t n, bool convert,
                                      std::vector<float>* out,
                                      std::string* error) const {
  // Every caller checks Entry::dtype before asking for float32 bytes; the
  // guard makes that contract local and keeps the reads below in one place.
  if (!e.dtype) {
    if (error)
      *error = "weight '" + std::string(name) + "' is stored as " + e.type_name +
               ", which the float32 weight store cannot supply";
    return nullptr;
  }
  const sx::DType dtype = *e.dtype;
  const std::uint64_t bytes = n * sizeof(float);
  if (dtype == sx::DType::Float32) {
    if (e.num_bytes != bytes) {
      if (error)
        *error = "weight '" + std::string(name) + "' holds " +
                 std::to_string(e.num_bytes) + " bytes but " +
                 std::to_string(n) + " float32 values need " +
                 std::to_string(bytes);
      return nullptr;
    }
    return e.data;
  }
  if (dtype != sx::DType::Float16 && dtype != sx::DType::BFloat16 &&
      dtype != sx::DType::Float64) {
    if (error)
      *error = "weight '" + std::string(name) + "' is stored as " + e.type_name +
               "; only float32, float16, bfloat16 and float64 can be supplied";
    return nullptr;
  }
  if (!convert) {
    if (error)
      *error = "weight '" + std::string(name) + "' is stored as " + e.type_name +
               "; open the store with convert_to_f32 to convert it to float32";
    return nullptr;
  }
  const std::uint64_t esz = sx::dtype_size(dtype);
  if (e.num_bytes != n * esz) {
    if (error)
      *error = "weight '" + std::string(name) + "' is " +
               std::to_string(e.num_bytes) + " bytes; " + std::to_string(n) +
               " values of " + e.type_name + " need " +
               std::to_string(n * esz);
    return nullptr;
  }
  out->resize(n);
  const std::byte* p = e.data;
  for (std::uint64_t i = 0; i < n; ++i) {
    if (dtype == sx::DType::Float16) {
      std::uint16_t h;
      std::memcpy(&h, p + i * 2, sizeof h);
      (*out)[i] = half_to_float(h);
    } else if (dtype == sx::DType::BFloat16) {
      std::uint16_t b;
      std::memcpy(&b, p + i * 2, sizeof b);
      const std::uint32_t bits = static_cast<std::uint32_t>(b) << 16;
      std::memcpy(&(*out)[i], &bits, sizeof(float));
    } else {
      double d;
      std::memcpy(&d, p + i * 8, sizeof d);
      (*out)[i] = static_cast<float>(d);
    }
  }
  return out->data();
}

bool WeightStore::host_f32(std::string_view name, const void** data,
                           std::uint64_t* bytes, std::string* error) {
  auto it = entries_.find(name);
  if (it == entries_.end()) {
    if (error)
      *error = "weight '" + std::string(name) + "' is not in '" +
               path_.string() + "'";
    return false;
  }
  const Entry& e = it->second;
  if (!e.dtype) {
    if (error)
      *error = "weight '" + std::string(name) + "' is stored as " + e.type_name +
               ", which the float32 weight store cannot supply";
    return false;
  }
  const std::uint64_t n = numel_of(e.shape);
  std::vector<float> scratch;
  std::vector<float>* convert_into = &scratch;
  bool cached_convert = false;
  if (*e.dtype != sx::DType::Float32) {
    auto itc = f32_cache_.try_emplace(std::string(name));
    cached_convert = itc.second;
    convert_into = &itc.first->second;
  }
  std::string msg;
  const void* src = f32_bytes_of(name, e, n, options_.convert_to_f32,
                                 convert_into, &msg);
  if (src == nullptr) {
    if (cached_convert) f32_cache_.erase(std::string(name));
    if (error) *error = msg;
    return false;
  }
  if (data) *data = src;
  if (bytes) *bytes = static_cast<std::uint64_t>(n) * sizeof(float);
  return true;
}

} // namespace sonicboom::device
