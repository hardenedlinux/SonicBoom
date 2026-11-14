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

#include "node_dump.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <utility>

namespace sonicboom::planner {

namespace {

bool is_integer(sx::DType d) {
  switch (d) {
    case sx::DType::Int8:
    case sx::DType::UInt8:
    case sx::DType::Int16:
    case sx::DType::Int32:
    case sx::DType::Int64:
    case sx::DType::Bool:
      return true;
    default:
      return false;
  }
}

// Element `k` of an integer buffer as int32. The graph's integer tensors are
// narrow (expert ids, token ids, masks); a 64-bit value past int32 does not
// occur and would be truncated, which the dump's int32 format implies.
int32_t read_int(const void *data, std::size_t k, sx::DType d) {
  const auto *p = static_cast<const std::byte *>(data) + k * sx::dtype_size(d);
  switch (d) {
    case sx::DType::Int8: {
      int8_t v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
    case sx::DType::UInt8:
    case sx::DType::Bool: {
      uint8_t v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
    case sx::DType::Int16: {
      int16_t v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
    case sx::DType::Int32: {
      int32_t v;
      std::memcpy(&v, p, sizeof(v));
      return v;
    }
    case sx::DType::Int64: {
      int64_t v;
      std::memcpy(&v, p, sizeof(v));
      return static_cast<int32_t>(v);
    }
    default:
      return 0;
  }
}

} // namespace

std::unique_ptr<NodeDump> NodeDump::create() {
  const char *names = std::getenv("SONICBOOM_DUMP_NODES");
  const char *dir = std::getenv("SONICBOOM_DUMP_DIR");
  if (names == nullptr || *names == '\0' || dir == nullptr || *dir == '\0')
    return nullptr;
  std::vector<std::string> prefixes;
  std::string_view rest{names};
  while (!rest.empty()) {
    const std::size_t comma = rest.find(',');
    const std::string_view prefix = rest.substr(0, comma);
    if (!prefix.empty()) prefixes.emplace_back(prefix);
    if (comma == std::string_view::npos) break;
    rest.remove_prefix(comma + 1);
  }
  if (prefixes.empty()) return nullptr;
  // A path that cannot be created is not fatal here: the writes then fail and
  // the run proceeds undumped.
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  return std::unique_ptr<NodeDump>(new NodeDump(dir, std::move(prefixes)));
}

NodeDump::NodeDump(std::string dir, std::vector<std::string> prefixes)
    : dir_(std::move(dir)), prefixes_(std::move(prefixes)) {}

bool NodeDump::matches(std::string_view node) const {
  for (const std::string &prefix : prefixes_)
    if (node.starts_with(prefix)) return true;
  return false;
}

void NodeDump::write(std::string_view node, const void *data,
                     std::size_t bytes, sx::DType dtype) const {
  if (data == nullptr || bytes == 0 || !matches(node)) return;

  // The dump format is float32 for a float32 output and int32 for every
  // integer width (the MoE route's expert ids are int64 in the graph).
  std::vector<int32_t> converted;
  const void *out = data;
  std::size_t out_bytes = bytes;
  if (dtype == sx::DType::Float32) {
    // The host bytes are already the format.
  } else if (is_integer(dtype)) {
    const std::size_t elem = sx::dtype_size(dtype);
    if (elem == 0 || bytes % elem != 0) return;
    const std::size_t n = bytes / elem;
    converted.resize(n);
    for (std::size_t k = 0; k < n; ++k) converted[k] = read_int(data, k, dtype);
    out = converted.data();
    out_bytes = converted.size() * sizeof(int32_t);
  } else {
    return; // float16/bfloat16/float64 outputs are outside the dump format
  }

  // begin_graph() runs before the nodes, so the execution in progress is one
  // past the count of completed ones: its index is step_ - 1.
  const uint64_t index = step_ == 0 ? 0 : step_ - 1;
  char prefix[24];
  std::snprintf(prefix, sizeof(prefix), "%06llu-",
                static_cast<unsigned long long>(index));
  std::string path = dir_;
  path += '/';
  path += prefix;
  path += node;
  path += ".bin";
  std::FILE *f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) return;
  std::fwrite(out, 1, out_bytes, f);
  std::fclose(f);
}

} // namespace sonicboom::planner
