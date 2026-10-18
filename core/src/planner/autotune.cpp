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

#include "autotune.h"

#include <sonicboom/runtime_features.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#ifdef SONICBOOM_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace sonicboom::tune {

namespace {

std::mutex g_mutex;
std::unordered_map<std::string, std::map<std::string, std::map<std::string, int>>>
    g_cache;  // op -> key -> values
std::unordered_map<std::string, bool> g_loaded;  // op -> file read already
std::string g_device_dir;

std::string sanitize(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  for (char c : in)
    out.push_back(std::isalnum(static_cast<unsigned char>(c)) || c == '.' ||
                          c == '_' || c == '-'
                      ? c
                      : '_');
  return out;
}

std::string compute_device_dir() {
#ifdef SONICBOOM_USE_CUDA
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return "cpu";
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return "cpu";
  const int arch = prop.major * 10 + prop.minor;
  int driver = 0;
  const std::string drv =
      (cuDriverGetVersion(&driver) == CUDA_SUCCESS && driver > 0)
          ? (std::to_string(driver / 1000) + "." +
             std::to_string((driver % 1000) / 10))
          : std::string("unknown");
  return sanitize(prop.name) + "-sm" + std::to_string(arch) + "-" + drv;
#else
  return "cpu";
#endif
}

std::string cache_root() {
  if (const char *e = std::getenv("SONICBOOM_TUNE_DIR"))
    if (*e != '\0') return e;
  if (const char *e = std::getenv("XDG_CACHE_HOME"))
    if (*e != '\0') return std::string(e) + "/sonicboom/tune";
  if (const char *e = std::getenv("HOME"))
    if (*e != '\0') return std::string(e) + "/.cache/sonicboom/tune";
  return ".sonicboom-tune";
}

// Caller holds g_mutex (the public device_dir() is the locking entry point;
// the cache-path helpers below run with the lock already held and must not
// take it again).
const std::string &device_dir_locked() {
  if (g_device_dir.empty()) g_device_dir = compute_device_dir();
  return g_device_dir;
}

std::string cache_path_locked(const std::string &op) {
  return (std::filesystem::path(cache_root()) / device_dir_locked() /
          (std::string(op) + ".json"))
      .string();
}

// --- minimal JSON (object of string -> object of string -> int) ------------

void skip_ws(std::string_view s, size_t &i) {
  while (i < s.size() &&
         std::isspace(static_cast<unsigned char>(s[i])))
    ++i;
}

bool parse_string(std::string_view s, size_t &i, std::string &out) {
  skip_ws(s, i);
  if (i >= s.size() || s[i] != '"') return false;
  ++i;
  out.clear();
  while (i < s.size() && s[i] != '"') {
    if (s[i] == '\\' && i + 1 < s.size()) {
      out.push_back(s[i + 1]);
      i += 2;
      continue;
    }
    out.push_back(s[i++]);
  }
  if (i >= s.size()) return false;
  ++i;  // closing quote
  return true;
}

bool parse_int(std::string_view s, size_t &i, int &out) {
  skip_ws(s, i);
  const size_t start = i;
  if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
  while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i]))) ++i;
  if (i == start) return false;
  out = std::atoi(std::string(s.substr(start, i - start)).c_str());
  return true;
}

bool expect(std::string_view s, size_t &i, char c) {
  skip_ws(s, i);
  if (i >= s.size() || s[i] != c) return false;
  ++i;
  return true;
}

// Parses one `<op>.json` body: {"key":{"k":v,...},...}. Malformed input
// yields whatever was parsed successfully; the cache is advisory, so a
// corrupt entry just costs a retune.
void parse_op_file(std::string_view s,
                   std::map<std::string, std::map<std::string, int>> &out) {
  size_t i = 0;
  if (!expect(s, i, '{')) return;
  skip_ws(s, i);
  if (i < s.size() && s[i] == '}') return;
  while (i < s.size()) {
    std::string key;
    if (!parse_string(s, i, key)) return;
    if (!expect(s, i, ':')) return;
    if (!expect(s, i, '{')) return;
    std::map<std::string, int> values;
    skip_ws(s, i);
    if (i < s.size() && s[i] == '}') {
      ++i;
    } else {
      while (i < s.size()) {
        std::string name;
        int value = 0;
        if (!parse_string(s, i, name) || !expect(s, i, ':') ||
            !parse_int(s, i, value))
          return;
        values[name] = value;
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') {
          ++i;
          continue;
        }
        if (!expect(s, i, '}')) return;
        break;
      }
    }
    out[key] = std::move(values);
    skip_ws(s, i);
    if (i < s.size() && s[i] == ',') {
      ++i;
      continue;
    }
    return;
  }
}

void load_op_locked(const std::string &op) {
  if (g_loaded[op]) return;
  g_loaded[op] = true;
  std::ifstream in(cache_path_locked(op), std::ios::binary);
  if (!in) return;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  parse_op_file(body, g_cache[op]);
}

void write_op_locked(const std::string &op) {
  const std::string path = cache_path_locked(op);
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path(), ec);
  std::ostringstream os;
  os << "{\n";
  bool first_key = true;
  for (const auto &[key, values] : g_cache[op]) {
    if (!first_key) os << ",\n";
    first_key = false;
    os << "  \"" << key << "\": {";
    bool first_value = true;
    for (const auto &[name, value] : values) {
      if (!first_value) os << ", ";
      first_value = false;
      os << "\"" << name << "\": " << value;
    }
    os << "}";
  }
  os << "\n}\n";
  // Write-then-rename so a concurrent process (ctest runs several GPU tests
  // in parallel) sees either the old or the new file, never a torn one.
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return;
    out << os.str();
    if (!out) return;
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
  }
}

} // namespace

bool enabled() {
  // SONICBOOM_FEATURE_DISABLE=autotune is the ablation-group alias of the
  // per-feature switch; either one turns the tuner off.
  if (::sonicboom::runtime::feature_disabled(
          ::sonicboom::runtime::Feature::Autotune))
    return false;
  const char *e = std::getenv("SONICBOOM_AUTOTUNE");
  return e == nullptr || std::strcmp(e, "0") != 0;
}

bool force() {
  const char *e = std::getenv("SONICBOOM_AUTOTUNE");
  return e != nullptr && std::strcmp(e, "force") == 0;
}

const std::string &device_dir() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return device_dir_locked();
}

std::string cache_path(std::string_view op) {
  std::lock_guard<std::mutex> lk(g_mutex);
  return cache_path_locked(std::string(op));
}

std::map<std::string, int> lookup(std::string_view op, std::string_view key) {
  if (!enabled() || force()) return {};
  std::lock_guard<std::mutex> lk(g_mutex);
  const std::string op_s(op);
  load_op_locked(op_s);
  const auto it = g_cache[op_s].find(std::string(key));
  if (it == g_cache[op_s].end()) return {};
  return it->second;
}

void store(std::string_view op, std::string_view key,
           const std::map<std::string, int> &values) {
  if (!enabled()) return;
  std::lock_guard<std::mutex> lk(g_mutex);
  const std::string op_s(op);
  load_op_locked(op_s);
  g_cache[op_s][std::string(key)] = values;
  write_op_locked(op_s);
}

void clear_memory_cache() {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_cache.clear();
  g_loaded.clear();
}

double median_launch_ms(const std::function<bool()> &launch,
                        cudaStream_t stream, int warmups, int reps) {
#ifdef SONICBOOM_USE_CUDA
  for (int i = 0; i < warmups; ++i) {
    // A candidate that cannot launch is dropped after one attempt; the loader
    // has already reported why, and retrying would repeat the message.
    if (!launch() || cudaGetLastError() != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      (void)cudaGetLastError();
      return -1.0;
    }
  }
  cudaEvent_t begin = nullptr, end = nullptr;
  if (cudaEventCreate(&begin) != cudaSuccess ||
      cudaEventCreate(&end) != cudaSuccess) {
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    return -1.0;
  }
  std::vector<double> samples;
  samples.reserve(static_cast<size_t>(std::max(reps, 1)));
  for (int r = 0; r < reps; ++r) {
    if (cudaEventRecord(begin, stream) != cudaSuccess || !launch() ||
        cudaEventRecord(end, stream) != cudaSuccess ||
        cudaEventSynchronize(end) != cudaSuccess) {
      (void)cudaGetLastError();
      break;
    }
    float ms = 0.0f;
    if (cudaEventElapsedTime(&ms, begin, end) != cudaSuccess) break;
    samples.push_back(double(ms));
  }
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  if (samples.empty()) return -1.0;
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
#else
  (void)launch;
  (void)warmups;
  (void)reps;
  return -1.0;
#endif
}

} // namespace sonicboom::tune
