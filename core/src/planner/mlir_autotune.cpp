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

#include "mlir_autotune.h"

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>

namespace sonicboom::mlir_tune {

namespace {

// The file is a fixed-layout record array behind a 16-byte header. Records are
// plain old data so a process can read them straight out of the mapping.
constexpr uint32_t kMagic = 0x31414253u;  // "SBA1"
constexpr uint32_t kVersion = 1;
constexpr uint32_t kCapacity = 256;
constexpr size_t kOpBytes = 16;
constexpr size_t kShapeBytes = 48;

struct Header {
  uint32_t magic;
  uint32_t version;
  uint32_t capacity;
  uint32_t count;
};

struct Record {
  char op[kOpBytes];
  char shape[kShapeBytes];
  int32_t sm;
  TileConfig cfg;
  int64_t stamp;
};

static_assert(sizeof(TileConfig) == 20, "TileConfig is five int32 fields");
static_assert(sizeof(Header) == 16, "cache header layout");
static_assert(sizeof(Record) == 96, "cache record layout");

constexpr size_t kFileBytes = sizeof(Header) + kCapacity * sizeof(Record);

std::string cache_dir() {
  if (const char* e = std::getenv("XDG_CACHE_HOME"); e && *e)
    return std::string(e) + "/sonicboom/autotune";
  if (const char* e = std::getenv("HOME"); e && *e)
    return std::string(e) + "/.cache/sonicboom/autotune";
  return "/tmp/sonicboom-autotune-" + std::to_string(::getuid());
}

std::string sm_file_name() {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "sm_%02d.bin", sm_version());
  return buf;
}

int open_cache_file() {
  const std::string dir = cache_dir();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  const std::string path = dir + "/" + sm_file_name();
  return ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
}

// Copy a key into its fixed-size field, truncating and always terminating.
void set_field(char* dst, size_t n, std::string_view s) {
  const size_t len = std::min(n - 1, s.size());
  std::memcpy(dst, s.data(), len);
  dst[len] = '\0';
}

bool field_matches(const char* field, size_t n, std::string_view s) {
  if (s.size() >= n) return false;  // longer keys are stored truncated
  return std::strlen(field) == s.size() &&
         std::memcmp(field, s.data(), s.size()) == 0;
}

bool op_matches(const Record& r, std::string_view op, std::string_view shape) {
  return field_matches(r.op, kOpBytes, op) &&
         field_matches(r.shape, kShapeBytes, shape);
}

} // namespace

std::string TileConfig::to_string() const {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "%d,%d,%d,%d,%d", tile_m, tile_n, tile_k,
                unroll, vectorize);
  return buf;
}

std::optional<TileConfig> TileConfig::from_string(std::string_view s) {
  int v[5] = {};
  size_t pos = 0;
  for (int i = 0; i < 5; ++i) {
    size_t end = s.find(',', pos);
    if (i < 4 && end == std::string_view::npos) return std::nullopt;
    if (i == 4) end = s.size();
    const std::string_view field = s.substr(pos, end - pos);
    if (field.empty()) return std::nullopt;
    char* tail = nullptr;
    const long x = std::strtol(std::string(field).c_str(), &tail, 10);
    if (!tail || *tail != '\0' || x <= 0 || x > 1 << 20) return std::nullopt;
    v[i] = int(x);
    pos = end + 1;
  }
  if (pos != s.size() + 1) return std::nullopt;  // trailing text after field 5
  TileConfig c{v[0], v[1], v[2], v[3], v[4]};
  return c;
}

bool enabled() {
  const char* e = std::getenv("SONICBOOM_AUTOTUNE");
  return e && std::strcmp(e, "1") == 0;
}

int sm_version() {
  static const int v = [] {
#ifdef SONICBOOM_USE_CUDA
    int major = 0, minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
            cudaSuccess &&
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0) ==
            cudaSuccess) {
      return major * 10 + minor;
    }
#endif
    return 0;
  }();
  return v;
}

TileConfig default_config(std::string_view op) {
  (void)op;
  // The pre-Q1 launch shape: tile_m 256 is the one-thread-per-super-block gemv
  // block, tile_k 32 is the quant super-block, one row per tile.
  return TileConfig{256, 1, 32, 1, 1};
}

std::span<const TileConfig> gemv_candidates() {
  static const TileConfig kCandidates[] = {
      {128, 1, 32, 1, 1}, {128, 1, 64, 1, 1}, {256, 1, 32, 1, 1},
      {256, 1, 64, 1, 1}, {512, 1, 32, 1, 1}, {512, 1, 64, 1, 1},
  };
  return kCandidates;
}

AutotuneCache& AutotuneCache::instance() {
  static AutotuneCache cache;
  return cache;
}

std::string AutotuneCache::path() const {
  if (path_.empty()) path_ = cache_dir() + "/" + sm_file_name();
  return path_;
}

void AutotuneCache::reset_for_test() { path_.clear(); }

std::optional<TileConfig> AutotuneCache::lookup(std::string_view op,
                                                std::string_view shape_class) {
  const int fd = open_cache_file();
  if (fd < 0) return std::nullopt;
  std::optional<TileConfig> found;
  if (::flock(fd, LOCK_SH) == 0) {
    struct stat st {};
    if (::fstat(fd, &st) == 0 && st.st_size >= off_t(sizeof(Header))) {
      const size_t bytes =
          std::min<size_t>(size_t(st.st_size), kFileBytes);
      void* m = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
      if (m != MAP_FAILED) {
        const Header* h = static_cast<const Header*>(m);
        const size_t avail = (bytes - sizeof(Header)) / sizeof(Record);
        const size_t count =
            std::min<size_t>(h->count, std::min<size_t>(h->capacity, avail));
        if (h->magic == kMagic && h->version == kVersion) {
          const Record* recs = reinterpret_cast<const Record*>(h + 1);
          for (size_t i = 0; i < count; ++i) {
            if (op_matches(recs[i], op, shape_class)) {
              found = recs[i].cfg;
              break;
            }
          }
        }
        ::munmap(m, bytes);
      }
    }
    ::flock(fd, LOCK_UN);
  }
  ::close(fd);
  return found;
}

bool AutotuneCache::store(std::string_view op, std::string_view shape_class,
                          const TileConfig& cfg) {
  const int fd = open_cache_file();
  if (fd < 0) return false;
  bool ok = false;
  if (::flock(fd, LOCK_EX) == 0) {
    struct stat st {};
    bool ready = false;
    if (::fstat(fd, &st) == 0) {
      ready = size_t(st.st_size) >= kFileBytes;
      if (!ready) ready = ::ftruncate(fd, off_t(kFileBytes)) == 0;
    }
    if (ready) {
      void* m = ::mmap(nullptr, kFileBytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                       fd, 0);
      if (m != MAP_FAILED) {
        Header* h = static_cast<Header*>(m);
        if (h->magic != kMagic || h->version != kVersion) {
          std::memset(m, 0, kFileBytes);
          h->magic = kMagic;
          h->version = kVersion;
          h->capacity = kCapacity;
          h->count = 0;
        }
        Record* recs = reinterpret_cast<Record*>(h + 1);
        const size_t count = std::min<size_t>(h->count, kCapacity);
        size_t slot = count;
        for (size_t i = 0; i < count; ++i) {
          if (op_matches(recs[i], op, shape_class)) {
            slot = i;
            break;
          }
        }
        if (slot == count) {
          if (count < kCapacity) {
            ++h->count;
          } else {
            // Full: replace the oldest record (smallest stamp).
            size_t oldest = 0;
            for (size_t i = 1; i < count; ++i)
              if (recs[i].stamp < recs[oldest].stamp) oldest = i;
            slot = oldest;
          }
        }
        Record& r = recs[slot];
        std::memset(&r, 0, sizeof(r));
        set_field(r.op, kOpBytes, op);
        set_field(r.shape, kShapeBytes, shape_class);
        r.sm = sm_version();
        r.cfg = cfg;
        r.stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        ok = ::msync(m, kFileBytes, MS_SYNC) == 0;
        ::munmap(m, kFileBytes);
      }
    }
    ::flock(fd, LOCK_UN);
  }
  ::close(fd);
  return ok;
}

std::string gemv_shape_class(int blocks_per_row, int num_rows) {
  // Rows bucket to the next power of two in [128, 16384]; the blocks-per-row
  // count already clusters the model's matvecs (8..64) and is kept exact.
  int rows_bucket = 128;
  while (rows_bucket < num_rows && rows_bucket < 16384) rows_bucket <<= 1;
  return "bpr" + std::to_string(blocks_per_row) + "-r" +
         std::to_string(rows_bucket);
}

TileConfig gemv_config(int blocks_per_row, int num_rows) {
  if (!enabled()) return default_config("gemv");
  static std::mutex mu;
  static std::unordered_map<std::string, TileConfig> memo;
  const std::string key = gemv_shape_class(blocks_per_row, num_rows);
  std::lock_guard<std::mutex> lock(mu);
  if (const auto it = memo.find(key); it != memo.end()) return it->second;
  TileConfig cfg =
      AutotuneCache::instance().lookup("gemv", key).value_or(
          default_config("gemv"));
  memo.emplace(key, cfg);
  return cfg;
}

std::optional<TileConfig> AutotuneRunner::run(const Problem& problem) {
  if (!enabled() || problem.candidates.empty() || !problem.compile)
    return std::nullopt;

  const int iters = problem.iters > 0 ? problem.iters : 5;
  std::vector<unsigned char> reference(problem.output_bytes);
  auto snapshot = [&] {
    if (problem.sync) problem.sync();
    if (problem.output && problem.output_bytes)
      std::memcpy(reference.data(), problem.output, problem.output_bytes);
  };

  // Candidate 0 is the reference: its output is the bit-identity window and
  // its time is the number to beat.
  auto best_launch = problem.compile(problem.candidates[0]);
  if (!best_launch) return std::nullopt;
  (*best_launch)();
  snapshot();

  auto time_launch = [&](const Launch& launch) {
    uint64_t best = std::numeric_limits<uint64_t>::max();
    for (int i = 0; i < iters; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      launch();
      if (problem.sync) problem.sync();
      const auto t1 = std::chrono::steady_clock::now();
      best = std::min<uint64_t>(
          best,
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    return best;
  };

  TileConfig best_cfg = problem.candidates[0];
  uint64_t best_ns = time_launch(*best_launch);

  for (size_t i = 1; i < problem.candidates.size(); ++i) {
    auto launch = problem.compile(problem.candidates[i]);
    if (!launch) continue;
    (*launch)();
    if (problem.sync) problem.sync();
    if (problem.output && problem.output_bytes &&
        std::memcmp(reference.data(), problem.output, problem.output_bytes) !=
            0) {
      continue;  // numerically different from the reference: not a candidate
    }
    const uint64_t ns = time_launch(*launch);
    if (ns < best_ns) {
      best_ns = ns;
      best_cfg = problem.candidates[i];
      best_launch = std::move(launch);
    }
  }
  // Probing a rejected candidate leaves its output in the window; run the
  // winner once more so the caller's buffer holds the returned config's
  // result, not the last probe's.
  (*best_launch)();
  if (problem.sync) problem.sync();
  return best_cfg;
}

} // namespace sonicboom::mlir_tune
