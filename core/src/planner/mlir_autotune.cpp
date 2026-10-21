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

#include "decision_registry.h"

#include <sonicboom/runtime_features.h>

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
#include <cmath>
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
// Version 2 widens the op field from 16 to 32 bytes so a full op name (V2's
// "spec_verify_ratio") fits. Version 3 changes the "attn_prefill" TileConfig
// meaning from (tile_seq, tile_head) to (br, dc, bc, as), so a version-2
// record must not be applied to the new grid. Version 4 splits gemv rows
// above 16384 into their own buckets, so a version-3 "bprN-r16384" record
// would be applied to a class it was not measured for. An older file misses
// on lookup and is rewritten by the first store, which is the cache's own
// reset path.
constexpr uint32_t kVersion = 4;
constexpr uint32_t kCapacity = 256;
constexpr size_t kOpBytes = 32;
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
static_assert(sizeof(Record) == 112, "cache record layout");

constexpr size_t kFileBytes = sizeof(Header) + kCapacity * sizeof(Record);

// The three choice enums share their value set; map it onto the registry's
// source vocabulary (design/decision-registry.md).
template <typename SourceT>
planner::decisions::Source registry_source(SourceT source) {
  switch (source) {
    case SourceT::Cache: return planner::decisions::Source::Cached;
    case SourceT::Measured: return planner::decisions::Source::Measured;
    case SourceT::Skipped: return planner::decisions::Source::Skipped;
    case SourceT::Default: break;
  }
  return planner::decisions::Source::Default;
}

// Every JIT-candidate pick is a decision for one device and shape class. The
// env field names the switch that overrides the whole service; a single class
// is overridden by retiming it (SONICBOOM_AUTOTUNE=force).
void record_choice(std::string_view op, const std::string& shape_class,
                   std::string_view value,
                   planner::decisions::Source source) {
  planner::decisions::record(op, shape_class, value, source,
                             "SONICBOOM_AUTOTUNE");
}

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
  // Same default as the hand-written tuner (autotune.h): on unless =0, with
  // the runtime `autotune` feature (SONICBOOM_FEATURE_DISABLE=autotune) as the
  // ablation switch covering both services.
  if (::sonicboom::runtime::feature_disabled(
          ::sonicboom::runtime::Feature::Autotune))
    return false;
  // The JIT-candidate service has its own ablation switch, so a run can keep
  // the hand-written tuner's choices and drop only the generated candidates
  // (the plan-record goldens use it).
  const char* j = std::getenv("SONICBOOM_AUTOTUNE_JIT");
  if (j && std::strcmp(j, "0") == 0) return false;
  const char* e = std::getenv("SONICBOOM_AUTOTUNE");
  return e == nullptr || std::strcmp(e, "0") != 0;
}

bool force() {
  const char* e = std::getenv("SONICBOOM_AUTOTUNE");
  return e && std::strcmp(e, "force") == 0;
}

namespace {

// One budget per process, started lazily by the first unresolved class. A
// warm process (every class cached) never calls budget_remaining_ms(), so it
// never starts the clock. Both the runner and the per-class choice functions
// consult this; the callers' confirmation re-measurement loops check it too, so
// the total cold cost stays inside the budget.
struct BudgetState {
  bool started = false;
  bool unlimited = false;
  std::chrono::steady_clock::time_point start;
  int64_t budget_ms = 0;
};

std::mutex& budget_mutex() {
  static std::mutex m;
  return m;
}

BudgetState& budget_state() {
  static BudgetState s;
  return s;
}

int64_t configured_budget_ms() {
  const char* e = std::getenv("SONICBOOM_AUTOTUNE_BUDGET_MS");
  if (e && *e) {
    char* tail = nullptr;
    const long long v = std::strtoll(e, &tail, 10);
    if (tail && *tail == '\0') return int64_t(v);
  }
  return 10000;  // a first run on a new device measures for at most ~10 s
}

} // namespace

int64_t budget_remaining_ms() {
  std::lock_guard<std::mutex> lock(budget_mutex());
  BudgetState& s = budget_state();
  if (!s.started) {
    s.started = true;
    s.start = std::chrono::steady_clock::now();
    s.budget_ms = configured_budget_ms();
    s.unlimited = s.budget_ms <= 0;
  }
  if (s.unlimited) return std::numeric_limits<int64_t>::max();
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now() - s.start)
                           .count();
  return std::max<int64_t>(0, s.budget_ms - int64_t(elapsed));
}

bool budget_exhausted() { return budget_remaining_ms() <= 0; }

void reset_budget_for_test() {
  std::lock_guard<std::mutex> lock(budget_mutex());
  budget_state() = BudgetState{};
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

std::vector<DiskRecord> disk_records() {
  std::vector<DiskRecord> out;
  const std::string file = cache_dir() + "/" + sm_file_name();
  const int fd = ::open(file.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) return out;
  if (::flock(fd, LOCK_SH) == 0) {
    struct stat st {};
    if (::fstat(fd, &st) == 0 && st.st_size >= off_t(sizeof(Header))) {
      const size_t bytes = std::min<size_t>(size_t(st.st_size), kFileBytes);
      void* m = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
      if (m != MAP_FAILED) {
        const Header* h = static_cast<const Header*>(m);
        const size_t avail = (bytes - sizeof(Header)) / sizeof(Record);
        const size_t count =
            std::min<size_t>(h->count, std::min<size_t>(h->capacity, avail));
        if (h->magic == kMagic && h->version == kVersion) {
          const Record* recs = reinterpret_cast<const Record*>(h + 1);
          out.reserve(count);
          for (size_t i = 0; i < count; ++i) {
            out.push_back(DiskRecord{
                std::string(recs[i].op, ::strnlen(recs[i].op, kOpBytes)),
                std::string(recs[i].shape,
                            ::strnlen(recs[i].shape, kShapeBytes)),
                recs[i].cfg});
          }
        }
        ::munmap(m, bytes);
      }
    }
    ::flock(fd, LOCK_UN);
  }
  ::close(fd);
  std::sort(out.begin(), out.end(),
            [](const DiskRecord& a, const DiskRecord& b) {
              return a.op != b.op ? a.op < b.op
                                  : a.shape_class < b.shape_class;
            });
  return out;
}

bool store_ratio(std::string_view op, std::string_view shape, double ratio) {
  if (!(ratio > 0.0)) return false;
  const double milli = ratio * 1000.0;
  if (milli < 1.0 || milli > 1048575.0) return false;
  // The scalar lives in tile_m; the remaining fields are a fixed placeholder
  // so the record is a well-formed TileConfig (see the header).
  const TileConfig cfg{int32_t(std::llround(milli)), 1, 1, 1, 1};
  if (!AutotuneCache::instance().store(op, shape, cfg)) return false;
  char value[32];
  std::snprintf(value, sizeof(value), "%.3f", ratio);
  planner::decisions::record(op, shape, value,
                             planner::decisions::Source::Measured);
  return true;
}

std::optional<double> lookup_ratio(std::string_view op,
                                   std::string_view shape) {
  const auto rec = AutotuneCache::instance().lookup(op, shape);
  if (!rec || rec->tile_n != 1 || rec->tile_k != 1 || rec->unroll != 1 ||
      rec->vectorize != 1)
    return std::nullopt;
  const double ratio = double(rec->tile_m) / 1000.0;
  char value[32];
  std::snprintf(value, sizeof(value), "%.3f", ratio);
  planner::decisions::record(op, shape, value,
                             planner::decisions::Source::Cached);
  return ratio;
}

std::string gemv_shape_class(int blocks_per_row, int num_rows) {
  // Rows bucket to the next power of two in [128, 262144]; the blocks-per-row
  // count already clusters the model's matvecs (8..64) and is kept exact.
  // The buckets above 16384 keep a tall output (the 262144-row logits
  // projection) from sharing a class with a 10240-row FFN, whose best launch
  // width is wider than gridDim.y allows for the tall shape.
  int rows_bucket = 128;
  while (rows_bucket < num_rows && rows_bucket < 262144) rows_bucket <<= 1;
  return "bpr" + std::to_string(blocks_per_row) + "-r" +
         std::to_string(rows_bucket);
}

namespace {
std::mutex& gemv_memo_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<std::string, GemvChoice>& gemv_memo() {
  static std::unordered_map<std::string, GemvChoice> memo;
  return memo;
}
} // namespace

GemvChoice gemv_choice(int blocks_per_row, int num_rows) {
  if (!enabled()) {
    record_choice("tune/gemv", gemv_shape_class(blocks_per_row, num_rows),
                  default_config("gemv").to_string(),
                  planner::decisions::Source::Default);
    return GemvChoice{default_config("gemv"), GemvSource::Default};
  }
  const std::string key = gemv_shape_class(blocks_per_row, num_rows);
  {
    std::lock_guard<std::mutex> lock(gemv_memo_mutex());
    const auto it = gemv_memo().find(key);
    if (it != gemv_memo().end()) return it->second;
  }
  if (!force()) {
    if (const auto cached = AutotuneCache::instance().lookup("gemv", key)) {
      const GemvChoice choice{*cached, GemvSource::Cache};
      record_choice("tune/gemv", key, cached->to_string(),
                    registry_source(choice.source));
      std::lock_guard<std::mutex> lock(gemv_memo_mutex());
      gemv_memo().emplace(key, choice);
      return choice;
    }
  }
  // Out of cold budget: keep the static default for this class for the rest of
  // the process. Memoized as Skipped (which resolved() reports true) so the
  // caller's per-launch check does not rebuild the probe problem every call.
  if (budget_exhausted()) {
    const GemvChoice choice{default_config("gemv"), GemvSource::Skipped};
    record_choice("tune/gemv", key, choice.cfg.to_string(),
                  registry_source(choice.source));
    std::lock_guard<std::mutex> lock(gemv_memo_mutex());
    gemv_memo().emplace(key, choice);
    return choice;
  }
  return GemvChoice{default_config("gemv"), GemvSource::Default};
}

bool gemv_choice_store(int blocks_per_row, int num_rows,
                       const TileConfig& cfg) {
  const std::string key = gemv_shape_class(blocks_per_row, num_rows);
  record_choice("tune/gemv", key, cfg.to_string(),
                planner::decisions::Source::Measured);
  {
    std::lock_guard<std::mutex> lock(gemv_memo_mutex());
    gemv_memo()[key] = GemvChoice{cfg, GemvSource::Measured};
  }
  return AutotuneCache::instance().store("gemv", key, cfg);
}

TileConfig gemv_config(int blocks_per_row, int num_rows) {
  return gemv_choice(blocks_per_row, num_rows).cfg;
}

TileConfig mmq_tc_default_config() { return TileConfig{64, 64, 32, 1, 1}; }

std::span<const TileConfig> mmq_tc_candidates() {
  // tile_k 16 is pruned: the kernel stages one m16n8k32 mma per K chunk, so a
  // 16-K chunk has no generator. candidates[0] is the smallest config and is
  // the runner's bit-identity reference.
  static const TileConfig kCandidates[] = {
      {16, 16, 32, 1, 1},  {16, 32, 32, 1, 1},  {16, 64, 32, 1, 1},
      {32, 16, 32, 1, 1},  {32, 32, 32, 1, 1},  {32, 64, 32, 1, 1},
      {64, 16, 32, 1, 1},  {64, 32, 32, 1, 1},  {64, 64, 32, 1, 1},
      {128, 16, 32, 1, 1}, {128, 32, 32, 1, 1}, {128, 64, 32, 1, 1},
  };
  return kCandidates;
}

std::string mmq_tc_shape_class(int m_tokens, int n_rows, int k) {
  // Token-count buckets, one per order of magnitude in the low range. A token
  // count below the first bucket (the unbatched path never reaches here) is
  // clamped to it.
  static constexpr struct {
    int hi;
    const char* name;
  } kBuckets[] = {
      {7, "4-7"},      {15, "8-15"},    {31, "16-31"},   {63, "32-63"},
      {127, "64-127"}, {255, "128-255"}, {511, "256-511"},
      {1023, "512-1023"}, {std::numeric_limits<int>::max(), "1024+"},
  };
  const char* bucket = kBuckets[0].name;
  for (const auto& b : kBuckets) {
    if (m_tokens <= b.hi) {
      bucket = b.name;
      break;
    }
  }
  // N and K stay exact: they are the weight's own dimensions and the model has
  // only a handful of each.
  return std::string("m") + bucket + "-n" + std::to_string(n_rows) + "-k" +
         std::to_string(k);
}

namespace {
std::mutex& mmq_tc_memo_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<std::string, MmqTcChoice>& mmq_tc_memo() {
  static std::unordered_map<std::string, MmqTcChoice> memo;
  return memo;
}
} // namespace

MmqTcChoice mmq_tc_choice(int m_tokens, int n_rows, int k) {
  if (!enabled()) {
    record_choice("tune/matmul_q8_tc",
                  mmq_tc_shape_class(m_tokens, n_rows, k),
                  mmq_tc_default_config().to_string(),
                  planner::decisions::Source::Default);
    return MmqTcChoice{mmq_tc_default_config(), MmqTcSource::Default};
  }
  const std::string key = mmq_tc_shape_class(m_tokens, n_rows, k);
  {
    std::lock_guard<std::mutex> lock(mmq_tc_memo_mutex());
    const auto it = mmq_tc_memo().find(key);
    if (it != mmq_tc_memo().end()) return it->second;
  }
  // force() retunes: the cache read is skipped and the caller measures.
  if (!force()) {
    if (const auto cached =
            AutotuneCache::instance().lookup("matmul_q8_tc", key)) {
      const MmqTcChoice choice{*cached, MmqTcSource::Cache};
      record_choice("tune/matmul_q8_tc", key, cached->to_string(),
                    registry_source(choice.source));
      std::lock_guard<std::mutex> lock(mmq_tc_memo_mutex());
      mmq_tc_memo().emplace(key, choice);
      return choice;
    }
  }
  // Out of cold budget: static default for this class for the rest of the
  // process, memoized so it is not probed again.
  if (budget_exhausted()) {
    const MmqTcChoice choice{mmq_tc_default_config(), MmqTcSource::Skipped};
    record_choice("tune/matmul_q8_tc", key, choice.cfg.to_string(),
                  registry_source(choice.source));
    std::lock_guard<std::mutex> lock(mmq_tc_memo_mutex());
    mmq_tc_memo().emplace(key, choice);
    return choice;
  }
  return MmqTcChoice{mmq_tc_default_config(), MmqTcSource::Default};
}

bool mmq_tc_choice_store(int m_tokens, int n_rows, int k,
                         const TileConfig& cfg) {
  const std::string key = mmq_tc_shape_class(m_tokens, n_rows, k);
  record_choice("tune/matmul_q8_tc", key, cfg.to_string(),
                planner::decisions::Source::Measured);
  {
    std::lock_guard<std::mutex> lock(mmq_tc_memo_mutex());
    mmq_tc_memo()[key] = MmqTcChoice{cfg, MmqTcSource::Measured};
  }
  return AutotuneCache::instance().store("matmul_q8_tc", key, cfg);
}

// --- Q3: prefill attention shape classes -----------------------------------

std::string attn_prefill_seq_bucket(int seq_len) {
  if (seq_len < 64) return "1-63";
  if (seq_len < 128) return "64-127";
  if (seq_len < 256) return "128-255";
  if (seq_len < 512) return "256-511";
  if (seq_len < 1024) return "512-1023";
  return "1024+";
}

std::string attn_prefill_heads_bucket(int n_heads) {
  if (n_heads < 8) return "1-7";
  if (n_heads < 16) return "8-15";
  return "16+";
}

std::string attn_prefill_shape_class(int head_dim, int seq_len, int n_heads) {
  return std::to_string(head_dim) + "/" + attn_prefill_seq_bucket(seq_len) +
         "/" + attn_prefill_heads_bucket(n_heads);
}

bool attn_prefill_tunable(int head_dim) {
  // The head dims the generator gives a default shape class (d64/gqa7,
  // d256/gqa4, d512/gqa4), plus the narrower classes its support rules allow.
  return head_dim == 32 || head_dim == 64 || head_dim == 128 ||
         head_dim == 256 || head_dim == 512;
}

std::vector<TileConfig> attn_prefill_candidates(int head_dim,
                                                const TileConfig &production) {
  // tile_m = br, tile_n = dc = head_dim/dps, tile_k = bc, unroll = as.
  // Production is first: it is the bit-identity reference and the time to
  // beat. dps is not varied (see mlir_autotune.h).
  std::vector<TileConfig> grid;
  grid.push_back(production);
  const int dc = production.tile_n;
  auto add = [&](int br, int bc, int as) {
    const TileConfig c{br, dc, bc, as, 1};
    if (c == production) return;
    for (const TileConfig &g : grid)
      if (g == c) return;
    grid.push_back(c);
  };
  const std::vector<int> brs = head_dim >= 512
                                   ? std::vector<int>{16, 32, 64}
                                   : std::vector<int>{32, 64};
  for (int br : brs)
    for (int bc : {production.tile_k, 32})
      for (int as : {production.unroll, 0, 2})
        add(br, bc, as);
  return grid;
}

namespace {
std::mutex& attn_prefill_memo_mutex() {
  static std::mutex m;
  return m;
}
std::unordered_map<std::string, AttnPrefillChoice>& attn_prefill_memo() {
  static std::unordered_map<std::string, AttnPrefillChoice> m;
  return m;
}
} // namespace

AttnPrefillChoice attn_prefill_choice(int head_dim, int seq_len, int n_heads) {
  if (!enabled()) {
    record_choice("tune/attn_prefill",
                  attn_prefill_shape_class(head_dim, seq_len, n_heads),
                  "dispatcher", planner::decisions::Source::Default);
    return AttnPrefillChoice{};
  }
  const std::string key = attn_prefill_shape_class(head_dim, seq_len, n_heads);
  {
    std::lock_guard<std::mutex> lock(attn_prefill_memo_mutex());
    const auto it = attn_prefill_memo().find(key);
    if (it != attn_prefill_memo().end()) return it->second;
  }
  // force() retunes: the cache read is skipped and the caller measures.
  if (!force()) {
    if (const auto cached =
            AutotuneCache::instance().lookup("attn_prefill", key)) {
      const AttnPrefillChoice choice{*cached, AttnPrefillSource::Cache};
      record_choice("tune/attn_prefill", key, cached->to_string(),
                    registry_source(choice.source));
      std::lock_guard<std::mutex> lock(attn_prefill_memo_mutex());
      attn_prefill_memo().emplace(key, choice);
      return choice;
    }
  }
  // Out of cold budget: keep the dispatcher's own shape for this class. The
  // empty cfg is rejected by the caller's apply(), so nothing is probed; the
  // Skipped record keeps the class out of the allocation path as well.
  if (budget_exhausted()) {
    const AttnPrefillChoice choice{TileConfig{}, AttnPrefillSource::Skipped};
    record_choice("tune/attn_prefill", key, "dispatcher",
                  registry_source(choice.source));
    std::lock_guard<std::mutex> lock(attn_prefill_memo_mutex());
    attn_prefill_memo().emplace(key, choice);
    return choice;
  }
  return AttnPrefillChoice{};
}

bool attn_prefill_choice_store(int head_dim, int seq_len, int n_heads,
                               const TileConfig& cfg) {
  const std::string key = attn_prefill_shape_class(head_dim, seq_len, n_heads);
  record_choice("tune/attn_prefill", key, cfg.to_string(),
                planner::decisions::Source::Measured);
  {
    std::lock_guard<std::mutex> lock(attn_prefill_memo_mutex());
    attn_prefill_memo()[key] = AttnPrefillChoice{cfg, AttnPrefillSource::Measured};
  }
  return AutotuneCache::instance().store("attn_prefill", key, cfg);
}

void attn_prefill_reset_for_test() {
  std::lock_guard<std::mutex> lock(attn_prefill_memo_mutex());
  attn_prefill_memo().clear();
}

std::optional<TileConfig> AutotuneRunner::run(const Problem& problem) {
  if (!enabled() || problem.candidates.empty() || !problem.compile)
    return std::nullopt;
  // Every probe starts its own clock check: the cold budget is shared by every
  // class tuned in this process, and an exhausted budget must not launch even
  // the reference candidate (the caller already has its default).
  if (budget_exhausted()) return std::nullopt;

  const int iters = problem.iters > 0 ? problem.iters : 5;
  std::vector<unsigned char> reference(problem.output_bytes);
  auto snapshot = [&] {
    if (problem.sync) problem.sync();
    if (problem.output && problem.output_bytes)
      std::memcpy(reference.data(), problem.output, problem.output_bytes);
  };

  // Candidate 0 is the reference: its output is the bit-identity window.
  struct Entry {
    TileConfig cfg;
    Launch launch;
    uint64_t ns = std::numeric_limits<uint64_t>::max();
  };
  std::vector<Entry> entries;

  auto ref_launch = problem.compile(problem.candidates[0]);
  if (!ref_launch) return std::nullopt;
  (*ref_launch)();
  snapshot();
  entries.push_back(Entry{problem.candidates[0], std::move(*ref_launch)});

  // Compile and bit-check every other candidate before any timing: a
  // candidate that cannot compile or that writes a different window is out.
  // Compiling a JIT candidate is the expensive part of a cold class, so the
  // budget is checked between compilations; the reference and every candidate
  // compiled so far stay in the race.
  for (size_t i = 1; i < problem.candidates.size(); ++i) {
    if (budget_exhausted()) break;
    auto launch = problem.compile(problem.candidates[i]);
    if (!launch) continue;
    (*launch)();
    if (problem.sync) problem.sync();
    if (problem.output && problem.output_bytes &&
        std::memcmp(reference.data(), problem.output, problem.output_bytes) !=
            0) {
      continue;  // numerically different from the reference: not a candidate
    }
    entries.push_back(Entry{problem.candidates[i], std::move(*launch)});
  }

  // Time every candidate once per round, alternating the direction each round,
  // so no candidate is always first or always last. Timing one candidate for
  // all its rounds before the next (the old order) let a clock ramp or host
  // load drift land on whichever end of the list ran first; at the few-percent
  // margins these tuners measure, that bias flipped the nomination between
  // processes. Each candidate keeps its fastest round.
  int rounds = 0;
  for (int r = 0; r < iters; ++r) {
    // The budget may expire mid-search; rank the rounds already measured. A
    // class with no complete round is left unresolved (the caller keeps its
    // default) rather than nominated from a partial one.
    if (budget_exhausted()) break;
    const bool forward = (r & 1) == 0;
    for (size_t j = 0; j < entries.size(); ++j) {
      Entry& e = entries[forward ? j : entries.size() - 1 - j];
      const auto t0 = std::chrono::steady_clock::now();
      e.launch();
      if (problem.timing_sync)
        problem.timing_sync();
      else if (problem.sync)
        problem.sync();
      const auto t1 = std::chrono::steady_clock::now();
      e.ns = std::min<uint64_t>(
          e.ns,
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }
    ++rounds;
  }
  if (rounds == 0) return std::nullopt;

  // Strict <: on a tie the earlier candidate (candidate 0, the reference) wins,
  // so a marginal measurement keeps the known config.
  size_t best = 0;
  for (size_t i = 1; i < entries.size(); ++i)
    if (entries[i].ns < entries[best].ns) best = i;

  if (std::getenv("SONICBOOM_AUTOTUNE_VERBOSE")) {
    std::fprintf(stderr, "[autotune] runner: %zu candidates, %d/%d rounds%s\n",
                 entries.size(), rounds, iters,
                 rounds < iters ? " (budget)" : "");
    for (const Entry& e : entries)
      std::fprintf(stderr, "  %d,%d,%d,%d,%d: %.4f ms%s\n", e.cfg.tile_m,
                   e.cfg.tile_n, e.cfg.tile_k, e.cfg.unroll, e.cfg.vectorize,
                   double(e.ns) / 1e6, &e == &entries[best] ? " <-" : "");
  }

  // Probing a rejected candidate leaves its output in the window; run the
  // winner once more so the caller's buffer holds the returned config's
  // result, not the last probe's.
  entries[best].launch();
  if (problem.sync) problem.sync();
  return entries[best].cfg;
}

} // namespace sonicboom::mlir_tune
