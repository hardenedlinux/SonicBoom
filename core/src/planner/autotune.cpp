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

#include "decision_registry.h"
#include "reduce_mode.h"

#include <sonicboom/runtime_features.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <sys/file.h>
#include <unistd.h>
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

std::string compute_device_dir() {
  // The reduction mode is part of the device identity: it changes the norm
  // kernels' instruction mix, so a choice tuned under one mode must not be
  // reused under another. The identity itself is shared with the reduction
  // decision cache (planner/reduce_mode.h), so both trees key the device the
  // same way.
  std::string dir = planner::device_identity() + "-reduce-" +
                    compiler::reduce_mode_name(planner::device_reduce_mode());
  // The numerics profile is NOT part of the device directory. It belongs in
  // the keys of the choices whose candidate set or kernel it actually changes
  // — today exactly one: the prefill attention class, whose f16-KV variant is
  // a distinct kernel (the boom_fattn JIT workload key carries the KV dtype).
  // A profile-wide directory would instead force every other class (mmq tiles,
  // dense arms, gemv widths) to retune under standard, and a cold retune picks
  // differently from a warm cache for reasons that have nothing to do with the
  // profile:
  // measured on DeepSeek d=4, the two directories' mmq-j picks alone moved a
  // prefill near-tie by 0.09 logits and flipped the first token. If a future
  // TF32 dense candidate arm is added, its key must gain the profile too.
  return dir;
}

std::string cache_root() { return planner::decision_cache_root(); }

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

// Declared here; both are defined with the minimal JSON reader below.
constexpr int kJsonFormatVersion = 1;
void parse_op_file(std::string_view s,
                   std::map<std::string, std::map<std::string, int>> &out);

// --- build identity, trace, resolution epoch ------------------------------

// Identity of the running library, the same size+mtime rule the spec cost
// cache uses: a rebuild moves it, so no record measured before a kernel change
// is reused after one.
std::string shared_object_identity() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void *>(&shared_object_identity), &info) == 0 ||
      info.dli_fname == nullptr)
    return "nostat";
  std::error_code ec;
  const auto size = std::filesystem::file_size(info.dli_fname, ec);
  if (ec) return "nostat";
  const auto mtime = std::filesystem::last_write_time(info.dli_fname, ec);
  if (ec) return "nostat";
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%llx-%llx", (unsigned long long)size,
                (unsigned long long)mtime.time_since_epoch().count());
  return buf;
}

long long fnv32(std::string_view s) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 0x100000001b3ull;
  }
  return static_cast<int>(static_cast<uint32_t>(h ^ (h >> 32)));
}

const char *exe_name() {
  static const std::string name = [] {
    std::ifstream in("/proc/self/cmdline", std::ios::binary);
    std::string s;
    std::getline(in, s, '\0');
    if (s.empty()) s = "?";
    return s;
  }();
  return name.c_str();
}

std::atomic<bool> g_sealed{false};
std::string g_sealed_miss_key;

bool numerics_locked_op(std::string_view op) {
  return op == "decode-family" || op == "mmq-splitk" || op == "mmq-cfg" ||
         op == "verify-gemv-wp" || op == "decode-attn" ||
         op == "attention-decode";
}

bool trace_line_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("SONICBOOM_TUNE_TRACE");
    return e != nullptr && e[0] != '\0' && std::strcmp(e, "0") != 0;
  }();
  return on;
}

std::mutex g_trace_mutex;

void trace_line(std::string_view event, std::string_view op,
                std::string_view key, std::string_view fp,
                const char *caller) {
  if (!trace_line_enabled()) return;
  std::string line;
  line.reserve(160 + key.size());
  line += "[tune] ";
  line += event;
  line += " file=";
  line += op;
  // The mlir autotune store names its own file; the JSON tables take <op>.json.
  if (op.find('.') == std::string_view::npos) line += ".json";
  line += " key=";
  line += key;
  line += " fp=";
  line += fp.empty() ? std::string_view("-") : fp;
  line += " pid=";
  line += std::to_string(static_cast<long long>(::getpid()));
  line += " exe=";
  line += exe_name();
  line += " caller=";
  line += caller != nullptr ? caller : "?";
  line += '\n';
  std::lock_guard<std::mutex> lk(g_trace_mutex);
  std::fwrite(line.data(), 1, line.size(), stderr);
  std::fflush(stderr);
}

std::string fp_text(long long fp) {
  return fp == kNoFingerprint ? std::string("-") : std::to_string(fp);
}

// The "k" a record carries: the effective fingerprint written next to the
// choice. It is the record's own "k" when the store did not name one.
long long effective_fp(const std::map<std::string, int> &values, long long fp) {
  if (fp != kNoFingerprint) return fp;
  const auto it = values.find("k");
  if (it == values.end()) return kNoFingerprint;
  return static_cast<long long>(it->second);
}

// Parse a `<op>.json` body from disk. Malformed bodies yield what parsed.
bool read_op_file(const std::string &path,
                  std::map<std::string, std::map<std::string, int>> &out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  parse_op_file(body, out);
  const auto fmt = out.find("__format__");
  if (fmt != out.end()) {
    const auto v = fmt->second.find("v");
    if (v == fmt->second.end() || v->second != kJsonFormatVersion) out.clear();
  }
  return true;
}

bool write_op_file(const std::string &path,
                   const std::map<std::string, std::map<std::string, int>> &entries) {
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path(), ec);
  std::ostringstream os;
  os << "{\n";
  bool first_key = true;
  for (const auto &[key, values] : entries) {
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
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << os.str();
    if (!out) return false;
  }
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
    return false;
  }
  return true;
}

// Exclusive advisory lock on `<op>.json.lock`, released on destruction. Every
// write is a read-modify-write of the file under this lock, so two processes
// storing different keys never drop each other's record.
struct FileLock {
  int fd = -1;
  explicit FileLock(const std::string &path) {
    const std::string lock = path + ".lock";
    fd = ::open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0644);
    if (fd >= 0 && ::flock(fd, LOCK_EX) != 0) {
      ::close(fd);
      fd = -1;
    }
  }
  ~FileLock() {
    if (fd >= 0) {
      ::flock(fd, LOCK_UN);
      ::close(fd);
    }
  }
  FileLock(const FileLock &) = delete;
  FileLock &operator=(const FileLock &) = delete;
};

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

// On-disk schema version of each <op>.json file. The reserved entry
// "__format__" records it next to the shape classes. A file without the entry
// predates the marker and is read as version 1 (its entries mean the same);
// a file with any other version is discarded, so a future change to a value's
// meaning invalidates old picks instead of being misread. A store always
// writes the current version, and the write is a temp file plus rename, so a
// reader sees the whole old file or the whole new one.
//
// kJsonFormatVersion is declared above (next to the fingerprint helpers).

void load_op_locked(const std::string &op) {
  if (g_loaded[op]) return;
  g_loaded[op] = true;
  std::ifstream in(cache_path_locked(op), std::ios::binary);
  if (!in) return;
  const std::string body((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  auto &entries = g_cache[op];
  parse_op_file(body, entries);
  const auto fmt = entries.find("__format__");
  if (fmt != entries.end()) {
    const auto v = fmt->second.find("v");
    if (v == fmt->second.end() || v->second != kJsonFormatVersion)
      entries.clear();
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

namespace {

// The registry value form of a tuner pick: "k=v,k=v" in key order.
std::string serialize_values(const std::map<std::string, int> &values) {
  std::string out;
  for (const auto &[name, value] : values) {
    if (!out.empty()) out += ',';
    out += name;
    out += '=';
    out += std::to_string(value);
  }
  return out;
}

void record_pick(std::string_view op, std::string_view key,
                 const std::map<std::string, int> &values,
                 planner::decisions::Source source) {
  planner::decisions::Entry entry;
  entry.name = "tune/" + std::string(op);
  entry.scope = std::string(key);
  entry.value = serialize_values(values);
  entry.source = source;
  entry.env = "SONICBOOM_AUTOTUNE";
  // The caller holds g_mutex; use the locked accessor.
  entry.inputs = "device=" + device_dir_locked() + ", class=" +
                 std::string(key);
  entry.candidates =
      source == planner::decisions::Source::Cached
          ? "the recorded pick"
          : "the tuner's eager-warm-up sweep";
  entry.reason = source == planner::decisions::Source::Cached
                     ? "per-device JSON cache record"
                     : "timed in the eager warm-up (3% win required)";
  entry.numerics = "kernel-specific; gated by the owning test suite";
  planner::decisions::record(entry);
}

} // namespace

const std::string &build_fingerprint() {
  static const std::string fp = [] {
    std::string s;
#ifdef SONICBOOM_VERSION_STRING
    s += SONICBOOM_VERSION_STRING;
#else
    s += "unknown";
#endif
    s += '|';
    s += shared_object_identity();
    return s;
  }();
  return fp;
}

long long build_fingerprint_key() {
  static const long long key = fnv32(build_fingerprint());
  return key;
}

bool trace_enabled() { return trace_line_enabled(); }

void trace_event(std::string_view event, std::string_view op,
                 std::string_view key, std::string_view fp,
                 const char *caller) {
  trace_line(event, op, key, fp, caller);
}

namespace {

// Caller holds g_mutex.
void late_miss_locked(std::string_view op, std::string_view key) {
  if (g_sealed_miss_key.empty()) g_sealed_miss_key = std::string(key);
  std::string line = "[tune] BUG late-miss key=";
  line += key;
  line += " file=";
  line += op;
  line += ".json pid=";
  line += std::to_string(static_cast<long long>(::getpid()));
  line += " exe=";
  line += exe_name();
  line += '\n';
  std::lock_guard<std::mutex> lk(g_trace_mutex);
  std::fwrite(line.data(), 1, line.size(), stderr);
  std::fflush(stderr);
}

void late_write_locked(std::string_view op, std::string_view key) {
  std::string line = "[tune] BUG late-write key=";
  line += key;
  line += " file=";
  line += op;
  line += ".json pid=";
  line += std::to_string(static_cast<long long>(::getpid()));
  line += " exe=";
  line += exe_name();
  line += '\n';
  std::lock_guard<std::mutex> lk(g_trace_mutex);
  std::fwrite(line.data(), 1, line.size(), stderr);
  std::fflush(stderr);
}

} // namespace

std::map<std::string, int> lookup(std::string_view op, std::string_view key,
                                  long long expect_fp, const char *caller) {
  if (!enabled()) return {};
  const std::string fp_s = fp_text(expect_fp);
  if (force()) {
    trace_line("miss", op, key, fp_s, caller);
    return {};
  }
  std::lock_guard<std::mutex> lk(g_mutex);
  const std::string op_s(op);
  const std::string key_s(key);
  load_op_locked(op_s);
  const auto it = g_cache[op_s].find(key_s);
  bool usable = it != g_cache[op_s].end();
  if (usable && expect_fp != kNoFingerprint) {
    const auto k = it->second.find("k");
    if (k == it->second.end() ||
        static_cast<long long>(k->second) != expect_fp) {
      usable = false;
      trace_line("fp-mismatch", op_s, key_s, fp_s, caller);
    }
  }
  if (!usable) {
    if (it == g_cache[op_s].end())
      trace_line("miss", op_s, key_s, fp_s, caller);
    if (sealed() && numerics_locked_op(op_s)) late_miss_locked(op_s, key_s);
    return {};
  }
  trace_line("hit", op_s, key_s, fp_s, caller);
  record_pick(op_s, key_s, it->second, planner::decisions::Source::Cached);
  return it->second;
}

void store(std::string_view op, std::string_view key,
           const std::map<std::string, int> &values, long long fp,
           const char *caller) {
  if (!enabled()) return;
  const std::string op_s(op);
  const std::string key_s(key);
  // The record's fingerprint: the explicit one, else the value the record
  // carries. A table without either is numerics-neutral by key.
  const long long new_fp = effective_fp(values, fp);
  const std::string fp_s = fp_text(new_fp);
  std::lock_guard<std::mutex> lk(g_mutex);
  // After the model-load epoch closed, a numerics-affecting table is read-only:
  // its choices were resolved at load and a write here means something missed.
  // Timing tables (spec-length) and the prefill-path tables (mmq-j,
  // attention-full, dense-*) resolve on first use and keep writing.
  if (sealed() && numerics_locked_op(op_s)) {
    late_write_locked(op_s, key_s);
    return;
  }
  const std::string path = cache_path_locked(op_s);
  // Read-modify-write under an exclusive lock: the on-disk file is the
  // authority, so a record another process wrote between our load and our
  // store is kept.
  FileLock file_lock(path);
  std::map<std::string, std::map<std::string, int>> entries;
  read_op_file(path, entries);
  const auto existing = entries.find(key_s);
  bool replaced = false;
  if (existing != entries.end()) {
    // Write once: a record already in the file is never replaced by this
    // build. Only a record measured for a different build (a fingerprint
    // mismatch) is replaced, which is how a rebuild re-measures each table
    // exactly once. An unfingerprinted table is frozen by its key, so the
    // build identity must be part of that key when it is meant to re-measure.
    // SONICBOOM_AUTOTUNE=force is the explicit "retune" escape hatch.
    const long long old_fp = effective_fp(existing->second, kNoFingerprint);
    const bool replace = force() || new_fp == kNoFingerprint ||
                         old_fp != new_fp;
    if (!replace) {
      trace_line("overwrite-refused", op_s, key_s, fp_s, caller);
      g_cache[op_s] = entries;
      g_loaded[op_s] = true;
      return;
    }
    replaced = true;
  }
  entries[key_s] = values;
  entries["__format__"] = {{"v", kJsonFormatVersion}};
  if (!write_op_file(path, entries)) return;
  g_cache[op_s] = std::move(entries);
  g_loaded[op_s] = true;
  trace_line(replaced ? "overwrite" : "write", op_s, key_s, fp_s, caller);
  record_pick(op_s, key_s, values, planner::decisions::Source::Measured);
}

void open() {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_sealed.store(false);
  g_sealed_miss_key.clear();
}

void seal() {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_sealed.store(true);
}

bool sealed() { return g_sealed.load(); }

bool sealed_missed() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return !g_sealed_miss_key.empty();
}

std::string sealed_miss_key() {
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_sealed_miss_key;
}

void clear_sealed_miss() {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_sealed_miss_key.clear();
}

void clear_memory_cache() {
  std::lock_guard<std::mutex> lk(g_mutex);
  g_cache.clear();
  g_loaded.clear();
}

std::vector<DiskChoice> disk_choices() {
  std::vector<DiskChoice> out;
  std::lock_guard<std::mutex> lk(g_mutex);
  const std::filesystem::path dir =
      std::filesystem::path(cache_root()) / device_dir_locked();
  std::error_code ec;
  if (!std::filesystem::is_directory(dir, ec)) return out;
  for (const auto &entry : std::filesystem::directory_iterator(dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec) || entry.path().extension() != ".json")
      continue;
    std::ifstream in(entry.path(), std::ios::binary);
    if (!in) continue;
    const std::string body((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    std::map<std::string, std::map<std::string, int>> parsed;
    parse_op_file(body, parsed);
    const auto fmt = parsed.find("__format__");
    if (fmt != parsed.end()) {
      const auto v = fmt->second.find("v");
      if (v == fmt->second.end() || v->second != kJsonFormatVersion) continue;
    }
    const std::string op = entry.path().stem().string();
    for (auto &[key, values] : parsed) {
      if (key == "__format__") continue;
      out.push_back(DiskChoice{op, key, std::move(values)});
    }
  }
  std::sort(out.begin(), out.end(), [](const DiskChoice &a, const DiskChoice &b) {
    return a.op != b.op ? a.op < b.op : a.key < b.key;
  });
  return out;
}

#ifdef SONICBOOM_USE_CUDA
double median_launch_ms(const std::function<bool()> &launch,
                        cudaStream_t stream, int warmups, int reps) {
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
}
#else
double median_launch_ms(const std::function<bool()> &launch, int warmups,
                        int reps) {
  (void)launch;
  (void)warmups;
  (void)reps;
  return -1.0;
}
#endif

} // namespace sonicboom::tune
