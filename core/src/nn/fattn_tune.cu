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

// boom_fattn candidate selection: legalise, compile (AOT or JIT), verify,
// time, persist, launch. See fattn_tune.h for the gates and the lifecycle.
//
// Shape class. A selection is keyed by three fingerprints (see the block above
// build_fingerprints): the device's capability and resource limits, the
// software that produced and launches the kernel, and the workload class --
// head dim, GQA, KV dtype, mask kind and a power-of-two bucket of the KV span a
// query can read. The query count is deliberately NOT part of any component:
// run rule R1 keeps one kernel per shape class, so a 130-row chunk and the same
// rows inside a full prefill resolve the same way.
//
// The span bucket is the one component that varies between the chunks of one
// prefill (their slots+queries bounds differ). That is sound only because every
// selectable candidate is bit-identical to the reference on this input class --
// measured as rel-L2 exactly 0 for all in-set candidates on both head dims, and
// asserted by the fattn report test. Two chunks of one prefill may therefore
// run different candidates and still produce the same bytes as an unchunked
// call. A future candidate that is not bit-identical must not be added to the
// tuning set while the span bucket is in the key.
//
// The class is resolved once per key per process from whatever call arrives
// first; the 2% incumbent margin keeps a low-signal decision (the candidates
// are within noise of each other at short context) on the reference.
//
// The reference candidate (kCands[0]) is the configuration measured best on an
// RTX 3050 and is the incumbent of every comparison: a challenger replaces it
// only with a >= kMarginPct better median. That is what makes a noise-level
// measurement incapable of selecting a regression.

#include "fattn_tune.h"

#include "cuda_stream.h"
#include "fattn_jit.h"

#include "../planner/mlir_autotune.h"
#include "../planner/plan_hook.h"  // SB_NOTE_LAUNCH (plan record; no-op when off)

#include <sonicboom/runtime_features.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

#ifndef SONICBOOM_VERSION_STRING
#define SONICBOOM_VERSION_STRING "unknown"
#endif

namespace sonicboom::nn::cuda::fattn_tune {
namespace {

// ---------------------------------------------------------------------------
// Tunables.
// ---------------------------------------------------------------------------
// The incumbent's protection: a challenger must be this much faster (percent)
// to be selected. Candidates that are equal within measurement noise therefore
// never displace the reference.
constexpr double kMarginPct = 2.0;
// Correctness gate against the reference output (relative L2 of the whole
// output). Layout-only candidates are bit-identical (0); the 4-warp phase-1b
// variant differs only by summation order (~1e-6 here); a broken candidate is
// off by orders of magnitude.
constexpr double kMaxRelL2 = 1e-2;
constexpr int kMaxReps = 5;
// A timed sample is a batch of launches (see time_launch) aimed at this
// duration, so the measurement is not dominated by launch latency and event
// granularity at short context.
constexpr double kTargetSampleMs = 0.5;
constexpr int kMaxIters = 64;
constexpr int64_t kDefaultBudgetMs = 4000;

// ---------------------------------------------------------------------------
// Candidates. num is the JIT entry-point suffix, the cache payload and the
// SONICBOOM_FATTN_CANDIDATE key; id is the stable name used by the report and
// by SONICBOOM_FATTN_CANDIDATE/CANDIDATES. The table is the experiment record:
// every variant that exists keeps its row, including the ones excluded from
// the default set and why.
// ---------------------------------------------------------------------------
struct Cand {
  int num;
  const char* id;
  int kpad;  // extra halves per K row
  int vpad;  // extra halves per V row
  int spw;   // phase-1b warps per unit (2 = row-half split, 8 warps)
  int pipe;  // 1 = QK/PV pipeline (V triple buffered)
  bool in_set;        // part of the default tuning set
  bool experimental;  // reachable only by forcing it by id
  const char* note;
};

constexpr Cand kCands[] = {
    {1, "k8-v0-s2-pipe", 8, 0, 2, 1, true, false,
     "RTX 3050 reference: KSTRIDE padding, 8-warp phase-1b row-half split, "
     "unpadded V, QK/PV pipeline"},
    {2, "k8-v0-s2-seq", 8, 0, 2, 0, true, false,
     "same layout without the QK/PV pipeline (V double buffered); the pipeline "
     "gain shows up at long context"},
    {3, "k8-v0-s1-pipe", 8, 0, 1, 1, true, false,
     "phase 1b on 4 warps (the row half is not split)"},
    {4, "k0-v0-s2-pipe", 0, 0, 2, 1, true, false,
     "K rows unpadded: 8-way bank conflict on every K fragment load"},
    {5, "k8-v8-s2-seq", 8, 8, 2, 0, true, false,
     "V rows padded to 48 B, sequential phases; needs the full sm_8.x opt-in cap"},
    {6, "k8-v8-s2-pipe", 8, 8, 2, 1, true, false,
     "V padding plus the pipeline's third V buffer: over the sm_8.x cap, legal on "
     "parts with a larger shared-memory budget"},
    {7, "k8-v4-s2-seq", 8, 4, 2, 0, false, true,
     "V rows padded to 40 B: 8-byte cp.async.ca, measured 20% slower on sm_86"},
};
const Cand* cand_by_num(int num) {
  for (const Cand& c : kCands)
    if (c.num == num) return &c;
  return nullptr;
}

const Cand* cand_by_id(std::string_view id) {
  for (const Cand& c : kCands)
    if (id == c.id) return &c;
  return nullptr;
}

const Cand* cand_by_name_or_num(const char* text) {
  char* tail = nullptr;
  const long v = std::strtol(text, &tail, 10);
  if (tail != nullptr && *tail == '\0' && tail != text) return cand_by_num(int(v));
  return cand_by_id(text);
}

// ---------------------------------------------------------------------------
// Environment.
// ---------------------------------------------------------------------------
const char* env(const char* name) {
  const char* e = std::getenv(name);
  return (e != nullptr && *e != '\0') ? e : nullptr;
}

bool env_is(const char* name, const char* value) {
  const char* e = env(name);
  return e != nullptr && std::strcmp(e, value) == 0;
}

bool report_enabled() { return !env_is("SONICBOOM_FATTN_REPORT", "0"); }

enum class Autotune { Off, On, Force };
enum class Jit { Off, Auto, Force };

Autotune autotune_mode() {
  if (const char* e = env("SONICBOOM_FATTN_AUTOTUNE"); e != nullptr) {
    if (std::strcmp(e, "0") == 0) return Autotune::Off;
    if (std::strcmp(e, "force") == 0) return Autotune::Force;
    return Autotune::On;
  }
  // The global gate covers this tuner too; the MLIR JIT ablation does not.
  if (::sonicboom::runtime::feature_disabled(
          ::sonicboom::runtime::Feature::Autotune))
    return Autotune::Off;
  if (const char* g = env("SONICBOOM_AUTOTUNE"); g != nullptr) {
    if (std::strcmp(g, "0") == 0) return Autotune::Off;
    if (std::strcmp(g, "force") == 0) return Autotune::Force;
  }
  return Autotune::On;
}

Jit jit_mode() {
  if (const char* e = env("SONICBOOM_FATTN_JIT"); e != nullptr) {
    if (std::strcmp(e, "off") == 0 || std::strcmp(e, "0") == 0) return Jit::Off;
    if (std::strcmp(e, "force") == 0) return Jit::Force;
    return Jit::Auto;
  }
  return Jit::Auto;
}

int64_t budget_ms() {
  if (const char* e = env("SONICBOOM_FATTN_AUTOTUNE_BUDGET_MS"); e != nullptr) {
    char* tail = nullptr;
    const long long v = std::strtoll(e, &tail, 10);
    if (tail != nullptr && *tail == '\0') return int64_t(v);
  }
  return kDefaultBudgetMs;
}

// ---------------------------------------------------------------------------
// Geometry (mirrors boom_fattn_qreg_body's shared-memory layout).
// ---------------------------------------------------------------------------
Geometry geometry(const Cand& c, int head_dim, int gqa) {
  constexpr int BR = 16;
  constexpr int BK = 16;
  constexpr int NW = 4;
  const int units = (BR / 16) * gqa;
  const int subs = BK / 16;
  const int vbuf = c.pipe ? 3 : 2;
  const int k_elems = 2 * BK * (head_dim + c.kpad);
  const int v_elems = vbuf * head_dim * (BK + c.vpad);
  const int p_elems = units * 16 * BK;
  const int rs_elems = units * subs * 16;
  Geometry g;
  g.br = BR;
  g.bk = BK;
  g.block = units * NW * 32;
  g.kstride = head_dim + c.kpad;
  g.bkp = BK + c.vpad;
  g.softmax_warps = c.spw * units;
  g.pipeline = c.pipe != 0;
  g.smem = (size_t)(k_elems + v_elems + p_elems) * sizeof(__half) +
           (size_t)(rs_elems + units * 16 + units * NW * subs * 256) *
               sizeof(float);
  return g;
}

// ---------------------------------------------------------------------------
// Device.
// ---------------------------------------------------------------------------
struct Dev {
  bool ok = false;
  std::string error;
  cudaDeviceProp prop{};
  int cc = 0;
  // CUDA version the installed driver supports (what JIT-compiles the PTX) and
  // the CUDA runtime this library was built against. -1 when not reported.
  int driver_cuda = -1;
  int runtime_cuda = -1;
};

const Dev& device() {
  static const Dev d = [] {
    Dev d;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
      d.error = "no CUDA device";
      return d;
    }
    if (cudaGetDeviceProperties(&d.prop, dev) != cudaSuccess) {
      d.error = "cudaGetDeviceProperties failed";
      return d;
    }
    d.cc = d.prop.major * 10 + d.prop.minor;
    int v = 0;
    if (cudaDriverGetVersion(&v) == cudaSuccess) d.driver_cuda = v;
    else cudaGetLastError();
    v = 0;
    if (cudaRuntimeGetVersion(&v) == cudaSuccess) d.runtime_cuda = v;
    else cudaGetLastError();
    d.ok = true;
    return d;
  }();
  return d;
}

std::string cc_string(const Dev& d) {
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d.%d", d.cc / 10, d.cc % 10);
  return buf;
}

// ---------------------------------------------------------------------------
// Fingerprints.
//
// A stored selection is valid only for the device it was measured on, the
// software that produced and launches the kernel, and the workload class it was
// measured for. Those are three independent facts that change for different
// reasons, so they are three components of the key, not one blob:
//
//   capability  what the hardware can run and how much of it is resident:
//               compute capability (cp.async needs 8.0), shared memory per
//               block the driver allows to opt in, registers per SM, max
//               threads per block, threads per SM, SM count. The first four
//               decide legality (a candidate the device cannot run is
//               eliminated); the last two decide how many blocks a candidate
//               keeps resident, i.e. the performance ordering.
//   software    which code object runs: the driver's CUDA version (it JITs the
//               PTX), the runtime the library was built against (AOT codegen),
//               the compile path (AOT cubin or NVRTC), the NVRTC version when
//               that path is used, the device-source hash (kernel body) and the
//               candidate registry version (launch parameters).
//   workload    what the kernel does: head dim, GQA, KV dtype, mask kind, and a
//               power-of-two bucket of the KV span a call can read. The span
//               bound is host-side (slots + queries for global attention,
//               min(slots, window) for a sliding window); it is a bucket, never
//               the live position, and the query count itself is not in the key
//               (run rule R1: one kernel per shape class, so a chunked prefill
//               and the same prefill in one call resolve the same way).
//
// The GPU product name is not a fact about the hardware: two cards sold under
// one name can differ in SM count, shared memory or clocks, and a modified card
// can report any name. It is printed in the report and used by nothing.
// ---------------------------------------------------------------------------
struct Fingerprints {
  std::string capability;
  std::string software;
  std::string workload;
  std::string key;
  bool sim_capability = false;
  bool sim_software = false;
  bool sim_workload = false;
  // The key does not fit the cache's fixed key field; the class still resolves
  // and memoizes, it is just not persisted.
  bool cache_disabled = false;
};

// "12.8" from the CUDA version integer (12080); "unknown" when unreported.
std::string cuda_version_text(int v) {
  if (v <= 0) return "unknown";
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%d.%d", v / 1000, (v % 1000) / 10);
  return buf;
}

std::string capability_fingerprint(const Dev& d) {
  if (!d.ok) return "unavailable";
  char buf[160];
  std::snprintf(buf, sizeof(buf), "cc%d-sm%d-shm%d:%d-reg%d-thr%d:%d", d.cc,
                d.prop.multiProcessorCount, d.prop.sharedMemPerBlockOptin,
                d.prop.sharedMemPerMultiprocessor, d.prop.regsPerMultiprocessor,
                d.prop.maxThreadsPerBlock, d.prop.maxThreadsPerMultiProcessor);
  return buf;
}

// True when this build's cubin has an image for (candidate, head_dim) that the
// driver accepts on the current device.
bool aot_image_exists(int candidate, int head_dim) {
  void* fn = aot_kernel(candidate, head_dim);
  if (fn == nullptr) return false;
  cudaFuncAttributes attr{};
  const cudaError_t e = cudaFuncGetAttributes(&attr, fn);
  if (e == cudaSuccess) return true;
  cudaGetLastError();  // cudaErrorInvalidDeviceFunction is the expected miss
  return false;
}

// The code path the reference candidate will take here, as resolve_target()
// decides it: a forced JIT, the AOT image when the cubin has one for this
// device, otherwise NVRTC. The selection key needs the answer before the
// resolution measures anything, because an NVRTC-compiled kernel is a different
// code object from the AOT one.
//
// The reference is representative for the whole registry: every candidate is
// instantiated in the same translation unit (fattn_boom_fattn.cu) with the same
// architecture list, so either the cubin has an image every candidate can use
// or it has none of them. If a candidate ever gains its own instantiation
// condition, this must become "jit when any in-set candidate would JIT", so the
// NVRTC version enters the software component.
const char* expected_compile_path(const Call& c) {
  const Jit m = jit_mode();
  if (m == Jit::Force) return "jit";
  if (aot_image_exists(kCands[0].num, c.head_dim)) return "aot";
  return "jit";
}

std::string software_fingerprint(const Call& c) {
  const Dev& d = device();
  const std::string path = expected_compile_path(c);
  std::string sw = "drv" + cuda_version_text(d.driver_cuda);
  sw += "-rt" + cuda_version_text(d.runtime_cuda);
  sw += "-" + path;
  if (path == "jit") sw += "-nvrtc" + fattn_jit::nvrtc_version_string();
  char tail[48];
  std::snprintf(tail, sizeof(tail), "-src%016llx-reg%d",
                static_cast<unsigned long long>(fattn_jit::source_hash()),
                kRegistryVersion);
  sw += tail;
  return sw;
}

// The KV rows a query can read at most: for global attention the slots already
// stored plus this call's queries; for a sliding window the ring it can reach.
uint64_t kv_span_bound(const Call& c) {
  if (c.window > 0) {
    const uint64_t ring = c.n_slots > 0 ? c.n_slots : c.window;
    return std::min<uint64_t>(ring, c.window);
  }
  return c.n_slots + uint64_t(c.n > 0 ? c.n : 0);
}

// Next power of two, saturating at 2^30: contexts above 2^30 rows share one
// class, which is beyond any device this kernel runs on.
unsigned span_bucket(uint64_t v) {
  unsigned b = 1;
  while (b < v && b < (1u << 30)) b <<= 1;
  return b;
}

std::string workload_fingerprint(const Call& c) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "hd%d-gqa%d-f16-%s-kv%u", c.head_dim, c.gqa,
                c.window > 0 ? "swa" : "global",
                span_bucket(kv_span_bound(c)));
  return buf;
}

// Test hook: SONICBOOM_FATTN_FP_SIMULATE="capability=<text>;software=<text>;
// workload=<text>" replaces a component in the key and the report. It is read
// on every call (no caching) so a test can change it between resolutions. It
// can never change legality, compilation or timing: those always use the
// device and the build that are actually present.
struct SimOverride {
  std::string capability;
  std::string software;
  std::string workload;
};

SimOverride sim_override() {
  SimOverride o;
  const char* e = env("SONICBOOM_FATTN_FP_SIMULATE");
  if (e == nullptr) return o;
  std::string_view text(e);
  while (!text.empty()) {
    const size_t semi = text.find(';');
    const std::string_view item = text.substr(0, semi);
    const size_t eq = item.find('=');
    if (eq != std::string_view::npos) {
      const std::string_view name = item.substr(0, eq);
      const std::string_view value = item.substr(eq + 1);
      if (!value.empty()) {
        if (name == "capability") o.capability = std::string(value);
        else if (name == "software") o.software = std::string(value);
        else if (name == "workload") o.workload = std::string(value);
      }
    }
    if (semi == std::string_view::npos) break;
    text.remove_prefix(semi + 1);
  }
  return o;
}

Fingerprints build_fingerprints(const Call& c) {
  Fingerprints f;
  f.capability = capability_fingerprint(device());
  f.software = software_fingerprint(c);
  f.workload = workload_fingerprint(c);
  const SimOverride sim = sim_override();
  if (!sim.capability.empty()) {
    f.capability = sim.capability;
    f.sim_capability = true;
  }
  if (!sim.software.empty()) {
    f.software = sim.software;
    f.sim_software = true;
  }
  if (!sim.workload.empty()) {
    f.workload = sim.workload;
    f.sim_workload = true;
  }
  f.key = "f" + std::to_string(kFingerprintVersion) + "/" + f.capability + "/" +
          f.software + "/" + f.workload;
  f.cache_disabled =
      f.key.size() > mlir_tune::AutotuneCache::max_shape_class_bytes();
  return f;
}

// ---------------------------------------------------------------------------
// JSON. Values are built as a tree and serialised by one writer, so strings
// are escaped and numbers formatted in exactly one place; no field of the
// report is assembled by concatenation.
// ---------------------------------------------------------------------------
class Json {
 public:
  enum class Kind { Null, Bool, Int, Real, Str, Arr, Obj };

  Json() = default;
  static Json boolean(bool v) {
    Json j;
    j.kind_ = Kind::Bool;
    j.b_ = v;
    return j;
  }
  static Json integer(long long v) {
    Json j;
    j.kind_ = Kind::Int;
    j.i_ = v;
    return j;
  }
  static Json real(double v) {
    Json j;
    j.kind_ = Kind::Real;
    j.r_ = v;
    return j;
  }
  static Json str(std::string v) {
    Json j;
    j.kind_ = Kind::Str;
    j.s_ = std::move(v);
    return j;
  }
  static Json arr() {
    Json j;
    j.kind_ = Kind::Arr;
    return j;
  }
  static Json obj() {
    Json j;
    j.kind_ = Kind::Obj;
    return j;
  }

  Json& push(Json v) {
    arr_.push_back(std::move(v));
    return *this;
  }
  Json& set(std::string key, Json v) {
    obj_.emplace_back(std::move(key), std::move(v));
    return *this;
  }

  std::string dump() const {
    std::string out;
    write(out);
    return out;
  }

 private:
  static void escape(std::string& out, const std::string& s) {
    out.push_back('"');
    for (const unsigned char c : s) {
      switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
          if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", unsigned(c));
            out += buf;
          } else {
            out.push_back(char(c));
          }
      }
    }
    out.push_back('"');
  }

  void write(std::string& out) const {
    switch (kind_) {
      case Kind::Null:
        out += "null";
        break;
      case Kind::Bool:
        out += b_ ? "true" : "false";
        break;
      case Kind::Int:
        out += std::to_string(i_);
        break;
      case Kind::Real: {
        if (!std::isfinite(r_)) {
          out += "null";
          break;
        }
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.6g", r_);
        out += buf;
        break;
      }
      case Kind::Str:
        escape(out, s_);
        break;
      case Kind::Arr: {
        out.push_back('[');
        for (size_t i = 0; i < arr_.size(); ++i) {
          if (i != 0) out.push_back(',');
          arr_[i].write(out);
        }
        out.push_back(']');
        break;
      }
      case Kind::Obj: {
        out.push_back('{');
        for (size_t i = 0; i < obj_.size(); ++i) {
          if (i != 0) out.push_back(',');
          escape(out, obj_[i].first);
          out.push_back(':');
          obj_[i].second.write(out);
        }
        out.push_back('}');
        break;
      }
    }
  }

  Kind kind_ = Kind::Null;
  bool b_ = false;
  long long i_ = 0;
  double r_ = 0.0;
  std::string s_;
  std::vector<Json> arr_;
  std::vector<std::pair<std::string, Json>> obj_;
};

// ---------------------------------------------------------------------------
// Output: the startup report. One human line and one JSON line, both written
// after all measurement is done, so logging never sits inside a timed region.
// ---------------------------------------------------------------------------
bool append_file(const std::string& path, const std::string& text) {
  const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
  if (fd < 0) return false;
  size_t off = 0;
  while (off < text.size()) {
    const ssize_t n = ::write(fd, text.data() + off, text.size() - off);
    if (n <= 0) {
      ::close(fd);
      return false;
    }
    off += size_t(n);
  }
  ::close(fd);
  return true;
}

std::string cache_root_dir() {
  if (const char* e = env("XDG_CACHE_HOME"); e != nullptr)
    return std::string(e) + "/sonicboom";
  if (const char* h = env("HOME"); h != nullptr)
    return std::string(h) + "/.cache/sonicboom";
  return "/tmp/sonicboom";
}

// The persistent report: one JSON object per line, appended when a class is
// resolved. It carries device features, the candidate results and the run
// conditions, and never model inputs or user paths.
std::string report_log_path() {
  return cache_root_dir() + "/autotune/fattn-reports.jsonl";
}

std::string now_iso8601() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
  return buf;
}

// ---------------------------------------------------------------------------
// One candidate result, as recorded during tuning.
// ---------------------------------------------------------------------------
struct CandResult {
  std::string id;
  int num = 0;
  std::string status;  // reference | timed | eliminated | compile-failed |
                       // correctness-failed | skipped-budget
  std::string reason;
  double median_ms = -1.0;
  double rel_l2 = -1.0;
  int regs = -1;
  int spill_bytes = -1;
  size_t smem = 0;
  std::string impl;
  std::string compile;
  int reps = 0;
};

struct ReportData {
  std::string source;  // measured | cache | forced | default | fallback
  bool fresh = false;
  std::string reason;
  std::string selected_id;
  int selected_num = 0;
  Geometry geo;
  int regs = -1;
  int spill_bytes = -1;
  std::string impl;
  std::string compile;
  std::string class_str;  // the full selection key

  // Fingerprints that form the key (see build_fingerprints). The simulated
  // flags say which components came from SONICBOOM_FATTN_FP_SIMULATE.
  Fingerprints fp;

  std::vector<CandResult> cands;
  int n_total = 0;
  int n_eliminated = 0;
  int n_compile_failed = 0;
  int n_correctness_failed = 0;
  int n_timed = 0;
  int warmup = 0;
  int reps = 0;
  int iters = 1;  // launches per timed sample
  bool failed = false;  // no candidate could run; nothing was selected
  int64_t budget_ms = 0;
  double margin_pct = kMarginPct;
  double ref_ms = -1.0;
  double probe_ms = -1.0;

  bool cache_hit = false;
  int cache_version = 0;
  // hit | miss | version-mismatch | unreadable | invalidated | disabled |
  // bypassed | unavailable (see docs/fattn-autotune.md).
  std::string cache_status = "miss";
  std::string cache_reason;

  // Workload (architecture only; no model input, no user path).
  int head_dim = 0;
  int gqa = 0;
  int queries = 0;
  int n_heads_q = 0;
  int n_heads_kv = 0;
  uint64_t window = 0;
  bool capturing = false;
};

const char* schedule_name(const Geometry& g) {
  return g.pipeline ? "pipeline" : "sequential";
}

std::string basis_text(const ReportData& r) {
  if (r.failed)
    return "no candidate could run on this device, so nothing was selected: " +
           r.reason;
  if (r.source == "measured")
    return "measured in this process: median kernel latency per candidate, "
           "the incumbent kept unless a challenger is " +
           std::to_string(int(r.margin_pct + 0.5)) + "% faster";
  if (r.source == "cache")
    return "reused from the per-device cache; its timings are historical, not "
           "measured in this process";
  if (r.source == "forced") return "forced by SONICBOOM_FATTN_CANDIDATE";
  if (r.source == "fallback")
    return "safe fallback to the reference candidate: " + r.reason;
  return "reference candidate: " + r.reason;
}

Json report_json(const ReportData& r) {
  const Dev& d = device();
  Json root = Json::obj();
  root.set("schema_version", Json::integer(2));
  root.set("tool", Json::str("sonicboom"));
  root.set("event", Json::str("fattn_selection"));
  root.set("time", Json::str(now_iso8601()));
  root.set("sonicboom_version", Json::str(SONICBOOM_VERSION_STRING));

  Json dev = Json::obj();
  dev.set("name", Json::str(d.ok ? d.prop.name : "unavailable"));
  dev.set("compute_capability", d.ok ? Json::str(cc_string(d)) : Json());
  dev.set("sm_version", d.ok ? Json::integer(d.cc) : Json());
  dev.set("sm_count", d.ok ? Json::integer(d.prop.multiProcessorCount) : Json());
  dev.set("shared_mem_per_block_optin",
          d.ok ? Json::integer(d.prop.sharedMemPerBlockOptin) : Json());
  dev.set("shared_mem_per_multiprocessor",
          d.ok ? Json::integer(d.prop.sharedMemPerMultiprocessor) : Json());
  dev.set("regs_per_multiprocessor",
          d.ok ? Json::integer(d.prop.regsPerMultiprocessor) : Json());
  dev.set("max_threads_per_block",
          d.ok ? Json::integer(d.prop.maxThreadsPerBlock) : Json());
  dev.set("max_threads_per_multiprocessor",
          d.ok ? Json::integer(d.prop.maxThreadsPerMultiProcessor) : Json());
  dev.set("driver_cuda_version",
          d.driver_cuda > 0 ? Json::str(cuda_version_text(d.driver_cuda)) : Json());
  dev.set("runtime_cuda_version",
          d.runtime_cuda > 0 ? Json::str(cuda_version_text(d.runtime_cuda))
                             : Json());
  // The product name is here for a human reader only; no decision reads it.
  dev.set("name_is_informational", Json::boolean(true));
  root.set("device", std::move(dev));

  Json fp = Json::obj();
  fp.set("version", Json::integer(kFingerprintVersion));
  fp.set("key", Json::str(r.class_str));
  fp.set("capability", Json::str(r.fp.capability));
  fp.set("software", Json::str(r.fp.software));
  fp.set("workload", Json::str(r.fp.workload));
  Json sim = Json::obj();
  sim.set("capability", Json::boolean(r.fp.sim_capability));
  sim.set("software", Json::boolean(r.fp.sim_software));
  sim.set("workload", Json::boolean(r.fp.sim_workload));
  fp.set("simulated", std::move(sim));
  root.set("fingerprint", std::move(fp));

  Json kern = Json::obj();
  kern.set("impl", Json::str("boom_fattn"));
  kern.set("candidate_registry_version", Json::integer(kRegistryVersion));
  const Jit jm = jit_mode();
  kern.set("jit", Json::str(jm == Jit::Force   ? "force"
                            : jm == Jit::Off ? "off"
                                             : "auto"));
  root.set("kernel", std::move(kern));

  // A failed resolution has nothing selected: every selection field is null so
  // a consumer can never read a failed configuration as the chosen one.
  const bool have = !r.failed;
  Json sel = Json::obj();
  sel.set("candidate", have ? Json::str(r.selected_id) : Json());
  sel.set("id", have ? Json::integer(r.selected_num) : Json());
  sel.set("status", Json::str(r.source));
  sel.set("fresh", Json::boolean(r.fresh));
  sel.set("reason", r.reason.empty() ? Json() : Json::str(r.reason));
  sel.set("br", have ? Json::integer(r.geo.br) : Json());
  sel.set("bk", have ? Json::integer(r.geo.bk) : Json());
  sel.set("kstride", have ? Json::integer(r.geo.kstride) : Json());
  sel.set("bkp", have ? Json::integer(r.geo.bkp) : Json());
  sel.set("softmax_warps", have ? Json::integer(r.geo.softmax_warps) : Json());
  sel.set("schedule", have ? Json::str(schedule_name(r.geo)) : Json());
  sel.set("block", have ? Json::integer(r.geo.block) : Json());
  Json compile = Json::obj();
  compile.set("impl", have ? Json::str(r.impl.empty() ? "aot" : r.impl) : Json());
  compile.set("artifact",
              have && !r.compile.empty() ? Json::str(r.compile) : Json());
  sel.set("compile", std::move(compile));
  sel.set("smem_bytes", have ? Json::integer((long long)r.geo.smem) : Json());
  sel.set("registers", r.regs >= 0 ? Json::integer(r.regs) : Json());
  sel.set("spill_bytes",
          r.spill_bytes >= 0 ? Json::integer(r.spill_bytes) : Json());
  sel.set("basis", Json::str(basis_text(r)));
  root.set("selection", std::move(sel));

  Json tune = Json::obj();
  tune.set("candidates_total", Json::integer(r.n_total));
  tune.set("eliminated_by_resource", Json::integer(r.n_eliminated));
  tune.set("compile_failures", Json::integer(r.n_compile_failed));
  tune.set("correctness_failures", Json::integer(r.n_correctness_failed));
  tune.set("timed", Json::integer(r.n_timed));
  tune.set("warmup_launches", Json::integer(r.warmup));
  tune.set("reps", Json::integer(r.reps));
  tune.set("launches_per_sample", Json::integer(r.iters));
  tune.set("budget_ms", Json::integer(r.budget_ms));
  tune.set("margin_pct", Json::real(r.margin_pct));
  tune.set("reference_median_ms", r.ref_ms >= 0.0 ? Json::real(r.ref_ms) : Json());
  tune.set("probe_median_ms", r.probe_ms >= 0.0 ? Json::real(r.probe_ms) : Json());
  Json wl = Json::obj();
  wl.set("head_dim", Json::integer(r.head_dim));
  wl.set("gqa", Json::integer(r.gqa));
  wl.set("queries", Json::integer(r.queries));
  wl.set("n_heads_q", Json::integer(r.n_heads_q));
  wl.set("n_heads_kv", Json::integer(r.n_heads_kv));
  wl.set("sliding_window", Json::integer((long long)r.window));
  wl.set("kv_dtype", Json::str("f16"));
  wl.set("under_graph_capture", Json::boolean(r.capturing));
  tune.set("workload", std::move(wl));
  Json list = Json::arr();
  for (const CandResult& c : r.cands) {
    Json o = Json::obj();
    o.set("candidate", Json::str(c.id));
    o.set("id", Json::integer(c.num));
    o.set("status", Json::str(c.status));
    o.set("reason", c.reason.empty() ? Json() : Json::str(c.reason));
    o.set("median_ms", c.median_ms >= 0.0 ? Json::real(c.median_ms) : Json());
    o.set("reps", Json::integer(c.reps));
    o.set("rel_l2", c.rel_l2 >= 0.0 ? Json::real(c.rel_l2) : Json());
    o.set("smem_bytes", Json::integer((long long)c.smem));
    o.set("registers", c.regs >= 0 ? Json::integer(c.regs) : Json());
    o.set("impl", c.impl.empty() ? Json() : Json::str(c.impl));
    o.set("compile", c.compile.empty() ? Json() : Json::str(c.compile));
    list.push(std::move(o));
  }
  tune.set("results", std::move(list));
  root.set("tuning", std::move(tune));

  Json cache = Json::obj();
  cache.set("op", Json::str("boom_fattn"));
  cache.set("shape_class", Json::str(r.class_str));
  cache.set("version", Json::integer(r.cache_version));
  cache.set("hit", Json::boolean(r.cache_hit));
  cache.set("status", Json::str(r.cache_status));
  cache.set("reason", r.cache_reason.empty() ? Json() : Json::str(r.cache_reason));
  root.set("cache", std::move(cache));
  return root;
}

void emit_report(const ReportData& r) {
  if (!report_enabled()) return;
  const Dev& d = device();
  std::string human = "[SonicBoom autotune] ";
  human += "GPU=\"" + std::string(d.ok ? d.prop.name : "unavailable") + "\"";
  human += " CC=" + (d.ok ? cc_string(d) : std::string("unavailable"));
  human += " SM=" + (d.ok ? std::to_string(d.prop.multiProcessorCount)
                          : std::string("unavailable"));
  human += " smem/block=" + (d.ok ? std::to_string(d.prop.sharedMemPerBlockOptin)
                                  : std::string("unavailable"));
  human += " smem/mp=" + (d.ok ? std::to_string(d.prop.sharedMemPerMultiprocessor)
                               : std::string("unavailable"));
  human += " regs/mp=" + (d.ok ? std::to_string(d.prop.regsPerMultiprocessor)
                               : std::string("unavailable"));
  human += " maxthr/block=" + (d.ok ? std::to_string(d.prop.maxThreadsPerBlock)
                                    : std::string("unavailable"));
  human += " driver_cuda=" + (d.driver_cuda > 0
                                  ? cuda_version_text(d.driver_cuda)
                                  : std::string("unavailable"));
  human += " runtime_cuda=" + (d.runtime_cuda > 0
                                   ? cuda_version_text(d.runtime_cuda)
                                   : std::string("unavailable"));
  human += " sonicboom=" SONICBOOM_VERSION_STRING;
  human += " impl=boom_fattn registry=" + std::to_string(kRegistryVersion);
  human += " source=" + r.source;
  const std::string na = "unavailable";
  human += " selected=" + (r.failed ? std::string("none") : r.selected_id);
  human += " id=" + (r.failed ? na : std::to_string(r.selected_num));
  human += " BR=" + (r.failed ? na : std::to_string(r.geo.br));
  human += " BK=" + (r.failed ? na : std::to_string(r.geo.bk));
  human += " KSTRIDE=" + (r.failed ? na : std::to_string(r.geo.kstride));
  human += " BKP=" + (r.failed ? na : std::to_string(r.geo.bkp));
  human += " softmax_warps=" +
           (r.failed ? na : std::to_string(r.geo.softmax_warps));
  human += std::string(" schedule=") +
           (r.failed ? na : std::string(schedule_name(r.geo)));
  human += " kernel=" + (r.failed ? na
                            : r.impl.empty() ? std::string("aot") : r.impl);
  human += " compile=" + (r.failed ? na : r.compile);
  human += " smem=" + (r.failed ? na : std::to_string(r.geo.smem));
  human += " regs=" + (r.regs >= 0 ? std::to_string(r.regs)
                                  : std::string("unavailable"));
  human += " spill=" + (r.spill_bytes >= 0 ? std::to_string(r.spill_bytes)
                                          : std::string("unavailable"));
  human += " n=" + std::to_string(r.queries);
  human += " candidates=" + std::to_string(r.n_total);
  human += " eliminated=" + std::to_string(r.n_eliminated);
  human += " compile_failed=" + std::to_string(r.n_compile_failed);
  human += " correctness_failed=" + std::to_string(r.n_correctness_failed);
  human += " timed=" + std::to_string(r.n_timed);
  human += " reps=" + std::to_string(r.reps);
  human += " launches_per_sample=" + std::to_string(r.iters);
  human += " cache=" + std::string(r.cache_hit ? "hit" : "miss");
  human += " cache_status=" + r.cache_status;
  human += " fingerprint=v" + std::to_string(kFingerprintVersion);
  human += " capability=" + r.fp.capability;
  human += " software=" + r.fp.software;
  human += " workload=" + r.fp.workload;
  if (r.fp.sim_capability || r.fp.sim_software || r.fp.sim_workload)
    human += " fingerprints_simulated=1";
  if (!r.cache_reason.empty()) human += " cache_reason=\"" + r.cache_reason + "\"";
  if (!r.reason.empty()) human += " reason=\"" + r.reason + "\"";
  human += "\n";

  const std::string json =
      "[SonicBoom autotune JSON] " + report_json(r).dump() + "\n";

  std::fputs(human.c_str(), stderr);
  std::fputs(json.c_str(), stderr);
  std::fflush(stderr);

  // The console keeps the readable line first; the exported file is a machine
  // record, so its first line is the JSON one.
  if (const char* path = env("SONICBOOM_FATTN_REPORT_FILE"); path != nullptr)
    append_file(path, json + human);

  const std::string log = report_log_path();
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(log).parent_path(), ec);
  append_file(log, json);
}

// ---------------------------------------------------------------------------
// Device launch target: an AOT kernel from this library's cubin, or a JIT
// module for exactly this device.
// ---------------------------------------------------------------------------
struct Target {
  void* fn = nullptr;
  bool jit = false;
  fattn_jit::Spec spec;
  int regs = -1;
  int spill_bytes = -1;
  int max_threads = -1;
  bool smem_set = false;
  std::string impl;
  std::string compile;
};

double* cmp_sums() {
  static double* p = [] {
    void* v = nullptr;
    if (cudaMalloc(&v, 3 * sizeof(double)) != cudaSuccess) {
      cudaGetLastError();
      return static_cast<double*>(nullptr);
    }
    return static_cast<double*>(v);
  }();
  return p;
}

bool resolve_target(const Call& c, const Cand& cand, int jit, Target* t,
                    std::string* why) {
  const Dev& d = device();
  t->spec.id = cand.num;
  t->spec.head_dim = c.head_dim;
  t->spec.gqa = c.gqa;
  t->spec.br = 16;
  t->spec.bk = 16;
  t->spec.nw = 4;
  t->spec.kpad = cand.kpad;
  t->spec.vpad = cand.vpad;
  t->spec.spw = cand.spw;
  t->spec.pipe = cand.pipe;

  if (jit != int(Jit::Force)) {
    void* fn = aot_kernel(cand.num, c.head_dim);
    if (fn != nullptr) {
      cudaFuncAttributes attr{};
      const cudaError_t e = cudaFuncGetAttributes(&attr, fn);
      if (e == cudaErrorInvalidDeviceFunction) {
        cudaGetLastError();  // expected on a device the cubin was not built for
      } else if (e != cudaSuccess) {
        *why = std::string("cudaFuncGetAttributes: ") + cudaGetErrorString(e);
        cudaGetLastError();
        return false;
      } else {
        t->fn = fn;
        t->jit = false;
        t->regs = attr.numRegs;
        t->spill_bytes = int(attr.localSizeBytes);
        t->max_threads = attr.maxThreadsPerBlock;
        t->impl = "aot";
        t->compile = "aot-cubin";
        return true;
      }
    }
    if (jit == int(Jit::Off)) {
      *why = "no AOT image for sm_" + std::to_string(d.cc) +
             " and SONICBOOM_FATTN_JIT=off";
      return false;
    }
  }

  if (!fattn_jit::available()) {
    *why = "no AOT image for sm_" + std::to_string(d.cc) + "; JIT unavailable (" +
           fattn_jit::unavailable_reason() + ")";
    return false;
  }
  const std::string path = fattn_jit::cache_path(t->spec);
  const bool cached = !path.empty() && ::access(path.c_str(), R_OK) == 0;
  std::string err;
  if (!fattn_jit::prepare(t->spec, &err)) {
    *why = "JIT: " + err;
    return false;
  }
  fattn_jit::attributes(t->spec, &t->regs, &t->spill_bytes, &t->max_threads);
  t->fn = nullptr;
  t->jit = true;
  t->impl = "jit";
  t->compile = cached ? "ptx-cache" : "nvrtc";
  return true;
}

// Launch a resolved target with the call's parameters. `out` is the destination
// (the caller's buffer, or the tuning scratch).
bool launch_target(Target& t, const Call& c, const Geometry& g, float* out) {
  cudaStream_t stream = ::sonicboom::cuda::sb_current_stream();
  const __half* q = c.q16;
  const __half* k = c.k_ws;
  const __half* v = c.v_ws;
  const uint64_t* start = c.start;
  uint64_t n_slots = c.n_slots;
  uint64_t window = c.window;
  int n = c.n;
  int nq = c.n_heads_q;
  int nkv = c.n_heads_kv;
  int ne_pad = c.ne_pad;
  float scale = c.scale;
  void* args[12] = {&q,  &k,   &v,  &start, &n_slots, &window,
                    &n,  &nq,  &nkv, &ne_pad, &scale,  &out};

  const dim3 grid((unsigned)((c.n + g.br - 1) / g.br), (unsigned)c.n_heads_kv);
  SB_NOTE_LAUNCH(boom_fattn_attn_qreg_kernel, grid, g.block, g.smem, stream);

  if (t.jit)
    return fattn_jit::launch(t.spec, grid, args, stream, g.smem, g.block);

  if (!t.smem_set) {
    if (cudaFuncSetAttribute(t.fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             int(g.smem)) != cudaSuccess) {
      cudaGetLastError();
      return false;
    }
    t.smem_set = true;
  }
  return cudaLaunchKernel(t.fn, grid, dim3((unsigned)g.block), args, g.smem,
                          stream) == cudaSuccess;
}

// ---------------------------------------------------------------------------
// Resource legality. Applied before anything is compiled or timed, and again
// to a candidate read from the cache: a record is a historical measurement, not
// permission to run a kernel the device cannot run.
// ---------------------------------------------------------------------------
bool resource_legal(const Dev& d, const Call& c, const Cand& cand,
                    const Geometry& g, std::string* why) {
  if (d.cc < 80) {
    *why = "cp.async needs compute capability 8.0+ (device is " + cc_string(d) +
           ")";
    return false;
  }
  if (g.smem > size_t(d.prop.sharedMemPerBlockOptin)) {
    *why = "needs " + std::to_string(g.smem) +
           " B shared memory; this device allows " +
           std::to_string(d.prop.sharedMemPerBlockOptin) + " B per block";
    return false;
  }
  if (g.block > d.prop.maxThreadsPerBlock) {
    *why = "needs a " + std::to_string(g.block) +
           "-thread block; the device allows " +
           std::to_string(d.prop.maxThreadsPerBlock);
    return false;
  }
  if (((2 * (c.head_dim + cand.kpad)) % 16) != 0 ||
      ((2 * (g.bk + cand.vpad)) % (cand.vpad == 4 ? 8 : 16)) != 0) {
    *why = "the K/V row lengths are not a whole number of cp.async transfers";
    return false;
  }
  return true;
}

// The register count is only known after the kernel is resolved. An unknown
// count is not permission to run: a candidate whose resource use cannot be
// established is eliminated, the same rule as a failed device probe. (Both the
// AOT and the JIT path obtain the count while resolving, so this only fires
// when the attributes could not be read at all.)
bool registers_legal(const Dev& d, const Target& t, const Geometry& g,
                     std::string* why) {
  if (t.regs <= 0) {
    *why = "the kernel's register usage could not be determined";
    return false;
  }
  if ((long long)t.regs * g.block > (long long)d.prop.regsPerMultiprocessor) {
    *why = "needs more registers per block than the device has";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Output comparison.
// ---------------------------------------------------------------------------
__device__ __forceinline__ bool sb_nonfinite(float v) {
  return !(v == v) || ::isinf(v);
}

__global__ void fattn_cmp_kernel(const float* __restrict__ a,
                                 const float* __restrict__ b, long long n,
                                 double* __restrict__ sums) {
  double sd = 0.0;
  double sr = 0.0;
  long long bad = 0;
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    const float x = a[i];
    const float y = b[i];
    if (sb_nonfinite(x) || sb_nonfinite(y)) {
      ++bad;
      continue;
    }
    const double diff = (double)x - (double)y;
    sd += diff * diff;
    sr += (double)y * (double)y;
  }
  __shared__ double sh_d[32];
  __shared__ double sh_r[32];
  __shared__ long long sh_b[32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int off = 16; off > 0; off >>= 1) {
    sd += __shfl_down_sync(0xffffffffu, sd, off);
    sr += __shfl_down_sync(0xffffffffu, sr, off);
    bad += __shfl_down_sync(0xffffffffu, bad, off);
  }
  if (lane == 0) {
    sh_d[warp] = sd;
    sh_r[warp] = sr;
    sh_b[warp] = bad;
  }
  __syncthreads();
  if (warp == 0) {
    const int warps = int(blockDim.x >> 5);
    sd = lane < warps ? sh_d[lane] : 0.0;
    sr = lane < warps ? sh_r[lane] : 0.0;
    bad = lane < warps ? sh_b[lane] : 0;
    for (int off = 16; off > 0; off >>= 1) {
      sd += __shfl_down_sync(0xffffffffu, sd, off);
      sr += __shfl_down_sync(0xffffffffu, sr, off);
      bad += __shfl_down_sync(0xffffffffu, bad, off);
    }
    if (lane == 0) {
      atomicAdd(&sums[0], sd);
      atomicAdd(&sums[1], sr);
      atomicAdd(&sums[2], (double)bad);
    }
  }
}

struct Compare {
  double rel_l2 = -1.0;
  long long nonfinite = -1;
  bool ok = false;
};

bool compare_outputs(const float* a, const float* b, size_t n,
                     cudaStream_t stream, Compare* out) {
  double* sums = cmp_sums();
  if (sums == nullptr) return false;
  if (cudaMemsetAsync(sums, 0, 3 * sizeof(double), stream) != cudaSuccess)
    return false;
  fattn_cmp_kernel<<<dim3(256), dim3(256), 0, stream>>>(a, b, (long long)n,
                                                        sums);
  double host[3] = {0.0, 0.0, 0.0};
  if (cudaMemcpyAsync(host, sums, sizeof(host), cudaMemcpyDeviceToHost,
                      stream) != cudaSuccess)
    return false;
  if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
  out->rel_l2 = std::sqrt(host[0] / (host[1] > 0.0 ? host[1] : 1.0));
  out->nonfinite = (long long)host[2];
  out->ok = true;
  return true;
}

// ---------------------------------------------------------------------------
// Timing.
// ---------------------------------------------------------------------------
struct Timer {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;

  bool init() {
    if (cudaEventCreate(&begin) != cudaSuccess) return false;
    if (cudaEventCreate(&end) != cudaSuccess) {
      cudaEventDestroy(begin);
      begin = nullptr;
      return false;
    }
    return true;
  }
  ~Timer() {
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
  }
};

// One timed sample = `iters` back-to-back launches of the same candidate; the
// reported value is per launch. At short context a single launch costs a few
// microseconds, which is the same order as launch latency and timer noise, so
// a one-launch sample cannot rank candidates. A batch makes the sample about
// kTargetSampleMs long and averages the launch path out of the comparison.
bool time_launch(Timer& t, Target& target, const Call& c, const Geometry& g,
                 float* out, int iters, double* ms) {
  if (cudaEventRecord(t.begin) != cudaSuccess) return false;
  for (int k = 0; k < iters; ++k)
    if (!launch_target(target, c, g, out)) return false;
  if (cudaEventRecord(t.end) != cudaSuccess) return false;
  if (cudaEventSynchronize(t.end) != cudaSuccess) return false;
  float elapsed = 0.0f;
  if (cudaEventElapsedTime(&elapsed, t.begin, t.end) != cudaSuccess) return false;
  *ms = double(elapsed) / double(iters);
  return true;
}

double median(std::vector<double> v) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// ---------------------------------------------------------------------------
// The process memo: one resolved class, launched with no further work.
// ---------------------------------------------------------------------------
struct Memo {
  bool resolved = false;
  Target target;
  Geometry geo;
  std::string source;
  std::string reason;
};

// The shape family a call belongs to regardless of its KV span: (head dim,
// GQA). It names the geometry selected() reports, not a selection.
uint64_t class_key(int head_dim, int gqa) {
  return (uint64_t(head_dim) << 8) | (uint64_t(gqa) & 0xffu);
}

std::mutex& memo_mutex() {
  static std::mutex m;
  return m;
}

// Resolved classes, keyed by the full selection key (fingerprints + span
// bucket). One process can therefore hold several entries for one (head_dim,
// GQA) -- e.g. a short-context prefill and a long-context one -- without
// resolving either twice.
std::map<std::string, Memo>& memo() {
  static std::map<std::string, Memo> m;
  return m;
}

// The geometry last used for each (head_dim, GQA) family, for selected().
std::map<uint64_t, Geometry>& last_geometry() {
  static std::map<uint64_t, Geometry> m;
  return m;
}

thread_local std::string t_last_error;

// Parse SONICBOOM_FATTN_CANDIDATES ("1,3,5" or "k8-v0-s2-pipe,...").
bool parse_ids(const char* text, std::vector<int>* out, std::string* why) {
  std::string_view s(text);
  while (!s.empty()) {
    const size_t comma = s.find(',');
    std::string_view item = s.substr(0, comma);
    while (!item.empty() && (item.front() == ' ' || item.front() == '\t'))
      item.remove_prefix(1);
    while (!item.empty() && (item.back() == ' ' || item.back() == '\t'))
      item.remove_suffix(1);
    if (!item.empty()) {
      const std::string copy(item);
      const Cand* cand = cand_by_name_or_num(copy.c_str());
      if (cand == nullptr) {
        *why = "unknown candidate '" + copy + "'";
        return false;
      }
      out->push_back(cand->num);
    }
    if (comma == std::string_view::npos) break;
    s.remove_prefix(comma + 1);
  }
  return true;
}

std::vector<int> tuning_set(std::string* why) {
  std::vector<int> ids;
  if (const char* e = env("SONICBOOM_FATTN_CANDIDATES"); e != nullptr) {
    if (!parse_ids(e, &ids, why)) return {};
    return ids;
  }
  for (const Cand& c : kCands)
    if (c.in_set) ids.push_back(c.num);
  return ids;
}

// ---------------------------------------------------------------------------
// Class resolution.
// ---------------------------------------------------------------------------
struct Tuned {
  bool ok = false;
  std::string source;
  std::string reason;
  const Cand* cand = &kCands[0];
  Target target;
  Geometry geo;
  ReportData report;
};

void report_prologue(ReportData* r, const Call& c, const Fingerprints& fp) {
  r->fp = fp;
  r->class_str = fp.key;
  r->budget_ms = budget_ms();
  r->head_dim = c.head_dim;
  r->gqa = c.gqa;
  r->queries = c.n;
  r->n_heads_q = c.n_heads_q;
  r->n_heads_kv = c.n_heads_kv;
  r->window = c.window;
  r->capturing = ::sonicboom::cuda::sb_capturing();
  r->cache_version = mlir_tune::cache_version();
  if (fp.cache_disabled) {
    r->cache_status = "unavailable";
    r->cache_reason = "the selection key (" + std::to_string(fp.key.size()) +
                      " bytes) does not fit the cache's " +
                      std::to_string(
                          mlir_tune::AutotuneCache::max_shape_class_bytes()) +
                      "-byte key field";
  }
}

// The reference selection: no measurement, just the incumbent candidate.
Tuned reference_tuned(const Call& c, const Fingerprints& fp,
                      const char* source, std::string reason) {
  Tuned t;
  t.cand = &kCands[0];
  t.geo = geometry(*t.cand, c.head_dim, c.gqa);
  std::string why;
  report_prologue(&t.report, c, fp);
  t.report.source = source;
  t.report.reason = reason;
  t.report.selected_id = t.cand->id;
  t.report.selected_num = t.cand->num;
  t.report.geo = t.geo;
  // The fallback is the reference geometry, not an exemption from the checks:
  // if the reference cannot legally run here, nothing runs and the caller's
  // own path is used. This is also what keeps a failed or unreliable device
  // probe from turning into a guess.
  if (!resource_legal(device(), c, *t.cand, t.geo, &why) ||
      !resolve_target(c, *t.cand, int(jit_mode()), &t.target, &why) ||
      !registers_legal(device(), t.target, t.geo, &why)) {
    t.ok = false;
    t.reason = why;
    return t;
  }
  t.ok = true;
  t.source = source;
  t.reason = std::move(reason);
  t.report.regs = t.target.regs;
  t.report.spill_bytes = t.target.spill_bytes;
  t.report.impl = t.target.impl;
  t.report.compile = t.target.compile;
  return t;
}

struct Ready {
  const Cand* cand = nullptr;
  Geometry geo;
  Target target;
  bool is_ref = false;
};

// Set a report result entry by candidate number.
CandResult* result_for(ReportData* rep, int num) {
  for (CandResult& r : rep->cands)
    if (r.num == num) return &r;
  return nullptr;
}

Tuned tune_class(const Call& c, const Fingerprints& fp) {
  const Dev& d = device();
  Tuned t;
  report_prologue(&t.report, c, fp);
  ReportData& rep = t.report;

  // ---- candidate list ----------------------------------------------------
  std::string parse_why;
  std::vector<int> ids = tuning_set(&parse_why);
  if (ids.empty()) {
    return reference_tuned(
        c, fp, "fallback",
        parse_why.empty() ? "the candidate set is empty" : parse_why);
  }
  // The reference is always first: it is the comparison basis and the fallback.
  ids.erase(std::remove(ids.begin(), ids.end(), kCands[0].num), ids.end());
  ids.insert(ids.begin(), kCands[0].num);
  rep.n_total = int(ids.size());
  for (const int num : ids) {
    const Cand* cand = cand_by_num(num);
    CandResult r;
    r.id = cand != nullptr ? cand->id : "?" + std::to_string(num);
    r.num = num;
    r.status = "excluded";
    rep.cands.push_back(std::move(r));
  }

  // ---- resource legality (before any compile) ----------------------------
  std::vector<const Cand*> legal;
  for (const int num : ids) {
    const Cand* cand = cand_by_num(num);
    CandResult* r = result_for(&rep, num);
    if (cand == nullptr) {
      r->reason = "not in this build's candidate table";
      ++rep.n_eliminated;
      continue;
    }
    const Geometry g = geometry(*cand, c.head_dim, c.gqa);
    r->smem = g.smem;
    std::string why;
    if (!resource_legal(d, c, *cand, g, &why)) {
      r->status = "eliminated";
      r->reason = why;
      ++rep.n_eliminated;
      continue;
    }
    legal.push_back(cand);
  }
  if (legal.empty() || legal.front()->num != kCands[0].num) {
    std::string why = "no candidate passed the resource check on this device";
    if (const CandResult* r = result_for(&rep, kCands[0].num);
        r != nullptr && !r->reason.empty())
      why = "the reference candidate cannot run: " + r->reason;
    return reference_tuned(c, fp, "fallback", why);
  }

  // ---- the reference, compiled and launched once -------------------------
  Ready ref{&kCands[0], geometry(kCands[0], c.head_dim, c.gqa), Target{}, true};
  {
    std::string why;
    if (!resolve_target(c, kCands[0], int(jit_mode()), &ref.target, &why)) {
      CandResult* r = result_for(&rep, kCands[0].num);
      r->status = "compile-failed";
      r->reason = why;
      ++rep.n_compile_failed;
      return reference_tuned(c, fp, "fallback",
                             "the reference candidate did not compile: " + why);
    }
    CandResult* r = result_for(&rep, kCands[0].num);
    r->regs = ref.target.regs;
    r->impl = ref.target.impl;
    r->compile = ref.target.compile;
    if (!registers_legal(d, ref.target, ref.geo, &why)) {
      r->status = "eliminated";
      r->reason = why;
      ++rep.n_eliminated;
      return reference_tuned(c, fp, "fallback",
                             "the reference candidate " + why);
    }
  }

  // ---- scratch -----------------------------------------------------------
  cudaStream_t stream = ::sonicboom::cuda::sb_current_stream();
  const size_t out_elems = (size_t)c.n * (size_t)c.n_heads_q * (size_t)c.head_dim;
  const size_t out_bytes = out_elems * sizeof(float);
  float* scratch_ref = nullptr;
  float* scratch_cand = nullptr;
  const bool alloc_ok = cudaMalloc(&scratch_ref, out_bytes) == cudaSuccess &&
                        cudaMalloc(&scratch_cand, out_bytes) == cudaSuccess;
  if (!alloc_ok) cudaGetLastError();
  struct Guard {
    float* a;
    float* b;
    ~Guard() {
      if (a) cudaFree(a);
      if (b) cudaFree(b);
    }
  } guard{scratch_ref, scratch_cand};
  if (!alloc_ok)
    return reference_tuned(c, fp, "default",
                           "the tuning scratch allocation failed; reference "
                           "candidate");
  Timer timer;
  if (!timer.init()) {
    cudaGetLastError();
    return reference_tuned(c, fp, "default",
                           "could not create CUDA events; reference candidate");
  }

  // The probe: the reference's first launch warms the class, produces the
  // comparison basis and measures the per-launch cost the budget is spent in.
  // It is untimed, as every candidate's first launch is.
  {
    const cudaError_t e = [&] {
      if (!launch_target(ref.target, c, ref.geo, scratch_ref))
        return cudaGetLastError();
      cudaEventRecord(timer.begin);
      if (!launch_target(ref.target, c, ref.geo, scratch_ref))
        return cudaGetLastError();
      cudaEventRecord(timer.end);
      cudaEventSynchronize(timer.end);
      return cudaGetLastError();
    }();
    if (e != cudaSuccess) {
      return reference_tuned(c, fp, "default",
                             std::string("the reference candidate failed to "
                                         "launch: ") +
                                 cudaGetErrorString(e));
    }
    float probe = 0.0f;
    cudaEventElapsedTime(&probe, timer.begin, timer.end);
    rep.probe_ms = double(probe);
    Compare cmp{};
    if (!compare_outputs(scratch_ref, scratch_ref, out_elems, stream, &cmp) ||
        cmp.nonfinite > 0) {
      return reference_tuned(c, fp, "fallback",
                             "the reference output is not finite on this input");
    }
  }

  // ---- budget ------------------------------------------------------------
  const double est = std::max(rep.probe_ms, 0.01);
  if (rep.budget_ms > 0 && est > double(rep.budget_ms) / 4.0) {
    CandResult* r = result_for(&rep, kCands[0].num);
    r->status = "reference";
    r->median_ms = rep.probe_ms;
    r->reps = 1;
    return reference_tuned(
        c, fp, "default",
        "one launch (" + std::to_string(int(est + 0.5)) +
            " ms) exceeds the cold-tuning budget share; reference candidate");
  }
  // How many launches one timed sample holds. Chosen so a sample lasts about
  // kTargetSampleMs regardless of the workload size; the cost of the tuning
  // pass is then ~ncand * (1 + reps * iters) launches.
  const int iters = std::clamp(int(std::ceil(kTargetSampleMs / est)), 1, kMaxIters);
  int ncand = int(legal.size());
  if (rep.budget_ms > 0) {
    const int cap =
        std::clamp(int(double(rep.budget_ms) / (est * (1.0 + iters))), 2, ncand);
    ncand = std::min(ncand, cap);
  }
  int reps = kMaxReps;
  if (rep.budget_ms > 0) {
    const double left = double(rep.budget_ms) - est * double(ncand);
    reps = int(left / (double(ncand) * double(iters) * est));
    reps = std::clamp(reps, 1, kMaxReps);
  }
  rep.reps = reps;
  rep.iters = iters;
  rep.warmup = 1;

  // ---- compile the candidates that will be timed -------------------------
  std::vector<Ready> ready;
  ready.push_back(ref);
  for (int i = 1; i < ncand; ++i) {
    const Cand* cand = legal[size_t(i)];
    CandResult* r = result_for(&rep, cand->num);
    Ready rd;
    rd.cand = cand;
    rd.geo = geometry(*cand, c.head_dim, c.gqa);
    rd.is_ref = false;
    std::string why;
    if (!resolve_target(c, *cand, int(jit_mode()), &rd.target, &why)) {
      r->status = "compile-failed";
      r->reason = why;
      ++rep.n_compile_failed;
      continue;
    }
    r->regs = rd.target.regs;
    r->impl = rd.target.impl;
    r->compile = rd.target.compile;
    if (!registers_legal(d, rd.target, rd.geo, &why)) {
      r->status = "eliminated";
      r->reason = why;
      ++rep.n_eliminated;
      continue;
    }
    ready.push_back(std::move(rd));
  }
  for (size_t i = size_t(std::max(ncand, 1)); i < legal.size(); ++i) {
    CandResult* r = result_for(&rep, legal[i]->num);
    if (r->status != "excluded") continue;
    r->status = "skipped-budget";
    r->reason = "the cold-tuning budget allows " + std::to_string(ncand) +
                " of " + std::to_string(legal.size()) + " candidates";
  }

  // ---- measure -----------------------------------------------------------
  // Every candidate gets one untimed warm-up launch (its first launch pays for
  // module/kernel first-touch) followed by `reps` timed samples of `iters`
  // launches each.
  auto measure = [&](Ready& rd, float* dst) -> bool {
    if (!launch_target(rd.target, c, rd.geo, dst)) return false;
    std::vector<double> samples;
    for (int k = 0; k < reps; ++k) {
      double ms = 0.0;
      if (!time_launch(timer, rd.target, c, rd.geo, dst, iters, &ms))
        return false;
      samples.push_back(ms);
    }
    CandResult* r = result_for(&rep, rd.cand->num);
    r->median_ms = median(samples);
    r->reps = int(samples.size());
    return true;
  };

  // The reference: its probe launch was the warm-up, so it goes straight to
  // the timed samples every challenger gets.
  CandResult* ref_r = result_for(&rep, kCands[0].num);
  {
    std::vector<double> samples;
    bool ok = true;
    for (int k = 0; k < reps && ok; ++k) {
      double ms = 0.0;
      ok = time_launch(timer, ref.target, c, ref.geo, scratch_ref, iters, &ms);
      if (ok) samples.push_back(ms);
    }
    if (!ok) {
      cudaGetLastError();
      return reference_tuned(c, fp, "default",
                             "the reference candidate failed while timing");
    }
    ref_r->median_ms = median(samples);
    ref_r->reps = int(samples.size());
  }
  ref_r->status = "reference";
  rep.ref_ms = ref_r->median_ms;

  for (size_t i = 1; i < ready.size(); ++i) {
    Ready& rd = ready[i];
    CandResult* r = result_for(&rep, rd.cand->num);
    // The warm-up launch also produces the output the correctness gate reads.
    if (!launch_target(rd.target, c, rd.geo, scratch_cand)) {
      cudaGetLastError();
      r->status = "compile-failed";
      r->reason = "the kernel failed to launch";
      ++rep.n_compile_failed;
      continue;
    }
    Compare cmp{};
    if (!compare_outputs(scratch_cand, scratch_ref, out_elems, stream, &cmp) ||
        cmp.nonfinite > 0 || cmp.rel_l2 > kMaxRelL2) {
      r->status = "correctness-failed";
      r->rel_l2 = cmp.rel_l2;
      r->reason = !cmp.ok ? "the output comparison failed"
                  : cmp.nonfinite > 0
                      ? std::to_string(cmp.nonfinite) +
                            " non-finite output elements vs the reference"
                      : "relative L2 " + std::to_string(cmp.rel_l2) +
                            " exceeds the gate";
      ++rep.n_correctness_failed;
      continue;
    }
    r->rel_l2 = cmp.rel_l2;
    std::vector<double> samples;
    bool failed = false;
    for (int k = 0; k < reps; ++k) {
      double ms = 0.0;
      if (!time_launch(timer, rd.target, c, rd.geo, scratch_cand, iters, &ms)) {
        failed = true;
        break;
      }
      samples.push_back(ms);
    }
    if (failed) {
      cudaGetLastError();
      r->status = "compile-failed";
      r->reason = "the kernel failed to launch while timing";
      ++rep.n_compile_failed;
      continue;
    }
    r->status = "timed";
    r->median_ms = median(samples);
    r->reps = int(samples.size());
    ++rep.n_timed;
  }

  // ---- choose (incumbent protection) ------------------------------------
  CandResult* best_r = ref_r;
  for (CandResult& r : rep.cands) {
    if (r.status != "timed") continue;
    if (r.median_ms < best_r->median_ms) best_r = &r;
  }
  // A challenger has to beat the reference by the margin to be selected. A
  // challenger that is ahead but inside the margin is a close race: both sides
  // get two more samples so the decision is not made on a single look. A
  // challenger that is merely at or behind the incumbent is left alone.
  if (best_r != ref_r && best_r->median_ms < ref_r->median_ms &&
      best_r->median_ms > ref_r->median_ms * (1.0 - kMarginPct / 100.0) &&
      rep.budget_ms > 0 &&
      double(rep.budget_ms) -
              est * double(ready.size()) * double(1 + reps * iters) >
          4.0 * est * iters) {
    Ready* best_ready = nullptr;
    for (Ready& rd : ready)
      if (rd.cand->num == best_r->num) best_ready = &rd;
    double extra_ref[2] = {0.0, 0.0};
    double extra_ch[2] = {0.0, 0.0};
    bool ok = best_ready != nullptr;
    for (int k = 0; k < 2 && ok; ++k)
      ok = time_launch(timer, ref.target, c, ref.geo, scratch_ref, iters,
                       &extra_ref[k]);
    for (int k = 0; k < 2 && ok; ++k)
      ok = time_launch(timer, best_ready->target, c, best_ready->geo,
                       scratch_cand, iters, &extra_ch[k]);
    if (ok) {
      const double ref_old = ref_r->median_ms;
      const double ch_old = best_r->median_ms;
      ref_r->median_ms = median({ref_old, ref_old, extra_ref[0], extra_ref[1]});
      ref_r->reps += 2;
      rep.ref_ms = ref_r->median_ms;
      best_r->median_ms = median({ch_old, ch_old, extra_ch[0], extra_ch[1]});
      best_r->reps += 2;
    }
  }
  const Cand* winner = best_r->median_ms <
                               ref_r->median_ms * (1.0 - kMarginPct / 100.0)
                           ? cand_by_num(best_r->num)
                           : &kCands[0];
  if (winner == nullptr) winner = &kCands[0];
  Ready* winner_ready = winner->num == kCands[0].num ? &ref : nullptr;
  for (Ready& rd : ready)
    if (rd.cand->num == winner->num) winner_ready = &rd;
  if (winner_ready == nullptr)
    return reference_tuned(c, fp, "fallback", "internal: winner not resolved");

  // ---- persist -----------------------------------------------------------
  if (!fp.cache_disabled) {
    mlir_tune::AutotuneCache::instance().store(
        "boom_fattn", fp.key,
        mlir_tune::TileConfig{winner->num, winner->kpad, winner->vpad,
                            winner->spw, winner->pipe});
  }

  t.ok = true;
  t.cand = winner;
  t.geo = winner_ready->geo;
  t.target = winner_ready->target;
  t.source = "measured";
  t.reason.clear();
  rep.source = "measured";
  rep.fresh = true;
  rep.reason.clear();
  rep.selected_id = winner->id;
  rep.selected_num = winner->num;
  rep.geo = t.geo;
  rep.regs = t.target.regs;
  rep.spill_bytes = t.target.spill_bytes;
  rep.impl = t.target.impl;
  rep.compile = t.target.compile;
  rep.cache_hit = false;
  return t;
}

// Resolve one class: cache (unless forced), then measurement. The caller (run)
// has already memoized, so this runs once per selection key per process.
//
// The report's cache.status says what happened to the persistent lookup:
//   hit               a record for this key was used
//   miss              no record for this key
//   version-mismatch  the file was written by another schema; nothing was read
//   unreadable        no file/lock/valid header
//   invalidated       a record existed but names a candidate/layout the device
//                     or this build cannot run; the class was measured again
//   unavailable       the key does not fit the record layout
//   bypassed          the lookup was skipped (autotune=force, forced candidate)
//   disabled          autotuning is off
Tuned resolve_class(const Call& c, const Fingerprints& fp) {
  const Dev& d = device();
  const std::string& cls = fp.key;
  const Autotune mode = autotune_mode();

  if (mode == Autotune::Off) {
    Tuned t = reference_tuned(c, fp, "default", "autotuning disabled");
    t.report.cache_status = "disabled";
    t.report.cache_reason = "autotuning disabled (SONICBOOM_FATTN_AUTOTUNE=0)";
    emit_report(t.report);
    return t;
  }

  // A forced candidate still goes through legality and compilation: it is a
  // selection override, not a way to run something the device cannot.
  if (const char* forced = env("SONICBOOM_FATTN_CANDIDATE"); forced != nullptr) {
    const std::string bypass = std::string("SONICBOOM_FATTN_CANDIDATE='") +
                               forced + "': the cache was not consulted";
    Tuned t;
    const Cand* cand = cand_by_name_or_num(forced);
    std::string why;
    if (cand == nullptr) {
      why = std::string("SONICBOOM_FATTN_CANDIDATE='") + forced +
            "' is not a candidate";
    } else {
      t.cand = cand;
      t.geo = geometry(*cand, c.head_dim, c.gqa);
      if (!resource_legal(d, c, *cand, t.geo, &why)) {
        // why already says what the device cannot provide
      } else if (!resolve_target(c, *cand, int(jit_mode()), &t.target, &why)) {
        why = "cannot run: " + why;
      } else if (!registers_legal(d, t.target, t.geo, &why)) {
        // as above
      }
    }
    if (!why.empty()) {
      Tuned fb = reference_tuned(c, fp, "fallback",
                                 "forced candidate could not run (" + why +
                                     "); reference candidate used");
      fb.report.cache_status = "bypassed";
      fb.report.cache_reason = bypass;
      emit_report(fb.report);
      return fb;
    }
    report_prologue(&t.report, c, fp);
    t.ok = true;
    t.source = "forced";
    t.report.source = "forced";
    t.report.selected_id = cand->id;
    t.report.selected_num = cand->num;
    t.report.geo = t.geo;
    t.report.regs = t.target.regs;
    t.report.spill_bytes = t.target.spill_bytes;
    t.report.impl = t.target.impl;
    t.report.compile = t.target.compile;
    t.report.n_total = 1;
    t.report.cache_status = "bypassed";
    t.report.cache_reason = bypass;
    emit_report(t.report);
    return t;
  }

  // Cache first, unless a fresh measurement was requested.
  std::string cache_status;
  std::string cache_reason;
  if (mode == Autotune::Force) {
    cache_status = "bypassed";
    cache_reason = "SONICBOOM_FATTN_AUTOTUNE=force: the cache was not consulted";
  } else if (fp.cache_disabled) {
    cache_status = "unavailable";
    cache_reason = "the selection key (" + std::to_string(cls.size()) +
                   " bytes) does not fit the cache's " +
                   std::to_string(mlir_tune::AutotuneCache::max_shape_class_bytes()) +
                   "-byte key field";
  } else {
    const auto look =
        mlir_tune::AutotuneCache::instance().lookup_ex("boom_fattn", cls);
    switch (look.status) {
      case mlir_tune::CacheStatus::Hit: {
        const mlir_tune::TileConfig& cfg = look.cfg;
        const Cand* cand = cand_by_num(cfg.tile_m);
        const bool match = cand != nullptr && cfg.tile_n == cand->kpad &&
                           cfg.tile_k == cand->vpad && cfg.unroll == cand->spw &&
                           cfg.vectorize == cand->pipe;
        if (!match) {
          cache_status = "invalidated";
          cache_reason = "the record names candidate " +
                         std::to_string(cfg.tile_m) + " (" +
                         cfg.to_string() + "), which this build does not have";
          break;
        }
        Tuned t;
        t.cand = cand;
        t.geo = geometry(*cand, c.head_dim, c.gqa);
        std::string why;
        // A record is a historical measurement, not permission to run a kernel
        // the device cannot run: legality, compile and registers are checked
        // again on the cache path.
        if (!resource_legal(d, c, *cand, t.geo, &why)) {
          cache_status = "invalidated";
          cache_reason = "the cached candidate cannot run here: " + why;
          break;
        }
        if (!resolve_target(c, *cand, int(jit_mode()), &t.target, &why)) {
          cache_status = "invalidated";
          cache_reason = "the cached candidate did not compile: " + why;
          break;
        }
        if (!registers_legal(d, t.target, t.geo, &why)) {
          cache_status = "invalidated";
          cache_reason = "the cached candidate cannot run here: " + why;
          break;
        }
        report_prologue(&t.report, c, fp);
        t.ok = true;
        t.source = "cache";
        t.report.source = "cache";
        t.report.cache_hit = true;
        t.report.cache_status = "hit";
        t.report.cache_reason = "record for this selection key";
        t.report.selected_id = cand->id;
        t.report.selected_num = cand->num;
        t.report.geo = t.geo;
        t.report.regs = t.target.regs;
        t.report.spill_bytes = t.target.spill_bytes;
        t.report.impl = t.target.impl;
        t.report.compile = t.target.compile;
        emit_report(t.report);
        return t;
      }
      case mlir_tune::CacheStatus::NotFound:
        cache_status = "miss";
        cache_reason = "no record for this selection key";
        break;
      case mlir_tune::CacheStatus::VersionMismatch:
        cache_status = "version-mismatch";
        cache_reason =
            "the cache file was written with schema " +
            std::to_string(look.file_version) + "; this build uses schema " +
            std::to_string(mlir_tune::cache_version()) +
            ", so the file was not read";
        break;
      case mlir_tune::CacheStatus::Unreadable:
        cache_status = "unreadable";
        cache_reason = "the cache file could not be read";
        break;
    }
  }

  if (env_is("SONICBOOM_FATTN_SHOW", "1")) {
    Tuned t = reference_tuned(c, fp, "default",
                              "no cached selection for this class "
                              "(SONICBOOM_FATTN_SHOW=1 does not tune)");
    t.report.cache_status = cache_status;
    t.report.cache_reason = cache_reason;
    emit_report(t.report);
    return t;
  }

  Tuned t = tune_class(c, fp);
  t.report.cache_status = cache_status;
  t.report.cache_reason = cache_reason;
  emit_report(t.report);
  return t;
}

}  // namespace

bool supported(int head_dim, int gqa) {
  return (head_dim == 256 || head_dim == 512) && gqa == 4;
}

bool run(const Call& call) {
  if (!supported(call.head_dim, call.gqa) || call.n <= 0 ||
      call.n_heads_q <= 0 || call.n_heads_kv <= 0 || call.q16 == nullptr ||
      call.k_ws == nullptr || call.v_ws == nullptr || call.start == nullptr ||
      call.out == nullptr) {
    t_last_error = "unsupported shape";
    return false;
  }
  if (call.n_heads_q != call.gqa * call.n_heads_kv) {
    t_last_error = "n_heads_q is not gqa * n_heads_kv";
    return false;
  }
  const Dev& d = device();
  if (!d.ok) {
    // No reliable device facts means no legality decision and no key: say so
    // instead of guessing, and let the caller use its own path.
    t_last_error = d.error;
    Fingerprints none;
    none.capability = none.software = none.workload = "unavailable";
    none.key = "unavailable";
    ReportData rep;
    report_prologue(&rep, call, none);
    rep.failed = true;
    rep.source = "error";
    rep.reason = d.error + std::string(": no candidate was selected");
    rep.cache_status = "unavailable";
    rep.cache_reason =
        "the device could not be queried, so the cache was not consulted";
    emit_report(rep);
    return false;
  }

  const Fingerprints fp = build_fingerprints(call);

  std::lock_guard<std::mutex> lock(memo_mutex());
  Memo& m = memo()[fp.key];

  if (!m.resolved) {
    // Under CUDA-graph capture a measurement would enqueue events and
    // allocations inside the capture. Take the reference and leave the class
    // unresolved, so the first non-captured call still tunes.
    if (::sonicboom::cuda::sb_capturing()) {
      Tuned t = reference_tuned(call, fp, "default",
                                "CUDA-graph capture; no tuning under capture");
      t.report.capturing = true;
      t.report.cache_status = "bypassed";
      t.report.cache_reason = "CUDA-graph capture: the cache is not consulted";
      emit_report(t.report);
      if (!t.ok) {
        t_last_error = t.reason;
        return false;
      }
      last_geometry()[class_key(call.head_dim, call.gqa)] = t.geo;
      return launch_target(t.target, call, t.geo, call.out);
    }
    Tuned t = resolve_class(call, fp);
    if (!t.ok) {
      t_last_error = t.reason.empty() ? "no runnable candidate" : t.reason;
      // Say so explicitly, and with nothing selected: a consumer must never
      // read a failed configuration as the chosen one.
      ReportData rep;
      report_prologue(&rep, call, fp);
      rep.failed = true;
      rep.source = "error";
      rep.reason = t_last_error;
      rep.cache_status = t.report.cache_status;
      rep.cache_reason = t.report.cache_reason;
      emit_report(rep);
      return false;
    }
    m.resolved = true;
    m.target = t.target;
    m.geo = t.geo;
    m.source = t.source;
    m.reason = t.reason;
  }

  last_geometry()[class_key(call.head_dim, call.gqa)] = m.geo;

  if (!launch_target(m.target, call, m.geo, call.out)) {
    cudaGetLastError();
    t_last_error = "the selected candidate failed to launch";
    return false;
  }
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    t_last_error = std::string("attention launch: ") + cudaGetErrorString(e);
    return false;
  }
  return true;
}

bool selected(int head_dim, int gqa, Geometry* geo) {
  if (!supported(head_dim, gqa)) return false;
  std::lock_guard<std::mutex> lock(memo_mutex());
  // The geometry of the last class this family ran. Several classes (KV span
  // buckets, software versions) can be resolved at once; this reports the one
  // that is actually in use, not an arbitrary map entry.
  const auto it = last_geometry().find(class_key(head_dim, gqa));
  if (it != last_geometry().end()) {
    if (geo) *geo = it->second;
    return true;
  }
  if (geo) *geo = geometry(kCands[0], head_dim, gqa);
  return true;
}

const char* last_error() { return t_last_error.c_str(); }

void reset_for_test() {
  std::lock_guard<std::mutex> lock(memo_mutex());
  memo().clear();
  t_last_error.clear();
}

}  // namespace sonicboom::nn::cuda::fattn_tune
