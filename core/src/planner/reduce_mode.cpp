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

#include "reduce_mode.h"

#include "decision_registry.h"

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#if SONICBOOM_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

namespace sonicboom::planner {
namespace {

// The performance-ratio threshold that makes a double accumulator affordable:
// the same 4x the device query uses (singleToDoublePrecisionPerfRatio is "how
// many f32 ops cost one f64 op"). Above it, DF32 is the default.
constexpr double kMaxDoubleCostRatio = 4.0;

std::optional<compiler::ReduceMode> env_override() {
  const char *env = std::getenv("SONICBOOM_REDUCE");
  if (env == nullptr || *env == '\0') return std::nullopt;
  if (auto mode = compiler::parse_reduce_mode(env)) return mode;
  std::fprintf(stderr,
               "SONICBOOM_REDUCE: unknown mode '%s' (f64|df32|f32); using auto\n",
               env);
  return std::nullopt;
}

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

// --- per-device acceptance verdict ------------------------------------------
//
// The cost ratio says whether a double accumulator is affordable; it does not
// say whether DF32 is numerically acceptable. That second question is measured
// once per device by tools/sb-numerics (the acceptance suite: per-kernel fp64
// errors, 3x128-token greedy streams and 50 top-1 prompts per model) and
// recorded here, in the same per-device tree as the autotune choices. Absent a
// verdict (a device that has never run the suite) DF32 is not selected: the
// safe choice is F64, which is what the committed goldens encode.

enum class Verdict { Absent, Accepted, Rejected };

Verdict verdict_file(const std::string &path) {
  std::ifstream f(path);
  std::string line;
  if (!std::getline(f, line)) return Verdict::Absent;
  if (line.rfind("df32 accepted", 0) == 0) return Verdict::Accepted;
  if (line.rfind("df32 rejected", 0) == 0) return Verdict::Rejected;
  return Verdict::Absent;
}

const char *verdict_name(Verdict verdict) {
  switch (verdict) {
    case Verdict::Accepted: return "accepted";
    case Verdict::Rejected: return "rejected";
    case Verdict::Absent: return "absent";
  }
  return "absent";
}

Verdict df32_verdict() {
  const std::string id = device_identity();
  return verdict_file(decision_cache_root() + "/" + id +
                      "/reduce-acceptance-v1");
}

// The hosted kernels are measured in the same run, but their verdict is keyed
// by the host (the CPU measurement and the CPU greedy streams), not by the
// device.
Verdict cpu_df32_verdict() {
  return verdict_file(decision_cache_root() + "/cpu/reduce-acceptance-cpu-v1");
}

void record_verdict(std::string_view scope, Verdict verdict) {
  decisions::record("reduce.df32-acceptance", scope, verdict_name(verdict),
                    verdict == Verdict::Absent ? decisions::Source::Default
                                               : decisions::Source::Verdict);
}

// --- CPU measurement -------------------------------------------------------

// Sum of squares over a representative norm row, both accumulators, timed by
// the caller. volatile prevents the optimizer from deleting the work while the
// inner loop stays the same code a regression of norm.cpp would run.
float g_bench_row[4096];

double time_f64(int n, int reps) {
  double sink = 0.0;
  for (int r = 0; r < reps; ++r) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
      const float v = g_bench_row[i];
      s += double(v * v);
    }
    sink += s;
  }
  return sink;
}

double time_df32(int n, int reps) {
  double sink = 0.0;
  for (int r = 0; r < reps; ++r) {
    compiler::Df32 s;
    for (int i = 0; i < n; ++i) {
      const float v = g_bench_row[i];
      s = compiler::df32_add_float(s, v * v);
    }
    sink += compiler::df32_value(s);
  }
  return sink;
}

template <typename F>
double best_ns(F&& body) {
  using clock = std::chrono::steady_clock;
  double best = 1e300;
  for (int trial = 0; trial < 3; ++trial) {
    const auto t0 = clock::now();
    const double sink = body();
    const auto t1 = clock::now();
    // Keep the result alive across the loop without a print.
    if (sink == 1234567.0) std::fprintf(stderr, " ");
    const double ns =
        std::chrono::duration<double, std::nano>(t1 - t0).count();
    if (ns < best) best = ns;
  }
  return best;
}

compiler::ReduceMode measure_cpu() {
  constexpr int n = 2560;
  constexpr int reps = 512;
  for (int i = 0; i < n; ++i) {
    // Deterministic, positive, ~1.0: the distribution of a norm row.
    g_bench_row[i] = 0.5f + float((i * 2654435761u) % 1000) / 1000.0f;
  }
  const double ns64 = best_ns([&] { return time_f64(n, reps); });
  const double nsdf =
      best_ns([&] { return time_df32(n, reps); });
  if (nsdf <= 0.0) return compiler::ReduceMode::F64;
  const double ratio = ns64 / nsdf;  // >1: f64 is the more expensive one
  return ratio <= kMaxDoubleCostRatio ? compiler::ReduceMode::F64
                                      : compiler::ReduceMode::Df32;
}

} // namespace

std::string device_identity() {
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

std::string decision_cache_root() {
  if (const char *e = std::getenv("SONICBOOM_TUNE_DIR"))
    if (*e != '\0') return e;
  if (const char *e = std::getenv("XDG_CACHE_HOME"))
    if (*e != '\0') return std::string(e) + "/sonicboom/tune";
  if (const char *e = std::getenv("HOME"))
    if (*e != '\0') return std::string(e) + "/.cache/sonicboom/tune";
  return ".sonicboom-tune";
}

bool record_df32_acceptance(bool accepted, std::string *error) {
  const std::string dir = decision_cache_root() + "/" + device_identity();
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    if (error) *error = "cannot create " + dir + ": " + ec.message();
    return false;
  }
  const std::string cpu_dir = decision_cache_root() + "/cpu";
  std::filesystem::create_directories(cpu_dir, ec);
  if (ec) {
    if (error) *error = "cannot create " + cpu_dir + ": " + ec.message();
    return false;
  }
  const char *body = accepted ? "df32 accepted\n" : "df32 rejected\n";
  std::ofstream f(dir + "/reduce-acceptance-v1", std::ios::trunc);
  std::ofstream g(cpu_dir + "/reduce-acceptance-cpu-v1", std::ios::trunc);
  if (!f || !g) {
    if (error) *error = "cannot write the acceptance verdict";
    return false;
  }
  f << body;
  g << body;
  return f.good() && g.good();
}

std::optional<compiler::ReduceMode> reduce_mode_env() { return env_override(); }

compiler::ReduceMode device_reduce_mode() {
  static const compiler::ReduceMode mode = [] {
    if (auto env = env_override()) {
      decisions::record("reduce.mode", "device",
                        compiler::reduce_mode_name(*env),
                        decisions::Source::Env, "SONICBOOM_REDUCE");
      return *env;
    }
#if SONICBOOM_USE_CUDA
    int device = 0;
    if (cudaGetDevice(&device) == cudaSuccess) {
      int ratio = 0;
      if (cudaDeviceGetAttribute(
              &ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) ==
              cudaSuccess &&
          ratio > 0) {
        // The ratio is the affordability test; the recorded acceptance verdict
        // is the numerics test. Both must pass for DF32.
        decisions::record("reduce.double-cost-ratio", "device",
                          std::to_string(ratio), decisions::Source::Default);
        const Verdict verdict = df32_verdict();
        record_verdict("device", verdict);
        const auto chosen =
            ratio <= kMaxDoubleCostRatio || verdict != Verdict::Accepted
                ? compiler::ReduceMode::F64
                : compiler::ReduceMode::Df32;
        decisions::record("reduce.mode", "device",
                          compiler::reduce_mode_name(chosen),
                          decisions::Source::Default, "SONICBOOM_REDUCE");
        return chosen;
      }
    }
#endif
    decisions::record("reduce.mode", "device", "f64",
                      decisions::Source::Default, "SONICBOOM_REDUCE");
    return compiler::ReduceMode::F64;
  }();
  return mode;
}

compiler::ReduceMode cpu_reduce_mode() {
  static const compiler::ReduceMode mode = [] {
    if (auto env = env_override()) {
      decisions::record("reduce.mode", "cpu",
                        compiler::reduce_mode_name(*env),
                        decisions::Source::Env, "SONICBOOM_REDUCE");
      return *env;
    }
    const compiler::ReduceMode measured = measure_cpu();
    decisions::record("reduce.cpu-measurement", "cpu",
                      compiler::reduce_mode_name(measured),
                      decisions::Source::Measured);
    const Verdict verdict = cpu_df32_verdict();
    record_verdict("cpu", verdict);
    // The host is subject to the same two tests as the device: DF32 only when
    // the measurement says doubles are expensive here AND sb-numerics recorded
    // that DF32 is acceptable here.
    const compiler::ReduceMode chosen =
        measured == compiler::ReduceMode::Df32 && verdict != Verdict::Accepted
            ? compiler::ReduceMode::F64
            : measured;
    decisions::record("reduce.mode", "cpu", compiler::reduce_mode_name(chosen),
                      decisions::Source::Measured, "SONICBOOM_REDUCE");
    return chosen;
  }();
  return mode;
}

} // namespace sonicboom::planner
