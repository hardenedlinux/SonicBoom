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

// V2 spec auto-detect: with SONICBOOM_SPEC unset, the first generation call
// measures one plain decode step against one k=1 verify step and selects
// auto (ratio < 1.5) or off. The ratio is cached per SM in the Q1 autotune
// file, so a later process loads it instead of measuring.
//
// The probe timings are injected through SONICBOOM_SPEC_PROBE_MS, so these
// cases never depend on the GPU's speed: the decision, the log line, the cache
// round-trip and the environment override are what is under test.
//
//   part 0  the ratio record round-trips in the per-SM cache file
//   part 1  forked children, each with its own SONICBOOM_SPEC / cache dir:
//           measured auto, cached auto (injection would have said off), off,
//           SONICBOOM_SPEC=off (no probe), SONICBOOM_SPEC=on (no probe)
//
// Skips cleanly when the Gemma GGUF is absent.

#include "../../core/src/planner/mlir_autotune.h"

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace mt = sonicboom::mlir_tune;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::string find_file(const char* env, const std::string& rel) {
  std::vector<std::string> candidates;
  if (const char* e = std::getenv(env)) candidates.push_back(e);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) + "/" + rel);
#endif
  candidates.push_back(rel);
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

std::string slurp(const std::string& path) {
  std::ifstream f(path);
  return std::string((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
}

// --- part 0: the ratio record in the Q1 per-SM file ------------------------

void ratio_round_trip(const std::string& xdg) {
  ::setenv("XDG_CACHE_HOME", xdg.c_str(), 1);
  auto& cache = mt::AutotuneCache::instance();
  cache.reset_for_test();
  check(!mt::lookup_ratio("spec_verify_ratio", "decode"), "a fresh cache misses");
  check(mt::store_ratio("spec_verify_ratio", "decode", 1.94), "store 1.94");
  const auto r = mt::lookup_ratio("spec_verify_ratio", "decode");
  check(r && std::abs(*r - 1.94) < 1e-9, "the ratio round-trips");
  check(mt::store_ratio("spec_verify_ratio", "decode", 0.42), "update to 0.42");
  const auto r2 = mt::lookup_ratio("spec_verify_ratio", "decode");
  check(r2 && std::abs(*r2 - 0.42) < 1e-9, "the update replaces the record");
  check(!mt::lookup_ratio("spec_verify_ratio", "prefill"), "another shape misses");
  check(!mt::store_ratio("spec_verify_ratio", "decode", 0.0),
        "a non-positive ratio is rejected");
  // The record lives in the Q1 file, not the older JSON tune service.
  check(std::filesystem::exists(cache.path()), "the per-SM file exists");
}

// --- part 1: the probe decision in a child process --------------------------

struct ChildCase {
  const char* tag;
  const char* spec_env;    // nullptr = unset
  const char* probe_ms;    // nullptr = no injection
  const std::string* xdg;  // null = no cache dir override
  bool expect_spec;        // the speculative loop must run
  const char* expect_log;  // substring of the probe log; null = must be absent
};

int child_body(const sbmodel::Gemma4Model& m, const ChildCase& c,
               const std::string& log_path) {
  // stderr carries the probe line (and any plan diagnostics).
  if (std::freopen(log_path.c_str(), "w", stderr) == nullptr) return 1;
  if (c.spec_env)
    ::setenv("SONICBOOM_SPEC", c.spec_env, 1);
  else
    ::unsetenv("SONICBOOM_SPEC");
  if (c.probe_ms)
    ::setenv("SONICBOOM_SPEC_PROBE_MS", c.probe_ms, 1);
  else
    ::unsetenv("SONICBOOM_SPEC_PROBE_MS");
  if (c.xdg) ::setenv("XDG_CACHE_HOME", c.xdg->c_str(), 1);

  constexpr uint64_t kNctx = 128;
  constexpr uint64_t kOut = 8;
  const uint64_t seed = m.config.vocab_size / 3;
  std::vector<int64_t> got(kOut, -1);
  sbmodel::spec_counters_reset();
  const uint64_t done = sbmodel::generate_spine_cuda(m, seed, 0, got, kNctx);
  const sbmodel::SpecCounters counters = sbmodel::spec_counters();
  const bool spec_ran = counters.iterations > 0;
  check(done == kOut, "the generation produced every requested token");
  if (c.expect_spec)
    check(spec_ran, "the speculative loop ran");
  else
    check(!spec_ran, "the plain loop ran (no speculative iterations)");
  return g_failures == 0 ? 0 : 1;
}

int run_child(const sbmodel::Gemma4Model& m, const ChildCase& c,
              const std::string& log_path) {
  std::cout.flush();
  std::cerr.flush();
  const pid_t pid = ::fork();
  if (pid < 0) {
    std::perror("  fork");
    return 1;
  }
  if (pid == 0) {
    int rc = 1;
    try {
      rc = child_body(m, c, log_path);
    } catch (...) {
      rc = 1;
    }
    std::cout.flush();
    std::cerr.flush();
    ::_exit(rc);
  }
  int status = 0;
  if (::waitpid(pid, &status, 0) < 0) return 1;
  if (!WIFEXITED(status)) return 1;
  const int rc = WEXITSTATUS(status);
  const std::string log = slurp(log_path);
  const bool has_line = log.find("spec probe: ratio=") != std::string::npos;
  if (c.expect_log) {
    if (log.find(c.expect_log) == std::string::npos) {
      std::cerr << "  FAIL: [" << c.tag << "] log lacks '" << c.expect_log
                << "'\n";
      ++g_failures;
    }
  } else if (has_line) {
    std::cerr << "  FAIL: [" << c.tag << "] probed with SONICBOOM_SPEC set\n";
    ++g_failures;
  }
  if (rc != 0) {
    std::cerr << "  FAIL: [" << c.tag << "] child failed; log:\n" << log;
    ++g_failures;
  }
  return rc;
}

std::string mkdtemp_dir(const std::string& base) {
  std::string tmpl = base + "/sb-spec-probe-XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  if (::mkdtemp(buf.data()) == nullptr) return {};
  return std::string(buf.data());
}

} // namespace

int main() {
  namespace fs = std::filesystem;
  const std::string gemma_path = find_file(
      "SONICBOOM_GEMMA_GGUF", "models/gemma-4-E4B-it-Q3_K_M.gguf");
  if (gemma_path.empty()) {
    std::cout << "  (skipped: Gemma GGUF not found)\n";
    std::cout << "test_spec_autoprobe OK\n";
    return 0;
  }

  const std::string root = mkdtemp_dir("/tmp");
  if (root.empty()) {
    std::cerr << "mkdtemp failed\n";
    return 1;
  }

  auto r_gemma = sbgguf::Reader::load(gemma_path);
  if (!r_gemma) {
    std::cerr << "gguf load failed\n";
    return 1;
  }
  auto loaded = sbmodel::load_gemma4(*r_gemma);
  if (!loaded) {
    std::cerr << "model load: " << loaded.error() << "\n";
    return 1;
  }
  const sbmodel::Gemma4Model& m = *loaded;

  // Shared by the measured and the cached child: the first writes 1.20 (auto),
  // the second injects 2.00 (which would be off) and must still read 1.20.
  const std::string xdg_shared = root + "/cache-shared";
  fs::create_directories(xdg_shared);
  const std::string xdg_off = root + "/cache-off";
  fs::create_directories(xdg_off);
  const std::string xdg_env_off = root + "/cache-env-off";
  fs::create_directories(xdg_env_off);
  const std::string xdg_env_on = root + "/cache-env-on";
  fs::create_directories(xdg_env_on);

  const ChildCase cases[] = {
      {"measured-auto", nullptr, "2.0,2.4", &xdg_shared, true,
       "ratio=1.20 → auto"},
      {"cached-auto", nullptr, "2.0,4.0", &xdg_shared, true,
       "ratio=1.20 → auto"},
      {"measured-off", nullptr, "2.0,3.4", &xdg_off, false, "ratio=1.70 → off"},
      {"env-off", "off", "2.0,1.0", &xdg_env_off, false, nullptr},
      {"env-on", "on", nullptr, &xdg_env_on, true, nullptr},
  };
  for (const ChildCase& c : cases)
    run_child(m, c, root + "/" + c.tag + ".log");

  // After the forks, never before: resolving the cache path asks the driver for
  // the SM version, and a process that has touched CUDA must not fork children
  // that use it.
  ratio_round_trip(root + "/cache-unit");

  if (g_failures == 0) {
    std::cout << "test_spec_autoprobe OK\n";
    return 0;
  }
  std::cerr << "test_spec_autoprobe FAILED: " << g_failures << " check(s)\n";
  return 1;
}
