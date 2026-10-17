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

// Runtime-feature switch tests. The disable list is parsed once per process, so
// the modes that need a different SONICBOOM_FEATURE_DISABLE value run as
// separate CTest invocations (see tests/core/CMakeLists.txt):
//
//   test_runtime_features              defaults, names, live aliases
//   test_runtime_features --disabled   SONICBOOM_FEATURE_DISABLE=list
//   test_runtime_features --unknown    unknown name must abort (WILL_FAIL)

#include <sonicboom/runtime_features.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>

using sonicboom::runtime::Feature;
using sonicboom::runtime::feature_disabled;
using sonicboom::runtime::feature_enabled;
using sonicboom::runtime::feature_name;

namespace {

int g_failures = 0;

void expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

constexpr Feature kAll[] = {
    Feature::CudaGraphDecode, Feature::CudaGraphPrefill, Feature::CaptureReuse,
    Feature::SplitKv,         Feature::PlanCache,       Feature::Q8_1Dispatch,
    Feature::BufferReuse,     Feature::Warmup,
};

bool known(const char* name) {
  for (Feature f : kAll)
    if (std::strcmp(feature_name(f), name) == 0) return true;
  return false;
}

} // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::strcmp(argv[1], "--disabled") == 0) {
    // CTest passes SONICBOOM_FEATURE_DISABLE=split_kv,plan_cache.
    expect(feature_disabled(Feature::SplitKv), "split_kv disabled");
    expect(feature_disabled(Feature::PlanCache), "plan_cache disabled");
    expect(feature_enabled(Feature::CudaGraphDecode), "cuda_graph_decode on");
    expect(feature_enabled(Feature::CudaGraphPrefill), "cuda_graph_prefill on");
    expect(feature_enabled(Feature::CaptureReuse), "capture_reuse on");
    expect(feature_enabled(Feature::Q8_1Dispatch), "q8_1_dispatch on");
    expect(feature_enabled(Feature::BufferReuse), "buffer_reuse on");
    expect(feature_enabled(Feature::Warmup), "warmup on");
    return g_failures == 0 ? 0 : 1;
  }

  // Unknown name: a child process sets the list before its first query (the
  // parse is once-per-process) and must abort. Forked so the parent's own parse
  // is still the default-env one.
  {
    const pid_t pid = fork();
    if (pid == 0) {
      setenv("SONICBOOM_FEATURE_DISABLE", "not_a_feature", 1);
      (void)feature_enabled(Feature::SplitKv);
      std::fprintf(stderr, "FAIL: unknown feature name was accepted\n");
      _exit(0);
    }
    int status = 0;
    expect(pid > 0 && waitpid(pid, &status, 0) == pid, "forked unknown-name child");
    expect(!(WIFEXITED(status) && WEXITSTATUS(status) == 0),
           "unknown feature name aborts");
  }

  // Default mode: every feature on, every name unique and known.
  for (Feature f : kAll) {
    expect(feature_enabled(f), feature_name(f));
    expect(feature_name(f) != nullptr, "feature has a name");
    expect(known(feature_name(f)), "feature name round-trips");
  }

  // Aliases are live (not folded into the once-parsed list): setting the old
  // variable mid-process flips its feature.
  setenv("SONICBOOM_GEMV_F32", "1", 1);
  expect(feature_disabled(Feature::Q8_1Dispatch), "SONICBOOM_GEMV_F32 alias");
  unsetenv("SONICBOOM_GEMV_F32");
  expect(feature_enabled(Feature::Q8_1Dispatch), "alias cleared");

  setenv("SONICBOOM_SPLIT_KV", "0", 1);
  expect(feature_disabled(Feature::SplitKv), "SONICBOOM_SPLIT_KV=0 alias");
  unsetenv("SONICBOOM_SPLIT_KV");
  expect(feature_enabled(Feature::SplitKv), "split_kv alias cleared");

  setenv("SONICBOOM_CUDA_GRAPHS", "0", 1);
  expect(feature_disabled(Feature::CudaGraphDecode),
         "SONICBOOM_CUDA_GRAPHS=0 disables decode capture");
  expect(feature_disabled(Feature::CudaGraphPrefill),
         "SONICBOOM_CUDA_GRAPHS=0 disables prefill capture");
  unsetenv("SONICBOOM_CUDA_GRAPHS");

  if (g_failures == 0) std::printf("runtime features: ok\n");
  return g_failures == 0 ? 0 : 1;
}
