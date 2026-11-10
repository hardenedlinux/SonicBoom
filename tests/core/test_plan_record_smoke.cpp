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

// Plan recording smoke test: the six model/mode cases, recorded twice each in
// one process.
//
// The plan record is the runtime's description of what it executes — every
// step, every node, every launch. Two properties must hold with nothing
// changed between the recordings:
//
//   1. Determinism: the same case recorded twice yields the same text. A
//      recording that reads device state, an unresolved tuning choice or a
//      pooled buffer that another case last wrote would show up here.
//   2. A clean context: no CUDA error survives a recording. A sticky error
//      would otherwise surface at whatever call comes next.
//
// The recordings themselves — device lines, kernel geometry, fusion and
// dispatch decisions — are engine internals. Their byte-exact goldens, and the
// pinned tune table they were recorded against, live in the private tree's
// test data, not here; this test compares a recording only against itself.
//
// Skips (exit 77) without a CUDA device or the three model GGUFs.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/plan_record.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

#ifndef SONICBOOM_SRC_DIR
#define SONICBOOM_SRC_DIR "."
#endif

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

namespace {

struct Case {
  const char* name;
  const char* env;   // SONICBOOM_*_GGUF
  const char* file;  // default file name under models/
  bool decode;
  uint64_t n_ctx;  // decode only
  uint64_t n;      // prefill only
};

const Case kCases[] = {
    {"gemma4-decode", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
     true, 512, 0},
    {"gemma4-prefill", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
     false, 0, 8},
    {"qwen2-decode", "SONICBOOM_QWEN_GGUF",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf", true, 512, 0},
    {"qwen2-prefill", "SONICBOOM_QWEN_GGUF",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf", false, 0, 8},
    {"deepseek-r1-decode", "SONICBOOM_DEEPSEEK_GGUF",
     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", true, 512, 0},
    {"deepseek-r1-prefill", "SONICBOOM_DEEPSEEK_GGUF",
     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", false, 0, 8},
};

int g_failures = 0;

void fail(const std::string& what) {
  std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  ++g_failures;
}

std::string model_path(const Case& c) {
  if (const char* env = std::getenv(c.env); env && *env) return env;
  return std::string(SONICBOOM_SRC_DIR) + "/models/" + c.file;
}

// The first line that differs, with both sides; the trailing lines are the
// recordings' common tail and say nothing about the divergence.
std::string first_difference(const std::string& a, const std::string& b) {
  std::size_t la = 0;
  std::size_t lb = 0;
  std::size_t line = 1;
  while (la < a.size() && lb < b.size()) {
    std::size_t ea = a.find('\n', la);
    std::size_t eb = b.find('\n', lb);
    if (ea == std::string::npos) ea = a.size();
    if (eb == std::string::npos) eb = b.size();
    const std::string x = a.substr(la, ea - la);
    const std::string y = b.substr(lb, eb - lb);
    if (x != y)
      return "line " + std::to_string(line) + ":\n- " + x + "\n+ " + y;
    la = ea + 1;
    lb = eb + 1;
    ++line;
  }
  if (la < a.size() || lb < b.size())
    return "one recording ends at line " + std::to_string(line);
  return {};
}

bool cuda_clean(const std::string& after) {
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    fail("CUDA error after " + after + ": " + cudaGetErrorString(e));
    return false;
  }
  return true;
}

std::string record(const Case& c, const sbmodel::Gemma4Model& model) {
  auto run = c.decode ? sbmodel::record_decode_plan(model, c.n_ctx, 1)
                      : sbmodel::record_prefill_plan(model, c.n);
  if (!run) return {};
  return sbplanner::serialize_plan_run(*run);
}

}  // namespace

int main() {
  // The recording pins the same two decisions the goldens pin: the JIT
  // candidate tuner's picks are a per-device measurement (they would depend on
  // the cold run's timing), and the f32 dense arm's choice is between two
  // different kernels. Neither may be left to whatever the tune cache holds.
  ::setenv("SONICBOOM_AUTOTUNE_JIT", "0", 1);
  ::setenv("SONICBOOM_DENSE_F32", "lt", 1);

  if (!sonicboom::quant::cuda_available()) {
    std::printf("test_plan_record_smoke SKIP (no CUDA device)\n");
    return 77;
  }

  for (const Case& c : kCases) {
    const std::string path = model_path(c);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      std::printf("test_plan_record_smoke SKIP (model absent: %s)\n",
                  path.c_str());
      return 77;
    }
  }

  for (const Case& c : kCases) {
    auto reader = sbgguf::Reader::load(model_path(c));
    if (!reader) {
      fail(std::string(c.name) + ": failed to load " + model_path(c));
      continue;
    }
    auto model = sbmodel::load_gemma4(*reader);
    if (!model) {
      fail(std::string(c.name) + ": model load error: " + model.error());
      continue;
    }

    const std::string first = record(c, *model);
    if (!cuda_clean(std::string(c.name) + " recording 1")) continue;
    if (first.empty()) {
      fail(std::string(c.name) + ": recording 1 produced no plan");
      continue;
    }
    const std::string second = record(c, *model);
    if (!cuda_clean(std::string(c.name) + " recording 2")) continue;
    if (second.empty()) {
      fail(std::string(c.name) + ": recording 2 produced no plan");
      continue;
    }

    if (const std::string d = first_difference(first, second); !d.empty()) {
      fail(std::string(c.name) + ": the two recordings differ: " + d);
      continue;
    }
    std::printf("  %s: %zu bytes, identical in-process\n", c.name, first.size());
    std::fflush(stdout);

    // The next case loads another model's weights into the same pool.
    sonicboom::quant::cuda::clear_cache();
  }

  if (g_failures == 0) {
    std::printf("test_plan_record_smoke OK\n");
    return 0;
  }
  std::fprintf(stderr, "test_plan_record_smoke FAILED: %d check(s)\n",
               g_failures);
  return 1;
}
