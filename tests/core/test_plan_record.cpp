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

// Execution-plan record acceptance (the plan format,
// docs/static-execution-plan-format-v0.md). Two properties per model/mode case:
//
//   1. Determinism — recording and serializing the same plan twice yields
//      byte-identical text. This is what makes the record diffable at all;
//      without it tools/plan_diff would report noise.
//   2. Golden — the regenerated text equals the committed file under
//      tests/data/plans/. This is the regression gate: any change to the
//      planner's steps, fusion, buffer assignment, or launched kernels shows up
//      as a plan diff.
//
// The golden comparison is skipped — never the determinism check — when the
// recording device differs from the golden's, because part of the record is
// device-dependent (the decode-attention split-KV grid is sized from the SM
// count). The device identity is line 9 of every record.
//
// Regenerate the goldens with:
//     test_plan_record --write-goldens
// The case parameters live here, so the goldens are by construction what this
// test compares against.
//
// Skips (exit 77) when there is no CUDA device (the record needs one) or when
// none of the model GGUFs is present.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/numerics.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/plan_record.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

namespace {

int g_failures = 0;

void fail(const std::string& what) {
  std::cerr << "  FAIL: " << what << "\n";
  ++g_failures;
}

// One model/mode case. `n_ctx`/`steps` are for decode, `n` for prefill.
struct Case {
  const char* name;
  const char* env;       // SONICBOOM_*_GGUF
  const char* file;      // default file name under models/
  const char* mode;      // "decode" | "prefill"
  uint64_t n_ctx = 0;
  uint64_t n = 0;
  uint64_t steps = 0;
};

const Case kCases[] = {
    {"gemma4-decode", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
     "decode", 512, 0, 1},
    {"gemma4-prefill", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
     "prefill", 0, 8, 0},
    {"qwen2-decode", "SONICBOOM_QWEN_GGUF",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf", "decode", 512, 0, 1},
    {"qwen2-prefill", "SONICBOOM_QWEN_GGUF",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf", "prefill", 0, 8, 0},
    // Third model: Qwen2 architecture at a different size, so it exercises the
    // same planner path with different shapes; no model-specific code.
    {"deepseek-r1-decode", "SONICBOOM_DEEPSEEK_GGUF",
     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", "decode", 512, 0, 1},
    {"deepseek-r1-prefill", "SONICBOOM_DEEPSEEK_GGUF",
     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", "prefill", 0, 8, 0},
};

std::string find_model(const Case& c) {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv(c.env)) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) + "/models/" + c.file);
#endif
  candidates.push_back(std::string("models/") + c.file);
  for (const auto& cand : candidates)
    if (std::filesystem::exists(cand)) return cand;
  return {};
}

std::string golden_path(const Case& c) {
  // One golden per numerics profile. Strict keeps the historical <name>.plan
  // names, so its bytes stay comparable with every pre-profile run; standard
  // records <name>.standard.plan, because f16 KV changes the launched attention
  // variants (`-kv16` classes) and their smem. The two profiles are allowed to
  // differ, so neither file may stand in for the other.
  const std::string suffix =
      sonicboom::numerics_profile() == sonicboom::NumericsProfile::Standard
          ? ".standard"
          : "";
#ifdef SONICBOOM_SRC_DIR
  return std::string(SONICBOOM_SRC_DIR) + "/tests/data/plans/" + c.name + suffix +
         ".plan";
#else
  return std::string("tests/data/plans/") + c.name + suffix + ".plan";
#endif
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  std::istringstream ss(s);
  std::string line;
  while (std::getline(ss, line)) out.push_back(line);
  return out;
}

// The value of the `device` line, or "" when the record has none.
std::string device_line(const std::string& text) {
  for (const std::string& l : split_lines(text))
    if (l.rfind("device ", 0) == 0) return l.substr(7);
  return {};
}

// Report the first differing line, mirroring tools/plan_diff — the tool and the
// test must not disagree about what "different" means, so keep them in step.
void report_diff(const std::string& want, const std::string& got,
                 const std::string& want_label, const std::string& got_label) {
  const std::vector<std::string> a = split_lines(want), b = split_lines(got);
  const std::size_t common = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < common && a[i] == b[i]) ++i;
  if (i >= common) {
    std::cerr << "  first difference: line count " << a.size() << " (" << want_label
              << ") vs " << b.size() << " (" << got_label << "), first extra line "
              << i + 1 << "\n";
    return;
  }
  std::cerr << "  first difference: line " << i + 1 << "\n"
            << "    " << want_label << ": " << a[i] << "\n"
            << "    " << got_label << ": " << b[i] << "\n";
}

// The device weight/tokenizer caches are process-global and keyed by host
// addresses. A case's model (and its host weight memory) dies at the end of its
// iteration, so the next case must drop the caches first: otherwise the mapped
// per-layer-embedding registration of the dead model still covers freed host
// memory, and the driver then fails unrelated later cudaMalloc calls with
// "resource already mapped" (208) — which truncates the next recording.
// quant::cuda::clear_cache is the documented teardown for a model reload.
struct CudaCacheGuard {
  ~CudaCacheGuard() { sonicboom::quant::cuda::clear_cache(); }
};

bool write_file(const std::string& path, const std::string& text) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << text;
  return f.good();
}

}  // namespace

int main(int argc, char** argv) {
  // The committed goldens record the hand-written plan: the JIT candidate
  // tuner's picks are a per-device measurement decision (recorded with the
  // other decisions, shown by sb-explain), not part of the plan-shape
  // contract. Leaving it on would make the golden text depend on the timing of
  // the cold run and on the JIT cache state, which is exactly what this test
  // exists to exclude. The hand-written tuner stays on, so the golden's own
  // generated-attention shape (what that tuner picked) is reproduced. The f32
  // dense arm is pinned because its two arms launch different kernels: the
  // golden records the Lt arm (no vendor tail kernels), independent of the
  // device's dense tune cache.
  ::setenv("SONICBOOM_AUTOTUNE_JIT", "0", 1);
  ::setenv("SONICBOOM_DENSE_F32", "lt", 1);
  bool write_goldens = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--write-goldens") {
      write_goldens = true;
    } else if (a == "-h" || a == "--help") {
      std::printf("usage: %s [--write-goldens]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown option %s\n", argv[0], argv[i]);
      return 2;
    }
  }

  if (!sonicboom::quant::cuda_available()) {
    std::cout << "(skipped: no CUDA device — the plan record needs one)\n";
    return 77;
  }

  int ran = 0;
  for (const Case& c : kCases) {
    CudaCacheGuard cache_guard;  // drops this case's caches when it goes out of scope
    const std::string path = find_model(c);
    if (path.empty()) {
      std::cout << "(" << c.name << ": skipped, " << c.env << " unset and models/"
                << c.file << " absent)\n";
      continue;
    }
    auto reader = sbgguf::Reader::load(path);
    if (!reader) {
      fail(std::string(c.name) + ": failed to load " + path);
      continue;
    }
    auto model = sbmodel::load_gemma4(*reader);
    if (!model) {
      fail(std::string(c.name) + ": model load error: " + model.error());
      continue;
    }

    ++ran;
    std::cout << c.name << " (" << c.mode << ")\n";

    auto record = [&]() -> std::string {
      auto run = std::string_view(c.mode) == "decode"
                     ? sbmodel::record_decode_plan(*model, c.n_ctx, c.steps)
                     : sbmodel::record_prefill_plan(*model, c.n);
      if (!run) return {};
      return sbplanner::serialize_plan_run(*run);
    };

    const std::string first = record();
    if (first.empty()) {
      fail(std::string(c.name) + ": recording produced no plan");
      continue;
    }

    // 1. Determinism: a second, independent recording must serialize identically.
    const std::string second = record();
    if (second != first) {
      fail(std::string(c.name) + ": two recordings differ (non-deterministic record)");
      report_diff(first, second, "run 1", "run 2");
    } else {
      std::cout << "  deterministic: " << split_lines(first).size() << " lines\n";
    }

    // 2. Golden.
    const std::string gpath = golden_path(c);
    if (write_goldens) {
      if (!write_file(gpath, first)) {
        fail(std::string(c.name) + ": cannot write golden " + gpath);
      } else {
        std::cout << "  wrote " << gpath << "\n";
      }
      continue;
    }
    const std::string golden = read_file(gpath);
    if (golden.empty()) {
      fail(std::string(c.name) + ": golden missing or empty: " + gpath);
      continue;
    }
    if (device_line(golden) != device_line(first)) {
      std::cout << "  golden skipped: recorded on '" << device_line(golden)
                << "', running on '" << device_line(first) << "'\n";
      continue;
    }
    if (golden != first) {
      fail(std::string(c.name) + ": plan differs from the golden " + gpath +
           " (compare the two with tools/plan_diff)");
      report_diff(golden, first, "golden", "current");
    } else {
      std::cout << "  matches golden\n";
    }
  }

  if (ran == 0 && g_failures == 0) {
    std::cout << "(skipped: no model GGUFs found)\n";
    return 77;
  }
  if (g_failures != 0) {
    std::cerr << g_failures << " plan-record check(s) failed\n";
    return 1;
  }
  std::cout << "plan-record: " << ran << " case(s) OK\n";
  return 0;
}
