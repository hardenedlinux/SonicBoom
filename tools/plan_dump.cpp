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

// plan_dump — write a model's executed plan to a text file.
//
//   plan_dump -m <model.gguf> --mode decode --n-ctx N [--steps S] [-o out]
//   plan_dump -m <model.gguf> --mode prefill --n N [-o out]
//
// The output is the deterministic serialization of a planner::PlanRunRecord
// (planner/plan_record.h): every executed step in order, with its node, op,
// attributes, buffers, and the kernels it launched plus their launch
// configuration. Two runs of the same model/mode produce byte-identical output,
// which is what tools/plan_diff and the plan-record ctest rely on.
//
// Links libsonicboom.so only, through the public Layer 2 headers — the same
// contract the tests use (see tools/baseline/build_sonicboom.sh).

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/plan_record.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;

namespace {

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s -m <model.gguf> --mode decode --n-ctx N [--steps S] [-o out]\n"
               "       %s -m <model.gguf> --mode prefill --n N [-o out]\n",
               argv0, argv0);
}

}  // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* out_path = nullptr;
  std::string_view mode;
  uint64_t n_ctx = 0, n = 0, steps = 64;

  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s: option %s needs a value\n", argv[0], argv[i]);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "-m") {
      model_path = next();
    } else if (a == "-o") {
      out_path = next();
    } else if (a == "--mode") {
      mode = next();
    } else if (a == "--n-ctx") {
      n_ctx = std::strtoull(next(), nullptr, 10);
    } else if (a == "--n") {
      n = std::strtoull(next(), nullptr, 10);
    } else if (a == "--steps") {
      steps = std::strtoull(next(), nullptr, 10);
    } else if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown option %s\n", argv[0], argv[i]);
      usage(argv[0]);
      return 2;
    }
  }

  if (!model_path || mode.empty() ||
      (mode != "decode" && mode != "prefill") ||
      (mode == "decode" && n_ctx == 0) || (mode == "prefill" && n == 0) ||
      steps == 0) {
    usage(argv[0]);
    return 2;
  }

  auto r = sbgguf::Reader::load(model_path);
  if (!r) {
    std::fprintf(stderr, "failed to load GGUF: %s\n", model_path);
    return 1;
  }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) {
    std::fprintf(stderr, "model load error: %s\n", m.error().c_str());
    return 1;
  }

  auto run = mode == "decode" ? sbmodel::record_decode_plan(*m, n_ctx, steps)
                              : sbmodel::record_prefill_plan(*m, n);
  if (!run) {
    std::fprintf(stderr,
                 "record failed (%s): no CUDA support, no weights, or the plan "
                 "did not build\n",
                 std::string(mode).c_str());
    return 1;
  }

  const std::string text = sonicboom::planner::serialize_plan_run(*run);
  if (out_path) {
    FILE* f = std::fopen(out_path, "w");
    if (!f) {
      std::fprintf(stderr, "failed to open output: %s\n", out_path);
      return 1;
    }
    const std::size_t written = std::fwrite(text.data(), 1, text.size(), f);
    std::fclose(f);
    if (written != text.size()) {
      std::fprintf(stderr, "short write to %s\n", out_path);
      return 1;
    }
  } else {
    std::fwrite(text.data(), 1, text.size(), stdout);
  }
  return 0;
}
