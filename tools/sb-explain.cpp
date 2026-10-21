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

// sb-explain — print the decisions the runtime would take on this device.
//
// Three sources, in the order the runtime consults them:
//   1. environment overrides (shown with the static default they replace);
//   2. the per-device caches: the hand-written tuner's <op>.json choices and
//      the JIT candidate tuner's sm_<NN>.bin records;
//   3. the decision registry for the decisions this process resolves
//      (design/decision-registry.md).
//
// It never tunes: the JIT cache is only read, and the reduce-mode entries come
// from the same once-per-process functions the runtime uses. Exit 0 on
// success; 2 when the decision cache cannot be inspected.

#include "planner/autotune.h"
#include "planner/decision_registry.h"
#include "planner/mlir_autotune.h"
#include "planner/reduce_mode.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

struct Knob {
  const char* var;
  const char* fallback;  // what the runtime does when the variable is unset
  const char* meaning;
};

// The env knobs that select an algorithm, tile, mode or threshold (the
// diagnostic-only switches are deliberately absent).
constexpr Knob kKnobs[] = {
    {"SONICBOOM_REDUCE", "auto (cost ratio + recorded acceptance verdict)",
     "reduction accumulator: f64|df32|f32"},
    {"SONICBOOM_AUTOTUNE", "on", "tuning service: 0 off, force retimes"},
    {"SONICBOOM_AUTOTUNE_JIT", "on", "JIT candidate tuner only: 0 off"},
    {"SONICBOOM_AUTOTUNE_BUDGET_MS", "10000",
     "process-wide cold-tuning budget (<=0 unlimited)"},
    {"SONICBOOM_MMQ", "on", "MMQ prefill matmul: 0 off"},
    {"SONICBOOM_MMQ_J", "tuner", "MMQ J split when not tuned"},
    {"SONICBOOM_MMQ_TC", "off", "generated tensor-core matmul"},
    {"SONICBOOM_MMQ_MIN_N", "2", "MMQ crossover token count"},
    {"SONICBOOM_ATTENTION_IMPL", "auto", "prefill/decode attention implementation"},
    {"SONICBOOM_ATTN_PREFILL", "auto", "prefill attention: auto|own|codegen"},
    {"SONICBOOM_ATTN_DECODE", "auto", "decode attention: auto|own|codegen"},
    {"SONICBOOM_ATTN_DECODE_CHUNK", "identity rule", "decode split-KV chunk"},
    {"SONICBOOM_DENSE_CUBLAS", "on", "dense GEMM uses cuBLAS"},
    {"SONICBOOM_DENSE_CUBLAS_MIN_N", "1", "dense GEMM cuBLAS crossover"},
    {"SONICBOOM_DENSE_LT", "on", "cuBLASLt for the dense GEMM"},
    {"SONICBOOM_DENSE_GEMM", "on", "SonicBoom dense GEMM"},
    {"SONICBOOM_GEMV_COOP_MIN_BPR", "20", "cooperative gemv blocks-per-row floor"},
    {"SONICBOOM_SPEC", "probe (auto)", "speculative decoding: off|on|auto"},
    {"SONICBOOM_ASYNC_PREFETCH", "off", "prefill prefetch markers"},
};

void section(const char* title) {
  std::cout << "\n== " << title << " ==\n";
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "-h" || a == "--help") {
      std::cout << "usage: sb-explain [--json]\n"
                   "Print the automatic choices for this device: the env\n"
                   "overrides, the per-device caches, and the decision\n"
                   "registry. --json prints the registry as JSON. Nothing is\n"
                   "tuned; the only write is the registry itself.\n";
      return 0;
    }
    if (a == "--json") {
      json = true;
      continue;
    }
    std::cerr << "sb-explain: unknown option " << a << "\n";
    return 2;
  }

  // The device and host decisions (recorded into the registry as a side
  // effect of resolving them once, exactly as the runtime would), then the
  // static knob table, so the registry covers every algorithm-selecting
  // variable.
  const std::string identity = sonicboom::planner::device_identity();
  const auto device_mode = sonicboom::planner::device_reduce_mode();
  const auto cpu_mode = sonicboom::planner::cpu_reduce_mode();
  sonicboom::planner::decisions::seed_static_knobs();

  if (json) {
    std::cout << sonicboom::planner::decisions::dump_json();
    return sonicboom::planner::decisions::conflict_count() > 0 ? 2 : 0;
  }

  std::cout << "device: " << identity << "\n"
            << "tune cache: " << sonicboom::planner::decision_cache_root()
            << "/" << sonicboom::tune::device_dir() << "\n"
            << "jit cache:  "
            << sonicboom::mlir_tune::AutotuneCache::instance().path() << "\n";

  section("environment overrides");
  for (const Knob& knob : kKnobs) {
    const char* value = std::getenv(knob.var);
    std::cout << knob.var << " = "
              << (value && *value ? value : "(unset)")
              << "  [default: " << knob.fallback << "]  " << knob.meaning
              << "\n";
  }

  section("reduce mode");
  std::cout << "device: "
            << sonicboom::compiler::reduce_mode_name(device_mode)
            << "\ncpu:    "
            << sonicboom::compiler::reduce_mode_name(cpu_mode) << "\n";

  section("decision registry (this process)");
  std::cout << sonicboom::planner::decisions::dump_text();

  section("hand-written tuner choices on disk");
  const auto choices = sonicboom::tune::disk_choices();
  if (choices.empty()) {
    std::cout << "(none: every class uses its static default)\n";
  } else {
    for (const auto& choice : choices) {
      std::cout << choice.op << ' ' << choice.key << " =";
      for (const auto& [name, value] : choice.values)
        std::cout << ' ' << name << '=' << value;
      std::cout << "\n";
    }
  }

  section("JIT candidate tuner records on disk");
  const auto records = sonicboom::mlir_tune::disk_records();
  if (records.empty()) {
    std::cout << "(none: every class uses its static default)\n";
  } else {
    for (const auto& record : records) {
      std::cout << record.op << ' ' << record.shape_class << " = "
                << record.cfg.to_string() << "\n";
    }
  }

  section("static JIT defaults");
  const auto gemv_default = sonicboom::mlir_tune::default_config("gemv");
  const auto mmq_default = sonicboom::mlir_tune::mmq_tc_default_config();
  std::cout << "gemv = " << gemv_default.to_string() << "\n"
            << "matmul_q8_tc = " << mmq_default.to_string() << "\n"
            << "attn_prefill = dispatcher shape\n";

  if (sonicboom::planner::decisions::conflict_count() > 0) {
    std::cerr << "sb-explain: "
              << sonicboom::planner::decisions::conflict_count()
              << " conflicting decision record(s) were dropped\n";
    return 2;
  }
  return 0;
}
