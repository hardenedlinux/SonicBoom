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

#include <sonicboom/numerics.h>

#include "analysis_driver.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

struct Knob {
  const char* var;
  const char* fallback;  // what the runtime does when the variable is unset
  const char* meaning;
};

// The env knobs that select an algorithm, tile, mode or threshold (the
// diagnostic-only switches are deliberately absent).
constexpr Knob kKnobs[] = {
    {"SONICBOOM_NUMERICS", "standard",
     "numerics profile: strict|standard (latched at first inference)"},
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

// The reduction rows, aggregated over the nodes that share an (op, K) pair:
// the worst bound per column, and whether every node is provable under each
// accumulation order. A row is a reduction kernel class; its nodes differ only
// in the values the calibration measured.
struct ReductionRow {
  std::size_t nodes = 0;
  double max_sum_abs = 0.0;
  double max_err_seq = 0.0;
  double max_err_tree = 0.0;
  bool seq_all = true;
  bool tree_all = true;
};

// Run the F2 analysis on MODEL with FACTS and print the facts it consumed and
// the reduction-safety table it produced. Analysis only: the plan is unchanged.
bool print_analysis(const std::string& model_path, const std::string& facts_path) {
  const std::string out =
      "/tmp/sb-explain-analysis-" + std::to_string(::getpid()) + ".txt";
  std::string error;
  if (!sb_tools::run_range_analysis(model_path, facts_path, out, &error)) {
    std::cerr << "sb-explain: " << error << "\n";
    std::remove(out.c_str());
    return false;
  }

  std::ifstream in(out);
  std::map<std::pair<std::string, long long>, ReductionRow> reductions;
  std::vector<std::string> units, notes, liveness;
  double epsilon = 0.0;
  bool have_epsilon = false;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "reduction") {
      std::string unit, node, op;
      long long k = 0;
      double sum_abs = 0.0, err_seq = 0.0, err_tree = 0.0;
      int within_seq = 0, within_tree = 0;
      ss >> unit >> node >> op >> k >> sum_abs >> err_seq >> err_tree >>
          within_seq >> within_tree;
      if (!ss) continue;
      ReductionRow& row = reductions[{op, k}];
      ++row.nodes;
      row.max_sum_abs = std::max(row.max_sum_abs, sum_abs);
      row.max_err_seq = std::max(row.max_err_seq, err_seq);
      row.max_err_tree = std::max(row.max_err_tree, err_tree);
      row.seq_all = row.seq_all && within_seq != 0;
      row.tree_all = row.tree_all && within_tree != 0;
    } else if (kind == "unit") {
      units.push_back(line);
      if (!have_epsilon) {
        const auto at = line.find("epsilon=");
        if (at != std::string::npos) {
          epsilon = std::atof(line.c_str() + at + 8);
          have_epsilon = true;
        }
      }
    } else if (kind == "note") {
      notes.push_back(line.substr(kind.size() + 1));
    } else if (kind == "liveness") {
      liveness.push_back(line);
    }
  }
  in.close();
  std::remove(out.c_str());

  // The facts the analysis consumed: one bound per (slot, layer) weight, one
  // interval per named activation.
  std::size_t weight_facts = 0, tensor_facts = 0;
  {
    std::ifstream facts(facts_path);
    std::string f;
    while (std::getline(facts, f))
      if (f.rfind("weight ", 0) == 0)
        ++weight_facts;
      else if (f.rfind("tensor ", 0) == 0)
        ++tensor_facts;
  }

  std::cout << "facts: " << weight_facts << " weight bounds, " << tensor_facts
            << " tensor intervals (" << facts_path << ")\n";
  for (const auto& u : units) std::cout << u << "\n";
  for (const auto& n : notes) std::cout << "  " << n << "\n";
  std::cout << "reduction safety (eps = " << epsilon
            << " relative to sum|ab|; seq = sequential f32, tree = blocked f32):\n";
  std::printf("  %-26s %6s %5s %12s %12s %12s %-4s %-4s\n", "op", "K", "nodes",
              "max sum|ab|", "seq bound", "tree bound", "seq", "tree");
  for (const auto& [key, row] : reductions)
    std::printf("  %-26s %6lld %5zu %12.6g %12.6g %12.6g %-4s %-4s\n",
                key.first.c_str(), key.second, row.nodes, row.max_sum_abs,
                row.max_err_seq, row.max_err_tree, row.seq_all ? "yes" : "NO",
                row.tree_all ? "yes" : "NO");
  if (liveness.empty()) {
    std::cout << "liveness: no disagreement between the in-place marks and "
                 "dense backward liveness\n";
  } else {
    std::cout << "liveness (" << liveness.size() << " row(s)):\n";
    for (const auto& l : liveness) std::cout << "  " << l << "\n";
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  bool json = false;
  std::string analysis_model, analysis_facts;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "-h" || a == "--help") {
      std::cout << "usage: sb-explain [--json] [--analysis MODEL FACTS]\n"
                   "Print the automatic choices for this device: the env\n"
                   "overrides, the per-device caches, and the decision\n"
                   "registry. --json prints the registry as JSON. --analysis\n"
                   "adds the Boom range/error facts for MODEL with FACTS and\n"
                   "the per-reduction accumulation-safety table (analysis\n"
                   "only: the plan is unchanged). Nothing is tuned; the only\n"
                   "write is the registry itself.\n";
      return 0;
    }
    if (a == "--json") {
      json = true;
      continue;
    }
    if (a == "--analysis") {
      if (i + 2 >= argc) {
        std::cerr << "sb-explain: --analysis needs MODEL and FACTS\n";
        return 2;
      }
      analysis_model = argv[++i];
      analysis_facts = argv[++i];
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
  // Resolving the profile records numerics.profile (and its env/application
  // origin) in the registry, so both output forms name it.
  const auto numerics = sonicboom::numerics_profile();
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

  section("numerics profile");
  std::cout << sonicboom::numerics_profile_name(numerics) << "\n"
            << (numerics == sonicboom::NumericsProfile::Strict
                    ? "  f32 KV; no TF32; f64 reductions; fp64-gated dense\n"
                    : "  f16 KV; TF32 dense where the standard gate accepts; "
                      "standard acceptance gate (ppl, top-1)\n");

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

  if (!analysis_model.empty()) {
    section("range/error analysis (facts for the reduction kernels)");
    if (!print_analysis(analysis_model, analysis_facts)) return 1;
  }

  if (sonicboom::planner::decisions::conflict_count() > 0) {
    std::cerr << "sb-explain: "
              << sonicboom::planner::decisions::conflict_count()
              << " conflicting decision record(s) were dropped\n";
    return 2;
  }
  return 0;
}
