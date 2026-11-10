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

// Graph -> Boom IR -> graph round-trip smoke test on the six model/mode cases
// (docs/boom-ir.md).
//
// The conversion is a translation, not a transformation: the graph that comes
// back must be the graph that went in — the same tensor table and the same
// nodes (id, op, operands, results, attributes, name, backend) in the same
// order — because the model fingerprint hashes all of it and every execution
// plan is derived from it. This test builds the emitter's graph for each case,
// sends it through Boom IR and back, and requires the two graphs to be
// structurally equal.
//
// Compiler-side and CPU-only: the graph is built host-side, and nothing here
// runs a plan or touches a device. The byte-exact plan goldens and the pinned
// tune table live in the private tree's test data; this test compares a graph
// only against itself.
//
// Skips (exit 77) when no model GGUF is found.

#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomOps.h>
#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/graph.h>

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifndef SONICBOOM_SRC_DIR
#define SONICBOOM_SRC_DIR "."
#endif

namespace sbcompiler = sonicboom::compiler;
namespace boom = sonicboom::compiler::boom;
namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

namespace {

struct Case {
  const char* name;
  const char* env;   // SONICBOOM_*_GGUF
  const char* file;  // default file name under models/
  bool decode;
  uint64_t n;  // prefill segment length
};

const Case kCases[] = {
    {"gemma4-decode", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
     true, 0},
    {"gemma4-prefill", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf",
     false, 8},
    {"qwen2-decode", "SONICBOOM_QWEN_GGUF",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf", true, 0},
    {"qwen2-prefill", "SONICBOOM_QWEN_GGUF",
     "qwen2.5-0.5b-instruct-q4_k_m.gguf", false, 8},
    {"deepseek-r1-decode", "SONICBOOM_DEEPSEEK_GGUF",
     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", true, 0},
    {"deepseek-r1-prefill", "SONICBOOM_DEEPSEEK_GGUF",
     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf", false, 8},
};

int g_failures = 0;

void fail(const std::string& what) {
  std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  ++g_failures;
}

std::string find_model(const Case& c) {
  if (const char* env = std::getenv(c.env); env && *env) return env;
  const std::string path = std::string(SONICBOOM_SRC_DIR) + "/models/" + c.file;
  std::error_code ec;
  if (std::filesystem::exists(path, ec)) return path;
  return {};
}

std::string join(const std::vector<sbplanner::TensorId>& v) {
  std::string out;
  for (const auto& x : v) {
    if (!out.empty()) out += ",";
    out += std::to_string(int64_t(x.value));
  }
  return out;
}

// The first structural difference between two graphs, or "" when they are the
// same graph: name, tensor table, then every node's id, kind, name, backend,
// operands, results and attributes in order.
std::string diffGraphs(const sbplanner::Graph& a, const sbplanner::Graph& b) {
  if (a.name != b.name) return "graph name: '" + a.name + "' vs '" + b.name + "'";

  if (a.tensors.size() != b.tensors.size())
    return "tensor count: " + std::to_string(a.tensors.size()) + " vs " +
           std::to_string(b.tensors.size());
  for (std::size_t i = 0; i < a.tensors.size(); ++i) {
    const auto& x = a.tensors[i];
    const auto& y = b.tensors[i];
    const std::string at = "tensor[" + std::to_string(i) + "] (id " +
                           std::to_string(x.id.value) + ")";
    if (int64_t(x.id.value) != int64_t(y.id.value))
      return at + ": id " + std::to_string(x.id.value) + " vs " +
             std::to_string(y.id.value);
    if (x.shape != y.shape) return at + ": shape differs";
    if (x.dtype != y.dtype) return at + ": dtype differs";
    if (x.size_bytes != y.size_bytes) return at + ": size_bytes differs";
    if (x.is_graph_input != y.is_graph_input ||
        x.is_graph_output != y.is_graph_output ||
        x.is_constant != y.is_constant || x.is_external != y.is_external)
      return at + ": a flag differs";
    if (x.name != y.name)
      return at + ": name '" + x.name + "' vs '" + y.name + "'";
  }

  if (a.nodes.size() != b.nodes.size())
    return "node count: " + std::to_string(a.nodes.size()) + " vs " +
           std::to_string(b.nodes.size());
  for (std::size_t i = 0; i < a.nodes.size(); ++i) {
    const auto& x = a.nodes[i];
    const auto& y = b.nodes[i];
    const std::string at = "node[" + std::to_string(i) + "] ('" + x.name + "')";
    if (int64_t(x.id.value) != int64_t(y.id.value)) return at + ": id differs";
    if (x.op != y.op) return at + ": op differs";
    if (x.name != y.name) return at + ": name differs";
    if (x.backend != y.backend) return at + ": backend differs";
    if (x.inputs.size() != y.inputs.size() ||
        !std::equal(x.inputs.begin(), x.inputs.end(), y.inputs.begin(),
                    [](sbplanner::TensorId p, sbplanner::TensorId q) {
                      return p.value == q.value;
                    }))
      return at + ": operands " + join(x.inputs) + " vs " + join(y.inputs);
    if (x.outputs.size() != y.outputs.size() ||
        !std::equal(x.outputs.begin(), x.outputs.end(), y.outputs.begin(),
                    [](sbplanner::TensorId p, sbplanner::TensorId q) {
                      return p.value == q.value;
                    }))
      return at + ": results " + join(x.outputs) + " vs " + join(y.outputs);
    if (x.attributes.size() != y.attributes.size())
      return at + ": attribute count differs";
    for (std::size_t k = 0; k < x.attributes.size(); ++k) {
      const auto& p = x.attributes[k];
      const auto& q = y.attributes[k];
      if (p.name != q.name)
        return at + ": attribute[" + std::to_string(k) + "] name '" + p.name +
               "' vs '" + q.name + "'";
      if (p.value.index() != q.value.index())
        return at + ": attribute '" + p.name + "' changed kind";
      if (p.value != q.value)
        return at + ": attribute '" + p.name + "' changed value";
    }
  }

  if (a.inputs.size() != b.inputs.size() ||
      a.outputs.size() != b.outputs.size())
    return "graph input/output count differs";
  for (std::size_t i = 0; i < a.inputs.size(); ++i)
    if (a.inputs[i].value != b.inputs[i].value) return "graph inputs differ";
  for (std::size_t i = 0; i < a.outputs.size(); ++i)
    if (a.outputs[i].value != b.outputs[i].value) return "graph outputs differ";
  return {};
}

}  // namespace

int main() {
  int ran = 0;
  for (const Case& c : kCases) {
    const std::string path = find_model(c);
    if (path.empty()) continue;

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
    std::printf("%s (%s)\n", c.name, c.decode ? "decode" : "prefill");

    const sbplanner::Graph pre =
        c.decode ? sbmodel::build_gemma4_graph(*model, /*greedy=*/true)
                 : sbmodel::build_gemma4_prefill_graph(*model, c.n);

    // MLIR diagnostics are collected rather than printed: a conversion that
    // fails is a test failure here, and the message belongs to it.
    mlir::MLIRContext context;
    context.getOrLoadDialect<boom::BoomDialect>();
    std::string diagnostics;
    mlir::ScopedDiagnosticHandler handler(&context, [&](mlir::Diagnostic& d) {
      if (d.getSeverity() != mlir::DiagnosticSeverity::Error) return;
      if (diagnostics.empty()) {
        std::string text;
        llvm::raw_string_ostream os(text);
        d.print(os);
        diagnostics = os.str();
      }
    });

    auto module = sbcompiler::graphToBoom(pre, &context);
    if (!module) {
      fail(std::string(c.name) + ": graph -> Boom IR failed" +
           (diagnostics.empty() ? "" : ": " + diagnostics));
      continue;
    }
    auto back = sbcompiler::boomToGraph(*module);
    if (!back) {
      fail(std::string(c.name) + ": Boom IR -> graph failed" +
           (diagnostics.empty() ? "" : ": " + diagnostics));
      continue;
    }

    const std::string d = diffGraphs(pre, *back);
    if (!d.empty()) {
      fail(std::string(c.name) + ": round trip: " + d);
      continue;
    }
    if (sbplanner::model_fingerprint(pre) !=
        sbplanner::model_fingerprint(*back)) {
      fail(std::string(c.name) + ": round trip: model fingerprint");
      continue;
    }
    std::printf("  %zu nodes, %zu tensors, round trip identical\n",
                pre.nodes.size(), pre.tensors.size());
  }

  if (ran == 0 && g_failures == 0) {
    std::printf("test_boom_conversion_smoke SKIP (no model GGUFs found)\n");
    return 77;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "test_boom_conversion_smoke FAILED: %d check(s)\n",
                 g_failures);
    return 1;
  }
  std::printf("test_boom_conversion_smoke OK (%d case(s))\n", ran);
  return 0;
}
