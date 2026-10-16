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

// Graph <-> Boom IR round trip acceptance (design/mlir-migration.md §7a).
//
// The conversion is a translation, not a transformation: a graph that goes
// through Boom IR and back must be the graph that went in — every tensor field,
// every node id, every attribute, in order — because the model fingerprint
// hashes all of it in that order and every execution plan is derived from it.
// Two properties per model/mode case:
//
//   1. Structural — the emitter's graph, and the graph the planner actually
//      plans, round-trip to a graph with the same name, the same tensor table,
//      the same nodes (id, op, operands, results, attributes, name, backend) in
//      the same order, and therefore the same model_fingerprint.
//   2. Plan — the round-tripped graph, fed back into the record path through the
//      GraphTransform seam (model/plan.h), yields a plan whose dump is
//      byte-identical to the committed golden under tests/data/plans/. The
//      planner's fusion passes already ran when the transform is applied, so the
//      round trip is on the critical path of the plan that follows: an attribute
//      reordered, an id renumbered or a type mistyped shows up here as a plan
//      diff, not as a lost kernel.
//
// The IR is dumped as text (MLIR's generic form) after every case, so a failure
// can be looked at. `--dump DIR` chooses where; the default is under the system
// temp directory.
//
// A third, model-free property is the one refusal the conversion makes: a
// vocabulary attribute (the names ODS declares) whose value is not an integer
// or a float has no typed form, and is refused with a diagnostic rather than
// dropped. That check is hand-built and runs before the gate, so a CPU-only
// build still covers it.
//
// As in test_plan_record, the golden comparison is skipped — never the round
// trip — when the recording device differs from the golden's, because part of
// the plan is device-dependent. Skips (exit 77) when there is no CUDA device
// or no model GGUFs.

#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomOps.h>
#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/planner/plan_record.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace sbcompiler = sonicboom::compiler;
namespace boom = sonicboom::compiler::boom;
namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;
namespace sxs = sonicboom::sx;

namespace {

int g_failures = 0;

void fail(const std::string& what) {
  std::cerr << "  FAIL: " << what << "\n";
  ++g_failures;
}

struct Case {
  const char* name;
  const char* env;   // SONICBOOM_*_GGUF
  const char* file;  // default file name under models/
  const char* mode;  // "decode" | "prefill"
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

std::string src_path(const std::string& rel) {
#ifdef SONICBOOM_SRC_DIR
  return std::string(SONICBOOM_SRC_DIR) + "/" + rel;
#else
  return rel;
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

std::string device_line(const std::string& text) {
  for (const std::string& l : split_lines(text))
    if (l.rfind("device ", 0) == 0) return l.substr(7);
  return {};
}

bool write_file(const std::string& path, const std::string& text) {
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path());
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << text;
  return f.good();
}

//--- the conversion, with the IR kept for inspection -------------------------

struct Trip {
  bool ok = false;
  std::string error;   // why not, when !ok
  std::string ir;      // the IR's text form
  sbplanner::Graph graph;  // the graph that came back
};

// Graph -> Boom IR -> Graph. The dialect is loaded first: an op cannot be
// created by name until its dialect is.
Trip roundTrip(const sbplanner::Graph& in, mlir::MLIRContext& context) {
  Trip out;
  context.getOrLoadDialect<boom::BoomDialect>();

  auto module = sbcompiler::graphToBoom(in, &context);
  if (!module) {
    out.error = "graph -> Boom IR failed";
    return out;
  }
  out.ir = sbcompiler::dumpBoom(module->getOperation());

  auto back = sbcompiler::boomToGraph(*module);
  if (!back) {
    out.error = "Boom IR -> graph failed";
    return out;
  }
  out.graph = std::move(*back);
  out.ok = true;
  return out;
}

//--- structural comparison ---------------------------------------------------

// The record path's plan as text; "" when it produced none.
std::string dumpBoomPlan(std::optional<sbplanner::PlanRunRecord> run) {
  if (!run) return {};
  return sbplanner::serialize_plan_run(*run);
}

template <typename T>
std::string join(const std::vector<T>& v) {
  std::string out;
  for (const T& x : v) {
    if (!out.empty()) out += ",";
    out += std::to_string(int64_t(x.value));
  }
  return out;
}

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
    if (x.name != y.name) return at + ": name '" + x.name + "' vs '" + y.name + "'";
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

  if (a.inputs.size() != b.inputs.size() || a.outputs.size() != b.outputs.size())
    return "graph input/output count differs";
  for (std::size_t i = 0; i < a.inputs.size(); ++i)
    if (a.inputs[i].value != b.inputs[i].value) return "graph inputs differ";
  for (std::size_t i = 0; i < a.outputs.size(); ++i)
    if (a.outputs[i].value != b.outputs[i].value) return "graph outputs differ";
  return {};
}

//--- the typed-attribute vocabulary's one refusal -----------------------------

// `node_attrs` carries every attribute a node holds, in order and whatever its
// kind; the *typed* view of them is the vocabulary ODS declares
// (BoomVerify.cpp, boomTypedAttrNames), each member of which is an integer or a
// float, because that is what the plan emitters and the fusion passes read. An
// attribute in that vocabulary carrying anything else — a string, a bool, any
// vector — has no typed form, and setting none would leave the op's typed
// attribute disagreeing with the node_attrs entry the verifier compares it
// against. So the conversion refuses it, naming the node and the attribute,
// instead of dropping a value the runtime may read or emitting IR that fails
// verification under a less specific message.
//
// No emitter writes such an attribute, so no model graph exercises this. It is
// built by hand: one node, one input, one output, and one vocabulary attribute
// whose *kind* is the only variable — the same graph with the same attribute
// name and an integer value must convert. Nothing about it needs a model or a
// device, so it runs before the gate in main.
void checkAttrKindRefusal() {
  // The graph, with `slot`'s value as the parameter. `slot` is a typed
  // vocabulary name declared I64Attr (BoomOps.td).
  auto graphWith = [](sxs::Attribute slot) {
    sbplanner::Graph g;
    g.name = "attr_kind";
    g.tensors.resize(2);
    g.tensors[0].id = sbplanner::TensorId{0};
    g.tensors[0].shape = {4};
    g.tensors[0].dtype = sxs::DType::Float32;
    g.tensors[0].size_bytes = 16;
    g.tensors[0].is_graph_input = true;
    g.tensors[0].name = "x";
    g.tensors[1].id = sbplanner::TensorId{1};
    g.tensors[1].shape = {4};
    g.tensors[1].dtype = sxs::DType::Float32;
    g.tensors[1].size_bytes = 16;
    g.tensors[1].is_graph_output = true;
    g.tensors[1].name = "y";

    sbplanner::GraphNodeDesc n;
    n.id = sbplanner::GraphNodeId{0};
    n.op = sbplanner::OpKind::Relu;
    n.inputs = {sbplanner::TensorId{0}};
    n.outputs = {sbplanner::TensorId{1}};
    n.attributes = {std::move(slot)};
    n.name = "relu0";
    g.nodes = {std::move(n)};
    g.inputs = {sbplanner::TensorId{0}};
    g.outputs = {sbplanner::TensorId{1}};
    return g;
  };

  mlir::MLIRContext context;
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

  // The control: the same graph and attribute name, an integer value.
  Trip control = roundTrip(graphWith(sxs::Attribute{"slot", int64_t(0)}), context);
  if (!control.ok)
    fail(std::string("attr-kind control (Int 'slot'): ") + control.error);
  else if (const std::string d = diffGraphs(
               graphWith(sxs::Attribute{"slot", int64_t(0)}), control.graph);
           !d.empty())
    fail("attr-kind control (Int 'slot'): round trip differs: " + d);
  else
    std::cout << "attr-kind control: Int 'slot' round-trips\n";

  // The refusal: a vocabulary attribute whose value is not an integer or a
  // float. The module must be empty and the diagnostic must name the node and
  // the attribute — a refusal that says only "conversion failed" would leave a
  // caller no way to find the attribute.
  diagnostics.clear();
  auto module = sbcompiler::graphToBoom(
      graphWith(sxs::Attribute{"slot", std::string("0")}), &context);
  if (module) {
    fail("attr-kind refusal: a String-valued 'slot' converted instead of "
         "being refused");
  } else if (diagnostics.empty()) {
    fail("attr-kind refusal: no diagnostic was emitted");
  } else if (diagnostics.find("'slot'") == std::string::npos ||
             diagnostics.find("relu0") == std::string::npos) {
    fail("attr-kind refusal: the diagnostic names neither the node nor the "
         "attribute: " + diagnostics);
  } else {
    std::cout << "attr-kind refusal: String-valued 'slot' refused — "
              << diagnostics << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path dump_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--dump" && i + 1 < argc) {
      dump_dir = argv[++i];
    } else if (a == "-h" || a == "--help") {
      std::printf("usage: %s [--dump DIR]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown option %s\n", argv[0], argv[i]);
      return 2;
    }
  }
  if (dump_dir.empty())
    dump_dir = std::filesystem::temp_directory_path() / "sonicboom-boom-ir";

  // The dialect-level check first: it needs no model and no device, so a
  // CPU-only run still covers it. Its failures are reported before the gate
  // below, so a skip never hides one.
  checkAttrKindRefusal();
  if (g_failures != 0) {
    std::cerr << "boom-conversion: " << g_failures
              << " dialect-level check(s) failed\n";
    return 1;
  }

  if (!sonicboom::quant::cuda_available()) {
    std::cout << "(skipped: no CUDA device — the plan gate needs one)\n";
    return 77;
  }

  int ran = 0;
  for (const Case& c : kCases) {
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

    // MLIR diagnostics are collected rather than printed: a conversion that
    // fails is a test failure here, and the message belongs to it.
    mlir::MLIRContext context;
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

    const bool decode = std::string_view(c.mode) == "decode";

    // 1. The emitter's graph, round-tripped. This is the widest op coverage
    //    available: the graph before the passes still has every node the emitter
    //    wrote, including the ones the passes later fuse away.
    const sbplanner::Graph pre = decode ? sbmodel::build_gemma4_graph(*model, true)
                                        : sbmodel::build_gemma4_prefill_graph(*model, c.n);
    Trip pre_trip = roundTrip(pre, context);
    if (!pre_trip.ok) {
      fail(std::string(c.name) + ": emitter graph: " + pre_trip.error +
           (diagnostics.empty() ? "" : ": " + diagnostics));
    } else {
      const std::string d = diffGraphs(pre, pre_trip.graph);
      if (!d.empty())
        fail(std::string(c.name) + ": emitter graph round trip: " + d);
      else if (sbplanner::model_fingerprint(pre) !=
               sbplanner::model_fingerprint(pre_trip.graph))
        fail(std::string(c.name) + ": emitter graph round trip: fingerprint");
      else
        std::cout << "  emitter graph: " << pre.nodes.size() << " nodes, "
                  << pre.tensors.size() << " tensors, round trip identical\n";
    }
    write_file((dump_dir / (std::string(c.name) + ".boom.mlir")).string(),
               pre_trip.ir);

    // 2. The plan the round-tripped graph yields. The transform runs inside the
    //    record path, after the passes, so `g` is replaced by the graph that
    //    went out and came back.
    std::string transform_error;
    const sbplanner::Graph* planned_in = nullptr;
    auto transform = [&](sbplanner::Graph& g) {
      planned_in = &g;
      Trip trip = roundTrip(g, context);
      if (!trip.ok) {
        transform_error = trip.error;
        return;
      }
      const std::string d = diffGraphs(g, trip.graph);
      if (!d.empty()) {
        transform_error = "round trip differs: " + d;
        return;
      }
      g = std::move(trip.graph);
      write_file((dump_dir / (std::string(c.name) + ".planned.boom.mlir")).string(),
                 trip.ir);
    };

    const std::string dump =
        decode ? dumpBoomPlan(sbmodel::record_decode_plan(*model, c.n_ctx, c.steps,
                                                          transform))
               : dumpBoomPlan(sbmodel::record_prefill_plan(*model, c.n, transform));
    if (!transform_error.empty()) {
      fail(std::string(c.name) + ": planned graph: " + transform_error +
           (diagnostics.empty() ? "" : ": " + diagnostics));
      continue;
    }
    if (planned_in == nullptr) {
      fail(std::string(c.name) + ": the transform was never applied (no plan)");
      continue;
    }
    if (dump.empty()) {
      fail(std::string(c.name) + ": recording the round-tripped plan produced no plan");
      continue;
    }

    // The golden gate. Same device-line rule as test_plan_record: the record
    // carries the device it was taken on, and part of the plan (the
    // decode-attention split-KV grid) is sized from the SM count.
    const std::string gpath = src_path("tests/data/plans/" + std::string(c.name) + ".plan");
    const std::string golden = read_file(gpath);
    if (golden.empty()) {
      fail(std::string(c.name) + ": golden missing or empty: " + gpath);
      continue;
    }
    const std::string dump_plan = (dump_dir / (std::string(c.name) + ".plan")).string();
    write_file(dump_plan, dump);

    if (device_line(golden) != device_line(dump)) {
      std::cout << "  plan: golden skipped (recorded on '" << device_line(golden)
                << "', running on '" << device_line(dump) << "')\n";
      continue;
    }
    if (golden != dump) {
      fail(std::string(c.name) + ": the round-tripped plan differs from " + gpath);
      continue;
    }

    // The plan is the golden, byte for byte. tools/plan_diff is the tool for
    // looking at a failure, so run it too — a disagreement between the test's
    // comparison and the tool's would be a test bug, not a plan bug.
    bool plan_diff_ran = false;
    const std::string plan_diff = src_path("tools/plan_diff");
    if (std::filesystem::exists(plan_diff)) {
      plan_diff_ran = true;
      const std::string cmd =
          plan_diff + " '" + gpath + "' '" + dump_plan + "' > /dev/null";
      if (std::system(cmd.c_str()) != 0) {
        fail(std::string(c.name) + ": tools/plan_diff disagrees (non-zero exit) for " +
             dump_plan);
        continue;
      }
    }
    std::cout << "  plan: matches golden"
              << (plan_diff_ran ? " (tools/plan_diff exit 0)" : "") << "\n";
  }

  std::cout << "IR dumped under " << dump_dir << "\n";

  if (ran == 0 && g_failures == 0) {
    std::cout << "(skipped: no model GGUFs found)\n";
    return 77;
  }
  if (g_failures != 0) {
    std::cerr << g_failures << " Boom conversion check(s) failed\n";
    return 1;
  }
  std::cout << "boom-conversion: " << ran << " case(s) OK\n";
  return 0;
}
