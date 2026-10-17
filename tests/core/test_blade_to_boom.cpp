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

// Blade (sx::Document) -> Boom IR lowering acceptance (design/mlir-migration.md
// §6.4 / §7d).
//
// The direct lowering and the legacy route — `bladeToBoom(doc)` versus
// `graphToBoom(adapt_graph(doc))` — must produce the same IR, byte for byte:
// same tensor table, same ids, same free-variable block arguments, same node
// records. That is the gate the migration's step (d) needs before the
// planner::Graph intermediate can be removed, and it is checkable without
// running anything.
//
// Each case is also lowered back (`boomToGraph`) and compared to the graph
// adapt_graph produced through the model fingerprint, so a field the IR text
// might print identically-by-accident (an id order, a byte size) is still
// compared by the hash the planner keys plans on.
//
// The refusals are checked too: a document adapt_graph rejects must be refused
// by the lowering as well, not lowered into a different-but-valid graph.
//
// CPU-only: no device, no model, no MLIR passes.

#include <sonicboom/compiler/BladeToBoom.h>
#include <sonicboom/compiler/BoomConversion.h>
#include <sonicboom/compiler/BoomDialect.h>
#include <sonicboom/planner/graph.h>
#include <sonicboom/sx/parser.h>

#include <mlir/IR/MLIRContext.h>

#include <cstdio>
#include <string>
#include <vector>

namespace sbcompiler = sonicboom::compiler;
namespace sbplanner = sonicboom::planner;
namespace sxs = sonicboom::sx;

namespace {

int g_failures = 0;

void fail(const std::string& what) {
  std::fprintf(stderr, "  FAIL: %s\n", what.c_str());
  ++g_failures;
}

// z = relu(softmax(x, axis=1)); x,y,z : float32 [2,3].
const char* kSoftmaxRelu =
    "(sonicboom-s-expr (version 0 1) (graph (name \"softmax_relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"y\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// z = relu(x + b); a constant parameter and a two-input node.
const char* kAddRelu =
    "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 8))))"
    " (outputs (output \"z\"))"
    " (parameters (parameter \"b\" (tensor float32 (shape 8))"
    "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
    " (nodes"
    "   (node add (inputs \"x\" \"b\")"
    "     (outputs (\"y\" (tensor float32 (shape 8)))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 8))))))))";

// A weight from an external sidecar: the table entry must be marked external.
const char* kExternalWeight =
    "(sonicboom-s-expr (version 0 1) (graph (name \"ext\")"
    " (inputs (input \"x\" (tensor float32 (shape 1 3 8 8))))"
    " (outputs (output \"y\"))"
    " (parameters (parameter \"w\" (tensor float32 (shape 4 3 3 3))"
    "   (data :external \"w.bin\" 0 432)))"
    " (nodes"
    "   (node conv (inputs \"x\" \"w\")"
    "     (outputs (\"y\" (tensor float32 (shape 1 4 6 6))))"
    "     (attrs (group (int 1)) (strides (ints 1 1))"
    "            (pad (ints 0 0 0 0)) (alpha (float 1.5)) (flag (bool #t))"
    "            (auto_pad (string \"NOTSET\")))))))";

// Every attribute kind the dialect's node record carries, on one node.
const char* kAllAttrs =
    "(sonicboom-s-expr (version 0 1) (graph (name \"attrs\")"
    " (inputs (input \"x\" (tensor float32 (shape 1))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes (node conv (inputs \"x\")"
    "   (outputs (\"y\" (tensor float32 (shape 1))))"
    "   (attrs (group (int 1)) (strides (ints 2 2))"
    "          (auto_pad (string \"NOTSET\")) (alpha (float 1.5))"
    "          (flag (bool #t)) (flags (bools #t #f #t)))))))";

// A graph whose output is one of its inputs: no node defines it, and the
// input's table entry is the marked output.
const char* kInputAsOutput =
    "(sonicboom-s-expr (version 0 1) (graph (name \"passthrough\")"
    " (inputs (input \"x\" (tensor int32 (shape 4))))"
    " (outputs (output \"x\"))"
    " (parameters)"
    " (nodes)))";

// A pre-quantized activation: the first operand of quantized_matmul_q8 is the
// q8_1 layout, which has its own Boom type.
const char* kQ8Activation =
    "(sonicboom-s-expr (version 0 1) (graph (name \"q8\")"
    " (inputs (input \"x\" (tensor float32 (shape 64))))"
    " (outputs (output \"y\"))"
    " (parameters (parameter \"w\" (tensor int8 (shape 32 64))"
    "   (data :external \"w.bin\" 0 2048)))"
    " (nodes"
    "   (node quantize (inputs \"x\")"
    "     (outputs (\"q\" (tensor float32 (shape 64)))))"
    "   (node quantized_matmul_q8 (inputs \"q\" \"w\")"
    "     (outputs (\"y\" (tensor float32 (shape 32))))))))";

struct Case {
  const char* name;
  const char* text;
};

const Case kCases[] = {
    {"softmax_relu", kSoftmaxRelu},
    {"add_relu", kAddRelu},
    {"external_weight", kExternalWeight},
    {"attributes", kAllAttrs},
    {"input_as_output", kInputAsOutput},
    {"q8_activation", kQ8Activation},
};

// Both routes from one parsed document, compared.
void check_case(const Case& c, mlir::MLIRContext& context) {
  auto doc = sxs::parse_document(c.text);
  if (!doc) {
    fail(std::string(c.name) + ": document does not parse: " +
         doc.error().message);
    return;
  }

  auto graph = sbplanner::adapt_graph(*doc);
  if (!graph) {
    fail(std::string(c.name) + ": adapt_graph refused a valid document");
    return;
  }
  auto legacy = sbcompiler::graphToBoom(*graph, &context);
  if (!legacy) {
    fail(std::string(c.name) + ": graphToBoom refused the adapted graph");
    return;
  }

  auto direct = sbcompiler::bladeToBoom(*doc, &context);
  if (!direct) {
    fail(std::string(c.name) + ": bladeToBoom refused a valid document");
    return;
  }

  const std::string legacy_ir = sbcompiler::dumpBoom(legacy->getOperation());
  const std::string direct_ir = sbcompiler::dumpBoom(direct->getOperation());
  if (legacy_ir != direct_ir) {
    fail(std::string(c.name) + ": direct lowering IR differs from "
                               "graphToBoom(adapt_graph)");
    std::fprintf(stderr, "--- legacy ---\n%s\n--- direct ---\n%s\n",
                 legacy_ir.c_str(), direct_ir.c_str());
    return;
  }

  // The IR must also lower back to the graph adapt_graph built.
  auto back = sbcompiler::boomToGraph(*direct);
  if (!back) {
    fail(std::string(c.name) + ": bladeToBoom output does not lower back");
    return;
  }
  if (sbplanner::model_fingerprint(*back) !=
      sbplanner::model_fingerprint(*graph)) {
    fail(std::string(c.name) + ": round-tripped graph fingerprint differs");
    std::fprintf(stderr, "  legacy fp %016lx, round-trip fp %016lx\n",
                 (unsigned long)sbplanner::model_fingerprint(*graph),
                 (unsigned long)sbplanner::model_fingerprint(*back));
    for (const auto& t : graph->tensors)
      std::fprintf(stderr, "  legacy tensor %u %s shape=%zu dtype=%d size=%lu "
                           "in=%d out=%d const=%d ext=%d\n",
                   t.id.value, t.name.c_str(), t.shape.size(),
                   (int)t.dtype, (unsigned long)t.size_bytes,
                   t.is_graph_input, t.is_graph_output, t.is_constant,
                   t.is_external);
    for (const auto& n : graph->nodes) {
      std::fprintf(stderr, "  legacy node %u op=%d name=%s attrs=%zu\n",
                   n.id.value, (int)n.op, n.name.c_str(), n.attributes.size());
      for (const auto& a : n.attributes)
        std::fprintf(stderr, "    attr %s kind=%d\n", a.name.c_str(),
                     (int)a.kind());
    }
    for (const auto& n : back->nodes) {
      std::fprintf(stderr, "  back   node %u op=%d name=%s attrs=%zu\n",
                   n.id.value, (int)n.op, n.name.c_str(), n.attributes.size());
      for (const auto& a : n.attributes)
        std::fprintf(stderr, "    attr %s kind=%d\n", a.name.c_str(),
                     (int)a.kind());
    }
    for (const auto& t : back->tensors)
      std::fprintf(stderr, "  back   tensor %u %s shape=%zu dtype=%d size=%lu "
                           "in=%d out=%d const=%d ext=%d\n",
                   t.id.value, t.name.c_str(), t.shape.size(),
                   (int)t.dtype, (unsigned long)t.size_bytes,
                   t.is_graph_input, t.is_graph_output, t.is_constant,
                   t.is_external);
    return;
  }
}

// A document the lowering must refuse (adapt_graph refuses it too).
void check_refused(const char* name, const char* text,
                   mlir::MLIRContext& context) {
  auto doc = sxs::parse_document(text);
  if (!doc) {
    // The parser refused it first: that is still a refusal, not a lowering.
    return;
  }
  if (sbplanner::adapt_graph(*doc)) {
    fail(std::string(name) + ": adapt_graph accepted a document the test "
                             "expects to be invalid");
    return;
  }
  if (sbcompiler::bladeToBoom(*doc, &context)) {
    fail(std::string(name) + ": bladeToBoom accepted an invalid document");
  }
}

const char* kUnknownOp =
    "(sonicboom-s-expr (version 0 1) (graph (name \"bad_op\")"
    " (inputs (input \"x\" (tensor float32 (shape 2))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes (node not_an_op (inputs \"x\")"
    "   (outputs (\"y\" (tensor float32 (shape 2))))))))";

const char* kDuplicateDefinition =
    "(sonicboom-s-expr (version 0 1) (graph (name \"dup\")"
    " (inputs (input \"x\" (tensor float32 (shape 2))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes (node relu (inputs \"x\")"
    "   (outputs (\"x\" (tensor float32 (shape 2))))))))";

const char* kUndefinedInput =
    "(sonicboom-s-expr (version 0 1) (graph (name \"undef\")"
    " (inputs (input \"x\" (tensor float32 (shape 2))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes (node relu (inputs \"nope\")"
    "   (outputs (\"y\" (tensor float32 (shape 2))))))))";

const char* kUnknownOutput =
    "(sonicboom-s-expr (version 0 1) (graph (name \"out\")"
    " (inputs (input \"x\" (tensor float32 (shape 2))))"
    " (outputs (output \"y\"))"
    " (parameters)"
    " (nodes)))";

} // namespace

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<sbcompiler::boom::BoomDialect>();

  for (const Case& c : kCases)
    check_case(c, context);

  check_refused("unknown_operator", kUnknownOp, context);
  check_refused("duplicate_definition", kDuplicateDefinition, context);
  check_refused("undefined_input", kUndefinedInput, context);
  check_refused("unknown_output", kUnknownOutput, context);

  if (g_failures == 0)
    std::printf("blade to boom: %zu documents lowered identically\n",
                sizeof(kCases) / sizeof(kCases[0]));
  return g_failures == 0 ? 0 : 1;
}
