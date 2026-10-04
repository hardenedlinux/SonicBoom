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

// test_s_expr.cpp — SonicBoom S-Expr v0.1 parser unit tests.
//
// Covers the 16 focused parsing/validation cases from the S-Expr parser task.
// Each case is a small self-contained document string; the parser must accept
// valid documents and reject invalid ones with a structured Error.
//
// NOTE: this file deliberately does NOT use <cassert> — the core build is a
// Release build (NDEBUG), which would disable assertions. Explicit checks plus
// a non-zero exit code keep the test meaningful in Release.

#include <sonicboom/sx/parser.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace sx = sonicboom::sx;

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

// Parse `src`; require success.
bool must_parse(const std::string& src, sx::Document& out, const char* what) {
  auto r = sx::parse_document(src);
  if (!r) {
    std::cerr << "  FAIL: " << what << " (unexpected error: "
              << r.error().message << ")\n";
    ++g_failures;
    return false;
  }
  out = std::move(*r);
  return true;
}

// Parse `src`; require failure with the given category.
void must_reject(const std::string& src, const char* what, sx::ErrorCategory cat) {
  auto r = sx::parse_document(src);
  if (r) {
    std::cerr << "  FAIL: " << what << " (expected rejection, but parsed OK)\n";
    ++g_failures;
    return;
  }
  if (r.error().category != cat) {
    std::cerr << "  FAIL: " << what << " (wrong category; got: "
              << r.error().message << ")\n";
    ++g_failures;
  }
}

// 1. minimal valid document
void test_minimal_document() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\") (inputs) (outputs) (parameters) (nodes)))",
          doc, "minimal document")) {
    return;
  }
  check(doc.version_major == 0 && doc.version_minor == 1, "minimal: version");
  check(doc.graph.name == "g", "minimal: graph name");
  check(doc.graph.inputs.empty() && doc.graph.outputs.empty() &&
            doc.graph.parameters.empty() && doc.graph.nodes.empty(),
        "minimal: empty sections");
}

// 2. valid tensor/type parsing
void test_tensor_type() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1 3 224 224))))"
          " (outputs) (parameters) (nodes)))",
          doc, "tensor type")) {
    return;
  }
  check(doc.graph.inputs.size() == 1, "type: one input");
  const auto& t = doc.graph.inputs[0].type;
  check(t.dtype == sx::DType::Float32, "type: dtype float32");
  check((t.shape.dims == std::vector<int64_t>{1, 3, 224, 224}), "type: shape");
  check(std::string(sx::dtype_name(t.dtype)) == "float32", "type: dtype_name");
}

// 3. valid parameter (inline :values)
void test_parameter_values() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs) (outputs)"
          " (parameters (parameter \"c\" (tensor int64 (shape 2)) (data :values -1 -2)))"
          " (nodes)))",
          doc, "values parameter")) {
    return;
  }
  check(doc.graph.parameters.size() == 1, "values-param: one");
  const auto& p = doc.graph.parameters[0];
  check(p.name == "c", "values-param: name");
  check(p.type.dtype == sx::DType::Int64, "values-param: dtype int64");
  check(std::holds_alternative<sx::ValuesData>(p.data), "values-param: values data");
  const auto& v = std::get<sx::ValuesData>(p.data);
  check(v.values.size() == 2, "values-param: two values");
  check(std::get<int64_t>(v.values[0]) == -1 &&
            std::get<int64_t>(v.values[1]) == -2,
        "values-param: values are -1 -2");
}

// 4. valid parameter (external sidecar)
void test_parameter_external() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs) (outputs)"
          " (parameters (parameter \"w\" (tensor float32 (shape 64 3 7 7))"
          "   (data :external \"w.bin\" 0 37632)))"
          " (nodes)))",
          doc, "external parameter")) {
    return;
  }
  const auto& p = doc.graph.parameters[0];
  check(std::holds_alternative<sx::ExternalData>(p.data), "ext-param: external data");
  const auto& e = std::get<sx::ExternalData>(p.data);
  check(e.file == "w.bin" && e.offset == 0 && e.length == 37632,
        "ext-param: file/offset/length");
}

// 5. valid node
void test_node() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1)))) (outputs)"
          " (parameters)"
          " (nodes (node relu (inputs \"x\") (outputs (\"y\" (tensor float32 (shape 1))))))))",
          doc, "node")) {
    return;
  }
  check(doc.graph.nodes.size() == 1, "node: one");
  const auto& n = doc.graph.nodes[0];
  check(n.op_name == "relu", "node: op name");
  check(n.inputs.size() == 1 && n.inputs[0] == "x", "node: inputs");
  check(n.outputs.size() == 1 && n.outputs[0].name == "y", "node: outputs");
  check(n.domain == "default", "node: default domain");
  check(!n.version.has_value(), "node: version defaults to unset");
}

// 6. valid attributes
void test_attributes() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1)))) (outputs)"
          " (parameters)"
          " (nodes (node conv (inputs \"x\") (outputs (\"y\" (tensor float32 (shape 1))))"
          "   (attrs (group (int 1)) (strides (ints 2 2)) (auto_pad (string \"NOTSET\"))"
          "          (alpha (float 1.5)) (flag (bool #t)))))))",
          doc, "attributes")) {
    return;
  }
  const auto& attrs = doc.graph.nodes[0].attributes;
  check(attrs.size() == 5, "attrs: count");
  check(attrs[0].name == "group" && std::get<int64_t>(attrs[0].value) == 1,
        "attrs: int");
  check(attrs[1].name == "strides" &&
            std::get<std::vector<int64_t>>(attrs[1].value) ==
                std::vector<int64_t>({2, 2}),
        "attrs: ints");
  check(attrs[2].name == "auto_pad" &&
            std::get<std::string>(attrs[2].value) == "NOTSET",
        "attrs: string");
  check(attrs[3].name == "alpha" && std::get<double>(attrs[3].value) == 1.5,
        "attrs: float");
  check(attrs[4].name == "flag" && std::get<bool>(attrs[4].value) == true,
        "attrs: bool");
}

// 7. graph input/output
void test_graph_io() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1))))"
          " (outputs (output \"x\"))"
          " (parameters) (nodes)))",
          doc, "graph io")) {
    return;
  }
  check(doc.graph.outputs.size() == 1 && doc.graph.outputs[0] == "x",
        "graph: output references input value");
}

// 8. multiple nodes in topological order
void test_topological_order() {
  sx::Document doc;
  if (!must_parse(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1)))) (outputs (output \"z\"))"
          " (parameters)"
          " (nodes"
          "   (node relu (inputs \"x\") (outputs (\"y\" (tensor float32 (shape 1)))))"
          "   (node add (inputs \"y\" \"x\") (outputs (\"z\" (tensor float32 (shape 1))))))))",
          doc, "topological order")) {
    return;
  }
  check(doc.graph.nodes.size() == 2, "topo: two nodes");
}

// 9. duplicate SSA definition → reject
void test_duplicate_definition() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float32 (shape 1))) (input \"x\" (tensor float32 (shape 1))))"
      " (outputs) (parameters) (nodes)))",
      "duplicate input name", sx::ErrorCategory::Semantic);
}

// 10. unknown SSA reference → reject
void test_unknown_reference() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs) (outputs) (parameters)"
      " (nodes (node relu (inputs \"nope\") (outputs (\"y\" (tensor float32 (shape 1))))))))",
      "unknown reference", sx::ErrorCategory::Semantic);
}

// 11. invalid dtype → reject
void test_invalid_dtype() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float8 (shape 1)))) (outputs) (parameters) (nodes)))",
      "invalid dtype", sx::ErrorCategory::Semantic);
}

// 12. invalid shape → reject
void test_invalid_shape() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float32 (shape 1 -1)))) (outputs) (parameters) (nodes)))",
      "negative dimension", sx::ErrorCategory::Semantic);
}

// 13. invalid node ordering → reject
void test_invalid_ordering() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float32 (shape 1)))) (outputs)"
      " (parameters)"
      " (nodes"
      "   (node add (inputs \"y\" \"x\") (outputs (\"z\" (tensor float32 (shape 1)))))"
      "   (node relu (inputs \"x\") (outputs (\"y\" (tensor float32 (shape 1))))))))",
      "consumer before producer", sx::ErrorCategory::Semantic);
}

// 14. unsupported version → reject
void test_unsupported_version() {
  must_reject(
      "(sonicboom-s-expr (version 1 0) (graph (name \"g\") (inputs) (outputs) (parameters) (nodes)))",
      "major version 1", sx::ErrorCategory::Unsupported);
  must_reject(
      "(sonicboom-s-expr (version 0 2) (graph (name \"g\") (inputs) (outputs) (parameters) (nodes)))",
      "minor version 2", sx::ErrorCategory::Unsupported);
}

// 15. malformed syntax → reject
void test_malformed_syntax() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\") (inputs) (outputs) (parameters) (nodes)",
      "unbalanced parenthesis", sx::ErrorCategory::Syntax);
}

// 16. unknown/unsupported form → reject
void test_unknown_form() {
  must_reject(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs) (outputs) (parameters) (nodes) (meta (key \"v\"))))",
      "unknown graph section", sx::ErrorCategory::Unsupported);
}

} // namespace

int main() {
  test_minimal_document();
  test_tensor_type();
  test_parameter_values();
  test_parameter_external();
  test_node();
  test_attributes();
  test_graph_io();
  test_topological_order();
  test_duplicate_definition();
  test_unknown_reference();
  test_invalid_dtype();
  test_invalid_shape();
  test_invalid_ordering();
  test_unsupported_version();
  test_malformed_syntax();
  test_unknown_form();

  if (g_failures == 0) {
    std::cout << "test_s_expr OK (16 cases)\n";
    return 0;
  }
  std::cerr << "test_s_expr FAILED: " << g_failures << " check(s)\n";
  return 1;
}
