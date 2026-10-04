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

// Graph adapter tests (P2): faithful adaptation of sx::Document into the
// planner graph, deterministic ID assignment, constant/external classification,
// and rejection of unresolved references / duplicates / unknown ops / overflow.

#include <sonicboom/planner/graph.h>

#include <sonicboom/sx/ir.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace pl = sonicboom::planner;
namespace sx = sonicboom::sx;

namespace {
int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

sx::TensorType ft(std::vector<int64_t> dims) {
  return {sx::DType::Float32, sx::Shape{std::move(dims)}};
}

sx::Document doc(const std::string& name) {
  sx::Document d;
  d.version_major = 0;
  d.version_minor = 1;
  d.graph.name = name;
  d.graph.opsets.push_back({"default", 20});
  return d;
}

sx::Node relu(const std::string& in, const std::string& out,
              std::vector<int64_t> dims) {
  sx::Node n;
  n.op_name = "relu";
  n.inputs = {in};
  n.outputs = {{out, ft(std::move(dims))}};
  return n;
}
} // namespace

int main() {
  // --- simple chain: x -> Relu -> y ----------------------------------------
  auto d = doc("g");
  d.graph.inputs.push_back({"x", ft({1, 3})});
  d.graph.nodes.push_back(relu("x", "y", {1, 3}));
  d.graph.outputs = {"y"};

  auto g = pl::adapt_graph(d);
  check(g.has_value(), "simple chain adapts");
  if (!g) {
    std::cerr << "unexpected: " << g.error().message << "\n";
    return 1;
  }
  check(g->tensors.size() == 2, "two tensors (x, y)");
  check(g->nodes.size() == 1, "one node");
  check(g->inputs.size() == 1 && g->outputs.size() == 1, "one in / one out");
  check(g->tensors[0].name == "x" && g->tensors[0].is_graph_input,
        "x is graph input");
  check(g->tensors[1].name == "y" && g->tensors[1].is_graph_output,
        "y is graph output");
  check(g->tensors[1].size_bytes == 1 * 3 * 4, "y size = 12 bytes");
  check(g->nodes[0].op == pl::OpKind::Relu, "node op = relu");
  check(g->nodes[0].inputs.size() == 1 && g->nodes[0].inputs[0] == pl::TensorId{0},
        "node input resolves to x (id 0)");

  // --- branching + constants + external ------------------------------------
  auto d2 = doc("g2");
  d2.graph.inputs.push_back({"x", ft({3})});
  sx::Parameter w;  // float32 weight -> external
  w.name = "w";
  w.type = ft({3});
  w.data = sx::ExternalData{"weights.bin", 0, 12};
  d2.graph.parameters.push_back(w);
  sx::Parameter c;  // int64 constant -> inline values
  c.name = "axes";
  c.type = {sx::DType::Int64, sx::Shape{{2}}};
  c.data = sx::ValuesData{{{int64_t{-1}, int64_t{-2}}}};
  d2.graph.parameters.push_back(c);

  sx::Node add;
  add.op_name = "add";
  add.inputs = {"x", "w"};
  add.outputs = {{"s", ft({3})}};
  d2.graph.nodes.push_back(add);
  d2.graph.outputs = {"s"};

  auto g2 = pl::adapt_graph(d2);
  check(g2.has_value(), "branch graph adapts");
  if (!g2)
    return 1;
  // ids: x=0, w=1 (external), axes=2 (constant), s=3
  const auto* w_t = g2->find_tensor(pl::TensorId{1});
  const auto* axes_t = g2->find_tensor(pl::TensorId{2});
  check(w_t && w_t->is_constant && w_t->is_external, "w is external constant");
  check(axes_t && axes_t->is_constant && !axes_t->is_external,
        "axes is inline constant");
  check(g2->tensors.size() == 4, "four tensors (x,w,axes,s)");

  // --- deterministic fingerprint -------------------------------------------
  auto g2b = pl::adapt_graph(d2);
  check(g2b.has_value() &&
            pl::model_fingerprint(*g2b) == pl::model_fingerprint(*g2),
        "model fingerprint deterministic");

  // --- unsupported operator ------------------------------------------------
  auto d3 = doc("g3");
  d3.graph.inputs.push_back({"x", ft({4})});
  d3.graph.nodes.push_back(relu("x", "y", {4}));
  d3.graph.nodes[0].op_name = "gelu";
  auto g3 = pl::adapt_graph(d3);
  check(!g3.has_value(), "unsupported operator rejected");
  check(g3.error().code == pl::PlannerErrorCode::UnsupportedOperator,
        "error code UnsupportedOperator");

  // --- unresolved reference ------------------------------------------------
  auto d4 = doc("g4");
  d4.graph.inputs.push_back({"x", ft({4})});
  d4.graph.nodes.push_back(relu("missing", "y", {4}));
  auto g4 = pl::adapt_graph(d4);
  check(!g4.has_value() && g4.error().code == pl::PlannerErrorCode::InvalidGraph,
        "unresolved reference rejected");

  // --- duplicate definition -------------------------------------------------
  auto d5 = doc("g5");
  d5.graph.inputs.push_back({"x", ft({4})});
  sx::Node n1 = relu("x", "y", {4});
  sx::Node n2 = relu("y", "y", {4});  // defines 'y' again
  d5.graph.nodes.push_back(n1);
  d5.graph.nodes.push_back(n2);
  auto g5 = pl::adapt_graph(d5);
  check(!g5.has_value() && g5.error().code == pl::PlannerErrorCode::InvalidGraph,
        "duplicate definition rejected");

  // --- tensor byte-size overflow -------------------------------------------
  auto d6 = doc("g6");
  d6.graph.inputs.push_back({"x", ft({std::numeric_limits<int64_t>::max(),
                                       std::numeric_limits<int64_t>::max()})});
  auto g6 = pl::adapt_graph(d6);
  check(!g6.has_value() &&
            g6.error().code == pl::PlannerErrorCode::ArithmeticOverflow,
        "tensor size overflow rejected");

  // --- attributes preserved -------------------------------------------------
  auto d7 = doc("g7");
  d7.graph.inputs.push_back({"x", ft({1, 1, 5, 5})});
  sx::Node cv;
  cv.op_name = "conv";
  cv.inputs = {"x"};
  cv.outputs = {{"y", ft({1, 1, 5, 5})}};
  cv.attributes.push_back({"strides", std::vector<int64_t>{1, 1}});
  cv.attributes.push_back({"pads", std::vector<int64_t>{1, 1, 1, 1}});
  d7.graph.nodes.push_back(cv);
  d7.graph.outputs = {"y"};
  auto g7 = pl::adapt_graph(d7);
  check(g7.has_value() && g7->nodes[0].attributes.size() == 2,
        "attributes preserved verbatim");

  if (g_failures == 0) {
    std::cout << "test_planner_graph OK\n";
    return 0;
  }
  std::cerr << "test_planner_graph FAILED: " << g_failures << " check(s)\n";
  return 1;
}
