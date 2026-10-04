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

// test_s_expr_resnet18.cpp — integration test: parse the frozen ResNet-18
// S-Expr v0.1 example document (design/s-expr-v0-resnet18.example.sx) end to
// end and verify it validates. No <cassert>: Release build disables it.

#include <sonicboom/sx/parser.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef SONICBOOM_SRC_DIR
#error "SONICBOOM_SRC_DIR must be defined to the repository root"
#endif

int main() {
  const std::string path =
      std::string(SONICBOOM_SRC_DIR) + "/design/s-expr-v0-resnet18.example.sx";
  std::ifstream in(path);
  if (!in) {
    std::cerr << "cannot open " << path << "\n";
    return 1;
  }
  std::ostringstream ss;
  ss << in.rdbuf();

  auto r = sonicboom::sx::parse_document(ss.str());
  if (!r) {
    std::cerr << "resnet18 parse failed: " << r.error().message;
    if (r.error().location.line != 0) {
      std::cerr << " (line " << r.error().location.line
                << ", col " << r.error().location.column << ")";
    }
    std::cerr << "\n";
    return 1;
  }

  int failures = 0;
  auto check = [&](bool ok, const char* what) {
    if (!ok) {
      std::cerr << "  FAIL: " << what << "\n";
      ++failures;
    }
  };

  const auto& g = r->graph;
  check(r->version_major == 0 && r->version_minor == 1, "document version 0.1");
  check(g.name == "main", "graph name");
  check(g.opsets.size() == 1 && g.opsets[0].domain == "default" &&
            g.opsets[0].version == 20,
        "opset default 20");
  check(g.inputs.size() == 1, "one graph input");
  check(g.outputs.size() == 1 && g.outputs[0] == "output", "one output 'output'");
  check(g.parameters.size() == 44, "44 parameters");
  check(g.nodes.size() == 49, "49 nodes");
  check(!g.nodes.empty() && g.nodes.front().op_name == "conv", "first node conv");
  check(g.nodes.back().op_name == "gemm", "last node gemm");
  check(g.inputs[0].type.dtype == sonicboom::sx::DType::Float32, "input dtype");

  if (failures == 0) {
    std::cout << "test_s_expr_resnet18 OK (49 nodes, 44 parameters)\n";
    return 0;
  }
  return 1;
}
