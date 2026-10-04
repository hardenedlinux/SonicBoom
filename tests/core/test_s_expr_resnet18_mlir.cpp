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

// test_s_expr_resnet18_mlir.cpp — integration test: parse the frozen ResNet-18
// S-Expr v0.1 example, lower it to MLIR, and verify structural properties of
// the produced (already verified) module. Deliberately NOT byte-for-byte.

#include <sonicboom/sx/lowering.h>
#include <sonicboom/sx/parser.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#ifndef SONICBOOM_SRC_DIR
#error "SONICBOOM_SRC_DIR must be defined to the repository root"
#endif

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

void check_contains(const std::string& text, const std::string& needle,
                    const char* what) {
  if (text.find(needle) == std::string::npos) {
    std::cerr << "  FAIL: " << what << " (missing \"" << needle << "\")\n";
    ++g_failures;
  }
}

// Count non-overlapping occurrences of `needle` in `text`.
int count_of(const std::string& text, const std::string& needle) {
  int n = 0;
  std::string::size_type pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) {
    ++n;
    pos += needle.size();
  }
  return n;
}

} // namespace

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

  auto doc = sonicboom::sx::parse_document(ss.str());
  if (!doc) {
    std::cerr << "resnet18 parse failed: " << doc.error().message << "\n";
    return 1;
  }

  // lower_to_mlir already runs full MLIR verification; a success value implies
  // the ResNet-18 graph produced a verified module.
  auto mlir = sonicboom::sx::lower_to_mlir(*doc);
  if (!mlir) {
    std::cerr << "resnet18 lowering failed: " << mlir.error().message << "\n";
    return 1;
  }
  const std::string& out = *mlir;

  check_contains(out, "module", "module wrapper");
  check_contains(out, "func.func @main", "single func.func @main");
  check_contains(out, "tensor<1x3x224x224xf32>", "input type tensor<1x3x224x224xf32>");
  check_contains(out, "tensor<1x1000xf32>", "output type tensor<1x1000xf32>");
  check_contains(out, "sonicboom.external_data", "external-weight metadata attr");

  // Operator census (ResNet-18: 20 conv, 17 relu, 1 max_pool, 1 reshape,
  // 1 reduce_mean, 1 gemm with transB, 8 add). Only the unambiguously named
  // linalg/arith ops are counted; addf/generic are shared with bias/reduction
  // plumbing and so are not asserted exactly.
  check(count_of(out, "linalg.conv_2d_nchw_fchw") == 20,
        "20 conv_2d_nchw_fchw");
  check(count_of(out, "arith.maximumf") == 17, "17 relu (arith.maximumf)");
  check(count_of(out, "linalg.pooling_nchw_max") == 1, "1 pooling_nchw_max");
  check(count_of(out, "tensor.reshape") == 1, "1 tensor.reshape");
  check(count_of(out, "linalg.matmul") == 1, "1 linalg.matmul");
  check(count_of(out, "linalg.transpose") == 1, "1 linalg.transpose (gemm transB)");

  if (g_failures == 0) {
    std::cout << "test_s_expr_resnet18_mlir OK (ResNet-18 lowers + verifies, "
              << out.size() << " bytes of MLIR)\n";
    return 0;
  }
  std::cerr << "test_s_expr_resnet18_mlir FAILED: " << g_failures
            << " check(s)\n";
  return 1;
}
