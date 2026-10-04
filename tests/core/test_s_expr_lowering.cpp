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

// test_s_expr_lowering.cpp — S-Expr v0.1 → MLIR lowering focused unit tests.
//
// Sixteen cases covering the frozen v0 operator subset, its lowering contract
// (type mapping, input/constant materialization, each of the seven operators,
// graph-return wiring), and the attribute/shape validation that keeps malformed
// inputs (empty strides/dilations, gemm rank mismatch, duplicate reduce_mean
// axes, unsupported operators) from reaching the MLIR builder out of bounds.
// Each case parses a small document, lowers it to textual MLIR, and checks
// structural properties (substrings), never byte-for-byte output.
//
// NOTE: no <cassert> — the core build is Release (NDEBUG); explicit checks plus
// a non-zero exit code keep the test meaningful.

#include <sonicboom/sx/lowering.h>
#include <sonicboom/sx/parser.h>

#include <iostream>
#include <string>

namespace sx = sonicboom::sx;

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

// Parse + lower `src`; require both to succeed and return the MLIR text.
bool must_lower(const std::string& src, std::string& out, const char* what) {
  auto doc = sx::parse_document(src);
  if (!doc) {
    std::cerr << "  FAIL: " << what << " (parse error: "
              << doc.error().message << ")\n";
    ++g_failures;
    return false;
  }
  auto mlir = sx::lower_to_mlir(*doc);
  if (!mlir) {
    std::cerr << "  FAIL: " << what << " (lower error: " << mlir.error().message
              << ")\n";
    ++g_failures;
    return false;
  }
  out = std::move(*mlir);
  return true;
}

// Parse `src`, then expect lowering to FAIL with an Operator error whose
// message contains `needle`. Used to prove malformed attributes / unsupported
// shapes fail cleanly (structured error) rather than indexing out of bounds.
void expect_lower_error(const std::string& src, const std::string& needle,
                        const char* what) {
  auto doc = sx::parse_document(src);
  if (!doc) {
    std::cerr << "  FAIL: " << what << " (unexpected parse error: "
              << doc.error().message << ")\n";
    ++g_failures;
    return;
  }
  auto mlir = sx::lower_to_mlir(*doc);
  if (mlir) {
    std::cerr << "  FAIL: " << what << " (expected lowering error, got MLIR)\n";
    ++g_failures;
    return;
  }
  check(mlir.error().kind == sx::LoweringErrorKind::Operator,
        (std::string(what) + ": error kind is Operator").c_str());
  check(mlir.error().message.find(needle) != std::string::npos,
        (std::string(what) + ": message contains \"" + needle + "\"").c_str());
}

// 1. dtype/shape → MLIR tensor type (float32 NCHW + int64 rank-1).
void test_type_mapping() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1 3 224 224))))"
          " (outputs (output \"x\"))"
          " (parameters (parameter \"c\" (tensor int64 (shape 2))"
          "   (data :values 1 2)))"
          " (nodes)))",
          out, "type mapping")) {
    return;
  }
  check_contains(out, "tensor<1x3x224x224xf32>", "type: f32 NCHW tensor");
  check_contains(out, "tensor<2xi64>", "type: i64 rank-1 tensor");
}

// 2. graph input becomes a func.func argument.
void test_input_argument() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 2 3))))"
          " (outputs (output \"a\")) (parameters) (nodes)))",
          out, "input argument")) {
    return;
  }
  check_contains(out, "func.func @g(", "input: function signature");
  check_contains(out, "tensor<2x3xf32>", "input: argument type");
  check_contains(out, "return", "input: return wired");
}

// 3. :values parameter becomes an arith.constant dense literal.
void test_constant() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs) (outputs (output \"c\"))"
          " (parameters (parameter \"c\" (tensor int64 (shape 2))"
          "   (data :values -1 -2)))"
          " (nodes)))",
          out, "constant")) {
    return;
  }
  check_contains(out, "arith.constant", "constant: arith.constant op");
  check_contains(out, "tensor<2xi64>", "constant: i64 type");
}

// 4. add → arith.addf.
void test_add() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 2 3)))"
          "         (input \"b\" (tensor float32 (shape 2 3))))"
          " (outputs (output \"c\")) (parameters)"
          " (nodes (node add (inputs \"a\" \"b\")"
          "   (outputs (\"c\" (tensor float32 (shape 2 3))))))))",
          out, "add")) {
    return;
  }
  check_contains(out, "arith.addf", "add: arith.addf");
}

// 5. relu → arith.maximumf(x, 0).
void test_relu() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 2 3))))"
          " (outputs (output \"c\")) (parameters)"
          " (nodes (node relu (inputs \"a\")"
          "   (outputs (\"c\" (tensor float32 (shape 2 3))))))))",
          out, "relu")) {
    return;
  }
  check_contains(out, "arith.maximumf", "relu: arith.maximumf");
}

// 6. reshape → tensor.reshape (shape derived from the output type).
void test_reshape() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 2 3))))"
          " (outputs (output \"c\")) (parameters)"
          " (nodes (node reshape (inputs \"a\")"
          "   (outputs (\"c\" (tensor float32 (shape 3 2))))))))",
          out, "reshape")) {
    return;
  }
  check_contains(out, "tensor.reshape", "reshape: tensor.reshape");
}

// 7. reduce_mean → linalg.generic sum + arith.divf (axes from a :values param).
void test_reduce_mean() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 4 3 3))))"
          " (outputs (output \"c\"))"
          " (parameters (parameter \"axes\" (tensor int64 (shape 2))"
          "   (data :values 1 2)))"
          " (nodes (node reduce_mean (inputs \"a\" \"axes\")"
          "   (outputs (\"c\" (tensor float32 (shape 4 1 1))))))))",
          out, "reduce_mean")) {
    return;
  }
  check_contains(out, "linalg.generic", "reduce_mean: linalg.generic");
  check_contains(out, "arith.divf", "reduce_mean: arith.divf");
}

// 8. conv → linalg.conv_2d_nchw_fchw (weight/bias as external placeholders).
void test_conv() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1 3 4 4))))"
          " (outputs (output \"y\"))"
          " (parameters"
          "   (parameter \"w\" (tensor float32 (shape 2 3 2 2))"
          "     (data :external \"w.bin\" 0 96))"
          "   (parameter \"b\" (tensor float32 (shape 2))"
          "     (data :external \"b.bin\" 0 8)))"
          " (nodes (node conv (inputs \"x\" \"w\" \"b\")"
          "   (outputs (\"y\" (tensor float32 (shape 1 2 3 3))))))))",
          out, "conv")) {
    return;
  }
  check_contains(out, "linalg.conv_2d_nchw_fchw", "conv: linalg.conv_2d_nchw_fchw");
}

// 9. max_pool → linalg.pooling_nchw_max.
void test_max_pool() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"x\" (tensor float32 (shape 1 2 4 4))))"
          " (outputs (output \"y\")) (parameters)"
          " (nodes (node max_pool (inputs \"x\")"
          "   (outputs (\"y\" (tensor float32 (shape 1 2 2 2))))"
          "   (attrs (kernel_shape (ints 2 2)) (strides (ints 2 2))"
          "          (pads (ints 0 0 0 0)))))))",
          out, "max_pool")) {
    return;
  }
  check_contains(out, "linalg.pooling_nchw_max", "max_pool: linalg.pooling_nchw_max");
}

// 10. gemm → linalg.matmul (+ broadcast bias).
void test_gemm() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 2 3))))"
          " (outputs (output \"c\"))"
          " (parameters"
          "   (parameter \"b\" (tensor float32 (shape 3 4))"
          "     (data :external \"b.bin\" 0 48))"
          "   (parameter \"bias\" (tensor float32 (shape 4))"
          "     (data :external \"bias.bin\" 0 16)))"
          " (nodes (node gemm (inputs \"a\" \"b\" \"bias\")"
          "   (outputs (\"c\" (tensor float32 (shape 2 4))))))))",
          out, "gemm")) {
    return;
  }
  check_contains(out, "linalg.matmul", "gemm: linalg.matmul");
}

// 11. graph outputs map to a func.return of the produced values.
void test_return_values() {
  std::string out;
  if (!must_lower(
          "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
          " (inputs (input \"a\" (tensor float32 (shape 2))))"
          " (outputs (output \"c\") (output \"a\")) (parameters)"
          " (nodes (node relu (inputs \"a\")"
          "   (outputs (\"c\" (tensor float32 (shape 2))))))))",
          out, "return values")) {
    return;
  }
  check_contains(out, "return", "return: func.return present");
}

// 12. unknown operator → explicit Operator lowering error.
void test_unsupported_op() {
  auto doc = sx::parse_document(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"a\" (tensor float32 (shape 2))))"
      " (outputs (output \"c\")) (parameters)"
      " (nodes (node foo (inputs \"a\")"
      "   (outputs (\"c\" (tensor float32 (shape 2))))))))");
  if (!doc) {
    std::cerr << "  FAIL: unsupported-op (unexpected parse error: "
              << doc.error().message << ")\n";
    ++g_failures;
    return;
  }
  auto mlir = sx::lower_to_mlir(*doc);
  if (mlir) {
    std::cerr << "  FAIL: unsupported-op (expected lowering error, got MLIR)\n";
    ++g_failures;
    return;
  }
  check(mlir.error().kind == sx::LoweringErrorKind::Operator,
        "unsupported-op: error kind Operator");
  check(mlir.error().message.find("unsupported operator 'foo'") !=
            std::string::npos,
        "unsupported-op: message names 'foo'");
}

// 13. conv with an empty strides attr → structured error, never OOB.
void test_conv_empty_strides() {
  expect_lower_error(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float32 (shape 1 3 4 4))))"
      " (outputs (output \"y\"))"
      " (parameters"
      "   (parameter \"w\" (tensor float32 (shape 2 3 2 2))"
      "     (data :external \"w.bin\" 0 96))"
      "   (parameter \"b\" (tensor float32 (shape 2))"
      "     (data :external \"b.bin\" 0 8)))"
      " (nodes (node conv (inputs \"x\" \"w\" \"b\")"
      "   (outputs (\"y\" (tensor float32 (shape 1 2 3 3))))"
      "   (attrs (strides (ints)))))))",
      "conv strides must have 2 entries", "conv empty strides");
}

// 14. conv with an empty dilations attr → structured error, never OOB.
void test_conv_empty_dilations() {
  expect_lower_error(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float32 (shape 1 3 4 4))))"
      " (outputs (output \"y\"))"
      " (parameters"
      "   (parameter \"w\" (tensor float32 (shape 2 3 2 2))"
      "     (data :external \"w.bin\" 0 96))"
      "   (parameter \"b\" (tensor float32 (shape 2))"
      "     (data :external \"b.bin\" 0 8)))"
      " (nodes (node conv (inputs \"x\" \"w\" \"b\")"
      "   (outputs (\"y\" (tensor float32 (shape 1 2 3 3))))"
      "   (attrs (dilations (ints)))))))",
      "conv dilations must have 2 entries", "conv empty dilations");
}

// 15. gemm with a rank-3 input → structured error (v0 gemm is strictly 2-D).
void test_gemm_rank_mismatch() {
  expect_lower_error(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"a\" (tensor float32 (shape 1 2 3))))"
      " (outputs (output \"c\"))"
      " (parameters"
      "   (parameter \"b\" (tensor float32 (shape 3 4))"
      "     (data :external \"b.bin\" 0 48)))"
      " (nodes (node gemm (inputs \"a\" \"b\")"
      "   (outputs (\"c\" (tensor float32 (shape 1 2 4))))))))",
      "gemm requires rank-2 A and B in v0", "gemm rank mismatch");
}

// 16. reduce_mean with a duplicated axis → structured error (no divisor bloat).
void test_reduce_mean_duplicate_axes() {
  expect_lower_error(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"a\" (tensor float32 (shape 4 3 3))))"
      " (outputs (output \"c\"))"
      " (parameters (parameter \"axes\" (tensor int64 (shape 2))"
      "   (data :values 1 1)))"
      " (nodes (node reduce_mean (inputs \"a\" \"axes\")"
      "   (outputs (\"c\" (tensor float32 (shape 4 1 3))))))))",
      "reduce_mean has duplicate axes", "reduce_mean duplicate axes");
}

} // namespace

int main() {
  test_type_mapping();
  test_input_argument();
  test_constant();
  test_add();
  test_relu();
  test_reshape();
  test_reduce_mean();
  test_conv();
  test_max_pool();
  test_gemm();
  test_return_values();
  test_unsupported_op();
  test_conv_empty_strides();
  test_conv_empty_dilations();
  test_gemm_rank_mismatch();
  test_reduce_mean_duplicate_axes();

  if (g_failures == 0) {
    std::cout << "test_s_expr_lowering OK (16 cases)\n";
    return 0;
  }
  std::cerr << "test_s_expr_lowering FAILED: " << g_failures << " check(s)\n";
  return 1;
}
