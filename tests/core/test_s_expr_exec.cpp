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

// test_s_expr_exec.cpp — S-Expr → CPU Boom executor execution.
//
// Proves the end-to-end S-Expr → Boom IR → planner graph → CPU kernel path
// computes correct results, against an independently hand-computed reference.
// It also exercises the v0 execution-scope guards (at-least-one f32 input,
// single f32 output), a two-input merge graph, the input-buffer size check, and
// the operator/attribute validation that must reject malformed documents at
// compile time.

#include <sonicboom/sx/exec.h>
#include <sonicboom/sx/parser.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
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

sx::Bytes pack(const std::vector<float>& v) {
  sx::Bytes b(v.size() * sizeof(float));
  std::memcpy(b.data(), v.data(), b.size());
  return b;
}

std::vector<float> unpack(const sx::Bytes& b) {
  std::vector<float> v(b.size() / sizeof(float));
  std::memcpy(v.data(), b.data(), b.size());
  return v;
}

// Zero-filled buffers for every `:external` parameter, so a compile reaches
// structural validation without sidecars on disk.
std::unordered_map<std::string, sx::Bytes> external_weights(
    const sx::Document& doc) {
  std::unordered_map<std::string, sx::Bytes> weights;
  for (const auto& p : doc.graph.parameters) {
    if (!std::holds_alternative<sx::ExternalData>(p.data)) continue;
    auto bytes = sx::tensor_byte_size(p.type);
    if (bytes) weights[p.name] = sx::Bytes(static_cast<std::size_t>(*bytes));
  }
  return weights;
}

// Compile `src` and require a Compile error whose message contains `needle`.
// Malformed attributes and unsupported shapes must fail cleanly at compile
// time, never index out of bounds at run time.
void expect_compile_error(const std::string& src, const std::string& needle,
                          const char* what) {
  auto doc = sx::parse_document(src);
  if (!doc) {
    std::cerr << "  FAIL: " << what << " (unexpected parse error: "
              << doc.error().message << ")\n";
    ++g_failures;
    return;
  }
  auto exe = sx::Executable::compile(*doc, external_weights(*doc));
  if (exe) {
    std::cerr << "  FAIL: " << what << " (expected a compile error)\n";
    ++g_failures;
    return;
  }
  check(exe.error().kind == sx::ExecErrorKind::Compile,
        (std::string(what) + ": error kind is Compile").c_str());
  check(exe.error().message.find(needle) != std::string::npos,
        (std::string(what) + ": message contains \"" + needle + "\"").c_str());
}

// z = relu(x + b), single f32 input/output, shape [8].
const char* ADD_RELU =
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

// 1. Add + Relu executes through the real JIT and matches a hand reference.
void test_add_relu_executes() {
  auto doc = sx::parse_document(ADD_RELU);
  if (!doc) {
    std::cerr << "  FAIL: add_relu (parse error: " << doc.error().message
              << ")\n";
    ++g_failures;
    return;
  }

  auto exe = sx::Executable::compile(*doc, {});
  if (!exe) {
    std::cerr << "  FAIL: add_relu (compile error: " << exe.error().message
              << ")\n";
    ++g_failures;
    return;
  }

  check((*exe)->input_type().dtype == sx::DType::Float32 &&
            (*exe)->input_type().shape.dims.size() == 1 &&
            (*exe)->input_type().shape.dims[0] == 8,
        "add_relu: input type is f32[8]");
  check((*exe)->output_type().dtype == sx::DType::Float32 &&
            (*exe)->output_type().shape.dims.size() == 1 &&
            (*exe)->output_type().shape.dims[0] == 8,
        "add_relu: output type is f32[8]");

  const std::vector<float> x = {1.0f, 2.0f, -3.0f, 4.0f,
                                -5.0f, 6.0f, -7.0f, 8.0f};
  sx::Bytes out;
  auto res = (*exe)->run(pack(x), out);
  if (!res) {
    std::cerr << "  FAIL: add_relu (run error: " << res.error().message
              << ")\n";
    ++g_failures;
    return;
  }

  // x + b = [1.5, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
  // relu  = [1.5, 1.0,  0.0, 1.0,  0.0, 1.0,  0.0, 1.0]
  const std::vector<float> expected = {1.5f, 1.0f, 0.0f, 1.0f,
                                       0.0f, 1.0f, 0.0f, 1.0f};
  std::vector<float> got = unpack(out);
  check(got.size() == 8, "add_relu: output has 8 elements");
  for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-5f) {
      std::cerr << "  FAIL: add_relu (elem " << i << ": got " << got[i]
                << ", expected " << expected[i] << ")\n";
      ++g_failures;
    }
  }
}

// 2. A two-input merge graph (`add(a, b)`) now compiles and runs — the v0
//    single-input scope was relaxed to support co-execution merge slices.
void test_two_input_add() {
  auto doc = sx::parse_document(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"a\" (tensor float32 (shape 2)))"
      "         (input \"b\" (tensor float32 (shape 2))))"
      " (outputs (output \"c\")) (parameters)"
      " (nodes (node add (inputs \"a\" \"b\")"
      "   (outputs (\"c\" (tensor float32 (shape 2))))))))");
  if (!doc) {
    std::cerr << "  FAIL: two-input (parse error: " << doc.error().message << ")\n";
    ++g_failures;
    return;
  }
  auto exe = sx::Executable::compile(*doc, {});
  if (!exe) {
    std::cerr << "  FAIL: two-input (compile error: " << exe.error().message << ")\n";
    ++g_failures;
    return;
  }

  const std::vector<sx::Bytes> ins = {pack({1.0f, 2.0f}), pack({3.0f, 4.0f})};
  std::vector<sx::Bytes> outs;
  auto res = (*exe)->run(ins, outs);
  if (!res) {
    std::cerr << "  FAIL: two-input (run error: " << res.error().message << ")\n";
    ++g_failures;
    return;
  }

  check(outs.size() == 1, "two-input: one output buffer");
  const std::vector<float> expected = {4.0f, 6.0f};
  std::vector<float> got = unpack(outs[0]);
  for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i)
    check(std::fabs(got[i] - expected[i]) <= 1e-5f, "two-input: add value correct");
}

// 3. A wrong-sized input buffer → Binding error (byte-count check).
void test_binding_size_check() {
  auto doc = sx::parse_document(ADD_RELU);
  if (!doc) {
    std::cerr << "  FAIL: binding (parse error: " << doc.error().message
              << ")\n";
    ++g_failures;
    return;
  }
  auto exe = sx::Executable::compile(*doc, {});
  if (!exe) {
    std::cerr << "  FAIL: binding (compile error: " << exe.error().message
              << ")\n";
    ++g_failures;
    return;
  }
  sx::Bytes out;
  auto res = (*exe)->run(pack({1.0f, 2.0f, 3.0f, 4.0f}), out);  // 16B ≠ 32B
  if (res) {
    std::cerr << "  FAIL: binding (expected Binding error for wrong size)\n";
    ++g_failures;
    return;
  }
  check(res.error().kind == sx::ExecErrorKind::Binding,
        "binding: error kind is Binding");
}

// 4. MaxPool with negative input and nonzero padding must pad with -inf, not
//    0.0 (the ONNX semantics); otherwise negative values would be dragged up
//    toward zero by the pad. Input [1,1,2,2] = [[-1,-2],[-3,-4]], kernel 2x2,
//    stride 1, pad 1 all around → [1,1,3,3] (hand-computed, pad = -inf).
const char* MAXPOOL_PAD =
    "(sonicboom-s-expr (version 0 1) (graph (name \"maxpool_pad\")"
    " (inputs (input \"x\" (tensor float32 (shape 1 1 2 2))))"
    " (outputs (output \"y\")) (parameters)"
    " (nodes (node max_pool (inputs \"x\")"
    "   (outputs (\"y\" (tensor float32 (shape 1 1 3 3))))"
    "   (attrs (kernel_shape (ints 2 2)) (strides (ints 1 1))"
    "          (pads (ints 1 1 1 1)))))))";

void test_max_pool_neg_inf_pad() {
  auto doc = sx::parse_document(MAXPOOL_PAD);
  if (!doc) {
    std::cerr << "  FAIL: maxpool-pad (parse error: " << doc.error().message
              << ")\n";
    ++g_failures;
    return;
  }
  auto exe = sx::Executable::compile(*doc, {});
  if (!exe) {
    std::cerr << "  FAIL: maxpool-pad (compile error: " << exe.error().message
              << ")\n";
    ++g_failures;
    return;
  }

  const std::vector<float> x = {-1.0f, -2.0f, -3.0f, -4.0f};  // NCHW 1x1x2x2
  sx::Bytes out;
  auto res = (*exe)->run(pack(x), out);
  if (!res) {
    std::cerr << "  FAIL: maxpool-pad (run error: " << res.error().message
              << ")\n";
    ++g_failures;
    return;
  }

  // With -inf padding every 2x2 window (stride 1, pad 1) maxes over the real
  // values plus -inf pads, yielding exactly the row/col-wise windowed maxima.
  const std::vector<float> expected = {-1.0f, -1.0f, -2.0f,
                                       -1.0f, -1.0f, -2.0f,
                                       -3.0f, -3.0f, -4.0f};
  std::vector<float> got = unpack(out);
  check(got.size() == 9, "maxpool-pad: output has 9 elements");
  for (std::size_t i = 0; i < expected.size() && i < got.size(); ++i) {
    if (std::fabs(got[i] - expected[i]) > 1e-5f) {
      std::cerr << "  FAIL: maxpool-pad (elem " << i << ": got " << got[i]
                << ", expected " << expected[i] << ")\n";
      ++g_failures;
    }
  }
}

// 5. An overflowing shape must be rejected at compile time (checked arithmetic)
//    rather than wrapping to a garbage size that reaches allocation/memcpy.
void test_overflow_rejected() {
  auto doc = sx::parse_document(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"x\" (tensor float32"
      "   (shape 2147483648 2147483648 2147483648))))"
      " (outputs (output \"x\")) (parameters) (nodes)))");
  if (!doc) {
    std::cerr << "  FAIL: overflow (parse error: " << doc.error().message
              << ")\n";
    ++g_failures;
    return;
  }
  auto exe = sx::Executable::compile(*doc, {});
  if (exe) {
    std::cerr << "  FAIL: overflow (expected Compile error for overflowing shape)\n";
    ++g_failures;
    return;
  }
  check(exe.error().kind == sx::ExecErrorKind::Compile,
        "overflow: error kind is Compile");
  check(exe.error().message.find("overflows") != std::string::npos,
        "overflow: message mentions overflow");
}

// 6. An operator outside the executor vocabulary → Compile error naming it.
void test_unsupported_op() {
  expect_compile_error(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"a\" (tensor float32 (shape 2))))"
      " (outputs (output \"c\")) (parameters)"
      " (nodes (node foo (inputs \"a\")"
      "   (outputs (\"c\" (tensor float32 (shape 2))))))))",
      "unsupported operator 'foo'", "unsupported op");
}

// 7. Conv with an empty strides attr → Compile error, never OOB.
void test_conv_empty_strides() {
  expect_compile_error(
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

// 8. Conv with an empty dilations attr → Compile error, never OOB.
void test_conv_empty_dilations() {
  expect_compile_error(
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
      "dilations", "conv empty dilations");
}

// 9. Gemm with a rank-3 input → Compile error (v0 gemm is strictly 2-D).
void test_gemm_rank_mismatch() {
  expect_compile_error(
      "(sonicboom-s-expr (version 0 1) (graph (name \"g\")"
      " (inputs (input \"a\" (tensor float32 (shape 1 2 3))))"
      " (outputs (output \"c\"))"
      " (parameters"
      "   (parameter \"b\" (tensor float32 (shape 3 4))"
      "     (data :external \"b.bin\" 0 48)))"
      " (nodes (node gemm (inputs \"a\" \"b\")"
      "   (outputs (\"c\" (tensor float32 (shape 1 2 4))))))))",
      "gemm requires rank-2 A and B", "gemm rank mismatch");
}

// 10. ReduceMean with a duplicated axis → Compile error (no divisor bloat).
void test_reduce_mean_duplicate_axes() {
  expect_compile_error(
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
  test_add_relu_executes();
  test_two_input_add();
  test_binding_size_check();
  test_max_pool_neg_inf_pad();
  test_overflow_rejected();
  test_unsupported_op();
  test_conv_empty_strides();
  test_conv_empty_dilations();
  test_gemm_rank_mismatch();
  test_reduce_mean_duplicate_axes();

  if (g_failures == 0) {
    std::cout << "test_s_expr_exec OK (10 cases)\n";
    return 0;
  }
  std::cerr << "test_s_expr_exec FAILED: " << g_failures << " check(s)\n";
  return 1;
}
