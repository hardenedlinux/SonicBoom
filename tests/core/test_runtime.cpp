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

// Unified C++ execution entry (Model) tests. Each scenario submits an S-Expr
// graph through the *single* Model API and verifies the numeric result plus the
// binding contract, covering both the homogeneous (whole-graph MLIR) path and
// the region-partitioned mixed (native-torch + MLIR) path, and the error paths
// (invalid graph, byte-count mismatch) that must fail meaningfully, not crash.

#include <sonicboom/runtime.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace sb = sonicboom;
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

// softmax([1,2,3]) row reference values.
const float P0 = 0.090030573f, P1 = 0.244728471f, P2 = 0.665240956f;
const std::vector<float> X = {1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f};

// mixed: softmax → relu. softmax routes to native-torch, relu to MLIR.
const char* SOFTMAX_RELU =
    "(sonicboom-s-expr (version 0 1) (graph (name \"sr\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// homogeneous: relu only. Negative entries must clamp to zero.
const char* RELU =
    "(sonicboom-s-expr (version 0 1) (graph (name \"relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node relu (inputs \"x\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// mixed: softmax → relu → softmax → relu (two separate MLIR regions).
const char* TWO_REGIONS =
    "(sonicboom-s-expr (version 0 1) (graph (name \"two_regions\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node softmax (inputs \"x\")"
    "     (outputs (\"s\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"s\")"
    "     (outputs (\"a\" (tensor float32 (shape 2 3)))))"
    "   (node softmax (inputs \"a\")"
    "     (outputs (\"t\" (tensor float32 (shape 2 3))))"
    "     (attrs (axis (int 1))))"
    "   (node relu (inputs \"t\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";

// invalid graph: an operator outside the planner vocabulary.
const char* UNKNOWN_OP =
    "(sonicboom-s-expr (version 0 1) (graph (name \"bad\")"
    " (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    " (outputs (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node frobnicate (inputs \"x\")"
    "     (outputs (\"z\" (tensor float32 (shape 2 3))))))))";
} // namespace

int main() {
  // --- mixed: native-torch softmax + MLIR relu through one API ---------------
  {
    auto model = sb::Model::compile(SOFTMAX_RELU);
    check(model.has_value(), "mixed: compiles from text");
    if (!model) {
      std::cerr << "  mixed compile error: " << model.error().message << "\n";
      return 1;
    }

    check(model->input_types().size() == 1 &&
              model->input_types()[0].dtype == sx::DType::Float32 &&
              model->input_types()[0].shape.dims == std::vector<int64_t>({2, 3}),
          "mixed: input binding contract");
    check(model->output_types().size() == 1 &&
              model->output_types()[0].shape.dims == std::vector<int64_t>({2, 3}),
          "mixed: output binding contract");

    auto r = model->run({pack(X)});
    check(r.has_value(), "mixed: run succeeds");
    if (r) {
      std::vector<float> got = unpack(r->at(0));
      const std::vector<float> want = {P0, P1, P2, P0, P1, P2};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-5f,
              "mixed: softmax→relu value correct");
    }

    // byte-count mismatch must fail meaningfully, not crash.
    sx::Bytes short_buf(10);
    auto bad = model->run({short_buf});
    check(!bad.has_value() && bad.error().stage == "execute" &&
              !bad.error().message.empty(),
          "mixed: byte-count mismatch rejected with a stage + message");
  }

  // --- homogeneous: relu only ------------------------------------------------
  {
    auto model = sb::Model::compile(RELU);
    check(model.has_value(), "homogeneous: compiles");
    if (!model)
      return 1;

    const std::vector<float> in = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f};
    auto r = model->run({pack(in)});
    check(r.has_value(), "homogeneous: run succeeds");
    if (r) {
      std::vector<float> got = unpack(r->at(0));
      const std::vector<float> want = {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-5f,
              "homogeneous: relu value correct");
    }
  }

  // --- two separate MLIR regions wired by per-task backends ------------------
  {
    auto model = sb::Model::compile(TWO_REGIONS);
    check(model.has_value(), "two-regions: compiles");
    if (!model)
      return 1;

    auto r = model->run({pack(X)});
    check(r.has_value(), "two-regions: run succeeds");
    if (r) {
      // softmax([1,2,3]) → P; relu → P; softmax(P) → Q; relu(Q) → Q.
      const float Q0 = 0.253497652f, Q1 = 0.295909144f, Q2 = 0.450593204f;
      std::vector<float> got = unpack(r->at(0));
      const std::vector<float> want = {Q0, Q1, Q2, Q0, Q1, Q2};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-4f,
              "two-regions: NT→MLIR→NT→MLIR value correct");
    }
  }

  // --- error path: invalid graph (unknown operator) --------------------------
  {
    auto model = sb::Model::compile(UNKNOWN_OP);
    check(!model.has_value() && model.error().stage == "plan" &&
              !model.error().message.empty(),
          "unknown op: compile fails at plan stage with a message");
  }

  // --- error path: malformed text --------------------------------------------
  {
    auto model = sb::Model::compile("(not a valid s-expr");
    check(!model.has_value() && model.error().stage == "parse",
          "malformed text: compile fails at parse stage");
  }

  if (g_failures == 0) {
    std::cout << "test_runtime OK\n";
    return 0;
  }
  std::cerr << "test_runtime FAILED: " << g_failures << " check(s)\n";
  return 1;
}
