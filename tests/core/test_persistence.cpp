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

// Persistence tests (Task 1.2): S-Expr round-trip serialization, and the
// export → load → run loop. Verifies that a graph + trained parameters export
// to a self-contained on-disk artifact (model.sx + weights.bin) that a
// standalone C++ `Model::load` recovers and executes correctly — no Guile or
// Python involved.

#include <sonicboom/runtime.h>
#include <sonicboom/sx/parser.h>
#include <sonicboom/sx/serialize.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
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

// Exercises every attribute typed-value form, both data layouts, an opset, and
// a node-level version — a stress test for lossless serialization.
const char* RICH =
    "(sonicboom-s-expr (version 0 1)"
    " (graph (name \"rich\")"
    "  (opset \"default\" 20)"
    "  (inputs (input \"x\" (tensor float32 (shape 2 3))))"
    "  (outputs (output \"z\"))"
    "  (parameters"
    "    (parameter \"w\" (tensor float32 (shape 3 2)) (data :external \"w.bin\" 0 24))"
    "    (parameter \"c\" (tensor int64 (shape 2)) (data :values -1 -2)))"
    "  (nodes"
    "    (node conv (inputs \"x\" \"w\")"
    "      (outputs (\"z\" (tensor float32 (shape 1 1))))"
    "      (attrs"
    "        (group (int 1)) (alpha (float 1.5))"
    "        (auto_pad (string \"SAME_UPPER\")) (ceil_mode (bool #t))"
    "        (strides (ints 2 2)) (dilations (floats 1.0 1.0))"
    "        (tags (strings \"a\" \"b\")) (flags (bools #t #f)))"
    "      (version 7)))))";

// A runnable graph with a trained parameter: z = x + w.
const char* ADD_W =
    "(sonicboom-s-expr (version 0 1)"
    " (graph (name \"add_w\")"
    "  (inputs (input \"x\" (tensor float32 (shape 3))))"
    "  (outputs (output \"z\"))"
    "  (parameters (parameter \"w\" (tensor float32 (shape 3)) (data :values 2 3 4)))"
    "  (nodes (node add (inputs \"x\" \"w\")"
    "          (outputs (\"z\" (tensor float32 (shape 3))))))))";
} // namespace

int main() {
  namespace fs = std::filesystem;

  // --- serialization round-trip (lossless, idempotent) ----------------------
  {
    auto d1 = sx::parse_document(RICH);
    check(d1.has_value(), "round-trip: RICH parses");
    if (!d1)
      return 1;

    auto s1 = sx::serialize_document(*d1);
    check(s1.has_value(), "round-trip: serialize succeeds");
    if (!s1)
      return 1;

    // Spot-check the emitted text covers every form.
    for (const char* needle : {":external \"w.bin\"", ":values -1 -2",
                               "(float 1.5)", "(int 1)", "(string \"SAME_UPPER\")",
                               "(bool #t)", "(ints 2 2)", "(floats 1.0 1.0)",
                               "(strings \"a\" \"b\")", "(bools #t #f)",
                               "(opset \"default\" 20)", "(version 7)"}) {
      check(s1->find(needle) != std::string::npos,
            "round-trip: text contains expected form");
    }

    auto d2 = sx::parse_document(*s1);
    check(d2.has_value(), "round-trip: re-parses");
    if (!d2)
      return 1;

    auto s2 = sx::serialize_document(*d2);
    check(s2.has_value() && *s2 == *s1, "round-trip: serialize is idempotent");
    check(d2->graph.name == "rich" && d2->graph.nodes.size() == 1 &&
              d2->graph.parameters.size() == 2,
          "round-trip: structure preserved");
  }

  // --- export → load → run ---------------------------------------------------
  {
    auto doc = sx::parse_document(ADD_W);
    check(doc.has_value(), "export: ADD_W parses");
    if (!doc)
      return 1;

    fs::path dir = fs::temp_directory_path() / "sonicboom_persist_test";
    std::error_code ec;
    fs::remove_all(dir, ec);
    ec.clear();

    // Trained value for w differs from its inline :values initial value, proving
    // the export actually replaces the parameter data.
    std::unordered_map<std::string, sx::Bytes> params;
    params["w"] = pack({10.0f, 20.0f, 30.0f});

    auto exp = sb::export_model(*doc, params, dir.string());
    check(exp.has_value(), "export: export_model succeeds");
    if (!exp)
      return 1;

    check(fs::exists(dir / "model.sx") && fs::exists(dir / "weights.bin"),
          "export: model.sx + weights.bin written");

    // The persisted graph must reference the sidecar, not inline values.
    std::ifstream sx_file(dir / "model.sx", std::ios::binary);
    std::string sx_text((std::istreambuf_iterator<char>(sx_file)),
                        std::istreambuf_iterator<char>());
    check(sx_text.find(":external \"weights.bin\"") != std::string::npos,
          "export: parameter externalized to weights.bin");

    auto model = sb::Model::load(dir.string());
    check(model.has_value(), "export: Model::load succeeds");
    if (!model)
      return 1;

    auto r = model->run({pack({1.0f, 2.0f, 3.0f})});
    check(r.has_value(), "export: run succeeds");
    if (r) {
      std::vector<float> got = unpack(r->at(0));
      const std::vector<float> want = {11.0f, 22.0f, 33.0f};
      for (std::size_t i = 0; i < want.size(); ++i)
        check(std::fabs(got[i] - want[i]) <= 1e-4f,
              "export: x + trained w correct");
    }

    // Export rejects a parameter whose byte count disagrees with its type.
    std::unordered_map<std::string, sx::Bytes> bad_params;
    bad_params["w"] = pack({1.0f, 2.0f});  // 2 floats, not 3
    auto bad = sb::export_model(*doc, bad_params, dir.string());
    check(!bad.has_value() && bad.error().stage == "export",
          "export: byte-count mismatch rejected");

    // Loading a missing artifact fails at the load stage.
    auto missing = sb::Model::load((dir / "nope").string());
    check(!missing.has_value() && missing.error().stage == "load",
          "export: missing model file rejected");

    fs::remove_all(dir, ec);
  }

  if (g_failures == 0) {
    std::cout << "test_persistence OK\n";
    return 0;
  }
  std::cerr << "test_persistence FAILED: " << g_failures << " check(s)\n";
  return 1;
}
