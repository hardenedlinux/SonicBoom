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

// End-to-end loader (the independent inference half of the Guile → C++ e2e
// loop). Loads an exported model artifact directory (model.sx + weights.bin,
// produced by the Guile training/exporter), runs the XOR inputs through the
// S-Expr → MLIR deployment path, and compares the outputs to `expected.bin`
// (the trained model's forward outputs, written by the Guile side). This proves
// the training engine (native-torch autograd) and the deployment engine
// (S-Expr → MLIR) agree on structure, naming, shapes and semantics — including
// that the trained weights reach the gemm nodes as external data.

#include <sonicboom/runtime.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
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

// XOR inputs in the same order the Guile trainer uses: [0,0] [0,1] [1,0] [1,1].
const std::vector<float> XOR_IN = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
} // namespace

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <export-dir>\n";
    return 2;
  }
  const std::string dir = argv[1];

  // The artifact must carry a gemm graph whose trained weights are externalized.
  {
    std::ifstream f(dir + "/model.sx", std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    check(f.good(), "artifact: model.sx present and readable");
    check(text.find("(node gemm") != std::string::npos,
          "artifact: graph contains gemm node(s)");
    check(text.find(":external \"weights.bin\"") != std::string::npos,
          "artifact: trained weights externalized to weights.bin");
  }

  auto model = sb::Model::load(dir);
  check(model.has_value(), "load: Model::load succeeds");
  if (!model) {
    std::cerr << "load error: " << model.error().message << "\n";
    return 1;
  }

  // Single input [4,2] float32 → single output [4,1] float32.
  check(model->input_types().size() == 1 && model->output_types().size() == 1,
        "load: one input and one output");
  check(model->input_types()[0].shape.dims == std::vector<int64_t>({4, 2}),
        "load: input shape is [4,2]");
  check(model->output_types()[0].shape.dims == std::vector<int64_t>({4, 1}),
        "load: output shape is [4,1]");

  auto r = model->run({pack(XOR_IN)});
  check(r.has_value(), "run: forward execution succeeds");
  if (!r) {
    std::cerr << "run error: " << r.error().message << "\n";
    return 1;
  }
  const std::vector<float> got = unpack(r->at(0));
  check(got.size() == 4, "run: 4 outputs produced");

  // The Guile trainer wrote the trained model's predictions here.
  std::vector<float> want(4, 0.0f);
  {
    std::ifstream f(dir + "/expected.bin", std::ios::binary);
    f.read(reinterpret_cast<char*>(want.data()), 4 * sizeof(float));
    check(f.good(), "compare: expected.bin present and readable");
  }

  if (got.size() == 4) {
    for (std::size_t i = 0; i < 4; ++i) {
      bool ok = std::fabs(got[i] - want[i]) <= 1e-2f;
      check(ok, "compare: trained prediction matches deployment (within 1e-2)");
      if (!ok)
        std::cerr << "    idx " << i << ": got " << got[i] << " want " << want[i]
                  << "\n";
    }
  }

  if (g_failures == 0) {
    std::cout << "test_e2e_load OK\n";
    return 0;
  }
  std::cerr << "test_e2e_load FAILED: " << g_failures << " check(s)\n";
  return 1;
}
