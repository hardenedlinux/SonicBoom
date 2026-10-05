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

// test_native_dispatch.cpp — G-3: prove libsonicboom.so actually exposes and
// can run the native-torch operator set through the Layer 1 adapter.
//
// This is distinct from test_operator.cpp (M2), which registers a SonicBoom-
// owned kernel via define_operator/register_kernel. Here we find a REAL
// `aten::` operator — one whose registration lives in the generated
// Register{Schema,CPU,...}*.cpp static initializers inside libaten_core.a —
// and boxed-dispatch it through OperatorHandle::call. If those initializers
// were dropped at link time (the G-1 whole-archive gap), find_operator throws.
//
// Passing this test is stronger evidence than an `nm` symbol count: it proves
// the operator is found by name AND its kernel runs and returns a correct
// value.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  using namespace nt;

  // --- 1. Registration breadth (find-by-name must succeed) ----------------
  // Each of these is registered by a generated Register*.cpp static
  // initializer. find_operator throws if the schema is not present.
  const OperatorHandle add = find_operator(OperatorName("aten::add", "Tensor"));
  const OperatorHandle mul = find_operator(OperatorName("aten::mul", "Tensor"));
  const OperatorHandle relu = find_operator(OperatorName("aten::relu"));
  assert(add.defined());
  assert(mul.defined());
  assert(relu.defined());
  assert(add.name() == "aten::add");
  assert(relu.name() == "aten::relu");

  // --- 2. Real dispatch: aten::add.Tensor(self, other, alpha=1) ----------
  Tensor a = empty({2, 3}, ScalarType::Float);
  Tensor b = empty({2, 3}, ScalarType::Float);
  assert(a.defined() && b.defined());
  assert(a.numel() == 6 && b.numel() == 6);
  assert(a.dtype() == ScalarType::Float);

  auto* pa = static_cast<float*>(a.data_ptr());
  auto* pb = static_cast<float*>(b.data_ptr());
  for (int i = 0; i < 6; ++i) {
    pa[i] = static_cast<float>(i + 1);
    pb[i] = 10.0f * static_cast<float>(i + 1);
  }

  ArgumentList args;
  args.push_back(Value(a));
  args.push_back(Value(b));
  args.push_back(Value(Scalar(1.0))); // alpha

  ResultList out = add.call(args);
  assert(out.size() == 1);
  assert(out[0].isTensor());

  Tensor c = out[0].toTensor();
  assert(c.defined());
  assert(c.sizes() == std::vector<int64_t>({2, 3}));
  assert(c.dtype() == ScalarType::Float);

  const auto* pc = static_cast<const float*>(c.data_ptr());
  for (int i = 0; i < 6; ++i) {
    // 11*(i+1) is exact in float, so == is safe.
    assert(pc[i] == 11.0f * static_cast<float>(i + 1));
  }

  // --- 3. A unary op for good measure: aten::relu(self) -------------------
  Tensor neg = empty({3}, ScalarType::Float);
  auto* pn = static_cast<float*>(neg.data_ptr());
  pn[0] = -2.0f;
  pn[1] = 3.0f;
  pn[2] = -0.0f;

  ArgumentList relu_args;
  relu_args.push_back(Value(neg));
  ResultList relu_out = relu.call(relu_args);
  assert(relu_out.size() == 1 && relu_out[0].isTensor());
  const auto* pr = static_cast<const float*>(relu_out[0].toTensor().data_ptr());
  assert(pr[0] == 0.0f);
  assert(pr[1] == 3.0f);
  assert(pr[2] == 0.0f);

  std::cout << "native-dispatch OK: found aten::add/mul/relu; "
               "aten::add.Tensor + aten::relu dispatched and computed "
               "correctly\n";
  return 0;
}
