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

// test_native_ops.cpp — G-4: six-category operator coverage through the
// libsonicboom.so native-torch dispatcher.
//
// Each category executes a real `aten::` operator via the Layer 1 adapter
// (OperatorHandle::call → redispatchBoxed(DispatchKeySet(CPU))). A category is
// only marked OK when the kernel runs and its numeric result / shape is
// asserted — not merely because the schema is registered.
//
// Reachability note: the v0 adapter dispatches through the computed dispatch
// table (OperatorEntry::lookup), which resolves both direct CPU kernels and the
// Composite{Implicit,Explicit}Autograd alias keys, so composite-only operators
// (conv2d, to.dtype) are reachable here. The error-path category proves that an
// operator with no CPU/composite kernel (CUDA-only cudnn_convolution) throws a
// catchable TORCH_CHECK_NOT_IMPLEMENTED instead of dereferencing an empty
// backend-fallback kernel.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool near(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

void fill_float(nt::Tensor& t, const std::vector<float>& vals) {
  auto* p = static_cast<float*>(t.data_ptr());
  for (size_t i = 0; i < vals.size(); ++i) {
    p[i] = vals[i];
  }
}

void fill_long(nt::Tensor& t, const std::vector<int64_t>& vals) {
  auto* p = static_cast<int64_t*>(t.data_ptr());
  for (size_t i = 0; i < vals.size(); ++i) {
    p[i] = vals[i];
  }
}

} // namespace

int main() {
  using namespace nt;

  // --- 1. matmul: aten::mm (CPU_0, direct CPU kernel) ---------------------
  {
    Tensor a = empty({2, 3}, ScalarType::Float);
    Tensor b = empty({3, 2}, ScalarType::Float);
    fill_float(a, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fill_float(b, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});

    OperatorHandle mm = find_operator(OperatorName("aten::mm"));
    assert(mm.defined());

    ArgumentList args;
    args.push_back(Value(a));
    args.push_back(Value(b));
    ResultList out = mm.call(args);

    assert(out.size() == 1 && out[0].isTensor());
    Tensor c = out[0].toTensor();
    assert(c.sizes() == std::vector<int64_t>({2, 2}));
    const auto* pc = static_cast<const float*>(c.data_ptr());
    // [[58, 64], [139, 154]]
    assert(near(pc[0], 58.0f) && near(pc[1], 64.0f));
    assert(near(pc[2], 139.0f) && near(pc[3], 154.0f));
    std::cout << "  matmul  OK (aten::mm)\n";
  }

  // --- 2. softmax: aten::_softmax (CPU_1, direct CPU kernel) --------------
  {
    Tensor x = empty({3}, ScalarType::Float);
    fill_float(x, {1.0f, 2.0f, 3.0f});

    OperatorHandle softmax = find_operator(OperatorName("aten::_softmax"));
    assert(softmax.defined());

    ArgumentList args;
    args.push_back(Value(x));
    args.push_back(Value(static_cast<int64_t>(0))); // dim
    args.push_back(Value(false));                    // half_to_float
    ResultList out = softmax.call(args);

    assert(out.size() == 1 && out[0].isTensor());
    const auto* ps = static_cast<const float*>(out[0].toTensor().data_ptr());
    // softmax([1,2,3]) ≈ [0.0900306, 0.2447285, 0.6652410]
    assert(near(ps[0], 0.0900306f) && near(ps[1], 0.2447285f) &&
           near(ps[2], 0.6652410f));
    assert(near(ps[0] + ps[1] + ps[2], 1.0f));
    std::cout << "  softmax OK (aten::_softmax)\n";
  }

  // --- 3. reduce: aten::max (CPU_3, direct CPU kernel) --------------------
  {
    Tensor x = empty({4}, ScalarType::Float);
    fill_float(x, {-3.0f, 5.0f, 2.0f, -1.0f});

    OperatorHandle max = find_operator(OperatorName("aten::max"));
    assert(max.defined());

    ArgumentList args;
    args.push_back(Value(x));
    ResultList out = max.call(args);

    assert(out.size() == 1 && out[0].isTensor());
    Tensor r = out[0].toTensor();
    assert(r.numel() == 1);
    assert(near(*static_cast<const float*>(r.data_ptr()), 5.0f));
    std::cout << "  reduce  OK (aten::max)\n";
  }

  // --- 4. index: aten::index_select (CPU_0, direct CPU kernel) ------------
  {
    Tensor x = empty({2, 3}, ScalarType::Float);
    fill_float(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    Tensor idx = empty({2}, ScalarType::Long);
    fill_long(idx, {1, 0});

    OperatorHandle sel = find_operator(OperatorName("aten::index_select"));
    assert(sel.defined());

    ArgumentList args;
    args.push_back(Value(x));
    args.push_back(Value(static_cast<int64_t>(0))); // dim
    args.push_back(Value(idx));
    ResultList out = sel.call(args);

    assert(out.size() == 1 && out[0].isTensor());
    Tensor r = out[0].toTensor();
    assert(r.sizes() == std::vector<int64_t>({2, 3}));
    const auto* pr = static_cast<const float*>(r.data_ptr());
    // rows [1,0] -> [[4,5,6],[1,2,3]]
    assert(pr[0] == 4.0f && pr[1] == 5.0f && pr[2] == 6.0f);
    assert(pr[3] == 1.0f && pr[4] == 2.0f && pr[5] == 3.0f);
    std::cout << "  index   OK (aten::index_select)\n";
  }

  // --- 5. conv: aten::conv2d (composite + SymInt[2] list args) -------------
  {
    // conv2d is CompositeImplicitAutograd with SymInt[2] stride/padding/dilation
    // and SymInt groups. The full boxed path resolves the composite kernel, and
    // the int↔SymInt IValue decay (toSymIntList/toSymInt accept plain ints) lets
    // us box those args as IntList / int — no SymInt representation needed.
    Tensor inp = empty({1, 1, 3, 3}, ScalarType::Float);
    fill_float(inp, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
    Tensor w = empty({1, 1, 2, 2}, ScalarType::Float);
    fill_float(w, {1.0f, 2.0f, 3.0f, 4.0f});

    OperatorHandle conv = find_operator(OperatorName("aten::conv2d"));
    assert(conv.defined());

    // Schema introspection must handle the SymInt[]/SymInt args (Phase C): they
    // map to IntList / Int, not throw.
    OperatorSchema s = conv.schema();
    assert(s.num_args() == 7);
    assert(s.arguments()[3].kind == ArgKind::IntList); // stride
    assert(s.arguments()[6].kind == ArgKind::Int);      // groups

    ArgumentList args;
    args.push_back(Value(inp));
    args.push_back(Value(w));
    args.push_back(Value());                           // bias = None
    args.push_back(Value(std::vector<int64_t>{1, 1})); // stride
    args.push_back(Value(std::vector<int64_t>{0, 0})); // padding
    args.push_back(Value(std::vector<int64_t>{1, 1})); // dilation
    args.push_back(Value(static_cast<int64_t>(1)));    // groups
    ResultList out = conv.call(args);

    assert(out.size() == 1 && out[0].isTensor());
    Tensor y = out[0].toTensor();
    assert(y.sizes() == std::vector<int64_t>({1, 1, 2, 2}));
    const auto* py = static_cast<const float*>(y.data_ptr());
    // 3x3 [[1,2,3],[4,5,6],[7,8,9]] ⊛ 2x2 [[1,2],[3,4]] = [[37,47],[67,77]]
    assert(near(py[0], 37.0f) && near(py[1], 47.0f));
    assert(near(py[2], 67.0f) && near(py[3], 77.0f));
    std::cout << "  conv    OK (aten::conv2d)\n";
  }

  // --- 6. cast: aten::to.dtype (composite, resolved via the full path) -----
  {
    // to.dtype is CompositeImplicitAutograd (no direct CPU kernel). The
    // computed dispatch table (OperatorEntry::lookup, reached via
    // redispatchBoxed) resolves it through the composite alias key to the math
    // kernel, which internally dispatches the dtype cast to the CPU copy
    // kernel. Under the old callBoxedForDispatchKey(CPU) this segfaulted.
    Tensor x = empty({3}, ScalarType::Float);
    fill_float(x, {1.0f, 2.0f, 3.0f});

    OperatorHandle to = find_operator(OperatorName("aten::to", "dtype"));
    assert(to.defined());

    ArgumentList args;
    args.push_back(Value(x));
    args.push_back(Value(ScalarType::Double)); // dtype
    args.push_back(Value(false));              // non_blocking
    args.push_back(Value(false));              // copy
    args.push_back(Value());                   // memory_format = None
    ResultList out = to.call(args);

    assert(out.size() == 1 && out[0].isTensor());
    Tensor y = out[0].toTensor();
    assert(y.dtype() == ScalarType::Double);
    const auto* py = static_cast<const double*>(y.data_ptr());
    assert(py[0] == 1.0 && py[1] == 2.0 && py[2] == 3.0);
    std::cout << "  cast    OK (aten::to.dtype: Float -> Double)\n";
  }

  // --- 7. error path: no kernel must throw, not segfault -------------------
  {
    // cudnn_convolution is CUDA-only (native_functions.yaml: dispatch: CUDA).
    // CUDA is not compiled in this build, so the op has a registered schema but
    // no CPU or composite kernel. Dispatching it with CPU tensors must fail
    // through lookup → reportError → TORCH_CHECK_NOT_IMPLEMENTED (a catchable
    // exception), proving the old empty-backend-fallback-kernel segfault is gone.
    OperatorHandle cudnn =
        find_operator(OperatorName("aten::cudnn_convolution"));
    assert(cudnn.defined());

    Tensor t = empty({1, 1, 2, 2}, ScalarType::Float);
    fill_float(t, {1.0f, 2.0f, 3.0f, 4.0f});

    ArgumentList args;
    args.push_back(Value(t));                          // self
    args.push_back(Value(t));                          // weight
    args.push_back(Value(std::vector<int64_t>{0, 0})); // padding
    args.push_back(Value(std::vector<int64_t>{1, 1})); // stride
    args.push_back(Value(std::vector<int64_t>{1, 1})); // dilation
    args.push_back(Value(static_cast<int64_t>(1)));    // groups
    args.push_back(Value(false));                      // benchmark
    args.push_back(Value(false));                      // deterministic
    args.push_back(Value(false));                      // allow_tf32

    bool caught = false;
    try {
      cudnn.call(args);
    } catch (const std::exception&) {
      caught = true;
    }
    assert(caught);
    std::cout << "  errpath OK (no-kernel dispatch throws, no segfault)\n";
  }

  std::cout << "native-ops: 6/6 categories executed and verified\n";
  return 0;
}
