// numeric_proof.cpp — SonicBoom Stage D numeric-correctness smoke test.
//
// Where dispatch_proof exercises only the dispatcher mechanism, this test goes
// through the real Tensor facade and native CPU kernels on concrete tensors:
//
//     factory (ones / full)  ->  binary (add / mul)
//         ->  unary (relu, dispatch-stub path)  ->  reduction (sum)
//
// and asserts concrete numeric results. It proves the aten_core build is not
// merely linkable but actually computes correctly end-to-end through the
// dispatcher, schema registration, and per-DispatchKey kernel registration.

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

bool close(float a, float b, float tol = 1e-5f) {
  return std::fabs(a - b) <= tol;
}

} // namespace

int main() {
  auto opts = at::TensorOptions().dtype(at::kFloat);

  // --- binary add: 1 + 2 = 3 everywhere ---
  at::Tensor a = at::ones({2, 3}, opts);
  at::Tensor b = at::full({2, 3}, 2.0f, opts);
  at::Tensor c = at::add(a, b);
  assert(c.numel() == 6);
  const float* cdata = c.data_ptr<float>();
  for (int64_t i = 0; i < c.numel(); ++i) {
    assert(close(cdata[i], 3.0f));
  }

  // --- binary mul: 3 * 3 = 9 everywhere ---
  at::Tensor d = at::mul(c, c);
  const float* ddata = d.data_ptr<float>();
  for (int64_t i = 0; i < d.numel(); ++i) {
    assert(close(ddata[i], 9.0f));
  }

  // --- unary relu (dispatch-stub path): negatives -> 0, positives -> keep ---
  at::Tensor neg = at::full({3}, -2.0f, opts);
  at::Tensor relu_neg = at::relu(neg);
  const float* negdata = relu_neg.data_ptr<float>();
  for (int64_t i = 0; i < relu_neg.numel(); ++i) {
    assert(close(negdata[i], 0.0f));
  }

  at::Tensor pos = at::full({3}, 2.0f, opts);
  at::Tensor relu_pos = at::relu(pos);
  const float* posdata = relu_pos.data_ptr<float>();
  for (int64_t i = 0; i < relu_pos.numel(); ++i) {
    assert(close(posdata[i], 2.0f));
  }

  // --- reduction sum: all-3.0 2x3 = 18 ---
  at::Tensor s = c.sum();
  assert(s.numel() == 1);
  assert(close(s.item<float>(), 18.0f));

  std::cout << "numeric proof OK: ones/full -> add/mul/relu -> sum all correct\n";
  return 0;
}
