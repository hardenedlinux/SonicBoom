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

// test_cuda.cpp — real GPU closed loop (RTX 3050, sm_86). Only built when
// SONICBOOM_USE_CUDA is on; skips cleanly at runtime when no CUDA device is
// present (so a CPU-only machine never goes red). The loop proves, end to end:
//   (1) to_device() moves a tensor CPU→CUDA and back without data loss,
//   (2) a tensor-driven dispatch reaches a real CUDA kernel (not a stub),
//   (3) the kernel result is read back to the CPU and numerically asserted.
// The kernels chosen (aten::add.Tensor, aten::mul.Tensor) are direct dense CUDA
// kernels, not CompositeImplicitAutograd aliases, so executing them proves the
// dense CUDA kernel layer is actually callable through the Layer 1 adapter.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <cmath>
#include <cstdint>
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

} // namespace

int main() {
  using namespace nt;

  // No GPU at runtime (or a CPU-only build): skip, never fail.
  if (!cuda_available()) {
    std::cout << "CUDA unavailable; skipping GPU loop\n";
    return 0;
  }

  // --- 1. Host ↔ device round-trip (factory + to_device) ------------------
  Tensor a = empty({4}, ScalarType::Float);
  fill_float(a, {1.0f, 2.0f, 3.0f, 4.0f});

  Tensor a_cuda = to_device(a, Device::cuda(0));
  assert(a_cuda.defined());
  assert(a_cuda.device().type == DeviceType::CUDA);
  assert(a_cuda.device().index == 0);
  assert(a_cuda.sizes() == std::vector<int64_t>({4}));

  Tensor a_back = to_device(a_cuda, Device::cpu());
  const auto* pa = static_cast<const float*>(a_back.data_ptr());
  assert(pa[0] == 1.0f && pa[1] == 2.0f && pa[2] == 3.0f && pa[3] == 4.0f);
  std::cout << "  to_device CPU→CUDA→CPU OK\n";

  // --- 2. Real CUDA kernel: aten::add.Tensor(self, other, alpha=1) --------
  Tensor b = empty({4}, ScalarType::Float);
  fill_float(b, {10.0f, 20.0f, 30.0f, 40.0f});
  Tensor b_cuda = to_device(b, Device::cuda(0));

  OperatorHandle add = find_operator(OperatorName("aten::add", "Tensor"));
  assert(add.defined());

  ArgumentList args;
  args.push_back(Value(a_cuda));
  args.push_back(Value(b_cuda));
  args.push_back(Value(Scalar(1.0))); // alpha

  ResultList out = add.call(args);
  assert(out.size() == 1 && out[0].isTensor());
  Tensor c = out[0].toTensor();
  assert(c.device().type == DeviceType::CUDA); // result stays on device

  Tensor c_cpu = to_device(c, Device::cpu());
  const auto* pc = static_cast<const float*>(c_cpu.data_ptr());
  assert(near(pc[0], 11.0f) && near(pc[1], 22.0f));
  assert(near(pc[2], 33.0f) && near(pc[3], 44.0f));
  std::cout << "  aten::add.Tensor CUDA kernel OK\n";

  // --- 3. A second independent kernel: aten::mul.Tensor(self, other) ------
  OperatorHandle mul = find_operator(OperatorName("aten::mul", "Tensor"));
  assert(mul.defined());

  ArgumentList mul_args;
  mul_args.push_back(Value(a_cuda));
  mul_args.push_back(Value(b_cuda));

  ResultList mul_out = mul.call(mul_args);
  assert(mul_out.size() == 1 && mul_out[0].isTensor());
  Tensor d = to_device(mul_out[0].toTensor(), Device::cpu());
  const auto* pd = static_cast<const float*>(d.data_ptr());
  // [1*10, 2*20, 3*30, 4*40]
  assert(near(pd[0], 10.0f) && near(pd[1], 40.0f));
  assert(near(pd[2], 90.0f) && near(pd[3], 160.0f));
  std::cout << "  aten::mul.Tensor CUDA kernel OK\n";

  // --- 4. Error path: an excluded operator must throw, not crash -----------
  {
    // cudnn_convolution is CUDA-only and backed by cuDNN (AT_CUDNN_ENABLED=0),
    // so v0 supplies a throwing stub (native/cuda/ExcludedKernelsStubs.cpp)
    // with the upstream signature. A controlled dispatch on CUDA tensors must
    // raise a catchable NotImplementedError with a readable message — not a
    // link failure, null dispatch, or segfault.
    OperatorHandle cudnn =
        find_operator(OperatorName("aten::cudnn_convolution"));
    assert(cudnn.defined());

    Tensor self4 = empty({1, 1, 2, 2}, ScalarType::Float);
    Tensor self4_cuda = to_device(self4, Device::cuda(0));
    Tensor w4 = empty({1, 1, 2, 2}, ScalarType::Float);
    Tensor w4_cuda = to_device(w4, Device::cuda(0));

    ArgumentList cargs;
    cargs.push_back(Value(self4_cuda));                      // self
    cargs.push_back(Value(w4_cuda));                         // weight
    cargs.push_back(Value(std::vector<int64_t>{0, 0}));      // padding
    cargs.push_back(Value(std::vector<int64_t>{1, 1}));      // stride
    cargs.push_back(Value(std::vector<int64_t>{1, 1}));      // dilation
    cargs.push_back(Value(static_cast<int64_t>(1)));         // groups
    cargs.push_back(Value(false));                           // benchmark
    cargs.push_back(Value(false));                           // deterministic
    cargs.push_back(Value(false));                           // allow_tf32

    bool caught = false;
    std::string msg;
    try {
      cudnn.call(cargs);
    } catch (const std::exception& e) {
      caught = true;
      msg = e.what();
    }
    assert(caught);
    assert(!msg.empty());
    // The stub names the operator and the exclusion reason.
    assert(msg.find("cudnn_convolution") != std::string::npos);
    std::cout << "  excluded-op errpath OK: " << msg << "\n";
  }

  std::cout << "CUDA loop OK: real GPU kernels executed and read back\n";
  return 0;
}
