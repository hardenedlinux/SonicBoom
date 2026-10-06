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

// test_autograd.cpp — Step 2.3: minimal reverse-mode autograd over nt::Tensor.
//
// Three checks, all against real native execution:
//   1. linear + mse gradient check (finite differences) over w, b, and x —
//      proves the VJPs are the actual derivatives, not fabricated values.
//   2. relu gradient check over a mixed positive/zero/negative input — proves
//      the mask gate.
//   3. a deterministic 2-layer MLP (relu hidden) trained on XOR by plain SGD:
//      loss must drop substantially over the run, proving forward + backward +
//      in-place parameter update form a working training loop.

#include <sonicboom/autograd.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/tensor.h>

#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>

namespace {

using sonicboom::Tape;

bool near(float a, float b, float eps = 1e-4f) {
  return std::fabs(a - b) <= eps;
}

void fill(nt::Tensor& t, const std::vector<float>& v) {
  assert(static_cast<size_t>(t.numel()) == v.size());
  std::memcpy(t.data_ptr(), v.data(), v.size() * sizeof(float));
}

float scalar_of(const nt::Tensor& t) {
  assert(t.numel() == 1);
  return *static_cast<const float*>(t.data_ptr());
}

nt::Tensor copy_tensor(const nt::Tensor& t) {
  nt::Tensor c = nt::empty(t.sizes(), t.dtype());
  std::memcpy(c.data_ptr(), t.data_ptr(),
              static_cast<size_t>(t.numel()) * sizeof(float));
  return c;
}

// Forward through a fresh tape, returning the scalar loss (no retained state).
float loss_of(const nt::Tensor& x, const nt::Tensor& w, const nt::Tensor& b,
              const nt::Tensor& target) {
  Tape tape;
  nt::Tensor y = sonicboom::linear(x, w, b, tape);
  nt::Tensor loss = sonicboom::mse_loss(y, target, tape);
  return scalar_of(loss);
}

void check_grad(const char* name, const nt::Tensor& param,
                const nt::Tensor& analytic,
                const std::function<float(const nt::Tensor&)>& fn) {
  const float* ga = static_cast<const float*>(analytic.data_ptr());
  const float eps = 1e-2f;
  for (int64_t i = 0; i < param.numel(); ++i) {
    nt::Tensor up = copy_tensor(param);
    nt::Tensor down = copy_tensor(param);
    static_cast<float*>(up.data_ptr())[i] += eps;
    static_cast<float*>(down.data_ptr())[i] -= eps;
    float numerical = (fn(up) - fn(down)) / (2.0f * eps);
    if (!near(numerical, ga[i], 5e-3f)) {
      std::cerr << "  gradient mismatch (" << name << "[" << i << "]): numerical "
                << numerical << " vs analytic " << ga[i] << "\n";
      std::abort();
    }
  }
  std::cout << "  " << name << " gradient OK (" << param.numel()
            << " element(s))\n";
}

} // namespace

int main() {
  using sonicboom::backward;
  using sonicboom::linear;
  using sonicboom::mse_loss;
  using sonicboom::relu;
  using sonicboom::sgd_step;

  // --- 1. linear + mse gradient check --------------------------------------
  {
    nt::Tensor x = nt::empty({3, 2}, nt::ScalarType::Float);
    nt::Tensor w = nt::empty({1, 2}, nt::ScalarType::Float);
    nt::Tensor b = nt::empty({1}, nt::ScalarType::Float);
    nt::Tensor tgt = nt::empty({3, 1}, nt::ScalarType::Float);
    fill(x, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    fill(w, {0.7f, -0.3f});
    fill(b, {0.2f});
    fill(tgt, {1.0f, 0.0f, 1.0f});

    Tape tape;
    nt::Tensor y = linear(x, w, b, tape);
    nt::Tensor loss = mse_loss(y, tgt, tape);
    backward(loss, tape);

    // loss is a function of w, b, x only (target fixed); perturb each.
    check_grad("w", w, tape.grad(w), [&](const nt::Tensor& p) {
      return loss_of(x, p, b, tgt);
    });
    check_grad("b", b, tape.grad(b), [&](const nt::Tensor& p) {
      return loss_of(x, w, p, tgt);
    });
    check_grad("x", x, tape.grad(x), [&](const nt::Tensor& p) {
      return loss_of(p, w, b, tgt);
    });
  }

  // --- 2. relu gradient check ----------------------------------------------
  {
    nt::Tensor x = nt::empty({4}, nt::ScalarType::Float);
    nt::Tensor tgt = nt::empty({4}, nt::ScalarType::Float);
    fill(x, {-2.0f, -1.0f, 0.0f, 3.0f});
    fill(tgt, {0.0f, 0.0f, 0.0f, 0.0f});

    Tape tape;
    nt::Tensor y = relu(x, tape);
    nt::Tensor loss = mse_loss(y, tgt, tape);
    backward(loss, tape);

    check_grad("relu-x", x, tape.grad(x), [&](const nt::Tensor& p) {
      Tape t;
      nt::Tensor yy = relu(p, t);
      return scalar_of(mse_loss(yy, tgt, t));
    });
  }

  // --- 3. XOR training loop (deterministic toy MLP) ------------------------
  {
    // 2-layer MLP: 2 -> 4 (relu) -> 1, trained on XOR with plain SGD + MSE.
    nt::Tensor x = nt::empty({4, 2}, nt::ScalarType::Float);
    nt::Tensor tgt = nt::empty({4, 1}, nt::ScalarType::Float);
    fill(x, {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f});
    fill(tgt, {0.0f, 1.0f, 1.0f, 0.0f});

    nt::Tensor w1 = nt::empty({4, 2}, nt::ScalarType::Float);
    nt::Tensor b1 = nt::empty({4}, nt::ScalarType::Float);
    nt::Tensor w2 = nt::empty({1, 4}, nt::ScalarType::Float);
    nt::Tensor b2 = nt::empty({1}, nt::ScalarType::Float);
    fill(w1, {0.6f, -0.3f, 0.2f, 0.5f, -0.5f, 0.4f, 0.3f, -0.7f});
    fill(b1, {0.1f, -0.1f, 0.2f, -0.2f});
    fill(w2, {0.5f, -0.6f, 0.3f, -0.4f});
    fill(b2, {0.0f});

    auto forward = [&](Tape& tape) {
      nt::Tensor h = relu(linear(x, w1, b1, tape), tape);
      nt::Tensor y = linear(h, w2, b2, tape);
      return mse_loss(y, tgt, tape);
    };

    Tape seed_tape;
    float initial = scalar_of(forward(seed_tape));
    float final = initial;
    const float lr = 0.5f;
    const int epochs = 3000;

    for (int e = 0; e < epochs; ++e) {
      Tape tape;
      nt::Tensor loss = forward(tape);
      backward(loss, tape);
      sgd_step(w1, tape.grad(w1), lr);
      sgd_step(b1, tape.grad(b1), lr);
      sgd_step(w2, tape.grad(w2), lr);
      sgd_step(b2, tape.grad(b2), lr);
      final = scalar_of(loss);
      if (e % 500 == 0)
        std::cout << "  epoch " << e << ": loss " << final << "\n";
    }

    std::cout << "  XOR: initial loss " << initial << " -> final loss " << final
              << "\n";
    assert(final < initial);
    assert(final < 0.05f);  // essentially solved
    std::cout << "  training loop OK\n";
  }

  std::cout << "autograd: 3/3 checks passed (gradients are real; loss decreases)\n";
  return 0;
}
