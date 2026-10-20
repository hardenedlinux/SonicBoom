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

// test_adamw.cpp — R2: fused AdamW parameter update through the C ABI.
//
// The SonicBoomParam path (sb_param_create / set_grad / step / read) is run for
// 10 steps with a deterministic gradient sequence and compared elementwise
// against an independent double-precision AdamW reference (the numpy formula).
// The gate is max-abs <= 1e-6. A zero_grad step then checks that a zeroed
// gradient produces the pure decoupled-weight-decay update. Error paths
// (mismatched gradient, short read buffer, mismatched init length) must fail
// with SB_ERR_BINDING and leave the parameter untouched.
//
// The parameter lives in device memory when a CUDA device is present and in
// host memory otherwise; both paths execute the same reference check.

#include <sonicboom/capi.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr std::size_t kN = 17; // odd: exercises the grid-stride tail

struct Ctx {
  bool ok = true;
  const char* what = "";
};

void check(Ctx& c, bool cond, const char* what) {
  if (!cond) {
    c.ok = false;
    c.what = what;
  }
}

std::vector<float> init_values() {
  std::vector<float> v(kN);
  for (std::size_t i = 0; i < kN; ++i)
    v[i] = 0.4f * std::sin(0.7f * static_cast<float>(i)) +
           0.05f * static_cast<float>(i % 3);
  return v;
}

// Deterministic gradient sequence: step t, element i. Deliberately non-uniform
// so a bug in the grid-stride indexing or the moment update shows up.
std::vector<float> grad_at(std::size_t t) {
  std::vector<float> g(kN);
  for (std::size_t i = 0; i < kN; ++i)
    g[i] = 0.5f * std::cos(0.31f * static_cast<float>(t) +
                           0.37f * static_cast<float>(i)) *
               (1.0f + 0.05f * static_cast<float>(i)) +
           0.01f * static_cast<float>(t);
  return g;
}

// Independent double-precision AdamW (decoupled weight decay), numpy's formula.
struct Ref {
  std::vector<double> p, g, m, v;

  explicit Ref(const std::vector<float>& init)
      : p(init.begin(), init.end()), g(kN, 0.0), m(kN, 0.0), v(kN, 0.0) {}

  void step(std::size_t t, float lr, float beta1, float beta2, float eps,
            float wd) {
    const double b1 = beta1, b2 = beta2;
    const double bc1 = 1.0 - std::pow(b1, static_cast<double>(t));
    const double bc2 = 1.0 - std::pow(b2, static_cast<double>(t));
    for (std::size_t i = 0; i < kN; ++i) {
      const double gi = g[i];
      m[i] = b1 * m[i] + (1.0 - b1) * gi;
      v[i] = b2 * v[i] + (1.0 - b2) * gi * gi;
      p[i] -= lr * (m[i] / bc1 / (std::sqrt(v[i] / bc2) + eps) + wd * p[i]);
    }
  }
};

float on_device = 0.0f;

void report_name(Ctx& c, sb_error* err, const char* where) {
  if (!err)
    return;
  std::fprintf(stderr, "%s: %s\n", where, sb_error_message(err));
  c.ok = false;
  c.what = where;
  sb_error_free(err);
}

} // namespace

int main() {
  Ctx c;
  sb_error* err = nullptr;

  const float lr = 0.02f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.01f;

  std::vector<float> init = init_values();
  const int64_t dims[1] = {static_cast<int64_t>(kN)};

  sb_param* p = nullptr;
  check(c, sb_param_create(dims, 1, init.data(), init.size() * sizeof(float),
                           &p, &err) == SB_OK,
        "sb_param_create");
  report_name(c, err, "sb_param_create");
  if (!p) {
    std::fprintf(stderr, "FAIL: %s\n", c.what);
    return 1;
  }

  on_device = sb_param_on_device(p) ? 1.0f : 0.0f;
  std::printf("adamw: %s path\n", on_device ? "device" : "host");

  std::size_t numel = 0;
  check(c, sb_param_numel(p, &numel) == SB_OK && numel == kN,
        "sb_param_numel");
  check(c, sb_param_step_count(p) == 0, "initial step_count == 0");

  Ref ref(init);
  float max_abs = 0.0f;

  for (std::size_t t = 1; t <= 10; ++t) {
    const std::vector<float> g = grad_at(t);
    check(c, sb_param_set_grad(p, g.data(), g.size() * sizeof(float), &err) ==
                  SB_OK,
          "sb_param_set_grad");
    report_name(c, err, "sb_param_set_grad");

    check(c, sb_param_step(p, lr, b1, b2, eps, wd, &err) == SB_OK,
          "sb_param_step");
    report_name(c, err, "sb_param_step");

    ref.g.assign(g.begin(), g.end());
    ref.step(t, lr, b1, b2, eps, wd);

    std::vector<float> got(kN);
    check(c, sb_param_read(p, got.data(), got.size() * sizeof(float), nullptr,
                           &err) == SB_OK,
          "sb_param_read");
    report_name(c, err, "sb_param_read");

    for (std::size_t i = 0; i < kN; ++i) {
      const float d = std::fabs(got[i] - static_cast<float>(ref.p[i]));
      if (d > max_abs)
        max_abs = d;
    }
    check(c, sb_param_step_count(p) == t, "step_count advances");
  }

  std::printf("adamw: 10-step max-abs vs double reference = %.3e\n", max_abs);
  check(c, max_abs <= 1e-6f, "10-step max-abs <= 1e-6");

  // zero_grad: a step with a zeroed gradient is pure decoupled weight decay.
  check(c, sb_param_zero_grad(p, &err) == SB_OK, "sb_param_zero_grad");
  report_name(c, err, "sb_param_zero_grad");
  check(c, sb_param_step(p, lr, b1, b2, eps, wd, &err) == SB_OK,
        "sb_param_step after zero_grad");
  report_name(c, err, "sb_param_step after zero_grad");

  ref.g.assign(kN, 0.0);
  ref.step(11, lr, b1, b2, eps, wd);

  std::vector<float> after(kN);
  check(c, sb_param_read(p, after.data(), after.size() * sizeof(float),
                         nullptr, &err) == SB_OK,
        "sb_param_read after zero_grad");
  report_name(c, err, "sb_param_read after zero_grad");
  float zero_grad_err = 0.0f;
  for (std::size_t i = 0; i < kN; ++i)
    zero_grad_err = std::max(
        zero_grad_err, std::fabs(after[i] - static_cast<float>(ref.p[i])));
  std::printf("adamw: zero_grad step max-abs = %.3e\n", zero_grad_err);
  check(c, zero_grad_err <= 1e-6f, "zero_grad step matches reference");

  // Error paths: a mismatched gradient is rejected and must not change the
  // parameter (a subsequent step still matches the reference).
  float before[kN];
  std::memcpy(before, after.data(), sizeof(before));
  const std::vector<float> bad(kN - 1, 0.5f);
  err = nullptr;
  check(c, sb_param_set_grad(p, bad.data(), bad.size() * sizeof(float),
                             &err) == SB_ERR_BINDING,
        "short gradient rejected");
  check(c, err != nullptr, "short gradient reports an error");
  if (err) {
    check(c, sb_error_get_kind(err) == SB_ERRKIND_BINDING,
          "error kind is binding");
    sb_error_free(err);
    err = nullptr;
  }

  check(c, sb_param_read(p, after.data(), (kN - 1) * sizeof(float), nullptr,
                         &err) == SB_ERR_BINDING,
        "short read buffer rejected");
  if (err) {
    sb_error_free(err);
    err = nullptr;
  }

  std::vector<float> unchanged(kN);
  check(c, sb_param_read(p, unchanged.data(), unchanged.size() * sizeof(float),
                         nullptr, &err) == SB_OK,
        "read after rejected calls");
  if (err) {
    sb_error_free(err);
    err = nullptr;
  }
  check(c, std::memcmp(before, unchanged.data(), sizeof(before)) == 0,
        "rejected calls leave the parameter untouched");

  // A mismatched initial buffer never creates a parameter.
  sb_param* bad_param = reinterpret_cast<sb_param*>(0x1);
  check(c, sb_param_create(dims, 1, init.data(), init.size() * sizeof(float) - 4,
                           &bad_param, &err) == SB_ERR_BINDING,
        "mismatched init length rejected");
  check(c, bad_param == nullptr, "failed create leaves out-param null");
  if (err) {
    sb_error_free(err);
    err = nullptr;
  }

  check(c, sb_param_numel(p, nullptr) == SB_ERR_INTERNAL,
        "null numel out-param rejected");

  // lr-input form: sb_param_step_lr reads the rate from a one-element float32
  // tensor. Two fresh parameters run the same gradient sequence, one updated
  // with the scalar form and one with the tensor form, and must agree bit for
  // bit (the same fused kernel and the same float -> double conversion).
  sb_param* a = nullptr;
  sb_param* b = nullptr;
  check(c, sb_param_create(dims, 1, init.data(), init.size() * sizeof(float),
                           &a, &err) == SB_OK,
        "lr-input: create scalar-form parameter");
  report_name(c, err, "lr-input create a");
  check(c, sb_param_create(dims, 1, init.data(), init.size() * sizeof(float),
                           &b, &err) == SB_OK,
        "lr-input: create tensor-form parameter");
  report_name(c, err, "lr-input create b");

  const float schedule[10] = {0.05f, 0.02f, 0.01f, 0.005f, 0.002f,
                              0.01f, 0.04f, 0.03f, 0.015f, 0.0075f};
  std::vector<float> last_a, last_b;
  bool lr_forms_agree = a && b;
  for (std::size_t t = 1; lr_forms_agree && t <= 10; ++t) {
    const std::vector<float> g = grad_at(t);
    const float lr_t = schedule[t - 1];
    check(c, sb_param_set_grad(a, g.data(), g.size() * sizeof(float), &err) ==
                  SB_OK,
          "lr-input: set_grad a");
    report_name(c, err, "lr-input set_grad a");
    check(c, sb_param_set_grad(b, g.data(), g.size() * sizeof(float), &err) ==
                  SB_OK,
          "lr-input: set_grad b");
    report_name(c, err, "lr-input set_grad b");

    check(c, sb_param_step(a, lr_t, b1, b2, eps, wd, &err) == SB_OK,
          "lr-input: scalar step");
    report_name(c, err, "lr-input scalar step");

    const int64_t lr_dims[1] = {1};
    sb_tensor* lr_tensor = nullptr;
    check(c, sb_tensor_from_f32(lr_dims, 1, &lr_t, sizeof(lr_t), &lr_tensor,
                                &err) == SB_OK,
          "lr-input: lr tensor");
    report_name(c, err, "lr-input lr tensor");
    check(c, sb_param_step_lr(b, lr_tensor, b1, b2, eps, wd, &err) == SB_OK,
          "lr-input: tensor step");
    report_name(c, err, "lr-input tensor step");
    sb_tensor_free(lr_tensor);

    last_a.assign(kN, 0.0f);
    last_b.assign(kN, 0.0f);
    check(c, sb_param_read(a, last_a.data(), last_a.size() * sizeof(float),
                           nullptr, &err) == SB_OK,
          "lr-input: read a");
    report_name(c, err, "lr-input read a");
    check(c, sb_param_read(b, last_b.data(), last_b.size() * sizeof(float),
                           nullptr, &err) == SB_OK,
          "lr-input: read b");
    report_name(c, err, "lr-input read b");
    lr_forms_agree = std::memcmp(last_a.data(), last_b.data(),
                                 last_a.size() * sizeof(float)) == 0;
    check(c, lr_forms_agree, "lr-input matches the scalar form every step");
  }
  check(c, sb_param_step_count(a) == 10 && sb_param_step_count(b) == 10,
        "lr-input: both forms advanced 10 steps");

  // A learning-rate tensor with more than one element is rejected before the
  // update, and leaves the step count alone.
  {
    const int64_t two[1] = {2};
    const float vals[2] = {0.1f, 0.2f};
    sb_tensor* lr_bad = nullptr;
    check(c, sb_tensor_from_f32(two, 1, vals, sizeof(vals), &lr_bad, &err) ==
                  SB_OK,
          "lr-input: two-element tensor created");
    report_name(c, err, "lr-input bad tensor");
    const uint64_t before_steps = sb_param_step_count(b);
    check(c, sb_param_step_lr(b, lr_bad, b1, b2, eps, wd, &err) ==
                  SB_ERR_BINDING,
          "lr-input: multi-element lr rejected");
    if (err) {
      sb_error_free(err);
      err = nullptr;
    }
    check(c, sb_param_step_count(b) == before_steps,
          "lr-input: rejected lr leaves the step count unchanged");
    sb_tensor_free(lr_bad);
  }

  sb_param_free(a);
  sb_param_free(b);
  sb_param_free(p);

  if (!c.ok) {
    std::fprintf(stderr, "FAIL: %s\n", c.what);
    return 1;
  }
  std::printf("PASS: adamw (%s)\n", on_device ? "device" : "host");
  return 0;
}
