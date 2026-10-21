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

// Direct native-torch CPU kernel calls for the CPU Boom executor.

#include "aten_cpu.h"

#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/max_pool2d_with_indices_native.h>
#include <ATen/ops/mm_native.h>

#include <c10/core/TensorOptions.h>

#include <array>
#include <optional>
#include <vector>

namespace sonicboom::cpu::detail {

namespace {

// A host buffer as a non-owning CPU float32 tensor.
at::Tensor wrap_f32(const float* p, at::IntArrayRef sizes) {
  return at::from_blob(const_cast<float*>(p), sizes,
                       at::TensorOptions().dtype(at::kFloat));
}

at::Tensor wrap_f32(float* p, at::IntArrayRef sizes) {
  return at::from_blob(p, sizes, at::TensorOptions().dtype(at::kFloat));
}

at::Tensor wrap_i64(std::int64_t* p, at::IntArrayRef sizes) {
  return at::from_blob(p, sizes, at::TensorOptions().dtype(at::kLong));
}

// The structured `_out` kernels are abstract: the generated dispatcher derives a
// concrete class whose set_output_* target the caller's out tensor. This is the
// same, minus the proxy path: a shape or dtype the caller did not allocate is a
// compile-time (graph) error here, not a resize.
template <class Base>
class DirectOut final : public Base {
 public:
  explicit DirectOut(at::Tensor& out) : out_(out) {}

  void set_output_strided(std::int64_t, at::IntArrayRef sizes,
                          at::IntArrayRef, at::TensorOptions options) override {
    check(sizes, options);
  }
  void set_output_raw_strided(std::int64_t, at::IntArrayRef sizes,
                              at::IntArrayRef,
                              at::TensorOptions options) override {
    check(sizes, options);
  }
  const at::Tensor& maybe_get_output(std::int64_t) override { return out_; }

 private:
  void check(at::IntArrayRef sizes, const at::TensorOptions& options) {
    TORCH_CHECK(options.dtype() == out_.dtype(), "aten_cpu: out dtype mismatch");
    TORCH_CHECK(options.device() == out_.device(),
                "aten_cpu: out device mismatch");
    TORCH_CHECK(out_.sizes().equals(sizes), "aten_cpu: out shape mismatch");
  }

  at::Tensor& out_;
};

// Two-output variant (max_pool2d_with_indices).
template <class Base>
class DirectOut2 final : public Base {
 public:
  DirectOut2(at::Tensor& out0, at::Tensor& out1) : out_{out0, out1} {}

  void set_output_strided(std::int64_t i, at::IntArrayRef sizes,
                          at::IntArrayRef, at::TensorOptions options) override {
    check(i, sizes, options);
  }
  void set_output_raw_strided(std::int64_t i, at::IntArrayRef sizes,
                              at::IntArrayRef,
                              at::TensorOptions options) override {
    check(i, sizes, options);
  }
  const at::Tensor& maybe_get_output(std::int64_t i) override {
    return out_[static_cast<std::size_t>(i)];
  }

 private:
  void check(std::int64_t i, at::IntArrayRef sizes,
             const at::TensorOptions& options) {
    at::Tensor& out = out_[static_cast<std::size_t>(i)];
    TORCH_CHECK(options.dtype() == out.dtype(), "aten_cpu: out dtype mismatch");
    TORCH_CHECK(options.device() == out.device(),
                "aten_cpu: out device mismatch");
    TORCH_CHECK(out.sizes().equals(sizes), "aten_cpu: out shape mismatch");
  }

  std::array<at::Tensor, 2> out_;
};

} // namespace

bool aten_cpu_mm(float* out, const float* a, const float* b, std::int64_t M,
                 std::int64_t K, std::int64_t N) {
  if (M <= 0 || K <= 0 || N <= 0) return false;
  try {
    const at::Tensor ta = wrap_f32(a, {M, K});
    const at::Tensor tb = wrap_f32(b, {K, N});
    at::Tensor to = wrap_f32(out, {M, N});
    DirectOut<at::native::structured_mm_out_cpu> st(to);
    st.meta(ta, tb);
    st.impl(ta, tb, st.maybe_get_output(0));
    return true;
  } catch (...) {
    return false;
  }
}

bool aten_cpu_bmm(float* out, const float* a, const float* b,
                  std::int64_t batch, std::int64_t M, std::int64_t K,
                  std::int64_t N) {
  if (batch <= 0 || M <= 0 || K <= 0 || N <= 0) return false;
  try {
    const at::Tensor ta = wrap_f32(a, {batch, M, K});
    const at::Tensor tb = wrap_f32(b, {batch, K, N});
    at::Tensor to = wrap_f32(out, {batch, M, N});
    DirectOut<at::native::structured_bmm_out_cpu> st(to);
    st.meta(ta, tb);
    st.impl(ta, tb, st.maybe_get_output(0));
    return true;
  } catch (...) {
    return false;
  }
}

bool aten_cpu_conv2d(float* out, const float* x, const float* w,
                     const float* bias, std::int64_t N, std::int64_t C,
                     std::int64_t H, std::int64_t W, std::int64_t O,
                     std::int64_t KH, std::int64_t KW, std::int64_t SH,
                     std::int64_t SW, std::int64_t OH, std::int64_t OW) {
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || O <= 0 || KH <= 0 || KW <= 0)
    return false;
  try {
    const at::Tensor tx = wrap_f32(x, {N, C, H, W});
    const at::Tensor tw = wrap_f32(w, {O, C, KH, KW});
    at::Tensor to = wrap_f32(out, {N, O, OH, OW});
    std::optional<at::Tensor> tb;
    if (bias != nullptr) tb = wrap_f32(bias, {O});
    at::native::slow_conv2d_forward_out_cpu(tx, tw, {KH, KW}, tb, {SH, SW},
                                            {0, 0}, to);
    return true;
  } catch (...) {
    return false;
  }
}

bool aten_cpu_max_pool2d(float* out, std::int64_t* indices, const float* x,
                         std::int64_t N, std::int64_t C, std::int64_t H,
                         std::int64_t W, std::int64_t KH, std::int64_t KW,
                         std::int64_t SH, std::int64_t SW, std::int64_t OH,
                         std::int64_t OW) {
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || KH <= 0 || KW <= 0 ||
      indices == nullptr)
    return false;
  try {
    const at::Tensor tx = wrap_f32(x, {N, C, H, W});
    at::Tensor to = wrap_f32(out, {N, C, OH, OW});
    at::Tensor ti = wrap_i64(indices, {N, C, OH, OW});
    DirectOut2<at::native::structured_max_pool2d_with_indices_out_cpu> st(to,
                                                                         ti);
    st.meta(tx, {KH, KW}, {SH, SW}, {0, 0}, {1, 1}, false);
    st.impl(tx, {KH, KW}, {SH, SW}, {0, 0}, {1, 1}, false,
            st.maybe_get_output(0), st.maybe_get_output(1));
    return true;
  } catch (...) {
    return false;
  }
}

} // namespace sonicboom::cpu::detail
