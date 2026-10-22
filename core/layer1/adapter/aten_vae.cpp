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

// Direct native-torch CUDA kernel calls for the VAE decoder.

#include "aten_vae.h"

#include <cstdio>
#include <cstdlib>

#if defined(SONICBOOM_USE_CUDA)

#include <ATen/TensorIterator.h>
#include <ATen/native/Activation.h>
#include <ATen/native/cuda/Copy.h>
#include <ATen/native/group_norm.h>
#include <ATen/ops/_slow_conv2d_forward_native.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/bmm_native.h>
#include <ATen/ops/from_blob.h>

#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>

#endif

namespace sonicboom::vae::detail {

#if defined(SONICBOOM_USE_CUDA)

namespace {

at::Tensor wrap_f32(VaeDevBuf ptr, at::IntArrayRef sizes) {
  return at::from_blob(
      reinterpret_cast<void*>(ptr), sizes,
      at::TensorOptions().dtype(at::kFloat).device(
          at::kCUDA, c10::cuda::current_device()));
}

at::Tensor wrap_f32(VaeDevBuf ptr, at::IntArrayRef sizes,
                    at::IntArrayRef strides) {
  return at::from_blob(
      reinterpret_cast<void*>(ptr), sizes, strides,
      at::TensorOptions().dtype(at::kFloat).device(
          at::kCUDA, c10::cuda::current_device()));
}

// The structured `_out` kernels derive their set_output target from this
// object; the adapter refuses any shape the caller did not allocate.
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
    TORCH_CHECK(options.dtype() == out_.dtype(), "aten_vae: out dtype mismatch");
    TORCH_CHECK(options.device() == out_.device(),
                "aten_vae: out device mismatch");
    TORCH_CHECK(out_.sizes().equals(sizes), "aten_vae: out shape mismatch");
  }

  at::Tensor& out_;
};

} // namespace

bool aten_vae_conv2d(VaeDevBuf out, VaeDevBuf x, VaeDevBuf w, VaeDevBuf bias,
                     std::int64_t N, std::int64_t C, std::int64_t H,
                     std::int64_t W, std::int64_t O, std::int64_t KH,
                     std::int64_t KW, std::int64_t SH, std::int64_t SW,
                     std::int64_t PH, std::int64_t PW, std::int64_t OH,
                     std::int64_t OW) {
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || O <= 0 || KH <= 0 || KW <= 0)
    return false;
  try {
    const at::Tensor tx = wrap_f32(x, {N, C, H, W});
    const at::Tensor tw = wrap_f32(w, {O, C, KH, KW});
    at::Tensor to = wrap_f32(out, {N, O, OH, OW});
    std::optional<at::Tensor> tb;
    if (bias != nullptr) tb = wrap_f32(bias, {O});
    at::native::slow_conv2d_forward_out_cuda(tx, tw, {KH, KW}, tb, {SH, SW},
                                             {PH, PW}, to);
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] conv2d: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_group_norm(VaeDevBuf out, VaeDevBuf mean, VaeDevBuf rstd,
                         VaeDevBuf x, VaeDevBuf gamma, VaeDevBuf beta,
                         std::int64_t N, std::int64_t C, std::int64_t HxW,
                         std::int64_t groups, double eps) {
  if (N <= 0 || C <= 0 || HxW <= 0 || groups <= 0 || C % groups != 0)
    return false;
  try {
    const at::Tensor tx = wrap_f32(x, {N, C, HxW});
    const at::Tensor tg = wrap_f32(gamma, {C});
    const at::Tensor tb = wrap_f32(beta, {C});
    at::Tensor ty = wrap_f32(out, {N, C, HxW});
    at::Tensor tm = wrap_f32(mean, {N, groups});
    at::Tensor tr = wrap_f32(rstd, {N, groups});
    at::native::GroupNormKernel(at::kCUDA, tx, tg, tb, N, C, HxW, groups, eps,
                                ty, tm, tr);
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] group_norm: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_silu(VaeDevBuf out, VaeDevBuf x, std::int64_t n) {
  if (n <= 0) return false;
  try {
    const at::Tensor tx = wrap_f32(x, {n});
    at::Tensor to = wrap_f32(out, {n});
    auto iter = at::TensorIteratorConfig().add_output(to).add_input(tx).build();
    at::native::silu_stub(at::kCUDA, iter);
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] silu: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_add_row(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b,
                      std::int64_t rows, std::int64_t cols) {
  if (rows <= 0 || cols <= 0) return false;
  try {
    const at::Tensor ta = wrap_f32(a, {rows, cols});
    const at::Tensor tb = wrap_f32(b, {rows, cols}, {0, 1});
    at::Tensor to = wrap_f32(out, {rows, cols});
    DirectOut<at::native::structured_ufunc_add_CUDA> st(to);
    st.meta(ta, tb, 1);
    st.impl(ta, tb, 1, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] add: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_add(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b, std::int64_t n) {
  if (n <= 0) return false;
  try {
    const at::Tensor ta = wrap_f32(a, {n});
    const at::Tensor tb = wrap_f32(b, {n});
    at::Tensor to = wrap_f32(out, {n});
    DirectOut<at::native::structured_ufunc_add_CUDA> st(to);
    st.meta(ta, tb, 1);
    st.impl(ta, tb, 1, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] add_elem: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_transpose(VaeDevBuf out, VaeDevBuf src, std::int64_t c,
                        std::int64_t s) {
  if (c <= 0 || s <= 0) return false;
  try {
    // A transposed view plus the strided copy kernel. The dispatcher's
    // copy_ fast path is skipped: its memcpy eligibility check reads a
    // from_blob buffer linearly, which is not the mapping this transpose needs.
    const at::Tensor ts = wrap_f32(src, {c, s});
    const at::Tensor view = ts.transpose(0, 1);
    at::Tensor to = wrap_f32(out, {s, c});
    auto iter = at::TensorIteratorConfig().add_output(to).add_input(view).build();
    at::native::direct_copy_kernel_cuda(iter);
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] transpose: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_upsample2x(VaeDevBuf out, VaeDevBuf x, std::int64_t N,
                         std::int64_t C, std::int64_t H, std::int64_t W) {
  if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return false;
  try {
    const at::Tensor tx = wrap_f32(x, {N, C, H, W});
    at::Tensor to = wrap_f32(out, {N, C, 2 * H, 2 * W});
    DirectOut<at::native::structured__upsample_nearest_exact2d_out_cuda> st(to);
    st.meta(tx, {2 * H, 2 * W}, std::nullopt, std::nullopt);
    st.impl(tx, {2 * H, 2 * W}, std::nullopt, std::nullopt,
            st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] upsample: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_bmm(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b, std::int64_t B,
                  std::int64_t M, std::int64_t K, std::int64_t N) {
  if (B <= 0 || M <= 0 || K <= 0 || N <= 0) return false;
  try {
    const at::Tensor ta = wrap_f32(a, {B, M, K});
    const at::Tensor tb = wrap_f32(b, {B, K, N});
    at::Tensor to = wrap_f32(out, {B, M, N});
    DirectOut<at::native::structured_bmm_out_cuda> st(to);
    st.meta(ta, tb);
    st.impl(ta, tb, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] bmm: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_bmm_nt(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b, std::int64_t B,
                     std::int64_t M, std::int64_t K, std::int64_t N) {
  if (B <= 0 || M <= 0 || K <= 0 || N <= 0) return false;
  try {
    const at::Tensor ta = wrap_f32(a, {B, M, K});
    // b is a [B, N, K] row-major buffer used as an [B, K, N] transposed view.
    const at::Tensor tb = wrap_f32(b, {B, K, N}, {N * K, 1, K});
    at::Tensor to = wrap_f32(out, {B, M, N});
    DirectOut<at::native::structured_bmm_out_cuda> st(to);
    st.meta(ta, tb);
    st.impl(ta, tb, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] bmm_nt: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_vae_softmax(VaeDevBuf out, VaeDevBuf x, std::int64_t rows,
                      std::int64_t cols) {
  if (rows <= 0 || cols <= 0) return false;
  try {
    const at::Tensor tx = wrap_f32(x, {rows, cols});
    at::Tensor to = wrap_f32(out, {rows, cols});
    DirectOut<at::native::structured_softmax_cuda_out> st(to);
    st.meta(tx, -1, false);
    st.impl(tx, -1, false, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_VAE_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-vae] softmax: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

#else // !SONICBOOM_USE_CUDA

bool aten_vae_conv2d(VaeDevBuf, VaeDevBuf, VaeDevBuf, VaeDevBuf, std::int64_t,
                     std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                     std::int64_t, std::int64_t, std::int64_t, std::int64_t,
                     std::int64_t, std::int64_t, std::int64_t, std::int64_t) {
  return false;
}
bool aten_vae_group_norm(VaeDevBuf, VaeDevBuf, VaeDevBuf, VaeDevBuf, VaeDevBuf,
                         VaeDevBuf, std::int64_t, std::int64_t, std::int64_t,
                         std::int64_t, double) {
  return false;
}
bool aten_vae_silu(VaeDevBuf, VaeDevBuf, std::int64_t) { return false; }
bool aten_vae_add_row(VaeDevBuf, VaeDevBuf, VaeDevBuf, std::int64_t,
                      std::int64_t) {
  return false;
}
bool aten_vae_add(VaeDevBuf, VaeDevBuf, VaeDevBuf, std::int64_t) {
  return false;
}
bool aten_vae_transpose(VaeDevBuf, VaeDevBuf, std::int64_t, std::int64_t) {
  return false;
}
bool aten_vae_upsample2x(VaeDevBuf, VaeDevBuf, std::int64_t, std::int64_t,
                         std::int64_t, std::int64_t) {
  return false;
}
bool aten_vae_bmm(VaeDevBuf, VaeDevBuf, VaeDevBuf, std::int64_t, std::int64_t,
                  std::int64_t, std::int64_t) {
  return false;
}
bool aten_vae_bmm_nt(VaeDevBuf, VaeDevBuf, VaeDevBuf, std::int64_t,
                     std::int64_t, std::int64_t, std::int64_t) {
  return false;
}
bool aten_vae_softmax(VaeDevBuf, VaeDevBuf, std::int64_t, std::int64_t) {
  return false;
}

#endif

} // namespace sonicboom::vae::detail
