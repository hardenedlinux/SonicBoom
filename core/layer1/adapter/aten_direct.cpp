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

// Direct native-torch CUDA kernel calls for the training executor.

#include "aten_direct.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#if defined(SONICBOOM_USE_CUDA)

#include <ATen/ops/_softmax_meta.h>
#include <ATen/ops/_softmax_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/mul_meta.h>
#include <ATen/ops/mul_native.h>

#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>

#include <ATen/native/Resize.h>

#include <vector>

#endif

namespace sonicboom::training::detail {

namespace {

// SONICBOOM_TRAIN_ATEN, read once: "", "0", "none" → no op; "1"/"all" → every
// op; otherwise a comma-separated list. Whitespace around entries is ignored.
const std::string& aten_selection() {
  static const std::string selection = [] {
    const char* raw = std::getenv("SONICBOOM_TRAIN_ATEN");
    if (!raw) return std::string{};
    std::string s;
    std::string token;
    auto flush = [&] {
      while (!token.empty() && (token.front() == ' ' || token.front() == '\t'))
        token.erase(token.begin());
      while (!token.empty() && (token.back() == ' ' || token.back() == '\t'))
        token.pop_back();
      if (!token.empty()) {
        if (!s.empty()) s += ',';
        s += token;
      }
      token.clear();
    };
    for (const char* p = raw; *p; ++p) {
      if (*p == ',') flush();
      else token += *p;
    }
    flush();
    return s;
  }();
  return selection;
}

} // namespace

bool aten_direct_enabled(const char* op) {
  const std::string& sel = aten_selection();
  if (sel.empty() || sel == "0" || sel == "none") return false;
  if (sel == "1" || sel == "all") return true;
  std::string needle{op};
  std::size_t start = 0;
  while (start <= sel.size()) {
    const std::size_t comma = sel.find(',', start);
    const std::string_view entry(sel.data() + start,
                                 (comma == std::string::npos ? sel.size() : comma) -
                                     start);
    if (entry == needle) return true;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return false;
}

#if defined(SONICBOOM_USE_CUDA)

namespace {

// The executor's device buffers, as non-owning float32 CUDA tensors. from_blob
// allocates only host-side TensorImpl/StorageImpl; no device memory, so the
// call is legal inside CUDA-graph capture.
at::Tensor wrap_f32(AtenDevBuf ptr, at::IntArrayRef sizes,
                    at::IntArrayRef strides, c10::DeviceIndex dev) {
  const auto options = at::TensorOptions().dtype(at::kFloat).device(
      at::kCUDA, dev);
  if (strides.empty())
    return at::from_blob(reinterpret_cast<void*>(ptr), sizes, options);
  return at::from_blob(reinterpret_cast<void*>(ptr), sizes, strides, options);
}

// The out tensor is a non-owning view of a buffer the executor already sized.
// Refuse a shape the kernel disagrees with rather than resize: a resize on a
// from_blob storage would allocate device memory, which CUDA-graph capture
// forbids, and the executor's own kernel is the correct fallback.
void check_out(const at::Tensor& out, at::IntArrayRef sizes,
               const at::TensorOptions& options) {
  TORCH_CHECK(options.dtype() == out.dtype(), "aten_direct: out dtype mismatch");
  TORCH_CHECK(options.device() == out.device(),
              "aten_direct: out device mismatch");
  TORCH_CHECK(out.sizes().equals(sizes), "aten_direct: out shape mismatch");
}

// The structured `_out` kernels are abstract: the generated dispatcher derives
// a concrete class whose set_output_* target the caller's out tensor (the
// copy RegisterCUDA_0.cpp's *_out wrapper makes). This is the same, minus the
// proxy path: check_out rejects anything that would need a resize or proxy.
template <class Base>
class DirectOut final : public Base {
 public:
  explicit DirectOut(at::Tensor& out) : out_(out) {}

  void set_output_strided(std::int64_t, at::IntArrayRef sizes,
                          at::IntArrayRef, at::TensorOptions options) override {
    check_out(out_, sizes, options);
  }
  void set_output_raw_strided(std::int64_t, at::IntArrayRef sizes,
                              at::IntArrayRef,
                              at::TensorOptions options) override {
    check_out(out_, sizes, options);
  }
  const at::Tensor& maybe_get_output(std::int64_t) override { return out_; }

 private:
  at::Tensor& out_;
};

} // namespace

bool aten_direct_softmax(AtenDevBuf out, AtenDevBuf in, std::int64_t rows,
                         std::int64_t cols, AtenDevBuf stream) {
  if (rows <= 0 || cols <= 0) return false;
  try {
    const c10::DeviceIndex dev = c10::cuda::current_device();
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream), dev));
    const std::vector<std::int64_t> dims{rows, cols};
    const at::Tensor x = wrap_f32(in, dims, {}, dev);
    const at::Tensor o = wrap_f32(out, dims, {}, dev);
    at::Tensor out_t = o;
    DirectOut<at::native::structured_softmax_cuda_out> st(out_t);
    st.meta(x, -1, false);
    st.impl(x, -1, false, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_TRAIN_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-direct] softmax fallback: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_direct_mul(AtenDevBuf out, AtenDevBuf a, AtenDevBuf b,
                     const std::int64_t* a_strides,
                     const std::int64_t* b_strides, const std::int64_t* dims,
                     std::int64_t rank, std::int64_t n, AtenDevBuf stream) {
  if (rank <= 0 || n <= 0) return false;
  try {
    const c10::DeviceIndex dev = c10::cuda::current_device();
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream), dev));
    const at::IntArrayRef sizes(dims, static_cast<std::size_t>(rank));
    const at::IntArrayRef sa(a_strides, static_cast<std::size_t>(rank));
    const at::IntArrayRef sb(b_strides, static_cast<std::size_t>(rank));
    const at::Tensor ta = wrap_f32(a, sizes, sa, dev);
    const at::Tensor tb = wrap_f32(b, sizes, sb, dev);
    at::Tensor to = wrap_f32(out, sizes, {}, dev);
    DirectOut<at::native::structured_mul_out> st(to);
    st.meta(ta, tb);
    st.impl(ta, tb, st.maybe_get_output(0));
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_TRAIN_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-direct] mul fallback: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

bool aten_direct_index_select(AtenDevBuf out, AtenDevBuf table, AtenDevBuf ids,
                              std::int64_t rows, std::int64_t table_rows,
                              std::int64_t dim, AtenDevBuf stream) {
  // Above 16 indices native index_select calls gather through the dispatcher;
  // the direct path is the small-index kernel only.
  if (rows <= 0 || rows > 16 || table_rows <= 0 || dim <= 0) return false;
  try {
    const c10::DeviceIndex dev = c10::cuda::current_device();
    const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
        reinterpret_cast<cudaStream_t>(stream), dev));
    const at::Tensor t = wrap_f32(table, {table_rows, dim}, {}, dev);
    const at::Tensor i =
        at::from_blob(reinterpret_cast<void*>(ids), {rows},
                      at::TensorOptions().dtype(at::kInt).device(at::kCUDA, dev));
    at::Tensor o = wrap_f32(out, {rows, dim}, {}, dev);
    at::native::index_select_out_cuda(t, 0, i, o);
    return true;
  } catch (const std::exception& e) {
    if (std::getenv("SONICBOOM_TRAIN_ATEN_DEBUG"))
      std::fprintf(stderr, "[aten-direct] index_select fallback: %s\n", e.what());
    return false;
  } catch (...) {
    return false;
  }
}

#else // !SONICBOOM_USE_CUDA

bool aten_direct_softmax(AtenDevBuf, AtenDevBuf, std::int64_t, std::int64_t,
                         AtenDevBuf) {
  return false;
}

bool aten_direct_mul(AtenDevBuf, AtenDevBuf, AtenDevBuf, const std::int64_t*,
                     const std::int64_t*, const std::int64_t*, std::int64_t,
                     std::int64_t, AtenDevBuf) {
  return false;
}

bool aten_direct_index_select(AtenDevBuf, AtenDevBuf, AtenDevBuf, std::int64_t,
                              std::int64_t, std::int64_t, AtenDevBuf) {
  return false;
}

#endif

} // namespace sonicboom::training::detail
