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

#pragma once

// Direct native-torch CUDA kernel calls for the MOSS-Image VAE decoder.
//
// The decoder owns device buffers and passes raw pointers plus dimensions. The
// adapter wraps them as non-owning float32 CUDA tensors (at::from_blob) and
// calls the native kernel directly — no dispatcher entry and no host/device
// copy. There is no cuDNN in the native-torch build, so convolution is the
// im2col + GEMM kernel (`slow_conv2d_forward_out_cuda`).
//
// This header is ATen-free: core/src/model/moss_vae.cpp includes it. Only
// core/layer1/adapter/aten_vae.cpp sees native-torch types.
//
// All functions run on the current native-torch CUDA stream (the default
// stream unless the caller installed another guard) and return false on a
// shape/dtype error instead of throwing.

#include <cstdint>

namespace sonicboom::vae::detail {

using VaeDevBuf = float*;

// out[N,O,OH,OW] = conv2d(x[N,C,H,W], w[O,C,KH,KW], bias[O] | null);
// stride (SH,SW), zero padding (PH,PW). A null `bias` means none.
bool aten_vae_conv2d(VaeDevBuf out, VaeDevBuf x, VaeDevBuf w, VaeDevBuf bias,
                     std::int64_t N, std::int64_t C, std::int64_t H,
                     std::int64_t W, std::int64_t O, std::int64_t KH,
                     std::int64_t KW, std::int64_t SH, std::int64_t SW,
                     std::int64_t PH, std::int64_t PW, std::int64_t OH,
                     std::int64_t OW);

// out[N,C,HxW] = group_norm(x[N,C,HxW], groups, eps, gamma[C], beta[C]).
// mean/rstd are scratch buffers of N*groups entries, float32.
bool aten_vae_group_norm(VaeDevBuf out, VaeDevBuf mean, VaeDevBuf rstd,
                         VaeDevBuf x, VaeDevBuf gamma, VaeDevBuf beta,
                         std::int64_t N, std::int64_t C, std::int64_t HxW,
                         std::int64_t groups, double eps);

// out[n] = silu(x[n]) elementwise.
bool aten_vae_silu(VaeDevBuf out, VaeDevBuf x, std::int64_t n);

// out[rows,cols] = a[rows,cols] + b[cols] (b broadcast over rows).
bool aten_vae_add_row(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b,
                      std::int64_t rows, std::int64_t cols);

// out[n] = a[n] + b[n] elementwise.
bool aten_vae_add(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b, std::int64_t n);

// out[s, c] = src[c, s]: transpose of a [C][S] row-major buffer.
bool aten_vae_transpose(VaeDevBuf out, VaeDevBuf src, std::int64_t c,
                        std::int64_t s);

// out[N,C,2H,2W] = nearest-exact x2 upsample of x[N,C,H,W].
bool aten_vae_upsample2x(VaeDevBuf out, VaeDevBuf x, std::int64_t N,
                         std::int64_t C, std::int64_t H, std::int64_t W);

// out[B,M,N] = a[B,M,K] @ b[B,K,N].
bool aten_vae_bmm(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b, std::int64_t B,
                  std::int64_t M, std::int64_t K, std::int64_t N);

// out[B,M,N] = a[B,M,K] @ b[B,N,K]^T (b is [B,N,K], used transposed).
bool aten_vae_bmm_nt(VaeDevBuf out, VaeDevBuf a, VaeDevBuf b, std::int64_t B,
                     std::int64_t M, std::int64_t K, std::int64_t N);

// out[rows,cols] = softmax(x[rows,cols], dim = -1).
bool aten_vae_softmax(VaeDevBuf out, VaeDevBuf x, std::int64_t rows,
                      std::int64_t cols);

} // namespace sonicboom::vae::detail
