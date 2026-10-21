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

// Direct calls into the native-torch CPU kernels from the CPU Boom executor.
//
// The executor owns host buffers and passes raw pointers plus dimensions. The
// adapter wraps them as non-owning CPU tensors (at::from_blob) and calls the
// native-torch `_out` kernel directly — no dispatcher entry, no allocation, no
// copy. Every op here is one native-torch has a CPU kernel for; the executor
// uses its own kernels (or the existing SonicBoom CPU kernels) everywhere else.
//
// This header is ATen-free: the executor (core/src/exec) includes it. Only
// core/layer1/adapter/aten_cpu.cpp sees native-torch types.

#include <cstdint>

namespace sonicboom::cpu::detail {

// out[M,N] = a[M,K] @ b[K,N], all float32 row-major.
bool aten_cpu_mm(float* out, const float* a, const float* b, std::int64_t M,
                 std::int64_t K, std::int64_t N);

// out[B,M,N] = a[B,M,K] @ b[B,K,N], all float32 row-major.
bool aten_cpu_bmm(float* out, const float* a, const float* b,
                  std::int64_t batch, std::int64_t M, std::int64_t K,
                  std::int64_t N);

// NCHW float32 convolution, OIHW weight, dilation 1, groups 1, no padding —
// the caller pads by the ONNX pads first. `bias` may be null (no bias).
bool aten_cpu_conv2d(float* out, const float* x, const float* w,
                     const float* bias, std::int64_t N, std::int64_t C,
                     std::int64_t H, std::int64_t W, std::int64_t O,
                     std::int64_t KH, std::int64_t KW, std::int64_t SH,
                     std::int64_t SW, std::int64_t OH, std::int64_t OW);

// NCHW float32 max pooling, dilation 1, no padding, ceil_mode false — the
// caller pads with -inf first. `indices` is scratch of OH*OW*N*C int64 entries
// (the kernel returns them; the executor does not use them).
bool aten_cpu_max_pool2d(float* out, std::int64_t* indices, const float* x,
                         std::int64_t N, std::int64_t C, std::int64_t H,
                         std::int64_t W, std::int64_t KH, std::int64_t KW,
                         std::int64_t SH, std::int64_t SW, std::int64_t OH,
                         std::int64_t OW);

} // namespace sonicboom::cpu::detail
