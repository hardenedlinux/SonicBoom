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

#include <cstdint>
#include <span>

#include <sonicboom/quant/quantized_tensor.h>

// CUDA backend for the quantized matvec (dequantize-on-the-fly). This is a
// SonicBoom-owned kernel: the packed K-quant weight is uploaded to the device
// once and cached, and each single-column matvec streams it through a
// dequantize-and-dot gemv kernel. It reproduces the arithmetic of the scalar
// f32-activation reference <sonicboom/quant/quantized_matmul.h> matvec_f32
// (f32 accumulation order differs on the device, so agreement is to fp32
// tolerance, not bit-exact).
//
// The declarations here use only Layer 2 types (QuantizedTensor, span) — no
// CUDA, c10 or native-torch types appear in the public header. The CUDA
// runtime and kernels live entirely in core/src/quant/cuda_quant_matmul.cu.
// These entry points are only linked when the build is configured with
// SONICBOOM_USE_CUDA=ON.

namespace sonicboom::quant::cuda {

// True when a usable CUDA device is present (cached after the first call).
bool available() noexcept;

// y = W @ x for a single-column activation, computed on the device. W's packed
// bytes are uploaded on first use and cached keyed by (data pointer, byte
// size), so repeated calls with the same weight reuse the device copy. x and y
// are host spans; the function uploads x, launches the gemv kernel, and copies
// y back.
//
// Returns false (writing nothing to y) when CUDA is unavailable, W is invalid
// or not 2-D, x/y are too short, or a device allocation/copy/kernel launch
// fails. Never reads past W.data() or x/y.
bool matvec_f32(const QuantizedTensor& W, std::span<const float> x,
                std::span<float> y);

// Drop all cached device weight copies (call when a model is reloaded at a new
// address so stale device buffers are not reused).
void clear_cache();

} // namespace sonicboom::quant::cuda
