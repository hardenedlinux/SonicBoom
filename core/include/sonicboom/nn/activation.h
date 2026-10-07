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

#include <span>

// Elementwise activation kernels for the native transformer path (float32).
// Pure C++23 reference arithmetic — no device backend, no ggml, no native-torch.

namespace sonicboom::nn {

// SiLU (a.k.a. swish): y[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])).
// Provided for completeness (Gemma 4's gated FFN uses GELU, not SiLU).
//
// Requires y.size() >= x.size(); returns false (writing nothing) otherwise.
bool silu(std::span<const float> x, std::span<float> y);

// GELU (tanh approximation, a.k.a. gelu_pytorch_tanh / gelu_new):
// y[i] = 0.5 * x[i] * (1 + tanh(sqrt(2/pi) * (x[i] + 0.044715 * x[i]^3))).
// This is the mathematically accurate reference.
bool gelu(std::span<const float> x, std::span<float> y);

// GELU (tanh) with llama.cpp's GGML_GELU_FP16 fp16-table semantics: the input is
// rounded to fp16 before the tanh-GELU, and the result is rounded back to fp16
// (with clamping: x <= -10 -> 0, x >= 10 -> x). llama.cpp's CPU GELU/GeGLU
// kernels use this fp16 lookup-table path (ggml_vec_gelu_f32 / ggml_vec_geglu_f32
// under GGML_GELU_FP16), which quantizes both the input and the result to fp16.
// Gemma 4's FFN and per-layer gates are computed through it; use gelu_fp16 to make
// run_block directly comparable to a llama.cpp baseline. The fp16 quantization
// introduces ~1e-3 relative error versus the accurate gelu() above.
bool gelu_fp16(std::span<const float> x, std::span<float> y);

} // namespace sonicboom::nn
