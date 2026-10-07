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

// Token embedding lookup for the native transformer path (float32).

namespace sonicboom::nn {

// Dequantize the `token_id`-th row of a 2-D quantized embedding matrix into
// `out`. `W` has GGUF shape [embd_dim, vocab]; the row is `embd_dim` contiguous
// floats. This is a thin wrapper over quant::dequantize_row_f32.
//
// Returns false (writing nothing) when `out.size() < W.dims()[0]`, the tensor
// is not 2-D/invalid, or `token_id >= W.dims()[1]`.
bool embedding_f32(const sonicboom::quant::QuantizedTensor& W, uint64_t token_id,
                   std::span<float> out);

// Per-layer embedding combine (the tail of the per-layer gate):
//   out[j] = (proj[j] + ple[j] * ple_scale) * combine_scale
// Gemma 4 fuses the rms_norm'd model projection `proj` with the per-layer token
// embedding row `ple` (scaled by ple_scale = sqrt(per_layer_input)) and rescales
// the sum by combine_scale = 1/sqrt(2). Both scales are parameters so the kernel
// stays a pure elementwise combine.
//
// Returns false (writing nothing) when ple.size() < proj.size() or
// out.size() < proj.size().
bool layer_combine(std::span<const float> proj, std::span<const float> ple,
                   float ple_scale, float combine_scale, std::span<float> out);

} // namespace sonicboom::nn
