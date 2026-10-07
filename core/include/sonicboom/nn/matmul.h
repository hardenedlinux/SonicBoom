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

// Dense float32 matrix-vector product for the native transformer path.

namespace sonicboom::nn {

// y = W @ x, where W is a float32 matrix in GGUF order [cols, rows]: `cols` is
// the contiguous (fastest-moving) dimension and `rows` stride by `cols`. `x`
// has length `cols`, `y` has length `rows`.
//
// The dot accumulates in double (ggml's ggml_float) with each product rounded
// to f32 first, matching llama.cpp's scalar ggml_vec_dot_f32; the result is
// cast back to f32 on write. This is the dense twin of quant::matvec_f32, used
// for the f32 gate/proj per-layer weights.
//
// Returns false (writing nothing) when W.size() < cols * y.size() or
// x.size() < cols.
bool matvec_f32(std::span<const float> W, uint64_t cols, std::span<const float> x,
                std::span<float> y);

// bf16 matvec (the per-layer model projection): y = W @ x where W holds raw
// bf16 halves (uint16_t) in the same GGUF [cols, rows] order, and `x` is the
// caller's bf16-rounded f32 activation. Each weight element is widened to f32
// before the product, which accumulates in double (ggml's ggml_float) exactly as
// matvec_f32 does. The result is the *unscaled* dot — the caller applies any
// projection scale (proj_scale) afterward, matching the CUDA bf16 gemv.
//
// Returns false (writing nothing) when W.size() < cols * y.size() or
// x.size() < cols.
bool matvec_bf16(std::span<const uint16_t> W, uint64_t cols,
                 std::span<const float> x, std::span<float> y);

} // namespace sonicboom::nn
