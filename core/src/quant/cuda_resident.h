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

#include <sonicboom/quant/quantized_tensor.h>

// Internal device-resident K-quant matvec (Phase 6a Option A). Mirrors
// quant::cuda::matvec_f32 but takes a device input and writes a device output
// with no H2D/D2H, so the caller can keep the activation resident on the
// device. W's packed bytes are uploaded once and cached by (data pointer, byte
// size) via the same cache the public path uses.
//
// NOT part of the public Layer 2 surface (lives under core/src/, may use raw
// device pointers). Included only from .cu translation units.

namespace sonicboom::quant::cuda {

// y = W @ x (single-column), device in/out. W is a 2-D QuantizedTensor
// ([cols, rows], cols contiguous). x device (cols floats), y device (rows
// floats, zero-initialized before accumulation). No host copies.
bool matvec_f32_dev(const QuantizedTensor& W, const float* x, float* y);

} // namespace sonicboom::quant::cuda
