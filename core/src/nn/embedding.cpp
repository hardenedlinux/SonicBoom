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

#include <sonicboom/nn/embedding.h>

#include <sonicboom/quant/dequant.h>

namespace sonicboom::nn {

bool embedding_f32(const sonicboom::quant::QuantizedTensor& W, uint64_t token_id,
                   std::span<float> out) {
  if (W.dims().size() != 2) return false;
  if (out.size() < W.dims()[0]) return false;
  return sonicboom::quant::dequantize_row_f32(W, token_id, out.data(), out.size());
}

bool layer_combine(std::span<const float> proj, std::span<const float> ple,
                   float ple_scale, float combine_scale, std::span<float> out) {
  if (ple.size() < proj.size() || out.size() < proj.size()) return false;
  for (size_t i = 0; i < proj.size(); ++i)
    out[i] = (proj[i] + ple[i] * ple_scale) * combine_scale;
  return true;
}

} // namespace sonicboom::nn
