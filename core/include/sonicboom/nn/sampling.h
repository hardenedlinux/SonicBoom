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

// Logit softcapping and greedy token selection for the decode loop (float32).

namespace sonicboom::nn {

// Gemma final-logit softcapping (final_logit_softcapping = 30 for Gemma 4):
//   y[i] = cap * tanh(x[i] / cap)
// Requires cap > 0 and y.size() >= x.size(); returns false otherwise.
bool softcap(std::span<const float> x, float cap, std::span<float> y);

// Index of the maximum element (ties broken toward the lowest index). Returns
// -1 for an empty span. This is the greedy (temperature-0) token selection.
int64_t argmax(std::span<const float> x);

} // namespace sonicboom::nn
