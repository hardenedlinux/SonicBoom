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

// Internal device entry points of the training module. Device buffers travel as
// the planner device pool's opaque handles (uint64_t); the concrete CUDA
// pointer never leaves the .cu translation unit. Only compiled and called when
// SONICBOOM_USE_CUDA is on.

#include <sonicboom/training/adamw.h>

#include <cstddef>
#include <cstdint>

namespace sonicboom::training::detail {

// In-place AdamW over `n` float32 elements held in device buffers: `param` and
// `grad` are inputs, `m1`/`m2` are the moment buffers (updated in place). Runs
// on the current SonicBoom stream. Returns false on a launch error.
bool adamw_dev(std::uint64_t param, std::uint64_t grad, std::uint64_t m1,
               std::uint64_t m2, std::size_t n, const AdamWConfig& cfg,
               std::uint64_t step);

} // namespace sonicboom::training::detail
