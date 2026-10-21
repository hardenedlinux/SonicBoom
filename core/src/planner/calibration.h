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

// Activation-range calibration for the Boom range/error analysis (boom-r4 F2).
//
// With SONICBOOM_BOOM_CALIBRATION set to a path, every f32 tensor the CPU
// SonicBackend produces is sampled for its minimum and maximum, keyed by the
// graph tensor's name — which is the name the Boom IR's tensor table carries,
// so the analysis can anchor its interval lattice at the measured value. The
// file is written at process exit in the facts format RangeFacts::load reads:
//
//   tensor <name> <lo> <hi> 0
//
// The error term is zero: a measured extremum is the value the arithmetic
// actually produced, not an approximation of it.
//
// Unset, record() is a branch on a cached flag; the runtime pays nothing.

#pragma once

#include <span>

namespace sonicboom::planner::calibration {

// True when a calibration run is configured in this process.
bool enabled();

// Fold one executed tensor's values into the sink. A no-op unless enabled().
void record(const char *name, std::span<const float> values);

// Write the accumulated facts to `path` now and clear them, so a tool can merge
// them in the same process. Returns false when the file cannot be written.
bool flush(const char *path);

} // namespace sonicboom::planner::calibration
