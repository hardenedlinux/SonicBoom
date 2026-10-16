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

// The Boom IR pipeline in a build without MLIR: the compiler library that
// carries the passes is not built, and the runtime side still names the
// pipeline's entry points — the emitter asks whether the IR path is available
// before it can ask for it, and the answer here is no.
//
// Compiled *instead of* the real thing, exactly one of the two (see
// core/CMakeLists.txt: the planner glob excludes this file and re-adds it only
// when SONICBOOM_ENABLE_MLIR is OFF; the same arrangement as
// sonic_cuda_backend.cpp).

#include <sonicboom/planner/ir_passes.h>

namespace sonicboom::planner {

bool ir_passes_available() { return false; }

// Unreachable: ir_passes_enabled() is false in this build, so nothing asks for
// a pipeline. Present so that the reference in the emitter links — and
// deliberately not a fallback that quietly does nothing useful, which is the
// same reason the real one has none (ir_passes.h).
void ir_pass_pipeline(Graph &, IrPassPhase) {}

} // namespace sonicboom::planner
