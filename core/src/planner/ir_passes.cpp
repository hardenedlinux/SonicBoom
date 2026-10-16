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

// The part of the Boom IR switch that is true of every build: reading the
// environment. The pipeline and its availability live with the passes, on the
// compiler side of the boundary (core/compiler/lib/BoomPasses.cpp), or in the
// stand-in compiled when this build has no compiler library
// (core/src/planner/ir_passes_stub.cpp). This file is MLIR-free like the rest
// of the planner.

#include <sonicboom/planner/ir_passes.h>

#include <cstdlib>
#include <string_view>

namespace sonicboom::planner {

bool ir_passes_enabled() {
  if (!ir_passes_available()) return false;
  const char *flag = std::getenv("SONICBOOM_IR_PASSES");
  return flag != nullptr && *flag != '\0' && std::string_view(flag) != "0";
}

} // namespace sonicboom::planner
