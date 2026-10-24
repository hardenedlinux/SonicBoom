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

#include <sonicboom/planner/partition.h>

namespace sonicboom::planner {

const char* backend_tag_name(BackendTag t) noexcept {
  switch (t) {
    case BackendTag::Mlir: return "mlir";
    case BackendTag::NativeTorch: return "native_torch";
    case BackendTag::Sonic: return "sonic";
  }
  return "?";
}

} // namespace sonicboom::planner
