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

// MLIR infrastructure smoke test: links against libsonicboom.so (which has the
// required MLIR libraries statically linked in) and verifies that an empty
// module can be created and printed.
#include "sonicboom/mlir.h"

#include <iostream>

int main() {
  const std::string text = sonicboom::mlir_emit_empty_module();
  std::cout << text;
  return text.find("module") == std::string::npos ? 1 : 0;
}
