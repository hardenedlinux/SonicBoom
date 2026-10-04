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

#include "adapter.h"

namespace nt {
namespace detail {

c10::DispatchKey to_dispatch_key(BackendId backend,
                                 Functionality functionality) {
  // v0: single backend (CPU), so "highest-priority applicable backend wins" is
  // trivially satisfied. The adapter maps (backend, functionality) onto the
  // native dispatch representation.
  TORCH_CHECK(backend == BackendId::cpu(), "v0 supports the CPU backend only");
  switch (functionality) {
    case Functionality::Dense:
      return c10::DispatchKey::CPU;
    case Functionality::Sparse:
      return c10::DispatchKey::Sparse;
    case Functionality::Quantized:
      return c10::DispatchKey::QuantizedCPU;
    case Functionality::Autograd:
      return c10::DispatchKey::Autograd;
  }
  TORCH_CHECK(false, "unsupported nt::Functionality");
  return c10::DispatchKey::CPU; // unreachable
}

} // namespace detail
} // namespace nt
