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

// test_tensor.cpp — M2 test E: the Layer 2 Tensor boundary.
//
// Allocates an nt::Tensor via the public nt::empty factory and inspects it with
// the public accessors. Only <sonicboom/...> is included, so no TensorImpl /
// StorageImpl / intrusive_ptr (or any c10/ATen type) is in scope.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>

int main() {
  using namespace nt;

  const Tensor t = empty({2, 3}, ScalarType::Float);
  assert(t.defined());

  assert(t.dim() == 2);
  assert(t.size(0) == 2);
  assert(t.size(1) == 3);
  assert(t.numel() == 6);

  const std::vector<int64_t> sizes = t.sizes();
  assert(sizes.size() == 2);
  assert(sizes[0] == 2);
  assert(sizes[1] == 3);

  assert(t.dtype() == ScalarType::Float);
  assert(t.device() == Device::cpu());
  assert(t.layout() == Layout::Strided);
  assert(t.data_ptr() != nullptr);

  // A default-constructed handle is an undefined tensor.
  const Tensor undef;
  assert(!undef.defined());

  std::cout << "M2 test_tensor OK\n";
  return 0;
}
