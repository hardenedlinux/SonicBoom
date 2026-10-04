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

// test_backend.cpp — M2 test D: the Layer 2 BackendId boundary.
//
// Verifies the public BackendId contract behind "highest-priority applicable
// backend wins" (manifest M2 §dispatch). The BackendId → native dispatch-key
// mapping itself lives behind the Layer 1 boundary (core/layer1/adapter/
// backend.cpp) and is exercised end-to-end by test_operator (register_kernel
// under BackendId::cpu() + Functionality::Dense fires the CPU kernel).

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

namespace {

// Highest-priority applicable backend wins. v0 has only CPU, so this is the
// ordering rule expressed on the public type; the adapter maps the winner onto
// the native dispatch representation.
nt::BackendId select(const std::vector<nt::BackendId>& backends) {
  nt::BackendId best = backends.front();
  for (const nt::BackendId b : backends) {
    if (b.priority > best.priority) best = b;
  }
  return best;
}

} // namespace

int main() {
  using namespace nt;

  const BackendId cpu = BackendId::cpu();
  assert(cpu.value == 0);
  assert(cpu.priority == 0);

  const BackendId a{1, 10};
  const BackendId b{2, 20};
  assert(a != b);
  assert((a == BackendId{1, 10})); // equality is by discriminator (value) only

  assert(select({a, b, cpu}) == b);
  assert(select({cpu, a}) == a);

  std::cout << "M2 test_backend OK\n";
  return 0;
}
