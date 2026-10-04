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

// test_value.cpp — M2 test B: the Layer 2 Value boundary.
//
// Verifies nt::Value is a self-contained std::variant over Native Torch-owned
// types (no c10::IValue in scope) with correct kind()/is*()/to*() accessors.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  using namespace nt;

  Value none;
  assert(none.isNone());
  assert(none.kind() == ValueKind::None);

  Value i(42);
  assert(i.isInt());
  assert(i.kind() == ValueKind::Int);
  assert(i.toInt() == 42);

  Value f(3.5);
  assert(f.isFloat());
  assert(f.kind() == ValueKind::Float);
  assert(f.toFloat() == 3.5);

  Value b(true);
  assert(b.isBool());
  assert(b.kind() == ValueKind::Bool);
  assert(b.toBool());

  Value s(std::string("hello"));
  assert(s.isString());
  assert(s.kind() == ValueKind::String);
  assert(s.toString() == "hello");

  Value d(Device::cpu());
  assert(d.isDevice());
  assert(d.kind() == ValueKind::Device);
  assert(d.toDevice() == Device::cpu());

  Value st(ScalarType::Float);
  assert(st.isScalarType());
  assert(st.kind() == ValueKind::ScalarType);
  assert(st.toScalarType() == ScalarType::Float);

  Value l(Layout::Strided);
  assert(l.isLayout());
  assert(l.kind() == ValueKind::Layout);
  assert(l.toLayout() == Layout::Strided);

  Value mf(MemoryFormat::Contiguous);
  assert(mf.isMemoryFormat());
  assert(mf.kind() == ValueKind::MemoryFormat);
  assert(mf.toMemoryFormat() == MemoryFormat::Contiguous);

  Value tl(std::vector<Tensor>{});
  assert(tl.isTensorList());
  assert(tl.kind() == ValueKind::TensorList);

  Value sc(Scalar(7));
  assert(sc.isScalar());
  assert(sc.kind() == ValueKind::Scalar);
  assert(sc.toScalar().isInt());
  assert(sc.toScalar().toInt() == 7);

  std::cout << "M2 test_value OK\n";
  return 0;
}
