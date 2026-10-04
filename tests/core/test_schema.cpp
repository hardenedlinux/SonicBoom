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

// test_schema.cpp — M2 test A: the Layer 2 OperatorSchema boundary.
//
// Builds an nt::OperatorSchema, registers it through the Layer 1 adapter, looks
// it up, and round-trips it back to nt::OperatorSchema. This file includes only
// <sonicboom/...>; no c10::FunctionSchema (or any native type) is in scope.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>

int main() {
  using namespace nt;

  OperatorSchema schema("m2::add", /*overload_name=*/"",
                        {Argument("a", ArgKind::Int),
                         Argument("b", ArgKind::Int)},
                        {Argument("0", ArgKind::Int)});

  assert(schema.name() == "m2::add");
  assert(schema.overload_name().empty());
  assert(schema.num_args() == 2);
  assert(schema.num_returns() == 1);
  assert(schema.arguments()[0].name == "a");
  assert(schema.arguments()[0].kind == ArgKind::Int);
  assert(schema.arguments()[1].kind == ArgKind::Int);
  assert(schema.returns()[0].kind == ArgKind::Int);

  // Register the schema in the runtime registry (Layer 1 → native dispatcher).
  auto reg = define_operator(schema);
  assert(reg.valid());

  // Look it back up and round-trip to nt::OperatorSchema.
  OperatorHandle op = find_operator(OperatorName("m2::add"));
  assert(op.defined());
  assert(op.name() == "m2::add");

  OperatorSchema got = op.schema();
  assert(got.name() == "m2::add");
  assert(got.num_args() == 2);
  assert(got.num_returns() == 1);
  assert(got.arguments()[0].name == "a");
  assert(got.arguments()[0].kind == ArgKind::Int);
  assert(got.arguments()[1].kind == ArgKind::Int);
  assert(got.returns()[0].kind == ArgKind::Int);

  std::cout << "M2 test_schema OK\n";
  return 0;
}
