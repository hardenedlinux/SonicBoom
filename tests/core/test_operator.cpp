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

// test_operator.cpp — M2 test C: end-to-end boxed operator dispatch.
//
// define_operator + register_kernel + OperatorHandle::call, going through the
// Layer 1 adapter onto the native dispatcher and back. The kernel operates on
// nt::ArgumentList / nt::ResultList only; no c10::Stack is in scope here.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

namespace {

nt::ResultList double_it(const nt::ArgumentList& args) {
  const double x = args[0].toFloat();
  return nt::ResultList(std::vector<nt::Value>{nt::Value(x * 2.0)});
}

} // namespace

int main() {
  using namespace nt;

  OperatorSchema schema("m2::scale", /*overload_name=*/"",
                        {Argument("x", ArgKind::Float)},
                        {Argument("0", ArgKind::Float)});

  auto def = define_operator(schema);
  assert(def.valid());

  auto impl = register_kernel(OperatorName("m2::scale"), BackendId::cpu(),
                              Functionality::Dense, &double_it);
  assert(impl.valid());

  OperatorHandle op = find_operator(OperatorName("m2::scale"));
  assert(op.defined());

  ArgumentList args;
  args.push_back(Value(3.0));

  ResultList out = op.call(args);
  assert(out.size() == 1);
  assert(out[0].isFloat());
  assert(out[0].toFloat() == 6.0);

  std::cout << "M2 test_operator OK\n";
  return 0;
}
