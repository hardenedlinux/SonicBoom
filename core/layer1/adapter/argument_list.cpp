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

c10::Stack to_aten(const ArgumentList& args) {
  c10::Stack stack;
  stack.reserve(args.size());
  for (const auto& v : args.values()) {
    stack.push_back(to_aten(v));
  }
  return stack;
}

ArgumentList from_aten(const c10::Stack &stack) {
  std::vector<Value> values;
  values.reserve(stack.size());
  for (const auto& iv : stack) {
    values.push_back(from_aten(iv));
  }
  return ArgumentList(std::move(values));
}

} // namespace detail
} // namespace nt
