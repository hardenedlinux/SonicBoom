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

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <sonicboom/schema.h>
#include <sonicboom/argument_list.h>

namespace nt {

namespace detail {
struct OperatorHandleImpl; // defined in the Layer 1 adapter; holds the native
                           // operator handle
} // namespace detail

struct OperatorName {
  std::string name;
  std::string overload_name;

  OperatorName() = default;
  OperatorName(std::string n, std::string o = "")
      : name(std::move(n)), overload_name(std::move(o)) {}
};

// Layer 2 operator handle: identity + schema access + boxed invocation. Does
// not expose c10::OperatorHandle.
class OperatorHandle {
 public:
  OperatorHandle(); // null handle
  explicit OperatorHandle(std::shared_ptr<detail::OperatorHandleImpl> impl);

  bool defined() const;

  std::string name() const;
  OperatorSchema schema() const;

  // Invoke the operator on the given arguments (boxed). v0 dispatches to the
  // CPU backend; returns the operator's results.
  ResultList call(ArgumentList args) const;

 private:
  std::shared_ptr<detail::OperatorHandleImpl> impl_;
};

// Look up a registered operator by name. Throws if not registered (v0).
OperatorHandle find_operator(const OperatorName& name);

} // namespace nt
