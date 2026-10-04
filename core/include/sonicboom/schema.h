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

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <sonicboom/argument.h>

namespace nt {

// Layer 2 operator schema. Does not expose c10::FunctionSchema; the adapter
// (core/layer1/adapter/schema.cpp) converts in both directions.
class OperatorSchema {
 public:
  OperatorSchema() = default;
  OperatorSchema(std::string name,
                 std::string overload_name,
                 std::vector<Argument> arguments,
                 std::vector<Argument> returns)
      : name_(std::move(name)),
        overload_name_(std::move(overload_name)),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)) {}

  const std::string& name() const { return name_; }
  const std::string& overload_name() const { return overload_name_; }
  const std::vector<Argument>& arguments() const { return arguments_; }
  const std::vector<Argument>& returns() const { return returns_; }
  size_t num_args() const { return arguments_.size(); }
  size_t num_returns() const { return returns_.size(); }

 private:
  std::string name_;
  std::string overload_name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
};

} // namespace nt
