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

#include <sonicboom/backend.h>
#include <sonicboom/schema.h>
#include <sonicboom/operator_handle.h>
#include <sonicboom/argument_list.h>

namespace nt {

namespace detail {
struct RegistrationState; // defined in the Layer 1 adapter; holds the native
                          // registration RAII
} // namespace detail

// A generic interpreter kernel: takes the operator's arguments and produces its
// results. The adapter bridges this to the boxed native dispatcher; typed /
// unboxed kernels can be layered on top later (manifest M2).
using KernelFn = ResultList (*)(const ArgumentList&);

// RAII handle for an operator or kernel registration. Empty until a
// registration succeeds; deregisters on destruction or on release().
class RegistrationHandle {
 public:
  RegistrationHandle();
  explicit RegistrationHandle(std::shared_ptr<detail::RegistrationState> state);
  RegistrationHandle(RegistrationHandle&&) noexcept;
  RegistrationHandle& operator=(RegistrationHandle&&) noexcept;
  ~RegistrationHandle();

  RegistrationHandle(const RegistrationHandle&) = delete;
  RegistrationHandle& operator=(const RegistrationHandle&) = delete;

  bool valid() const;
  void release();

 private:
  std::shared_ptr<detail::RegistrationState> state_;
};

// Register an operator schema in the runtime registry.
RegistrationHandle define_operator(const OperatorSchema& schema);

// Register a kernel for an operator under (backend, functionality).
RegistrationHandle register_kernel(const OperatorName& name,
                                   BackendId backend,
                                   Functionality functionality,
                                   KernelFn kernel);

} // namespace nt
