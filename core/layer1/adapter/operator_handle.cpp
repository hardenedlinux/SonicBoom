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

OperatorHandle::OperatorHandle() = default;
OperatorHandle::OperatorHandle(std::shared_ptr<detail::OperatorHandleImpl> impl)
    : impl_(std::move(impl)) {}

bool OperatorHandle::defined() const {
  return impl_ && impl_->handle.hasSchema();
}

std::string OperatorHandle::name() const {
  return impl_->handle.schema().name();
}

OperatorSchema OperatorHandle::schema() const {
  return detail::from_aten(impl_->handle.schema());
}

ResultList OperatorHandle::call(ArgumentList args) const {
  c10::Stack stack = detail::to_aten(args);
  // v0 is CPU-only. A tensor input would carry DispatchKey::CPU and the generic
  // backend-selection path would pick it; that path is deferred (backend.cpp).
  impl_->handle.callBoxedForDispatchKey(c10::DispatchKey::CPU, stack);
  return detail::from_aten(std::move(stack));
}

OperatorHandle find_operator(const OperatorName& name) {
  c10::OperatorHandle handle = c10::Dispatcher::singleton().findSchemaOrThrow(
      name.name.c_str(), name.overload_name.c_str());
  return OperatorHandle(std::make_shared<detail::OperatorHandleImpl>(
      detail::OperatorHandleImpl{std::move(handle)}));
}

} // namespace nt
