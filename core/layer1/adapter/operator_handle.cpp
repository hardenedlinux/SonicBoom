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

#include <ATen/core/dispatch/DispatchKeyExtractor.h>

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
  // Tensor-driven dispatch: derive the dispatch key set from the tensor
  // arguments, then resolve the kernel through redispatchBoxed →
  // OperatorEntry::lookup — the computed dispatch table, including the
  // Composite{Implicit,Explicit}Autograd alias keys and backend fallback.
  //
  // We must use the operator's OWN registered DispatchKeyExtractor (via
  // OperatorHandle::dispatchKeyExtractor()), NOT a fresh one built from the
  // schema: makeFallthrough() has already called setOperatorHasFallthroughForKey
  // on it, masking BackendSelect / ADInplaceOrView / Autograd* out of the
  // derived key set. A fresh extractor leaves those fallthrough keys in, so
  // lookup() would return fallthrough_kernel and redispatchBoxed (which, unlike
  // callBoxed, does not short-circuit fallthrough) would execute it → throw.
  //
  // For scalar-only operators (e.g. the M2 m2::scale test) the tensor-derived
  // key set is empty; an empty key set would resolve to Undefined under lookup,
  // so fall back to an explicit CPU key set (a CPU-registered kernel is
  // unreachable otherwise). A missing kernel throws a clean
  // TORCH_CHECK_NOT_IMPLEMENTED via lookup → reportError, never a null call
  // target.
  c10::DispatchKeySet keys =
      impl_->handle.dispatchKeyExtractor().getDispatchKeySetBoxed(&stack);
  if (keys.empty()) {
    keys = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  impl_->handle.redispatchBoxed(keys, &stack);
  return detail::from_aten(std::move(stack));
}

OperatorHandle find_operator(const OperatorName& name) {
  c10::OperatorHandle handle = c10::Dispatcher::singleton().findSchemaOrThrow(
      name.name.c_str(), name.overload_name.c_str());
  return OperatorHandle(std::make_shared<detail::OperatorHandleImpl>(
      detail::OperatorHandleImpl{std::move(handle)}));
}

} // namespace nt
