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

#include "../adapter/adapter.h"

namespace nt {

RegistrationHandle::RegistrationHandle() = default;
RegistrationHandle::RegistrationHandle(
    std::shared_ptr<detail::RegistrationState> state)
    : state_(std::move(state)) {}
RegistrationHandle::RegistrationHandle(RegistrationHandle&&) noexcept = default;
RegistrationHandle& RegistrationHandle::operator=(RegistrationHandle&&) noexcept =
    default;
RegistrationHandle::~RegistrationHandle() = default;

bool RegistrationHandle::valid() const { return state_ != nullptr; }
void RegistrationHandle::release() { state_.reset(); }

namespace {

// Boxed native kernel bridging nt::KernelFn onto the c10 dispatcher. It is a
// stateful functor (holds the nt-level function pointer), registered through
// BoxedKernel::makeFromFunctor (a runtime callable path — the compile-time
// makeFromBoxedFunction<&func>() path cannot capture state).
struct NtKernel final : c10::OperatorKernel {
  KernelFn fn;
  explicit NtKernel(KernelFn f) : fn(f) {}

  void operator()(const c10::OperatorHandle& /*op*/,
                  c10::DispatchKeySet /*keys*/,
                  c10::Stack* stack) {
    ArgumentList args = detail::from_aten(*stack);
    ResultList results = fn(args);
    *stack = detail::to_aten(results);
  }
};

} // namespace

RegistrationHandle define_operator(const OperatorSchema& schema) {
  auto raii = c10::Dispatcher::singleton().registerDef(
      detail::to_aten(schema), "SonicBoom M2");
  return RegistrationHandle(std::make_shared<detail::RegistrationState>(
      detail::RegistrationState{std::move(raii)}));
}

RegistrationHandle register_kernel(const OperatorName& name,
                                   BackendId backend,
                                   Functionality functionality,
                                   KernelFn kernel) {
  c10::DispatchKey key = detail::to_dispatch_key(backend, functionality);

  auto boxed =
      c10::BoxedKernel::makeFromFunctor(std::make_unique<NtKernel>(kernel));
  auto kf = c10::KernelFunction::makeFromBoxedKernel(std::move(boxed));

  auto raii = c10::Dispatcher::singleton().registerImpl(
      c10::OperatorName(name.name, name.overload_name), key, std::move(kf),
      /*cpp_signature=*/std::nullopt,
      /*inferred_function_schema=*/nullptr, "SonicBoom M2");
  return RegistrationHandle(std::make_shared<detail::RegistrationState>(
      detail::RegistrationState{std::move(raii)}));
}

} // namespace nt
