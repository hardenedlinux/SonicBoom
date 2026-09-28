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
