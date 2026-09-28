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
