#include <ATen/core/OpRegistrationTrampoline.h>

namespace at::impl {

std::atomic<c10::impl::SubstrateInterpreter*>
    OpRegistrationTrampoline::interpreter_{nullptr};

c10::impl::SubstrateInterpreter* OpRegistrationTrampoline::getInterpreter() {
  return OpRegistrationTrampoline::interpreter_.load();
}

bool OpRegistrationTrampoline::registerInterpreter(
    c10::impl::SubstrateInterpreter* interp) {
  c10::impl::SubstrateInterpreter* expected = nullptr;
  return interpreter_.compare_exchange_strong(expected, interp);
}

} // namespace at::impl
