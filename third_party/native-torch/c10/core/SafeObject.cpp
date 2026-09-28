#include <c10/core/SafeObject.h>

namespace c10 {

PyObject* SafeObject::ptr(const c10::impl::SubstrateInterpreter* interpreter) const {
  TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
  return data_;
}

PyObject* SafePyHandle::ptr(const c10::impl::SubstrateInterpreter* interpreter) const {
  TORCH_INTERNAL_ASSERT(interpreter == pyinterpreter_);
  return data_;
}

} // namespace c10
