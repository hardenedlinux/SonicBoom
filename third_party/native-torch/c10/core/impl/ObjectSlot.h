#pragma once

#include <c10/core/impl/SubstrateInterpreter.h>
#include <c10/core/impl/SubstrateInterpreterHooks.h>
#include <c10/util/object_stub.h>
#include <optional>

#include <atomic>

namespace torch::utils {
class PyObjectPreservation;
}

namespace c10::impl {

struct C10_API ObjectSlot {
 public:
  ObjectSlot() : pyobj_(nullptr) {}

  PyObject* load_pyobj() const {
    return pyobj_.load(std::memory_order_acquire);
  }

  void store_pyobj(PyObject* obj) {
    pyobj_.store(obj, std::memory_order_release);
  }

  bool has_unique_reference() const {
    PyObject* pyobj = load_pyobj();
    return pyobj != nullptr && (*getGlobalSubstrateInterpreter())->refcnt(pyobj) == 1;
  }

  void clear() {
    pyobj_.store(nullptr, std::memory_order_relaxed);
  }

  // Helper methods for incref/decref/try_incref of the stored PyObject.
  // Used by intrusive_ptr_target subclasses (TensorImpl, StorageImpl, Node)
  // to implement their virtual pyobject refcount overrides.
  void incref() const noexcept {
    // Because intrusive_ptr incref uses relaxed memory order, we need to
    // do an acquire fence to ensure that the kHasPyObject bit was
    // observed before the load of the PyObject* below.
    // NB: This is a no-op on x86/x86-64
    std::atomic_thread_fence(std::memory_order_acquire);
    PyObject* obj = load_pyobj();
    (*c10::impl::getGlobalSubstrateInterpreter())->incref(obj);
  }

  void decref() const noexcept {
    PyObject* obj = load_pyobj();
    (*c10::impl::getGlobalSubstrateInterpreter())->decref(obj);
  }

  bool try_incref() const noexcept {
    SubstrateInterpreter* interp = c10::impl::getGlobalSubstrateInterpreter();
    if (C10_UNLIKELY(!interp)) {
      return false;
    }
    return (*interp)->try_incref(*this);
  }

 private:
  // The PyObject representing this Tensor or nullptr. Ownership is managed
  // by intrusive_ptr. By the time the ObjectSlot is destroyed, this
  // reference is already dead.
  std::atomic<PyObject*> pyobj_;

  friend class torch::utils::PyObjectPreservation;
};

} // namespace c10::impl
