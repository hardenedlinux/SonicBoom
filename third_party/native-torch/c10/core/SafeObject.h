#pragma once

#include <c10/core/impl/SubstrateInterpreter.h>
#include <c10/macros/Export.h>
#include <c10/util/object_stub.h>
#include <utility>

namespace c10 {

// This is an safe owning holder for a PyObject, akin to pybind11's
// py::object, with two major differences:
//
//  - It is in c10/core; i.e., you can use this type in contexts where
//    you do not have a libpython dependency
//
//  - It is multi-interpreter safe (ala torchdeploy); when you fetch
//    the underlying PyObject* you are required to specify what the current
//    interpreter context is and we will check that you match it.
//
// It is INVALID to store a reference to a Tensor object in this way;
// you should just use TensorImpl directly in that case!
struct C10_API SafeObject {
  // Steals a reference to data
  SafeObject(PyObject* data, c10::impl::SubstrateInterpreter* pyinterpreter)
      : data_(data), pyinterpreter_(pyinterpreter) {}
  SafeObject(SafeObject&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        pyinterpreter_(other.pyinterpreter_) {}
  // For now it's not used, so we just disallow it.
  SafeObject& operator=(SafeObject&&) = delete;

  SafeObject(SafeObject const& other)
      : data_(other.data_), pyinterpreter_(other.pyinterpreter_) {
    if (data_ != nullptr) {
      (*pyinterpreter_)->incref(data_);
    }
  }

  SafeObject& operator=(SafeObject const& other) {
    if (this == &other) {
      return *this; // Handle self-assignment
    }
    if (other.data_ != nullptr) {
      (*other.pyinterpreter_)->incref(other.data_);
    }
    if (data_ != nullptr) {
      (*pyinterpreter_)->decref(data_);
    }
    data_ = other.data_;
    pyinterpreter_ = other.pyinterpreter_;
    return *this;
  }

  ~SafeObject() {
    if (data_ != nullptr) {
      (*pyinterpreter_)->decref(data_);
    }
  }

  c10::impl::SubstrateInterpreter& pyinterpreter() const {
    return *pyinterpreter_;
  }
  PyObject* ptr(const c10::impl::SubstrateInterpreter* /*interpreter*/) const;

  // stop tracking the current object, and return it
  PyObject* release() {
    auto rv = data_;
    data_ = nullptr;
    return rv;
  }

 private:
  PyObject* data_;
  c10::impl::SubstrateInterpreter* pyinterpreter_;
};

// A newtype wrapper around SafeObject for type safety when a python object
// represents a specific type. Note that `T` is only used as a tag and isn't
// actually used for any true purpose.
template <typename T>
struct SafeObjectT : private SafeObject {
  SafeObjectT(PyObject* data, c10::impl::SubstrateInterpreter* pyinterpreter)
      : SafeObject(data, pyinterpreter) {}
  ~SafeObjectT() = default;
  SafeObjectT(SafeObjectT&& other) noexcept : SafeObject(other) {}
  SafeObjectT(SafeObjectT const&) = delete;
  SafeObjectT& operator=(SafeObjectT const&) = delete;
  SafeObjectT& operator=(SafeObjectT&&) = delete;

  using SafeObject::ptr;
  using SafeObject::pyinterpreter;
  using SafeObject::release;
};

// Like SafeObject, but non-owning.  Good for references to global PyObjects
// that will be leaked on interpreter exit.  You get a copy constructor/assign
// this way.
struct C10_API SafePyHandle {
  SafePyHandle() : data_(nullptr), pyinterpreter_(nullptr) {}
  SafePyHandle(PyObject* data, c10::impl::SubstrateInterpreter* pyinterpreter)
      : data_(data), pyinterpreter_(pyinterpreter) {}

  c10::impl::SubstrateInterpreter& pyinterpreter() const {
    return *pyinterpreter_;
  }
  PyObject* ptr(const c10::impl::SubstrateInterpreter* /*interpreter*/) const;
  void reset() {
    data_ = nullptr;
    pyinterpreter_ = nullptr;
  }
  operator bool() {
    return data_;
  }

 private:
  PyObject* data_;
  c10::impl::SubstrateInterpreter* pyinterpreter_;
};

} // namespace c10
