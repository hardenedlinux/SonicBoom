#pragma once

// Layer 1 adapter internal header. NOT public — included only by adapter .cpp
// files under core/layer1/. This is the ONLY place where native-torch
// (c10/ATen) types meet the Native Torch Layer 2 (nt::) types.

#include <sonicboom/sonicboom.h>

#include <ATen/core/TensorBody.h>
#include <ATen/core/boxing/BoxedKernel.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/stack.h>

#include <c10/core/Allocator.h>
#include <c10/core/CPUAllocator.h>
#include <c10/core/Device.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <c10/core/Storage.h>
#include <c10/core/SymInt.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>

namespace nt {
namespace detail {

// --- opaque impls referenced by the public headers ---

struct TensorImpl {
  at::Tensor t;
};

struct OperatorHandleImpl {
  c10::OperatorHandle handle;
};

struct RegistrationState {
  c10::RegistrationHandleRAII raii;
  explicit RegistrationState(c10::RegistrationHandleRAII&& r)
      : raii(std::move(r)) {}
};

// --- friend access point for Tensor (declared friend in tensor.h) ---
struct Adapter {
  static const std::shared_ptr<TensorImpl>& impl_of(const Tensor& t) {
    return t.impl_;
  }
  static Tensor from_impl(std::shared_ptr<TensorImpl> i) {
    return Tensor(std::move(i));
  }
};

// --- scalar-type conversions ---
c10::ScalarType to_aten(ScalarType s);
ScalarType from_aten(c10::ScalarType s);

c10::Device to_aten(const Device& d);
Device from_aten(const c10::Device& d);

c10::Layout to_aten(Layout l);
Layout from_aten(c10::Layout l);

c10::MemoryFormat to_aten(MemoryFormat m);
MemoryFormat from_aten(c10::MemoryFormat m);

c10::Scalar to_aten(const Scalar& s);
Scalar from_aten(const c10::Scalar& s);

// --- value / tensor / list ---
c10::IValue to_aten(const Value& v);
Value from_aten(const c10::IValue& iv);

at::Tensor to_aten(const Tensor& t);
Tensor from_aten(at::Tensor t);

c10::Stack to_aten(const ArgumentList& args);
ArgumentList from_aten(c10::Stack stack);

// --- schema ---
c10::Argument to_aten(const Argument& a);
c10::FunctionSchema to_aten(const OperatorSchema& s);
OperatorSchema from_aten(const c10::FunctionSchema& s);

// --- backend dispatch mapping ---
c10::DispatchKey to_dispatch_key(BackendId backend,
                                 Functionality functionality);

// --- minimal tensor construction (adapter / tests) ---
Tensor make_cpu_tensor(const std::vector<int64_t>& sizes, ScalarType dtype);

} // namespace detail
} // namespace nt
