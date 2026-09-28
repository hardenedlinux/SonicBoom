#include "adapter.h"

namespace nt {
namespace detail {

c10::IValue to_aten(const Value& v) {
  switch (v.kind()) {
    case ValueKind::None:
      return c10::IValue();
    case ValueKind::Tensor:
      return c10::IValue(to_aten(v.toTensor()));
    case ValueKind::Scalar:
      return c10::IValue(to_aten(v.toScalar()));
    case ValueKind::Int:
      return c10::IValue(v.toInt());
    case ValueKind::Float:
      return c10::IValue(v.toFloat());
    case ValueKind::Bool:
      return c10::IValue(v.toBool());
    case ValueKind::String:
      return c10::IValue(v.toString());
    case ValueKind::Device:
      return c10::IValue(to_aten(v.toDevice()));
    case ValueKind::ScalarType:
      return c10::IValue(to_aten(v.toScalarType()));
    case ValueKind::Layout:
      return c10::IValue(to_aten(v.toLayout()));
    case ValueKind::MemoryFormat:
      return c10::IValue(to_aten(v.toMemoryFormat()));
    case ValueKind::TensorList: {
      std::vector<at::Tensor> vec;
      vec.reserve(v.toTensorList().size());
      for (const auto& t : v.toTensorList()) {
        vec.push_back(to_aten(t));
      }
      return c10::IValue(std::move(vec));
    }
  }
  TORCH_CHECK(false, "unsupported nt::ValueKind");
  return c10::IValue(); // unreachable
}

Value from_aten(const c10::IValue& iv) {
  if (iv.isNone()) {
    return Value();
  }
  if (iv.isTensor()) {
    return Value(from_aten(iv.toTensor()));
  }
  if (iv.isInt()) {
    return Value(iv.toInt());
  }
  if (iv.isDouble()) {
    return Value(iv.toDouble());
  }
  if (iv.isBool()) {
    return Value(iv.toBool());
  }
  if (iv.isString()) {
    return Value(iv.toStringRef());
  }
  if (iv.isDevice()) {
    return Value(from_aten(iv.toDevice()));
  }
  if (iv.isTensorList()) {
    std::vector<Tensor> vec;
    for (const auto& t : iv.toTensorVector()) {
      vec.push_back(from_aten(t));
    }
    return Value(std::move(vec));
  }
  if (iv.isScalar()) {
    return Value(from_aten(iv.toScalar()));
  }
  TORCH_CHECK(false, "unsupported c10::IValue kind");
  return Value(); // unreachable
}

} // namespace detail
} // namespace nt
