#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace nt {

// Native Torch-owned operator argument kind. v0 required kinds (manifest M2).
enum class ArgKind : uint8_t {
  Tensor,
  OptionalTensor,
  TensorList,
  OptionalTensorList,
  Scalar,
  Int,
  IntList,
  Float,
  Bool,
  String,
  Device,
  ScalarType,
  Layout,
  MemoryFormat,
  None,
};

struct Argument {
  std::string name;
  ArgKind kind = ArgKind::None;

  Argument() = default;
  Argument(std::string n, ArgKind k) : name(std::move(n)), kind(k) {}
};

} // namespace nt
