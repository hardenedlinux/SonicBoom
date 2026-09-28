#pragma once

#include <cstdint>

namespace nt {

// Layer 2 scalar type (dtype). A Native Torch-owned enum; it does not alias
// c10::ScalarType. The Layer 1 adapter maps it (see core/layer1/adapter).
// v0 subset: the dtypes the minimal runtime needs; float8 / unsigned dtypes are
// deferred per the manifest.
enum class ScalarType : uint8_t {
  Byte,
  Char,
  Short,
  Int,
  Long,
  Half,
  Float,
  Double,
  ComplexFloat,
  ComplexDouble,
  Bool,
  BFloat16,
};

} // namespace nt
