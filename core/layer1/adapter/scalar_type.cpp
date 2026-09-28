#include "adapter.h"

namespace nt {
namespace detail {

c10::ScalarType to_aten(ScalarType s) {
  switch (s) {
    case ScalarType::Byte:
      return c10::ScalarType::Byte;
    case ScalarType::Char:
      return c10::ScalarType::Char;
    case ScalarType::Short:
      return c10::ScalarType::Short;
    case ScalarType::Int:
      return c10::ScalarType::Int;
    case ScalarType::Long:
      return c10::ScalarType::Long;
    case ScalarType::Half:
      return c10::ScalarType::Half;
    case ScalarType::Float:
      return c10::ScalarType::Float;
    case ScalarType::Double:
      return c10::ScalarType::Double;
    case ScalarType::ComplexFloat:
      return c10::ScalarType::ComplexFloat;
    case ScalarType::ComplexDouble:
      return c10::ScalarType::ComplexDouble;
    case ScalarType::Bool:
      return c10::ScalarType::Bool;
    case ScalarType::BFloat16:
      return c10::ScalarType::BFloat16;
  }
  TORCH_CHECK(false, "unsupported nt::ScalarType");
  return c10::ScalarType::Float; // unreachable
}

ScalarType from_aten(c10::ScalarType s) {
  switch (s) {
    case c10::ScalarType::Byte:
      return ScalarType::Byte;
    case c10::ScalarType::Char:
      return ScalarType::Char;
    case c10::ScalarType::Short:
      return ScalarType::Short;
    case c10::ScalarType::Int:
      return ScalarType::Int;
    case c10::ScalarType::Long:
      return ScalarType::Long;
    case c10::ScalarType::Half:
      return ScalarType::Half;
    case c10::ScalarType::Float:
      return ScalarType::Float;
    case c10::ScalarType::Double:
      return ScalarType::Double;
    case c10::ScalarType::ComplexFloat:
      return ScalarType::ComplexFloat;
    case c10::ScalarType::ComplexDouble:
      return ScalarType::ComplexDouble;
    case c10::ScalarType::Bool:
      return ScalarType::Bool;
    case c10::ScalarType::BFloat16:
      return ScalarType::BFloat16;
    default:
      break;
  }
  TORCH_CHECK(false, "unsupported c10::ScalarType");
  return ScalarType::Float; // unreachable
}

} // namespace detail
} // namespace nt
