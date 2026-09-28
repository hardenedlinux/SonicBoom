#include "adapter.h"

namespace nt {
namespace detail {

c10::Scalar to_aten(const Scalar& s) {
  if (s.isInt()) {
    return c10::Scalar(s.toInt());
  }
  if (s.isDouble()) {
    return c10::Scalar(s.toDouble());
  }
  return c10::Scalar(s.toBool());
}

Scalar from_aten(const c10::Scalar& s) {
  if (s.isBoolean()) {
    return Scalar(s.to<bool>());
  }
  if (s.isFloatingPoint()) {
    return Scalar(s.to<double>());
  }
  return Scalar(s.to<int64_t>());
}

} // namespace detail
} // namespace nt
