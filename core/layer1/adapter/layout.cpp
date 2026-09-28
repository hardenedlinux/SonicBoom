#include "adapter.h"

namespace nt {
namespace detail {

c10::Layout to_aten(Layout l) {
  switch (l) {
    case Layout::Strided:
      return c10::Layout::Strided;
    case Layout::Sparse:
      return c10::Layout::Sparse;
  }
  TORCH_CHECK(false, "unsupported nt::Layout");
  return c10::Layout::Strided; // unreachable
}

Layout from_aten(c10::Layout l) {
  switch (l) {
    case c10::Layout::Strided:
      return Layout::Strided;
    case c10::Layout::Sparse:
      return Layout::Sparse;
    default:
      break;
  }
  TORCH_CHECK(false, "unsupported c10::Layout");
  return Layout::Strided; // unreachable
}

} // namespace detail
} // namespace nt
