#include "adapter.h"

namespace nt {
namespace detail {

c10::MemoryFormat to_aten(MemoryFormat m) {
  switch (m) {
    case MemoryFormat::Contiguous:
      return c10::MemoryFormat::Contiguous;
    case MemoryFormat::Preserve:
      return c10::MemoryFormat::Preserve;
    case MemoryFormat::ChannelsLast:
      return c10::MemoryFormat::ChannelsLast;
  }
  TORCH_CHECK(false, "unsupported nt::MemoryFormat");
  return c10::MemoryFormat::Contiguous; // unreachable
}

MemoryFormat from_aten(c10::MemoryFormat m) {
  switch (m) {
    case c10::MemoryFormat::Contiguous:
      return MemoryFormat::Contiguous;
    case c10::MemoryFormat::Preserve:
      return MemoryFormat::Preserve;
    case c10::MemoryFormat::ChannelsLast:
      return MemoryFormat::ChannelsLast;
    default:
      break;
  }
  TORCH_CHECK(false, "unsupported c10::MemoryFormat");
  return MemoryFormat::Contiguous; // unreachable
}

} // namespace detail
} // namespace nt
