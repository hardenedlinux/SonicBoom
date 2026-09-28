#pragma once

#include <cstdint>

namespace nt {

enum class MemoryFormat : uint8_t {
  Contiguous,
  Preserve,
  ChannelsLast,
};

} // namespace nt
