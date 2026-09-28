#pragma once

#include <cstdint>

namespace nt {

// v0 is CPU-only; CUDA is a deferred manifest amendment and must not appear
// here as an abstraction.
enum class DeviceType : uint8_t {
  CPU,
};

struct Device {
  DeviceType type = DeviceType::CPU;
  int32_t index = 0; // device ordinal; always 0 for CPU in v0.

  static constexpr Device cpu() { return Device{DeviceType::CPU, 0}; }
};

inline bool operator==(const Device& a, const Device& b) {
  return a.type == b.type && a.index == b.index;
}
inline bool operator!=(const Device& a, const Device& b) { return !(a == b); }

} // namespace nt
