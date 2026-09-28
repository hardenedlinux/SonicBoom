#include "adapter.h"

namespace nt {
namespace detail {

c10::Device to_aten(const Device& d) {
  TORCH_CHECK(d.type == DeviceType::CPU, "v0 supports the CPU device only");
  return c10::Device(c10::DeviceType::CPU, static_cast<int8_t>(d.index));
}

Device from_aten(const c10::Device& d) {
  TORCH_CHECK(d.type() == c10::DeviceType::CPU, "v0 supports the CPU device only");
  // c10 uses index -1 to mean "current/default" device; v0 CPU is
  // single-device, so normalize to index 0 (matching Device::cpu()).
  return Device{DeviceType::CPU, 0};
}

} // namespace detail
} // namespace nt
