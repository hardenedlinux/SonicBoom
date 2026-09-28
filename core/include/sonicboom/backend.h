#pragma once

#include <cstdint>

namespace nt {

// Native Torch backend identity. This is SonicBoom-owned: it is not an alias
// for c10::DispatchKey. The Layer 1 adapter maps it onto the native dispatch
// representation (see core/layer1/adapter/backend.cpp).
struct BackendId {
  uint16_t value;   // backend discriminator (0 = CPU in v0)
  int32_t priority; // ordering key; highest-priority applicable backend wins

  static constexpr BackendId cpu() { return BackendId{0, 0}; }

  friend bool operator==(const BackendId& a, const BackendId& b) {
    return a.value == b.value;
  }
  friend bool operator!=(const BackendId& a, const BackendId& b) {
    return !(a == b);
  }
};

// Functionality selects a dispatch path within a backend (manifest §3).
enum class Functionality : uint8_t {
  Dense,
  Sparse,
  Quantized,
  Autograd,
};

} // namespace nt
