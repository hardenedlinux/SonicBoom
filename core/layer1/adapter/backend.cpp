#include "adapter.h"

namespace nt {
namespace detail {

c10::DispatchKey to_dispatch_key(BackendId backend,
                                 Functionality functionality) {
  // v0: single backend (CPU), so "highest-priority applicable backend wins" is
  // trivially satisfied. The adapter maps (backend, functionality) onto the
  // native dispatch representation.
  TORCH_CHECK(backend == BackendId::cpu(), "v0 supports the CPU backend only");
  switch (functionality) {
    case Functionality::Dense:
      return c10::DispatchKey::CPU;
    case Functionality::Sparse:
      return c10::DispatchKey::Sparse;
    case Functionality::Quantized:
      return c10::DispatchKey::QuantizedCPU;
    case Functionality::Autograd:
      return c10::DispatchKey::Autograd;
  }
  TORCH_CHECK(false, "unsupported nt::Functionality");
  return c10::DispatchKey::CPU; // unreachable
}

} // namespace detail
} // namespace nt
