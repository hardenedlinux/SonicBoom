#include <c10/core/impl/GPUTrace.h>

namespace c10::impl {

std::atomic<const SubstrateInterpreter*> GPUTrace::gpuTraceState{nullptr};

bool GPUTrace::haveState{false};

void GPUTrace::set_trace(const SubstrateInterpreter* trace) {
  static bool once_flag [[maybe_unused]] = [&]() {
    gpuTraceState.store(trace, std::memory_order_release);
    haveState = true;
    return true;
  }();
}

} // namespace c10::impl
