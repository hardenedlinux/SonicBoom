#include <c10/core/impl/SubstrateInterpreterHooks.h>

namespace c10::impl {

// Define the registry
C10_DEFINE_REGISTRY(SubstrateInterpreterHooksRegistry, SubstrateInterpreterHooksInterface)

const SubstrateInterpreterHooksInterface& getSubstrateInterpreterHooks() {
  auto create_impl = [] {
#if !defined C10_MOBILE
    auto hooks = SubstrateInterpreterHooksRegistry()->Create("SubstrateInterpreterHooks");
    if (hooks) {
      return hooks;
    }
#endif
    // Return stub implementation that will throw errors when methods are called
    return std::make_unique<SubstrateInterpreterHooksInterface>();
  };
  static auto hooks = create_impl();
  return *hooks;
}

// Main function to get global SubstrateInterpreter
SubstrateInterpreter* getGlobalSubstrateInterpreter() {
  static SubstrateInterpreter* cached = getSubstrateInterpreterHooks().getSubstrateInterpreter();
  return cached;
}

} // namespace c10::impl
