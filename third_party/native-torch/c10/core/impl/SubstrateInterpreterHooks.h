#pragma once

#include <c10/core/impl/SubstrateInterpreter.h>
#include <c10/macros/Export.h>
#include <c10/util/Registry.h>
#include <memory>

namespace c10::impl {

// Minimal interface for SubstrateInterpreter hooks
struct C10_API SubstrateInterpreterHooksInterface {
  virtual ~SubstrateInterpreterHooksInterface() = default;

  // Get the SubstrateInterpreter instance
  // Stub implementation throws error when Python is not available
  virtual SubstrateInterpreter* getSubstrateInterpreter() const {
    TORCH_CHECK(
        false,
        "PyTorch was compiled without Python support. "
        "Cannot access Python interpreter from C++.");
  }
};

// Deprecated: no longer used internally, kept for ABI compatibility.
struct C10_API SubstrateInterpreterHooksArgs{};

C10_DECLARE_REGISTRY(SubstrateInterpreterHooksRegistry, SubstrateInterpreterHooksInterface);

#define REGISTER_PYTHON_HOOKS(clsname) \
  C10_REGISTER_CLASS(SubstrateInterpreterHooksRegistry, clsname, clsname)

// Get the global SubstrateInterpreter hooks instance
C10_API const SubstrateInterpreterHooksInterface& getSubstrateInterpreterHooks();

// Helper function to get the global interpreter
C10_API SubstrateInterpreter* getGlobalSubstrateInterpreter();

} // namespace c10::impl
