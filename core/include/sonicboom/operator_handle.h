#pragma once

#include <memory>
#include <string>
#include <utility>

#include <sonicboom/schema.h>
#include <sonicboom/argument_list.h>

namespace nt {

namespace detail {
struct OperatorHandleImpl; // defined in the Layer 1 adapter; holds the native
                           // operator handle
} // namespace detail

struct OperatorName {
  std::string name;
  std::string overload_name;

  OperatorName() = default;
  OperatorName(std::string n, std::string o = "")
      : name(std::move(n)), overload_name(std::move(o)) {}
};

// Layer 2 operator handle: identity + schema access + boxed invocation. Does
// not expose c10::OperatorHandle.
class OperatorHandle {
 public:
  OperatorHandle(); // null handle
  explicit OperatorHandle(std::shared_ptr<detail::OperatorHandleImpl> impl);

  bool defined() const;

  std::string name() const;
  OperatorSchema schema() const;

  // Invoke the operator on the given arguments (boxed). v0 dispatches to the
  // CPU backend; returns the operator's results.
  ResultList call(ArgumentList args) const;

 private:
  std::shared_ptr<detail::OperatorHandleImpl> impl_;
};

// Look up a registered operator by name. Throws if not registered (v0).
OperatorHandle find_operator(const OperatorName& name);

} // namespace nt
