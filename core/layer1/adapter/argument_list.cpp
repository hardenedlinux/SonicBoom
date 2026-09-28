#include "adapter.h"

namespace nt {
namespace detail {

c10::Stack to_aten(const ArgumentList& args) {
  c10::Stack stack;
  stack.reserve(args.size());
  for (const auto& v : args.values()) {
    stack.push_back(to_aten(v));
  }
  return stack;
}

ArgumentList from_aten(c10::Stack stack) {
  std::vector<Value> values;
  values.reserve(stack.size());
  for (const auto& iv : stack) {
    values.push_back(from_aten(iv));
  }
  return ArgumentList(std::move(values));
}

} // namespace detail
} // namespace nt
