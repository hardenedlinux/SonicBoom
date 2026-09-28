#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include <sonicboom/value.h>

namespace nt {

// Layer 2 operator argument/result containers. Native Torch-owned; they do not
// expose c10::Stack. The adapter converts to/from the native stack.
class ArgumentList {
 public:
  ArgumentList() = default;
  explicit ArgumentList(std::vector<Value> values) : values_(std::move(values)) {}

  size_t size() const { return values_.size(); }
  bool empty() const { return values_.empty(); }
  const Value& operator[](size_t i) const { return values_[i]; }
  Value& operator[](size_t i) { return values_[i]; }
  void push_back(Value v) { values_.push_back(std::move(v)); }

  const std::vector<Value>& values() const { return values_; }
  std::vector<Value>& values() { return values_; }

 private:
  std::vector<Value> values_;
};

// Same shape as ArgumentList; a distinct name for clarity at call sites. v0
// does not need them to diverge.
using ResultList = ArgumentList;

} // namespace nt
