#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include <sonicboom/argument.h>

namespace nt {

// Layer 2 operator schema. Does not expose c10::FunctionSchema; the adapter
// (core/layer1/adapter/schema.cpp) converts in both directions.
class OperatorSchema {
 public:
  OperatorSchema() = default;
  OperatorSchema(std::string name,
                 std::string overload_name,
                 std::vector<Argument> arguments,
                 std::vector<Argument> returns)
      : name_(std::move(name)),
        overload_name_(std::move(overload_name)),
        arguments_(std::move(arguments)),
        returns_(std::move(returns)) {}

  const std::string& name() const { return name_; }
  const std::string& overload_name() const { return overload_name_; }
  const std::vector<Argument>& arguments() const { return arguments_; }
  const std::vector<Argument>& returns() const { return returns_; }
  size_t num_args() const { return arguments_.size(); }
  size_t num_returns() const { return returns_.size(); }

 private:
  std::string name_;
  std::string overload_name_;
  std::vector<Argument> arguments_;
  std::vector<Argument> returns_;
};

} // namespace nt
