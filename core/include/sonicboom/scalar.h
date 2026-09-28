#pragma once

#include <cstdint>
#include <variant>

namespace nt {

// Type-erased scalar value (int / float / bool). Complex is deferred (v0).
class Scalar {
 public:
  Scalar() = default;
  Scalar(int i) : data_(static_cast<int64_t>(i)) {} // disambiguate int literals
  Scalar(int64_t i) : data_(i) {}
  Scalar(double d) : data_(d) {}
  Scalar(bool b) : data_(b) {}

  bool isInt() const { return std::holds_alternative<int64_t>(data_); }
  bool isDouble() const { return std::holds_alternative<double>(data_); }
  bool isBool() const { return std::holds_alternative<bool>(data_); }

  int64_t toInt() const { return std::get<int64_t>(data_); }
  double toDouble() const { return std::get<double>(data_); }
  bool toBool() const { return std::get<bool>(data_); }

 private:
  std::variant<int64_t, double, bool> data_;
};

} // namespace nt
