#include <ATen/core/op_registration/infer_schema.h>

#include <c10/util/irange.h>

#include <sstream>

// SonicBoom M1 ADAPT: c10::findSchemaDifferences, fmt-free.
//
// Upstream this lives in op_registration/infer_schema.cpp and formats its
// diagnostics with fmt::format. SonicBoom v0 is fmt-free (manifest M4), so the
// same pure logic is reimplemented here with std::ostringstream. It is pulled
// into the closure by dispatch/OperatorEntry.cpp (checkSchema), which reports
// schema-compatibility differences during registerDef.

namespace c10 {

std::optional<std::string> findSchemaDifferences(
    const FunctionSchema& lhs,
    const FunctionSchema& rhs) {
  if (lhs.arguments().size() != rhs.arguments().size()) {
    std::ostringstream oss;
    oss << "The number of arguments is different. " << lhs.arguments().size()
        << " vs " << rhs.arguments().size() << ".";
    return oss.str();
  }
  if (lhs.returns().size() != rhs.returns().size()) {
    std::ostringstream oss;
    oss << "The number of returns is different. " << lhs.returns().size()
        << " vs " << rhs.returns().size() << ".";
    return oss.str();
  }

  for (const auto i : c10::irange(lhs.arguments().size())) {
    const TypePtr& leftType = lhs.arguments()[i].type();
    const TypePtr& rightType = rhs.arguments()[i].type();
    // Type::operator== is virtual. Comparing pointers first is cheaper,
    // particularly for singleton types like NumberType/AnyType.
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      std::ostringstream oss;
      oss << "Type mismatch in argument " << (i + 1) << ": "
          << lhs.arguments()[i].type()->str() << " vs "
          << rhs.arguments()[i].type()->str() << ".";
      return oss.str();
    }
  }

  for (const auto i : c10::irange(lhs.returns().size())) {
    const TypePtr& leftType = lhs.returns()[i].type();
    const TypePtr& rightType = rhs.returns()[i].type();
    if (leftType.get() != rightType.get() && *leftType != *rightType) {
      std::ostringstream oss;
      oss << "Type mismatch in return " << (i + 1) << ": "
          << lhs.returns()[i].type()->str() << " vs "
          << rhs.returns()[i].type()->str() << ".";
      return oss.str();
    }
  }

  // no differences found
  return std::nullopt;
}

} // namespace c10
