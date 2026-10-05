#include <ATen/core/op_registration/infer_schema.h>
#include <c10/util/irange.h>
#include <fmt/format.h>

namespace c10 {

namespace detail::infer_schema {
namespace {

std::vector<Argument> createArgumentVector(c10::ArrayRef<ArgumentDef> args) {
  std::vector<Argument> result;
  result.reserve(args.size());
  for (const auto i : c10::irange(args.size())) {
    // Arguments are named "_<index>"
    result.emplace_back(
        fmt::format("_{}", i),
        (*args[i].getFakeTypeFn)(),
        (*args[i].getTypeFn)());
  }
  return result;
}
} // namespace
// This is intentionally a separate function and in a .cpp file
// because then the template is smaller and that benefits binary size
FunctionSchema make_function_schema(
    std::string&& name,
    std::string&& overload_name,
    c10::ArrayRef<ArgumentDef> arguments,
    c10::ArrayRef<ArgumentDef> returns) {
  return FunctionSchema(
      std::move(name),
      std::move(overload_name),
      createArgumentVector(arguments),
      createArgumentVector(returns));
}

FunctionSchema make_function_schema(
    c10::ArrayRef<ArgumentDef> arguments,
    c10::ArrayRef<ArgumentDef> returns) {
  return make_function_schema("", "", arguments, returns);
}
} // namespace detail

// SonicBoom ADAPT: c10::findSchemaDifferences is provided fmt-free by
// core/op_registration/find_schema_differences.cpp (M1). Upstream also defined
// it here; keeping both causes a multiple-definition link error, so it is
// removed from this translation unit (make_function_schema above is retained —
// it backs inferFunctionSchemaFromFunctor used by the fallback kernels).

} // namespace c10
