// dispatch_proof.cpp — SonicBoom M1 proof: registration + dispatcher lookup +
// boxed invocation on the native-torch dispatcher core.
//
// Demonstrates the full operator path required by manifest M1:
//
//     FunctionSchema  ->  Dispatcher::registerDef
//         ->  Dispatcher::registerImpl (boxed kernel)
//         ->  findSchema / findOp (dispatcher lookup)
//         ->  OperatorHandle::callBoxed (boxed invocation)
//
// This exercises only the dispatcher mechanism (schema, registration, lookup,
// boxed dispatch). It deliberately does NOT go through the Tensor op facade or
// any Layer 1 adapter — those are outside the M1 boundary.

#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/operator_name.h>
#include <ATen/core/stack.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DeviceType.h>

#include <cassert>
#include <iostream>

namespace {

// Boxed relu kernel: relu(x) = max(x, 0). Operates on the type-erased Stack.
void relu_boxed(const c10::OperatorHandle& op, c10::Stack* stack) {
  (void)op;
  const double x = stack->back().toDouble();
  stack->clear();
  stack->push_back(x < 0.0 ? 0.0 : x);
}

} // namespace

int main() {
  using namespace c10;

  // 1. Operator schema: aten::relu(float x) -> float.
  FunctionSchema schema(
      /*name=*/"aten::relu",
      /*overload_name=*/"",
      /*arguments=*/{Argument("x", FloatType::get())},
      /*returns=*/{Argument("0", FloatType::get())});

  // 2. Register the schema.
  auto def_raii =
      Dispatcher::singleton().registerDef(std::move(schema), "SonicBoom M1 proof");

  // 3. Register a boxed kernel under DispatchKey::CPU.
  auto kernel = KernelFunction::makeFromBoxedFunction<&relu_boxed>();
  auto impl_raii = Dispatcher::singleton().registerImpl(
      OperatorName("aten::relu", ""),
      DispatchKey::CPU,
      std::move(kernel),
      /*cpp_signature=*/std::nullopt,
      /*inferred_function_schema=*/nullptr,
      "SonicBoom M1 proof");

  // 4. Dispatcher lookup.
  OperatorHandle op =
      Dispatcher::singleton().findSchemaOrThrow("aten::relu", "");
  assert(op.schema().name() == "aten::relu");
  assert(op.schema().overload_name().empty());
  auto found =
      Dispatcher::singleton().findSchema(OperatorName("aten::relu", ""));
  assert(found.has_value());

  // 5. Boxed invocation.
  //
  // The stack carries a plain double (no Tensor), so DispatchKeyExtractor
  // cannot infer a backend from it. We dispatch explicitly to the CPU kernel
  // we registered above; a real tensor input would carry DispatchKey::CPU and
  // the generic callBoxed path would select it automatically.
  Stack stack;
  stack.emplace_back(-3.0);
  op.callBoxedForDispatchKey(DispatchKey::CPU, stack);
  assert(stack.size() == 1);
  assert(stack[0].isDouble());
  assert(stack[0].toDouble() == 0.0);

  stack.clear();
  stack.emplace_back(5.0);
  op.callBoxedForDispatchKey(DispatchKey::CPU, stack);
  assert(stack.size() == 1);
  assert(stack[0].toDouble() == 5.0);

  std::cout << "M1 dispatch proof OK: schema -> registerDef -> registerImpl -> "
               "findSchema -> callBoxed\n";
  return 0;
}
