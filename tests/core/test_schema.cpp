// test_schema.cpp — M2 test A: the Layer 2 OperatorSchema boundary.
//
// Builds an nt::OperatorSchema, registers it through the Layer 1 adapter, looks
// it up, and round-trips it back to nt::OperatorSchema. This file includes only
// <sonicboom/...>; no c10::FunctionSchema (or any native type) is in scope.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>

int main() {
  using namespace nt;

  OperatorSchema schema("m2::add", /*overload_name=*/"",
                        {Argument("a", ArgKind::Int),
                         Argument("b", ArgKind::Int)},
                        {Argument("0", ArgKind::Int)});

  assert(schema.name() == "m2::add");
  assert(schema.overload_name().empty());
  assert(schema.num_args() == 2);
  assert(schema.num_returns() == 1);
  assert(schema.arguments()[0].name == "a");
  assert(schema.arguments()[0].kind == ArgKind::Int);
  assert(schema.arguments()[1].kind == ArgKind::Int);
  assert(schema.returns()[0].kind == ArgKind::Int);

  // Register the schema in the runtime registry (Layer 1 → native dispatcher).
  auto reg = define_operator(schema);
  assert(reg.valid());

  // Look it back up and round-trip to nt::OperatorSchema.
  OperatorHandle op = find_operator(OperatorName("m2::add"));
  assert(op.defined());
  assert(op.name() == "m2::add");

  OperatorSchema got = op.schema();
  assert(got.name() == "m2::add");
  assert(got.num_args() == 2);
  assert(got.num_returns() == 1);
  assert(got.arguments()[0].name == "a");
  assert(got.arguments()[0].kind == ArgKind::Int);
  assert(got.arguments()[1].kind == ArgKind::Int);
  assert(got.returns()[0].kind == ArgKind::Int);

  std::cout << "M2 test_schema OK\n";
  return 0;
}
