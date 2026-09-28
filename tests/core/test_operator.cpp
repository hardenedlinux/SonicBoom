// test_operator.cpp — M2 test C: end-to-end boxed operator dispatch.
//
// define_operator + register_kernel + OperatorHandle::call, going through the
// Layer 1 adapter onto the native dispatcher and back. The kernel operates on
// nt::ArgumentList / nt::ResultList only; no c10::Stack is in scope here.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

namespace {

nt::ResultList double_it(const nt::ArgumentList& args) {
  const double x = args[0].toFloat();
  return nt::ResultList(std::vector<nt::Value>{nt::Value(x * 2.0)});
}

} // namespace

int main() {
  using namespace nt;

  OperatorSchema schema("m2::scale", /*overload_name=*/"",
                        {Argument("x", ArgKind::Float)},
                        {Argument("0", ArgKind::Float)});

  auto def = define_operator(schema);
  assert(def.valid());

  auto impl = register_kernel(OperatorName("m2::scale"), BackendId::cpu(),
                              Functionality::Dense, &double_it);
  assert(impl.valid());

  OperatorHandle op = find_operator(OperatorName("m2::scale"));
  assert(op.defined());

  ArgumentList args;
  args.push_back(Value(3.0));

  ResultList out = op.call(args);
  assert(out.size() == 1);
  assert(out[0].isFloat());
  assert(out[0].toFloat() == 6.0);

  std::cout << "M2 test_operator OK\n";
  return 0;
}
