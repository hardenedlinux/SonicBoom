// test_value.cpp — M2 test B: the Layer 2 Value boundary.
//
// Verifies nt::Value is a self-contained std::variant over Native Torch-owned
// types (no c10::IValue in scope) with correct kind()/is*()/to*() accessors.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

int main() {
  using namespace nt;

  Value none;
  assert(none.isNone());
  assert(none.kind() == ValueKind::None);

  Value i(42);
  assert(i.isInt());
  assert(i.kind() == ValueKind::Int);
  assert(i.toInt() == 42);

  Value f(3.5);
  assert(f.isFloat());
  assert(f.kind() == ValueKind::Float);
  assert(f.toFloat() == 3.5);

  Value b(true);
  assert(b.isBool());
  assert(b.kind() == ValueKind::Bool);
  assert(b.toBool());

  Value s(std::string("hello"));
  assert(s.isString());
  assert(s.kind() == ValueKind::String);
  assert(s.toString() == "hello");

  Value d(Device::cpu());
  assert(d.isDevice());
  assert(d.kind() == ValueKind::Device);
  assert(d.toDevice() == Device::cpu());

  Value st(ScalarType::Float);
  assert(st.isScalarType());
  assert(st.kind() == ValueKind::ScalarType);
  assert(st.toScalarType() == ScalarType::Float);

  Value l(Layout::Strided);
  assert(l.isLayout());
  assert(l.kind() == ValueKind::Layout);
  assert(l.toLayout() == Layout::Strided);

  Value mf(MemoryFormat::Contiguous);
  assert(mf.isMemoryFormat());
  assert(mf.kind() == ValueKind::MemoryFormat);
  assert(mf.toMemoryFormat() == MemoryFormat::Contiguous);

  Value tl(std::vector<Tensor>{});
  assert(tl.isTensorList());
  assert(tl.kind() == ValueKind::TensorList);

  Value sc(Scalar(7));
  assert(sc.isScalar());
  assert(sc.kind() == ValueKind::Scalar);
  assert(sc.toScalar().isInt());
  assert(sc.toScalar().toInt() == 7);

  std::cout << "M2 test_value OK\n";
  return 0;
}
