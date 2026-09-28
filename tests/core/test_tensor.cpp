// test_tensor.cpp — M2 test E: the Layer 2 Tensor boundary.
//
// Allocates an nt::Tensor via the public nt::empty factory and inspects it with
// the public accessors. Only <sonicboom/...> is included, so no TensorImpl /
// StorageImpl / intrusive_ptr (or any c10/ATen type) is in scope.

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>

int main() {
  using namespace nt;

  const Tensor t = empty({2, 3}, ScalarType::Float);
  assert(t.defined());

  assert(t.dim() == 2);
  assert(t.size(0) == 2);
  assert(t.size(1) == 3);
  assert(t.numel() == 6);

  const std::vector<int64_t> sizes = t.sizes();
  assert(sizes.size() == 2);
  assert(sizes[0] == 2);
  assert(sizes[1] == 3);

  assert(t.dtype() == ScalarType::Float);
  assert(t.device() == Device::cpu());
  assert(t.layout() == Layout::Strided);
  assert(t.data_ptr() != nullptr);

  // A default-constructed handle is an undefined tensor.
  const Tensor undef;
  assert(!undef.defined());

  std::cout << "M2 test_tensor OK\n";
  return 0;
}
