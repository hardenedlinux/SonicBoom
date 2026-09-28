// test_backend.cpp — M2 test D: the Layer 2 BackendId boundary.
//
// Verifies the public BackendId contract behind "highest-priority applicable
// backend wins" (manifest M2 §dispatch). The BackendId → native dispatch-key
// mapping itself lives behind the Layer 1 boundary (core/layer1/adapter/
// backend.cpp) and is exercised end-to-end by test_operator (register_kernel
// under BackendId::cpu() + Functionality::Dense fires the CPU kernel).

#include <sonicboom/sonicboom.h>

#include <cassert>
#include <iostream>
#include <vector>

namespace {

// Highest-priority applicable backend wins. v0 has only CPU, so this is the
// ordering rule expressed on the public type; the adapter maps the winner onto
// the native dispatch representation.
nt::BackendId select(const std::vector<nt::BackendId>& backends) {
  nt::BackendId best = backends.front();
  for (const nt::BackendId b : backends) {
    if (b.priority > best.priority) best = b;
  }
  return best;
}

} // namespace

int main() {
  using namespace nt;

  const BackendId cpu = BackendId::cpu();
  assert(cpu.value == 0);
  assert(cpu.priority == 0);

  const BackendId a{1, 10};
  const BackendId b{2, 20};
  assert(a != b);
  assert((a == BackendId{1, 10})); // equality is by discriminator (value) only

  assert(select({a, b, cpu}) == b);
  assert(select({cpu, a}) == a);

  std::cout << "M2 test_backend OK\n";
  return 0;
}
