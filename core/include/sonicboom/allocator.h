#pragma once

#include <cstddef>

namespace nt {

// Layer 2 allocator boundary (manifest M2). Define the interface only; the
// complete allocator system is deferred. The ABI-level deleter is a plain C
// function pointer (ctx, data) — NOT std::function — so it can cross the C ABI
// boundary later.
class Allocator {
 public:
  using DeleterFn = void (*)(void* ctx, void* data);

  virtual ~Allocator() = default;

  virtual void* allocate(size_t nbytes) = 0;
  virtual void deallocate(void* data) = 0;

  // C-compatible deleter + context for memory returned by allocate().
  virtual DeleterFn deleter() const = 0;
  virtual void* context() const = 0;
};

} // namespace nt
