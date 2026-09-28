#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <sonicboom/scalar_type.h>
#include <sonicboom/device.h>
#include <sonicboom/layout.h>

namespace nt {

namespace detail {
struct TensorImpl; // defined in the Layer 1 adapter; holds the native tensor
struct Adapter;    // Layer 1 access point (friend; defined in the adapter)
} // namespace detail

// Stable Layer 2 tensor handle. Opaque: the underlying implementation tensor
// lives behind the Layer 1 boundary, so no c10/ATen type (TensorImpl,
// StorageImpl, intrusive_ptr, …) appears in this header.
class Tensor {
 public:
  Tensor(); // undefined tensor
  explicit Tensor(std::shared_ptr<detail::TensorImpl> impl);

  bool defined() const;

  ScalarType dtype() const;
  Device device() const;
  Layout layout() const;

  int64_t dim() const;
  int64_t size(int64_t dim) const;
  std::vector<int64_t> sizes() const;
  int64_t numel() const;

  void* data_ptr() const;

 private:
  std::shared_ptr<detail::TensorImpl> impl_;
  friend struct detail::Adapter;
};

// Allocate an (uninitialized) CPU tensor of the given sizes and dtype. v0
// factory; the underlying storage lives behind the Layer 1 boundary.
Tensor empty(const std::vector<int64_t>& sizes, ScalarType dtype);

} // namespace nt
