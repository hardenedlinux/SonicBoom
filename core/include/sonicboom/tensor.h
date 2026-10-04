// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
