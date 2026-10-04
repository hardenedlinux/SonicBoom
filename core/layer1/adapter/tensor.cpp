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

#include "adapter.h"

namespace nt {

Tensor::Tensor() = default;
Tensor::Tensor(std::shared_ptr<detail::TensorImpl> impl) : impl_(std::move(impl)) {}

bool Tensor::defined() const { return impl_ && impl_->t.defined(); }

ScalarType Tensor::dtype() const { return detail::from_aten(impl_->t.scalar_type()); }
Device Tensor::device() const { return detail::from_aten(impl_->t.device()); }
Layout Tensor::layout() const { return detail::from_aten(impl_->t.layout()); }

int64_t Tensor::dim() const { return impl_->t.dim(); }
int64_t Tensor::size(int64_t d) const { return impl_->t.size(d); }
std::vector<int64_t> Tensor::sizes() const {
  auto s = impl_->t.sizes();
  return std::vector<int64_t>(s.begin(), s.end());
}
int64_t Tensor::numel() const { return impl_->t.numel(); }
void* Tensor::data_ptr() const { return impl_->t.data_ptr(); }

Tensor empty(const std::vector<int64_t>& sizes, ScalarType dtype) {
  return detail::make_cpu_tensor(sizes, dtype);
}

} // namespace nt

namespace nt {
namespace detail {

at::Tensor to_aten(const Tensor& t) {
  return Adapter::impl_of(t)->t;
}

Tensor from_aten(at::Tensor t) {
  return Adapter::from_impl(
      std::make_shared<TensorImpl>(TensorImpl{std::move(t)}));
}

Tensor make_cpu_tensor(const std::vector<int64_t>& sizes, ScalarType dtype) {
  int64_t numel = 1;
  for (auto s : sizes) {
    numel *= s;
  }
  auto meta = c10::scalarTypeToTypeMeta(to_aten(dtype));
  size_t nbytes = static_cast<size_t>(numel) * meta.itemsize();

  auto* allocator = c10::GetCPUAllocator();
  c10::Storage storage(c10::Storage::use_byte_size_t{},
                       c10::SymInt(static_cast<int64_t>(nbytes)), allocator,
                       /*resizable=*/false);
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::CPU), meta);
  impl->set_sizes_contiguous(c10::IntArrayRef(sizes.data(), sizes.size()));
  return from_aten(at::Tensor(std::move(impl)));
}

} // namespace detail
} // namespace nt
