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

// A Param's storage: the planner's device buffer pool when a CUDA device is
// present, host vectors otherwise. The pool speaks opaque handles, so this
// translation unit carries no CUDA types either; only adamw.cu does.

#include <sonicboom/training/param.h>

#include "../planner/device_pool.h"

#ifdef SONICBOOM_USE_CUDA
#include "training_internal.h"
#endif

#include <algorithm>
#include <cstring>
#include <utility>
#include <vector>

namespace sonicboom::training {

namespace device = sonicboom::planner::device;

struct Param::Impl {
  std::size_t n = 0;
  bool on_device = false;
  std::uint64_t step = 0;

  // Host-resident storage (on_device == false).
  std::vector<float> hp, hg, hm1, hm2;

  // Device-resident storage (on_device == true), as pool handles.
  std::uint64_t dp = 0, dg = 0, dm1 = 0, dm2 = 0;

  std::uint64_t bytes() const {
    return static_cast<std::uint64_t>(n) * sizeof(float);
  }

  ~Impl() {
    if (!on_device)
      return;
    for (std::uint64_t h : {dp, dg, dm1, dm2})
      device::release(h, bytes());
  }

  bool upload_param(const float* src) {
    if (!on_device) {
      std::memcpy(hp.data(), src, bytes());
      return true;
    }
    return device::upload(src, dp, bytes());
  }

  bool download(const std::uint64_t handle, const std::vector<float>& host,
                float* out) const {
    if (!on_device) {
      std::memcpy(out, host.data(), bytes());
      return true;
    }
    return device::download(handle, out, bytes());
  }
};

Param::Param() : impl_(std::make_unique<Impl>()) {}
Param::~Param() = default;
Param::Param(Param&&) noexcept = default;
Param& Param::operator=(Param&&) noexcept = default;

std::expected<std::unique_ptr<Param>, std::string> Param::create(
    std::span<const float> init) {
  if (init.empty())
    return std::unexpected("AdamW parameter must have at least one element");

  auto p = std::unique_ptr<Param>(new Param());
  Impl &im = *p->impl_;
  im.n = init.size();
  const std::uint64_t bytes = im.bytes();

  if (device::available()) {
    im.dp = device::acquire(bytes);
    im.dg = device::acquire(bytes);
    im.dm1 = device::acquire(bytes);
    im.dm2 = device::acquire(bytes);
    if (im.dp && im.dg && im.dm1 && im.dm2) {
      im.on_device = true;
      const bool ok = device::upload(init.data(), im.dp, bytes) &&
                      device::memset_device(im.dg, 0, bytes) &&
                      device::memset_device(im.dm1, 0, bytes) &&
                      device::memset_device(im.dm2, 0, bytes);
      if (ok)
        return p;
      im.on_device = false;
    }
    // A partial pool allocation: return everything and use host memory.
    for (std::uint64_t h : {im.dp, im.dg, im.dm1, im.dm2})
      device::release(h, bytes);
    im.dp = im.dg = im.dm1 = im.dm2 = 0;
  }

  im.hp.assign(init.begin(), init.end());
  im.hg.assign(im.n, 0.0f);
  im.hm1.assign(im.n, 0.0f);
  im.hm2.assign(im.n, 0.0f);
  return p;
}

std::size_t Param::size() const noexcept { return impl_->n; }
bool Param::on_device() const noexcept { return impl_->on_device; }
std::uint64_t Param::step_count() const noexcept { return impl_->step; }

std::uint64_t Param::param_handle() const noexcept { return impl_->dp; }
std::uint64_t Param::grad_handle() const noexcept { return impl_->dg; }
std::uint64_t Param::m1_handle() const noexcept { return impl_->dm1; }
std::uint64_t Param::m2_handle() const noexcept { return impl_->dm2; }

void Param::zero_grad() {
  if (impl_->on_device) {
    device::memset_device(impl_->dg, 0, impl_->bytes());
  } else {
    std::fill(impl_->hg.begin(), impl_->hg.end(), 0.0f);
  }
}

bool Param::set_grad(std::span<const float> grad) {
  if (grad.size() != impl_->n)
    return false;
  if (impl_->on_device)
    return device::upload(grad.data(), impl_->dg, impl_->bytes());
  std::memcpy(impl_->hg.data(), grad.data(), impl_->bytes());
  return true;
}

bool Param::step(const AdamWConfig &cfg) {
  const std::uint64_t next = impl_->step + 1;
  bool ok = false;
  if (impl_->on_device) {
#ifdef SONICBOOM_USE_CUDA
    ok = detail::adamw_dev(impl_->dp, impl_->dg, impl_->dm1, impl_->dm2,
                           impl_->n, cfg, next);
#endif
  } else {
    adamw_f32(impl_->hp.data(), impl_->hg.data(), impl_->hm1.data(),
              impl_->hm2.data(), impl_->n, cfg, next);
    ok = true;
  }
  if (ok)
    impl_->step = next;
  return ok;
}

bool Param::read_param(std::span<float> out) const {
  if (out.size() != impl_->n)
    return false;
  return impl_->download(impl_->dp, impl_->hp, out.data());
}

bool Param::read_grad(std::span<float> out) const {
  if (out.size() != impl_->n)
    return false;
  return impl_->download(impl_->dg, impl_->hg, out.data());
}

bool Param::read_m1(std::span<float> out) const {
  if (out.size() != impl_->n)
    return false;
  return impl_->download(impl_->dm1, impl_->hm1, out.data());
}

bool Param::read_m2(std::span<float> out) const {
  if (out.size() != impl_->n)
    return false;
  return impl_->download(impl_->dm2, impl_->hm2, out.data());
}

} // namespace sonicboom::training
