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

// The device allocator: the CUDA-arena implementation of hal::Allocator. It
// wraps the internal device pool (device_pool.h), which already provides the
// stable-address guarantee a pre-recorded step relies on — buffers are pooled by
// aligned size and never freed until clear_all(), so a replayed CUDA graph keeps
// referencing valid addresses. hal::Buffer::id is the device pool's opaque
// handle (a CUDA device pointer when CUDA is present).
//
// NOT part of the public Layer 2 surface: lives under core/src/. With no CUDA,
// every entry point is a safe no-op / false, mirroring device_pool.

#include <sonicboom/hal/buffer.h>

namespace sonicboom::hal {

class DeviceAllocator : public Allocator {
 public:
  Buffer allocate(uint64_t bytes) override;
  void deallocate(const Buffer& buf) override;

  bool copy_to_device(const Buffer& dst, const void* src, uint64_t bytes) override;
  bool copy_from_device(void* dst, const Buffer& src, uint64_t bytes) override;
  bool device_copy(const Buffer& dst, const Buffer& src, uint64_t bytes) override;
  bool fill(const Buffer& dst, int value, uint64_t bytes) override;
  bool synchronize() override;
};

}  // namespace sonicboom::hal
