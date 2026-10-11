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

#include "device_allocator.h"

#include "device_pool.h"

namespace sonicboom::hal {

Buffer DeviceAllocator::allocate(uint64_t bytes) {
  const uint64_t handle = planner::device::acquire(bytes);
  return Buffer{handle, bytes, DeviceId{0}, MemoryKind::DeviceLocal};
}

void DeviceAllocator::deallocate(const Buffer& buf) {
  planner::device::release(buf.id, buf.size_bytes);
}

bool DeviceAllocator::copy_to_device(const Buffer& dst, const void* src,
                                     uint64_t bytes) {
  return planner::device::upload(src, dst.id, bytes);
}

bool DeviceAllocator::copy_from_device(void* dst, const Buffer& src,
                                       uint64_t bytes) {
  return planner::device::download(src.id, dst, bytes);
}

bool DeviceAllocator::device_copy(const Buffer& dst, const Buffer& src,
                                  uint64_t bytes) {
  return planner::device::device_copy(src.id, dst.id, bytes);
}

bool DeviceAllocator::fill(const Buffer& dst, int value, uint64_t bytes) {
  return planner::device::memset_device(dst.id, value, bytes);
}

bool DeviceAllocator::synchronize() {
  return planner::device::synchronize();
}

}  // namespace sonicboom::hal
