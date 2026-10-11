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

// HAL buffer + allocator surface (Step 1). A backend owns memory in one or more
// memory spaces; an Allocator mints buffers with a stable address for their
// lifetime, which is the property a pre-recorded command buffer (a CUDA graph,
// a Vulkan command buffer) relies on when it is replayed later: the recorded
// commands reference the same buffer ids every replay.
//
// `Buffer::id` is opaque to Layer 2 — a CUDA device pointer, a Vulkan
// VkDeviceMemory handle, or a host arena index, resolved behind the backend.
// No CUDA / Vulkan / native-torch type appears here.

#include <cstdint>

namespace sonicboom::hal {

// An opaque, HAL-minted device id. Not a CUDA ordinal or a Vulkan index.
struct DeviceId {
  uint64_t value = 0;
  friend bool operator==(DeviceId a, DeviceId b) { return a.value == b.value; }
  friend bool operator!=(DeviceId a, DeviceId b) { return a.value != b.value; }
};

// The memory space a buffer lives in.
enum class MemoryKind { Host, DeviceLocal, HostVisible, Managed };

// A stable-address buffer allocated from an Allocator.
struct Buffer {
  uint64_t id = 0;      // HAL-assigned, stable for the buffer's lifetime
  uint64_t size_bytes = 0;
  DeviceId device;
  MemoryKind kind = MemoryKind::DeviceLocal;

  explicit operator bool() const { return id != 0; }
};

// Allocates and moves buffers within a backend's memory spaces. The
// "allocate once with a fixed address, keep alive for the plan" contract is what
// lets a pre-recorded step be replayed against valid memory. The synchronous
// copy forms are the simple API; async copies go through the command recorder
// (hal/command.h's StepProgram is the pre-record/replay seam today).
class Allocator {
 public:
  virtual ~Allocator() = default;

  // Allocate a buffer of at least `bytes` (backend rounds up). id == 0 on
  // failure. The address is stable until deallocate.
  virtual Buffer allocate(uint64_t bytes) = 0;

  // Return a buffer's storage to the backend (pooled or freed at its choice).
  virtual void deallocate(const Buffer& buf) = 0;

  // Host <-> device / device <-> device copies. Return false on failure.
  virtual bool copy_to_device(const Buffer& dst, const void* src,
                              uint64_t bytes) = 0;   // H2D
  virtual bool copy_from_device(void* dst, const Buffer& src,
                                uint64_t bytes) = 0;  // D2H
  virtual bool device_copy(const Buffer& dst, const Buffer& src,
                           uint64_t bytes) = 0;       // D2D

  // Zero / value-fill a device buffer. Must be recordable into a command
  // buffer (so it is async on the backend's current stream when it has one).
  virtual bool fill(const Buffer& dst, int value, uint64_t bytes) = 0;

  // Block until all backend work completes (a fence).
  virtual bool synchronize() = 0;
};

}  // namespace sonicboom::hal
