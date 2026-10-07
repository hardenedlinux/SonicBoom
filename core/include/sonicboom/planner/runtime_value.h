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

// Runtime tensor value (Phase 6 M3). The executor's live-set holds one of these
// per tensor instead of a bare sx::Bytes, so a tensor can be resident in a
// non-host memory space (a device buffer) rather than host float32 bytes. The
// device handle is opaque at Layer 2: the concrete interpretation (a CUDA
// device pointer) lives below the boundary in core/src/ (the GPU SonicBackend
// and the executor's transfer/alloc tasks). Host storage stays the v0 default,
// so the existing CPU backends are unchanged apart from unwrapping `.host`.

#include <sonicboom/sx/exec.h>

#include <cstdint>
#include <utility>

namespace sonicboom::planner {

// An opaque non-host buffer handle. `handle` holds the raw device address (or a
// pool index, depending on the backend); `size_bytes` is its byte length.
struct DeviceBuffer {
  uint64_t handle = 0;
  uint64_t size_bytes = 0;
};

// A runtime tensor value: exactly one storage location. A tensor moves between
// host and device via an explicit Transfer task; a value's `on_device` flag says
// which storage is authoritative.
struct TensorValue {
  bool on_device = false;
  sx::Bytes host;       // host float32 bytes (on_device == false)
  DeviceBuffer device;  // device buffer (on_device == true)

  static TensorValue from_host(sx::Bytes b) {
    TensorValue v;
    v.host = std::move(b);
    return v;
  }

  static TensorValue from_device(uint64_t handle, uint64_t size_bytes) {
    TensorValue v;
    v.on_device = true;
    v.device.handle = handle;
    v.device.size_bytes = size_bytes;
    return v;
  }
};

} // namespace sonicboom::planner
