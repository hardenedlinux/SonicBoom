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

// Internal device buffer pool (Phase 6 M3). NOT part of the public Layer 2
// surface: it lives under core/src/ and speaks in raw opaque handles (uint64_t)
// so the executor, the GPU SonicBackend, and the transfer tasks can allocate,
// reuse, and move device-resident activation buffers without any CUDA type
// leaking into <sonicboom/>. The concrete interpretation of a handle (a CUDA
// device pointer) is resolved inside device_pool.cpp; when the build has no
// CUDA, every entry point is a safe no-op / false so callers degrade gracefully.
//
// Buffers are pooled by aligned byte size and returned to the free list rather
// than freed, so a single-token decode's working set is reused across steps
// instead of being re-malloc'd per node. clear_all() returns them to the driver.

namespace sonicboom::planner::device {

// True when a usable device is present (cached). Always false without CUDA.
bool available() noexcept;

// Acquire a device buffer of at least `bytes` (rounded up internally). Returns
// an opaque handle, or 0 on failure.
uint64_t acquire(uint64_t bytes);

// Return `handle` (of the given size) to the free list. No-op on handle == 0.
void release(uint64_t handle, uint64_t bytes);

// Copy between host and device. Return false on failure.
bool upload(const void* host, uint64_t handle, uint64_t bytes);      // H2D
bool download(uint64_t handle, void* host, uint64_t bytes);          // D2H
bool device_copy(uint64_t src, uint64_t dst, uint64_t bytes);        // D2D

// Block until all device work is complete.
bool synchronize();

// Free every pooled buffer (used at teardown / model reload).
void clear_all();

} // namespace sonicboom::planner::device
