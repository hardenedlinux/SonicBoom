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

// Device-resource ownership (see <sonicboom/model/plan.h>). A model's device
// memory has to return to the driver when the model does: the pool's free-list
// retention is what makes a second session on one model cheap and a second,
// different model OOM. Each model creates an owner id and runs its load and
// session calls under a DeviceOwnerScope, so every block and cached device
// weight they allocate carries the id; release_device_owner() then hands that
// id's memory back exactly once, when the model's last reference goes.
//
// This is the CUDA-free entry point the C API calls, so it compiles in both
// configurations: without a device every call degrades to a no-op.

#include <sonicboom/model/plan.h>

#include "../planner/device_pool.h"

#ifdef SONICBOOM_USE_CUDA
#include <c10/cuda/CUDACachingAllocator.h>

#include <sonicboom/nn/cuda_elementwise.h>
#include <sonicboom/quant/quantized_matmul_cuda.h>

#include "cuda_block.h"
#endif

namespace sonicboom::model {

uint64_t device_owner_create() {
#ifdef SONICBOOM_USE_CUDA
  return planner::device::create_owner();
#else
  return 0;
#endif
}

uint64_t device_set_current_owner(uint64_t owner) noexcept {
#ifdef SONICBOOM_USE_CUDA
  return planner::device::set_current_owner(owner);
#else
  (void)owner;
  return 0;
#endif
}

void release_device_owner(uint64_t owner) {
#ifdef SONICBOOM_USE_CUDA
  if (owner == 0) return;
  // Serialize against execution: the owner's last reference drop (the C API's
  // model bundle) only happens once its sessions are gone, but the execution
  // lock makes that ordering explicit and waits out any operation still inside
  // the core. Recursive, so a caller may hold it already.
  std::lock_guard<std::recursive_mutex> lk(execution_mutex());
  // The pool first: it synchronizes the device, then frees the owner's blocks
  // (live and retained) and forgets its blocks for good. Nothing may allocate
  // for this owner after this point.
  planner::device::release_owner(owner);
  // Device copies of the model's weights: the per-shape caches (quantized and
  // dense) hold them, and the pinned per-layer-embedding mapping points into
  // the model's own buffer, which the caller still holds here.
  quant::cuda::clear_cache_owner(owner);
  nn::cuda::clear_cache_owner(owner);
  if (planner::device::live_owners() != 0) return;
  // Last owner gone: drop the process-lifetime buffers too, so VRAM returns
  // to what it was before the first model. The kernel scratch and the resident
  // activation pool are no-ops while a capture is live (its graph has their
  // pointers baked in — the same rule the scratch growth path follows), and
  // the weight caches only hold untagged entries by now (a tool's, not a
  // model's). Everything regrows on demand.
  cuda_resident_clear();
  quant::cuda::clear_scratch();
  nn::cuda::clear_scratch();
  quant::cuda::clear_cache();
  nn::cuda::clear_cache();
  // Native-torch's caching allocator keeps freed blocks for its own reuse; they
  // are as much "held for reuse" as the pool's free list, and a released model
  // must not leave them behind. Safe with live tensors: only unused blocks go.
  c10::cuda::CUDACachingAllocator::emptyCache();
#else
  (void)owner;
#endif
}

void device_trim() {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::recursive_mutex> lk(execution_mutex());
#endif
  planner::device::trim();
}

} // namespace sonicboom::model
