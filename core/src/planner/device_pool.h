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
//
// Ownership. A model's device memory has to go back to the driver when the
// model does -- the free-list retention that makes one session cheap makes a
// second, different model OOM. Every block therefore records the owner active
// when it was acquired: a target installs an OwnerScope for the duration of its
// load and of each of its session/execution calls, and blocks acquired before
// any model exists stay unowned (0). release_owner() hands every block tagged
// with that owner -- live or parked in the free list -- back to the driver, so
// call it only when the owner's last session is gone (the C API ties it to the
// model bundle's last reference). trim() returns every retained block
// regardless of owner without touching anything live.

namespace sonicboom::planner::device {

// True when a usable device is present (cached). Always false without CUDA.
bool available() noexcept;

// Acquire a device buffer of at least `bytes` (rounded up internally). Returns
// an opaque handle, or 0 on failure.
uint64_t acquire(uint64_t bytes);

// Like acquire(), but never satisfied from the free list. A captured graph's
// launches keep pointers into blocks the pool has handed back after it, so a
// long-lived buffer that took a free block could be overwritten by a replay of
// an older graph. Buffers that outlive captures (the grammar gather table, for
// one) take a fresh allocation instead.
uint64_t acquire_fresh(uint64_t bytes);

// Return `handle` (of the given size) to the free list. No-op on handle == 0.
void release(uint64_t handle, uint64_t bytes);

// Copy between host and device. Return false on failure.
bool upload(const void* host, uint64_t handle, uint64_t bytes);      // H2D
bool download(uint64_t handle, void* host, uint64_t bytes);          // D2H
bool device_copy(uint64_t src, uint64_t dst, uint64_t bytes);        // D2D

// Host bytes pinned in place for the staging pipeline's direct copies. A range
// registered here can be the source of an asynchronous H2D copy as it sits --
// no host memcpy into the pinned half and no second copy of the model in a
// pinned arena. The expert pool registers the GGUF reader's in-RAM expert bytes
// once at load; the registration rounds out to page boundaries, so the caller
// passes the same pointer and length to register and to unregister. A range the
// driver refuses (not enough pinnable memory) is not registered and keeps the
// staged path. host_is_registered() is false for a range that crosses the edge
// of a registered one. Without CUDA registration fails and the query is false.
bool host_register_pages(const void* p, uint64_t bytes);
bool host_is_registered(const void* p, uint64_t bytes) noexcept;
void host_unregister_pages(const void* p, uint64_t bytes);

// Zero-fill a device buffer. Runs on the current stream (see
// core/src/nn/cuda_stream.h), so it is capturable into a CUDA Graph. On the
// legacy default stream this is synchronous, matching the old cudaMemset.
bool memset_device(uint64_t handle, int value, uint64_t bytes);

// A pinned, double-buffered staging pipeline for large host-to-device copies
// (the routed experts' matrices). The host copies pageable bytes into a pinned
// half; the half is then sent to the device asynchronously on a dedicated
// non-blocking copy stream. A copy larger than a half is split into chunks
// across the two halves. The ordering is carried by events, not by host
// synchronisation:
//
//   staging_begin()   the copy stream waits for everything already launched on
//                     the current stream (so a slot read by an earlier node's
//                     kernel is not overwritten under it)
//   staging_upload()  host memcpy into the pinned half + cudaMemcpyAsync
//   staging_commit()  the current stream waits for every issued copy
//   staging_fence()   record the same "all issued copies" event without making
//                     the current stream wait (a prefetch pipeline runs ahead)
//   staging_wait()    make the current stream wait for that event later
//   staging_chain()   make one pipeline's copy stream wait for another's fence,
//                     so two pipelines never write a slot concurrently
//
// The only host wait is the reuse of a pinned half whose previous copy has not
// completed yet (cudaEventQuery/Sync at that point); the default 16 MiB holds
// a whole node's misses, so it is not reached within a node. An upload whose
// bytes lie entirely in a range registered with host_register_pages() is sent
// straight from the host bytes: no host memcpy and no pinned half consumed
// (the half bookkeeping is simply left alone for that call).
struct Staging;  // opaque; the CUDA types stay in device_pool.cpp

// `bytes` of pinned host memory (split in two). Returns null without a device.
Staging* staging_create(uint64_t bytes);
void staging_destroy(Staging* s);
bool staging_begin(Staging* s);
bool staging_upload(Staging* s, const void* host, uint64_t handle,
                    uint64_t bytes);
bool staging_commit(Staging* s);
bool staging_fence(Staging* s);
bool staging_wait(Staging* s);
bool staging_chain(Staging* s, Staging* after);

// A completion mark for one upload (an opaque cudaEvent_t handle; 0 on
// failure). event_record() records the mark on the pipeline's copy stream, so
// it comes after every copy issued there so far. The expert pool uses one mark
// per selection entry; the routed-expert kernel makes the compute stream wait
// for it just before using the entry's expert, so a resident expert computes
// while a missing one is still travelling instead of the node waiting for its
// last upload. Re-recording a mark is allowed: a wait keeps the state the event
// had when it was issued.
uint64_t event_create();
void event_destroy(uint64_t ev);
bool event_record(uint64_t ev, Staging* s);

// Block until all device work is complete.
bool synchronize();

// Free device memory right now (what a new allocation could take). False when
// no device is present or the query fails.
bool free_bytes(uint64_t* bytes);

// Pool accounting. "live" is what callers currently hold (KV caches, staging,
// activations); "retained" is what was released back to the pool and can be
// handed to a new session without a cudaMalloc. The two are reported
// separately: a session's live bytes must fall to zero after release even while
// its buffers remain retained for reuse.
struct Stats {
  uint64_t live_bytes = 0;
  uint64_t live_blocks = 0;
  uint64_t retained_bytes = 0;
  uint64_t retained_blocks = 0;
};
Stats stats() noexcept;

// Free every pooled buffer (used at teardown / model reload).
void clear_all();

// --- Ownership --------------------------------------------------------------

// A new nonzero owner id. Ids are never reused, so a released owner's blocks
// cannot be confused with a later model's.
uint64_t create_owner();

// The owner acquire()/acquire_fresh() will tag a block with; 0 means unowned.
uint64_t current_owner() noexcept;

// Install `owner` as the current owner and return the previous one (the
// building block behind the RAII scope in <sonicboom/model/plan.h>).
uint64_t set_current_owner(uint64_t owner) noexcept;

// How many owners have been created and not yet released.
uint32_t live_owners() noexcept;

// Installs `owner` as the current owner for its lifetime (nesting restores the
// previous owner). Every model load and every session call that can allocate
// runs under one.
class OwnerScope {
 public:
  explicit OwnerScope(uint64_t owner) noexcept;
  ~OwnerScope();
  OwnerScope(const OwnerScope &) = delete;
  OwnerScope &operator=(const OwnerScope &) = delete;

 private:
  uint64_t prev_;
};

// Return every block tagged with `owner` -- live or in the free list -- to the
// driver, after synchronizing the device. Invalid (0) is a no-op.
void release_owner(uint64_t owner);

// Return every retained (free-list) block to the driver now. Live blocks are
// untouched; a model still using the device simply allocates again on demand.
void trim();

} // namespace sonicboom::planner::device
