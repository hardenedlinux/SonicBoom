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

#include "device_pool.h"

#ifdef SONICBOOM_USE_CUDA
#include <cuda_runtime.h>
#include "../nn/cuda_stream.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sonicboom::planner::device {

namespace {

constexpr uint64_t kAlignment = 256;

uint64_t align_up(uint64_t n) {
  const uint64_t rem = n % kAlignment;
  return rem == 0 ? n : n + (kAlignment - rem);
}

#ifdef SONICBOOM_USE_CUDA

std::mutex g_mutex;
bool g_checked = false;
bool g_available = false;
// Free buffers keyed by aligned size.
std::unordered_map<uint64_t, std::vector<uint64_t>> g_free;

// One record per block the pool owns, live or parked in the free list. `size`
// is the aligned acquire-time size, so release() accounts a block exactly even
// when the caller passes a size that does not match the acquire (the executor
// has one release path that passes 0). `owner` is the owner active at acquire;
// reusing a free block retags it. `live` distinguishes held from retained and
// makes a second release of the same handle a no-op.
struct Block {
  uint64_t size = 0;
  uint64_t owner = 0;
  bool live = false;
};
std::unordered_map<uint64_t, Block> g_blocks;
Stats g_stats;

thread_local uint64_t t_owner = 0;
std::atomic<uint64_t> g_next_owner{1};
std::atomic<uint32_t> g_live_owners{0};

// Return the doomed blocks to the driver, first unhooking each from its free
// list. The caller holds g_mutex. Accounting follows the block's state: a live
// block leaves the live figures, a retained one the retained figures.
void free_blocks_locked(const std::unordered_set<uint64_t>& doomed) {
  if (doomed.empty()) return;
  for (auto& kv : g_free) {
    auto& v = kv.second;
    v.erase(std::remove_if(v.begin(), v.end(),
                           [&](uint64_t h) { return doomed.count(h) != 0; }),
            v.end());
  }
  for (uint64_t h : doomed) {
    auto it = g_blocks.find(h);
    if (it == g_blocks.end()) continue;
    const Block b = it->second;
    if (b.live) {
      g_stats.live_blocks--;
      g_stats.live_bytes -= b.size;
    } else {
      g_stats.retained_blocks--;
      g_stats.retained_bytes -= b.size;
    }
    cudaFree(reinterpret_cast<void*>(h));
    g_blocks.erase(it);
  }
}

#endif

} // namespace

bool available() noexcept {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::mutex> lk(g_mutex);
  if (!g_checked) {
    int n = 0;
    g_available = (cudaGetDeviceCount(&n) == cudaSuccess && n > 0);
    g_checked = true;
  }
  return g_available;
#else
  return false;
#endif
}

uint64_t acquire(uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (bytes == 0 || !available()) return 0;
  const uint64_t size = align_up(bytes);
  std::lock_guard<std::mutex> lk(g_mutex);
  auto it = g_free.find(size);
  if (it != g_free.end() && !it->second.empty()) {
    const uint64_t h = it->second.back();
    it->second.pop_back();
    g_stats.retained_blocks--;
    g_stats.retained_bytes -= size;
    g_stats.live_blocks++;
    g_stats.live_bytes += size;
    Block &b = g_blocks[h];
    b.size = size;
    b.owner = t_owner;  // reuse retags the block to the acquiring owner
    b.live = true;
    return h;
  }
  void* p = nullptr;
  if (cudaMalloc(&p, size) != cudaSuccess) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "[device-pool] acquire %llu bytes failed: %s "
                         "(free %llu of %llu)\n",
                 (unsigned long long)size, cudaGetErrorString(cudaGetLastError()),
                 (unsigned long long)free_b, (unsigned long long)total_b);
    return 0;
  }
  const uint64_t h = reinterpret_cast<uint64_t>(p);
  g_blocks[h] = Block{size, t_owner, true};
  g_stats.live_blocks++;
  g_stats.live_bytes += size;
  return h;
#else
  (void)bytes;
  return 0;
#endif
}

uint64_t acquire_fresh(uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (bytes == 0 || !available()) return 0;
  const uint64_t size = align_up(bytes);
  std::lock_guard<std::mutex> lk(g_mutex);
  void* p = nullptr;
  if (cudaMalloc(&p, size) != cudaSuccess) {
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "[device-pool] acquire_fresh %llu bytes failed: %s "
                         "(free %llu of %llu)\n",
                 (unsigned long long)size, cudaGetErrorString(cudaGetLastError()),
                 (unsigned long long)free_b, (unsigned long long)total_b);
    return 0;
  }
  const uint64_t h = reinterpret_cast<uint64_t>(p);
  g_blocks[h] = Block{size, t_owner, true};
  g_stats.live_blocks++;
  g_stats.live_bytes += size;
  return h;
#else
  (void)bytes;
  return 0;
#endif
}

void release(uint64_t handle, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (handle == 0) return;
  std::lock_guard<std::mutex> lk(g_mutex);
  // The acquire-time size is authoritative: one executor path releases with 0.
  auto it = g_blocks.find(handle);
  if (it == g_blocks.end() || !it->second.live)
    return;  // unknown or already released
  const uint64_t size = it->second.size;
  (void)bytes;
  it->second.live = false;
  g_free[size].push_back(handle);
  g_stats.live_blocks--;
  g_stats.live_bytes -= size;
  g_stats.retained_blocks++;
  g_stats.retained_bytes += size;
#else
  (void)handle;
  (void)bytes;
#endif
}

bool upload(const void* host, uint64_t handle, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!host || handle == 0 || bytes == 0) return false;
  return cudaMemcpy(reinterpret_cast<void*>(handle), host, bytes,
                    cudaMemcpyHostToDevice) == cudaSuccess;
#else
  (void)host; (void)handle; (void)bytes;
  return false;
#endif
}

bool download(uint64_t handle, void* host, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!host || handle == 0 || bytes == 0) return false;
  // A synchronous cudaMemcpy cannot be recorded into a CUDA graph. Under
  // capture (see cuda::CaptureScope) skip the copy: the caller recovers the
  // result from a persistent device slot after replay, so the graph records
  // only the capturable device work.
  if (sonicboom::cuda::boom_capturing()) return true;
  return cudaMemcpy(host, reinterpret_cast<const void*>(handle), bytes,
                    cudaMemcpyDeviceToHost) == cudaSuccess;
#else
  (void)handle; (void)host; (void)bytes;
  return false;
#endif
}

bool device_copy(uint64_t src, uint64_t dst, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (src == 0 || dst == 0 || bytes == 0) return false;
  return cudaMemcpyAsync(reinterpret_cast<void*>(dst),
                         reinterpret_cast<const void*>(src), bytes,
                         cudaMemcpyDeviceToDevice,
                         sonicboom::cuda::boom_current_stream()) == cudaSuccess;
#else
  (void)src; (void)dst; (void)bytes;
  return false;
#endif
}

bool memset_device(uint64_t handle, int value, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (handle == 0 || bytes == 0) return false;
  return cudaMemsetAsync(reinterpret_cast<void*>(handle), value, bytes,
                         sonicboom::cuda::boom_current_stream()) == cudaSuccess;
#else
  (void)handle; (void)value; (void)bytes;
  return false;
#endif
}

// --- pinned staging pipeline -------------------------------------------------

#ifdef SONICBOOM_USE_CUDA

struct Staging {
  unsigned char* pinned = nullptr;
  uint64_t half = 0;      // bytes per half; the two halves are contiguous
  uint64_t used = 0;      // bytes written in the current half
  int cur = 0;
  cudaStream_t copy = nullptr;
  cudaEvent_t ev[2] = {nullptr, nullptr};  // last copy issued out of each half
  bool pending[2] = {false, false};
  cudaEvent_t wake = nullptr;   // compute stream -> copy stream
  cudaEvent_t ready = nullptr;  // copy stream -> compute stream
};

#endif

Staging* staging_create(uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!available()) return nullptr;
  // Two halves of 256-byte multiples; a very small request is still usable.
  uint64_t half = (bytes / 2) & ~uint64_t(kAlignment - 1);
  if (half < kAlignment) half = kAlignment;
  Staging* s = new (std::nothrow) Staging;
  if (!s) return nullptr;
  s->half = half;
  if (cudaMallocHost(reinterpret_cast<void**>(&s->pinned), 2 * half) !=
      cudaSuccess) {
    delete s;
    return nullptr;
  }
  if (cudaStreamCreateWithFlags(&s->copy, cudaStreamNonBlocking) !=
      cudaSuccess) {
    cudaFreeHost(s->pinned);
    delete s;
    return nullptr;
  }
  for (int i = 0; i < 2; ++i)
    cudaEventCreateWithFlags(&s->ev[i], cudaEventDisableTiming);
  cudaEventCreateWithFlags(&s->wake, cudaEventDisableTiming);
  cudaEventCreateWithFlags(&s->ready, cudaEventDisableTiming);
  return s;
#else
  (void)bytes;
  return nullptr;
#endif
}

void staging_destroy(Staging* s) {
#ifdef SONICBOOM_USE_CUDA
  if (!s) return;
  cudaStreamSynchronize(s->copy);
  for (int i = 0; i < 2; ++i)
    if (s->ev[i]) cudaEventDestroy(s->ev[i]);
  if (s->wake) cudaEventDestroy(s->wake);
  if (s->ready) cudaEventDestroy(s->ready);
  if (s->copy) cudaStreamDestroy(s->copy);
  if (s->pinned) cudaFreeHost(s->pinned);
  delete s;
#else
  (void)s;
#endif
}

bool staging_begin(Staging* s) {
#ifdef SONICBOOM_USE_CUDA
  if (!s) return false;
  // Everything already launched on the compute stream has to be ordered before
  // the copies: a slot handed back by the LRU may still be read by an earlier
  // node's kernel. The synchronous cudaMemcpy this replaces got that ordering
  // implicitly; here it is an event.
  if (cudaEventRecord(s->wake, sonicboom::cuda::boom_current_stream()) !=
      cudaSuccess)
    return false;
  return cudaStreamWaitEvent(s->copy, s->wake, 0) == cudaSuccess;
#else
  (void)s;
  return false;
#endif
}

bool staging_upload(Staging* s, const void* host, uint64_t handle,
                    uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!s || (!host && bytes != 0) || handle == 0) return false;
  if (bytes == 0) return true;
  // Test hook: SONICBOOM_EXPERT_UPLOAD_DELAY_US spaces the issued chunks, so a
  // caller that assumes a fixed upload rate fails visibly. Off unless set.
  static const uint64_t delay_us = [] {
    const char* env = std::getenv("SONICBOOM_EXPERT_UPLOAD_DELAY_US");
    if (!env || !*env) return uint64_t{0};
    char* end = nullptr;
    const unsigned long long v = std::strtoull(env, &end, 10);
    return (end == env || *end != '\0') ? uint64_t{0} : uint64_t(v);
  }();
  const unsigned char* src = static_cast<const unsigned char*>(host);
  uint64_t off = 0;
  while (off < bytes) {
    if (s->used == s->half) {
      s->cur ^= 1;
      s->used = 0;
      if (s->pending[s->cur]) {
        // The half is about to be overwritten by the host. With the default
        // 16 MiB this is the reuse of the other half between nodes; a small
        // staging buffer (the tests use 64 KiB) reaches it mid-node.
        if (cudaEventSynchronize(s->ev[s->cur]) != cudaSuccess) return false;
        s->pending[s->cur] = false;
      }
    }
    const uint64_t chunk = std::min<uint64_t>(bytes - off, s->half - s->used);
    unsigned char* slot =
        s->pinned + static_cast<uint64_t>(s->cur) * s->half + s->used;
    std::memcpy(slot, src + off, static_cast<size_t>(chunk));
    if (cudaMemcpyAsync(reinterpret_cast<void*>(handle + off), slot, chunk,
                        cudaMemcpyHostToDevice, s->copy) != cudaSuccess)
      return false;
    if (cudaEventRecord(s->ev[s->cur], s->copy) != cudaSuccess) return false;
    s->pending[s->cur] = true;
    s->used += chunk;
    off += chunk;
    if (delay_us != 0)
      std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  }
  return true;
#else
  (void)s; (void)host; (void)handle; (void)bytes;
  return false;
#endif
}

bool staging_commit(Staging* s) {
#ifdef SONICBOOM_USE_CUDA
  if (!s) return false;
  if (cudaEventRecord(s->ready, s->copy) != cudaSuccess) return false;
  return cudaStreamWaitEvent(sonicboom::cuda::boom_current_stream(), s->ready,
                             0) == cudaSuccess;
#else
  (void)s;
  return false;
#endif
}

bool staging_fence(Staging* s) {
#ifdef SONICBOOM_USE_CUDA
  if (!s) return false;
  return cudaEventRecord(s->ready, s->copy) == cudaSuccess;
#else
  (void)s;
  return false;
#endif
}

bool staging_wait(Staging* s) {
#ifdef SONICBOOM_USE_CUDA
  if (!s) return false;
  return cudaStreamWaitEvent(sonicboom::cuda::boom_current_stream(), s->ready,
                             0) == cudaSuccess;
#else
  (void)s;
  return false;
#endif
}

bool staging_chain(Staging* s, Staging* after) {
#ifdef SONICBOOM_USE_CUDA
  if (!s || !after) return false;
  return cudaStreamWaitEvent(s->copy, after->ready, 0) == cudaSuccess;
#else
  (void)s; (void)after;
  return false;
#endif
}

bool synchronize() {
#ifdef SONICBOOM_USE_CUDA
  return cudaStreamSynchronize(0) == cudaSuccess;
#else
  return false;
#endif
}

bool free_bytes(uint64_t* bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!bytes || !available()) return false;
  size_t free_mem = 0;
  size_t total = 0;
  if (cudaMemGetInfo(&free_mem, &total) != cudaSuccess) return false;
  *bytes = static_cast<uint64_t>(free_mem);
  return true;
#else
  (void)bytes;
  return false;
#endif
}

Stats stats() noexcept {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_stats;
#else
  return Stats{};
#endif
}

void clear_all() { trim(); }

void trim() {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::mutex> lk(g_mutex);
  std::unordered_set<uint64_t> doomed;
  for (const auto& kv : g_blocks)
    if (!kv.second.live) doomed.insert(kv.first);
  free_blocks_locked(doomed);
#endif
}

uint64_t create_owner() {
#ifdef SONICBOOM_USE_CUDA
  g_live_owners.fetch_add(1);
  return g_next_owner.fetch_add(1);
#else
  return 0;
#endif
}

uint64_t current_owner() noexcept {
#ifdef SONICBOOM_USE_CUDA
  return t_owner;
#else
  return 0;
#endif
}

uint64_t set_current_owner(uint64_t owner) noexcept {
#ifdef SONICBOOM_USE_CUDA
  const uint64_t prev = t_owner;
  t_owner = owner;
  return prev;
#else
  (void)owner;
  return 0;
#endif
}

uint32_t live_owners() noexcept {
#ifdef SONICBOOM_USE_CUDA
  return g_live_owners.load();
#else
  return 0;
#endif
}

OwnerScope::OwnerScope(uint64_t owner) noexcept : prev_(current_owner()) {
#ifdef SONICBOOM_USE_CUDA
  t_owner = owner;
#else
  (void)owner;
#endif
}

OwnerScope::~OwnerScope() {
#ifdef SONICBOOM_USE_CUDA
  t_owner = prev_;
#endif
}

void release_owner(uint64_t owner) {
#ifdef SONICBOOM_USE_CUDA
  if (owner == 0) return;
  // The owner's runtime is gone, so nothing should be executing against its
  // blocks; synchronize anyway so a launch that outlived its handle cannot read
  // memory the driver is about to reuse.
  cudaDeviceSynchronize();
  std::lock_guard<std::mutex> lk(g_mutex);
  std::unordered_set<uint64_t> doomed;
  for (const auto& kv : g_blocks)
    if (kv.second.owner == owner) doomed.insert(kv.first);
  free_blocks_locked(doomed);
  g_live_owners.fetch_sub(1);
#else
  (void)owner;
#endif
}

} // namespace sonicboom::planner::device
