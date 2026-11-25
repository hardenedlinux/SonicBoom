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
#include <utility>
#include <vector>

#include <unistd.h>

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

// Host byte ranges pinned in place for direct H2D copies (host_register_pages).
// [begin, end) are page-aligned and lie inside the caller's allocation, so a
// page the driver pins is never shared with another registration here.
struct HostRange {
  uintptr_t begin;
  uintptr_t end;
};
std::vector<HostRange> g_host_ranges;  // guarded by g_mutex

uintptr_t page_size() {
  static const uintptr_t page = [] {
    const long v = ::sysconf(_SC_PAGESIZE);
    return v > 0 ? uintptr_t(v) : uintptr_t(4096);
  }();
  return page;
}

// True when [p, p + bytes) intersects a registered range without lying wholly
// inside one. upload() stages such a copy through ordinary memory: the driver
// refuses a pageable H2D copy whose source only partly overlaps pinned pages
// (cudaErrorInvalidValue). Called with no lock held.
bool host_partially_registered(const void* p, uint64_t bytes) noexcept {
  const uintptr_t b = reinterpret_cast<uintptr_t>(p);
  const uintptr_t e = b + bytes;
  try {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const HostRange& r : g_host_ranges)
      if (b < r.end && e > r.begin)
        return !(b >= r.begin && e <= r.end);
    return false;
  } catch (...) {
    return false;
  }
}

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
  if (host_partially_registered(host, bytes)) {
    // A pageable copy whose source straddles the edge of a registered range
    // fails with cudaErrorInvalidValue (measured: the first bytes of the tensor
    // after an expert share the expert range's last pinned page). Reader tensors
    // are 32-byte aligned, so the tensors either side of a registered range
    // share a page with it by construction. Stage the bytes in ordinary memory
    // instead: this is a load-time weight upload, never a per-step copy.
    std::vector<std::byte> tmp(static_cast<size_t>(bytes));
    std::memcpy(tmp.data(), host, static_cast<size_t>(bytes));
    if (cudaMemcpy(reinterpret_cast<void*>(handle), tmp.data(), bytes,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      (void)cudaGetLastError();  // reported by the return value
      return false;
    }
    return true;
  }
  if (cudaMemcpy(reinterpret_cast<void*>(handle), host, bytes,
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return true;
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

bool host_register_pages(const void* p, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!p || bytes == 0) return false;
  const uintptr_t pg = page_size();
  const uintptr_t begin = reinterpret_cast<uintptr_t>(p) & ~(pg - 1);
  const uintptr_t end =
      (reinterpret_cast<uintptr_t>(p) + bytes + pg - 1) & ~(pg - 1);
  {
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const HostRange& r : g_host_ranges)
      if (begin >= r.begin && end <= r.end) return true;  // already pinned
  }
  if (cudaHostRegister(reinterpret_cast<void*>(begin), end - begin,
                       cudaHostRegisterDefault) != cudaSuccess) {
    // The refusal (e.g. a read-only file mapping the driver will not pin) is
    // handled by leaving the range pageable. Clear the driver's error first:
    // it is sticky, and an unrelated cudaGetLastError() later in the process
    // would otherwise report this refusal.
    (void)cudaGetLastError();
    return false;
  }
  // The registration is the source of a copy only; the host keeps writing these
  // bytes exactly as before, so a caller that registered the same pages twice
  // (two pools over one weights object) is folded into one range.
  std::lock_guard<std::mutex> lk(g_mutex);
  g_host_ranges.push_back(HostRange{begin, end});
  return true;
#else
  (void)p; (void)bytes;
  return false;
#endif
}

bool host_is_registered(const void* p, uint64_t bytes) noexcept {
#ifdef SONICBOOM_USE_CUDA
  if (!p || bytes == 0) return false;
  try {
    const uintptr_t pg = page_size();
    const uintptr_t begin = reinterpret_cast<uintptr_t>(p) & ~(pg - 1);
    const uintptr_t end =
        (reinterpret_cast<uintptr_t>(p) + bytes + pg - 1) & ~(pg - 1);
    std::lock_guard<std::mutex> lk(g_mutex);
    for (const HostRange& r : g_host_ranges)
      if (begin >= r.begin && end <= r.end) return true;
    return false;
  } catch (...) {  // a noexcept query: never let the lock surface an error
    return false;
  }
#else
  (void)p; (void)bytes;
  return false;
#endif
}

void host_unregister_pages(const void* p, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (!p || bytes == 0) return;
  const uintptr_t pg = page_size();
  const uintptr_t begin = reinterpret_cast<uintptr_t>(p) & ~(pg - 1);
  const uintptr_t end =
      (reinterpret_cast<uintptr_t>(p) + bytes + pg - 1) & ~(pg - 1);
  std::lock_guard<std::mutex> lk(g_mutex);
  for (auto it = g_host_ranges.begin(); it != g_host_ranges.end(); ++it) {
    if (it->begin != begin || it->end != end) continue;
    g_host_ranges.erase(it);
    if (cudaHostUnregister(reinterpret_cast<void*>(begin)) != cudaSuccess)
      (void)cudaGetLastError();  // teardown: a refusal is not a later error
    return;
  }
#else
  (void)p; (void)bytes;
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
  // Test hooks: SONICBOOM_EXPERT_UPLOAD_DELAY_US spaces the issued copies by a
  // fixed amount, and SONICBOOM_EXPERT_UPLOAD_JITTER_US=<seed>:<max_us> adds a
  // seeded random sleep of up to max_us to each. Both exist so a caller whose
  // result depends on upload timing fails visibly; an output must be
  // bit-identical with either (or both) on. Off unless set.
  static const uint64_t delay_us = [] {
    const char* env = std::getenv("SONICBOOM_EXPERT_UPLOAD_DELAY_US");
    if (!env || !*env) return uint64_t{0};
    char* end = nullptr;
    const unsigned long long v = std::strtoull(env, &end, 10);
    return (end == env || *end != '\0') ? uint64_t{0} : uint64_t(v);
  }();
  struct Jitter {
    uint64_t max_us = 0;
    uint64_t state = 0x9E3779B97F4A7C15ull;
  };
  static Jitter jitter = [] {
    Jitter j;
    const char* env = std::getenv("SONICBOOM_EXPERT_UPLOAD_JITTER_US");
    if (!env || !*env) return j;
    char* end = nullptr;
    const unsigned long long seed = std::strtoull(env, &end, 10);
    if (end == env || *end != ':') return Jitter{};
    const unsigned long long max_us = std::strtoull(end + 1, &end, 10);
    if (*end != '\0') return Jitter{};
    j.max_us = uint64_t(max_us);
    j.state = uint64_t(seed) * 0x9E3779B97F4A7C15ull + 0x1234567ull;
    if (j.state == 0) j.state = 1;
    return j;
  }();
  const auto sleep_after_copy = [] {
    if (jitter.max_us != 0) {
      jitter.state ^= jitter.state >> 12;
      jitter.state ^= jitter.state << 25;
      jitter.state ^= jitter.state >> 27;
      const uint64_t r = jitter.state * 0x2545F4914F6CDD1Dull;
      std::this_thread::sleep_for(
          std::chrono::microseconds(r % (jitter.max_us + 1)));
    }
    if (delay_us != 0)
      std::this_thread::sleep_for(std::chrono::microseconds(delay_us));
  };
  const unsigned char* src = static_cast<const unsigned char*>(host);
  if (host_is_registered(src, bytes)) {
    // Direct DMA out of the registered host bytes: no host memcpy, no pinned
    // half. The event still marks the end of the copy on the copy stream, so
    // commit()/fence() order it against the compute stream as usual.
    if (cudaMemcpyAsync(reinterpret_cast<void*>(handle), src, bytes,
                        cudaMemcpyHostToDevice, s->copy) != cudaSuccess ||
        cudaEventRecord(s->ev[s->cur], s->copy) != cudaSuccess) {
      (void)cudaGetLastError();  // reported by the return value
      return false;
    }
    s->pending[s->cur] = true;
    sleep_after_copy();
    return true;
  }
  uint64_t off = 0;
  while (off < bytes) {
    if (s->used == s->half) {
      s->cur ^= 1;
      s->used = 0;
      if (s->pending[s->cur]) {
        // The half is about to be overwritten by the host. With the default
        // 16 MiB this is the reuse of the other half between nodes; a small
        // staging buffer (the tests use 64 KiB) reaches it mid-node.
        if (cudaEventSynchronize(s->ev[s->cur]) != cudaSuccess) {
          (void)cudaGetLastError();  // reported by the return value
          return false;
        }
        s->pending[s->cur] = false;
      }
    }
    const uint64_t chunk = std::min<uint64_t>(bytes - off, s->half - s->used);
    unsigned char* slot =
        s->pinned + static_cast<uint64_t>(s->cur) * s->half + s->used;
    std::memcpy(slot, src + off, static_cast<size_t>(chunk));
    if (cudaMemcpyAsync(reinterpret_cast<void*>(handle + off), slot, chunk,
                        cudaMemcpyHostToDevice, s->copy) != cudaSuccess ||
        cudaEventRecord(s->ev[s->cur], s->copy) != cudaSuccess) {
      (void)cudaGetLastError();  // reported by the return value
      return false;
    }
    s->pending[s->cur] = true;
    s->used += chunk;
    off += chunk;
    sleep_after_copy();
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

uint64_t event_create() {
#ifdef SONICBOOM_USE_CUDA
  cudaEvent_t e = nullptr;
  if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return reinterpret_cast<uint64_t>(e);
#else
  return 0;
#endif
}

void event_destroy(uint64_t ev) {
#ifdef SONICBOOM_USE_CUDA
  if (!ev) return;
  if (cudaEventDestroy(reinterpret_cast<cudaEvent_t>(ev)) != cudaSuccess)
    (void)cudaGetLastError();
#else
  (void)ev;
#endif
}

bool event_record(uint64_t ev, Staging* s) {
#ifdef SONICBOOM_USE_CUDA
  if (!ev || !s) return false;
  if (cudaEventRecord(reinterpret_cast<cudaEvent_t>(ev), s->copy) !=
      cudaSuccess) {
    (void)cudaGetLastError();  // reported by the return value; not sticky
    return false;
  }
  return true;
#else
  (void)ev; (void)s;
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
