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

#include <mutex>
#include <unordered_map>
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
// The aligned size each live handle was acquired at, so release() accounts it
// exactly even when the caller passes a size that does not match the acquire
// (the executor has one release path that passes 0).
std::unordered_map<uint64_t, uint64_t> g_sizes;
Stats g_stats;

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
    g_sizes[h] = size;
    return h;
  }
  void* p = nullptr;
  if (cudaMalloc(&p, size) != cudaSuccess) return 0;
  const uint64_t h = reinterpret_cast<uint64_t>(p);
  g_sizes[h] = size;
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
  if (cudaMalloc(&p, size) != cudaSuccess) return 0;
  const uint64_t h = reinterpret_cast<uint64_t>(p);
  g_sizes[h] = size;
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
  auto sit = g_sizes.find(handle);
  if (sit == g_sizes.end()) return;  // unknown or already released
  const uint64_t size = sit->second;
  (void)bytes;
  g_sizes.erase(sit);
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
  if (sonicboom::cuda::sb_capturing()) return true;
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
                         sonicboom::cuda::sb_current_stream()) == cudaSuccess;
#else
  (void)src; (void)dst; (void)bytes;
  return false;
#endif
}

bool memset_device(uint64_t handle, int value, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (handle == 0 || bytes == 0) return false;
  return cudaMemsetAsync(reinterpret_cast<void*>(handle), value, bytes,
                         sonicboom::cuda::sb_current_stream()) == cudaSuccess;
#else
  (void)handle; (void)value; (void)bytes;
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

Stats stats() noexcept {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::mutex> lk(g_mutex);
  return g_stats;
#else
  return Stats{};
#endif
}

void clear_all() {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::mutex> lk(g_mutex);
  for (auto& kv : g_free)
    for (uint64_t h : kv.second) {
      cudaFree(reinterpret_cast<void*>(h));
      g_sizes.erase(h);
    }
  g_free.clear();
  g_stats.retained_blocks = 0;
  g_stats.retained_bytes = 0;
#endif
}

} // namespace sonicboom::planner::device
