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
    return h;
  }
  void* p = nullptr;
  if (cudaMalloc(&p, size) != cudaSuccess) return 0;
  return reinterpret_cast<uint64_t>(p);
#else
  (void)bytes;
  return 0;
#endif
}

void release(uint64_t handle, uint64_t bytes) {
#ifdef SONICBOOM_USE_CUDA
  if (handle == 0) return;
  const uint64_t size = align_up(bytes);
  std::lock_guard<std::mutex> lk(g_mutex);
  g_free[size].push_back(handle);
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
  return cudaMemcpy(reinterpret_cast<void*>(dst),
                    reinterpret_cast<const void*>(src), bytes,
                    cudaMemcpyDeviceToDevice) == cudaSuccess;
#else
  (void)src; (void)dst; (void)bytes;
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

void clear_all() {
#ifdef SONICBOOM_USE_CUDA
  std::lock_guard<std::mutex> lk(g_mutex);
  for (auto& kv : g_free)
    for (uint64_t h : kv.second)
      cudaFree(reinterpret_cast<void*>(h));
  g_free.clear();
#endif
}

} // namespace sonicboom::planner::device
