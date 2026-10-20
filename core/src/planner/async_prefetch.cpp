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

#include "async_prefetch.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#ifdef SONICBOOM_USE_CUDA
#include "../nn/cuda_stream.h"  // cuda::sb_capturing
#endif

namespace sonicboom::planner {

//--- LRU stub ---------------------------------------------------------------

void LRUEvictionPolicy::touch(int layer) {
  order_.erase(std::remove(order_.begin(), order_.end(), layer), order_.end());
  order_.push_back(layer);
  last_ = layer;
}

std::optional<int> LRUEvictionPolicy::evict_candidate(
    std::span<const int> resident_layers, uint64_t /*needed_bytes*/) {
  // The earliest-touched layer that is actually resident; an order entry that
  // has since left the resident set is skipped rather than evicted.
  for (int layer : order_)
    for (int resident : resident_layers)
      if (resident == layer) return layer;
  return std::nullopt;
}

void LRUEvictionPolicy::reset() {
  last_ = -1;
  order_.clear();
}

//--- runtime ----------------------------------------------------------------

AsyncPrefetchRuntime& AsyncPrefetchRuntime::instance() {
  static AsyncPrefetchRuntime runtime;
  return runtime;
}

bool AsyncPrefetchRuntime::enabled() const {
  const char* env = std::getenv("SONICBOOM_ASYNC_PREFETCH");
  return env != nullptr && std::strcmp(env, "on") == 0;
}

void AsyncPrefetchRuntime::begin(int layer, uint64_t bytes) {
  if (!enabled()) return;

#ifdef SONICBOOM_USE_CUDA
  // Create the stream lazily, outside capture: a stream created into a capture
  // would make the captured graph's lifetime depend on this object.
  bool create = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++begins_;
    bytes_ += bytes;
    if (policy_) policy_->touch(layer);
    create = stream_ == nullptr && !cuda::sb_capturing();
  }
  if (create) {
    cudaStream_t s = nullptr;
    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) == cudaSuccess) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stream_ == nullptr) stream_ = s;
      else cudaStreamDestroy(s);  // another thread won the race
    } else {
      (void)cudaGetLastError();  // no device: the skeleton stays streamless
    }
  }
#else
  std::lock_guard<std::mutex> lock(mutex_);
  ++begins_;
  bytes_ += bytes;
  if (policy_) policy_->touch(layer);
#endif
}

void AsyncPrefetchRuntime::wait(int layer) {
  if (!enabled()) return;

  void* stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    ++waits_;
    if (policy_) policy_->touch(layer);
    stream = stream_;
  }
#ifdef SONICBOOM_USE_CUDA
  // Nothing is queued on the stream yet; the synchronization is the hook the
  // real prefetch issues its copies through. Never synchronize into a capture:
  // a sync cannot be recorded into a CUDA graph.
  if (stream != nullptr && !cuda::sb_capturing())
    (void)cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
#endif
}

void* AsyncPrefetchRuntime::stream() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stream_;
}

void AsyncPrefetchRuntime::set_eviction_policy(
    std::unique_ptr<LayerEvictionPolicy> policy) {
  std::lock_guard<std::mutex> lock(mutex_);
  policy_ = std::move(policy);
}

LayerEvictionPolicy* AsyncPrefetchRuntime::eviction_policy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return policy_.get();
}

uint64_t AsyncPrefetchRuntime::begins() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return begins_;
}

uint64_t AsyncPrefetchRuntime::waits() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return waits_;
}

uint64_t AsyncPrefetchRuntime::prefetched_bytes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return bytes_;
}

void AsyncPrefetchRuntime::reset_for_test() {
  void* stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream = stream_;
    stream_ = nullptr;
    policy_.reset();
    begins_ = waits_ = bytes_ = 0;
  }
#ifdef SONICBOOM_USE_CUDA
  if (stream != nullptr) (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream));
#else
  (void)stream;
#endif
}

AsyncPrefetchRuntime::AsyncPrefetchRuntime() = default;

AsyncPrefetchRuntime::~AsyncPrefetchRuntime() {
  if (stream_ != nullptr) {
#ifdef SONICBOOM_USE_CUDA
    (void)cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
#endif
    stream_ = nullptr;
  }
}

} // namespace sonicboom::planner
