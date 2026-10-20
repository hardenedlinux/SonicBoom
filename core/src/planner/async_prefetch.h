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

// VS1 — VRAM async-prefetch skeleton.
//
// AsyncPrefetchPass (compiler/BoomPasses.h) inserts boom.async_weight_prefetch
// and boom.sync_prefetch markers around the layer boundaries of a graph whose
// model-bound weights exceed SONICBOOM_VRAM_THRESHOLD_MB. The backend executes
// a marker by telling this runtime to begin or wait a layer on its own
// prefetch stream.
//
// The runtime is a no-op unless SONICBOOM_ASYNC_PREFETCH is "on": begin() and
// wait() then only maintain the stream and the counters. No weight data is
// copied yet; the stream exists so the eventual prefetch implementation has
// one place to issue its copies and the synchronization point it needs.
//
// LayerEvictionPolicy is the seam for the residency policy a real prefetch
// implementation will consult (which layer's weights may be dropped to make
// room). LRUEvictionPolicy is the stub: it records the touch order and offers
// the least-recently-touched resident layer as the eviction candidate.
//
// Layer placement: internal runtime machinery (core/src/planner), not part of
// the Layer 2 public headers; no CUDA types in this interface (the stream is
// an opaque void*).

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <vector>

namespace sonicboom::planner {

// Which resident layer's weights may be evicted. The runtime does not track
// residency yet (the skeleton performs no copies); a policy is what a real
// implementation will ask before it reuses a layer's VRAM. `needed_bytes` is
// the size the caller is about to make resident.
class LayerEvictionPolicy {
 public:
  virtual ~LayerEvictionPolicy() = default;

  // The layer was just prefetched or used.
  virtual void touch(int layer) = 0;

  // The layer to evict from `resident_layers`, or nullopt when evicting any of
  // them would not help (a policy may also return nullopt when the set is
  // empty). The stub ignores `needed_bytes`; it is part of the signature so a
  // real policy need not change it.
  virtual std::optional<int> evict_candidate(
      std::span<const int> resident_layers, uint64_t needed_bytes) = 0;
};

// Least-recently-used stub: the earliest-touched layer in `resident_layers`.
// Touches the runtime reports (begin/wait) are the recency it orders by. A
// layer touched again moves to the back.
class LRUEvictionPolicy final : public LayerEvictionPolicy {
 public:
  void touch(int layer) override;
  std::optional<int> evict_candidate(std::span<const int> resident_layers,
                                     uint64_t needed_bytes) override;

  // The most recent touch, or -1 before the first one (diagnostics).
  int last_touched() const { return last_; }
  void reset();

 private:
  int last_ = -1;
  std::vector<int> order_;  // oldest first
};

class AsyncPrefetchRuntime {
 public:
  static AsyncPrefetchRuntime& instance();

  // SONICBOOM_ASYNC_PREFETCH == "on" (exact string). Read per call: the switch
  // is an ablation input and a caller may change it between runs. Unset, "off"
  // or anything else is off.
  bool enabled() const;

  // Start prefetching `layer`'s weights (the marker's `prefetch_bytes`). When
  // the runtime is disabled this is a no-op and no stream is created.
  void begin(int layer, uint64_t bytes);

  // Wait for `layer`'s prefetched weights before its compute runs.
  void wait(int layer);

  // The prefetch stream (a cudaStream_t), or nullptr when none exists: the
  // build has no CUDA, the runtime is disabled, or stream creation failed. The
  // stream is created lazily on the first enabled begin() outside CUDA graph
  // capture (a stream cannot be created into a capture).
  void* stream() const;

  void set_eviction_policy(std::unique_ptr<LayerEvictionPolicy> policy);
  LayerEvictionPolicy* eviction_policy() const;

  // Counters (the skeleton's observable behavior).
  uint64_t begins() const;
  uint64_t waits() const;
  uint64_t prefetched_bytes() const;

  // Clear the counters and the policy; destroy the stream. For tests.
  void reset_for_test();

  ~AsyncPrefetchRuntime();
  AsyncPrefetchRuntime(const AsyncPrefetchRuntime&) = delete;
  AsyncPrefetchRuntime& operator=(const AsyncPrefetchRuntime&) = delete;

 private:
  AsyncPrefetchRuntime();

  mutable std::mutex mutex_;
  void* stream_ = nullptr;  // cudaStream_t under SONICBOOM_USE_CUDA
  std::unique_ptr<LayerEvictionPolicy> policy_;
  uint64_t begins_ = 0;
  uint64_t waits_ = 0;
  uint64_t bytes_ = 0;
};

// The switch shared with AsyncPrefetchPass (compiler/BoomPasses.h, where the
// pass reads SONICBOOM_VRAM_THRESHOLD_MB): SONICBOOM_ASYNC_PREFETCH == "on".
// The pass and the runtime each read it, so a run that disables it inserts no
// markers and a marker that reaches the backend anyway is still a no-op.

} // namespace sonicboom::planner
