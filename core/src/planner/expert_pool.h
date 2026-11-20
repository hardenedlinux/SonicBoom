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

// The bounded device pool of routed experts (design: the Qwen3.6 run, §9).
//
// A mixture-of-experts model keeps its routed experts in host memory (they are
// most of the file) and a bounded pool on the device holds the ones in use,
// one slot per (block, expert):
//
//   pool_gate [N * B_gate]   pool_up [N * B_up]   pool_down [N * B_down]
//
// B_role is the largest byte size of one expert's matrix of that role over all
// blocks, rounded up to the kernels' 256-byte read-ahead tail (see
// register_weight_device); the bytes after a matrix's packed bytes in its slot
// are kept zero. N is the pool bytes divided by B_gate + B_up + B_down, where
// the pool bytes are SONICBOOM_EXPERT_POOL_MB when set and otherwise the free
// device memory at creation minus the model's non-expert weights minus
// 1024 MB, raised to one block's worth of experts when that figure is
// smaller: the default always covers every expert one block can select at
// once, taking the floor out of the margin if needed. A setting below that
// floor is an error that names the missing bytes.
//
// Before a node's kernel runs, select() reads its selected expert ids (the
// caller has already read them from the device), copies each absent expert's
// three matrices from host memory into a slot — evicting the least recently
// used slot this node does not need — and writes the slot of every selected
// expert to a device buffer. A scenario in which residency cannot change a
// result: the kernel's arithmetic, including the ascending-expert summation
// order, is unchanged, and only where a matrix is found moves.
//
// The copies go through a pinned, double-buffered staging pipeline (device::
// Staging): the host fills a pinned half, the half travels on a dedicated copy
// stream, and events order it against the compute stream. The arrays are zeroed
// once at creation; a slot's read-ahead tail is still zeroed when it is filled,
// because a smaller matrix may follow a larger one there. SONICBOOM_EXPERT_STAGE_KB
// sizes the staging buffer (default 16384 KiB; the tests use 64).
//
// The pool is internal runtime machinery (core/src/planner), not part of the
// Layer 2 surface: no CUDA type appears here.

#include <sonicboom/model/weights.h>

#include "device_pool.h"

#include <cstdint>
#include <expected>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

namespace sonicboom::planner {

class ExpertPool {
 public:
  // Counters reported when SONICBOOM_EXPERT_STATS=1. hits/misses count
  // selection entries (a repeated (block, expert) within one node is a hit
  // after the first), so they are a function of the token history alone.
  struct Stats {
    uint64_t nodes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t bytes = 0;
  };

  // Size the pool from `weights` and allocate its three device arrays.
  // Returns an error naming the missing bytes when the pool cannot hold every
  // expert a block may select at once, or when no device is present.
  static std::expected<std::unique_ptr<ExpertPool>, std::string> create(
      const model::Gemma4Weights& weights);
  ~ExpertPool();
  ExpertPool(const ExpertPool&) = delete;
  ExpertPool& operator=(const ExpertPool&) = delete;

  uint64_t slots() const noexcept { return n_slots_; }
  uint64_t slot_bytes(uint32_t role) const noexcept { return stride_[role]; }

  // The three device arrays (slot s of role r starts at data(r) + s *
  // slot_bytes(r)).
  const void* data(uint32_t role) const noexcept;

  // The device buffer holding one slot per selection entry, in the order of
  // the node's `ids`, as written by the last select().
  const void* selection() const noexcept;

  // The host copy of that buffer, valid until the next select().
  const int32_t* selection_host() const noexcept { return sel_host_.data(); }

  // Steps 2-4 of a node: bring every selected expert resident (evicting the
  // lowest-stamped slot this node does not need), stamp every slot this node
  // uses, and write the selection buffer. `block` indexes the weights' blocks;
  // `ids` is the node's rows*used selected experts in node order. Thread-safe.
  std::expected<void, std::string> select(uint32_t block,
                                          std::span<const int64_t> ids);

  Stats stats() const;

  // Test hook: forget what is resident and zero the counters; the arrays stay
  // allocated (every selection then misses again).
  void clear();

  // Role indices for data()/slot_bytes().
  static constexpr uint32_t kGate = 0;
  static constexpr uint32_t kUp = 1;
  static constexpr uint32_t kDown = 2;

 private:
  ExpertPool() = default;

  struct BlockInfo {
    uint32_t experts = 0;
    uint32_t key_base = 0;  // global key of expert 0 of this block
  };

  std::expected<void, std::string> allocate(uint64_t pool_bytes);

  mutable std::mutex mutex_;
  std::vector<BlockInfo> blocks_;
  uint32_t total_keys_ = 0;
  // Per global key: the host view whose type and shape the kernel reads.
  std::vector<const quant::QuantizedTensor*> gate_of_;
  std::vector<const quant::QuantizedTensor*> up_of_;
  std::vector<const quant::QuantizedTensor*> down_of_;
  // Per global key: its slot, or -1 when not resident.
  std::vector<int32_t> slot_of_;
  // Per slot: the global key it holds (-1 empty) and its use stamp.
  std::vector<int64_t> owner_;
  std::vector<uint64_t> stamp_;

  uint64_t handles_[3] = {0, 0, 0};
  uint64_t stride_[3] = {0, 0, 0};
  uint64_t n_slots_ = 0;
  // The selection buffer: one int32 slot per selection entry.
  uint64_t sel_handle_ = 0;
  uint64_t sel_entries_ = 0;
  std::vector<int32_t> sel_host_;
  // The pinned staging pipeline; null when it could not be created, in which
  // case select() falls back to the synchronous copy.
  device::Staging* staging_ = nullptr;

  Stats stats_;
};

// Wall-clock seconds the CUDA backend spent in the mixture-of-experts stages,
// accumulated only when SONICBOOM_EXPERT_STATS is on: inside ExpertPool::select,
// inside the routed-expert kernel call, and inside a routing node. Printed with
// the counters at exit.
struct ExpertTiming {
  double select_s = 0.0;
  double experts_s = 0.0;
  double route_s = 0.0;
};

void add_expert_timing(const ExpertTiming& t);

// The process's pool for this weights set, created on first use and kept on
// the same lifetime as the device weight cache (one per model; a reload at a
// new address is a different weights object and so a different pool).
std::expected<ExpertPool*, std::string> expert_pool_for(
    const model::Gemma4Weights& weights);

// Drop the process's pool (tests, teardown).
void clear_expert_pool();

} // namespace sonicboom::planner
