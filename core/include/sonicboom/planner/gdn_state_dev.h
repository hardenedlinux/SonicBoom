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

// The device-resident variant of planner/gdn_state.h: the same state, the same
// operations and the same meanings, with the committed histogram and memory in
// device buffers and every scan running the CUDA kernel (nn/cuda_gdn.h).
// Scratch and journal buffers are allocated on the first tentative scan.
// save()/restore() are whole copies between the device and the host.
//
// This header names only Layer 2 types plus opaque HAL buffers, so it carries no
// CUDA type; the implementation lives in core/src/planner/gdn_state_dev.cpp and
// is compiled only when the build is configured with SONICBOOM_USE_CUDA=ON.
//
// A state owns its device buffers, so it is movable but not copyable. A copy of
// a state is a GdnStateSnapshot (save()/restore()).

#include <sonicboom/hal/buffer.h>
#include <sonicboom/nn/gdn.h>
#include <sonicboom/planner/gdn_state.h>

#include <cstdint>
#include <expected>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sonicboom::planner {

class GdnStateDev {
 public:
  GdnStateDev() = default;
  explicit GdnStateDev(const nn::GdnDims& dims);
  ~GdnStateDev();

  GdnStateDev(GdnStateDev&& other) noexcept { swap(other); }
  GdnStateDev& operator=(GdnStateDev&& other) noexcept {
    swap(other);
    return *this;
  }
  GdnStateDev(const GdnStateDev&) = delete;
  GdnStateDev& operator=(const GdnStateDev&) = delete;

  void swap(GdnStateDev& other) noexcept;

  const nn::GdnDims& dims() const noexcept { return dims_; }
  uint64_t pos() const noexcept { return pos_; }
  // Rows of a tentative scan whose accept is still pending; 0 when none.
  uint64_t pending() const noexcept { return pending_; }
  // True once the committed device buffers were allocated.
  bool valid() const noexcept {
    return (hist_floats_ == 0 || static_cast<bool>(hist_)) &&
           (mem_floats_ == 0 || static_cast<bool>(mem_));
  }

  // Scan `rows` rows of device memory with the position of the first one,
  // updating the committed state in place. `out` receives the rows' outputs.
  std::expected<void, std::string> commit(
      uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      const float* qkv, const float* bl, const float* al, int rows, float* out);

  // Scan `rows` rows into scratch, leaving the committed state untouched; the
  // inputs are journaled for a later partial accept.
  std::expected<void, std::string> tentative(
      uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      const float* qkv, const float* bl, const float* al, int rows, float* out);

  // Accept the first `keep` rows of the pending tentative scan (1 <= keep <=
  // pending rows).
  std::expected<void, std::string> accept(uint32_t layer, uint64_t keep);

  // Drop the pending tentative scan; the committed state is unchanged.
  void discard() noexcept;

  // Zero the state.
  void reset();

  // Whole copies between the device and the host.
  GdnStateSnapshot save() const;
  std::expected<void, std::string> restore(const GdnStateSnapshot& snapshot);

 private:
  std::expected<void, std::string> position_error(uint32_t layer,
                                                  uint64_t pos) const;
  // Grow the journal to this many rows (no-op when it already holds them).
  bool ensure_journal(uint64_t rows);

  // The device allocator behind every buffer (the process-wide pool).
  std::shared_ptr<hal::Allocator> alloc_;
  nn::GdnDims dims_{};
  hal::Buffer hist_;
  hal::Buffer mem_;
  uint64_t hist_floats_ = 0;
  uint64_t mem_floats_ = 0;
  uint64_t pos_ = 0;

  // Allocated on the first tentative scan.
  hal::Buffer scratch_hist_;
  hal::Buffer scratch_mem_;
  hal::Buffer journal_qkv_;
  hal::Buffer journal_bl_;
  hal::Buffer journal_al_;
  hal::Buffer scratch_out_;
  uint64_t journal_rows_ = 0;
  // The pending tentative scan: its weights and position.
  nn::GdnWeights journal_weights_{};
  uint64_t journal_pos_ = 0;

  uint64_t pending_ = 0;
};

// One device state per recurrent block, in the order the recipe numbers them.
class StateSetDev {
 public:
  StateSetDev() = default;
  explicit StateSetDev(std::vector<nn::GdnDims> dims);

  size_t size() const noexcept { return states_.size(); }
  GdnStateDev& state(size_t index) { return states_[index]; }
  const GdnStateDev& state(size_t index) const { return states_[index]; }
  // The committed position every state in the set stands at (the set advances
  // as one); 0 for an empty set. A prefix-reuse caller reads it to tell a live
  // continuation from a history the caller has rewound.
  uint64_t pos() const noexcept;
  bool valid() const noexcept;  // every state's committed buffers exist

  std::expected<void, std::string> commit(
      size_t index, uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      const float* qkv, const float* bl, const float* al, int rows,
      float* out) {
    return states_[index].commit(layer, pos, weights, qkv, bl, al, rows, out);
  }
  std::expected<void, std::string> tentative(
      size_t index, uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      const float* qkv, const float* bl, const float* al, int rows,
      float* out) {
    return states_[index].tentative(layer, pos, weights, qkv, bl, al, rows,
                                   out);
  }
  std::expected<void, std::string> accept(size_t index, uint32_t layer,
                                          uint64_t keep) {
    return states_[index].accept(layer, keep);
  }
  void discard(size_t index) { states_[index].discard(); }
  void reset() {
    for (GdnStateDev& s : states_) s.reset();
  }

  std::vector<GdnStateSnapshot> save() const;
  std::expected<void, std::string> restore(
      const std::vector<GdnStateSnapshot>& snapshot);

 private:
  std::vector<GdnStateDev> states_;
};

// Host checkpoints of a recurrent state set, keyed by the position they follow
// (the design doc's section 11). The store belongs to a session (next to its
// `states`), not to the state set: the device backend may replace the state set
// it runs with, while the checkpoints outlive that. checkpoint() copies the
// committed device state to the host; the store holds what
// SONICBOOM_STATE_CKPT_MB (MiB, default 1024; <= 0 disables) allows, dropping
// the lowest position when a new one does not fit. A position already stored is
// replaced. The rest of the API reads the store: the highest position at or
// below a bound, every position, and the removal of the positions above a bound
// (the abandoned continuation).
class StateCheckpointStoreDev {
 public:
  StateCheckpointStoreDev();

  void checkpoint(const StateSetDev& states, uint64_t position);
  void drop_checkpoints_above(uint64_t position);
  bool checkpoint_below(uint64_t bound, uint64_t& position,
                        const std::vector<GdnStateSnapshot>*& snapshot) const;
  std::vector<uint64_t> checkpoint_positions() const;

 private:
  std::map<uint64_t, std::vector<GdnStateSnapshot>> checkpoints_;
  uint64_t checkpoint_bytes_ = 0;
  uint64_t checkpoint_budget_ = 0;
};

} // namespace sonicboom::planner
