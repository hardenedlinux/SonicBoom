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

// The per-sequence state of a recurrent (gated DeltaNet) block: the
// convolution histogram, the memory, and the number of tokens absorbed. A
// StateSet is one of these per recurrent block, indexed the way a recipe's
// GatedDeltaNet::state indexes them.
//
// The state is where the session lives for such a block — there are no rows to
// inspect — so every scan states the absolute position of its first row and a
// state that disagrees refuses the update, naming the layer and both positions.
//
// Commit updates the committed buffers in place. Tentative reads them and
// writes scratch instead, keeping the scan's inputs in a journal; accept(k)
// either swaps the two (k == the whole batch) or re-runs the kernel over the
// first k journal rows from the same committed state, which performs exactly
// the operations the tentative scan performed for those rows and yields the
// same bits. Nothing is snapshotted on the accept path.

#include <sonicboom/nn/gdn.h>

#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace sonicboom::planner {

// A host copy of one block's committed state.
struct GdnStateSnapshot {
  std::vector<float> hist;
  std::vector<float> mem;
  uint64_t pos = 0;
};

class GdnState {
 public:
  GdnState() = default;
  explicit GdnState(const nn::GdnDims& dims);

  const nn::GdnDims& dims() const noexcept { return dims_; }
  std::span<const float> hist() const noexcept { return hist_; }
  std::span<const float> mem() const noexcept { return mem_; }
  uint64_t pos() const noexcept { return pos_; }
  // Rows of a tentative scan whose accept is still pending; 0 when none.
  uint64_t pending() const noexcept { return pending_; }

  // Scan n rows with the position of the first one, updating the committed
  // state in place. `out` receives the rows' outputs.
  std::expected<void, std::string> commit(
      uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      std::span<const float> qkv, std::span<const float> bl,
      std::span<const float> al, std::span<float> out);

  // Scan n rows into scratch, leaving the committed state untouched.
  std::expected<void, std::string> tentative(
      uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      std::span<const float> qkv, std::span<const float> bl,
      std::span<const float> al, std::span<float> out);

  // Accept the first `keep` rows of the pending tentative scan (the accepted
  // drafts plus one; 1 <= keep <= pending rows).
  std::expected<void, std::string> accept(uint32_t layer, uint64_t keep);

  // Drop the pending tentative scan; the committed state is unchanged.
  void discard() noexcept;

  // Zero the state.
  void reset() noexcept;

  GdnStateSnapshot save() const;
  std::expected<void, std::string> restore(const GdnStateSnapshot& snapshot);

 private:
  std::expected<void, std::string> position_error(uint32_t layer,
                                                  uint64_t pos) const;

  nn::GdnDims dims_{};
  std::vector<float> hist_;
  std::vector<float> mem_;
  uint64_t pos_ = 0;

  // Allocated on the first tentative scan.
  std::vector<float> scratch_hist_;
  std::vector<float> scratch_mem_;
  // The pending tentative scan: its inputs, weights and position.
  std::vector<float> journal_qkv_;
  std::vector<float> journal_bl_;
  std::vector<float> journal_al_;
  nn::GdnWeights journal_weights_{};
  uint64_t journal_pos_ = 0;

  uint64_t pending_ = 0;
};

// One state per recurrent block, in the order the recipe numbers them.
class StateSet {
 public:
  StateSet() = default;
  explicit StateSet(const std::vector<nn::GdnDims>& dims);

  size_t size() const noexcept { return states_.size(); }
  GdnState& state(size_t index) { return states_[index]; }
  const GdnState& state(size_t index) const { return states_[index]; }

  std::span<const float> hist(size_t index) const {
    return states_[index].hist();
  }
  std::span<const float> mem(size_t index) const {
    return states_[index].mem();
  }
  uint64_t pos(size_t index) const { return states_[index].pos(); }
  uint64_t pending(size_t index) const { return states_[index].pending(); }

  std::expected<void, std::string> commit(
      size_t index, uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      std::span<const float> qkv, std::span<const float> bl,
      std::span<const float> al, std::span<float> out) {
    return states_[index].commit(layer, pos, weights, qkv, bl, al, out);
  }
  std::expected<void, std::string> tentative(
      size_t index, uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
      std::span<const float> qkv, std::span<const float> bl,
      std::span<const float> al, std::span<float> out) {
    return states_[index].tentative(layer, pos, weights, qkv, bl, al, out);
  }
  std::expected<void, std::string> accept(size_t index, uint32_t layer,
                                          uint64_t keep) {
    return states_[index].accept(layer, keep);
  }
  void discard(size_t index) { states_[index].discard(); }
  void reset() {
    for (GdnState& s : states_) s.reset();
  }

  std::vector<GdnStateSnapshot> save() const;
  std::expected<void, std::string> restore(
      const std::vector<GdnStateSnapshot>& snapshot);

 private:
  std::vector<GdnState> states_;
};

} // namespace sonicboom::planner
