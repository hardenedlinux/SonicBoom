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

// The device-resident per-sequence state of a recurrent block
// (planner/gdn_state_dev.h): the host twin's operations over device buffers.

#include <sonicboom/planner/gdn_state_dev.h>

#include "device_allocator.h"

#include <sonicboom/nn/cuda_gdn.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace sonicboom::planner {

namespace {

uint64_t channels(const nn::GdnDims& dims) {
  return 2 * uint64_t(dims.heads_k) * dims.dim + uint64_t(dims.heads_v) * dims.dim;
}

uint64_t value_dim(const nn::GdnDims& dims) {
  return uint64_t(dims.heads_v) * dims.dim;
}

std::string rejected(uint32_t layer, const char* what) {
  return "gdn_scan: layer " + std::to_string(layer) +
         " received spans that do not match its geometry (" + what + ")";
}

const float* dev(const hal::Buffer& buf) {
  return reinterpret_cast<const float*>(buf.id);
}

float* dev_mut(const hal::Buffer& buf) {
  return reinterpret_cast<float*>(buf.id);
}

// A HAL view of a device pointer the caller owns; only the id is read.
hal::Buffer view(const float* p, uint64_t bytes) {
  return hal::Buffer{reinterpret_cast<uint64_t>(p), bytes, hal::DeviceId{},
                     hal::MemoryKind::DeviceLocal};
}

// The host checkpoint budget: SONICBOOM_STATE_CKPT_MB in MiB, default 1024.
// <= 0 disables checkpoints; an unparsable value keeps the default.
uint64_t state_ckpt_budget_bytes() {
  constexpr uint64_t kDefaultMiB = 1024;
  const char* e = std::getenv("SONICBOOM_STATE_CKPT_MB");
  if (!e || !*e) return kDefaultMiB << 20;
  char* end = nullptr;
  const long long mib = std::strtoll(e, &end, 10);
  if (end == e || *end != '\0') return kDefaultMiB << 20;
  if (mib <= 0) return 0;
  return uint64_t(mib) << 20;
}

// The host bytes a snapshot holds.
uint64_t snapshot_bytes(const std::vector<GdnStateSnapshot>& snap) {
  uint64_t bytes = 0;
  for (const GdnStateSnapshot& s : snap)
    bytes += (s.hist.size() + s.mem.size()) * sizeof(float);
  return bytes;
}

} // namespace

GdnStateDev::GdnStateDev(const nn::GdnDims& dims) : dims_(dims) {
  alloc_ = std::make_shared<hal::DeviceAllocator>();
  const uint64_t C = channels(dims);
  hist_floats_ = dims.kernel > 0 ? uint64_t(dims.kernel - 1) * C : 0;
  mem_floats_ = value_dim(dims) * dims.dim;
  if (hist_floats_ != 0) {
    hist_ = alloc_->allocate(hist_floats_ * sizeof(float));
    if (hist_) alloc_->fill(hist_, 0, hist_floats_ * sizeof(float));
  }
  if (mem_floats_ != 0) {
    mem_ = alloc_->allocate(mem_floats_ * sizeof(float));
    if (mem_) alloc_->fill(mem_, 0, mem_floats_ * sizeof(float));
  }
}

GdnStateDev::~GdnStateDev() {
  if (!alloc_) return;
  for (hal::Buffer* buf : {&hist_, &mem_, &scratch_hist_, &scratch_mem_,
                           &journal_qkv_, &journal_bl_, &journal_al_,
                           &scratch_out_}) {
    if (*buf) alloc_->deallocate(*buf);
    *buf = {};
  }
}

void GdnStateDev::swap(GdnStateDev& other) noexcept {
  std::swap(alloc_, other.alloc_);
  std::swap(dims_, other.dims_);
  std::swap(hist_, other.hist_);
  std::swap(mem_, other.mem_);
  std::swap(hist_floats_, other.hist_floats_);
  std::swap(mem_floats_, other.mem_floats_);
  std::swap(pos_, other.pos_);
  std::swap(scratch_hist_, other.scratch_hist_);
  std::swap(scratch_mem_, other.scratch_mem_);
  std::swap(journal_qkv_, other.journal_qkv_);
  std::swap(journal_bl_, other.journal_bl_);
  std::swap(journal_al_, other.journal_al_);
  std::swap(scratch_out_, other.scratch_out_);
  std::swap(journal_rows_, other.journal_rows_);
  std::swap(journal_weights_, other.journal_weights_);
  std::swap(journal_pos_, other.journal_pos_);
  std::swap(pending_, other.pending_);
}

std::expected<void, std::string> GdnStateDev::position_error(
    uint32_t layer, uint64_t pos) const {
  return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                         " scans from position " + std::to_string(pos) +
                         ", but its state is at position " +
                         std::to_string(pos_));
}

std::expected<void, std::string> GdnStateDev::commit(
    uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
    const float* qkv, const float* bl, const float* al, int rows,
    float* out) {
  if (!valid())
    return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                           " has no device state (allocation failed)");
  if (pending_ != 0)
    return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                           " commits while a tentative scan of " +
                           std::to_string(pending_) + " row(s) is pending");
  if (pos != pos_) return position_error(layer, pos);
  if (rows < 0) return std::unexpected(rejected(layer, "row count"));
  if (rows == 0) return {};
  if (!nn::cuda::gdn_scan_dev(dims_, weights, qkv, bl, al, rows, dev(hist_),
                              dev_mut(hist_), dev(mem_), dev_mut(mem_), out))
    return std::unexpected(rejected(layer, "input, state or output size"));
  pos_ += uint64_t(rows);
  return {};
}

bool GdnStateDev::ensure_journal(uint64_t rows) {
  if (journal_rows_ >= rows && journal_qkv_ && journal_bl_ && journal_al_ &&
      scratch_out_)
    return true;
  for (hal::Buffer* buf : {&journal_qkv_, &journal_bl_, &journal_al_,
                           &scratch_out_}) {
    if (*buf) alloc_->deallocate(*buf);
    *buf = {};
  }
  journal_rows_ = 0;
  const uint64_t C = channels(dims_);
  const uint64_t hv = dims_.heads_v;
  journal_qkv_ = alloc_->allocate(rows * C * sizeof(float));
  journal_bl_ = alloc_->allocate(rows * hv * sizeof(float));
  journal_al_ = alloc_->allocate(rows * hv * sizeof(float));
  scratch_out_ = alloc_->allocate(rows * value_dim(dims_) * sizeof(float));
  if (!journal_qkv_ || !journal_bl_ || !journal_al_ || !scratch_out_) {
    for (hal::Buffer* buf : {&journal_qkv_, &journal_bl_, &journal_al_,
                             &scratch_out_}) {
      if (*buf) alloc_->deallocate(*buf);
      *buf = {};
    }
    return false;
  }
  journal_rows_ = rows;
  return true;
}

std::expected<void, std::string> GdnStateDev::tentative(
    uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
    const float* qkv, const float* bl, const float* al, int rows, float* out) {
  if (!valid())
    return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                           " has no device state (allocation failed)");
  if (pending_ != 0)
    return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                           " starts a tentative scan while a tentative scan "
                           "of " +
                           std::to_string(pending_) + " row(s) is pending");
  if (pos != pos_) return position_error(layer, pos);
  if (rows < 0) return std::unexpected(rejected(layer, "row count"));
  if (rows == 0) return {};

  // The tentative scan reads the committed state and writes scratch.
  const uint64_t hist_bytes = hist_floats_ * sizeof(float);
  const uint64_t mem_bytes = mem_floats_ * sizeof(float);
  if (!scratch_hist_)
    scratch_hist_ = alloc_->allocate(hist_bytes);
  if (!scratch_mem_) scratch_mem_ = alloc_->allocate(mem_bytes);
  if ((hist_bytes != 0 && !scratch_hist_) ||
      (mem_bytes != 0 && !scratch_mem_))
    return std::unexpected(rejected(layer, "scratch allocation"));
  if ((hist_bytes != 0 &&
       !alloc_->device_copy(scratch_hist_, hist_, hist_bytes)) ||
      (mem_bytes != 0 && !alloc_->device_copy(scratch_mem_, mem_, mem_bytes)))
    return std::unexpected(rejected(layer, "scratch copy"));

  const uint64_t C = channels(dims_);
  const uint64_t hv = dims_.heads_v;
  if (!ensure_journal(uint64_t(rows)))
    return std::unexpected(rejected(layer, "journal allocation"));
  if (!alloc_->device_copy(journal_qkv_, view(qkv, rows * C * sizeof(float)),
                           rows * C * sizeof(float)) ||
      !alloc_->device_copy(journal_bl_, view(bl, rows * hv * sizeof(float)),
                           rows * hv * sizeof(float)) ||
      !alloc_->device_copy(journal_al_, view(al, rows * hv * sizeof(float)),
                           rows * hv * sizeof(float)))
    return std::unexpected(rejected(layer, "journal copy"));

  if (!nn::cuda::gdn_scan_dev(dims_, weights, qkv, bl, al, rows, dev(hist_),
                              dev_mut(scratch_hist_), dev(mem_),
                              dev_mut(scratch_mem_), out))
    return std::unexpected(rejected(layer, "input, state or output size"));

  journal_weights_ = weights;
  journal_pos_ = pos_;
  pending_ = uint64_t(rows);
  return {};
}

std::expected<void, std::string> GdnStateDev::accept(uint32_t layer,
                                                     uint64_t keep) {
  if (pending_ == 0)
    return std::unexpected("gdn_accept: layer " + std::to_string(layer) +
                           " has no pending tentative scan");
  if (keep == 0 || keep > pending_)
    return std::unexpected("gdn_accept: layer " + std::to_string(layer) +
                           " accepts " + std::to_string(keep) +
                           " row(s) of a tentative scan of " +
                           std::to_string(pending_));

  if (keep == pending_) {
    // The whole batch is accepted: the scratch state is the state after it.
    std::swap(hist_, scratch_hist_);
    std::swap(mem_, scratch_mem_);
  } else {
    // Perform the accepted prefix again from the same committed state. The
    // kernel runs row by row, so the first `keep` rows produce exactly what
    // the tentative scan produced for them.
    const int n = int(keep);
    if (!nn::cuda::gdn_scan_dev(dims_, journal_weights_, dev(journal_qkv_),
                                dev(journal_bl_), dev(journal_al_), n,
                                dev(hist_), dev_mut(hist_), dev(mem_),
                                dev_mut(mem_), dev_mut(scratch_out_)))
      return std::unexpected(rejected(layer, "journal rows"));
  }
  pos_ = journal_pos_ + keep;
  pending_ = 0;
  return {};
}

void GdnStateDev::discard() noexcept { pending_ = 0; }

void GdnStateDev::reset() {
  pending_ = 0;
  pos_ = 0;
  if (!valid() || !alloc_) return;
  alloc_->fill(hist_, 0, hist_floats_ * sizeof(float));
  alloc_->fill(mem_, 0, mem_floats_ * sizeof(float));
}

GdnStateSnapshot GdnStateDev::save() const {
  GdnStateSnapshot snapshot;
  snapshot.pos = pos_;
  if (!valid() || !alloc_) return snapshot;
  snapshot.hist.resize(hist_floats_);
  snapshot.mem.resize(mem_floats_);
  if (hist_floats_ != 0 &&
      !alloc_->copy_from_device(snapshot.hist.data(), hist_,
                                hist_floats_ * sizeof(float))) {
    snapshot.hist.clear();
    snapshot.mem.clear();
    return snapshot;
  }
  if (mem_floats_ != 0 &&
      !alloc_->copy_from_device(snapshot.mem.data(), mem_,
                                mem_floats_ * sizeof(float))) {
    snapshot.hist.clear();
    snapshot.mem.clear();
  }
  return snapshot;
}

std::expected<void, std::string> GdnStateDev::restore(
    const GdnStateSnapshot& snapshot) {
  if (snapshot.hist.size() != hist_floats_ ||
      snapshot.mem.size() != mem_floats_)
    return std::unexpected(
        "gdn_restore: snapshot size does not match the state's geometry (" +
        std::to_string(snapshot.hist.size()) + " histogram and " +
        std::to_string(snapshot.mem.size()) + " memory floats, expected " +
        std::to_string(hist_floats_) + " and " + std::to_string(mem_floats_) +
        ")");
  if (!valid() || !alloc_)
    return std::unexpected(
        "gdn_restore: the device state has no buffers (allocation failed)");
  if ((hist_floats_ != 0 &&
       !alloc_->copy_to_device(hist_, snapshot.hist.data(),
                               hist_floats_ * sizeof(float))) ||
      (mem_floats_ != 0 &&
       !alloc_->copy_to_device(mem_, snapshot.mem.data(),
                               mem_floats_ * sizeof(float))))
    return std::unexpected("gdn_restore: the copy to the device failed");
  pos_ = snapshot.pos;
  pending_ = 0;
  return {};
}

StateSetDev::StateSetDev(std::vector<nn::GdnDims> dims) {
  states_.reserve(dims.size());
  for (const nn::GdnDims& d : dims) states_.emplace_back(d);
}

uint64_t StateSetDev::pos() const noexcept {
  return states_.empty() ? 0 : states_.front().pos();
}

bool StateSetDev::valid() const noexcept {
  for (const GdnStateDev& s : states_) {
    if (!s.valid()) return false;
  }
  return true;
}

std::vector<GdnStateSnapshot> StateSetDev::save() const {
  std::vector<GdnStateSnapshot> out;
  out.reserve(states_.size());
  for (const GdnStateDev& s : states_) out.push_back(s.save());
  return out;
}

std::expected<void, std::string> StateSetDev::restore(
    const std::vector<GdnStateSnapshot>& snapshot) {
  if (snapshot.size() != states_.size())
    return std::unexpected(
        "gdn_restore: snapshot holds " + std::to_string(snapshot.size()) +
        " state(s), the state set holds " + std::to_string(states_.size()));
  for (size_t i = 0; i < states_.size(); ++i) {
    if (auto r = states_[i].restore(snapshot[i]); !r) return r;
  }
  return {};
}

StateCheckpointStoreDev::StateCheckpointStoreDev()
    : checkpoint_budget_(state_ckpt_budget_bytes()) {}

void StateCheckpointStoreDev::checkpoint(const StateSetDev& states,
                                         uint64_t position) {
  if (checkpoint_budget_ == 0 || states.size() == 0) return;
  std::vector<GdnStateSnapshot> snap = states.save();
  const uint64_t bytes = snapshot_bytes(snap);
  // Nothing was allocated (or the copy failed): there is no state to keep.
  if (bytes == 0) return;
  auto it = checkpoints_.find(position);
  if (it != checkpoints_.end()) {
    // The same boundary recomputed: keep the newer copy.
    checkpoint_bytes_ -= snapshot_bytes(it->second);
    it->second = std::move(snap);
    checkpoint_bytes_ += bytes;
    return;
  }
  // Make room by dropping the lowest position. A checkpoint larger than the
  // whole budget is not kept at all.
  while (!checkpoints_.empty() &&
         checkpoint_bytes_ + bytes > checkpoint_budget_) {
    auto lowest = checkpoints_.begin();
    checkpoint_bytes_ -= snapshot_bytes(lowest->second);
    checkpoints_.erase(lowest);
  }
  if (bytes > checkpoint_budget_) return;
  checkpoints_.emplace(position, std::move(snap));
  checkpoint_bytes_ += bytes;
}

void StateCheckpointStoreDev::drop_checkpoints_above(uint64_t position) {
  auto it = checkpoints_.upper_bound(position);
  while (it != checkpoints_.end()) {
    checkpoint_bytes_ -= snapshot_bytes(it->second);
    it = checkpoints_.erase(it);
  }
}

bool StateCheckpointStoreDev::checkpoint_below(
    uint64_t bound, uint64_t& position,
    const std::vector<GdnStateSnapshot>*& snapshot) const {
  auto it = checkpoints_.upper_bound(bound);
  if (it == checkpoints_.begin()) return false;
  --it;
  position = it->first;
  snapshot = &it->second;
  return true;
}

std::vector<uint64_t> StateCheckpointStoreDev::checkpoint_positions() const {
  std::vector<uint64_t> out;
  out.reserve(checkpoints_.size());
  for (const auto& [position, snapshot] : checkpoints_) out.push_back(position);
  return out;
}

} // namespace sonicboom::planner
