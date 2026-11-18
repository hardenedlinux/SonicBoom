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

// Per-sequence state of a recurrent block (planner/gdn_state.h).

#include <sonicboom/planner/gdn_state.h>

#include <algorithm>
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

} // namespace

GdnState::GdnState(const nn::GdnDims& dims) : dims_(dims) {
  const uint64_t C = channels(dims);
  hist_.assign(dims.kernel > 0 ? uint64_t(dims.kernel - 1) * C : 0, 0.0f);
  mem_.assign(value_dim(dims) * dims.dim, 0.0f);
}

std::expected<void, std::string> GdnState::position_error(uint32_t layer,
                                                          uint64_t pos) const {
  return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                         " scans from position " + std::to_string(pos) +
                         ", but its state is at position " +
                         std::to_string(pos_));
}

std::expected<void, std::string> GdnState::commit(
    uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
    std::span<const float> qkv, std::span<const float> bl,
    std::span<const float> al, std::span<float> out) {
  if (pending_ != 0)
    return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                           " commits while a tentative scan of " +
                           std::to_string(pending_) + " row(s) is pending");
  if (pos != pos_) return position_error(layer, pos);
  const uint64_t C = channels(dims_);
  if (C == 0 || qkv.size() % C != 0)
    return std::unexpected(rejected(layer, "qkv rows"));
  if (!nn::gdn_scan(dims_, weights, qkv, bl, al, hist_, mem_, out))
    return std::unexpected(rejected(layer, "input, state or output size"));
  pos_ += qkv.size() / C;
  return {};
}

std::expected<void, std::string> GdnState::tentative(
    uint32_t layer, uint64_t pos, const nn::GdnWeights& weights,
    std::span<const float> qkv, std::span<const float> bl,
    std::span<const float> al, std::span<float> out) {
  if (pending_ != 0)
    return std::unexpected("gdn_scan: layer " + std::to_string(layer) +
                           " starts a tentative scan while a tentative scan "
                           "of " +
                           std::to_string(pending_) + " row(s) is pending");
  if (pos != pos_) return position_error(layer, pos);

  // The tentative scan reads the committed state and writes scratch.
  scratch_hist_ = hist_;
  scratch_mem_ = mem_;
  if (!nn::gdn_scan(dims_, weights, qkv, bl, al, scratch_hist_, scratch_mem_,
                    out))
    return std::unexpected(rejected(layer, "input, state or output size"));

  journal_qkv_.assign(qkv.begin(), qkv.end());
  journal_bl_.assign(bl.begin(), bl.end());
  journal_al_.assign(al.begin(), al.end());
  journal_weights_ = weights;
  journal_pos_ = pos_;
  pending_ = qkv.size() / channels(dims_);
  return {};
}

std::expected<void, std::string> GdnState::accept(uint32_t layer,
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
    const uint64_t C = channels(dims_);
    const uint64_t hv = dims_.heads_v;
    std::vector<float> unused(keep * value_dim(dims_));
    const std::span<const float> qkv(journal_qkv_.data(), keep * C);
    const std::span<const float> bl(journal_bl_.data(), keep * hv);
    const std::span<const float> al(journal_al_.data(), keep * hv);
    if (!nn::gdn_scan(dims_, journal_weights_, qkv, bl, al, hist_, mem_,
                      unused))
      return std::unexpected(rejected(layer, "journal rows"));
  }
  pos_ = journal_pos_ + keep;
  pending_ = 0;
  return {};
}

void GdnState::discard() noexcept { pending_ = 0; }

void GdnState::reset() noexcept {
  std::fill(hist_.begin(), hist_.end(), 0.0f);
  std::fill(mem_.begin(), mem_.end(), 0.0f);
  pos_ = 0;
  pending_ = 0;
}

GdnStateSnapshot GdnState::save() const {
  return GdnStateSnapshot{hist_, mem_, pos_};
}

std::expected<void, std::string> GdnState::restore(
    const GdnStateSnapshot& snapshot) {
  if (snapshot.hist.size() != hist_.size() ||
      snapshot.mem.size() != mem_.size())
    return std::unexpected(
        "gdn_restore: snapshot size does not match the state's geometry (" +
        std::to_string(snapshot.hist.size()) + " histogram and " +
        std::to_string(snapshot.mem.size()) + " memory floats, expected " +
        std::to_string(hist_.size()) + " and " + std::to_string(mem_.size()) +
        ")");
  hist_ = snapshot.hist;
  mem_ = snapshot.mem;
  pos_ = snapshot.pos;
  pending_ = 0;
  return {};
}

StateSet::StateSet(const std::vector<nn::GdnDims>& dims) {
  states_.reserve(dims.size());
  for (const nn::GdnDims& d : dims) states_.emplace_back(d);
}

std::vector<GdnStateSnapshot> StateSet::save() const {
  std::vector<GdnStateSnapshot> out;
  out.reserve(states_.size());
  for (const GdnState& s : states_) out.push_back(s.save());
  return out;
}

std::expected<void, std::string> StateSet::restore(
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

} // namespace sonicboom::planner
