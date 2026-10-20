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

// DraftModel: the drafter interface of the speculative decode loop (SD1).
//
// The loop in `run_spec_program_steps` (gemma4_plan.cpp) asks the drafter for
// the next k tokens, feeds `[last, drafts...]` through the row-invariant verify
// graph over k+1 rows, and accepts the longest matching prefix plus the
// verifier's own token. **The verify/accept path is drafter-agnostic and is
// reused unchanged for every DraftModel**: a source only proposes tokens, reads
// the accepted history it is given (never the KV cache or device state), and a
// rejected draft is simply discarded work. That is why a model-backed drafter
// can be added behind this interface without touching verification, the
// accept rule, or the emitted stream (which stays the plain greedy stream).
//
// Implementations:
//   * NgramDraftModel — the existing prompt-lookup NGramDrafter (B5).
//   * LlmDraftModel  — skeleton; constructing it throws
//                      "LlmDraftModel not yet implemented". Selected by the
//                      SONICBOOM_DRAFT_MODEL environment variable.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "sonicboom/planner/drafter.h"

namespace sonicboom::planner {

// Drafter contract. Semantics are those of NGramDrafter: the tables describe
// accepted tokens only; a rejected draft is never pushed; propose returns 0
// when there is nothing to propose and the caller then takes a plain decode
// step. Proposals must be deterministic for a given history.
class DraftModel {
 public:
  virtual ~DraftModel() = default;

  virtual void reset() noexcept = 0;
  virtual void push(uint64_t token) = 0;
  virtual void push(std::span<const uint64_t> tokens) = 0;

  virtual size_t max_drafts() const noexcept = 0;
  virtual void set_max_drafts(size_t k) noexcept = 0;

  virtual size_t propose(std::span<uint64_t> out) const = 0;
  // As above, also reporting the length of the matched suffix n-gram
  // (diagnostic only; the proposal is identical).
  virtual size_t propose(std::span<uint64_t> out, size_t* matched_n) const = 0;
};

// The prompt-lookup n-gram drafter, unchanged behind the interface.
class NgramDraftModel final : public DraftModel {
 public:
  explicit NgramDraftModel(size_t max_drafts = 7) : drafter_(max_drafts) {}

  void reset() noexcept override { drafter_.reset(); }
  void push(uint64_t token) override { drafter_.push(token); }
  void push(std::span<const uint64_t> tokens) override { drafter_.push(tokens); }
  size_t max_drafts() const noexcept override { return drafter_.max_drafts(); }
  void set_max_drafts(size_t k) noexcept override {
    drafter_.set_max_drafts(k);
  }
  size_t propose(std::span<uint64_t> out) const override {
    return drafter_.propose(out);
  }
  size_t propose(std::span<uint64_t> out, size_t* matched_n) const override {
    return drafter_.propose(out, matched_n);
  }

  const NGramDrafter &ngram() const noexcept { return drafter_; }

 private:
  NGramDrafter drafter_;
};

// Model-backed drafter. Not implemented: every entry point (including the
// constructor) throws std::runtime_error("LlmDraftModel not yet implemented").
// It exists so the interface, the environment switch, and the call sites are
// fixed before the drafter model itself lands.
class LlmDraftModel final : public DraftModel {
 public:
  explicit LlmDraftModel(std::string model_path, size_t max_drafts = 7);

  void reset() noexcept override;
  void push(uint64_t token) override;
  void push(std::span<const uint64_t> tokens) override;
  size_t max_drafts() const noexcept override;
  void set_max_drafts(size_t k) noexcept override;
  size_t propose(std::span<uint64_t> out) const override;
  size_t propose(std::span<uint64_t> out, size_t* matched_n) const override;
};

// SONICBOOM_DRAFT_MODEL unset or empty  -> NgramDraftModel(max_drafts).
// SONICBOOM_DRAFT_MODEL set to a path  -> LlmDraftModel(path, max_drafts),
//                                         which throws (not implemented).
std::unique_ptr<DraftModel> make_draft_model(size_t max_drafts);

} // namespace sonicboom::planner
