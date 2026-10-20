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

#include <sonicboom/planner/draft_model.h>

#include <cstdlib>
#include <stdexcept>

namespace sonicboom::planner {

namespace {
constexpr const char *kLlmNotImplemented = "LlmDraftModel not yet implemented";
} // namespace

// The constructor is the whole skeleton: selecting the model-backed drafter
// fails loudly here rather than silently falling back to the n-gram drafter,
// so a run that asks for it cannot be mistaken for an n-gram run.
LlmDraftModel::LlmDraftModel(std::string model_path, size_t max_drafts) {
  (void)model_path;
  (void)max_drafts;
  throw std::runtime_error(kLlmNotImplemented);
}

// Unreachable (construction always throws). The noexcept methods must not
// throw without terminating, so they are inert.
void LlmDraftModel::reset() noexcept {}
size_t LlmDraftModel::max_drafts() const noexcept { return 0; }
void LlmDraftModel::set_max_drafts(size_t k) noexcept { (void)k; }

void LlmDraftModel::push(uint64_t token) {
  (void)token;
  throw std::runtime_error(kLlmNotImplemented);
}

void LlmDraftModel::push(std::span<const uint64_t> tokens) {
  (void)tokens;
  throw std::runtime_error(kLlmNotImplemented);
}

size_t LlmDraftModel::propose(std::span<uint64_t> out) const {
  (void)out;
  throw std::runtime_error(kLlmNotImplemented);
}

size_t LlmDraftModel::propose(std::span<uint64_t> out,
                              size_t *matched_n) const {
  (void)out;
  (void)matched_n;
  throw std::runtime_error(kLlmNotImplemented);
}

std::unique_ptr<DraftModel> make_draft_model(size_t max_drafts) {
  const char *env = std::getenv("SONICBOOM_DRAFT_MODEL");
  if (env != nullptr && *env != '\0')
    return std::make_unique<LlmDraftModel>(env, max_drafts);
  return std::make_unique<NgramDraftModel>(max_drafts);
}

} // namespace sonicboom::planner
