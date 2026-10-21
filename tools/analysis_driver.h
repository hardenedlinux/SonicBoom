// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Shared driver for the boom-r4 F2 analysis tools. It sets the lowering's
// analysis variables (facts file, all units, marks, program lift, output
// path), loads a model, and runs one short prefill so every graph — and the
// lifted boom.program — is lowered once and analyzed. Nothing else observes
// the run: the analysis writes no attribute and the plan is unchanged.

#pragma once

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/vocab/vocab.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace sb_tools {

// The prompt the analysis runs: short, so the reports stay readable; the
// calibrated facts, not this prompt, are what bound the values.
inline constexpr const char *kAnalysisPrompt = "The capital of France is";

inline bool run_range_analysis(const std::string &model_path,
                               const std::string &facts_path,
                               const std::string &out_path, std::string *error) {
  std::filesystem::remove(out_path);
  setenv("SONICBOOM_BOOM_ANALYSIS_FACTS", facts_path.c_str(), 1);
  setenv("SONICBOOM_BOOM_ANALYSIS_UNITS", "all", 1);
  setenv("SONICBOOM_BOOM_ANALYSIS_OUT", out_path.c_str(), 1);
  setenv("SONICBOOM_BOOM_ANALYSIS_MARK", "1", 1);
  setenv("SONICBOOM_BOOM_ANALYSIS_PROGRAM", "1", 1);

  auto reader = sonicboom::gguf::Reader::load(model_path);
  if (!reader) {
    *error = "cannot read " + model_path;
    return false;
  }
  auto model = sonicboom::model::load_gemma4(*reader);
  if (!model) {
    *error = "cannot load model weights from " + model_path;
    return false;
  }
  auto vocab = sonicboom::vocab::load_vocab(*reader);
  if (!vocab) {
    *error = "cannot load vocabulary from " + model_path;
    return false;
  }
  const auto ids = sonicboom::vocab::tokenize(*vocab, kAnalysisPrompt);
  std::vector<uint64_t> prompt;
  prompt.reserve(ids.size());
  for (int32_t id : ids) prompt.push_back(uint64_t(id));
  if (prompt.empty()) {
    *error = "the analysis prompt tokenizes to nothing";
    return false;
  }
  std::vector<int64_t> tokens(4, -1);
  sonicboom::model::generate_prefill_spine(*model, prompt, 0, tokens, 0, true);
  return true;
}

} // namespace sb_tools
