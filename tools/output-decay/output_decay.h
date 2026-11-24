// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Output Decay (OD): how much of an engine's own answer survives its
// acceleration. For one prompt, the reference is that engine's output with
// acceleration off and the compared output is the same engine's output with it
// on. p is the length in bytes of the longest common prefix; N is the
// reference's length in bytes; OD = 1 - p/N. A run-to-run OD uses two fresh
// processes with the same settings.
//
// Every prompt is one file in each directory, paired by file name; the
// comparison is over the file bytes exactly as the engine wrote them. This
// header and its .cpp depend on nothing in SonicBoom, so the metric can be
// applied to any engine's saved outputs.

#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sonicboom::tools::od {

struct PromptResult {
  std::string name;
  std::uint64_t reference_bytes = 0;
  std::uint64_t prefix_bytes = 0;
  // 0-based byte offset of the first byte that differs, or where one output
  // ends; `identical` is true instead when the two outputs are equal.
  std::uint64_t first_divergence = 0;
  bool identical = true;
  // 0..100. A prompt with an empty reference is 0% when the compared output is
  // empty too (nothing decayed) and 100% otherwise (everything the answer
  // would have contained is gone).
  double od_percent = 0.0;
};

struct Summary {
  std::vector<PromptResult> prompts;
  // Arithmetic mean of the per-prompt percentages.
  double od_mean_percent = 0.0;
  // Byte-weighted: 100 * (sum N - sum p) / sum N over prompts with N > 0.
  double od_pooled_percent = 0.0;
};

// Length in bytes of the longest common prefix.
std::uint64_t common_prefix_length(std::string_view a, std::string_view b);

PromptResult compare_outputs(std::string name, std::string_view reference,
                             std::string_view compared);

// Pair every regular file in `reference_dir` with the file of the same name in
// `compared_dir`, in name order, and compute the per-prompt results and the
// summary. Throws std::runtime_error when a directory cannot be read or the
// two directories do not hold the same file names.
Summary compare_directories(const std::filesystem::path& reference_dir,
                            const std::filesystem::path& compared_dir);

// Text of one output as the engine wrote it (binary read, no newline changes).
std::string read_output(const std::filesystem::path& path);

} // namespace sonicboom::tools::od
