// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>

#include "output_decay.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <set>
#include <system_error>

namespace sonicboom::tools::od {

std::uint64_t common_prefix_length(std::string_view a, std::string_view b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

PromptResult compare_outputs(std::string name, std::string_view reference,
                             std::string_view compared) {
  PromptResult r;
  r.name = std::move(name);
  r.reference_bytes = reference.size();
  r.prefix_bytes = common_prefix_length(reference, compared);
  r.identical = reference == compared;
  if (!r.identical) {
    r.first_divergence = r.prefix_bytes;
  }
  if (r.reference_bytes == 0) {
    r.od_percent = compared.empty() ? 0.0 : 100.0;
  } else {
    r.od_percent =
        100.0 *
        static_cast<double>(r.reference_bytes - r.prefix_bytes) /
        static_cast<double>(r.reference_bytes);
  }
  return r;
}

std::string read_output(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  return std::string(std::istreambuf_iterator<char>(in),
                     std::istreambuf_iterator<char>());
}

namespace {

std::set<std::string> output_names(const std::filesystem::path& dir) {
  std::set<std::string> names;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
    if (!entry.is_regular_file()) continue;
    names.insert(entry.path().filename().string());
  }
  if (ec)
    throw std::runtime_error("cannot read " + dir.string() + ": " + ec.message());
  return names;
}

} // namespace

Summary compare_directories(const std::filesystem::path& reference_dir,
                            const std::filesystem::path& compared_dir) {
  const std::set<std::string> ref_names = output_names(reference_dir);
  const std::set<std::string> cmp_names = output_names(compared_dir);
  if (ref_names.empty())
    throw std::runtime_error("no prompt outputs in " + reference_dir.string());
  if (ref_names != cmp_names) {
    std::string missing, extra;
    for (const auto& n : ref_names)
      if (!cmp_names.count(n)) missing += (missing.empty() ? " " : ", ") + n;
    for (const auto& n : cmp_names)
      if (!ref_names.count(n)) extra += (extra.empty() ? " " : ", ") + n;
    throw std::runtime_error("the directories do not hold the same prompt "
                             "outputs; missing:" +
                             (missing.empty() ? " none" : missing) +
                             "; extra:" + (extra.empty() ? " none" : extra));
  }

  Summary s;
  double mean_sum = 0.0;
  std::uint64_t total_ref = 0, total_prefix = 0;
  for (const std::string& name : ref_names) {
    const std::string ref = read_output(reference_dir / name);
    const std::string cmp = read_output(compared_dir / name);
    PromptResult r = compare_outputs(name, ref, cmp);
    mean_sum += r.od_percent;
    if (r.reference_bytes != 0) {
      total_ref += r.reference_bytes;
      total_prefix += r.prefix_bytes;
    }
    s.prompts.push_back(std::move(r));
  }
  s.od_mean_percent = mean_sum / static_cast<double>(s.prompts.size());
  s.od_pooled_percent =
      total_ref == 0 ? 0.0
                     : 100.0 * static_cast<double>(total_ref - total_prefix) /
                           static_cast<double>(total_ref);
  return s;
}

} // namespace sonicboom::tools::od
