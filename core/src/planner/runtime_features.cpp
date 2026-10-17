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

#include <sonicboom/runtime_features.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace sonicboom::runtime {

namespace {

struct FeatureName {
  const char* name;
  Feature feature;
};

// One table for the parser, the error message, and feature_name(), so they
// cannot disagree about the accepted names.
constexpr FeatureName kFeatureNames[] = {
    {"cuda_graph_decode", Feature::CudaGraphDecode},
    {"cuda_graph_prefill", Feature::CudaGraphPrefill},
    {"capture_reuse", Feature::CaptureReuse},
    {"split_kv", Feature::SplitKv},
    {"plan_cache", Feature::PlanCache},
    {"q8_1_dispatch", Feature::Q8_1Dispatch},
    {"buffer_reuse", Feature::BufferReuse},
    {"warmup", Feature::Warmup},
};

constexpr std::size_t kFeatureCount =
    sizeof(kFeatureNames) / sizeof(kFeatureNames[0]);

// The pre-existing single-purpose switches, kept as aliases. Each one only
// ever turned its feature *off*; that direction is preserved exactly.
bool alias_disabled(Feature feature) noexcept {
  switch (feature) {
    case Feature::CudaGraphDecode:
    case Feature::CudaGraphPrefill: {
      const char* e = std::getenv("SONICBOOM_CUDA_GRAPHS");
      return e && std::string_view(e) == "0";
    }
    case Feature::CaptureReuse: {
      const char* e = std::getenv("SONICBOOM_CAPTURE_REUSE");
      return e && std::string_view(e) == "0";
    }
    case Feature::SplitKv: {
      const char* e = std::getenv("SONICBOOM_SPLIT_KV");
      return e && std::string_view(e) == "0";
    }
    case Feature::PlanCache: {
      const char* e = std::getenv("SONICBOOM_PLAN_CACHE");
      return e && std::string_view(e) == "0";
    }
    case Feature::Q8_1Dispatch:
      // Any value, including "0", selects the fp32 path (the historical
      // semantics of SONICBOOM_GEMV_F32).
      return std::getenv("SONICBOOM_GEMV_F32") != nullptr;
    case Feature::BufferReuse: {
      const char* e = std::getenv("SONICBOOM_BUFFER_REUSE");
      return e && std::string_view(e) == "0";
    }
    case Feature::Warmup: {
      const char* e = std::getenv("SONICBOOM_WARMUP");
      return e && std::string_view(e) == "0";
    }
  }
  return false;
}

// Parse SONICBOOM_FEATURE_DISABLE once. The list is tiny and the feature is
// queried per dispatch, so a bitset fold over the static table is kept.
std::uint32_t parse_disabled_mask() {
  std::uint32_t mask = 0;
  const char* env = std::getenv("SONICBOOM_FEATURE_DISABLE");
  if (env == nullptr || *env == '\0') return mask;
  const std::string_view list(env);
  std::string_view unknown;
  for (std::size_t pos = 0;;) {
    const std::size_t comma = list.find(',', pos);
    std::string_view name =
        list.substr(pos, comma == std::string_view::npos ? comma : comma - pos);
    while (!name.empty() && (name.front() == ' ' || name.front() == '\t'))
      name.remove_prefix(1);
    while (!name.empty() && (name.back() == ' ' || name.back() == '\t'))
      name.remove_suffix(1);
    if (!name.empty()) {
      bool matched = false;
      for (std::size_t i = 0; i < kFeatureCount; ++i) {
        if (name == kFeatureNames[i].name) {
          mask |= (1u << static_cast<unsigned>(kFeatureNames[i].feature));
          matched = true;
          break;
        }
      }
      if (!matched && unknown.empty()) unknown = name;
    }
    if (comma == std::string_view::npos) break;
    pos = comma + 1;
  }
  if (!unknown.empty()) {
    std::fprintf(stderr,
                 "SONICBOOM_FEATURE_DISABLE: unknown feature '%.*s'; valid "
                 "features are:",
                 static_cast<int>(unknown.size()), unknown.data());
    for (std::size_t i = 0; i < kFeatureCount; ++i)
      std::fprintf(stderr, " %s", kFeatureNames[i].name);
    std::fprintf(stderr, "\n");
    std::abort();
  }
  return mask;
}

std::uint32_t disabled_mask() {
  static const std::uint32_t mask = parse_disabled_mask();
  return mask;
}

} // namespace

bool feature_enabled(Feature feature) noexcept {
  const unsigned bit = static_cast<unsigned>(feature);
  if ((disabled_mask() & (1u << bit)) != 0) return false;
  return !alias_disabled(feature);
}

bool feature_disabled(Feature feature) noexcept {
  return !feature_enabled(feature);
}

const char* feature_name(Feature feature) noexcept {
  for (std::size_t i = 0; i < kFeatureCount; ++i)
    if (kFeatureNames[i].feature == feature) return kFeatureNames[i].name;
  return nullptr;
}

} // namespace sonicboom::runtime
