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

#include <sonicboom/numerics.h>

#include "decision_registry.h"

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

namespace sonicboom {
namespace {

constexpr const char *kEnv = "SONICBOOM_NUMERICS";

struct State {
  std::mutex mutex;
  bool latched = false;
  NumericsProfile profile = kDefaultNumericsProfile;
  std::optional<NumericsProfile> application;
  bool kv_latched = false;
  KvStorage kv = KvStorage::F16;
};

State &state() {
  static State s;
  return s;
}

// Why the profile was chosen, as the registry record's reason (M1: the
// application-selected case reads exactly "application").
struct Origin {
  planner::decisions::Source source = planner::decisions::Source::Default;
  const char *reason = "the default profile";
  std::string inputs = "none";
};

} // namespace

const char *numerics_profile_name(NumericsProfile profile) noexcept {
  switch (profile) {
    case NumericsProfile::Strict: return "strict";
    case NumericsProfile::Standard: return "standard";
  }
  return "standard";
}

std::optional<NumericsProfile> parse_numerics_profile(std::string_view name) {
  if (name == "strict") return NumericsProfile::Strict;
  if (name == "standard") return NumericsProfile::Standard;
  return std::nullopt;
}

bool numerics_profile_resolved() {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  return s.latched;
}

bool set_numerics_profile(NumericsProfile profile, std::string *error) {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.latched) {
    if (s.profile == profile) return true;
    if (error)
      *error = std::string("numerics profile already resolved as ") +
               numerics_profile_name(s.profile) + "; cannot switch to " +
               numerics_profile_name(profile) + " after startup";
    return false;
  }
  s.application = profile;
  return true;
}

NumericsProfile numerics_profile() {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.latched) return s.profile;

  NumericsProfile chosen = kDefaultNumericsProfile;
  Origin origin;

  // The environment is the test/CI switch and, like every other SONICBOOM_*
  // knob, it overrides the static default. An unrecognised value is a warning
  // plus the default, never a silent other profile.
  if (const char *env = std::getenv(kEnv); env != nullptr && *env != '\0') {
    if (auto parsed = parse_numerics_profile(env)) {
      chosen = *parsed;
      origin.source = planner::decisions::Source::Env;
      origin.reason = "SONICBOOM_NUMERICS set explicitly";
      origin.inputs = std::string(kEnv) + "=" + env;
    } else {
      std::fprintf(stderr,
                   "SONICBOOM_NUMERICS: unknown profile '%s' "
                   "(strict|standard); using %s\n",
                   env, numerics_profile_name(chosen));
      origin.inputs = std::string(kEnv) + " invalid";
    }
  } else if (s.application) {
    chosen = *s.application;
    origin.source = planner::decisions::Source::Application;
    origin.reason = "application";
    origin.inputs = "sb_set_numerics_profile";
  }

  planner::decisions::Entry entry;
  entry.name = "numerics.profile";
  entry.scope = "process";
  entry.value = numerics_profile_name(chosen);
  entry.source = origin.source;
  entry.env = kEnv;
  entry.inputs = std::move(origin.inputs);
  entry.candidates = "strict|standard";
  entry.reason = origin.reason;
  entry.numerics =
      chosen == NumericsProfile::Strict
          ? "historical contract: f32 KV; f64 reductions; fp64-gated dense"
          : "f16 KV storage; the standard acceptance gate (ppl, top-1)";
  planner::decisions::record(entry);

  s.profile = chosen;
  s.latched = true;
  return s.profile;
}

bool numerics_profile_f16_kv() {
  return numerics_profile() == NumericsProfile::Standard;
}

const char *numerics_kv_storage_name(KvStorage storage) noexcept {
  switch (storage) {
    case KvStorage::F32: return "f32";
    case KvStorage::F16: return "f16";
    case KvStorage::Q8: return "q8";
  }
  return "f16";
}

KvStorage numerics_kv_storage() {
  // Resolve the profile first: numerics_profile() takes the same lock.
  const NumericsProfile profile = numerics_profile();
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  if (s.kv_latched) return s.kv;

  // The profile decides what the default means: strict pins f32 (its contract
  // is the pre-rounding values); standard stores f16, or q8 once selected.
  KvStorage chosen = profile == NumericsProfile::Strict ? KvStorage::F32
                                                        : KvStorage::F16;
  planner::decisions::Source source = planner::decisions::Source::Default;
  std::string inputs = "none";
  const char *reason = "the numerics profile";

  // SONICBOOM_KV is the test/CI switch, same rules as SONICBOOM_NUMERICS: the
  // environment overrides the default, an unknown value warns and keeps it.
  if (const char *env = std::getenv("SONICBOOM_KV"); env != nullptr && *env != '\0') {
    if (std::string_view(env) == "f32") {
      chosen = KvStorage::F32;
    } else if (std::string_view(env) == "f16") {
      chosen = KvStorage::F16;
    } else if (std::string_view(env) == "q8") {
      chosen = KvStorage::Q8;
    } else if (std::string_view(env) != "auto") {
      std::fprintf(stderr,
                   "SONICBOOM_KV: unknown storage '%s' (auto|f32|f16|q8); "
                   "using %s\n",
                   env, numerics_kv_storage_name(chosen));
      inputs = std::string("SONICBOOM_KV invalid: ") + env;
    }
    if (inputs == "none" && std::string_view(env) != "auto") {
      source = planner::decisions::Source::Env;
      reason = "SONICBOOM_KV set explicitly";
      inputs = std::string("SONICBOOM_KV=") + env;
    }
  }

  planner::decisions::Entry entry;
  entry.name = "kv.storage";
  entry.scope = "process";
  entry.value = numerics_kv_storage_name(chosen);
  entry.source = source;
  entry.env = "SONICBOOM_KV";
  entry.inputs = std::move(inputs);
  entry.candidates = "f32|f16|q8";
  entry.reason = reason;
  entry.numerics =
      chosen == KvStorage::Q8
          ? "q8 re-quantizes the fp16-rounded K/V (per-row scale = amax/127)"
          : chosen == KvStorage::F16
                ? "f16 storage is exact for the fp16-rounded K/V values"
                : "f32 storage (the pre-rounding values)";
  planner::decisions::record(entry);

  s.kv = chosen;
  s.kv_latched = true;
  return s.kv;
}

void reset_numerics_kv_storage_for_test() {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.kv_latched = false;
  s.kv = KvStorage::F16;
  planner::decisions::forget_for_test("kv.storage", "process");
}

void reset_numerics_profile_for_test() {
  State &s = state();
  std::lock_guard<std::mutex> lock(s.mutex);
  s.latched = false;
  s.application.reset();
  s.profile = kDefaultNumericsProfile;
  planner::decisions::forget_for_test("numerics.profile", "process");
}

} // namespace sonicboom
