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

// The numerics profile: which numeric contract the runtime executes under.
//
//   strict    the historical behavior, unchanged: f32 KV storage, f64
//             reductions on sm_86, fp64-gated dense. Every golden that
//             predates the profile switch is a strict golden.
//   standard  the default. f16 KV storage (the values stored were already
//             fp16-rounded, so the storage change is exact) and the device
//             reduction mode as resolved for the device. It is deterministic
//             and batch-invariant on a given device exactly as strict is; the
//             two profiles may differ from each other, and standard is the
//             one held to the perceptual gate (perplexity, top-1 agreement)
//             rather than bit identity.
//
// Selection, resolved once per process:
//
//   set_numerics_profile()          an application choosing explicitly
//   SONICBOOM_NUMERICS=strict|standard   the test/CI switch
//   otherwise                       standard
//
// The resolved profile is part of the plan cache key, the attention class's
// tune key (its f16-KV variant is a distinct kernel) and the glue/attention
// PTX cache keys; it is recorded in the decision registry. It is deliberately
// not part of the whole autotune device directory: a profile-wide directory
// would retune every unrelated class (mmq tiles, dense arms, gemv widths) and
// replace timing-stable picks with fresh ones. Each class's key must carry the
// profile only when the profile changes that class's kernel or candidates.
//
// The switch is process-global and latched at first use: set it before the
// library builds its first runtime. A later set with a different value is
// rejected, because half a process running one profile and half the other is
// exactly the nondeterminism the profile exists to exclude.

#include <optional>
#include <string>
#include <string_view>

namespace sonicboom {

enum class NumericsProfile {
  Strict,
  Standard,
};

// The profile used when nothing selects another.
inline constexpr NumericsProfile kDefaultNumericsProfile =
    NumericsProfile::Standard;

const char *numerics_profile_name(NumericsProfile profile) noexcept;

std::optional<NumericsProfile> parse_numerics_profile(std::string_view name);

// The resolved profile. Resolves and records the decision on first call.
NumericsProfile numerics_profile();

// True once the profile has been resolved (a set may no longer change it).
bool numerics_profile_resolved();

// Application override. Must run before the first numerics_profile() call
// (in practice: before any model/session is created). Returns false and fills
// `error` when the profile is already latched to a different value.
bool set_numerics_profile(NumericsProfile profile, std::string *error);

// The profile's consequences, in one place so no caller re-derives them:
//   standard stores the KV cache in f16 (the values were fp16-rounded already)
//   and launches the f16-KV attention variants; strict does neither. This is
//   the profile's only arithmetic difference today: the dense and matmul
//   thresholds are the same in both profiles (the TF32 dense arm is not a
//   profile default -- it is gate-blocked, reachable only through
//   SONICBOOM_DENSE_F32=tf32).
bool numerics_profile_f16_kv();

// Tests only: forget the resolution, so the next call re-reads the
// environment. The registry record is dropped with it.
void reset_numerics_profile_for_test();

} // namespace sonicboom
