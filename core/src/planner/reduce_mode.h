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

// The device's reduction-accumulator choice (compiler/ReduceMode.h), decided
// once per process:
//
//   SONICBOOM_REDUCE=f64|df32|f32 overrides both selections verbatim.
//   device_reduce_mode(): F64 when cudaDeviceProp.singleToDoublePrecisionPerfRatio
//     is <= 4 (double accumulators are affordable). Above 4, DF32 is the
//     candidate, but only if tools/sb-numerics recorded that DF32 is
//     acceptable on this device (reduce-acceptance-v1 in the decision cache).
//   cpu_reduce_mode(): times both accumulators on this CPU, applies the same
//     threshold to the measured f64/df32 ratio, and requires the same kind of
//     recorded verdict for the host.
//
// Both decisions are made once per process and recorded per device, in the
// same cache tree as the autotune choices (planner/autotune.cpp).
//
// F32 is never chosen automatically: it exists as the performance reference
// for the DF32 budget and as a manual override.

#include <sonicboom/compiler/ReduceMode.h>

#include <string>

namespace sonicboom::planner {

// The accumulator for device (CUDA) kernels.
compiler::ReduceMode device_reduce_mode();

// The accumulator for the hosted (CPU) norm / matvec kernels.
compiler::ReduceMode cpu_reduce_mode();

// The env override alone, if set (no device query, no measurement).
std::optional<compiler::ReduceMode> reduce_mode_env();

// The per-device identity "<sanitized-gpu-name>-sm<arch>-<driver>" (or "cpu").
// Shared with the autotune cache, so every per-device decision lives under one
// directory.
std::string device_identity();

// The root of the per-device decision / autotune cache.
std::string decision_cache_root();

// Record the DF32 acceptance verdict for this device and this host. Written by
// tools/sb-numerics after a full comparison; returns false and fills `error`
// when the cache is not writable.
bool record_df32_acceptance(bool accepted, std::string *error);

} // namespace sonicboom::planner
