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

// The accumulator a norm / attention-prep row reduction uses for its sum of
// squares (the f64 "double" accumulator in the reference kernels and in the
// generated glue regions).
//
//   F64  — double accumulator. The historical path; bit-identical reference.
//   DF32 — a double-float pair (hi, lo) of f32 values, accumulated with
//          Knuth's TwoSum. Same reduction tree shape as F64 (strided thread
//          partials, shuffle-down warp tree, shared-memory warp partials,
//          warp-0 tree), so only the per-add arithmetic differs.
//   F32  — plain float accumulator. Never selected automatically; it exists
//          as the performance reference for the DF32 budget and as a manual
//          override.
//
// The mode is chosen once per device (core/src/planner/reduce_mode.cpp) and
// is part of the plan cache key and the glue JIT cache header.

#include <optional>
#include <string_view>

namespace sonicboom::compiler {

enum class ReduceMode : int { F64 = 0, Df32 = 1, F32 = 2 };

inline constexpr const char *reduce_mode_name(ReduceMode mode) noexcept {
  switch (mode) {
    case ReduceMode::F64:
      return "f64";
    case ReduceMode::Df32:
      return "df32";
    case ReduceMode::F32:
      return "f32";
  }
  return "?";
}

inline std::optional<ReduceMode> parse_reduce_mode(std::string_view name) {
  if (name == "f64") return ReduceMode::F64;
  if (name == "df32") return ReduceMode::Df32;
  if (name == "f32") return ReduceMode::F32;
  return std::nullopt;
}

// A double-float value: an unevaluated sum hi + lo, both f32. Used by the
// hosted DF32 reduction (core/src/nn/norm.cpp and matmul.cpp) and, through
// reduce_sum.cuh, by the CUDA kernels. Deliberately a plain aggregate so the
// device side can keep it in two registers.
struct Df32 {
  float hi = 0.0f;
  float lo = 0.0f;
};

#if defined(__CUDACC__)
#define SB_REDUCE_HD __host__ __device__
#else
#define SB_REDUCE_HD
#endif

// Knuth's TwoSum on plain additions: s = fl(a.hi + b.hi) and the pair (s, e)
// represents a.hi + b.hi exactly. Both low parts are folded into e with one
// rounding each. No multiplication here, so -fmad cannot change the result;
// plain `+`/`-` round to nearest and are never reassociated.
SB_REDUCE_HD inline Df32 df32_add(Df32 a, Df32 b) noexcept {
  const float s = a.hi + b.hi;
  const float v = s - a.hi;
  const float e0 = (a.hi - (s - v)) + (b.hi - v);
  const float e = e0 + (a.lo + b.lo);
  return {s, e};
}

// TwoSum with a pure-f32 addend (lo = 0): one fewer add than df32_add.
SB_REDUCE_HD inline Df32 df32_add_float(Df32 a, float b) noexcept {
  const float s = a.hi + b;
  const float v = s - a.hi;
  const float e = ((a.hi - (s - v)) + (b - v)) + a.lo;
  return {s, e};
}

// The pair's value, rounded once to f32.
SB_REDUCE_HD inline float df32_value(Df32 a) noexcept { return a.hi + a.lo; }

#undef SB_REDUCE_HD

} // namespace sonicboom::compiler
