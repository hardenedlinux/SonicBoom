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

// Launch hook for the plan record (plan_record.h). Included by the CUDA kernels
// only. SB_NOTE_LAUNCH is inserted immediately before a kernel launch; it costs
// one thread-local load and a not-taken branch when no recorder is installed
// (recording off), and does no allocation or string work in that case.

#include <sonicboom/planner/plan_record.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string_view>
#include <utility>

namespace sonicboom::planner::plan_detail {

// Format "a=1,b=2" into a thread-local scratch buffer, for the instantiation
// label of a template kernel. Only ever called when a recorder is installed.
inline std::string_view tp_label(
    std::initializer_list<std::pair<const char*, long long>> kv) {
  static thread_local char buf[192];
  int off = 0;
  for (const auto& [k, v] : kv) {
    if (off > 0) buf[off++] = ',';
    const int n = std::snprintf(buf + off, sizeof buf - (std::size_t)off,
                                "%s=%lld", k, v);
    if (n < 0) break;
    off += n;
    if ((std::size_t)off >= sizeof buf) {
      off = (int)sizeof buf - 1;
      break;
    }
  }
  buf[off] = '\0';
  return std::string_view(buf, (std::size_t)off);
}

// A grid/block reaches a launch as an int, an unsigned, or a dim3. Unpack each
// into the components the record stores. The dim3 overload is a better match
// than the template for a dim3 argument, so it always wins for dim3.
template <class T>
inline void unpack(T v, unsigned& x, unsigned& y, unsigned& z) {
  x = static_cast<unsigned>(v);
  y = 1;
  z = 1;
}

inline void unpack(const dim3& d, unsigned& x, unsigned& y, unsigned& z) {
  x = d.x;
  y = d.y;
  z = d.z;
}

} // namespace sonicboom::planner::plan_detail

// Record the launch that immediately follows: the kernel's name, its launch
// configuration, and the stream. KERNEL must be a plain identifier (a
// template-id with commas would split the macro arguments — use
// SB_NOTE_LAUNCH_TP with an explicit instantiation label for those).
#define SB_NOTE_LAUNCH(KERNEL, GRID, BLOCK, SMEM, STREAM)                     \
  do {                                                                        \
    if (auto* sb_rec_ = ::sonicboom::planner::recorder()) {                   \
      unsigned sb_gx_, sb_gy_, sb_gz_, sb_bx_, sb_by_, sb_bz_;                \
      ::sonicboom::planner::plan_detail::unpack((GRID), sb_gx_, sb_gy_,       \
                                                sb_gz_);                      \
      ::sonicboom::planner::plan_detail::unpack((BLOCK), sb_bx_, sb_by_,      \
                                                sb_bz_);                      \
      sb_rec_->add_launch(#KERNEL, "", sb_gx_, sb_gy_, sb_gz_, sb_bx_, sb_by_,\
                          sb_bz_, static_cast<std::uint64_t>(SMEM),           \
                          reinterpret_cast<const void*>(STREAM));             \
    }                                                                         \
  } while (0)

// The template-instantiation form: TPARAMS is a C string naming the concrete
// instantiation the launch selects (e.g. "ds=ds4"). Evaluated only when a
// recorder is installed.
#define SB_NOTE_LAUNCH_TP(KERNEL, TPARAMS, GRID, BLOCK, SMEM, STREAM)          \
  do {                                                                        \
    if (auto* sb_rec_ = ::sonicboom::planner::recorder()) {                   \
      unsigned sb_gx_, sb_gy_, sb_gz_, sb_bx_, sb_by_, sb_bz_;                \
      ::sonicboom::planner::plan_detail::unpack((GRID), sb_gx_, sb_gy_,       \
                                                sb_gz_);                      \
      ::sonicboom::planner::plan_detail::unpack((BLOCK), sb_bx_, sb_by_,      \
                                                sb_bz_);                      \
      sb_rec_->add_launch(KERNEL, (TPARAMS), sb_gx_, sb_gy_, sb_gz_, sb_bx_,  \
                          sb_by_, sb_bz_, static_cast<std::uint64_t>(SMEM),   \
                          reinterpret_cast<const void*>(STREAM));             \
    }                                                                         \
  } while (0)
