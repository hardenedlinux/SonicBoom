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

// Key-axis partitions for split decode attention.
//
// The key axis [0, inf) is cut into absolute chunks by three segments:
// [0, e0) in steps of s0, [e0, e1) in steps of s1 and [e1, inf) in steps of s2.
// Every boundary is an absolute key position, so a query's chunk indices -- and
// therefore its reduction tree -- depend only on the keys it reads, never on
// the cache capacity, the launch depth or a prefix-reuse split. A uniform
// chunk size C is the degenerate partition (e0 = e1 = 0, s0 = s1 = s2 = C).
//
// Both sides of the boundary use the same closed forms: the device kernel maps
// a block to `chunk_of(start0) + blockIdx.x` and derives its key range with
// chunk_start/chunk_len; the merge folds the chunks of a row's window in
// ascending index. Host code sizes the grid with chunk_of(cap - 1) + spare.
//
// Header-only and free of CUDA builtins so the host launcher and private
// microbenchmarks can share the exact functions the kernel runs.

#pragma once

#include <cstdint>

namespace sonicboom::nn::cuda {

#if defined(__CUDACC__)
#define SB_FDEC_HD __host__ __device__
#else
#define SB_FDEC_HD
#endif

struct FdecPartition {
  uint64_t e0 = 0;  // end of segment 0 (0: no segment 0)
  uint64_t e1 = 0;  // end of segment 1 (0: no segment 1)
  uint64_t s0 = 1;  // chunk size in segment 0
  uint64_t s1 = 1;  // chunk size in segment 1
  uint64_t s2 = 1;  // chunk size in segment 2 (unbounded)
  uint64_t n0 = 0;  // number of chunks in segment 0
  uint64_t n1 = 0;  // number of chunks in segment 1

  // Absolute chunk index of key `j`.
  SB_FDEC_HD constexpr uint64_t chunk_of(uint64_t j) const {
    if (j < e0) return j / s0;
    if (j < e1) return n0 + (j - e0) / s1;
    return n0 + n1 + (j - e1) / s2;
  }

  // First key position of absolute chunk `c`.
  SB_FDEC_HD constexpr uint64_t chunk_start(uint64_t c) const {
    if (c < n0) return c * s0;
    if (c < n0 + n1) return e0 + (c - n0) * s1;
    return e1 + (c - n0 - n1) * s2;
  }

  // Number of keys in absolute chunk `c` (the last chunk of a window may be
  // short; the kernel clamps to the query's position).
  SB_FDEC_HD constexpr uint64_t chunk_len(uint64_t c) const {
    return c < n0 ? s0 : c < n0 + n1 ? s1 : s2;
  }
};

// Segment sizes must divide the segment length exactly, so a boundary never
// falls inside a chunk.
SB_FDEC_HD constexpr FdecPartition fdec_partition(uint64_t s0, uint64_t e0,
                                                  uint64_t s1, uint64_t e1,
                                                  uint64_t s2) {
  return FdecPartition{e0,
                       e1,
                       s0,
                       s1,
                       s2,
                       s0 == 0 ? 0 : e0 / s0,
                       s1 == 0 ? 0 : (e1 - e0) / s1};
}

SB_FDEC_HD constexpr FdecPartition fdec_uniform_partition(uint64_t c) {
  return fdec_partition(c, 0, c, 0, c);
}

// Partition identities a class can choose between. The first five are uniform
// chunk sizes, the last two grow the chunks with the distance from the query.
enum FdecPartId : int {
  kFdecPartC32 = 1,
  kFdecPartC64 = 2,
  kFdecPartC128 = 3,
  kFdecPartC256 = 4,
  kFdecPartC512 = 5,
  kFdecPartP1 = 6,  // (512, 64), (4096, 256), (inf, 512)
  kFdecPartP2 = 7,  // (1024, 128), (8192, 512), (inf, 1024)
};

SB_FDEC_HD constexpr FdecPartition fdec_partition_of(int id) {
  switch (id) {
    case kFdecPartC32:
      return fdec_uniform_partition(32);
    case kFdecPartC64:
      return fdec_uniform_partition(64);
    case kFdecPartC256:
      return fdec_uniform_partition(256);
    case kFdecPartC512:
      return fdec_uniform_partition(512);
    case kFdecPartP1:
      return fdec_partition(64, 512, 256, 4096, 512);
    case kFdecPartP2:
      return fdec_partition(128, 1024, 512, 8192, 1024);
    case kFdecPartC128:
    default:
      return fdec_uniform_partition(128);
  }
}

SB_FDEC_HD constexpr const char* fdec_part_name(int id) {
  switch (id) {
    case kFdecPartC32:
      return "c32";
    case kFdecPartC64:
      return "c64";
    case kFdecPartC128:
      return "c128";
    case kFdecPartC256:
      return "c256";
    case kFdecPartC512:
      return "c512";
    case kFdecPartP1:
      return "p1";
    case kFdecPartP2:
      return "p2";
  }
  return "?";
}

// Every segment size a multiple of the class's key tile, so a chunk boundary is
// also a tile boundary.
SB_FDEC_HD constexpr bool fdec_part_kt_ok(const FdecPartition& p, int kt) {
  const uint64_t k = uint64_t(kt);
  return k != 0 && p.s0 % k == 0 && p.s1 % k == 0 && p.s2 % k == 0 &&
         (p.e1 <= p.e0 || (p.e1 - p.e0) % p.s1 == 0) &&
         (p.e0 == 0 || p.e0 % p.s0 == 0);
}

// Candidate partitions per mask class (host side; filtered by key tile size).
inline constexpr int kFdecCausalParts[] = {kFdecPartC128, kFdecPartC256,
                                           kFdecPartC512, kFdecPartP1,
                                           kFdecPartP2};
inline constexpr int kFdecWindowParts[] = {kFdecPartC32, kFdecPartC64,
                                           kFdecPartC128};

}  // namespace sonicboom::nn::cuda
