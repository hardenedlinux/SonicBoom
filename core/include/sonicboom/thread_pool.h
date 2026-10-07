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

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Minimal persistent CPU thread pool for data-parallel work (Phase 6a).
//
// This is a conventional, SonicBoom-owned runtime primitive, not a copy of any
// external implementation. parallel_for() statically partitions [begin, end)
// into one contiguous slice per worker and blocks until every worker finishes.
//
// Determinism: each index in [begin, end) is processed by exactly one worker,
// so a pure row-wise/elementwise function produces the same result for any
// worker count (the order within a slice is unchanged; only which thread runs a
// slice differs). This is what lets the quantized matmul split its row loop
// across threads while staying bit-exact with the single-threaded oracle path.
//
// The pool is process-wide and lazily spawned on first use. parallel_for() is
// NOT reentrant: it must not be invoked concurrently from multiple threads
// (the model decode loop is single-threaded at the dispatch level, so this
// holds for the current call sites).

namespace sonicboom {

class ThreadPool {
public:
  // The process-wide instance (lazily spawned hardware_concurrency workers).
  static ThreadPool& instance();

  int num_workers() const { return n_workers_; }

  // Run fn(slice_begin, slice_end) over contiguous slices of [begin, end);
  // blocks until all slices complete. `fn` is not retained and is valid only
  // for the duration of the call. Tiny ranges run inline (no pool handshake).
  void parallel_for(uint64_t begin, uint64_t end,
                    const std::function<void(uint64_t, uint64_t)>& fn);

private:
  ThreadPool();
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  void worker_loop(int id);

  int n_workers_;
  std::vector<std::thread> workers_;

  std::mutex mu_;
  std::condition_variable cv_;       // workers wake on a new generation
  std::condition_variable done_cv_;  // caller waits for remaining_ == 0

  const std::function<void(uint64_t, uint64_t)>* fn_ = nullptr;
  uint64_t begin_ = 0;
  uint64_t end_ = 0;
  uint64_t generation_ = 0;
  uint64_t remaining_ = 0;
  bool shutting_down_ = false;
};

} // namespace sonicboom
