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

#include <sonicboom/thread_pool.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace sonicboom {

namespace {
// Below this many indices the pool handshake costs more than the work; run
// inline on the calling thread instead.
constexpr uint64_t k_min_parallel_elements = 64;

// --- temporary profiling (remove after measuring) ---------------------------
struct PoolProf {
  long inline_calls = 0;
  long dispatch_calls = 0;
  double handshake_ms = 0.0;
  double work_ms = 0.0;  // sum of per-worker slice durations (≈ n_workers x wall)
  ~PoolProf() {
    std::fprintf(stderr,
                 "[threadpool-prof] %ld inline, %ld dispatch, handshake %.2f ms, "
                 "work %.2f ms\n",
                 inline_calls, dispatch_calls, handshake_ms, work_ms);
  }
} g_pool_prof;
} // namespace

ThreadPool& ThreadPool::instance() {
  static ThreadPool pool;
  return pool;
}

ThreadPool::ThreadPool() {
  const unsigned hw = std::thread::hardware_concurrency();
  n_workers_ = static_cast<int>(hw >= 2 ? hw : 1);
  workers_.reserve(static_cast<size_t>(n_workers_));
  for (int i = 0; i < n_workers_; ++i)
    workers_.emplace_back([this, i] { worker_loop(i); });
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lk(mu_);
    shutting_down_ = true;
  }
  cv_.notify_all();
  for (std::thread& w : workers_) w.join();
}

void ThreadPool::parallel_for(
    uint64_t begin, uint64_t end,
    const std::function<void(uint64_t, uint64_t)>& fn) {
  if (end <= begin) return;

  if (n_workers_ <= 1 || (end - begin) < k_min_parallel_elements) {
    ++g_pool_prof.inline_calls;
    fn(begin, end);
    return;
  }

  ++g_pool_prof.dispatch_calls;
  const auto t0 = std::chrono::steady_clock::now();
  {
    std::lock_guard<std::mutex> lk(mu_);
    fn_ = &fn;
    begin_ = begin;
    end_ = end;
    remaining_ = static_cast<uint64_t>(n_workers_);
    ++generation_;
  }
  cv_.notify_all();

  std::unique_lock<std::mutex> lk(mu_);
  done_cv_.wait(lk, [this] { return remaining_ == 0; });
  g_pool_prof.handshake_ms +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0)
          .count();
}

void ThreadPool::worker_loop(int id) {
  uint64_t seen = 0;
  for (;;) {
    const std::function<void(uint64_t, uint64_t)>* f = nullptr;
    uint64_t s = 0, e = 0;
    {
      std::unique_lock<std::mutex> lk(mu_);
      cv_.wait(lk, [this, &seen] { return shutting_down_ || generation_ != seen; });
      if (shutting_down_) return;
      seen = generation_;

      // Static contiguous slice for this worker. Read fn_/range under the lock,
      // then run the slice outside it so workers compute concurrently.
      const uint64_t count = end_ - begin_;
      const uint64_t slice = (count + static_cast<uint64_t>(n_workers_) - 1) /
                             static_cast<uint64_t>(n_workers_);
      s = begin_ + static_cast<uint64_t>(id) * slice;
      e = std::min(s + slice, end_);
      f = fn_;
    }
    if (f != nullptr && s < e) {
      const auto w0 = std::chrono::steady_clock::now();
      (*f)(s, e);
      g_pool_prof.work_ms +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - w0)
              .count();
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (--remaining_ == 0) done_cv_.notify_all();
    }
  }
}

} // namespace sonicboom
