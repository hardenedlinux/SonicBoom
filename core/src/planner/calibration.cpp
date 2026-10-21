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

#include "calibration.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

namespace sonicboom::planner::calibration {

namespace {

struct Entry {
  double lo = 0.0;
  double hi = 0.0;
  bool seen = false;
};

// One sink per process, flushed when it goes out of scope at exit. The path is
// read once: an unset variable leaves `path` empty and the sink inert.
struct Sink {
  std::string path;
  std::map<std::string, Entry> entries;
  std::mutex mutex;
  bool flushed = false;

  Sink() {
    if (const char *e = std::getenv("SONICBOOM_BOOM_CALIBRATION"))
      if (*e) path = e;
  }

  ~Sink() {
    if (path.empty() || flushed) return;
    write(path.c_str());
  }

  bool write(const char *target) {
    std::FILE *f = std::fopen(target, "w");
    if (!f) return false;
    for (const auto &[name, e] : entries)
      if (e.seen)
        std::fprintf(f, "tensor %s %.17g %.17g 0\n", name.c_str(), e.lo, e.hi);
    std::fclose(f);
    return true;
  }
};

Sink &sink() {
  static Sink s;
  return s;
}

} // namespace

bool enabled() { return !sink().path.empty(); }

void record(const char *name, std::span<const float> values) {
  if (!name || values.empty() || !enabled()) return;
  double lo = 0.0, hi = 0.0;
  bool any = false;
  for (const float v : values) {
    if (!std::isfinite(v)) continue;
    if (!any) {
      lo = hi = v;
      any = true;
    } else {
      lo = std::min(lo, double(v));
      hi = std::max(hi, double(v));
    }
  }
  if (!any) return;
  Sink &s = sink();
  std::lock_guard<std::mutex> lock(s.mutex);
  auto it = s.entries.find(name);
  if (it == s.entries.end()) {
    s.entries.emplace(name, Entry{lo, hi, true});
    return;
  }
  it->second.lo = std::min(it->second.lo, lo);
  it->second.hi = std::max(it->second.hi, hi);
}

bool flush(const char *path) {
  if (!path || !*path) return false;
  Sink &s = sink();
  std::lock_guard<std::mutex> lock(s.mutex);
  const bool ok = s.write(path);
  s.flushed = true;
  s.entries.clear();
  return ok;
}

} // namespace sonicboom::planner::calibration
