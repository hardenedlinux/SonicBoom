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

#include "decision_registry.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>
#include <utility>

namespace sonicboom::planner::decisions {

namespace {

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

// (name, scope) -> entry.
std::map<std::pair<std::string, std::string>, Entry>& registry() {
  static std::map<std::pair<std::string, std::string>, Entry> r;
  return r;
}

std::size_t& conflicts() {
  static std::size_t c = 0;
  return c;
}

} // namespace

std::string_view source_name(Source source) {
  switch (source) {
    case Source::Default: return "default";
    case Source::Env: return "env";
    case Source::Cached: return "cached";
    case Source::Measured: return "measured";
    case Source::Verdict: return "verdict";
    case Source::Skipped: return "skipped";
  }
  return "unknown";
}

bool record(std::string_view name, std::string_view scope,
            std::string_view value, Source source, std::string_view env) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  const auto key = std::make_pair(std::string(name), std::string(scope));
  const auto it = registry().find(key);
  if (it != registry().end()) {
    if (it->second.value == value) return true;
    ++conflicts();
    return false;
  }
  Entry entry;
  entry.name = std::string(name);
  entry.scope = std::string(scope);
  entry.value = std::string(value);
  entry.source = source;
  entry.env = std::string(env);
  registry().emplace(key, std::move(entry));
  return true;
}

std::vector<Entry> snapshot() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  std::vector<Entry> out;
  out.reserve(registry().size());
  for (const auto& [key, entry] : registry()) out.push_back(entry);
  return out;
}

std::size_t conflict_count() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  return conflicts();
}

uint64_t fingerprint() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  if (registry().empty()) return 0;
  uint64_t h = 1469598103934665603ull;
  for (const auto& [key, entry] : registry()) {
    for (std::string_view field : {std::string_view(entry.name),
                                   std::string_view(entry.scope),
                                   std::string_view(entry.value)}) {
      for (const char c : field) {
        h ^= static_cast<unsigned char>(c);
        h *= 1099511628211ull;
      }
      h ^= 0xff;  // field separator
      h *= 1099511628211ull;
    }
  }
  return h;
}

std::string dump_text() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  std::ostringstream os;
  for (const auto& [key, entry] : registry()) {
    os << entry.name << ' ' << entry.scope << " = " << entry.value << " ["
       << source_name(entry.source) << ']';
    if (!entry.env.empty()) os << " env=" << entry.env;
    os << '\n';
  }
  if (conflicts() > 0)
    os << conflicts() << " conflicting record(s) dropped\n";
  return os.str();
}

void reset_for_test() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().clear();
  conflicts() = 0;
}

} // namespace sonicboom::planner::decisions
