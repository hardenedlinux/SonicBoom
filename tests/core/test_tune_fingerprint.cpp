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

// The tune-record fingerprint is a function of the version string and the
// registered tune-key patterns, not of the library file: a rebuild keeps it,
// changing the version string moves it, and two computations in one process
// agree. The same-process example is the old failure mode this rules out: a
// fingerprint that embedded the file's size/mtime could change between two
// reads if the file was replaced while the process ran.

#include "../../core/src/planner/autotune.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

bool all_hex(const std::string& s) {
  for (char c : s) {
    const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    if (!hex) return false;
  }
  return !s.empty();
}

} // namespace

int main() {
  using sonicboom::tune::fingerprint_for_version;

  // Two computations in the same process agree.
  const std::string a = fingerprint_for_version("clo-test-version");
  const std::string b = fingerprint_for_version("clo-test-version");
  check(a == b, "fingerprint: repeated computation is equal");

  // Changing the version string changes the fingerprint.
  const std::string c = fingerprint_for_version("clo-test-version-2");
  check(a != c, "fingerprint: a different version string differs");

  // The shape is <version>|<8 hex digits>, and the digits are the registered
  // key-pattern digest.
  const auto bar = a.find('|');
  check(bar == std::string::npos ? false
                                 : a.substr(0, bar) == "clo-test-version",
        "fingerprint: carries the version string");
  const std::string digest =
      bar == std::string::npos ? std::string() : a.substr(bar + 1);
  check(digest.size() == 8 && all_hex(digest),
        "fingerprint: digest is 8 hex digits");
  check(digest == sonicboom::tune::key_pattern_digest(),
        "fingerprint: digest is the key-pattern hash");
  check(!sonicboom::tune::key_pattern_list().empty(),
        "fingerprint: the tune keys are registered");

  // The build's own fingerprint follows the same composition.
  const std::string& build = sonicboom::tune::build_fingerprint();
  const auto build_bar = build.find('|');
  check(build_bar != std::string::npos &&
            build.substr(build_bar + 1) == digest,
        "fingerprint: the build fingerprint uses the same digest");
  check(sonicboom::tune::build_fingerprint_key() ==
            sonicboom::tune::build_fingerprint_key(),
        "fingerprint: the record key is stable in one process");

  if (g_failures == 0) {
    std::cout << "test_tune_fingerprint OK\n";
    return 0;
  }
  std::cerr << "test_tune_fingerprint FAILED: " << g_failures << " check(s)\n";
  return 1;
}
