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

// plan_diff — compare two plan-record files (tools/plan_dump output).
//
//   plan_diff <a.plan> <b.plan>
//
// Exit status: 0 when the files are byte-identical, 1 when they differ, 2 on a
// usage or I/O error. On a difference it reports the first differing line, the
// step that line belongs to, and a few lines of context from each side — the
// point being to see *which step* a plan change moved, not just that the files
// differ. The records are line-oriented and deterministically ordered, so a
// plain line-by-line comparison is the whole algorithm; no LCS is needed
// because the first mismatch is the answer being asked for.

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kContext = 4;  // lines of context on each side

bool read_lines(const char* path, std::vector<std::string>* out) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) out->push_back(std::move(line));
  return true;
}

bool is_step_header(std::string_view s) {
  return s.size() >= 5 && s.compare(0, 5, "step ") == 0;
}

// The step header at or before `at`, or npos. A step block spans many lines
// (attributes, buffers, launches), so the header is what names the position in
// the plan; the window below is centred on the differing line instead, since a
// difference can sit far below its header.
std::size_t enclosing_step(const std::vector<std::string>& v, std::size_t at) {
  if (v.empty()) return std::string::npos;
  at = std::min(at, v.size() - 1);
  for (std::size_t k = at + 1; k-- > 0;)
    if (is_step_header(v[k])) return k;
  return std::string::npos;
}

void print_anchor(const char* label, const std::vector<std::string>& v,
                  std::size_t at) {
  const std::size_t h = enclosing_step(v, at);
  if (h == std::string::npos) {
    std::printf("%s: no step header\n", label);
    return;
  }
  std::printf("%s: %s\n", label, v[h].c_str());
}

// A window of lines centred on `mark`, with `mark` flagged.
void print_window(const char* label, const std::vector<std::string>& v,
                  std::size_t mark) {
  std::printf("--- %s\n", label);
  const std::size_t size = v.size();
  const std::size_t focus = std::min(mark, size == 0 ? 0 : size - 1);
  const std::size_t from = focus > kContext ? focus - kContext : 0;
  const std::size_t end = std::min(size, focus + kContext + 1);
  for (std::size_t k = from; k < end; ++k)
    std::printf("%c %6zu | %s\n", k == focus ? '>' : ' ', k + 1, v[k].c_str());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <a.plan> <b.plan>\n", argv[0]);
    return 2;
  }

  std::vector<std::string> a, b;
  if (!read_lines(argv[1], &a)) {
    std::fprintf(stderr, "%s: cannot read\n", argv[1]);
    return 2;
  }
  if (!read_lines(argv[2], &b)) {
    std::fprintf(stderr, "%s: cannot read\n", argv[2]);
    return 2;
  }

  const std::size_t common = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < common && a[i] == b[i]) ++i;

  if (i == common && a.size() == b.size()) {
    std::printf("identical: %zu lines\n", a.size());
    return 0;
  }

  if (i >= common) {
    // One file ran out first: the difference is the extra lines.
    const bool a_shorter = a.size() < b.size();
    std::printf("differ: %s is %zu lines, %s is %zu lines; first extra line %zu\n",
                argv[1], a.size(), argv[2], b.size(), i + 1);
  } else {
    std::printf("differ: first at line %zu\n", i + 1);
    std::printf("  - %s: %s\n", argv[1], a[i].c_str());
    std::printf("  + %s: %s\n", argv[2], b[i].c_str());
  }

  print_anchor("  step in A", a, i);
  print_anchor("  step in B", b, i);

  std::printf("\n");
  print_window(argv[1], a, i);
  std::printf("\n");
  print_window(argv[2], b, i);
  return 1;
}
