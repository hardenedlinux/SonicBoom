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

// sb-tune — move the warm-up table between machines or cache directories.
//
// export packs every record family the cache writers produce, enumerated by
// sonicboom::tune::table_files() from the writers themselves rather than from
// a directory listing:
//   tune/<op>.json            the hand-written tuner's per-shape-class choices
//                             (every family in tune::record_families())
//   tune/reduce-acceptance-v1 the device reduction's acceptance verdict
//   tune/reduce-acceptance-cpu-v1
//                             the hosted (CPU) acceptance verdict
//   autotune/sm_<NN>.bin      the JIT candidate tuner's per-shape records
// Each file is packed only when it exists — a family this run did not touch
// stays absent. A record file the table does not name is still packed, so a
// family added by a future writer travels with the table before this list
// learns about it. All of it goes into one file with the build fingerprint and
// the device directory it was measured on. Append-only logs
// (decisions.jsonl, fattn-reports.jsonl), the table lock files, and the JIT
// PTX/cubin caches are bookkeeping or derived artifacts, not choices, and are
// not part of the table.
//
// import writes such a file into this process's cache locations. It refuses a
// table recorded by another build or another device (the record keys embed
// both, so a cross-build table would be ignored or, worse, read as a foreign
// decision), and it refuses to overwrite a non-empty table unless --replace
// is given.
//
// The archive is never interpreted: the runtime's own reader decides what a
// record means, exactly as it does for the files the tuner writes itself.

#include "planner/autotune.h"
#include "planner/mlir_autotune.h"
#include "planner/reduce_mode.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr std::string_view kMagic = "SBTUNE01\n";
constexpr uint32_t kFormatVersion = 1;

struct Entry {
  std::string name;  // "tune/<file>" or "autotune/<file>"
  std::string data;
};

std::string tune_dir() {
  return (fs::path(sonicboom::planner::decision_cache_root()) /
          sonicboom::tune::device_dir())
      .string();
}

std::string autotune_file() {
  return sonicboom::mlir_tune::AutotuneCache::instance().path();
}

std::string sm_file_name() {
  char buf[32];
  std::snprintf(buf, sizeof buf, "sm_%02d.bin",
                sonicboom::mlir_tune::sm_version());
  return buf;
}

// A record file of the hand-written tuner: <op>.json, or the reduce verdict.
// Lock files and the append-only logs are runtime bookkeeping, not choices.
bool is_tune_record(const fs::directory_entry& e) {
  if (!e.is_regular_file()) return false;
  const std::string name = e.path().filename().string();
  if (name.size() > 5 && name.compare(name.size() - 5, 5, ".lock") == 0)
    return false;
  if (name.size() > 6 && name.compare(name.size() - 6, 6, ".jsonl") == 0)
    return false;
  return true;
}

bool read_file(const fs::path& p, std::string& out) {
  std::ifstream in(p, std::ios::binary);
  if (!in) return false;
  out.assign(std::istreambuf_iterator<char>(in),
             std::istreambuf_iterator<char>());
  return in.good() || in.eof();
}

bool write_file(const fs::path& p, std::string_view data) {
  std::error_code ec;
  fs::create_directories(p.parent_path(), ec);
  const std::string tmp = p.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out) return false;
  }
  fs::rename(tmp, p, ec);
  if (ec) {
    std::error_code ignored;
    fs::remove(tmp, ignored);
    return false;
  }
  return true;
}

// Fixed-width little-endian fields; both ends of the table are x86-64 hosts
// (the device directory already pins the GPU), so no byte swapping is needed,
// but the widths are explicit so the file is not layout-dependent.
void put_u32(std::string& out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out.push_back(static_cast<char>(v >> (8 * i)));
}
void put_u64(std::string& out, uint64_t v) {
  for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>(v >> (8 * i)));
}
bool get_u32(std::string_view s, size_t& i, uint32_t& v) {
  if (i + 4 > s.size()) return false;
  v = 0;
  for (int b = 0; b < 4; ++b)
    v |= static_cast<uint32_t>(static_cast<unsigned char>(s[i + b])) << (8 * b);
  i += 4;
  return true;
}
bool get_u64(std::string_view s, size_t& i, uint64_t& v) {
  if (i + 8 > s.size()) return false;
  v = 0;
  for (int b = 0; b < 8; ++b)
    v |= static_cast<uint64_t>(static_cast<unsigned char>(s[i + b])) << (8 * b);
  i += 8;
  return true;
}

void put_str(std::string& out, std::string_view s) {
  put_u64(out, s.size());
  out.append(s);
}
bool get_str(std::string_view s, size_t& i, std::string& out) {
  uint64_t n = 0;
  if (!get_u64(s, i, n) || n > s.size() - i) return false;
  out.assign(s.substr(i, static_cast<size_t>(n)));
  i += static_cast<size_t>(n);
  return true;
}

int usage() {
  std::cout
      << "usage: sb-tune export --out FILE\n"
         "       sb-tune import --in FILE [--replace]\n"
         "\n"
         "export packs this device's warm-up table (every tuner <op>.json\n"
         "family, the device and hosted reduction verdicts, and\n"
         "autotune/sm_<NN>.bin) with the build fingerprint and device it was\n"
         "measured on.\n"
         "import writes it back into this process's cache locations; it\n"
         "refuses a table from another build or device, and refuses to\n"
         "overwrite a non-empty table unless --replace is given.\n";
  return 0;
}

int cmd_export(const std::string& out_path) {
  const std::string fingerprint = sonicboom::tune::build_fingerprint();
  const std::string device = sonicboom::tune::device_dir();
  const std::string sm_name = sm_file_name();

  // The table is defined by the code that writes it, not by what happens to
  // be in the directory: a family the run did not touch is simply absent.
  std::vector<Entry> entries;
  std::set<std::string> packed;
  std::error_code ec;
  for (const auto& file : sonicboom::tune::table_files()) {
    if (!fs::is_regular_file(file.path, ec)) continue;
    Entry entry;
    entry.name = file.name;
    if (!read_file(file.path, entry.data)) {
      std::cerr << "sb-tune: cannot read " << file.path << "\n";
      return 2;
    }
    packed.insert(entry.name);
    entries.push_back(std::move(entry));
  }
  // Forward compatibility: a record family a future writer adds without
  // updating record_families() still travels with the table. Bookkeeping
  // (lock files, append-only logs) is named and skipped.
  const fs::path tdir = tune_dir();
  if (fs::is_directory(tdir, ec)) {
    std::vector<fs::path> paths;
    for (const auto& e : fs::directory_iterator(tdir, ec))
      if (is_tune_record(e)) paths.push_back(e.path());
    std::sort(paths.begin(), paths.end());
    for (const auto& p : paths) {
      Entry entry;
      entry.name = "tune/" + p.filename().string();
      if (packed.count(entry.name)) continue;
      if (!read_file(p, entry.data)) {
        std::cerr << "sb-tune: cannot read " << p << "\n";
        return 2;
      }
      entries.push_back(std::move(entry));
    }
  }

  std::string body;
  body.append(kMagic);
  put_u32(body, kFormatVersion);
  put_u32(body, static_cast<uint32_t>(entries.size()));
  put_str(body, fingerprint);
  put_str(body, device);
  put_str(body, sm_name);
  for (const auto& entry : entries) {
    put_str(body, entry.name);
    put_str(body, entry.data);
  }

  if (!write_file(out_path, body)) {
    std::cerr << "sb-tune: cannot write " << out_path << "\n";
    return 2;
  }

  std::cout << "exported " << entries.size() << " file(s) to " << out_path
            << "\n"
            << "  fingerprint: " << fingerprint << "\n"
            << "  device:      " << device << "\n";
  for (const auto& entry : entries)
    std::cout << "  " << entry.name << " (" << entry.data.size() << " bytes)\n";
  return 0;
}

struct Archive {
  std::string fingerprint;
  std::string device;
  std::string sm_name;
  std::vector<Entry> entries;
};

bool parse_archive(const std::string& body, Archive& out) {
  if (body.size() < kMagic.size() ||
      std::memcmp(body.data(), kMagic.data(), kMagic.size()) != 0) {
    std::cerr << "sb-tune: not a tune archive\n";
    return false;
  }
  size_t i = kMagic.size();
  uint32_t version = 0, count = 0;
  if (!get_u32(body, i, version) || version != kFormatVersion) {
    std::cerr << "sb-tune: unsupported archive version\n";
    return false;
  }
  if (!get_u32(body, i, count)) {
    std::cerr << "sb-tune: truncated archive\n";
    return false;
  }
  if (!get_str(body, i, out.fingerprint) || !get_str(body, i, out.device) ||
      !get_str(body, i, out.sm_name)) {
    std::cerr << "sb-tune: truncated archive header\n";
    return false;
  }
  for (uint32_t n = 0; n < count; ++n) {
    Entry entry;
    if (!get_str(body, i, entry.name) || !get_str(body, i, entry.data)) {
      std::cerr << "sb-tune: truncated archive at entry " << n << "\n";
      return false;
    }
    out.entries.push_back(std::move(entry));
  }
  if (i != body.size()) {
    std::cerr << "sb-tune: trailing bytes in archive\n";
    return false;
  }
  return true;
}

int cmd_import(const std::string& in_path, bool replace) {
  std::string body;
  if (!read_file(in_path, body)) {
    std::cerr << "sb-tune: cannot read " << in_path << "\n";
    return 2;
  }
  Archive archive;
  if (!parse_archive(body, archive)) return 2;

  const std::string fingerprint = sonicboom::tune::build_fingerprint();
  const std::string device = sonicboom::tune::device_dir();
  const std::string sm_name = sm_file_name();
  if (archive.fingerprint != fingerprint) {
    std::cerr << "sb-tune: fingerprint mismatch\n"
              << "  archive: " << archive.fingerprint << "\n"
              << "  current: " << fingerprint << "\n";
    return 2;
  }
  if (archive.device != device || archive.sm_name != sm_name) {
    std::cerr << "sb-tune: device mismatch\n"
              << "  archive: " << archive.device << " (" << archive.sm_name
              << ")\n"
              << "  current: " << device << " (" << sm_name << ")\n";
    return 2;
  }

  // The archive's names are resolved through the same code-derived table the
  // export used, so importing cannot invent a path and a record family this
  // build no longer knows is refused instead of written.
  std::map<std::string, std::string> targets;
  for (const auto& file : sonicboom::tune::table_files())
    targets.emplace(file.name, file.path);
  const fs::path tdir = tune_dir();
  const fs::path adir = fs::path(autotune_file()).parent_path();

  // Non-empty means the destination already holds a record the runtime could
  // read.
  bool non_empty = false;
  std::error_code ec;
  if (fs::is_directory(tdir, ec)) {
    for (const auto& e : fs::directory_iterator(tdir, ec))
      if (is_tune_record(e)) non_empty = true;
  }
  for (const auto& file : sonicboom::tune::table_files())
    if (fs::is_regular_file(file.path, ec)) non_empty = true;
  if (non_empty && !replace) {
    std::cerr << "sb-tune: the destination table is not empty (pass "
                 "--replace to overwrite it)\n";
    return 2;
  }

  for (const auto& entry : archive.entries) {
    fs::path target;
    const auto it = targets.find(entry.name);
    if (it != targets.end()) {
      target = it->second;
    } else if (entry.name.rfind("tune/", 0) == 0) {
      // A record family this build does not enumerate (see export's
      // forward-compatibility note); it lives in the tuner's device directory.
      const std::string leaf = entry.name.substr(5);
      if (leaf.empty() || leaf.find('/') != std::string::npos ||
          leaf == "." || leaf == "..") {
        std::cerr << "sb-tune: invalid entry name " << entry.name << "\n";
        return 2;
      }
      target = tdir / leaf;
    } else if (entry.name.rfind("autotune/", 0) == 0) {
      const std::string leaf = entry.name.substr(9);
      if (leaf != sm_name) {
        std::cerr << "sb-tune: unexpected autotune record " << leaf << "\n";
        return 2;
      }
      target = adir / leaf;
    } else {
      std::cerr << "sb-tune: invalid entry name " << entry.name << "\n";
      return 2;
    }
    if (!write_file(target, entry.data)) {
      std::cerr << "sb-tune: cannot write " << target << "\n";
      return 2;
    }
  }

  std::cout << "imported " << archive.entries.size() << " file(s) from "
            << in_path << "\n"
            << "  fingerprint: " << fingerprint << "\n"
            << "  device:      " << device << "\n";
  for (const auto& entry : archive.entries)
    std::cout << "  " << entry.name << " (" << entry.data.size() << " bytes)\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  const std::string_view cmd = argv[1];
  if (cmd == "-h" || cmd == "--help") return usage();
  if (cmd != "export" && cmd != "import") {
    std::cerr << "sb-tune: unknown command " << cmd << "\n";
    return 2;
  }
  std::string path;
  bool replace = false;
  for (int i = 2; i < argc; ++i) {
    const std::string_view a = argv[i];
    if ((a == "--out" || a == "-o") && i + 1 < argc) {
      path = argv[++i];
    } else if ((a == "--in" || a == "-i") && i + 1 < argc) {
      path = argv[++i];
    } else if (a == "--replace" && cmd == "import") {
      replace = true;
    } else if (a == "-h" || a == "--help") {
      return usage();
    } else {
      std::cerr << "sb-tune: unknown option " << a << "\n";
      return 2;
    }
  }
  if (path.empty()) {
    std::cerr << "sb-tune: " << cmd << " needs " << (cmd == "export" ? "--out" : "--in")
              << " FILE\n";
    return 2;
  }
  return cmd == "export" ? cmd_export(path) : cmd_import(path, replace);
}
