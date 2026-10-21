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

// The decision registry (docs/decision-registry.md): one decision per
// (name, scope), conflicts counted and dropped, a fingerprint that follows the
// decision set, and the automatic choices — the hand-written tuner's picks and
// the JIT candidate tuner's cached/skipped classes — recorded as they resolve.
// Everything runs against a temporary decision cache, so the host cache is not
// touched.

#include "../../core/src/planner/autotune.h"
#include "../../core/src/planner/decision_registry.h"
#include "../../core/src/planner/mlir_autotune.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

namespace dr = sonicboom::planner::decisions;
namespace mt = sonicboom::mlir_tune;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

const dr::Entry* find(const std::vector<dr::Entry>& entries,
                      const std::string& name, const std::string& scope) {
  for (const dr::Entry& entry : entries)
    if (entry.name == name && entry.scope == scope) return &entry;
  return nullptr;
}

} // namespace

int main() {
  namespace fs = std::filesystem;

  char tmpl[] = "/tmp/sb-decision-registry-XXXXXX";
  if (::mkdtemp(tmpl) == nullptr) {
    std::cerr << "mkdtemp failed\n";
    return 1;
  }
  const fs::path root(tmpl);
  ::setenv("XDG_CACHE_HOME", root.c_str(), 1);
  ::unsetenv("SONICBOOM_TUNE_DIR");
  ::unsetenv("SONICBOOM_AUTOTUNE");
  ::unsetenv("SONICBOOM_AUTOTUNE_JIT");
  ::unsetenv("SONICBOOM_REDUCE");

  // --- registry mechanics ---------------------------------------------------
  dr::reset_for_test();
  check(dr::snapshot().empty(), "a fresh registry is empty");
  check(dr::fingerprint() == 0, "an empty registry has fingerprint 0");
  check(dr::record("tune/x", "class-a", "1,2,3", dr::Source::Cached),
        "the first record is stored");
  const uint64_t fp_a = dr::fingerprint();
  check(fp_a != 0, "a recorded decision has a fingerprint");
  check(dr::record("tune/x", "class-a", "1,2,3", dr::Source::Measured),
        "the same value again is a no-op");
  check(dr::conflict_count() == 0, "an identical record is not a conflict");
  check(!dr::record("tune/x", "class-a", "9,9,9", dr::Source::Measured),
        "a different value for the same scope is refused");
  check(dr::conflict_count() == 1, "the conflict is counted");
  const auto after_conflict = dr::snapshot();
  check(after_conflict.size() == 1 &&
            after_conflict[0].value == "1,2,3" &&
            after_conflict[0].source == dr::Source::Cached,
        "the first decision wins");
  check(dr::record("tune/x", "class-b", "4,5,6", dr::Source::Measured,
                   "SONICBOOM_X"),
        "a different scope is a different decision");
  check(dr::fingerprint() != fp_a, "the fingerprint follows the set");
  const auto all = dr::snapshot();
  check(all.size() == 2 && all[0].scope == "class-a" && all[1].scope == "class-b",
        "the snapshot is sorted by (name, scope)");
  check(all[1].env == "SONICBOOM_X" && all[1].source == dr::Source::Measured,
        "the env override and the source round-trip");
  const std::string dump = dr::dump_text();
  check(dump.find("tune/x class-b = 4,5,6 [measured] env=SONICBOOM_X") !=
            std::string::npos,
        "the dump names the choice, source and env");
  check(dump.find("conflicting") != std::string::npos,
        "the dump reports the dropped conflict");

  // --- a full decision: inputs, candidates, reason, costs, persistence ------
  {
    dr::Entry entry;
    entry.name = "attention.decode";
    entry.scope = "d256-gqa4";
    entry.value = "codegen chunk=16";
    entry.source = dr::Source::Measured;
    entry.env = "SONICBOOM_ATTN_DECODE";
    entry.inputs = "sm=86, class=d256-gqa4";
    entry.candidates = "own|codegen";
    entry.reason = "the measured arm within the token-identity gate";
    entry.numerics = "token-identical";
    entry.vram_bytes = 4096;
    entry.tuning_ms = 7;
    check(dr::record(entry), "the full decision is stored");
    const std::string text = dr::dump_text();
    check(text.find("reason=\"the measured arm within the token-identity gate\"") !=
              std::string::npos &&
              text.find("vram_bytes=4096") != std::string::npos &&
              text.find("tuning_ms=7") != std::string::npos,
          "the human table carries the reason and the declared costs");
    const std::string json = dr::dump_json();
    check(json.find("\"name\":\"attention.decode\"") != std::string::npos &&
              json.find("\"scope\":\"d256-gqa4\"") != std::string::npos &&
              json.find("\"vram_bytes\":4096") != std::string::npos &&
              json.find("\"tuning_ms\":7") != std::string::npos,
          "the JSON carries the same decision");

    // A new process reloads the persisted record as historical: the dump marks
    // it, and this process may replace it with a different (env) value.
    const fs::path persisted = dr::path();
    check(fs::exists(persisted), "the decision is persisted for the device");
    dr::drop_memory_for_test();
    const auto reloaded = dr::snapshot();
    const dr::Entry* hist = find(reloaded, "attention.decode", "d256-gqa4");
    check(hist && !hist->current && hist->vram_bytes == 4096 &&
              hist->reason == "the measured arm within the token-identity gate",
          "a persisted record reloads with its metadata");
    check(dr::dump_text().find("(persisted)") != std::string::npos,
          "the table marks a reloaded record");
    dr::Entry override_entry = entry;
    override_entry.value = "own";
    override_entry.source = dr::Source::Env;
    check(dr::record(override_entry),
          "this process replaces the historical record");
    check(dr::conflict_count() == 0,
          "replacing a historical record is not a conflict");
    check(find(dr::snapshot(), "attention.decode", "d256-gqa4")->source ==
              dr::Source::Env,
          "the new record wins");
    dr::reset_for_test();
  }

  // --- the T0b knob table is a set of decisions ------------------------------
  {
    dr::reset_for_test();
    ::unsetenv("SONICBOOM_MMQ");
    ::setenv("SONICBOOM_MMQ_MIN_N", "4", 1);
    dr::seed_static_knobs();
    const auto entries = dr::snapshot();
    const dr::Entry* mmq = find(entries, "mmq.enabled", "device");
    check(mmq && mmq->source == dr::Source::Default && mmq->value == "on" &&
              mmq->env == "SONICBOOM_MMQ",
          "an unset knob records its static default");
    const dr::Entry* min_n = find(entries, "mmq.min-n", "device");
    check(min_n && min_n->source == dr::Source::Env && min_n->value == "4",
          "a set knob records the env override");
    dr::seed_static_knobs();
    check(dr::conflict_count() == 0,
          "seeding twice is idempotent, not a conflict");
    ::unsetenv("SONICBOOM_MMQ_MIN_N");
    dr::reset_for_test();
  }

  // --- the hand-written tuner records its picks -----------------------------
  {
    dr::reset_for_test();
    ::setenv("SONICBOOM_AUTOTUNE", "1", 1);
    sonicboom::tune::store("mmq-j", "m512-n4864-k896",
                           {{"j", 64}, {"split", 2}});
    const auto stored = dr::snapshot();
    const dr::Entry* entry =
        find(stored, "tune/mmq-j", "m512-n4864-k896");
    check(entry != nullptr, "a stored pick is recorded");
    check(entry && entry->source == dr::Source::Measured,
          "a pick timed here is Measured");
    check(entry && entry->value == "j=64,split=2",
          "the value is the serialized pick");

    // The next process reads the same pick from disk: the registry says
    // Cached. Clearing the in-memory maps (not the files) models that read.
    dr::reset_for_test();
    sonicboom::tune::clear_memory_cache();
    const auto cached = sonicboom::tune::lookup("mmq-j", "m512-n4864-k896");
    check(cached.size() == 2 && cached.at("j") == 64,
          "the pick reads back from disk");
    const auto from_disk = dr::snapshot();
    const dr::Entry* hit = find(from_disk, "tune/mmq-j", "m512-n4864-k896");
    check(hit && hit->source == dr::Source::Cached &&
              hit->value == "j=64,split=2",
          "a cached pick is recorded as Cached");
    ::unsetenv("SONICBOOM_AUTOTUNE");
  }

  // --- the JIT candidate tuner records cached and skipped classes -----------
  {
    dr::reset_for_test();
    auto& cache = mt::AutotuneCache::instance();
    cache.reset_for_test();
    const std::string cls = mt::gemv_shape_class(33, 3000);
    const mt::TileConfig winner{128, 1, 64, 1, 1};
    check(cache.store("gemv", cls, winner), "seed a JIT record");
    const mt::GemvChoice choice = mt::gemv_choice(33, 3000);
    check(choice.source == mt::GemvSource::Cache && choice.cfg == winner,
          "the seeded gemv class resolves from the cache");
    const auto entries = dr::snapshot();
    const dr::Entry* entry = find(entries, "tune/gemv", cls);
    check(entry != nullptr, "a cached JIT pick is recorded");
    check(entry && entry->source == dr::Source::Cached &&
              entry->value == winner.to_string(),
          "the JIT value and source are recorded");

    // An exhausted cold budget resolves the class to the static default and
    // records it as Skipped, not as a measurement.
    dr::reset_for_test();
    ::setenv("SONICBOOM_AUTOTUNE_BUDGET_MS", "1", 1);
    mt::reset_budget_for_test();
    check(mt::budget_remaining_ms() <= 1, "the 1 ms budget starts");
    ::usleep(5000);
    const mt::MmqTcChoice skipped = mt::mmq_tc_choice(512, 4864, 896);
    check(skipped.source == mt::MmqTcSource::Skipped,
          "the budget expires before an unresolved class");
    const auto after_skip = dr::snapshot();
    const dr::Entry* skip_entry = find(
        after_skip, "tune/matmul_q8_tc", mt::mmq_tc_shape_class(512, 4864, 896));
    check(skip_entry && skip_entry->source == dr::Source::Skipped &&
              skip_entry->value == mt::mmq_tc_default_config().to_string(),
          "a skipped class records the static default and the source");
    ::unsetenv("SONICBOOM_AUTOTUNE_BUDGET_MS");
    mt::reset_budget_for_test();
  }

  std::error_code ec;
  fs::remove_all(root, ec);

  if (g_failures == 0) {
    std::cout << "test_decision_registry OK\n";
    return 0;
  }
  std::cerr << "test_decision_registry FAILED: " << g_failures << " check(s)\n";
  return 1;
}
