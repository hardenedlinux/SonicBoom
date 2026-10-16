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

// Prefill segment cache acceptance (`SONICBOOM_PLAN_CACHE`, planner/plan.h).
//
// The prefill path builds one graph + rewrite pipeline + execution plan per
// segment length and caches it in the decode runtime, keyed by segment length
// and the planner environment (see gemma4_plan.cpp). "Cached" must mean
// *invisible*: a cache hit has to produce the same plan, and the same tokens,
// as a freshly built one. Byte comparison of the execution-plan record
// (planner/plan_record.h) is the check, because it is the one artifact that
// names every step, buffer, attribute and launched kernel the plan encodes.
//
// Properties per model:
//   1. cold == hit          — the first call for a (model, n_ctx) builds, the
//                             second hits, and the two records are byte-equal.
//   2. hit  == uncached     — with SONICBOOM_PLAN_CACHE=0 the plan is rebuilt
//                             per call; that record must equal the hit record.
//   3. tokens unvarying     — cache on and cache off generate the same tokens.
//   4. the key discriminates— a different segment length must not reuse the
//                             entry (N=511 vs N=512 records differ).
//   5. n_ctx is not a plan input — the same segment under a different n_ctx
//                             has a byte-identical record *body* (only the
//                             record header names n_ctx).
//   6. agreement with the fresh-build path — `record_prefill_plan` (the plan_dump
//                             machinery) plans the same segment independently;
//                             its record must agree with the cached one, minus
//                             the final norm + lm_head launches the real path
//                             appends to its last step.
//   7. the offset is not a plan input — the same segment length at off=0 and
//                             off=448 shares one entry (the phase log shows the
//                             second call hitting, not rebuilding) and both
//                             offsets produce the uncached tokens. The offset
//                             reaches the kernels as an execute-time binding.
//
// Skips (exit 77) when there is no CUDA device or no model GGUF is present.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/planner/plan_record.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

namespace {

int g_failures = 0;

void fail(const std::string& what) {
  std::cerr << "  FAIL: " << what << "\n";
  ++g_failures;
}

std::vector<std::string> split_lines(const std::string& s) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == '\n') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) out.push_back(cur);
  return out;
}

bool starts_with(std::string_view s, std::string_view p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// The record's step lines, with the run header dropped: the header names the
// model, mode, n_ctx and n, which two records of the same plan may disagree on.
std::vector<std::string> body_lines(const std::string& text) {
  std::vector<std::string> lines = split_lines(text);
  std::size_t i = 0;
  while (i < lines.size() && !starts_with(lines[i], "step ")) ++i;
  lines.erase(lines.begin(), lines.begin() + std::ptrdiff_t(i));
  return lines;
}

// Report the first differing line, as tools/plan_diff does.
void report_diff(const std::vector<std::string>& a,
                 const std::vector<std::string>& b, const std::string& la,
                 const std::string& lb) {
  const std::size_t common = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < common && a[i] == b[i]) ++i;
  if (i >= common) {
    std::cerr << "  first difference: line count " << a.size() << " (" << la
              << ") vs " << b.size() << " (" << lb << ")\n";
    return;
  }
  std::cerr << "  first difference: line " << i + 1 << "\n    " << la << ": "
            << a[i] << "\n    " << lb << ": " << b[i] << "\n";
}

struct ModelCase {
  const char* name;
  const char* env;
  const char* file;
};

const ModelCase kModels[] = {
    {"gemma4", "SONICBOOM_GEMMA_GGUF", "gemma-4-E4B-it-Q3_K_M.gguf"},
    {"qwen2", "SONICBOOM_QWEN_GGUF", "qwen2.5-0.5b-instruct-q4_k_m.gguf"},
};

// A model together with the reader it borrows from. Gemma4Model's weight spans
// point into the reader's buffer, so the reader must outlive it — and both must
// outlive the runtimes built from them (see main).
struct LoadedModel {
  const ModelCase* c;
  std::unique_ptr<sbgguf::Reader> reader;
  std::unique_ptr<sbmodel::Gemma4Model> model;
};

std::string find_model(const ModelCase& c) {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv(c.env)) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) + "/models/" + c.file);
#endif
  candidates.push_back(std::string("models/") + c.file);
  for (const auto& cand : candidates)
    if (std::filesystem::exists(cand)) return cand;
  return {};
}

// Deterministic prompt: token values do not reach the plan, only shapes do.
std::vector<uint64_t> make_prompt(uint64_t n) {
  std::vector<uint64_t> p(n);
  for (uint64_t i = 0; i < n; ++i) p[i] = 2 + (i % 500);
  return p;
}

struct RunOut {
  std::vector<int64_t> tokens;
  std::string plan;  // serialized record, empty when not recording
  bool ok = false;
};

// One real prefill call: a fresh KV session, optionally recorded. This is the
// path under test — not record_prefill_plan, which builds its own throwaway
// runtime and never touches the cache.
RunOut run_prefill(const sbmodel::Gemma4Model& m, uint64_t n, uint64_t off,
                   uint64_t n_ctx, uint64_t steps, bool record) {
  const std::vector<uint64_t> prompt = make_prompt(n);
  std::vector<int64_t> toks(steps, -1);
  sbplanner::KvCacheSession session;
  session.n_ctx = n_ctx;

  sbplanner::PlanRecorder rec;
  if (record) {
    rec.set_run("test", "prefill", n_ctx, n, 0, 0, "cuda");
    rec.set_device(sbmodel::cuda_device_identity());
    sbplanner::set_thread_recorder(&rec);
  }
  const uint64_t done = sbmodel::generate_prefill_spine_cuda_prefix(
      m, prompt, off, toks, session, true);
  if (record) sbplanner::set_thread_recorder(nullptr);
  RunOut out;
  out.ok = done != 0;
  if (!out.ok) return out;
  toks.resize(done);
  out.tokens = std::move(toks);
  if (record) out.plan = sbplanner::serialize_plan_run(rec.finish());
  return out;
}

// Run `fn` with stderr diverted to a temporary file and return what it wrote.
// The planner's SONICBOOM_PREFILL_PHASE_TIMING lines go to stderr, and they are
// the only place a cache *hit* is distinguishable from a rebuild — both produce
// the same plan record, so the record alone cannot show that an entry was
// shared.
std::string capture_stderr(const std::function<void()>& fn) {
  std::fflush(stderr);
  const int saved = dup(fileno(stderr));
  FILE* tmp = std::tmpfile();
  if (saved < 0 || tmp == nullptr) return {};
  dup2(fileno(tmp), fileno(stderr));
  fn();
  std::fflush(stderr);
  dup2(saved, fileno(stderr));
  close(saved);
  std::fseek(tmp, 0, SEEK_SET);
  std::string out;
  char buf[4096];
  std::size_t got = 0;
  while ((got = std::fread(buf, 1, sizeof buf, tmp)) > 0) out.append(buf, got);
  std::fclose(tmp);
  return out;
}

bool has_line(const std::string& text, std::string_view needle) {
  return text.find(needle) != std::string::npos;
}

void set_cache(bool on) {
  if (on)
    setenv("SONICBOOM_PLAN_CACHE", "1", 1);
  else
    setenv("SONICBOOM_PLAN_CACHE", "0", 1);
}

std::string toks_text(const std::vector<int64_t>& t) {
  std::string s;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (i) s += ' ';
    s += std::to_string(t[i]);
  }
  return s;
}

void check_model(const ModelCase& c, const sbmodel::Gemma4Model& m) {
  std::cout << c.name << "\n";

  const uint64_t N = 512, OFF = 0, NCTX = 513, NCTX4 = 516;
  set_cache(true);

  // Warm up before recording anything. The first prefill call for a (model,
  // n_ctx) builds the decode runtime and captures its decode step, and both of
  // those execute the decode plan through the executor — under a recorder they
  // would show up as steps belonging to no prefill segment. A short prefill at
  // a different segment length creates that runtime without caching the segment
  // length under test.
  if (!run_prefill(m, 64, OFF, NCTX, 1, false).ok) {
    fail(std::string(c.name) + ": warm-up prefill failed");
    return;
  }

  // 1. cold then hit, both recorded.
  const RunOut cold = run_prefill(m, N, OFF, NCTX, 1, true);
  if (!cold.ok) { fail(std::string(c.name) + ": prefill (cold) produced no token"); return; }
  const RunOut hit = run_prefill(m, N, OFF, NCTX, 1, true);
  if (!hit.ok) { fail(std::string(c.name) + ": prefill (hit) produced no token"); return; }
  if (cold.plan != hit.plan) {
    fail(std::string(c.name) + ": cache hit plans a different segment than the cold build");
    report_diff(split_lines(cold.plan), split_lines(hit.plan), "cold", "hit");
  } else {
    std::cout << "  cold == hit: " << split_lines(hit.plan).size() << " lines\n";
  }

  // 2. and 3. the same plan, and the same tokens, with the cache disabled.
  set_cache(false);
  const RunOut uncached = run_prefill(m, N, OFF, NCTX, 1, true);
  if (!uncached.ok) { fail(std::string(c.name) + ": prefill (uncached) produced no token"); return; }
  if (uncached.plan != hit.plan) {
    fail(std::string(c.name) + ": SONICBOOM_PLAN_CACHE=0 plans a different segment");
    report_diff(split_lines(hit.plan), split_lines(uncached.plan), "cached", "uncached");
  } else {
    std::cout << "  hit == uncached\n";
  }

  // Token streams compared over a few decode steps (a longer session, so its
  // own n_ctx: the prefill path rejects a request that does not fit the cache).
  const RunOut gen_uncached = run_prefill(m, N, OFF, NCTX4, 4, false);
  set_cache(true);
  const RunOut gen_cached = run_prefill(m, N, OFF, NCTX4, 4, false);
  if (!gen_uncached.ok || !gen_cached.ok) {
    fail(std::string(c.name) + ": 4-step prefill failed (cached "
         + std::to_string(gen_cached.ok) + ", uncached "
         + std::to_string(gen_uncached.ok) + ")");
    return;
  }
  if (gen_cached.tokens != gen_uncached.tokens) {
    fail(std::string(c.name) + ": cache on/off generate different tokens\n"
         "    cached:   " + toks_text(gen_cached.tokens) +
         "\n    uncached: " + toks_text(gen_uncached.tokens));
  } else {
    std::cout << "  tokens identical cache on/off (" << gen_cached.tokens.size()
              << " steps)\n";
  }

  // 4. a different segment length must not be served by the N=512 entry.
  const RunOut shorter = run_prefill(m, N - 1, OFF, NCTX, 1, true);
  if (!shorter.ok) { fail(std::string(c.name) + ": prefill (N-1) failed"); return; }
  if (shorter.plan == hit.plan) {
    fail(std::string(c.name) + ": N=511 returned the N=512 plan (cache key too loose)");
  } else {
    std::cout << "  N=511 is not served by the N=512 entry\n";
  }

  // 5. n_ctx is not a plan input: the body is the same under a different n_ctx.
  //    n_ctx=1024 is a new decode runtime, so warm it up first — its own decode
  //    runtime warm-up would otherwise be recorded ahead of the segment.
  if (!run_prefill(m, 64, OFF, 1024, 1, false).ok) {
    fail(std::string(c.name) + ": warm-up prefill (n_ctx=1024) failed");
    return;
  }
  const RunOut other_ctx = run_prefill(m, N, OFF, 1024, 1, true);
  if (!other_ctx.ok) { fail(std::string(c.name) + ": prefill (n_ctx=1024) failed"); return; }
  const std::vector<std::string> body_a = body_lines(hit.plan);
  const std::vector<std::string> body_b = body_lines(other_ctx.plan);
  if (body_a != body_b) {
    fail(std::string(c.name) + ": n_ctx changed the plan body");
    report_diff(body_a, body_b, "n_ctx=513", "n_ctx=1024");
  } else {
    std::cout << "  plan body independent of n_ctx (" << body_a.size() << " lines)\n";
  }

  // 6. the same segment planned independently by the plan_dump path.
  auto fresh = sbmodel::record_prefill_plan(m, N);
  if (!fresh) {
    fail(std::string(c.name) + ": record_prefill_plan produced no plan");
    return;
  }
  std::vector<std::string> fresh_body = body_lines(sbplanner::serialize_plan_run(*fresh));
  // The real path's final step carries the final norm + lm_head launches on top
  // of the freshly planned record's last step, so drop the freshly planned
  // record's trailing launch lines and require the rest to be a prefix.
  while (!fresh_body.empty() && starts_with(fresh_body.back(), "  launch "))
    fresh_body.pop_back();
  if (fresh_body.size() > body_a.size() ||
      !std::equal(fresh_body.begin(), fresh_body.end(), body_a.begin())) {
    fail(std::string(c.name) +
         ": the cached segment disagrees with a freshly built one (record_prefill_plan)");
    report_diff(fresh_body, body_a, "fresh", "cached");
  } else {
    std::cout << "  agrees with record_prefill_plan (" << fresh_body.size()
              << " of " << body_a.size() << " lines)\n";
  }

  // 7. the execute offset is not part of the cache key. The n_ctx=1024 warm-up
  //    above built the 64-token entry at off=0; a 64-token call at off=448 must
  //    be served by it — same entry, no rebuild — and produce the tokens the
  //    uncached path produces at that offset. The phase log is what shows the
  //    hit: a rebuilt segment records the same plan, so the record cannot
  //    distinguish the two.
  const uint64_t OFF448 = 448;
  set_cache(true);
  setenv("SONICBOOM_PREFILL_PHASE_TIMING", "1", 1);
  RunOut at0;
  capture_stderr([&] { at0 = run_prefill(m, 64, OFF, 1024, 1, true); });
  RunOut at448;
  const std::string log448 =
      capture_stderr([&] { at448 = run_prefill(m, 64, OFF448, 1024, 1, true); });
  unsetenv("SONICBOOM_PREFILL_PHASE_TIMING");
  if (!at0.ok || !at448.ok) {
    fail(std::string(c.name) + ": 64-token prefill at off=0/off=448 failed");
    return;
  }
  if (!has_line(log448, "segment_cache_hit") ||
      has_line(log448, "segment_build_graph")) {
    fail(std::string(c.name) +
         ": off=448 did not reuse the off=0 entry (cache key depends on the offset)");
  } else {
    std::cout << "  off=448 reuses the off=0 entry (no rebuild)\n";
  }
  if (at0.plan != at448.plan) {
    fail(std::string(c.name) + ": the plan records a different offset");
    report_diff(split_lines(at0.plan), split_lines(at448.plan), "off=0", "off=448");
  }
  set_cache(false);
  const RunOut at448_uncached = run_prefill(m, 64, OFF448, 1024, 1, false);
  set_cache(true);
  if (!at448_uncached.ok || at448_uncached.tokens != at448.tokens) {
    fail(std::string(c.name) +
         ": off=448 cached/uncached tokens differ\n    cached:   "
         + toks_text(at448.tokens)
         + "\n    uncached: " + toks_text(at448_uncached.tokens));
  } else {
    std::cout << "  off=448 tokens identical cache on/off ("
              << at448.tokens.size() << " steps)\n";
  }
}

}  // namespace

int main() {
  if (!sonicboom::quant::cuda_available()) {
    std::cout << "(skipped: no CUDA device — the plan record needs one)\n";
    return 77;
  }
  // Every model is loaded up front and kept alive for the whole run. Two
  // lifetimes are at stake: the weights borrow the reader's buffer, and the
  // prefill/decode runtimes are cached per model *address* and hold raw model
  // pointers. Destroying either early — or letting the next model land on a
  // freed address — hands a later model the earlier one's plans and device
  // weights.
  std::vector<LoadedModel> loaded;
  for (const ModelCase& c : kModels) {
    const std::string path = find_model(c);
    if (path.empty()) {
      std::cout << "(" << c.name << ": skipped, " << c.env
                << " unset and models/" << c.file << " absent)\n";
      continue;
    }
    auto reader = sbgguf::Reader::load(path);
    if (!reader) {
      fail(std::string(c.name) + ": failed to load " + path);
      continue;
    }
    auto model = sbmodel::load_gemma4(*reader);
    if (!model) {
      fail(std::string(c.name) + ": model load error: " + model.error());
      continue;
    }
    LoadedModel lm;
    lm.c = &c;
    lm.reader = std::make_unique<sbgguf::Reader>(std::move(*reader));
    lm.model = std::make_unique<sbmodel::Gemma4Model>(std::move(*model));
    loaded.push_back(std::move(lm));
  }

  const std::size_t ran = loaded.size();
  for (const LoadedModel& lm : loaded) check_model(*lm.c, *lm.model);
  if (ran == 0 && g_failures == 0) {
    std::cout << "(skipped: no model GGUFs found)\n";
    return 77;
  }
  if (g_failures != 0) {
    std::cerr << g_failures << " plan-cache check(s) failed\n";
    return 1;
  }
  std::cout << "plan-cache: " << ran << " model(s) OK\n";
  return 0;
}
