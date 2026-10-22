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

#include "reduce_mode.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <utility>

namespace sonicboom::planner::decisions {

namespace {

using Key = std::pair<std::string, std::string>;

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

std::map<Key, Entry>& registry() {
  static std::map<Key, Entry> r;
  return r;
}

std::size_t& conflicts() {
  static std::size_t c = 0;
  return c;
}

bool& loaded() {
  static bool l = false;
  return l;
}

std::string json_escape(std::string_view in) {
  std::string out;
  out.reserve(in.size() + 8);
  for (const char c : in) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      case '\r': out += "\\r"; break;
      default: out.push_back(c); break;
    }
  }
  return out;
}

bool field_string(std::string_view line, std::string_view key,
                  std::string& out) {
  const std::string pattern = "\"" + std::string(key) + "\":\"";
  const std::size_t start = line.find(pattern);
  if (start == std::string_view::npos) return false;
  std::size_t pos = start + pattern.size();
  std::string value;
  while (pos < line.size()) {
    const char c = line[pos++];
    if (c == '\\' && pos < line.size()) {
      const char next = line[pos++];
      switch (next) {
        case 'n': value.push_back('\n'); break;
        case 't': value.push_back('\t'); break;
        case 'r': value.push_back('\r'); break;
        default: value.push_back(next); break;
      }
    } else if (c == '"') {
      out = std::move(value);
      return true;
    } else {
      value.push_back(c);
    }
  }
  return false;
}

bool field_int(std::string_view line, std::string_view key, int64_t& out) {
  const std::string pattern = "\"" + std::string(key) + "\":";
  std::size_t pos = line.find(pattern);
  if (pos == std::string_view::npos) return false;
  pos += pattern.size();
  while (pos < line.size() && line[pos] == ' ') ++pos;
  const std::string text(line.substr(pos));
  char* tail = nullptr;
  const long long value = std::strtoll(text.c_str(), &tail, 10);
  if (tail == nullptr || tail == text.c_str()) return false;
  out = int64_t(value);
  return true;
}

std::string object_json(const Entry& entry) {
  std::ostringstream os;
  os << "{\"name\":\"" << json_escape(entry.name) << "\",\"scope\":\""
     << json_escape(entry.scope) << "\",\"value\":\""
     << json_escape(entry.value) << "\",\"source\":\""
     << source_name(entry.source) << "\",\"env\":\"" << json_escape(entry.env)
     << "\",\"inputs\":\"" << json_escape(entry.inputs)
     << "\",\"candidates\":\"" << json_escape(entry.candidates)
     << "\",\"reason\":\"" << json_escape(entry.reason)
     << "\",\"numerics\":\"" << json_escape(entry.numerics)
     << "\",\"vram_bytes\":" << entry.vram_bytes
     << ",\"tuning_ms\":" << entry.tuning_ms << "}";
  return os.str();
}

std::string registry_path() {
  return (std::filesystem::path(decision_cache_root()) / device_identity() /
          "decisions.jsonl")
      .string();
}

void persist_locked() {
  const std::string file = registry_path();
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(file).parent_path(), ec);
  if (ec) return;
  std::ostringstream os;
  os << "{\"version\":1,\"device\":\"" << json_escape(device_identity())
     << "\"}\n";
  for (const auto& [key, entry] : registry()) os << object_json(entry) << '\n';
  const std::string tmp = file + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return;
    out << os.str();
    if (!out) return;
  }
  std::filesystem::rename(tmp, file, ec);
  if (ec) {
    std::error_code ignored;
    std::filesystem::remove(tmp, ignored);
  }
}

Source parse_source(std::string_view name) {
  if (name == "env") return Source::Env;
  if (name == "cached") return Source::Cached;
  if (name == "measured") return Source::Measured;
  if (name == "verdict") return Source::Verdict;
  if (name == "skipped") return Source::Skipped;
  if (name == "application") return Source::Application;
  return Source::Default;
}

void load_persisted_locked() {
  if (loaded()) return;
  loaded() = true;
  std::ifstream in(registry_path());
  if (!in) return;
  std::string line;
  while (std::getline(in, line)) {
    Entry entry;
    if (!field_string(line, "name", entry.name) || entry.name.empty())
      continue;  // the header line
    if (!field_string(line, "scope", entry.scope)) continue;
    if (!field_string(line, "value", entry.value)) continue;
    std::string source;
    if (field_string(line, "source", source))
      entry.source = parse_source(source);
    field_string(line, "env", entry.env);
    field_string(line, "inputs", entry.inputs);
    field_string(line, "candidates", entry.candidates);
    field_string(line, "reason", entry.reason);
    field_string(line, "numerics", entry.numerics);
    field_int(line, "vram_bytes", entry.vram_bytes);
    field_int(line, "tuning_ms", entry.tuning_ms);
    entry.current = false;
    registry().emplace(Key{entry.name, entry.scope}, std::move(entry));
  }
}

void ensure_loaded_locked() { load_persisted_locked(); }

} // namespace

std::string_view source_name(Source source) {
  switch (source) {
    case Source::Default: return "default";
    case Source::Env: return "env";
    case Source::Cached: return "cached";
    case Source::Measured: return "measured";
    case Source::Verdict: return "verdict";
    case Source::Skipped: return "skipped";
    case Source::Application: return "application";
  }
  return "unknown";
}

bool record(std::string_view name, std::string_view scope,
            std::string_view value, Source source, std::string_view env) {
  Entry entry;
  entry.name = std::string(name);
  entry.scope = std::string(scope);
  entry.value = std::string(value);
  entry.source = source;
  entry.env = std::string(env);
  return record(entry);
}

bool record(const Entry& entry) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  ensure_loaded_locked();
  const Key key{entry.name, entry.scope};
  const auto it = registry().find(key);
  Entry next = entry;
  next.current = true;
  if (it != registry().end()) {
    if (it->second.current) {
      if (it->second.value == next.value &&
          it->second.source == next.source &&
          it->second.env == next.env)
        return true;
      if (it->second.value == next.value) {
        // Same value, richer metadata: keep the first record (the first
        // decision wins), so the metadata that explained it is stable.
        return true;
      }
      ++conflicts();
      return false;
    }
    // A persisted entry: this process decided again and replaces it.
    it->second = std::move(next);
    persist_locked();
    return true;
  }
  registry().emplace(key, std::move(next));
  persist_locked();
  return true;
}

std::vector<Entry> snapshot() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  ensure_loaded_locked();
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
  ensure_loaded_locked();
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
  ensure_loaded_locked();
  std::ostringstream os;
  for (const auto& [key, entry] : registry()) {
    os << entry.name << ' ' << entry.scope << " = " << entry.value << " ["
       << source_name(entry.source) << ']';
    if (!entry.env.empty()) os << " env=" << entry.env;
    if (!entry.inputs.empty()) os << " inputs=\"" << entry.inputs << '"';
    if (!entry.candidates.empty())
      os << " candidates=\"" << entry.candidates << '"';
    if (!entry.reason.empty()) os << " reason=\"" << entry.reason << '"';
    if (!entry.numerics.empty()) os << " numerics=\"" << entry.numerics << '"';
    if (entry.vram_bytes > 0) os << " vram_bytes=" << entry.vram_bytes;
    if (entry.tuning_ms > 0) os << " tuning_ms=" << entry.tuning_ms;
    if (!entry.current) os << " (persisted)";
    os << '\n';
  }
  if (conflicts() > 0)
    os << conflicts() << " conflicting record(s) dropped\n";
  return os.str();
}

std::string dump_json() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  ensure_loaded_locked();
  std::ostringstream os;
  os << "{\n  \"version\": 1,\n  \"device\": \"" << json_escape(device_identity())
     << "\",\n  \"decisions\": [\n";
  bool first = true;
  for (const auto& [key, entry] : registry()) {
    if (!first) os << ",\n";
    first = false;
    os << "    " << object_json(entry);
  }
  os << "\n  ]\n}\n";
  return os.str();
}

namespace {

// One row of the T0b knob table (docs/decision-registry.md). `fallback` is
// the value the runtime uses with the variable unset; the extra text is what
// sb-explain shows next to the decision.
struct KnobRow {
  const char* name;
  const char* var;
  const char* fallback;
  const char* candidates;
  const char* reason;
  const char* numerics;
  int64_t vram_bytes;
  int64_t tuning_ms;
};

constexpr KnobRow kKnobRows[] = {
    {"autotune.service", "SONICBOOM_AUTOTUNE", "on", "on|off|force",
     "tuning service; force retimes every class", "bit-identical gate in the JIT tuner", 0, 0},
    {"autotune.jit", "SONICBOOM_AUTOTUNE_JIT", "on", "on|off",
     "JIT candidate tuner; off = static geometry", "bit-identical gate", 0, 0},
    {"autotune.budget-ms", "SONICBOOM_AUTOTUNE_BUDGET_MS", "10000", "",
     "process-wide cold-tuning budget; <=0 unlimited", "none (timing only)", 0, 0},
    {"cuda-graph.decode", "SONICBOOM_CUDA_GRAPHS", "on", "on|off",
     "capture/replay one decode step", "bit-identical to the launch path", 0, 0},
    {"cuda-graph.prefill", "SONICBOOM_CUDA_GRAPHS", "on", "on|off",
     "capture/replay the prefill segment", "bit-identical to the launch path", 0, 0},
    {"cuda-graph.capture-reuse", "SONICBOOM_CAPTURE_REUSE", "on", "on|off",
     "reuse a cached prefill capture", "bit-identical", 0, 0},
    {"split-kv", "SONICBOOM_SPLIT_KV", "on", "on|off",
     "decode attention split-KV", "split order fixed per class (R1)", 0, 0},
    {"plan-cache", "SONICBOOM_PLAN_CACHE", "on", "on|off",
     "cache prefill segment plans", "none (no arithmetic change)", 0, 0},
    {"symbolic-prefill", "", "on", "on|off",
     "symbolic prefill lowering", "bit-identical to concrete plans", 0, 0},
    {"q8_1-dispatch", "SONICBOOM_GEMV_F32", "on", "on|off",
     "q8_1 activation quant + dp4a/MMQ", "project's own quantized path", 0, 0},
    {"buffer-reuse", "SONICBOOM_BUFFER_REUSE", "on", "on|off",
     "return an activation buffer at last read", "none (memory only)", 0, 0},
    {"warmup", "SONICBOOM_WARMUP", "on", "on|off",
     "one-time decode warm-up before timing", "none (timing only)", 0, 0},
    {"glue-codegen", "", "on", "on|off",
     "generated glue regions", "bit-identical gate per region", 0, 0},
    {"quant.cublas", "SONICBOOM_CUBLAS", "on", "on|off",
     "cuBLAS fp16 path for the quant GEMMs", "same operands, different kernel", 0, 0},
    {"quant.cublas-min-n", "SONICBOOM_CUBLAS_MIN_N", "32", "1|...",
     "cuBLAS crossover token count", "algorithm choice only", 0, 0},
    {"gemv.q6k-epilogue", "SONICBOOM_GEMV_Q6K_EPILOGUE", "on", "on|off",
     "Q6_K gemv epilogue form", "same arithmetic per element", 0, 0},
    {"gemv.sibling-kquant", "SONICBOOM_GEMV_SIBLING_KQUANT", "on", "on|off",
     "K-quant sibling-block gemv", "same arithmetic per element", 0, 0},
    {"mmq.tc", "SONICBOOM_MMQ_TC", "off (auto would need the VRAM budget)",
     "on|off|auto", "MMQ1 generated tensor-core matmul", "fp64-ref gate",
     0, 0},
    {"mmq.tc-max-mb", "SONICBOOM_MMQ_TC_MAX_MB", "2048", "MiB",
     "q8_1 weight-copy budget for the tensor-core path", "none (VRAM only)",
     2048 * 1024 * 1024, 0},
    {"async.vram-threshold-mb", "SONICBOOM_VRAM_THRESHOLD_MB", "6000", "MiB",
     "prefetch only above this VRAM threshold", "none (VRAM only)", 0, 0},
    {"glue.cache", "SONICBOOM_GLUE_CACHE", "on", "on|off",
     "cache generated glue kernels on disk", "none (rebuild only)", 0, 0},
    {"attention.codegen", "SONICBOOM_ATTN_CODEGEN", "auto (tuned tiles)",
     "tile spec|auto", "pin generated prefill tiles (sweeps)", "token-identical",
     0, 0},
    {"mmq.enabled", "SONICBOOM_MMQ", "on", "mmq|cublas-fp16",
     "MMQ measured faster than cuBLAS fp16 at every n", "project's own quantized path", 0, 0},
    {"mmq.min-n", "SONICBOOM_MMQ_MIN_N", "2", "1|2|...",
     "n stays below 2 only for the single-column matvec", "batch-independent MMQ columns", 0, 0},
    {"mmq.j", "SONICBOOM_MMQ_J", "tuner", "8|16|...|128",
     "J tile: tuner when not pinned", "J does not change summation order", 0, 0},
    {"mmq.decomp", "SONICBOOM_MMQ_DECOMP", "auto (tiled policy)", "tiled|stream_k",
     "class-wide decomposition choice", "decomposition changes order; class-wide", 0, 0},
    {"gemv.coop-min-bpr", "SONICBOOM_GEMV_COOP_MIN_BPR", "20", "blocks per row",
     "cooperative gemv blocks-per-row floor", "same arithmetic per element", 0, 0},
    {"attention.impl", "SONICBOOM_ATTENTION_IMPL", "auto", "auto|own|codegen",
     "shape-supported codegen with own fallback", "token-identical per class", 0, 0},
    {"attention.prefill", "SONICBOOM_ATTN_PREFILL", "auto", "auto|own|codegen",
     "per-class prefill attention choice", "fp64-ref error gate", 0, 0},
    {"attention.prefill-d64", "SONICBOOM_ATTN_PREFILL_D64", "auto", "auto|own",
     "decode-head-dim-64 prefill arm", "token-identical", 0, 0},
    {"attention.decode", "SONICBOOM_ATTN_DECODE", "auto (codegen when supported)",
     "auto|own|codegen", "codegen matches the own pair's rounding",
     "token-identical per class", 0, 0},
    {"attention.decode-chunk", "SONICBOOM_ATTN_DECODE_CHUNK", "identity rule",
     "1..16", "chunk scales with the window, not the step", "identity rule", 0, 0},
    {"attention.exp2", "SONICBOOM_ATTN_EXP2", "off", "off|on",
     "exp2-based softmax arm (ablation)", "changes rounding; ablation only", 0, 0},
    {"attention.tune-nonidentical", "SONICBOOM_ATTN_TUNE_NONIDENTICAL", "off",
     "off|on", "runtime chunk tuner", "non-token-identical arm", 0, 0},
    {"dense.cublas", "SONICBOOM_DENSE_CUBLAS", "on", "on|off",
     "dense GEMM via cuBLAS", "fixed-width blocking keeps n-independence", 0, 0},
    {"dense.cublas-min-n", "SONICBOOM_DENSE_CUBLAS_MIN_N", "1", "1|...",
     "cuBLAS crossover", "fixed-width blocking", 0, 0},
    {"dense.lt", "SONICBOOM_DENSE_LT", "on", "on|off",
     "cuBLASLt heuristic for the dense GEMM", "algorithm choice only", 0, 0},
    {"dense.gemm", "SONICBOOM_DENSE_GEMM", "on", "on|off",
     "SonicBoom dense GEMM vs cuBLASLt", "fp64-ref gate per class", 0, 0},
    {"spec.mode", "SONICBOOM_SPEC", "probe", "off|on|auto",
     "probe compares one plain step to one verify step", "verify is exact (greedy)", 0, 0},
    {"spec.bootstrap", "SONICBOOM_SPEC_BOOTSTRAP", "50", "0..",
     "forced k=1 iterations while the EMA warms up", "greedy stream unchanged", 0, 0},
    {"spec.bootstrap-abort", "SONICBOOM_SPEC_BOOTSTRAP_ABORT", "5", "0..",
     "consecutive zero-accept bootstraps end the bootstrap", "greedy stream unchanged", 0, 0},
    {"spec.probe-ms", "SONICBOOM_SPEC_PROBE_MS", "measured", "<plain>,<verify>",
     "test seam: injected probe pair", "measurement only", 0, 0},
    {"spec.draft-model", "SONICBOOM_DRAFT_MODEL", "ngram", "ngram|path",
     "drafter selection", "accept rule is drafter-agnostic", 0, 0},
    {"spec.verify-batch", "SONICBOOM_SPEC_VERIFY_BATCH", "batched", "batched|single",
     "one batched mmvq per projection", "bit-identical launch shapes", 0, 0},
    {"decode.chunk", "SONICBOOM_DECODE_CHUNK", "1", "1..16",
     "plain decode rows per step", "chunk replay is exact", 0, 0},
    {"prefix.align", "SONICBOOM_PREFIX_ALIGN", "on", "on|off",
     "hold the reuse boundary on a prefill chunk", "restores identical reuse", 0, 0},
    {"train.graph", "SONICBOOM_TRAIN_GRAPH", "on", "on|off",
     "capture the training step", "bit-identical launches", 0, 0},
    {"train.aten", "SONICBOOM_TRAIN_ATEN", "off", "off|on",
     "native-torch kernels directly in training", "same kernels, different glue", 0, 0},
    {"train.adamw-lr-attr", "SONICBOOM_ADAMW_LR_ATTR", "off", "off|on",
     "lr from the node attribute (ablation)", "same arithmetic", 0, 0},
    {"async.prefetch", "SONICBOOM_ASYNC_PREFETCH", "off", "on|off",
     "prefill prefetch markers", "no arithmetic change", 0, 0},
};

}  // namespace

void seed_static_knobs() {
  static std::once_flag once;
  std::call_once(once, [] {
    for (const KnobRow& row : kKnobRows) {
      const char* env = (row.var != nullptr && *row.var != '\0')
                          ? std::getenv(row.var)
                          : nullptr;
      const bool set = env != nullptr && *env != '\0';
      Entry entry;
      entry.name = row.name;
      entry.scope = "device";
      entry.value = set ? env : row.fallback;
      entry.source = set ? Source::Env : Source::Default;
      entry.env = row.var != nullptr ? row.var : "";
      entry.inputs = "device=" + device_identity();
      entry.candidates = row.candidates;
      entry.reason = row.reason;
      entry.numerics = row.numerics;
      entry.vram_bytes = row.vram_bytes;
      entry.tuning_ms = row.tuning_ms;
      record(entry);
    }
  });
}

std::string path() { return registry_path(); }

void forget_for_test(std::string_view name, std::string_view scope) {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().erase(Key{std::string(name), std::string(scope)});
}

void reset_for_test() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().clear();
  conflicts() = 0;
  loaded() = false;
  // The tests reset to a clean *registry*, so the persisted file goes too;
  // otherwise the next call would load exactly what was just dropped. Tests
  // point the decision root at a temporary directory.
  std::error_code ec;
  std::filesystem::remove(registry_path(), ec);
}

void drop_memory_for_test() {
  std::lock_guard<std::mutex> lock(registry_mutex());
  registry().clear();
  conflicts() = 0;
  loaded() = false;
}

} // namespace sonicboom::planner::decisions
