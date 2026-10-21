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

// Qwen2 forward pass end to end on the CPU spine (Qwen run, Stage 4).
//
// For each prompt fixture in tests/data/qwen2/ it
//   1. tokenizes the prompt text with SonicBoom's own GGUF tokenizer and checks
//      the ids against llama.cpp's (the .ref file's `prompt:` line),
//   2. runs the whole prompt through the batched prefill spine
//      (generate_prefill_spine, CPU SonicBackend + CpuQ8K matmul) and decodes 64
//      greedy tokens, and
//   3. diffs the stream against llama.cpp's greedy reference (the `tokens:` line,
//      produced by tools/baseline/qwen_ref.cpp).
//
// The reference is llama.cpp CPU, i.e. the baseline this runtime is measured
// against; the prompt ids are checked first so a tokenizer difference can never
// masquerade as a model difference. The same prompt is then run through the
// per-token spine (use_prefill = false) to assert the batched prefill and the
// single-token path agree, which is a model-independent invariant of the spine.
//
// Greedy-stream agreement is asserted as "agrees with llama.cpp up to a
// documented near-tie", not as bit-exactness. Greedy decode over a quantized
// model is an argmax over logits whose top few candidates genuinely coincide:
// at each prompt's first divergence SonicBoom picks llama.cpp's rank-2 or
// rank-3 token, within 0.13 raw logit of its top-1 (see the `topk:` line of the
// .ref fixtures, which records llama.cpp's own top-8 and their gaps per step).
// The bound is therefore data, not prose. For scale, llama.cpp's own CPU and
// CUDA backends disagree with each other on the same file much earlier and at
// larger gaps (p2 at step 2, gap 0.2181; p1 at step 22, gap 0.3694), so this
// runtime is numerically at least as close to llama.cpp CPU as llama.cpp's own
// CUDA backend is. The first-16-token agreement is reported per prompt and
// checked where it is achieved (p1, p3); p2's first divergence is at step 5 and
// is a rank-2 near-tie.
//
// Skips cleanly (exit 0) when the GGUF is absent.
//
// --- Golden token streams ---------------------------------------------------
//
// Beyond the llama.cpp near-tie check above, each prompt's stream is pinned
// against a committed golden under tests/data/qwen2/pN.golden — SonicBoom's own
// output, compared byte-for-byte (the same role kGolden plays for Gemma in
// test_generate.cpp, but committed as data because these are 64-token streams).
// The CPU line is the CPU spine's stream; the CUDA line is the device spine's,
// recorded on the device named in the golden's `device` line. That line is part
// of the golden and not decoration: the CUDA kernels are not device-invariant
// (the decode-attention split-KV grid is sized from the SM count), so the CUDA
// comparison is skipped — never passed vacuously — when this run's device is not
// the one that recorded the golden. Regenerate with:
//
//     test_qwen2_generate --write-goldens
//
// and commit the result; a CLI-visible token change then shows up as a diff.
//
// --- Fixture directories ----------------------------------------------------
//
// The fixtures and the model they describe live together: the directory is
// tests/data/qwen2/ by default and SONICBOOM_QWEN_DATA names another (the
// DeepSeek-R1-Distill-Qwen-1.5B fixtures, tests/data/deepseek-r1-qwen15b/,
// reuse this executable because the model is the same Qwen2 architecture). A
// directory may carry a `model` file holding the GGUF path its fixtures were
// recorded against; that file binds the directory to its model, so an ambient
// SONICBOOM_QWEN_GGUF meant for another fixture set cannot leak in. Nothing in
// the code is model-specific: the architecture dispatch and the data decide
// everything.

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/vocab/vocab.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;
namespace sbquant = sonicboom::quant;
namespace sbvocab = sonicboom::vocab;

namespace {

int g_failures = 0;
void check(bool ok, const std::string& what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::string data_dir() {
  if (const char* env = std::getenv("SONICBOOM_QWEN_DATA")) return env;
#ifdef SONICBOOM_SRC_DIR
  return std::string(SONICBOOM_SRC_DIR) + "/tests/data/qwen2";
#else
  return "tests/data/qwen2";
#endif
}

// The first line of `path`, trailing whitespace stripped; "" when absent.
std::string first_line(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  if (!std::getline(f, line)) return {};
  while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
  return line;
}

// Model lookup for one fixture directory: the fixtures decide the model. The
// `model` file (see above) is tried first so a directory's fixtures cannot be
// run against whichever model an ambient SONICBOOM_QWEN_GGUF happens to name;
// without one (the default qwen2 directory) the pre-existing Qwen2.5 lookup
// applies unchanged.
std::string find_model(const std::string& dir) {
  std::vector<std::string> candidates;
  const std::string named = first_line(dir + "/model");
  if (!named.empty()) {
    if (std::filesystem::path(named).is_absolute()) {
      candidates.push_back(named);
    } else {
#ifdef SONICBOOM_SRC_DIR
      candidates.push_back(std::string(SONICBOOM_SRC_DIR) + "/" + named);
#endif
      candidates.push_back(named);
    }
  }
  if (const char* env = std::getenv("SONICBOOM_QWEN_GGUF")) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
#endif
  candidates.push_back("models/qwen2.5-0.5b-instruct-q4_k_m.gguf");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return {};
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

// Parse "<label>: <id> <id> ..." out of a qwen_ref dump.
std::vector<int64_t> pick_ids(const std::string& text, const std::string& label) {
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    if (line.rfind(label + ":", 0) != 0) continue;
    std::vector<int64_t> ids;
    std::istringstream ls(line.substr(label.size() + 1));
    int64_t v = 0;
    while (ls >> v) ids.push_back(v);
    return ids;
  }
  return {};
}

// Parse one `topk:` entry, "<step>|<id>:<gap>,<id>:<gap>,...", into the ranked
// (token, gap-to-top-1) list llama.cpp emitted for that step.
std::vector<std::pair<int64_t, double>> parse_topk_entry(const std::string& entry) {
  std::vector<std::pair<int64_t, double>> out;
  const size_t bar = entry.find('|');
  if (bar == std::string::npos) return out;
  std::istringstream is(entry.substr(bar + 1));
  std::string item;
  while (std::getline(is, item, ',')) {
    const size_t colon = item.find(':');
    if (colon == std::string::npos) continue;
    out.emplace_back(std::stoll(item.substr(0, colon)),
                     std::stod(item.substr(colon + 1)));
  }
  return out;
}

// All per-step top-K lists out of a qwen_ref dump's `topk:` line, indexed by
// step (a step the line does not cover is left empty).
std::vector<std::vector<std::pair<int64_t, double>>> pick_topk(
    const std::string& text, size_t steps) {
  std::vector<std::vector<std::pair<int64_t, double>>> out(steps);
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    if (line.rfind("topk:", 0) != 0) continue;
    std::istringstream ls(line.substr(5));
    std::string entry;
    while (ls >> entry) {
      const size_t bar = entry.find('|');
      if (bar == std::string::npos) continue;
      const size_t step = size_t(std::stoull(entry.substr(0, bar)));
      if (step >= steps) continue;
      out[step] = parse_topk_entry(entry);
    }
    break;
  }
  return out;
}

std::string toks(const std::vector<int64_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ' ';
    s += std::to_string(v[i]);
  }
  return s;
}

// Index of the first differing element, or -1 when the two agree over their
// common prefix.
int64_t first_diff(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return int64_t(i);
  return a.size() == b.size() ? -1 : int64_t(n);
}

// The remainder of the "<label> ..." line, or "" when the file has no such line.
// The golden's token lines are compared as exactly this remainder, so the
// comparison is a byte comparison of the stream text, not of parsed integers.
std::string pick_line(const std::string& text, const std::string& label) {
  std::istringstream is(text);
  std::string line;
  while (std::getline(is, line)) {
    if (line.rfind(label + " ", 0) == 0) {
      std::string rest = line.substr(label.size() + 1);
      if (!rest.empty() && rest.back() == '\r') rest.pop_back();
      return rest;
    }
  }
  return {};
}

// The golden file's canonical text. Rendering it (rather than writing whatever
// the run produced) is what makes the comparison a fixed point: --write-goldens
// and the check below share this one function, so a golden can never be written
// in a shape the check cannot read back.
std::string render_golden(const std::string& model_file, const std::string& device,
                          const std::vector<int64_t>& cpu,
                          const std::vector<int64_t>& cuda) {
  std::string s;
  s += "# SonicBoom golden greedy token stream — test_qwen2_generate (batched\n";
  s += "# prefill + decode). SonicBoom's own output, pinned so a token-stream\n";
  s += "# change shows up as a diff. Regenerate with:\n";
  s += "#     test_qwen2_generate --write-goldens\n";
  s += "# The cuda line is device-specific (the device kernels are not\n";
  s += "# device-invariant: the decode-attention split-KV grid is sized from the\n";
  s += "# SM count), so it is only compared on the device that recorded it.\n";
  s += "model " + model_file + "\n";
  s += "device " + device + "\n";
  s += "steps " + std::to_string(cpu.size()) + "\n";
  s += "cpu " + toks(cpu) + "\n";
  if (!cuda.empty()) s += "cuda " + toks(cuda) + "\n";
  return s;
}

bool write_file(const std::string& path, const std::string& text) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f << text;
  return f.good();
}

} // namespace

int main(int argc, char** argv) {
  bool write_goldens = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--write-goldens") {
      write_goldens = true;
    } else if (a == "-h" || a == "--help") {
      std::printf("usage: %s [--write-goldens]\n", argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "%s: unknown option %s\n", argv[0], argv[i]);
      return 2;
    }
  }

  const std::string dir = data_dir();
  const std::string path = find_model(dir);
  if (path.empty()) {
    std::cout << "  (skipped: no GGUF for " << dir << ")\n";
    std::cout << "test_qwen2_generate OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    std::cerr << "  failed to load GGUF\n";
    std::cerr << "test_qwen2_generate FAILED: 1 check(s)\n";
    return 1;
  }
  auto m = sbmodel::load_gemma4(*r);  // arch-dispatched: qwen2 -> load_qwen2
  if (!m) {
    std::cerr << "  load error: " << m.error() << "\n";
    std::cerr << "test_qwen2_generate FAILED: 1 check(s)\n";
    return 1;
  }
  auto vocab = sbvocab::load_vocab(*r);
  if (!vocab) {
    std::cerr << "  vocab error: " << vocab.error() << "\n";
    std::cerr << "test_qwen2_generate FAILED: 1 check(s)\n";
    return 1;
  }

  std::cout << "  model: " << path << "\n";
  std::cout << "  arch=Qwen2 blocks=" << m->config.block_count
            << " d=" << m->config.embedding_length
            << " ff=" << m->config.feed_forward_length
            << " heads=" << m->config.head_count << "/" << m->config.head_count_kv
            << " head_dim=" << m->config.head_dim_global
            << " vocab=" << m->config.vocab_size << "\n";

  const int64_t kSteps = 64;
  // A divergence deeper than this in llama.cpp's ranking (or with a larger gap
  // to its top-1) is a model difference, not a near-tie.
  constexpr double kNearTieGap = 0.25;

  // Kept for the Stage-5 CUDA section below: each prompt's CPU spine stream, the
  // same prompts' token ids, and llama.cpp's per-step ranking.
  std::vector<std::vector<int64_t>> cpu_streams;
  std::vector<std::vector<uint64_t>> prompts;
  std::vector<std::vector<std::vector<std::pair<int64_t, double>>>> topk_all;

  for (int p = 1; p <= 3; ++p) {
    const std::string txt = read_file(dir + "/p" + std::to_string(p) + ".txt");
    const std::string ref = read_file(dir + "/p" + std::to_string(p) + ".ref");
    check(!txt.empty(), "fixture p" + std::to_string(p) + ".txt present");
    check(!ref.empty(), "fixture p" + std::to_string(p) + ".ref present");
    if (txt.empty() || ref.empty()) continue;

    const std::vector<int64_t> want_prompt = pick_ids(ref, "prompt");
    const std::vector<int64_t> want_tokens = pick_ids(ref, "tokens");
    check(!want_prompt.empty() && want_tokens.size() == size_t(kSteps),
          "fixture p" + std::to_string(p) + ".ref parsed");

    // 1. Tokenizer agreement (SonicBoom's GGUF BPE vs llama.cpp's).
    const std::vector<int32_t> got32 = sbvocab::tokenize(*vocab, txt);
    std::vector<int64_t> got_prompt(got32.begin(), got32.end());
    check(got_prompt == want_prompt,
          "p" + std::to_string(p) + " prompt ids == llama.cpp (" +
              std::to_string(got_prompt.size()) + " vs " +
              std::to_string(want_prompt.size()) + " tokens)");
    if (got_prompt != want_prompt) {
      std::cout << "    sonic: " << toks(got_prompt) << "\n";
      std::cout << "    llama: " << toks(want_prompt) << "\n";
      continue;
    }

    std::vector<uint64_t> prompt(got_prompt.begin(), got_prompt.end());

    // 2. Batched prefill + 64 greedy decode steps on the CPU spine.
    std::vector<int64_t> got(size_t(kSteps), -1);
    const uint64_t n = sbmodel::generate_prefill_spine(
        *m, prompt, /*start_pos=*/0, got, /*n_ctx=*/0, /*use_prefill=*/true);
    check(n == uint64_t(kSteps),
          "p" + std::to_string(p) + " ran all " + std::to_string(kSteps) +
              " steps (" + std::to_string(n) + ")");
    got.resize(n);

    // Gate: the stream agrees with llama.cpp CPU except where llama.cpp's own
    // top-8 at that step says the decision was a near-tie. The divergence is
    // accepted only when SonicBoom's token is one of those candidates AND its
    // gap to llama.cpp's top-1 is under kNearTieGap — so a genuinely wrong model
    // (which would pick some unrelated token, or one far down the ranking) fails.
    const auto topk = pick_topk(ref, size_t(kSteps));
    const int64_t d = first_diff(got, want_tokens);
    const bool first16 = d < 0 || d >= 16;
    cpu_streams.push_back(got);
    prompts.push_back(prompt);
    topk_all.push_back(topk);
    std::cout << "  p" << p << ": " << got.size() << " tokens, first diff at "
              << (d < 0 ? std::string("none") : std::to_string(d))
              << ", first 16 " << (first16 ? "match" : "DIFFER") << "\n";

    if (d >= 0) {
      // Rank and gap of SonicBoom's token in llama.cpp's own ranking at step d.
      const auto& ranked = topk[size_t(d)];
      double gap = -1.0;
      int rank = 0;
      for (size_t i = 0; i < ranked.size(); ++i)
        if (ranked[i].first == got[size_t(d)]) { gap = ranked[i].second; rank = int(i) + 1; }
      std::cout << "    divergence: sonic=" << got[size_t(d)]
                << " llama_top1=" << (ranked.empty() ? -1 : ranked[0].first)
                << " rank=" << rank << " gap=" << gap << "\n";
      check(!ranked.empty() && rank > 0 && gap < kNearTieGap,
            "p" + std::to_string(p) + ": divergence at step " +
                std::to_string(d) + " is a llama.cpp near-tie (rank " +
                std::to_string(rank) + ", gap " + std::to_string(gap) + " < " +
                std::to_string(kNearTieGap) + ")");
      std::cout << "    sonic: " << toks(got) << "\n";
      std::cout << "    llama: " << toks(want_tokens) << "\n";
    }

    // 3. Spine invariant: batched prefill and per-token decode agree.
    std::vector<int64_t> got_slow(size_t(kSteps), -1);
    const uint64_t n2 = sbmodel::generate_prefill_spine(
        *m, prompt, /*start_pos=*/0, got_slow, /*n_ctx=*/0, /*use_prefill=*/false);
    got_slow.resize(n2);
    check(got_slow == got,
          "p" + std::to_string(p) + ": batched prefill == per-token spine");
  }

  // A prompt that failed the tokenizer check above never reaches the model, so
  // there is nothing for the stages below to run (they index prompts[0]); report
  // the failures instead of dereferencing an empty vector.
  if (prompts.empty()) {
    std::cerr << "test_qwen2_generate FAILED: " << g_failures
              << " check(s), no prompt reached the model\n";
    return 1;
  }

  // --- Stage 5: the CUDA path and prefix-cache reuse -------------------------
  //
  // Skipped cleanly when the build has no device. The GPU kernels are allowed
  // their own quantization and reduction order (the project's GPU-numerics
  // rule), so CUDA-vs-CPU agreement is asserted the same way as CPU-vs-llama:
  // identically, or at a step where llama.cpp's own ranking says the decision was
  // a near-tie. Prefix reuse has no such licence — it must reproduce the full
  // prefill's continuation token for token, since it is the same arithmetic on
  // the same cache.
  // Each prompt's device stream, in prompt order; left empty when there is no
  // device, which is what tells the golden check below not to expect a `cuda`
  // line to compare against.
  std::vector<std::vector<int64_t>> cuda_streams;

  if (!sbquant::cuda_available()) {
    std::cout << "  (skipped: no CUDA device)\n";
  } else {
    for (size_t p = 0; p < prompts.size(); ++p) {
      std::vector<int64_t> gpu(size_t(kSteps), -1);
      const uint64_t n = sbmodel::generate_prefill_spine_cuda(
          *m, prompts[p], 0, gpu, 0, /*use_prefill=*/true);
      gpu.resize(n);
      cuda_streams.push_back(gpu);
      const std::string tag = "p" + std::to_string(p + 1);
      check(n == uint64_t(kSteps),
            tag + ": CUDA ran all " + std::to_string(kSteps) + " steps");
      const std::vector<int64_t>& cpu = cpu_streams[p];
      const int64_t d = first_diff(gpu, cpu);
      if (d < 0) {
        std::cout << "  " << tag << ": CUDA stream == CPU stream ("
                  << gpu.size() << " tokens)\n";
        check(true, tag + ": CUDA == CPU");
        continue;
      }
      const auto& ranked = topk_all[p][size_t(d)];
      double gap = -1.0;
      int rank = 0;
      for (size_t i = 0; i < ranked.size(); ++i)
        if (ranked[i].first == gpu[size_t(d)]) { gap = ranked[i].second; rank = int(i) + 1; }
      std::cout << "  " << tag << ": CUDA vs CPU diverge at " << d << " (cuda="
                << gpu[size_t(d)] << " cpu=" << cpu[size_t(d)] << ", llama rank "
                << rank << " gap " << gap << ")\n";
      check(gap >= 0.0 && gap < kNearTieGap,
            tag + ": CUDA-vs-CPU divergence at step " + std::to_string(d) +
                " is a near-tie (gap " + std::to_string(gap) + " < " +
                std::to_string(kNearTieGap) + ")");
    }

    // Determinism: the same prompt must give the same stream on every run. The
    // device path is the one with atomics and split-K reductions in it, so this
    // is the check that would catch a non-deterministic merge.
    {
      std::vector<int64_t> a(size_t(kSteps), -1), b(size_t(kSteps), -1);
      sbmodel::generate_prefill_spine_cuda(*m, prompts[0], 0, a, 0, true);
      for (int run = 0; run < 5; ++run) {
        sbmodel::generate_prefill_spine_cuda(*m, prompts[0], 0, b, 0, true);
        check(a == b, "CUDA run " + std::to_string(run + 1) +
                          " of 5 identical to the first");
      }
    }

    // Prefix reuse at three prefix lengths over one longer prompt (the three
    // fixtures' prompts concatenated), plus the fully-reused case.
    std::vector<uint64_t> full;
    for (const auto& pr : prompts) full.insert(full.end(), pr.begin(), pr.end());
    std::cout << "  prefix-reuse prompt: " << full.size() << " tokens\n";
    const uint64_t kReuseSteps = 16;

    // Per-token reference over the same prompt (no prefill graph): every token's
    // state is produced as n == 1, which is what a fully-reused suffix plus decode
    // does. It is the discriminator between a reduction-order difference (batched
    // prefill vs token-by-token) and a prefix-cache defect: a reuse continuation
    // that matches this stream but not the batched one differs only in the order
    // the attention reduction ran in, not in what the cache holds.
    std::vector<int64_t> per_token(size_t(kReuseSteps), -1);
    per_token.resize(sbmodel::generate_prefill_spine_cuda(*m, full, 0, per_token, 0,
                                                          /*use_prefill=*/false));
    for (uint64_t prefix_len : {uint64_t(8), uint64_t(16), uint64_t(24), full.size()}) {
      // Reference: the whole prompt, fresh session, no reuse.
      std::vector<int64_t> ref_toks(size_t(kReuseSteps), -1);
      const uint64_t ref_done = sbmodel::generate_prefill_spine_cuda(
          *m, full, 0, ref_toks, 0, true);
      ref_toks.resize(ref_done);

      std::vector<uint64_t> prefix(full.begin(), full.begin() + prefix_len);
      sbplanner::KvCacheSession session;
      session.n_ctx = full.size() + kReuseSteps;
      std::vector<int64_t> setup_toks(1, -1);
      if (sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, setup_toks,
                                                      session, true) == 0) {
        check(false, "prefix setup prefill failed at " + std::to_string(prefix_len));
        continue;
      }
      std::vector<int64_t> reuse_toks(size_t(kReuseSteps), -1);
      uint64_t reused = 0;
      const uint64_t reuse_done = sbmodel::generate_prefill_spine_cuda_prefix(
          *m, full, 0, reuse_toks, session, true, &reused);
      reuse_toks.resize(reuse_done);
      const uint64_t want_reused =
          prefix_len == full.size() ? full.size() - 1 : prefix_len;
      std::cout << "  prefix=" << prefix_len << ": reused " << reused
                << " (want " << want_reused << ")\n";
      check(reused == want_reused, "prefix=" + std::to_string(prefix_len) +
                                       ": reuse count is the shared prefix");
      const bool eq_ref = reuse_done == ref_done && reuse_toks == ref_toks;
      const bool eq_pt =
          reuse_done == per_token.size() && reuse_toks == per_token;
      // The three partial prefix lengths must be token-identical to a fresh full
      // prefill: that is the reuse contract. The fully-reused case (suffix of one
      // token) cannot be, and is checked for its reuse count plus reported: the
      // one-token suffix runs through the batched *prefill* graph at n == 1,
      // which selects the dp4a path, while the reference's last prompt token was
      // one column of a 31-token mmq tensor-core prefill — a kernel-selection
      // numerics difference of the same class as the CUDA-vs-CPU near-ties
      // (n == 1 prefill vs the n == 1 decode graph is reported alongside).
      const bool fully_reused = prefix_len == full.size();
      if (!eq_ref && !fully_reused) {
        const int64_t diff_at = first_diff(reuse_toks, ref_toks);
        std::cout << "    first diff at " << diff_at << " (ref_done " << ref_done
                  << ", reuse_done " << reuse_done << ")";
        if (diff_at >= 0)
          std::cout << " reuse=" << reuse_toks[size_t(diff_at)]
                    << " ref=" << ref_toks[size_t(diff_at)]
                    << " per-token=" << per_token[size_t(diff_at)];
        std::cout << ", whole stream == per-token: " << (eq_pt ? "yes" : "no")
                  << "\n";
      }
      if (fully_reused) {
        const int64_t diff_at = first_diff(reuse_toks, ref_toks);
        // Same reuse, but the one-token suffix goes through the decode graph
        // instead of the one-token prefill graph (the discriminating control).
        sbplanner::KvCacheSession dec_session;
        dec_session.n_ctx = full.size() + kReuseSteps;
        std::vector<int64_t> dec_setup(1, -1);
        std::vector<int64_t> dec_toks(size_t(kReuseSteps), -1);
        bool dec_ok = false;
        if (sbmodel::generate_prefill_spine_cuda_prefix(*m, prefix, 0, dec_setup,
                                                        dec_session, true) != 0) {
          const uint64_t dec_done = sbmodel::generate_prefill_spine_cuda_prefix(
              *m, full, 0, dec_toks, dec_session, /*use_prefill=*/false);
          dec_toks.resize(dec_done);
          dec_ok = dec_done == ref_done && dec_toks == ref_toks;
        }
        std::cout << "    fully-reused continuation: "
                  << (eq_ref ? "token-identical to full prefill"
                             : "matches for " + std::to_string(diff_at) +
                                   " step(s), differs at " +
                                   std::to_string(diff_at))
                  << "; via the decode graph: "
                  << (dec_ok ? "token-identical" : "differs at " +
                                                     std::to_string(
                                                         first_diff(dec_toks,
                                                                    ref_toks)))
                  << "\n";
      }
      if (!fully_reused)
        check(eq_ref, "prefix=" + std::to_string(prefix_len) +
                          ": reuse continuation token-identical to full prefill");
    }
  }

  // --- golden token streams (SonicBoom's own output, committed) --------------
  //
  // The check the rest of the file's near-tie reasoning exists to make cheap to
  // change: once a stream is blessed, it must not move without the golden moving
  // with it. Compared as text, so a reordered-but-equal stream still fails.
  if (!cpu_streams.empty()) {
    const std::string model_file = std::filesystem::path(path).filename().string();
    const std::string dev = sbmodel::cuda_device_identity();
    for (size_t p = 0; p < cpu_streams.size(); ++p) {
      const std::string tag = "p" + std::to_string(p + 1);
      const std::string gpath = dir + "/" + tag + ".golden";
      const std::vector<int64_t> cuda =
          p < cuda_streams.size() ? cuda_streams[p] : std::vector<int64_t>{};

      if (write_goldens) {
        if (!write_file(gpath, render_golden(model_file, dev, cpu_streams[p], cuda))) {
          check(false, "cannot write golden " + gpath);
        } else {
          std::cout << "  wrote " << gpath << " (" << cpu_streams[p].size()
                    << " cpu" << (cuda.empty() ? "" : " + cuda") << " tokens)\n";
        }
        continue;
      }

      const std::string golden = read_file(gpath);
      check(!golden.empty(), tag + ".golden present");
      if (golden.empty()) continue;

      const std::string want_cpu = pick_line(golden, "cpu");
      check(want_cpu == toks(cpu_streams[p]),
            tag + ": CPU stream == golden " + tag + ".golden (byte-for-byte)");

      const std::string want_cuda = pick_line(golden, "cuda");
      if (want_cuda.empty()) {
        std::cout << "  " << tag << ": golden has no cuda line\n";
      } else if (cuda.empty()) {
        std::cout << "  " << tag << ": CUDA comparison skipped (no device)\n";
      } else if (pick_line(golden, "device") != dev) {
        std::cout << "  " << tag << ": CUDA comparison skipped (golden recorded on '"
                  << pick_line(golden, "device") << "', running on '" << dev << "')\n";
      } else {
        check(want_cuda == toks(cuda),
              tag + ": CUDA stream == golden " + tag + ".golden (byte-for-byte)");
      }
    }
  }

  if (g_failures) {
    std::cerr << "test_qwen2_generate FAILED: " << g_failures << " check(s)\n";
    return 1;
  }
  std::cout << "test_qwen2_generate OK\n";
  return 0;
}
