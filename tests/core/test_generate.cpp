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

// Phase 6 integration: autoregressive generation (forward + lm_head + argmax)
// through the fused spine, plus tokens/sec timing.
//
// Loads the Gemma 4 model (skips cleanly when absent) and cross-checks the
// token stream across the two runtime paths, which now *differ* by design:
//   - bespoke single-token decode: model::generate_reference  (the pre-Phase-6
//     oracle — no KV cache, so each step attends over one key and the stream is
//     236761 236761 236761 236761 for the default seed)
//   - spine multi-token decode:    model::generate_spine / generate_spine_cuda
//     (backend-held KV cache — reproduces llama.cpp's incremental decode)
// The spine must reproduce the llama.cpp golden stream, and the public
// model::generate() must route to the spine (M4): generate(CpuQ8K) ==
// generate_spine and generate(Cuda) == generate_spine_cuda. Reports decode
// tokens/sec for each backend.
//
// No tokenizer is in scope, so the seed token and step count are dev-test
// knobs:
//   SONICBOOM_GEN_TOKEN  seed vocab id (default 2)
//   SONICBOOM_GEN_STEPS  decode steps (default 4)

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/exec.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/quant/quantized_matmul.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbquant = sonicboom::quant;

namespace {

int g_failures = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "  FAIL: " << what << "\n";
    ++g_failures;
  }
}

std::string find_model() {
  std::vector<std::string> candidates;
  if (const char* env = std::getenv("SONICBOOM_GEMMA_GGUF")) candidates.push_back(env);
#ifdef SONICBOOM_SRC_DIR
  candidates.push_back(std::string(SONICBOOM_SRC_DIR) +
                       "/models/gemma-4-E4B-it-Q3_K_M.gguf");
#endif
  candidates.push_back("models/gemma-4-E4B-it-Q3_K_M.gguf");
  for (const auto& c : candidates)
    if (std::filesystem::exists(c)) return c;
  return {};
}

uint64_t env_u64(const char* name, uint64_t dflt) {
  const char* s = std::getenv(name);
  if (!s || !*s) return dflt;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s) return dflt;
  return static_cast<uint64_t>(v);
}

// llama.cpp's llama-bench golden for the default dev-test knobs (seed token 2,
// n_ctx 512, 4 decode steps): the multi-token KV-cache stream the spine must
// reproduce. The single-token oracle yields 236761 236761 236761 236761.
const std::vector<int64_t> kGolden = {236761, 107, 236769, 236776};

std::string toks(const std::vector<int64_t>& v) {
  std::string s;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i) s += ' ';
    s += std::to_string(v[i]);
  }
  return s;
}

// Run `steps` decode steps and report tokens/sec. Returns the generated tokens.
std::vector<int64_t> run_generate(const sbmodel::Gemma4Model& m, uint64_t token,
                                  uint64_t steps, sbquant::MatmulBackend backend,
                                  double* tok_per_sec) {
  // One warm-up decode step first: this triggers the backend's one-time costs
  // (CUDA device weight upload + context init; CPU page-cache / AVX2 dispatch)
  // so the timed region measures steady-state decode, not amortized setup. The
  // warm-up output is discarded; the timed run restarts from the same seed token.
  std::vector<int64_t> warmup(1, -1);
  sbmodel::generate_reference(m, token, 0, warmup, backend);

  std::vector<int64_t> tokens(steps, -1);
  const auto t0 = std::chrono::steady_clock::now();
  const uint64_t done = sbmodel::generate_reference(m, token, 0, tokens, backend);
  const auto t1 = std::chrono::steady_clock::now();
  tokens.resize(done);
  const double secs = std::chrono::duration<double>(t1 - t0).count();
  *tok_per_sec = secs > 0.0 ? double(done) / secs : 0.0;
  return tokens;
}

} // namespace

int main() {
  const std::string path = find_model();
  if (path.empty()) {
    std::cout << "  (skipped: Gemma 4 model not found)\n";
    std::cout << "test_generate OK\n";
    return 0;
  }

  auto r = sbgguf::Reader::load(path);
  if (!r) {
    std::cerr << "  failed to load GGUF\n";
    std::cerr << "test_generate FAILED: 1 check(s)\n";
    return 1;
  }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) {
    std::cerr << "  load_gemma4 error: " << m.error() << "\n";
    std::cerr << "test_generate FAILED: 1 check(s)\n";
    return 1;
  }

  const uint64_t seed = env_u64("SONICBOOM_GEN_TOKEN", 2);
  const uint64_t steps = env_u64("SONICBOOM_GEN_STEPS", 4);
  const uint64_t vocab = m->config.vocab_size;

  // --- CPU q8_K oracle (the reference token stream + baseline timing) --------
  double cpu_tps = 0.0;
  std::vector<int64_t> cpu_tokens;
  if (!std::getenv("SONICBOOM_SKIP_CPU")) {
    cpu_tokens =
        run_generate(*m, seed, steps, sbquant::MatmulBackend::CpuQ8K, &cpu_tps);

    check(!cpu_tokens.empty(), "generate(CpuQ8K): produced at least one token");
    check(cpu_tokens.size() == steps, "generate(CpuQ8K): ran all requested steps");
    bool valid = true;
    for (int64_t t : cpu_tokens)
      if (t < 0 || uint64_t(t) >= vocab) valid = false;
    check(valid, "generate(CpuQ8K): all tokens in vocab range");

    std::cout << "  [cpu q8_K] " << cpu_tokens.size() << " tokens (" << toks(cpu_tokens)
              << "), " << cpu_tps << " tok/s (seed " << seed << ")\n";
  }

  // --- CUDA (only when a device is actually present) ------------------------
  if (!sbquant::cuda_available()) {
    std::cout << "  (skipped: no CUDA device; Cuda backend would fall back to CPU)\n";
  } else {
    double cuda_tps = 0.0;
    const auto cuda_tokens =
        run_generate(*m, seed, steps, sbquant::MatmulBackend::Cuda, &cuda_tps);

    if (!cpu_tokens.empty()) {
      check(cuda_tokens.size() == cpu_tokens.size(),
            "generate(Cuda): same step count as CPU");
      check(cuda_tokens == cpu_tokens,
            "generate(Cuda): token stream identical to CPU q8_K oracle");
    }

    std::cout << "  [cuda]     " << cuda_tokens.size() << " tokens, "
              << cuda_tps << " tok/s (" << (cuda_tps / std::max(cpu_tps, 1e-9))
              << "x vs cpu)\n";
  }

  // --- spine (planner + SonicBackend: the Phase 6 fused decode path) --------
  {
    double spine_tps = 0.0;
    std::vector<int64_t> warmup(1, -1);
    sbmodel::generate_spine(*m, seed, 0, warmup);

    std::vector<int64_t> spine_tokens(steps, -1);
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t done = sbmodel::generate_spine(*m, seed, 0, spine_tokens);
    const auto t1 = std::chrono::steady_clock::now();
    spine_tokens.resize(done);
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spine_tps = secs > 0.0 ? double(done) / secs : 0.0;

    check(!spine_tokens.empty(), "generate_spine: produced at least one token");
    check(spine_tokens.size() == steps, "generate_spine: ran all requested steps");
    bool valid = true;
    for (int64_t t : spine_tokens)
      if (t < 0 || uint64_t(t) >= vocab) valid = false;
    check(valid, "generate_spine: all tokens in vocab range");
    // The spine is the multi-token path: it must reproduce llama.cpp's
    // incremental decode (the golden), not the single-token oracle.
    if (seed == 2 && steps == kGolden.size())
      check(spine_tokens == kGolden,
            "generate_spine: token stream matches llama.cpp golden");

    // M4: the public generate() now routes to the spine.
    {
      std::vector<int64_t> routed(steps, -1);
      const uint64_t done = sbmodel::generate(*m, seed, 0, routed,
                                              sbquant::MatmulBackend::CpuQ8K);
      routed.resize(done);
      check(routed == spine_tokens, "generate(CpuQ8K): routes to the spine");
    }

    std::cout << "  [spine]    " << spine_tokens.size() << " tokens (" << toks(spine_tokens)
              << "), " << spine_tps << " tok/s\n";
  }

  // --- prefill (batched prompt) vs per-token prompt through the spine -------
  // The prefill path processes the whole prompt in one batched pass (batched
  // q8_K matmul + full-sequence causal attention) and fills the KV cache in one
  // shot; the reference (use_prefill=false) feeds the prompt one token at a time
  // through the same per-token spine. The two must produce the identical
  // continuation token stream (the M3 gate, CPU).
  {
    const uint64_t pn = env_u64("SONICBOOM_PREFILL_N", 3);
    std::vector<uint64_t> prompt(pn, seed);

    std::vector<int64_t> pf_tokens(steps, -1);
    const uint64_t pf_done =
        sbmodel::generate_prefill_spine(*m, prompt, 0, pf_tokens, 0, /*use_prefill=*/true);
    pf_tokens.resize(pf_done);

    std::vector<int64_t> ref_tokens(steps, -1);
    const uint64_t ref_done =
        sbmodel::generate_prefill_spine(*m, prompt, 0, ref_tokens, 0, /*use_prefill=*/false);
    ref_tokens.resize(ref_done);

    check(pf_done == ref_done, "prefill: same step count as per-token prompt");
    check(pf_tokens == ref_tokens,
          "prefill: token stream identical to per-token prompt");

    // M4: the public generate_prompt() routes a prompt to the prefill spine.
    {
      std::vector<int64_t> routed(steps, -1);
      const uint64_t done = sbmodel::generate_prompt(
          *m, prompt, 0, routed, sbquant::MatmulBackend::CpuQ8K);
      routed.resize(done);
      check(routed == pf_tokens, "generate_prompt(CpuQ8K): routes to the prefill spine");
    }

    std::cout << "  [prefill]  " << pf_tokens.size() << " tokens (" << toks(pf_tokens)
              << ") over a " << pn << "-token prompt\n";
  }

  // --- spine (CUDA): the Phase 6 fused decode path on device ----------------
  if (!sbquant::cuda_available()) {
    std::cout << "  (skipped: no CUDA device; Cuda spine would not run on device)\n";
  } else {
    double spine_cuda_tps = 0.0;
    std::vector<int64_t> warmup(1, -1);
    sbmodel::generate_spine_cuda(*m, seed, 0, warmup);

    std::vector<int64_t> spine_cuda_tokens(steps, -1);
    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t done =
        sbmodel::generate_spine_cuda(*m, seed, 0, spine_cuda_tokens);
    const auto t1 = std::chrono::steady_clock::now();
    spine_cuda_tokens.resize(done);
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    spine_cuda_tps = secs > 0.0 ? double(done) / secs : 0.0;

    check(!spine_cuda_tokens.empty(),
          "generate_spine_cuda: produced at least one token");
    check(spine_cuda_tokens.size() == steps,
          "generate_spine_cuda: ran all requested steps");
    bool valid = true;
    for (int64_t t : spine_cuda_tokens)
      if (t < 0 || uint64_t(t) >= vocab) valid = false;
    check(valid, "generate_spine_cuda: all tokens in vocab range");
    if (seed == 2 && steps == kGolden.size())
      check(spine_cuda_tokens == kGolden,
            "generate_spine_cuda: token stream matches llama.cpp golden");

    // M4: the public generate() routes to the CUDA spine when a device is present.
    {
      std::vector<int64_t> routed(steps, -1);
      const uint64_t done = sbmodel::generate(*m, seed, 0, routed,
                                              sbquant::MatmulBackend::Cuda);
      routed.resize(done);
      check(routed == spine_cuda_tokens, "generate(Cuda): routes to the CUDA spine");
    }

    std::cout << "  [spine-cuda] " << spine_cuda_tokens.size() << " tokens ("
              << toks(spine_cuda_tokens) << "), " << spine_cuda_tps << " tok/s\n";

    // --- prefill (CUDA): batched prompt vs per-token prompt ------------------
    // The CUDA prefill processes the whole prompt in one batched pass (batched
    // f32-activation matmul + flash attention) and fills the device KV caches in
    // one shot; the reference feeds the prompt one token at a time through the
    // per-token CUDA spine. The two must produce the identical continuation token
    // stream (the M3 gate, CUDA).
    {
      const uint64_t pn = env_u64("SONICBOOM_PREFILL_N", 3);
      std::vector<uint64_t> prompt(pn, seed);

      std::vector<int64_t> pf_tokens(steps, -1);
      const uint64_t pf_done = sbmodel::generate_prefill_spine_cuda(
          *m, prompt, 0, pf_tokens, 0, /*use_prefill=*/true);
      pf_tokens.resize(pf_done);

      std::vector<int64_t> ref_tokens(steps, -1);
      const uint64_t ref_done = sbmodel::generate_prefill_spine_cuda(
          *m, prompt, 0, ref_tokens, 0, /*use_prefill=*/false);
      ref_tokens.resize(ref_done);

      check(pf_done == ref_done, "prefill(Cuda): same step count as per-token prompt");
      check(pf_tokens == ref_tokens,
            "prefill(Cuda): token stream identical to per-token prompt");

      // M4: the public generate_prompt() routes a prompt to the CUDA prefill spine.
      {
        std::vector<int64_t> routed(steps, -1);
        const uint64_t done = sbmodel::generate_prompt(
            *m, prompt, 0, routed, sbquant::MatmulBackend::Cuda);
        routed.resize(done);
        check(routed == pf_tokens, "generate_prompt(Cuda): routes to the CUDA prefill spine");
      }

      std::cout << "  [prefill-cuda] " << pf_tokens.size() << " tokens ("
                << toks(pf_tokens) << ") over a " << pn << "-token prompt\n";
    }
  }

  if (g_failures == 0) {
    std::cout << "test_generate OK\n";
    return 0;
  }
  std::cerr << "test_generate FAILED: " << g_failures << " check(s)\n";
  return 1;
}
