// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// gen_from_tokens — SonicBoom greedy continuation of a token-id prompt through
// the CUDA prefill spine, printing the generated ids so they can be compared
// token for token against llama_gen_tokens on the same prompt. Built for the M3
// rope-scaling check (generic linear/YaRN metadata): a frequency-scaling bug or
// a missing scaling key changes the continuation, so an identical stream is a
// strong statement that the graph computes what llama.cpp computes.
//
// Usage: gen_from_tokens -m model.gguf --tokens FILE -n N [--ctx N]
// Prints the generated ids on stdout, space separated, one line.
//
// Build: tools/baseline/build_gen_from_tokens.sh

#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/planner/kv_cache.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sbgguf = sonicboom::gguf;
namespace sbmodel = sonicboom::model;
namespace sbplanner = sonicboom::planner;

int main(int argc, char** argv) {
  std::string model_path;
  std::string tokens_path;
  int64_t n_steps = 16;
  uint64_t n_ctx = 0;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-m")) model_path = argv[++i];
    else if (!std::strcmp(argv[i], "--tokens")) tokens_path = argv[++i];
    else if (!std::strcmp(argv[i], "-n")) n_steps = std::atoll(argv[++i]);
    else if (!std::strcmp(argv[i], "--ctx")) n_ctx = std::atoll(argv[++i]);
  }
  if (model_path.empty() || tokens_path.empty()) {
    fprintf(stderr,
            "usage: gen_from_tokens -m model.gguf --tokens FILE -n N "
            "[--ctx N]\n");
    return 2;
  }

  std::vector<uint64_t> prompt;
  {
    std::ifstream in(tokens_path);
    if (!in) {
      fprintf(stderr, "tokens file not found: %s\n", tokens_path.c_str());
      return 2;
    }
    uint64_t tok;
    while (in >> tok) prompt.push_back(tok);
    if (prompt.empty()) {
      fprintf(stderr, "tokens file empty: %s\n", tokens_path.c_str());
      return 2;
    }
  }
  if (n_ctx == 0) n_ctx = prompt.size() + (uint64_t)n_steps + 8;

  auto r = sbgguf::Reader::load(model_path);
  if (!r) { fprintf(stderr, "failed to load GGUF\n"); return 1; }
  auto m = sbmodel::load_gemma4(*r);
  if (!m) { fprintf(stderr, "model load error\n"); return 1; }

  std::fprintf(stderr, "rope_freqs: %zu entries", m->weights.rope_freqs.size());
  if (!m->weights.rope_freqs.empty())
    std::fprintf(stderr, " first=%.6f last=%.6f", m->weights.rope_freqs.front(),
                 m->weights.rope_freqs.back());
  std::fprintf(stderr, " attn_scale=%.6f", m->weights.rope_attn_scale);
  std::fprintf(stderr, "\n");
  if (!m->plan.empty())
    std::fprintf(stderr, "layer0: attention=%d head_dim=%u rope_base=%g "
                 "sliding=%llu\n", (int)m->plan[0].attention,
                 m->plan[0].head_dim, m->plan[0].rope_base,
                 (unsigned long long)m->plan[0].sliding_window);

  std::vector<int64_t> out((size_t)n_steps, -1);
  sbplanner::KvCacheSession session;
  session.n_ctx = n_ctx;
  const uint64_t done = sbmodel::generate_prefill_spine_cuda_prefix(
      *m, prompt, 0, out, session, /*use_prefill=*/true);
  if (done == 0) { fprintf(stderr, "generation failed\n"); return 1; }

  for (uint64_t i = 0; i < done; ++i) {
    if (i) printf(" ");
    printf("%lld", (long long)out[(size_t)i]);
  }
  printf("\n");
  return 0;
}
