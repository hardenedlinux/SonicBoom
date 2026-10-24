// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// llama_gen_tokens — llama.cpp greedy continuation of a token-id prompt, as
// the baseline reference for SonicBoom's own long-context generation (the M3
// rope-scaling token comparison). The prompt arrives as token ids rather than
// text so both engines see the identical token sequence and no tokenizer
// difference can mask or fake a logits difference.
//
// Usage: llama_gen_tokens -m model.gguf --tokens FILE -n N [--ctx N]
// Prints the generated ids on stdout, space separated, one line.
//
// Build: tools/baseline/build_llama_gen_tokens.sh  (dev-time, links llama.cpp)

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* tokens_path = nullptr;
  int32_t n_steps = 16;
  int32_t n_ctx = 0;
  bool detok = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "-m")) model_path = argv[++i];
    else if (!std::strcmp(argv[i], "--tokens")) tokens_path = argv[++i];
    else if (!std::strcmp(argv[i], "-n")) n_steps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--ctx")) n_ctx = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--detok")) detok = true;
  }
  if (!model_path || !tokens_path) {
    std::fprintf(stderr,
                 "usage: llama_gen_tokens -m model.gguf --tokens FILE -n N "
                 "[--ctx N] [--detok]\n");
    return 2;
  }

  std::vector<llama_token> prompt;
  {
    std::ifstream in(tokens_path);
    if (!in) {
      std::fprintf(stderr, "tokens file not found: %s\n", tokens_path);
      return 2;
    }
    uint64_t tok;
    while (in >> tok) prompt.push_back((llama_token)tok);
    if (prompt.empty()) {
      std::fprintf(stderr, "tokens file empty: %s\n", tokens_path);
      return 2;
    }
  }
  if (n_ctx <= 0) n_ctx = (int32_t)prompt.size() + n_steps + 8;

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();

  if (detok) {
    // Vocabulary-only mode: decode a token-id file back to text (the needle
    // test reads the model's continuation as text, not ids). Weights stay on
    // the CPU because none of them are used.
    mparams.n_gpu_layers = 0;
    llama_model* vm = llama_model_load_from_file(model_path, mparams);
    if (!vm) {
      std::fprintf(stderr, "failed to load model: %s\n", model_path);
      return 1;
    }
    const llama_vocab* vocab = llama_model_get_vocab(vm);
    for (const llama_token t : prompt) {
      char buf[256];
      const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
      if (n > 0) std::fwrite(buf, 1, (size_t)n, stdout);
    }
    std::fputc('\n', stdout);
    llama_model_free(vm);
    llama_backend_free();
    return 0;
  }

  mparams.n_gpu_layers = 99;
  llama_model* model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    std::fprintf(stderr, "failed to load model: %s\n", model_path);
    return 1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = (uint32_t)n_ctx;
  cparams.n_batch = (uint32_t)prompt.size();
  cparams.n_ubatch = (uint32_t)prompt.size();
  // The comparison is over numerics, not speed: fp32 KV keeps llama.cpp on its
  // most precise path and removes its own cache quantization from the diff.
  cparams.type_k = GGML_TYPE_F32;
  cparams.type_v = GGML_TYPE_F32;
  cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
  llama_context* ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    std::fprintf(stderr, "failed to create context\n");
    llama_model_free(model);
    return 1;
  }

  llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
  llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

  std::vector<int32_t> out;
  llama_batch batch = llama_batch_get_one(prompt.data(), (int32_t)prompt.size());
  if (llama_decode(ctx, batch) != 0) {
    std::fprintf(stderr, "prompt decode failed\n");
    return 1;
  }
  for (int32_t i = 0; i < n_steps; ++i) {
    llama_token id = llama_sampler_sample(sampler, ctx, -1);
    out.push_back((int32_t)id);
    if (llama_vocab_is_eog(llama_model_get_vocab(model), id)) break;
    llama_batch one = llama_batch_get_one(&id, 1);
    if (llama_decode(ctx, one) != 0) {
      std::fprintf(stderr, "decode failed at step %d\n", i);
      return 1;
    }
  }

  for (size_t i = 0; i < out.size(); ++i) {
    if (i) std::printf(" ");
    std::printf("%d", out[i]);
  }
  std::printf("\n");

  llama_sampler_free(sampler);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
