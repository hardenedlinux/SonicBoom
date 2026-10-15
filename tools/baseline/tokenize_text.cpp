// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// tokenize_text — encode a real text file into GGUF vocab token ids using the
// llama.cpp tokenizer (the baseline reference tokenizer for the same GGUF that
// SonicBoom loads). Used to build a real-text prompt for the prefix-cache C.4
// tests, replacing the synthetic token sequence.
//
// Usage: tokenize_text <model.gguf> <text-file> [max_tokens]
// Prints space-separated token ids on stdout (one line), count on stderr.
//
// Build: tools/baseline/build_tokenize_text.sh  (dev-time, links llama.cpp)

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool slurp(const char* path, std::string& out) {
  FILE* f = std::fopen(path, "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0) { std::fclose(f); return false; }
  out.resize((size_t)n);
  if (n > 0 && std::fread(&out[0], 1, (size_t)n, f) != (size_t)n) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: tokenize_text <model.gguf> <text-file> [max_tokens]\n");
    return 2;
  }
  const char* model_path = argv[1];
  const char* text_path = argv[2];
  int32_t max_tokens = argc > 3 ? (int32_t)std::atoi(argv[3]) : 0;  // 0 = unlimited

  std::string text;
  if (!slurp(text_path, text)) {
    std::fprintf(stderr, "failed to read text: %s\n", text_path);
    return 1;
  }

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = 0;  // tokenizer only; keep weights on CPU (cheap)

  llama_model* model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    std::fprintf(stderr, "failed to load model: %s\n", model_path);
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);

  if (max_tokens <= 0) max_tokens = 1 << 20;
  // TOKENIZE_PARSE_SPECIAL=1 turns on llama.cpp's parse_special: occurrences of
  // the vocab's CONTROL / USER_DEFINED pieces (e.g. "<|turn>", "<turn|>") are
  // emitted as their own token id instead of being BPE'd as text. Off by
  // default, matching the golden ids recorded for the tokenizer tests.
  const char* ps = std::getenv("TOKENIZE_PARSE_SPECIAL");
  const bool parse_special = ps && *ps && std::strcmp(ps, "0") != 0;

  std::vector<llama_token> tokens((size_t)max_tokens);
  const int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                   tokens.data(), max_tokens,
                                   /*add_special=*/true, parse_special);
  if (n < 0) {
    std::fprintf(stderr, "tokenize failed (need %d tokens)\n", -n);
    llama_model_free(model);
    return 1;
  }

  for (int32_t i = 0; i < n; ++i) {
    if (i) std::printf(" ");
    std::printf("%d", tokens[i]);
  }
  std::printf("\n");
  std::fprintf(stderr, "tokenized %zu chars -> %d tokens\n", text.size(), n);

  llama_model_free(model);
  return 0;
}
