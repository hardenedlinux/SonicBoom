// llama_bench — measure llama.cpp CPU decode throughput on the Gemma 4 model.
//
// Dev-time reference only. This tool links against the llama.cpp CPU-only build
// (NOT SonicBoom) and reports greedy-decode tokens/sec for direct comparison
// against SonicBoom's model::generate() (tests/core/test_generate.cpp). It is
// NOT part of libsonicboom.so and adds no llama.cpp dependency to the runtime.
//
// Greedy decode loop: for each step, decode the current token at position pos
// (incremental KV cache, so real causal attention over all prior tokens), then
// argmax over the raw vocab logits. Same shape as test_generate: a seed vocab
// id decoded for `n` steps from position 0.
//
// Usage: llama_bench -m <model.gguf> [-n steps] [--token id] [--ctx n]
//
// Build: tools/oracle/build_llama_bench.sh

#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// Greedy argmax over `n` logits (ties toward the lowest index).
int argmax(const float* logits, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (logits[i] > logits[best]) best = i;
  return best;
}

} // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  int steps = 4;
  int token = 2;
  int n_ctx = 512;
  int n_threads = 0;   // 0 == llama.cpp default (4 threads)
  int n_gpu_layers = 0;  // 0 == CPU only; -1/999 == all layers offloaded

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-m")) {
      model_path = argv[++i];
    } else if (!strcmp(argv[i], "-n")) {
      steps = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--token")) {
      token = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--ctx")) {
      n_ctx = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-t")) {
      n_threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-ngl")) {
      n_gpu_layers = atoi(argv[++i]);
    }
  }
  if (!model_path) {
    fprintf(stderr,
            "usage: %s -m <model.gguf> [-n steps] [--token id] [--ctx n] [-t threads] [-ngl n]\n",
            argv[0]);
    return 2;
  }
  if (steps <= 0 || token < 0 || n_ctx < 1 || n_threads < 0) {
    fprintf(stderr, "invalid args: steps=%d token=%d ctx=%d threads=%d\n", steps,
            token, n_ctx, n_threads);
    return 2;
  }

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = n_gpu_layers;
  llama_model* model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    fprintf(stderr, "failed to load model: %s\n", model_path);
    return 1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = n_ctx;
  if (n_threads > 0) {
    cparams.n_threads = n_threads;
    cparams.n_threads_batch = n_threads;
  }
  llama_context* ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    fprintf(stderr, "failed to create context\n");
    llama_model_free(model);
    return 1;
  }

  const int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  llama_batch batch = llama_batch_init(1, 0, 1);
  batch.n_tokens = 1;
  batch.n_seq_id[0] = 1;
  batch.seq_id[0][0] = 0;
  batch.logits[0] = 1;

  std::vector<int> tokens;
  tokens.reserve(steps);

  const auto t0 = std::chrono::steady_clock::now();
  int cur = token;
  int pos = 0;
  for (int s = 0; s < steps; ++s) {
    batch.token[0] = cur;
    batch.pos[0] = pos;

    if (llama_decode(ctx, batch) != 0) {
      fprintf(stderr, "decode failed at step %d (pos %d)\n", s, pos);
      break;
    }

    const float* logits = llama_get_logits_ith(ctx, 0);
    cur = argmax(logits, vocab);
    tokens.push_back(cur);
    ++pos;
  }
  const auto t1 = std::chrono::steady_clock::now();

  const double secs = std::chrono::duration<double>(t1 - t0).count();
  const double tok_per_sec = secs > 0.0 ? double(tokens.size()) / secs : 0.0;

  printf("llama.cpp decode: %zu tokens in %.3fs = %.3f tok/s (seed %d, %d threads, %d gpu layers)\n",
         tokens.size(), secs, tok_per_sec, token, llama_n_threads(ctx), n_gpu_layers);
  printf("tokens:");
  for (int t : tokens) printf(" %d", t);
  printf("\n");

  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
