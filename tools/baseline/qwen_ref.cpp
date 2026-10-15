// qwen_ref — greedy llama.cpp reference token streams for the Qwen2 tests.
//
// Dev-time reference only. This tool links against the llama.cpp CPU-only build
// (NOT SonicBoom) and emits, for one prompt, the tokenized prompt ids and the
// first `-n` greedy (argmax) continuation ids. tests/core/test_qwen2_generate.cpp
// replays the same prompts through SonicBoom's CPU spine and diffs the streams;
// the goldens it reads are this tool's output, checked into tests/data/qwen2/.
//
// The prompt is read from a file so the shell never touches it, and tokenized
// without special tokens (the Qwen2 tokenizer sets add_bos_token = false, and
// SonicBoom's own tokenizer inserts none) — that is the exact byte sequence
// SonicBoom's GGUF tokenizer is expected to reproduce.
//
// Greedy decode: the prompt is prefilled in one batch (logits only on its last
// token), then each step decodes one token at position pos with the incremental
// KV cache and takes the argmax over the raw vocab logits — the same greedy loop
// as SonicBoom's prefill spine, so the two streams are directly comparable.
//
// Usage: qwen_ref -m <model.gguf> -f <prompt.txt> [-n steps] [-t threads]
//                  [-ngl n] [--margin] [--topk K]
// Output (stdout):
//   prompt: <id> <id> ...
//   tokens: <id> <id> ...
//   margin: <step>:<gap>(top1=<id> top2=<id>) ...        (with --margin)
//   topk:   <step>|<id>:<gap>,<id>:<gap>, ... ...        (with --topk K)
//
// Build: tools/baseline/build_qwen_ref.sh

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int argmax(const float* logits, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (logits[i] > logits[best]) best = i;
  return best;
}

} // namespace

int main(int argc, char** argv) {
  const char* model_path = nullptr;
  const char* prompt_path = nullptr;
  int steps = 64;
  int n_threads = 0;
  int n_gpu_layers = 0;   // 0 == CPU; >0 offloads that many layers (dev knob used
                          // to obtain a second, numerically independent llama.cpp
                          // reference for the near-tie analysis)
  bool margin = false;    // print the top-1/top-2 logit margin per step
  int topk = 0;           // also print the top-K ids and their gap to the top-1
                          // logit per step (0 = off)

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "-m")) {
      model_path = argv[++i];
    } else if (!strcmp(argv[i], "-f")) {
      prompt_path = argv[++i];
    } else if (!strcmp(argv[i], "-n")) {
      steps = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-t")) {
      n_threads = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-ngl")) {
      n_gpu_layers = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--margin")) {
      margin = true;
    } else if (!strcmp(argv[i], "--topk")) {
      topk = atoi(argv[++i]);
    }
  }
  if (!model_path || !prompt_path || steps <= 0) {
    fprintf(stderr, "usage: %s -m <model.gguf> -f <prompt.txt> [-n steps] [-t threads]\n",
            argv[0]);
    return 2;
  }

  std::ifstream pf(prompt_path, std::ios::binary);
  if (!pf) {
    fprintf(stderr, "cannot open prompt: %s\n", prompt_path);
    return 2;
  }
  std::ostringstream ss;
  ss << pf.rdbuf();
  const std::string prompt = ss.str();

  llama_backend_init();

  llama_model_params mparams = llama_model_default_params();
  mparams.n_gpu_layers = n_gpu_layers;
  llama_model* model = llama_model_load_from_file(model_path, mparams);
  if (!model) {
    fprintf(stderr, "failed to load model: %s\n", model_path);
    return 1;
  }
  const llama_vocab* vocab = llama_model_get_vocab(model);
  const int n_vocab = llama_vocab_n_tokens(vocab);

  // Tokenize without specials; `parse_special` is false too, so a prompt made of
  // plain text is a pure byte-level encode (matching the SonicBoom path).
  std::vector<llama_token> ptoks(prompt.size() + 8);
  int n_p = llama_tokenize(vocab, prompt.data(), int(prompt.size()), ptoks.data(),
                           int(ptoks.size()), /*add_special=*/false,
                           /*parse_special=*/false);
  if (n_p < 0) {
    fprintf(stderr, "tokenize failed (needs %d tokens)\n", -n_p);
    return 1;
  }
  ptoks.resize(n_p);
  if (ptoks.empty()) {
    fprintf(stderr, "empty prompt\n");
    return 1;
  }

  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = uint32_t(n_p + steps + 8);
  cparams.n_batch = uint32_t(n_p);
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

  // Prefill the whole prompt in one batch; logits only on the last token.
  llama_batch  batch = llama_batch_init(int(ptoks.size()) + 1, 0, 1);
  int pos = 0;
  for (int i = 0; i < n_p; ++i) {
    batch.token[i] = ptoks[i];
    batch.pos[i] = i;
    batch.n_seq_id[i] = 1;
    batch.seq_id[i][0] = 0;
    batch.logits[i] = (i == n_p - 1) ? 1 : 0;
  }
  batch.n_tokens = n_p;
  if (llama_decode(ctx, batch) != 0) {
    fprintf(stderr, "prefill decode failed\n");
    return 1;
  }
  pos = n_p;
  const float* last = llama_get_logits_ith(ctx, n_p - 1);
  int cur = argmax(last, n_vocab);

  std::vector<int> out;
  std::vector<std::string> margins;
  std::vector<std::string> topks;
  out.reserve(steps);
  llama_batch one = llama_batch_init(1, 0, 1);
  for (int s = 0; s < steps; ++s) {
    // Top-1/top-2 raw-logit margin of the distribution the step sampled from:
    // a divergent stream whose steps are near ties is a numerics story, not a
    // model bug.
    if (topk > 0) {
      // Top-K ids and their gap to the top-1 logit. A runtime that picks a
      // different token at this step is within that gap of the reference, i.e.
      // its own logit error is at least half of it — so the gap is a direct
      // bound on the divergence, and it is recorded per step as test data.
      std::vector<int> idx(n_vocab);
      for (int i = 0; i < n_vocab; ++i) idx[i] = i;
      const int k = topk < n_vocab ? topk : n_vocab;
      std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                        [&](int a, int b) { return last[a] > last[b]; });
      std::string line = std::to_string(s) + "|";
      for (int j = 0; j < k; ++j) {
        if (j) line += ',';
        char buf[80];
        snprintf(buf, sizeof(buf), "%d:%.4f", idx[j], last[idx[0]] - last[idx[j]]);
        line += buf;
      }
      topks.push_back(line);
    }
    if (margin) {
      int first = 0, second = -1;
      for (int i = 1; i < n_vocab; ++i) {
        if (last[i] > last[first]) {
          second = first;
          first = i;
        } else if (second < 0 || last[i] > last[second]) {
          second = i;
        }
      }
      char buf[128];
      snprintf(buf, sizeof(buf), "%d:%.4f(top1=%d top2=%d)", s,
               second < 0 ? 0.0f : last[first] - last[second], first, second);
      margins.push_back(buf);
    }
    out.push_back(cur);
    one.token[0] = cur;
    one.pos[0] = pos;
    one.n_tokens = 1;
    one.n_seq_id[0] = 1;
    one.seq_id[0][0] = 0;
    one.logits[0] = 1;
    if (llama_decode(ctx, one) != 0) {
      fprintf(stderr, "decode failed at pos %d\n", pos);
      return 1;
    }
    last = llama_get_logits_ith(ctx, 0);
    cur = argmax(last, n_vocab);
    ++pos;
  }

  printf("prompt:");
  for (llama_token t : ptoks) printf(" %d", int(t));
  printf("\n");
  printf("tokens:");
  for (int t : out) printf(" %d", t);
  printf("\n");
  if (margin) {
    printf("margin:");
    for (const std::string& s : margins) printf(" %s", s.c_str());
    printf("\n");
  }
  if (topk > 0) {
    printf("topk:");
    for (const std::string& s : topks) printf(" %s", s.c_str());
    printf("\n");
  }

  llama_batch_free(one);
  llama_batch_free(batch);
  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
