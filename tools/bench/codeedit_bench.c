// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// codeedit_bench: the public code-edit workload through the stable C ABI.
//
// Ten independent edit requests, each pasting one frozen source file into a
// session and generating that file's own token count back, greedy. Every turn
// starts at position 0; the session's KV cache is reused across turns and reps
// the way a server reuses its slot for stateless requests. Speculation is
// selected by the environment (SONICBOOM_SPEC=off|auto), like the rest of the
// runtime, and does not change the emitted tokens.
//
// Usage:
//   codeedit_bench --model GGUF --data DIR --reps N [--ctx-only]
//
//   --ctx-only  print the llama-server -c rule for this data and exit:
//               ctx = max_t(prompt_tokens_t + gen_tokens_t + 1) + 2
//
// Output, one line per turn per rep, then one line per rep:
//   turn=NN rep=R prompt_tokens=P gen_tokens=G decode_ms=T hash=H
//   rep=R tok_s=<sum G / sum T over the ten turns * 1000>
//
// `hash` is the token hash chained over the turns of that rep, starting from
// the same initial value the workload harness uses.

#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include "sonicboom/capi.h"

#define HASH_INIT 1469598103934665603ull
#define HASH_MUL 1099511628211ull

static uint64_t hash_token(uint64_t h, uint64_t token) {
  h ^= token;
  return h * HASH_MUL;
}

static void die(const char *what, const sb_error *err) {
  fprintf(stderr, "codeedit_bench: %s: %s\n", what,
          err ? sb_error_message(err) : "(no error object)");
  exit(1);
}

static char *read_file(const char *path, size_t *len_out) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "codeedit_bench: cannot read %s: %s\n", path,
            strerror(errno));
    return NULL;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  const long end = ftell(f);
  if (end < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  char *buf = (char *)malloc((size_t)end + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  const size_t got = fread(buf, 1, (size_t)end, f);
  fclose(f);
  if (got != (size_t)end) {
    free(buf);
    return NULL;
  }
  buf[got] = '\0';
  *len_out = got;
  return buf;
}

// Tokenize with explicit flags; *out is malloc'd and valid when the call
// succeeds (NULL for an empty encoding).
static int tokenize(const sb_llm_model *model, const char *text, size_t len,
                    int flags, sb_token **out, size_t *n_out) {
  size_t need = 0;
  const sb_status st =
      sb_llm_tokenize_ex(model, text, len, flags, NULL, 0, &need, NULL);
  if (st != SB_ERR_BINDING && st != SB_OK) {
    fprintf(stderr, "codeedit_bench: tokenize sizing failed (%d)\n", (int)st);
    return 0;
  }
  sb_token *tok = need ? (sb_token *)malloc(need * sizeof(sb_token)) : NULL;
  if (need && !tok) return 0;
  if (need) {
    sb_error *err = NULL;
    size_t got = 0;
    if (sb_llm_tokenize_ex(model, text, len, flags, tok, need, &got, &err) !=
        SB_OK)
      die("tokenize", err);
    sb_error_free(err);
    if (got != need) {
      fprintf(stderr, "codeedit_bench: tokenize count moved\n");
      free(tok);
      return 0;
    }
  }
  *out = tok;
  *n_out = need;
  return 1;
}

struct turn {
  char index[4];       // "00" .. "09"
  int add_special;     // only turn 0 carries <bos>
  char *prompt;        // instruction bytes followed by the file bytes
  size_t prompt_len;
  char *file;          // the file bytes alone (the generation length)
  size_t file_len;
  size_t prompt_tokens;
  size_t file_tokens;  // the turn's generation length
};

struct collect {
  sb_token *tokens;
  size_t count;
  size_t cap;
  int overflow;
};

static int on_token(void *user, sb_token token) {
  struct collect *c = (struct collect *)user;
  if (c->count == c->cap) {
    c->overflow = 1;
    return 1;  // abort: the decode produced more than requested
  }
  c->tokens[c->count++] = token;
  return 0;
}

// Measurement aid: write one turn's token ids, for cross-tool identity checks.
// Off unless CODEEDIT_IDS_DIR is set.
static void dump_ids(const char *index, const sb_token *first,
                     const struct collect *out) {
  const char *dir = getenv("CODEEDIT_IDS_DIR");
  if (!dir || !*dir) return;
  mkdir(dir, 0777);  // EEXIST is fine
  char path[4096];
  snprintf(path, sizeof path, "%s/turn%s.ids", dir, index);
  FILE *f = fopen(path, "wb");
  if (!f) return;
  fprintf(f, "%d\n", (int)*first);
  for (size_t i = 0; i < out->count; ++i) fprintf(f, "%d\n", (int)out->tokens[i]);
  fclose(f);
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static void usage(void) {
  fprintf(stderr,
          "usage: codeedit_bench --model GGUF --data DIR --reps N "
          "[--ctx-only]\n");
}

// Tokenize one turn's prompt (instruction bytes followed by the file bytes);
// only turn 0 adds <bos>.
static sb_token *turn_prompt_ids(const sb_llm_model *model,
                                 const struct turn *t, size_t *n_out) {
  sb_token *ids = NULL;
  if (!tokenize(model, t->prompt, t->prompt_len,
                t->add_special ? SB_TOKENIZE_ADD_SPECIAL : 0, &ids, n_out))
    exit(1);
  return ids;
}

static void prefill_ids(sb_llm_session *session, const sb_token *ids, size_t n) {
  sb_error *err = NULL;
  uint64_t reused = 0;
  if (sb_llm_prefill(session, ids, n, &reused, &err) != SB_OK)
    die("prefill", err);
  sb_error_free(err);
}

int main(int argc, char **argv) {
  const char *model_path = NULL;
  const char *data_dir = NULL;
  unsigned long long reps = 0;
  int ctx_only = 0;

  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--model") && i + 1 < argc) {
      model_path = argv[++i];
    } else if (!strcmp(argv[i], "--data") && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (!strcmp(argv[i], "--reps") && i + 1 < argc) {
      reps = strtoull(argv[++i], NULL, 10);
    } else if (!strcmp(argv[i], "--ctx-only")) {
      ctx_only = 1;
    } else {
      usage();
      return 2;
    }
  }
  if (!model_path || !data_dir || (!ctx_only && reps == 0)) {
    usage();
    return 2;
  }

  // Read the frozen requests: NN.instruction, NN.file and manifest.tsv
  // (index, source path, sha256, add_special).
  char manifest_path[4096];
  snprintf(manifest_path, sizeof manifest_path, "%s/manifest.tsv", data_dir);
  FILE *mf = fopen(manifest_path, "rb");
  if (!mf) {
    fprintf(stderr, "codeedit_bench: cannot read %s: %s\n", manifest_path,
            strerror(errno));
    return 2;
  }

  struct turn turns[64];
  size_t n_turns = 0;
  char *line = NULL;
  size_t line_cap = 0;
  while (getline(&line, &line_cap, mf) != -1 && n_turns < 64) {
    char *p = line;
    char *fields[4] = {0};
    for (int f = 0; f < 4; ++f) {
      fields[f] = p;
      char *tab = strchr(p, '\t');
      if (!tab) {
        if (f < 3) fields[f] = NULL;
        break;
      }
      *tab = '\0';
      p = tab + 1;
    }
    if (!fields[0] || !fields[3]) continue;
    char *nl = strchr(fields[3], '\n');
    if (nl) *nl = '\0';

    struct turn *t = &turns[n_turns];
    memset(t, 0, sizeof *t);
    snprintf(t->index, sizeof t->index, "%s", fields[0]);
    t->add_special = atoi(fields[3]) != 0;

    char path[4096];
    snprintf(path, sizeof path, "%s/%s.instruction", data_dir, t->index);
    size_t ins_len = 0;
    char *ins = read_file(path, &ins_len);
    if (!ins) return 2;
    snprintf(path, sizeof path, "%s/%s.file", data_dir, t->index);
    char *file = read_file(path, &t->file_len);
    if (!file) {
      free(ins);
      return 2;
    }
    t->file = file;
    // The prompt is exactly the instruction bytes followed by the file bytes.
    t->prompt_len = ins_len + t->file_len;
    t->prompt = (char *)malloc(t->prompt_len + 1);
    if (!t->prompt) {
      free(ins);
      return 1;
    }
    memcpy(t->prompt, ins, ins_len);
    memcpy(t->prompt + ins_len, t->file, t->file_len);
    t->prompt[t->prompt_len] = '\0';
    free(ins);
    ++n_turns;
  }
  free(line);
  fclose(mf);
  if (n_turns == 0) {
    fprintf(stderr, "codeedit_bench: no turns in %s\n", manifest_path);
    return 2;
  }

  // The model's tokenizer is needed even for --ctx-only.
  sb_error *err = NULL;
  sb_llm_model *model = NULL;
  if (sb_llm_model_load(model_path, &model, &err) != SB_OK)
    die("model load", err);
  sb_error_free(err);
  err = NULL;

  unsigned long long max_need = 0;
  for (size_t i = 0; i < n_turns; ++i) {
    sb_token *ids = NULL;
    size_t n = 0;
    if (!tokenize(model, turns[i].prompt, turns[i].prompt_len,
                  turns[i].add_special ? SB_TOKENIZE_ADD_SPECIAL : 0, &ids,
                  &n))
      return 1;
    free(ids);
    turns[i].prompt_tokens = n;

    // The generation length is the file's own token count (tokenized alone,
    // without <bos>).
    ids = NULL;
    n = 0;
    if (!tokenize(model, turns[i].file, turns[i].file_len, 0, &ids, &n))
      return 1;
    free(ids);
    turns[i].file_tokens = n;

    const unsigned long long need =
        (unsigned long long)n + (unsigned long long)turns[i].prompt_tokens + 1;
    if (need > max_need) max_need = need;
  }
  const unsigned long long n_ctx = max_need + 2;

  if (ctx_only) {
    printf("ctx=%llu\n", n_ctx);
    for (size_t i = 0; i < n_turns; ++i) {
      free(turns[i].prompt);
      free(turns[i].file);
    }
    sb_llm_model_free(model);
    return 0;
  }

  // One session for every turn and rep, allocated and warmed at construction
  // (the --warm-session / --warmup behaviour of the workload harness).
  sb_llm_session *session = NULL;
  if (sb_llm_session_new(model, n_ctx, &session, &err) != SB_OK)
    die("session new", err);
  sb_error_free(err);
  err = NULL;

  sb_llm_sampling greedy;
  memset(&greedy, 0, sizeof greedy);
  greedy.struct_size = sizeof greedy;
  greedy.temperature = 0.0f;
  greedy.top_p = 1.0f;
  greedy.top_k = 0;
  greedy.seed = 1234;
  greedy.max_tokens = 0;  // set per call
  greedy.min_p = 0.0f;
  greedy.repeat_penalty = 1.0f;
  greedy.repeat_last_n = 64;
  greedy.presence_penalty = 0.0f;
  greedy.frequency_penalty = 0.0f;

  sb_token first_buf[1];
  struct collect first = {first_buf, 0, 1, 0};

  // Warm the plan/capture caches on the first turn, generation shortened to
  // 64 tokens (the harness's --warmup warm turn).
  {
    size_t prompt_n = 0;
    sb_token *prompt_ids = turn_prompt_ids(model, &turns[0], &prompt_n);
    prefill_ids(session, prompt_ids, prompt_n);
    first.count = 0;
    first.overflow = 0;
    greedy.max_tokens = 1;
    uint32_t made = 0;
    if (sb_llm_decode(session, &greedy, on_token, &first, &made, &err) != SB_OK)
      die("warm decode (first)", err);
    sb_error_free(err);
    err = NULL;

    // The second call prefills the prompt with the carried token appended, the
    // way the workload harness's turn is structured.
    sb_token *extended = (sb_token *)malloc((prompt_n + 1) * sizeof(sb_token));
    if (!extended) return 1;
    memcpy(extended, prompt_ids, prompt_n * sizeof(sb_token));
    extended[prompt_n] = first.tokens[0];
    prefill_ids(session, extended, prompt_n + 1);
    free(extended);
    free(prompt_ids);

    const size_t warm_gen =
        turns[0].file_tokens < 64 ? turns[0].file_tokens : 64;
    sb_token *warm = (sb_token *)malloc((warm_gen ? warm_gen : 1) *
                                        sizeof(sb_token));
    if (!warm) return 1;
    struct collect warm_c = {warm, 0, warm_gen, 0};
    greedy.max_tokens = (uint32_t)warm_gen;
    made = 0;
    if (sb_llm_decode(session, &greedy, on_token, &warm_c, &made, &err) != SB_OK)
      die("warm decode", err);
    sb_error_free(err);
    err = NULL;
    free(warm);
  }

  // One output buffer per turn, reused across turns.
  size_t max_gen = 0;
  for (size_t i = 0; i < n_turns; ++i)
    if (turns[i].file_tokens > max_gen) max_gen = turns[i].file_tokens;
  sb_token *decoded =
      (sb_token *)malloc((max_gen ? max_gen : 1) * sizeof(sb_token));
  if (!decoded) return 1;
  struct collect out = {decoded, 0, max_gen, 0};

  for (unsigned long long rep = 0; rep < reps; ++rep) {
    uint64_t hash = HASH_INIT;
    unsigned long long gen_total = 0;
    double ms_total = 0.0;

    for (size_t t = 0; t < n_turns; ++t) {
      size_t prompt_n = 0;
      sb_token *prompt_ids = turn_prompt_ids(model, &turns[t], &prompt_n);
      prefill_ids(session, prompt_ids, prompt_n);

      // The prefill's carried token is the turn's first generated token.
      first.count = 0;
      first.overflow = 0;
      greedy.max_tokens = 1;
      uint32_t made = 0;
      if (sb_llm_decode(session, &greedy, on_token, &first, &made, &err) !=
          SB_OK)
        die("decode (first)", err);
      sb_error_free(err);
      err = NULL;
      if (made != 1 || first.overflow) {
        fprintf(stderr, "codeedit_bench: turn %s produced no first token\n",
                turns[t].index);
        return 1;
      }
      hash = hash_token(hash, (uint64_t)first.tokens[0]);

      // The decode phase proper: prefill the prompt with the carried token
      // appended (the harness's second call), then generate the file's own
      // token count from it.
      sb_token *extended = (sb_token *)malloc((prompt_n + 1) * sizeof(sb_token));
      if (!extended) return 1;
      memcpy(extended, prompt_ids, prompt_n * sizeof(sb_token));
      extended[prompt_n] = first.tokens[0];
      free(prompt_ids);

      out.count = 0;
      out.overflow = 0;
      greedy.max_tokens = (uint32_t)turns[t].file_tokens;
      made = 0;
      const double t0 = now_ms();
      prefill_ids(session, extended, prompt_n + 1);
      const sb_status dec =
          sb_llm_decode(session, &greedy, on_token, &out, &made, &err);
      const double decode_ms = now_ms() - t0;
      free(extended);
      if (dec != SB_OK) die("decode", err);
      sb_error_free(err);
      err = NULL;
      if (out.overflow || made != turns[t].file_tokens) {
        fprintf(stderr, "codeedit_bench: turn %s produced %u of %zu tokens\n",
                turns[t].index, made, turns[t].file_tokens);
        return 1;
      }
      for (size_t i = 0; i < out.count; ++i)
        hash = hash_token(hash, (uint64_t)out.tokens[i]);
      dump_ids(turns[t].index, &first.tokens[0], &out);

      printf("turn=%s rep=%llu prompt_tokens=%llu gen_tokens=%llu "
             "decode_ms=%.3f hash=%016llx\n",
             turns[t].index, rep,
             (unsigned long long)turns[t].prompt_tokens,
             (unsigned long long)turns[t].file_tokens, decode_ms,
             (unsigned long long)hash);
      gen_total += turns[t].file_tokens;
      ms_total += decode_ms;
    }
    printf("rep=%llu tok_s=%.3f\n", rep,
           ms_total > 0.0 ? (double)gen_total * 1000.0 / ms_total : 0.0);
  }

  free(decoded);
  sb_llm_session_free(session);
  sb_llm_model_free(model);
  for (size_t i = 0; i < n_turns; ++i) {
    free(turns[i].prompt);
    free(turns[i].file);
  }
  return 0;
}
