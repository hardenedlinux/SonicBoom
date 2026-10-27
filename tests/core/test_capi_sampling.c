// C ABI sampling tests (qwen2.5-0.5b, CUDA).
//
//  1. the same (temperature, top_p, seed) twice produces identical tokens;
//  2. a different seed diverges within 64 tokens;
//  3. prefix reuse: a prefill of prompt + the first 16 sampled tokens followed
//     by 16 more tokens reproduces tokens 17..32 of the first run;
//  4. temperature 0 is still the historical greedy stream (the p1 golden);
//  5. constrained decoding + sampling produces text the grammar accepts;
//  6. a zero-initialized struct (struct_size + max_tokens only) is greedy:
//     repeat_penalty 0 means off and repeat_last_n 0 disables penalties;
//  7. sampled decode runs on the session path: on a 2048-token prompt its
//     time-to-first-token (prefill + the first callback) stays within 2x the
//     greedy path's, i.e. the prompt is never re-run token by token;
//  8. a second turn on the same session reuses the prefix and the same
//     (seed, position, history) reproduces the same stream;
//  9. grammar + sampling on CUDA: the output is accepted by the grammar.
//
// Skips (exit 77) when the model, the golden or a CUDA device is absent.

#include <sonicboom/capi.h>

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_failures = 0;

static void check(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

static void die(const char *what, sb_error *err) {
  fprintf(stderr, "%s: %s\n", what, err ? sb_error_message(err) : "(none)");
  if (err) sb_error_free(err);
  exit(1);
}

static void check_status(sb_status st, sb_error *err, const char *what) {
  if (st != SB_OK) die(what, err);
}

#define MAX_TOK 64

typedef struct {
  sb_token tokens[MAX_TOK];
  uint32_t n;
  char text[4096];
} run_result;

static sb_llm_sampling sampling(float temperature, float top_p, uint32_t top_k,
                                uint64_t seed, uint32_t max_tokens) {
  sb_llm_sampling s;
  memset(&s, 0, sizeof s);
  s.struct_size = sizeof s;
  s.temperature = temperature;
  s.top_p = top_p;
  s.top_k = top_k;
  s.seed = seed;
  s.max_tokens = max_tokens;
  s.min_p = 0.0f;
  s.repeat_penalty = 1.0f;
  s.repeat_last_n = 64;
  s.presence_penalty = 0.0f;
  s.frequency_penalty = 0.0f;
  return s;
}

static int token_cb(void *user, sb_token token) {
  run_result *r = (run_result *)user;
  if (r->n < MAX_TOK) r->tokens[r->n++] = token;
  return 0;
}

// Wall clock for the time-to-first-token comparison.
static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

typedef struct {
  double first_ms;  // when the first token arrived
  int n;
} first_token_state;

static int first_token_cb(void *user, sb_token token) {
  (void)token;
  first_token_state *f = (first_token_state *)user;
  if (f->n++ == 0) f->first_ms = now_ms();
  return 0;
}

// Time from the start of the prefill to the first decode callback on a fresh
// session: the whole time-to-first-token of one turn, so the comparison covers
// the same work on both the greedy and the sampled path.
static double ttft_ms(sb_llm_model *model, const sb_token *prompt,
                      size_t n_prompt, uint64_t n_ctx, sb_llm_sampling s,
                      double *prefill_ms, double *decode_ms) {
  sb_error *err = NULL;
  sb_llm_session *session = NULL;
  check_status(sb_llm_session_new(model, n_ctx, &session, &err), err,
               "session_new (ttft)");
  const double t0 = now_ms();
  uint64_t reused = 0;
  check_status(sb_llm_prefill(session, prompt, n_prompt, &reused, &err), err,
               "prefill (ttft)");
  const double t1 = now_ms();
  first_token_state f = {0.0, 0};
  uint32_t n = 0;
  check_status(sb_llm_decode(session, &s, first_token_cb, &f, &n, &err), err,
               "decode (ttft)");
  check(f.n >= 1, "(7) decode produced a first token");
  sb_llm_session_free(session);
  if (prefill_ms) *prefill_ms = t1 - t0;
  if (decode_ms) *decode_ms = (f.n ? f.first_ms - t1 : 0.0);
  return f.n ? f.first_ms - t0 : 1e12;
}

// Run one generation: a fresh session, prefill `prompt`, decode `max_tokens`.
static run_result run_plain(sb_llm_model *model, const sb_token *prompt,
                            size_t n_prompt, sb_llm_sampling s) {
  run_result r;
  memset(&r, 0, sizeof r);
  sb_error *err = NULL;
  sb_llm_session *session = NULL;
  check_status(sb_llm_session_new(model, 256, &session, &err), err,
               "session_new");
  uint64_t reused = 0;
  check_status(sb_llm_prefill(session, prompt, n_prompt, &reused, &err), err,
               "prefill");
  uint32_t n = 0;
  sb_status st = sb_llm_decode(session, &s, token_cb, &r, &n, &err);
  if (st != SB_OK) die("decode", err);
  size_t len = 0;
  if (r.n) sb_llm_detokenize(model, r.tokens, r.n, r.text, sizeof r.text, &len, &err);
  sb_llm_session_free(session);
  return r;
}

#ifndef SONICBOOM_PRIVATE_DATA_DIR
#define SONICBOOM_PRIVATE_DATA_DIR ""
#endif

// Read the greedy golden's `cuda` token line from p1.standard.golden. The
// golden is test data and lives with the private suites; a public-only build
// has none and the caller skips.
static int read_golden(int64_t *out, size_t cap, size_t *n_out) {
  if (SONICBOOM_PRIVATE_DATA_DIR[0] == '\0') return 0;
  char path[4096];
  snprintf(path, sizeof path, "%s/qwen2/p1.standard.golden",
           SONICBOOM_PRIVATE_DATA_DIR);
  FILE *f = fopen(path, "r");
  if (!f) return 0;
  char line[8192];
  int found = 0;
  while (fgets(line, sizeof line, f)) {
    if (strncmp(line, "cuda ", 5) != 0) continue;
    size_t n = 0;
    char *p = line + 5;
    while (*p && n < cap) {
      char *end = NULL;
      long v = strtol(p, &end, 10);
      if (end == p) break;
      out[n++] = v;
      p = end;
    }
    *n_out = n;
    found = 1;
    break;
  }
  fclose(f);
  return found;
}

#ifndef SONICBOOM_SRC_DIR
#define SONICBOOM_SRC_DIR "."
#endif

int main(void) {
  // Pin the numerics-bearing choices the private qwen2 goldens were recorded
  // under (the same pins test_qwen2_generate sets).
  setenv("SONICBOOM_MMQ_DECOMP", "tiled", 0);
  setenv("SONICBOOM_ATTN_DECODE_CHUNK", "256", 0);
  const char *root = SONICBOOM_SRC_DIR;
  char model_path[4096];
  snprintf(model_path, sizeof model_path, "%s/models/%s", root,
           "qwen2.5-0.5b-instruct-q4_k_m.gguf");

  FILE *mf = fopen(model_path, "rb");
  if (!mf) {
    fprintf(stderr, "SKIP: %s absent\n", model_path);
    return 77;
  }
  fclose(mf);

  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    fprintf(stderr, "SKIP: no CUDA device\n");
    return 77;
  }

  int64_t golden[256];
  size_t n_golden = 0;
  if (!read_golden(golden, 256, &n_golden) || n_golden < 32) {
    fprintf(stderr, "SKIP: golden absent\n");
    return 77;
  }

  sb_error *err = NULL;
  sb_llm_model *model = NULL;
  check_status(sb_llm_model_load(model_path, &model, &err), err,
               "model_load");

  const char *text = "The capital of France is";
  sb_token prompt[256];
  size_t n_prompt = 0;
  check_status(sb_llm_tokenize(model, text, strlen(text), prompt, 256,
                               &n_prompt, &err),
               err, "tokenize");

  // (1) the same sampling twice is identical.
  const sb_llm_sampling s7 = sampling(0.8f, 0.9f, 0, 7, 32);
  run_result a = run_plain(model, prompt, n_prompt, s7);
  run_result b = run_plain(model, prompt, n_prompt, s7);
  check(a.n == 32 && b.n == a.n, "(1) produced 32 tokens");
  check(a.n == b.n && memcmp(a.tokens, b.tokens, a.n * sizeof(sb_token)) == 0,
        "(1) same seed -> identical tokens");

  // (2) a different seed differs within 64 tokens.
  run_result c = run_plain(model, prompt, n_prompt, sampling(0.8f, 0.9f, 0, 8, 64));
  run_result d = run_plain(model, prompt, n_prompt, sampling(0.8f, 0.9f, 0, 7, 64));
  int differs = 0;
  for (uint32_t i = 0; i < c.n && i < d.n; ++i)
    if (c.tokens[i] != d.tokens[i]) differs = 1;
  check(differs, "(2) a different seed changes the stream within 64 tokens");

  // (3) prefix reuse reproduces tokens 17..32.
  {
    sb_llm_session *session = NULL;
    check_status(sb_llm_session_new(model, 256, &session, &err), err,
                 "session_new (reuse)");
    uint64_t reused = 0;
    check_status(sb_llm_prefill(session, prompt, n_prompt, &reused, &err), err,
                 "prefill (reuse 1)");
    run_result r1;
    memset(&r1, 0, sizeof r1);
    uint32_t n = 0;
    check_status(sb_llm_decode(session, &s7, token_cb, &r1, &n, &err), err,
                 "decode (reuse 1)");
    check(r1.n >= 32, "(3) first run produced at least 32 tokens");

    // New prefill: prompt + the first 16 generated tokens (prefix reuse).
    sb_token extended[256];
    memcpy(extended, prompt, n_prompt * sizeof(sb_token));
    memcpy(extended + n_prompt, r1.tokens, 16 * sizeof(sb_token));
    check_status(sb_llm_prefill(session, extended, n_prompt + 16, &reused, &err),
                 err, "prefill (reuse 2)");
    run_result r2;
    memset(&r2, 0, sizeof r2);
    check_status(sb_llm_decode(session, &s7, token_cb, &r2, &n, &err), err,
                 "decode (reuse 2)");
    check(r2.n >= 16, "(3) second run produced at least 16 tokens");
    int same = r1.n >= 32 && r2.n >= 16;
    for (int i = 0; same && i < 16; ++i)
      same = r1.tokens[16 + i] == r2.tokens[i];
    check(same, "(3) tokens 17..32 are identical after prefix reuse");
    sb_llm_session_free(session);
  }

  // (4) temperature 0 is the historical greedy stream (the p1 golden).
  {
    run_result g = run_plain(model, prompt, n_prompt, sampling(0.0f, 1.0f, 0, 0, 32));
    check(g.n == 32, "(4) greedy produced 32 tokens");
    int same = g.n == 32;
    for (int i = 0; same && i < 32; ++i)
      same = (int64_t)g.tokens[i] == golden[i];
    if (!same) {
      for (int i = 0; i < 32; ++i)
        fprintf(stderr, "    [%d] got %lld want %lld%s\n", i,
                (long long)g.tokens[i], (long long)golden[i],
                (int64_t)g.tokens[i] == golden[i] ? "" : "  <--");
    }
    check(same, "(4) temperature 0 == the p1 greedy golden");
  }

  // (5) grammar + sampling: the output must be accepted by the grammar.
  {
    const char *gbnf = "root ::= \"yes\" | \"no\"";
    sb_llm_session *session = NULL;
    check_status(sb_llm_session_new(model, 256, &session, &err), err,
                 "session_new (grammar)");
    check_status(sb_llm_session_set_grammar(session, gbnf, &err), err,
                 "set_grammar");
    uint64_t reused = 0;
    check_status(sb_llm_prefill(session, prompt, n_prompt, &reused, &err), err,
                 "prefill (grammar)");
    run_result g;
    memset(&g, 0, sizeof g);
    uint32_t n = 0;
    check_status(sb_llm_decode(session, &s7, token_cb, &g, &n, &err), err,
                 "decode (grammar)");
    char out[128] = {0};
    sb_error *de = NULL;
    size_t len = 0;
    sb_status dst = sb_llm_detokenize(model, g.tokens, g.n, out, sizeof out, &len, &de);
    fprintf(stderr, "    grammar tokens:");
    for (uint32_t i = 0; i < g.n; ++i) fprintf(stderr, " %d", g.tokens[i]);
    fprintf(stderr, " (detok st=%d msg=%s)\n", (int)dst, de ? sb_error_message(de) : "");
    if (de) sb_error_free(de);
    if (!(strcmp(out, "yes") == 0 || strcmp(out, "no") == 0))
      fprintf(stderr, "    grammar output: [%s] (%u tokens)\n", out, g.n);
    check(strcmp(out, "yes") == 0 || strcmp(out, "no") == 0,
          "(5) grammar-constrained sampled text is accepted");
    sb_llm_session_free(session);
  }

  // (6) zero-initialized fields mean "off": only struct_size and max_tokens
  // are set, so repeat_penalty 0 and repeat_last_n 0 must behave as plain
  // greedy (and must not be rejected).
  {
    sb_llm_sampling z;
    memset(&z, 0, sizeof z);
    z.struct_size = sizeof z;
    z.max_tokens = 32;
    run_result g = run_plain(model, prompt, n_prompt, z);
    check(g.n == 32, "(6) zero-initialized struct produced 32 tokens");
    int same = g.n == 32;
    for (int i = 0; same && i < 32; ++i)
      same = (int64_t)g.tokens[i] == golden[i];
    check(same, "(6) zero-initialized struct == the p1 greedy golden");
  }

  // (7) sampled decode reuses the session path: on a 2048-token prompt the
  // sampled turn's time-to-first-token (prefill + first callback) stays within
  // 2x the greedy turn's. The old sampled path re-ran the whole prompt token by
  // token through the decode graph inside sb_llm_decode, so it could not pass.
  {
    const size_t n_big = 2048;
    sb_token *big = (sb_token *)malloc(n_big * sizeof(sb_token));
    check(big != NULL, "(7) prompt allocation");
    if (big) {
      for (size_t i = 0; i < n_big; ++i)
        big[i] = prompt[i % n_prompt];
      double g_pf = 0.0, g_dec = 0.0, s_pf = 0.0, s_dec = 0.0;
      sb_llm_sampling greedy;
      memset(&greedy, 0, sizeof greedy);
      greedy.struct_size = sizeof greedy;
      greedy.max_tokens = 1;
      const double g_ms = ttft_ms(model, big, n_big, 2304, greedy, &g_pf, &g_dec);
      const double s_ms = ttft_ms(model, big, n_big, 2304,
                                  sampling(0.8f, 0.9f, 0, 7, 1), &s_pf, &s_dec);
      fprintf(stderr,
              "    ttft 2048: greedy %.1f ms (prefill %.1f, decode %.1f), "
              "sampled %.1f ms (prefill %.1f, decode %.1f)\n",
              g_ms, g_pf, g_dec, s_ms, s_pf, s_dec);
      check(s_ms < 2.0 * g_ms,
            "(7) sampled TTFT on a 2048-token prompt is within 2x greedy");
      free(big);
    }
  }

  // (8) a second turn on the same session reuses the prefix (the cache holds
  // the first prompt) and the same (seed, position, history) reproduces the
  // same stream when the turn is run twice.
  {
    sb_llm_session *session = NULL;
    check_status(sb_llm_session_new(model, 512, &session, &err), err,
                 "session_new (turn 2)");
    uint64_t reused = 0;
    check_status(sb_llm_prefill(session, prompt, n_prompt, &reused, &err), err,
                 "prefill (turn 1)");
    run_result r1;
    memset(&r1, 0, sizeof r1);
    uint32_t n = 0;
    check_status(sb_llm_decode(session, &s7, token_cb, &r1, &n, &err), err,
                 "decode (turn 1)");
    check(r1.n >= 32, "(8) turn 1 produced 32 tokens");

    sb_token ext[512];
    memcpy(ext, prompt, n_prompt * sizeof(sb_token));
    memcpy(ext + n_prompt, r1.tokens, 16 * sizeof(sb_token));
    const size_t n_ext = n_prompt + 16;
    run_result r2a, r2b;
    uint64_t reused_a = 0, reused_b = 0;
    for (int pass = 0; pass < 2; ++pass) {
      uint64_t *reused_p = pass == 0 ? &reused_a : &reused_b;
      run_result *out = pass == 0 ? &r2a : &r2b;
      memset(out, 0, sizeof *out);
      check_status(sb_llm_prefill(session, ext, n_ext, reused_p, &err), err,
                   pass == 0 ? "prefill (turn 2a)" : "prefill (turn 2b)");
      check_status(sb_llm_decode(session, &s7, token_cb, out, &n, &err), err,
                   pass == 0 ? "decode (turn 2a)" : "decode (turn 2b)");
    }
    fprintf(stderr, "    turn 2: reused %llu and %llu of a %zu-token prompt\n",
            (unsigned long long)reused_a, (unsigned long long)reused_b, n_ext);
    check(reused_a >= n_prompt,
          "(8) turn 2 reuses at least the previous prompt's prefix");
    check(reused_b >= n_prompt,
          "(8) a re-run of turn 2 still reuses the previous prompt's prefix");
    check(r2a.n >= 16 && r2b.n == r2a.n &&
              memcmp(r2a.tokens, r2b.tokens, r2a.n * sizeof(sb_token)) == 0,
          "(8) running turn 2 twice reproduces the same tokens");
    sb_llm_session_free(session);
  }

  // (9) grammar + sampling on CUDA: the constrained stream stays inside the
  // grammar (the device SampleSelect applies the same allowed-token bitmap the
  // greedy Argmax node applies).
  {
    const char *gbnf = "root ::= \"yes\" | \"no\" | \"maybe\"";
    sb_llm_session *session = NULL;
    check_status(sb_llm_session_new(model, 256, &session, &err), err,
                 "session_new (grammar 9)");
    check_status(sb_llm_session_set_grammar(session, gbnf, &err), err,
                 "set_grammar (9)");
    uint64_t reused = 0;
    check_status(sb_llm_prefill(session, prompt, n_prompt, &reused, &err), err,
                 "prefill (grammar 9)");
    run_result g;
    memset(&g, 0, sizeof g);
    uint32_t n = 0;
    check_status(sb_llm_decode(session, &s7, token_cb, &g, &n, &err), err,
                 "decode (grammar 9)");
    char out[128] = {0};
    sb_error *de = NULL;
    size_t len = 0;
    sb_status dst = sb_llm_detokenize(model, g.tokens, g.n, out, sizeof out,
                                      &len, &de);
    if (de) sb_error_free(de);
    check(dst == SB_OK && g.n >= 1,
          "(9) grammar-constrained sampled CUDA output detokenizes");
    check(strcmp(out, "yes") == 0 || strcmp(out, "no") == 0 ||
              strcmp(out, "maybe") == 0,
          "(9) grammar-constrained sampled CUDA text is in the grammar");
    sb_llm_session_free(session);
  }

  sb_llm_model_free(model);
  if (g_failures) {
    fprintf(stderr, "%d C API sampling test(s) failed\n", g_failures);
    return 1;
  }
  printf("capi sampling tests passed\n");
  return 0;
}
