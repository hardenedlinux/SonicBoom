/* SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/* test_capi.c — Stage D: the stable C ABI, exercised from real C.
 *
 * This file is compiled as C (not C++) to prove that <sonicboom/capi.h> is a
 * genuine C header and that libsonicboom.so exports its entry points with C
 * linkage. It drives the full parse → compile → bind → execute → retrieve →
 * release flow over the Add+Relu graph (no weight sidecar needed) and checks a
 * handful of error paths.
 */

#include <sonicboom/capi.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures = 0;

static void check(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

/* The value of a top-level numeric field ("key": 123) in a model-info
 * document; 0 when the field is absent. The tests only read fields whose
 * values are non-negative integers. */
static unsigned long long json_number(const char *json, const char *key) {
  char pattern[64];
  snprintf(pattern, sizeof pattern, "\"%s\": ", key);
  const char *p = strstr(json, pattern);
  return p ? strtoull(p + strlen(pattern), NULL, 10) : 0;
}

/* The sum of every "<key>": 123 in the document: the weights of the quant
 * entries add up to the model's parameter count. */
static unsigned long long json_sum(const char *json, const char *key) {
  char pattern[64];
  snprintf(pattern, sizeof pattern, "\"%s\": ", key);
  const size_t plen = strlen(pattern);
  unsigned long long sum = 0;
  for (const char *p = json; (p = strstr(p, pattern)) != NULL; p += plen)
    sum += strtoull(p + plen, NULL, 10);
  return sum;
}

/* The value of a top-level string field ("key": "value"); empty when absent. */
static void json_string(const char *json, const char *key, char *out,
                        size_t cap) {
  char pattern[64];
  snprintf(pattern, sizeof pattern, "\"%s\": \"", key);
  const char *p = strstr(json, pattern);
  out[0] = '\0';
  if (p == NULL) return;
  p += strlen(pattern);
  const char *e = strchr(p, '"');
  size_t n = e ? (size_t)(e - p) : 0;
  if (n >= cap) n = cap - 1;
  memcpy(out, p, n);
  out[n] = '\0';
}

/* z = relu(x + b), single f32 input/output, shape [8]. */
static const char *ADD_RELU =
    "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu\")"
    " (inputs (input \"x\" (tensor float32 (shape 8))))"
    " (outputs (output \"z\"))"
    " (parameters (parameter \"b\" (tensor float32 (shape 8))"
    "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
    " (nodes"
    "   (node add (inputs \"x\" \"b\")"
    "     (outputs (\"y\" (tensor float32 (shape 8)))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 8))))))))";

static void test_end_to_end(void) {
  boom_document *doc = NULL;
  boom_executable *exe = NULL;
  boom_error *err = NULL;

  if (boom_parse(ADD_RELU, strlen(ADD_RELU), &doc, &err) != BOOM_OK) {
    fprintf(stderr, "  FAIL: parse (err: %s)\n", boom_error_message(err));
    ++g_failures;
    boom_error_free(err);
    return;
  }
  check(doc != NULL, "parse: document handle non-NULL");
  check(err == NULL, "parse: no error handle on success");

  if (boom_compile(doc, "", &exe, &err) != BOOM_OK) {
    fprintf(stderr, "  FAIL: compile (err: %s)\n", boom_error_message(err));
    ++g_failures;
    boom_error_free(err);
    boom_document_free(doc);
    return;
  }
  check(exe != NULL, "compile: executable handle non-NULL");

  /* Introspect input/output. */
  boom_dtype dtype = BOOM_DTYPE_FLOAT32;
  int64_t rank = 0;
  const int64_t *dims = NULL;
  uint64_t num_bytes = 0;
  check(boom_input_info(exe, &dtype, &rank, &dims, &num_bytes) == BOOM_OK,
        "input_info: success");
  check(dtype == BOOM_DTYPE_FLOAT32 && rank == 1 && dims && dims[0] == 8 &&
            num_bytes == 32,
        "input_info: f32[8], 32 bytes");
  check(boom_output_info(exe, &dtype, &rank, &dims, &num_bytes) == BOOM_OK,
        "output_info: success");
  check(dtype == BOOM_DTYPE_FLOAT32 && rank == 1 && dims && dims[0] == 8 &&
            num_bytes == 32,
        "output_info: f32[8], 32 bytes");

  /* Bind, execute, retrieve. */
  const float x[8] = {1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
  check(boom_bind_input(exe, x, sizeof(x), &err) == BOOM_OK, "bind_input: success");

  check(boom_execute(exe, &err) == BOOM_OK, "execute: success");

  float got[8];
  size_t written = 0;
  check(boom_retrieve_output(exe, got, sizeof(got), &written, &err) == BOOM_OK,
        "retrieve_output: success");
  check(written == 32, "retrieve_output: wrote 32 bytes");

  /* x + b = [1.5, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
   * relu  = [1.5, 1.0,  0.0, 1.0,  0.0, 1.0,  0.0, 1.0] */
  const float expected[8] = {1.5f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  for (int i = 0; i < 8; ++i) {
    if (fabsf(got[i] - expected[i]) > 1e-5f) {
      fprintf(stderr, "  FAIL: output elem %d (got %f, expected %f)\n", i,
              got[i], expected[i]);
      ++g_failures;
    }
  }

  boom_executable_free(exe);
  boom_document_free(doc);
}

static void test_binding_error(void) {
  boom_document *doc = NULL;
  boom_executable *exe = NULL;
  boom_error *err = NULL;

  if (boom_parse(ADD_RELU, strlen(ADD_RELU), &doc, &err) != BOOM_OK ||
      boom_compile(doc, "", &exe, &err) != BOOM_OK) {
    fprintf(stderr, "  FAIL: binding-error setup (err: %s)\n",
            boom_error_message(err));
    ++g_failures;
    boom_error_free(err);
    return;
  }

  /* Wrong input size → BOOM_ERR_BINDING. */
  float small[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  check(boom_bind_input(exe, small, sizeof(small), &err) == BOOM_ERR_BINDING,
        "bind_input: wrong size rejected");
  check(err != NULL && boom_error_get_kind(err) == BOOM_ERRKIND_BINDING,
        "bind_input: error kind is BINDING");
  boom_error_free(err);
  err = NULL;

  /* Execute before binding → BOOM_ERR_BINDING. */
  boom_executable *exe2 = NULL;
  if (boom_compile(doc, "", &exe2, &err) != BOOM_OK) {
    fprintf(stderr, "  FAIL: second compile (err: %s)\n", boom_error_message(err));
    ++g_failures;
    boom_error_free(err);
    boom_executable_free(exe);
    boom_document_free(doc);
    return;
  }
  check(boom_execute(exe2, &err) == BOOM_ERR_BINDING,
        "execute: unbound input rejected");
  boom_error_free(err);

  boom_executable_free(exe2);
  boom_executable_free(exe);
  boom_document_free(doc);
}

static void test_parse_error(void) {
  boom_document *doc = NULL;
  boom_error *err = NULL;

  /* Malformed s-expression → BOOM_ERR_PARSE. */
  const char *bad = "(sonicboom-s-expr (version 0 1) (graph";
  check(boom_parse(bad, strlen(bad), &doc, &err) == BOOM_ERR_PARSE,
        "parse: malformed text rejected");
  check(doc == NULL, "parse: no document on failure");
  check(err != NULL && boom_error_message(err)[0] != '\0',
        "parse: error message non-empty");
  boom_error_free(err);
}

/* Error-output slot reuse: two failures into the SAME err slot without freeing
 * in between. The implementation frees the prior handle before writing the new
 * one, so the slot must reflect the second error (not crash, not retain a stale
 * handle). A leak in this path is only visible under ASan; this test proves the
 * reuse discipline is safe and that the slot is genuinely replaced. */
static void test_error_slot_reuse(void) {
  boom_document *doc = NULL;
  boom_error *err = NULL;

  /* First failure: unterminated string literal → lexical error. */
  const char *bad1 = "\"unterminated";
  check(boom_parse(bad1, strlen(bad1), &doc, &err) == BOOM_ERR_PARSE,
        "reuse: first parse rejected");
  check(err != NULL && boom_error_get_kind(err) == BOOM_ERRKIND_PARSE_LEXICAL,
        "reuse: first error kind is LEXICAL");

  /* Second failure, same slot, no boom_error_free between the calls. */
  const char *bad2 = "(sonicboom-s-expr)";
  check(boom_parse(bad2, strlen(bad2), &doc, &err) == BOOM_ERR_PARSE,
        "reuse: second parse rejected");
  check(err != NULL && boom_error_get_kind(err) == BOOM_ERRKIND_PARSE_SYNTAX,
        "reuse: slot replaced with second error kind (SYNTAX)");

  boom_error_free(err);
}

/* A prompt the session already holds in full must not fail: boom_llm_prefill
 * re-prefills the last token (its activation is what produces the next-token
 * logits) and reports n-1 reused tokens. Needs the Gemma 4 GGUF, so it skips
 * cleanly when SONICBOOM_GEMMA_GGUF is unset, like the model-backed tests. */
static void test_llm_prefill_full_reuse(void) {
  const char *path = getenv("SONICBOOM_GEMMA_GGUF");
  FILE *f = path ? fopen(path, "rb") : NULL;
  if (!f) {
    printf("  (skipped: SONICBOOM_GEMMA_GGUF not set or unreadable)\n");
    return;
  }
  fclose(f);

  boom_llm_model *model = NULL;
  boom_llm_session *session = NULL;
  boom_error *err = NULL;
  boom_token tok[64];
  size_t n = 0;
  const char *text = "The quick brown fox jumps over the lazy dog";

  if (boom_llm_model_load(path, &model, &err) != BOOM_OK) {
    check(0, "llm: model load succeeds");
    boom_error_free(err);
    return;
  }
  if (boom_llm_session_new(model, 512, &session, &err) != BOOM_OK) {
    check(0, "llm: session created");
    boom_error_free(err);
    boom_llm_model_free(model);
    return;
  }
  if (boom_llm_tokenize(model, text, strlen(text), tok, 64, &n, &err) != BOOM_OK ||
      n < 2) {
    check(0, "llm: prompt tokenized");
    boom_error_free(err);
    boom_llm_session_free(session);
    boom_llm_model_free(model);
    return;
  }

  uint64_t reused = 12345;
  check(boom_llm_prefill(session, tok, n, &reused, &err) == BOOM_OK,
        "llm: first prefill succeeds");
  check(reused == 0, "llm: the first prefill reuses nothing");

  /* The same prompt again: fully resident, so the reuse is capped at n-1. */
  reused = 12345;
  check(boom_llm_prefill(session, tok, n, &reused, &err) == BOOM_OK,
        "llm: a fully reused prompt prefills successfully");
  check(reused == n - 1, "llm: a fully reused prompt reports n-1 reused");
  boom_error_free(err);

  boom_llm_session_free(session);
  boom_llm_model_free(model);
}

/* A minimal, independent GGUF metadata reader: the test compares the
 * model-info document with what the file itself says, not with the library
 * that produced it. Only the header's key/value pairs are walked (the tensor
 * table after them is not read). */
typedef struct {
  char arch[64];
  char context_key[128];
  unsigned long long context_length;
  int tool_template; /* the chat template branches on tools */
} gguf_metadata_view;

static int read_le(FILE *f, unsigned char *buf, size_t n) {
  return fread(buf, 1, n, f) == n;
}

static int read_u32_le(FILE *f, unsigned int *v) {
  unsigned char b[4];
  if (!read_le(f, b, 4)) return 0;
  *v = (unsigned)b[0] | (unsigned)b[1] << 8 | (unsigned)b[2] << 16 |
       (unsigned)b[3] << 24;
  return 1;
}

static int read_u64_le(FILE *f, unsigned long long *v) {
  unsigned char b[8];
  if (!read_le(f, b, 8)) return 0;
  unsigned long long x = 0;
  for (int i = 7; i >= 0; --i) x = (x << 8) | b[i];
  *v = x;
  return 1;
}

/* Skip one GGUF value of TYPE (0..12: the scalar types, string, array). */
static int skip_gguf_value(FILE *f, unsigned int type) {
  size_t size;
  switch (type) {
    case 0: case 1: case 7: size = 1; break;   /* u8, i8, bool */
    case 2: case 3: size = 2; break;           /* u16, i16 */
    case 4: case 5: case 6: size = 4; break;   /* u32, i32, f32 */
    case 10: case 11: case 12: size = 8; break; /* u64, i64, f64 */
    case 8: {                                  /* string */
      unsigned long long n;
      if (!read_u64_le(f, &n)) return 0;
      return fseek(f, (long)n, SEEK_CUR) == 0;
    }
    case 9: {                                  /* array */
      unsigned int et;
      unsigned long long n;
      if (!read_u32_le(f, &et) || !read_u64_le(f, &n)) return 0;
      for (unsigned long long i = 0; i < n; ++i)
        if (!skip_gguf_value(f, et)) return 0;
      return 1;
    }
    default:
      return 0;
  }
  return fseek(f, (long)size, SEEK_CUR) == 0;
}

/* Read a scalar value of TYPE; only the integer types a context length uses. */
static int read_gguf_scalar(FILE *f, unsigned int type,
                            unsigned long long *out) {
  switch (type) {
    case 4: {
      unsigned int v;
      if (!read_u32_le(f, &v)) return 0;
      *out = v;
      return 1;
    }
    case 5: {
      unsigned int v;
      if (!read_u32_le(f, &v)) return 0;
      *out = (unsigned long long)(long long)(int)v;
      return 1;
    }
    case 10:
    case 11:
      return read_u64_le(f, out);
    default:
      return 0;
  }
}

static int read_gguf_metadata(const char *path, gguf_metadata_view *m) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) return 0;
  memset(m, 0, sizeof *m);
  unsigned int magic = 0, version = 0;
  unsigned long long tensors = 0, kv_count = 0;
  int ok = read_u32_le(f, &magic) && magic == 0x46554747u &&
           read_u32_le(f, &version) && read_u64_le(f, &tensors) &&
           read_u64_le(f, &kv_count);
  for (unsigned long long i = 0; ok && i < kv_count; ++i) {
    unsigned long long klen;
    char key[256];
    unsigned int type;
    ok = read_u64_le(f, &klen) && klen < sizeof key &&
         fread(key, 1, (size_t)klen, f) == klen;
    if (!ok) break;
    key[klen] = '\0';
    if (!read_u32_le(f, &type)) {
      ok = 0;
      break;
    }
    if (type == 8) {                            /* string value */
      unsigned long long n;
      if (!read_u64_le(f, &n)) {
        ok = 0;
        break;
      }
      if (strcmp(key, "general.architecture") == 0 && n < sizeof m->arch) {
        if (fread(m->arch, 1, (size_t)n, f) != n) ok = 0;
        m->arch[n] = '\0';
      } else if (strcmp(key, "tokenizer.chat_template") == 0 && n < (1u << 22)) {
        char *s = (char *)malloc((size_t)n + 1);
        if (s == NULL || fread(s, 1, (size_t)n, f) != n) {
          ok = 0;
        } else {
          s[n] = '\0';
          m->tool_template = strstr(s, "tools") != NULL ||
                             strstr(s, "tool_call") != NULL;
        }
        free(s);
      } else {
        ok = fseek(f, (long)n, SEEK_CUR) == 0;
      }
    } else if (strstr(key, ".context_length") != NULL) {
      int fits = (type == 4 || type == 5 || type == 10 || type == 11);
      ok = fits && read_gguf_scalar(f, type, &m->context_length);
      if (ok) snprintf(m->context_key, sizeof m->context_key, "%s", key);
    } else {
      ok = skip_gguf_value(f, type);
    }
  }
  fclose(f);
  return ok;
}

/* boom_llm_model_info describes the loaded GGUF: the two-call sizing contract
 * (reporting the length including the NUL), the architecture, the parameter
 * count, the context length, the vocabulary, and the weight-type mixture.
 * The values are compared with the file's own header, read independently by
 * read_gguf_metadata. One model per supported family (Gemma 4, Qwen2); each
 * skips cleanly when its GGUF is not configured. */
static void llm_model_info_for(const char *env, const char *label) {
  const char *path = getenv(env);
  FILE *f = path ? fopen(path, "rb") : NULL;
  if (!f) {
    printf("  (skipped: %s not set or unreadable)\n", env);
    return;
  }
  fclose(f);
  printf("  model info: %s (%s)\n", label, path);

  gguf_metadata_view meta;
  check(read_gguf_metadata(path, &meta), "info: the GGUF header is readable");
  check(strcmp(meta.context_key, "") != 0,
        "info: the GGUF declares the context length");

  boom_llm_model *model = NULL;
  boom_error *err = NULL;
  if (boom_llm_model_load(path, &model, &err) != BOOM_OK) {
    check(0, "info: model load succeeds");
    boom_error_free(err);
    return;
  }

  size_t need = 0;
  check(boom_llm_model_info(model, NULL, 0, &need, &err) == BOOM_ERR_BINDING,
        "info: a NULL buffer reports the needed size");
  check(need > 1, "info: the document has content");
  check(err != NULL && boom_error_get_kind(err) == BOOM_ERRKIND_BINDING,
        "info: the size request is a binding error");
  boom_error_free(err);
  err = NULL;

  char *json = (char *)malloc(need);
  check(json != NULL, "info: the buffer allocates");
  if (json != NULL) {
    size_t written = 0;
    check(boom_llm_model_info(model, json, need, &written, &err) == BOOM_OK,
          "info: the JSON fits the reported size");
    check(written == need && json[written - 1] == '\0',
          "info: *len_out includes the NUL");
    char arch[64];
    json_string(json, "arch", arch, sizeof arch);
    check(strcmp(arch, meta.arch) == 0,
          "info: the architecture is the GGUF's general.architecture");
    char key_arch[64] = "";
    (void)sscanf(meta.context_key, "%63[^.].context_length", key_arch);
    check(strcmp(key_arch, meta.arch) == 0,
          "info: the GGUF's context length is the architecture's");
    check(json_number(json, "params") > 0,
          "info: params counts the weights");
    check(json_number(json, "context_length") == meta.context_length,
          "info: the context length is the GGUF's <arch>.context_length");
    check(json_number(json, "vocab_size") > 0,
          "info: the vocabulary size is declared");
    check((strstr(json, "\"has_tool_template\": true") != NULL) ==
              (meta.tool_template != 0),
          "info: the tool-template flag matches the chat template");
    check(strstr(json, "\"has_vision\": ") != NULL,
          "info: the vision flag is present");
    check(strstr(json, "\"quant\": [") != NULL &&
              strstr(json, "\"type\": \"") != NULL,
          "info: the weight-type mixture is present");
    check(json_sum(json, "weights") == json_number(json, "params"),
          "info: the quant entries' weights add up to params");

    /* A short buffer is rejected as a binding error. */
    check(boom_llm_model_info(model, json, need - 1, &written, &err) ==
              BOOM_ERR_BINDING,
          "info: a short buffer is rejected");
    boom_error_free(err);
    err = NULL;
    free(json);
  }

  boom_llm_model_free(model);
}

static void test_llm_model_info(void) {
  llm_model_info_for("SONICBOOM_GEMMA_GGUF", "gemma4");
  llm_model_info_for("SONICBOOM_QWEN_GGUF", "qwen2");
}

/* The decision registry is readable through the C ABI as a table and as JSON,
 * with the same two-call sizing contract as the other string calls. */
static void test_explain(void) {
  size_t need = 0;
  boom_error *err = NULL;
  check(boom_explain_table(NULL, 0, &need, &err) == BOOM_ERR_BINDING,
        "explain: a NULL buffer reports the needed size");
  check(need > 1, "explain: the table has content");
  check(boom_error_get_kind(err) == BOOM_ERRKIND_BINDING,
        "explain: the short-buffer error is a binding error");
  boom_error_free(err);
  err = NULL;

  char *table = (char *)malloc(need);
  check(table != NULL, "explain: the table buffer allocates");
  if (table != NULL) {
    size_t written = 0;
    check(boom_explain_table(table, need, &written, &err) == BOOM_OK,
          "explain: the table fits the reported size");
    check(written == need && table[written - 1] == '\0',
          "explain: the table is NUL-terminated");
    check(strstr(table, "mmq.enabled device = ") != NULL &&
              strstr(table, "reason=\"") != NULL,
          "explain: the table names a decision and its reason");
    free(table);
  }

  need = 0;
  check(boom_explain_json(NULL, 0, &need, &err) == BOOM_ERR_BINDING,
        "explain: the JSON reports the needed size");
  char *json = (char *)malloc(need);
  check(json != NULL, "explain: the JSON buffer allocates");
  if (json != NULL) {
    size_t written = 0;
    check(boom_explain_json(json, need, &written, &err) == BOOM_OK,
          "explain: the JSON fits the reported size");
    check(strstr(json, "\"decisions\": [") != NULL &&
              strstr(json, "\"vram_bytes\":") != NULL &&
              strstr(json, "\"tuning_ms\":") != NULL,
          "explain: the JSON carries decisions and their declared costs");
    free(json);
  }
  boom_error_free(err);
}

/* boom_tune_digest names the tuning tables (not the runtime logs) as 16 hex
 * digits and is stable for the process. */
static void test_tune_digest(void) {
  const char *digest = boom_tune_digest();
  check(digest != NULL, "tune digest: returns a string");
  check(digest != NULL && strlen(digest) == 16,
        "tune digest: is 16 hex digits");
  for (size_t i = 0; digest != NULL && digest[i] != '\0'; ++i) {
    char c = digest[i];
    check((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'),
          "tune digest: every character is lowercase hex");
  }
  check(boom_tune_digest() == digest, "tune digest: stable for the process");
}

int main(void) {
  test_end_to_end();
  test_binding_error();
  test_parse_error();
  test_error_slot_reuse();
  test_llm_prefill_full_reuse();
  test_llm_model_info();
  test_explain();
  test_tune_digest();

  if (g_failures == 0) {
    printf("test_capi OK (C ABI parse/compile/bind/execute/retrieve/release)\n");
    return 0;
  }
  fprintf(stderr, "test_capi FAILED: %d check(s)\n", g_failures);
  return 1;
}
