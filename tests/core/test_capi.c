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
#include <string.h>

static int g_failures = 0;

static void check(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
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
  sb_document *doc = NULL;
  sb_executable *exe = NULL;
  sb_error *err = NULL;

  if (sb_parse(ADD_RELU, strlen(ADD_RELU), &doc, &err) != SB_OK) {
    fprintf(stderr, "  FAIL: parse (err: %s)\n", sb_error_message(err));
    ++g_failures;
    sb_error_free(err);
    return;
  }
  check(doc != NULL, "parse: document handle non-NULL");
  check(err == NULL, "parse: no error handle on success");

  if (sb_compile(doc, "", &exe, &err) != SB_OK) {
    fprintf(stderr, "  FAIL: compile (err: %s)\n", sb_error_message(err));
    ++g_failures;
    sb_error_free(err);
    sb_document_free(doc);
    return;
  }
  check(exe != NULL, "compile: executable handle non-NULL");

  /* Introspect input/output. */
  sb_dtype dtype = SB_DTYPE_FLOAT32;
  int64_t rank = 0;
  const int64_t *dims = NULL;
  uint64_t num_bytes = 0;
  check(sb_input_info(exe, &dtype, &rank, &dims, &num_bytes) == SB_OK,
        "input_info: success");
  check(dtype == SB_DTYPE_FLOAT32 && rank == 1 && dims && dims[0] == 8 &&
            num_bytes == 32,
        "input_info: f32[8], 32 bytes");
  check(sb_output_info(exe, &dtype, &rank, &dims, &num_bytes) == SB_OK,
        "output_info: success");
  check(dtype == SB_DTYPE_FLOAT32 && rank == 1 && dims && dims[0] == 8 &&
            num_bytes == 32,
        "output_info: f32[8], 32 bytes");

  /* Bind, execute, retrieve. */
  const float x[8] = {1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
  check(sb_bind_input(exe, x, sizeof(x), &err) == SB_OK, "bind_input: success");

  check(sb_execute(exe, &err) == SB_OK, "execute: success");

  float got[8];
  size_t written = 0;
  check(sb_retrieve_output(exe, got, sizeof(got), &written, &err) == SB_OK,
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

  sb_executable_free(exe);
  sb_document_free(doc);
}

static void test_binding_error(void) {
  sb_document *doc = NULL;
  sb_executable *exe = NULL;
  sb_error *err = NULL;

  if (sb_parse(ADD_RELU, strlen(ADD_RELU), &doc, &err) != SB_OK ||
      sb_compile(doc, "", &exe, &err) != SB_OK) {
    fprintf(stderr, "  FAIL: binding-error setup (err: %s)\n",
            sb_error_message(err));
    ++g_failures;
    sb_error_free(err);
    return;
  }

  /* Wrong input size → SB_ERR_BINDING. */
  float small[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  check(sb_bind_input(exe, small, sizeof(small), &err) == SB_ERR_BINDING,
        "bind_input: wrong size rejected");
  check(err != NULL && sb_error_get_kind(err) == SB_ERRKIND_BINDING,
        "bind_input: error kind is BINDING");
  sb_error_free(err);
  err = NULL;

  /* Execute before binding → SB_ERR_BINDING. */
  sb_executable *exe2 = NULL;
  if (sb_compile(doc, "", &exe2, &err) != SB_OK) {
    fprintf(stderr, "  FAIL: second compile (err: %s)\n", sb_error_message(err));
    ++g_failures;
    sb_error_free(err);
    sb_executable_free(exe);
    sb_document_free(doc);
    return;
  }
  check(sb_execute(exe2, &err) == SB_ERR_BINDING,
        "execute: unbound input rejected");
  sb_error_free(err);

  sb_executable_free(exe2);
  sb_executable_free(exe);
  sb_document_free(doc);
}

static void test_parse_error(void) {
  sb_document *doc = NULL;
  sb_error *err = NULL;

  /* Malformed s-expression → SB_ERR_PARSE. */
  const char *bad = "(sonicboom-s-expr (version 0 1) (graph";
  check(sb_parse(bad, strlen(bad), &doc, &err) == SB_ERR_PARSE,
        "parse: malformed text rejected");
  check(doc == NULL, "parse: no document on failure");
  check(err != NULL && sb_error_message(err)[0] != '\0',
        "parse: error message non-empty");
  sb_error_free(err);
}

/* Error-output slot reuse: two failures into the SAME err slot without freeing
 * in between. The implementation frees the prior handle before writing the new
 * one, so the slot must reflect the second error (not crash, not retain a stale
 * handle). A leak in this path is only visible under ASan; this test proves the
 * reuse discipline is safe and that the slot is genuinely replaced. */
static void test_error_slot_reuse(void) {
  sb_document *doc = NULL;
  sb_error *err = NULL;

  /* First failure: unterminated string literal → lexical error. */
  const char *bad1 = "\"unterminated";
  check(sb_parse(bad1, strlen(bad1), &doc, &err) == SB_ERR_PARSE,
        "reuse: first parse rejected");
  check(err != NULL && sb_error_get_kind(err) == SB_ERRKIND_PARSE_LEXICAL,
        "reuse: first error kind is LEXICAL");

  /* Second failure, same slot, no sb_error_free between the calls. */
  const char *bad2 = "(sonicboom-s-expr)";
  check(sb_parse(bad2, strlen(bad2), &doc, &err) == SB_ERR_PARSE,
        "reuse: second parse rejected");
  check(err != NULL && sb_error_get_kind(err) == SB_ERRKIND_PARSE_SYNTAX,
        "reuse: slot replaced with second error kind (SYNTAX)");

  sb_error_free(err);
}

int main(void) {
  test_end_to_end();
  test_binding_error();
  test_parse_error();
  test_error_slot_reuse();

  if (g_failures == 0) {
    printf("test_capi OK (C ABI parse/compile/bind/execute/retrieve/release)\n");
    return 0;
  }
  fprintf(stderr, "test_capi FAILED: %d check(s)\n", g_failures);
  return 1;
}
