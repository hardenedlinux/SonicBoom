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

/* test_capi_multi.c — the multi-input/multi-output C ABI.
 *
 * One Blade graph takes two f32[8] inputs and returns both `y = a + b` and
 * `z = relu(b)` (the JIT result struct is exercised with N = 2). The v0.1
 * interpreter supports add and relu, so those are the two ops used here; the
 * point is the two-input/two-output ABI, not the operator set. Compiled as C
 * to keep proving the header is genuine C and the symbols have C linkage.
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

/* y = a + b, z = relu(b): two inputs, two outputs, shape [8]. */
static const char *ADD_RELU_MULTI =
    "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_multi\")"
    " (inputs (input \"a\" (tensor float32 (shape 8)))"
    "         (input \"b\" (tensor float32 (shape 8))))"
    " (outputs (output \"y\") (output \"z\"))"
    " (parameters)"
    " (nodes"
    "   (node add (inputs \"a\" \"b\")"
    "     (outputs (\"y\" (tensor float32 (shape 8)))))"
    "   (node relu (inputs \"b\")"
    "     (outputs (\"z\" (tensor float32 (shape 8))))))))";

static sb_executable *compile_add_mul(sb_document **doc_out) {
  sb_document *doc = NULL;
  sb_executable *exe = NULL;
  sb_error *err = NULL;

  if (sb_parse(ADD_RELU_MULTI, strlen(ADD_RELU_MULTI), &doc, &err) != SB_OK) {
    fprintf(stderr, "  FAIL: parse (err: %s)\n", sb_error_message(err));
    ++g_failures;
    sb_error_free(err);
    return NULL;
  }
  if (sb_compile(doc, "", &exe, &err) != SB_OK) {
    fprintf(stderr, "  FAIL: compile (err: %s)\n", sb_error_message(err));
    ++g_failures;
    sb_error_free(err);
    sb_document_free(doc);
    return NULL;
  }
  *doc_out = doc;
  return exe;
}

static void test_arity_and_run(void) {
  sb_document *doc = NULL;
  sb_executable *exe = compile_add_mul(&doc);
  if (!exe)
    return;

  size_t n_in = 0, n_out = 0;
  check(sb_input_count(exe, &n_in) == SB_OK && n_in == 2,
        "input_count: 2");
  check(sb_output_count(exe, &n_out) == SB_OK && n_out == 2,
        "output_count: 2");

  sb_dtype dtype = SB_DTYPE_INT64;
  int64_t rank = -1;
  const int64_t *dims = NULL;
  uint64_t nbytes = 0;
  check(sb_input_info_at(exe, 1, &dtype, &rank, &dims, &nbytes) == SB_OK,
        "input_info_at(1): success");
  check(dtype == SB_DTYPE_FLOAT32 && rank == 1 && dims && dims[0] == 8 &&
            nbytes == 32,
        "input_info_at(1): f32[8], 32 bytes");
  check(sb_output_info_at(exe, 1, &dtype, &rank, &dims, &nbytes) == SB_OK,
        "output_info_at(1): success");
  check(dtype == SB_DTYPE_FLOAT32 && rank == 1 && dims && dims[0] == 8 &&
            nbytes == 32,
        "output_info_at(1): f32[8], 32 bytes");

  /* Index 0 through the old accessors still describes the first tensor. */
  check(sb_input_info(exe, &dtype, &rank, &dims, &nbytes) == SB_OK &&
            nbytes == 32,
        "input_info: index 0 still works");
  check(sb_output_info(exe, &dtype, &rank, &dims, &nbytes) == SB_OK &&
            nbytes == 32,
        "output_info: index 0 still works");
  check(sb_input_info_at(exe, 2, NULL, NULL, NULL, NULL) == SB_ERR_INTERNAL,
        "input_info_at(2): out of range rejected");
  check(sb_output_info_at(exe, 2, NULL, NULL, NULL, NULL) == SB_ERR_INTERNAL,
        "output_info_at(2): out of range rejected");

  const float a[8] = {1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
  const float b[8] = {0.5f, -1.0f, 2.0f, 3.0f, -4.0f, 5.0f, 6.0f, -7.0f};
  float y[8], z[8];
  const void *ins[2] = {a, b};
  size_t lens[2] = {sizeof(a), sizeof(b)};
  void *outs[2] = {y, z};
  size_t caps[2] = {sizeof(y), sizeof(z)};
  size_t written[2] = {0, 0};

  sb_error *err = NULL;
  check(sb_run_multi(exe, ins, lens, 2, outs, caps, 2, written, &err) == SB_OK,
        "run_multi: success");
  sb_error_free(err);
  err = NULL;
  check(written[0] == 32 && written[1] == 32, "run_multi: wrote 32+32 bytes");

  for (int i = 0; i < 8; ++i) {
    if (fabsf(y[i] - (a[i] + b[i])) > 1e-6f) {
      fprintf(stderr, "  FAIL: y[%d] = %f, expected %f\n", i, y[i],
              a[i] + b[i]);
      ++g_failures;
    }
    if (fabsf(z[i] - (b[i] > 0.0f ? b[i] : 0.0f)) > 1e-6f) {
      fprintf(stderr, "  FAIL: z[%d] = %f, expected %f\n", i, z[i],
              b[i] > 0.0f ? b[i] : 0.0f);
      ++g_failures;
    }
  }

  /* The single-in/single-out wrapper refuses a two-output graph instead of
   * silently returning the first result. */
  check(sb_bind_input(exe, a, sizeof(a), &err) == SB_OK,
        "bind_input: still works on a multi graph");
  check(sb_execute(exe, &err) == SB_ERR_BINDING,
        "execute: 1-out wrapper rejects a 2-output graph");
  sb_error_free(err);
  err = NULL;

  /* Error paths: arity, input length, output capacity. None may touch the
   * output buffers. */
  y[0] = z[0] = 123.0f;
  check(sb_run_multi(exe, ins, lens, 1, outs, caps, 2, written, &err) ==
            SB_ERR_BINDING,
        "run_multi: wrong input count rejected");
  sb_error_free(err);
  err = NULL;

  size_t bad_lens[2] = {sizeof(a), 4};
  check(sb_run_multi(exe, ins, bad_lens, 2, outs, caps, 2, written, &err) ==
            SB_ERR_BINDING,
        "run_multi: wrong input length rejected");
  sb_error_free(err);
  err = NULL;

  size_t small_caps[2] = {sizeof(y), 4};
  check(sb_run_multi(exe, ins, lens, 2, outs, small_caps, 2, written, &err) ==
            SB_ERR_BINDING,
        "run_multi: short output buffer rejected");
  sb_error_free(err);
  err = NULL;
  check(y[0] == 123.0f && z[0] == 123.0f,
        "run_multi: failed binding leaves outputs untouched");

  sb_executable_free(exe);
  sb_document_free(doc);
}

int main(void) {
  test_arity_and_run();

  if (g_failures != 0) {
    fprintf(stderr, "%d check(s) failed\n", g_failures);
    return 1;
  }
  return 0;
}
