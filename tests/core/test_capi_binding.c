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

/* test_capi_binding.c — deferred (:bound) parameters (Stage 3).
 *
 * Compiled as C: the new surface must stay a genuine C header. The graph is
 * the Stage 2 Add+Relu graph with `b` declared (data :bound "b") instead of
 * inline values.
 *
 * The contract under test:
 *   - compile succeeds without the bytes; bindings_complete is 0;
 *   - run before binding is SB_ERR_BINDING;
 *   - wrong size / unknown name is SB_ERR_BINDING, NULL is SB_ERR_INTERNAL;
 *   - binding fills the slot, completes the bindings and makes run work;
 *   - rebinding before the first successful run replaces the bytes;
 *   - binding copies: later writes to the caller's buffer are not observed;
 *   - rebinding after the first successful run is SB_ERR_BINDING;
 *   - two executables bind independently (no shared state);
 *   - a parameter view shows the bound bytes and outlives the handle.
 */

#include <sonicboom/capi.h>

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

/* z = relu(x + b); b is a deferred f32[8]. */
static const char *ADD_RELU_BOUND =
    "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_bound\")"
    " (inputs (input \"x\" (tensor float32 (shape 8))))"
    " (outputs (output \"z\"))"
    " (parameters (parameter \"b\" (tensor float32 (shape 8))"
    "   (data :bound \"b\")))"
    " (nodes"
    "   (node add (inputs \"x\" \"b\")"
    "     (outputs (\"y\" (tensor float32 (shape 8)))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 8))))))))";

/* The same graph with inline values: no binding step. */
static const char *ADD_RELU_VALUES =
    "(sonicboom-s-expr (version 0 1) (graph (name \"add_relu_values\")"
    " (inputs (input \"x\" (tensor float32 (shape 8))))"
    " (outputs (output \"z\"))"
    " (parameters (parameter \"b\" (tensor float32 (shape 8))"
    "   (data :values 0.5 -1.0 2.0 -3.0 4.0 -5.0 6.0 -7.0)))"
    " (nodes"
    "   (node add (inputs \"x\" \"b\")"
    "     (outputs (\"y\" (tensor float32 (shape 8)))))"
    "   (node relu (inputs \"y\")"
    "     (outputs (\"z\" (tensor float32 (shape 8))))))))";

static const float B[8] = {0.5f, -1.0f, 2.0f, -3.0f,
                           4.0f, -5.0f, 6.0f, -7.0f};
static const float X[8] = {1.0f, 2.0f, -3.0f, 4.0f,
                           -5.0f, 6.0f, -7.0f, 8.0f};
static const float EXPECTED[8] = {1.5f, 1.0f, 0.0f, 1.0f,
                                  0.0f, 1.0f, 0.0f, 1.0f};

static sb_executable *compile_graph(const char *text) {
  sb_document *doc = NULL;
  sb_executable *exe = NULL;
  sb_error *err = NULL;

  if (sb_parse(text, strlen(text), &doc, &err) != SB_OK) {
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
  sb_document_free(doc);
  return exe;
}

static int run_once(sb_executable *exe, float out[8]) {
  sb_error *err = NULL;
  size_t written = 0;
  if (sb_bind_input(exe, X, sizeof(X), &err) != SB_OK) {
    sb_error_free(err);
    return 0;
  }
  if (sb_execute(exe, &err) != SB_OK) {
    sb_error_free(err);
    return 0;
  }
  int ok = sb_retrieve_output(exe, out, sizeof(float) * 8, &written, &err) ==
               SB_OK &&
           written == sizeof(X);
  sb_error_free(err);
  return ok;
}

static sb_tensor_desc make_desc(void) {
  sb_tensor_desc d;
  memset(&d, 0, sizeof(d));
  d.struct_size = sizeof(sb_tensor_desc);
  return d;
}

int main(void) {
  sb_error *err = NULL;

  check(sb_api_version() == SB_API_VERSION,
        "sb_api_version reports the header's SB_API_VERSION");

  sb_executable *exe = compile_graph(ADD_RELU_BOUND);
  if (!exe) {
    fprintf(stderr, "test_capi_binding FAILED to compile the graph\n");
    return 1;
  }

  /* --- the declaration survives compile without bytes --------------------- */
  size_t count = 0;
  check(sb_executable_parameter_count(exe, &count, &err) == SB_OK && count == 1,
        "parameter_count: one bound parameter");
  const char *name = NULL;
  check(sb_executable_parameter_name_at(exe, 0, &name, &err) == SB_OK &&
            name != NULL && strcmp(name, "b") == 0,
        "parameter_name_at(0): \"b\"");
  sb_tensor_desc desc = make_desc();
  check(sb_executable_parameter_desc(exe, 0, &desc, &err) == SB_OK &&
            desc.dtype == SB_DTYPE_FLOAT32 && desc.rank == 1 &&
            desc.num_bytes == 32,
        "parameter_desc: f32[8], 32 bytes");

  int complete = -1;
  check(sb_bindings_complete(exe, &complete, &err) == SB_OK && complete == 0,
        "bindings_complete: not complete before binding");

  /* --- running before binding fails with a binding error ------------------ */
  float out[8] = {0};
  check(!run_once(exe, out), "run before binding fails");

  sb_bind_input(exe, X, sizeof(X), &err);
  check(sb_execute(exe, &err) == SB_ERR_BINDING,
        "sb_execute before binding: SB_ERR_BINDING");
  sb_error_free(err);
  err = NULL;

  /* --- binding errors ------------------------------------------------------ */
  check(sb_bind_parameter(exe, "nope", B, sizeof(B), &err) == SB_ERR_BINDING,
        "unknown binding name: SB_ERR_BINDING");
  sb_error_free(err);
  err = NULL;
  check(sb_bind_parameter(exe, "b", B, sizeof(B) - 1, &err) == SB_ERR_BINDING,
        "wrong byte count: SB_ERR_BINDING");
  sb_error_free(err);
  err = NULL;
  check(sb_bind_parameter(exe, "b", NULL, sizeof(B), &err) == SB_ERR_INTERNAL,
        "NULL data: SB_ERR_INTERNAL");
  sb_error_free(err);
  err = NULL;
  check(sb_bindings_complete(exe, &complete, &err) == SB_OK && complete == 0,
        "a failed bind leaves the graph incomplete");

  /* --- a successful bind makes the graph runnable ------------------------- */
  static const float B2[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  check(sb_bind_parameter(exe, "b", B, sizeof(B), &err) == SB_OK,
        "bind_parameter: success");
  check(sb_bindings_complete(exe, &complete, &err) == SB_OK && complete == 1,
        "bindings_complete: complete after binding");

  /* Rebinding before the first successful run replaces the bytes. */
  check(sb_bind_parameter(exe, "b", B2, sizeof(B2), &err) == SB_OK,
        "rebinding before the first run: success");
  sb_storage *rebound = NULL;
  const void *rb_data = NULL;
  uint64_t rb_bytes = 0;
  check(sb_executable_parameter_view(exe, 0, &rebound, &err) == SB_OK &&
            sb_storage_data(rebound, &rb_data, &rb_bytes, &err) == SB_OK &&
            rb_bytes == sizeof(B2) && memcmp(rb_data, B2, sizeof(B2)) == 0,
        "the rebound bytes replace the first binding");
  sb_storage_release(rebound);

  /* Binding copies: writing through the caller's buffer afterwards changes
     nothing. */
  float scratch[8];
  memcpy(scratch, B, sizeof(B));
  check(sb_bind_parameter(exe, "b", scratch, sizeof(scratch), &err) == SB_OK,
        "binding from a caller-owned buffer");
  memset(scratch, 0, sizeof(scratch));

  /* The view observes the bound bytes without copying. */
  sb_storage *view = NULL;
  check(sb_executable_parameter_view(exe, 0, &view, &err) == SB_OK,
        "parameter_view on a bound parameter");
  const void *view_data = NULL;
  uint64_t view_bytes = 0;
  check(sb_storage_data(view, &view_data, &view_bytes, &err) == SB_OK &&
            view_bytes == sizeof(B) &&
            memcmp(view_data, B, sizeof(B)) == 0,
        "the view sees the bound bytes, not later writes to the caller's "
        "buffer");

  check(run_once(exe, out) && memcmp(out, EXPECTED, sizeof(EXPECTED)) == 0,
        "run after binding produces the :values result");

  /* --- binding is a one-time initialization ------------------------------- */
  check(sb_bind_parameter(exe, "b", B, sizeof(B), &err) == SB_ERR_BINDING,
        "rebinding after a successful run: SB_ERR_BINDING");
  sb_error_free(err);
  err = NULL;

  /* --- the view outlives the executable handle ----------------------------- */
  sb_executable_free(exe);
  exe = NULL;
  check(sb_storage_data(view, &view_data, &view_bytes, &err) == SB_OK &&
            view_bytes == sizeof(B) &&
            memcmp(view_data, B, sizeof(B)) == 0,
        "the view survives sb_executable_free");
  sb_storage_release(view);
  view = NULL;

  /* --- two executables bind independently --------------------------------- */
  sb_executable *a = compile_graph(ADD_RELU_BOUND);
  sb_executable *c = compile_graph(ADD_RELU_BOUND);
  if (a && c) {
    static const float B2[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    static const float A_OUT[8] = {1.5f, 1.0f, 0.0f, 1.0f,
                                   0.0f, 1.0f, 0.0f, 1.0f};
    static const float C_OUT[8] = {2.0f, 3.0f, 0.0f, 5.0f,
                                   0.0f, 7.0f, 0.0f, 9.0f};
    float oa[8], oc[8];
    check(sb_bind_parameter(a, "b", B, sizeof(B), &err) == SB_OK,
          "executable A binds its own bytes");
    check(sb_bind_parameter(c, "b", B2, sizeof(B2), &err) == SB_OK,
          "executable C binds different bytes");
    check(run_once(a, oa) && memcmp(oa, A_OUT, sizeof(A_OUT)) == 0,
          "executable A keeps its binding");
    check(run_once(c, oc) && memcmp(oc, C_OUT, sizeof(C_OUT)) == 0,
          "executable C keeps its own binding");
    check(run_once(a, oa) && memcmp(oa, A_OUT, sizeof(A_OUT)) == 0,
          "executable A is unaffected by C's run");
    sb_executable_free(a);
    sb_executable_free(c);
  } else {
    check(0, "independent executables compiled");
    sb_executable_free(a);
    sb_executable_free(c);
  }

  /* --- argument checks ----------------------------------------------------- */
  check(sb_bind_parameter(NULL, "b", B, sizeof(B), &err) == SB_ERR_INTERNAL,
        "NULL executable rejected");
  sb_error_free(err);
  err = NULL;
  check(sb_bindings_complete(NULL, &complete, &err) == SB_ERR_INTERNAL,
        "bindings_complete(NULL) rejected");
  sb_error_free(err);
  err = NULL;

  /* A graph without bound parameters is always complete and runs as before. */
  sb_executable *plain = compile_graph(ADD_RELU_VALUES);
  if (plain) {
    float po[8];
    check(sb_bindings_complete(plain, &complete, &err) == SB_OK &&
              complete == 1,
          "an all-:values graph is complete");
    check(run_once(plain, po) && memcmp(po, EXPECTED, sizeof(EXPECTED)) == 0,
          "an all-:values graph runs without binding");
    sb_executable_free(plain);
  } else {
    check(0, "the :values graph compiled");
  }

  if (g_failures == 0) {
    printf("test_capi_binding OK (deferred :bound parameters)\n");
    return 0;
  }
  fprintf(stderr, "test_capi_binding FAILED: %d check(s)\n", g_failures);
  return 1;
}
