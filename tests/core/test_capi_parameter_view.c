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

/* test_capi_parameter_view.c — read-only parameter storage views (Stage 2).
 *
 * Compiled as C: the new surface must stay a genuine C header. The graph is
 * the same Add+Relu graph as test_capi.c, whose parameter `b` is f32[8].
 *
 * The lifetime contract under test:
 *   - a view's bytes equal the parameter's bytes;
 *   - several views pin the same executable independently;
 *   - freeing the executable handle does not invalidate outstanding views;
 *   - releasing one view leaves the others valid, and release(NULL) is safe.
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

/* z = relu(x + b); b is a compile-time constant f32[8]. */
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

static const float EXPECTED[8] = {0.5f, -1.0f, 2.0f, -3.0f,
                                  4.0f, -5.0f, 6.0f, -7.0f};

static sb_executable *compile_add_relu(void) {
  sb_document *doc = NULL;
  sb_executable *exe = NULL;
  sb_error *err = NULL;

  if (sb_parse(ADD_RELU, strlen(ADD_RELU), &doc, &err) != SB_OK) {
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
  sb_document_free(doc); /* sb_compile copies what it needs */
  return exe;
}

static sb_tensor_desc make_desc(void) {
  sb_tensor_desc d;
  memset(&d, 0, sizeof(d));
  d.struct_size = sizeof(sb_tensor_desc);
  return d;
}

static int bytes_equal(const void *p, const float *expected, size_t n) {
  return p != NULL && memcmp(p, expected, n) == 0;
}

int main(void) {
  sb_error *err = NULL;

  check(sb_api_version() == SB_API_VERSION,
        "sb_api_version reports the header's SB_API_VERSION");

  sb_executable *exe = compile_add_relu();
  if (!exe) {
    fprintf(stderr, "test_capi_parameter_view FAILED to compile the graph\n");
    return 1;
  }

  /* --- enumeration and descriptor ---------------------------------------- */
  size_t count = 0;
  check(sb_executable_parameter_count(exe, &count, &err) == SB_OK && count == 1,
        "parameter_count: one parameter");

  const char *name = NULL;
  check(sb_executable_parameter_name_at(exe, 0, &name, &err) == SB_OK &&
            name != NULL && strcmp(name, "b") == 0,
        "parameter_name_at(0): \"b\"");

  sb_tensor_desc desc = make_desc();
  check(sb_executable_parameter_desc(exe, 0, &desc, &err) == SB_OK,
        "parameter_desc(0): success");
  check(desc.struct_size == sizeof(sb_tensor_desc),
        "parameter_desc: struct_size reports the library's size");
  check(desc.dtype == SB_DTYPE_FLOAT32 && desc.rank == 1 && desc.dims &&
            desc.dims[0] == 8 && desc.num_bytes == 32,
        "parameter_desc: f32[8], 32 bytes");
  check((desc.flags & SB_TF_READ_ONLY) != 0 && desc.quant == SB_QUANT_NONE &&
            desc.quant_block == 0,
        "parameter_desc: read-only, dense");
  check(desc.name != NULL && strcmp(desc.name, "b") == 0,
        "parameter_desc: name");

  /* A struct_size smaller than this library's struct is rejected, not
   * silently overrun. */
  sb_tensor_desc small = make_desc();
  small.struct_size = sizeof(sb_tensor_desc) - 1;
  check(sb_executable_parameter_desc(exe, 0, &small, &err) == SB_ERR_INTERNAL,
        "parameter_desc: a too-small struct_size is rejected");
  sb_error_free(err);
  err = NULL;
  sb_tensor_desc dummy = make_desc();
  check(sb_executable_parameter_desc(exe, 1, &dummy, &err) == SB_ERR_INTERNAL,
        "parameter_desc(1): out of range rejected");
  sb_error_free(err);
  err = NULL;

  /* --- two independent views --------------------------------------------- */
  sb_storage *v1 = NULL, *v2 = NULL;
  check(sb_executable_parameter_view(exe, 0, &v1, &err) == SB_OK && v1 != NULL,
        "parameter_view(0): first view");
  check(sb_executable_parameter_view(exe, 0, &v2, &err) == SB_OK && v2 != NULL,
        "parameter_view(0): second view");

  const void *data1 = NULL;
  uint64_t bytes1 = 0;
  check(sb_storage_data(v1, &data1, &bytes1, &err) == SB_OK &&
            bytes1 == 32 && bytes_equal(data1, EXPECTED, 32),
        "storage_data: view bytes equal the parameter bytes");

  sb_tensor_desc vdesc = make_desc();
  check(sb_storage_desc(v1, &vdesc, &err) == SB_OK &&
            vdesc.dtype == SB_DTYPE_FLOAT32 && vdesc.rank == 1 &&
            vdesc.num_bytes == 32 && (vdesc.flags & SB_TF_READ_ONLY) != 0,
        "storage_desc: matches the parameter descriptor");

  /* Releasing one view must not disturb the other. */
  sb_storage_release(v1);
  v1 = NULL;
  const void *data2 = NULL;
  check(sb_storage_data(v2, &data2, NULL, &err) == SB_OK &&
            bytes_equal(data2, EXPECTED, 32),
        "storage_data: the second view survives the first's release");
  sb_error_free(err);
  err = NULL;

  /* --- the executable handle may die first -------------------------------- */
  sb_executable_free(exe);
  exe = NULL;

  const void *after_free = NULL;
  uint64_t bytes_after = 0;
  check(sb_storage_data(v2, &after_free, &bytes_after, &err) == SB_OK &&
            bytes_after == 32 && bytes_equal(after_free, EXPECTED, 32),
        "storage_data: valid after sb_executable_free");
  sb_tensor_desc after_desc = make_desc();
  check(sb_storage_desc(v2, &after_desc, &err) == SB_OK &&
            after_desc.num_bytes == 32,
        "storage_desc: valid after sb_executable_free");

  /* --- release discipline -------------------------------------------------- */
  sb_storage_release(NULL);
  sb_storage_release(NULL);
  check(1, "storage_release(NULL): safe, twice");

  sb_storage *v3 = NULL;
  sb_executable *exe2 = compile_add_relu();
  if (exe2) {
    check(sb_executable_parameter_view(exe2, 0, &v3, &err) == SB_OK,
          "parameter_view: third view on a second executable");
    sb_storage_release(v3);
    v3 = NULL;
    sb_storage_release(v2); /* the other view, still valid */
    v2 = NULL;
    sb_executable_free(exe2);
  }

  check(sb_storage_data(NULL, NULL, NULL, &err) == SB_ERR_INTERNAL,
        "storage_data: NULL view rejected");
  sb_error_free(err);

  if (g_failures == 0) {
    printf("test_capi_parameter_view OK (read-only parameter views)\n");
    return 0;
  }
  fprintf(stderr, "test_capi_parameter_view FAILED: %d check(s)\n",
          g_failures);
  return 1;
}
