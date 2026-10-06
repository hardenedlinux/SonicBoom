// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// SonicBoom stable C ABI (Layer 3).
//
// The long-term binary-compatibility boundary consumed by language bindings
// (first Guile) and other C callers. It is a thin facade over the SonicBoom
// core: S-Expr v0.1 text → parse → compile → bind → execute → retrieve, with
// opaque handles and explicit status/error returns. This header is pure C (no
// C++ classes/templates/STL, no MLIR/LLVM/ATen/c10 types, no exceptions across
// the boundary). ABI stability is the contract here, unlike the C++ Layer 2
// headers which are source-level only.

#ifndef SONICBOOM_CAPI_H
#define SONICBOOM_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. Contents are private to the implementation. */
typedef struct sb_document sb_document;      /* a parsed S-Expr v0.1 document  */
typedef struct sb_executable sb_executable;  /* a compiled, runnable graph     */
typedef struct sb_error sb_error;            /* a failure detail               */
typedef struct sb_tensor sb_tensor;          /* a float32 tensor (autograd)     */
typedef struct sb_tape sb_tape;              /* a differentiable-op tape        */

/* Status codes. 0 == success; every non-zero value is a failure. */
typedef enum sb_status {
  SB_OK = 0,
  SB_ERR_PARSE = 1,    /* the S-Expr text was malformed or unsupported        */
  SB_ERR_WEIGHT = 2,   /* a weight sidecar could not be read/resolved         */
  SB_ERR_COMPILE = 3,  /* lowering / pass pipeline / JIT compilation failed   */
  SB_ERR_BINDING = 4,  /* input/output buffer size or type mismatch           */
  SB_ERR_RUNTIME = 5,  /* the JITted function failed or was unavailable       */
  SB_ERR_INTERNAL = 6, /* invalid argument to the C API, or unexpected error  */
} sb_status;

/* Error categories (the `message` carries the human-readable detail). */
typedef enum sb_error_kind {
  SB_ERRKIND_NONE = 0,
  SB_ERRKIND_PARSE_LEXICAL = 1,
  SB_ERRKIND_PARSE_SYNTAX = 2,
  SB_ERRKIND_PARSE_SEMANTIC = 3,
  SB_ERRKIND_PARSE_UNSUPPORTED = 4,
  SB_ERRKIND_WEIGHT = 5,
  SB_ERRKIND_COMPILE = 6,
  SB_ERRKIND_BINDING = 7,
  SB_ERRKIND_RUNTIME = 8,
  SB_ERRKIND_INTERNAL = 9,
} sb_error_kind;

/* The frozen v0 dtype set. */
typedef enum sb_dtype {
  SB_DTYPE_FLOAT32 = 0,
  SB_DTYPE_FLOAT16 = 1,
  SB_DTYPE_BFLOAT16 = 2,
  SB_DTYPE_FLOAT64 = 3,
  SB_DTYPE_INT8 = 4,
  SB_DTYPE_UINT8 = 5,
  SB_DTYPE_INT16 = 6,
  SB_DTYPE_INT32 = 7,
  SB_DTYPE_INT64 = 8,
  SB_DTYPE_BOOL = 9,
} sb_dtype;

/* --- Error inspection ------------------------------------------------------ */

/* Error handles are owned by the caller. On failure, a status-returning call
 * writes a fresh error handle to *err_out *only if* err_out is non-NULL; the
 * caller must release it with sb_error_free. Ownership discipline: if the same
 * error-slot variable is reused across calls without freeing the previous
 * handle in between, the implementation frees the previous handle before
 * writing a new one, so reuse never leaks. A slot that has never held an error
 * (or was NULLed after a free) is simply overwritten; the implementation never
 * dereferences or frees an uninitialized or caller-owned pointer. The caller
 * must still free the last handle written to a slot.
 *
 * Exception containment: no C++ exception escapes any exported sb_* function.
 * Every status-returning function is wrapped so that an expected failure (the
 * normal error kinds above) and any unexpected exception (including bad_alloc)
 * are both mapped to a status code plus, when err_out is non-NULL, an error
 * handle. Unexpected exceptions surface as SB_ERR_INTERNAL.
 *
 * Limitation: handling is not fully reliable if error-object construction
 * itself fails. When allocation of the error handle or its message fails under
 * memory pressure, the slot is left NULL and the failure is still reported by
 * the returned status code, but there is no diagnostic message to retrieve. */

/* NUL-terminated message; valid until sb_error_free(). Never NULL ("" if the
 * handle is NULL). */
const char *sb_error_message(const sb_error *err);
sb_error_kind sb_error_get_kind(const sb_error *err);
void sb_error_free(sb_error *err);

/* --- Parse ----------------------------------------------------------------- */

/* Parse an S-Expr v0.1 document. `text` need not be NUL-terminated; `len` is
 * its byte length. On success *doc_out receives a new handle (release with
 * sb_document_free). On failure *doc_out is set to NULL and, if `err_out` is
 * non-NULL, *err_out receives a new error handle (release with sb_error_free). */
sb_status sb_parse(const char *text, size_t len, sb_document **doc_out,
                   sb_error **err_out);

/* --- Compile --------------------------------------------------------------- */

/* Compile a parsed document for CPU execution. `base_dir` resolves relative
 * weight-sidecar paths (pass "" to resolve against the process working
 * directory). On success *exe_out receives a new handle (release with
 * sb_executable_free). */
sb_status sb_compile(const sb_document *doc, const char *base_dir,
                     sb_executable **exe_out, sb_error **err_out);

/* --- I/O introspection (bind) ---------------------------------------------- */

/* Query the single input/output tensor. `dims_out`, if non-NULL, is set to a
 * borrowed pointer to the dimension array (valid until the executable is
 * freed). Any output pointer may be NULL if that field is not needed.
 * v0 execution scope is exactly one input and one output tensor. */
sb_status sb_input_info(const sb_executable *exe, sb_dtype *dtype_out,
                        int64_t *rank_out, const int64_t **dims_out,
                        uint64_t *num_bytes_out);
sb_status sb_output_info(const sb_executable *exe, sb_dtype *dtype_out,
                         int64_t *rank_out, const int64_t **dims_out,
                         uint64_t *num_bytes_out);

/* --- Execute / retrieve ---------------------------------------------------- */

/* Bind the input tensor buffer. `data` is copied, so the caller may free it
 * immediately after this returns. `len` must equal the input's num_bytes. */
sb_status sb_bind_input(sb_executable *exe, const void *data, size_t len,
                        sb_error **err_out);

/* Run the bound graph. The output is retained internally until the next
 * sb_execute or until the executable is freed. */
sb_status sb_execute(sb_executable *exe, sb_error **err_out);

/* Copy the retained output into `out` (capacity `cap` bytes). On success
 * `*written_out` is the output byte count (== the output num_bytes). */
sb_status sb_retrieve_output(sb_executable *exe, void *out, size_t cap,
                             size_t *written_out, sb_error **err_out);

/* --- Release --------------------------------------------------------------- */

void sb_document_free(sb_document *doc);
void sb_executable_free(sb_executable *exe);

/* --- Training (minimal autograd) ------------------------------------------- */
/* A float32-only reverse-mode autograd surface for a small MLP: linear/relu/
 * mse forward ops recorded on a tape, backward (VJP replay), and an in-place
 * SGD step. Guile (or another binding) drives the training loop; all numeric
 * work happens in native C++ behind these handles. No tensor is exposed as a
 * raw pointer; data crosses the boundary only as little-endian byte buffers. */

/* Create a float32 tensor of the given shape (dims[0..rank)) from little-endian
 * bytes. `len` must equal numel * 4. The handle owns the native tensor. */
sb_status sb_tensor_from_f32(const int64_t *dims, int64_t rank, const void *data,
                             size_t len, sb_tensor **out, sb_error **err_out);

/* Copy a tensor's little-endian float32 bytes into `out` (capacity `cap`). On
 * success *written_out is the byte count (== numel * 4). */
sb_status sb_tensor_bytes(const sb_tensor *t, void *out, size_t cap,
                          size_t *written_out, sb_error **err_out);

void sb_tensor_free(sb_tensor *t);

/* Create an empty tape (no recorded ops, no gradients). */
sb_status sb_tape_new(sb_tape **out, sb_error **err_out);

void sb_tape_free(sb_tape *tape);

/* Clear the tape's recorded ops and accumulated gradients. */
void sb_tape_zero_grad(sb_tape *tape);

/* Forward ops: each returns a new tensor handle and records a node on `tape`.
 * linear: y = x @ Wᵀ + b (x [N,in], W [out,in], b [out] → [N,out]). */
sb_status sb_linear(const sb_tensor *x, const sb_tensor *w, const sb_tensor *b,
                    sb_tape *tape, sb_tensor **out, sb_error **err_out);

/* relu: y = max(x, 0), elementwise. */
sb_status sb_relu(const sb_tensor *x, sb_tape *tape, sb_tensor **out,
                  sb_error **err_out);

/* mse_loss: scalar = mean((pred - target)²); returns a 0-dim tensor. */
sb_status sb_mse_loss(const sb_tensor *pred, const sb_tensor *target,
                      sb_tape *tape, sb_tensor **out, sb_error **err_out);

/* Reverse-mode backprop from `loss` (seeds dL/dL = 1), accumulating gradients
 * into `tape`. */
sb_status sb_backward(const sb_tensor *loss, sb_tape *tape, sb_error **err_out);

/* In-place SGD update param <- param - lr * grad, using `param`'s accumulated
 * gradient in `tape`. The parameter keeps its identity across calls. */
sb_status sb_sgd_step(sb_tape *tape, sb_tensor *param, double lr,
                      sb_error **err_out);

/* Read `param`'s accumulated gradient as a new tensor handle. Returns
 * SB_ERR_RUNTIME if no gradient has been accumulated for `param`. */
sb_status sb_grad(const sb_tape *tape, const sb_tensor *param, sb_tensor **out,
                  sb_error **err_out);

/* --- Export ----------------------------------------------------------------- */

/* Persist a trained model as a self-contained artifact directory (`dir`):
 *
 *   <dir>/model.sx    — the S-Expr v0.1 graph text; parameters whose name is in
 *                       `names` are rewritten to :external references into
 *                       weights.bin
 *   <dir>/weights.bin — concatenated little-endian float32 bytes of those
 *                       parameters, in graph parameter-declaration order
 *
 * `sx_text`/`sx_len` is the graph (the same text the nn composition emits).
 * `names[i]` names the parameter; `params[i]` is its trained float32 tensor,
 * whose byte count must equal the declared shape (SB_ERR_INTERNAL otherwise).
 * Parameters not listed keep their existing inline/external data. `dir` is
 * created if needed. The artifact is loadable later with sb_compile using
 * `dir` as base_dir, with no Guile or Python at load time. */
sb_status sb_export_model(const char *sx_text, size_t sx_len,
                          const char *const *names, const sb_tensor *const *params,
                          size_t count, const char *dir, sb_error **err_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SONICBOOM_CAPI_H */
