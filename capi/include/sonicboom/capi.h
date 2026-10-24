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
typedef struct sb_param sb_param;            /* a trainable param + AdamW state */

/* Status codes. 0 == success; every non-zero value is a failure. */
typedef enum sb_status {
  SB_OK = 0,
  SB_ERR_PARSE = 1,    /* the S-Expr text was malformed or unsupported        */
  SB_ERR_WEIGHT = 2,   /* a weight sidecar could not be read/resolved         */
  SB_ERR_COMPILE = 3,  /* lowering / pass pipeline / JIT compilation failed   */
  SB_ERR_BINDING = 4,  /* input/output buffer size or type mismatch           */
  SB_ERR_RUNTIME = 5,  /* the JITted function failed or was unavailable       */
  SB_ERR_INTERNAL = 6, /* invalid argument to the C API, or unexpected error  */
  SB_ERR_UNSUPPORTED = 7, /* a requested feature this build/backend lacks     */
  SB_ERR_CANCELLED = 8, /* the caller's progress callback asked to stop      */
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
  SB_ERRKIND_CANCELLED = 10,
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

/* --- Build info ------------------------------------------------------------ */

/* A JSON object describing how this libsonicboom.so was built. Returns a static
 * string, valid for the lifetime of the process; never NULL and never an error.
 *
 * It exists so that a benchmark result or a bug report can name the exact
 * binary it came from: which commit, how it was configured, which CUDA
 * architectures the binary really contains (CMake's CMAKE_CUDA_ARCHITECTURES
 * may be the token `native`, which is resolved at build time), and which
 * LLVM/MLIR is linked in. The field set may grow — parse by name, not by
 * position.
 *
 * {"git_commit":"84bb114","git_tag":"v0.3-12-g84bb114",
 *  "build_type":"Release",
 *  "cuda":{"enabled":true,"archs":["86"],"toolkit":"12.0"},
 *  "llvm":{"version":"23.1.2","major":23,"origin":"external"},
 *  "platform":"x86_64"}
 *
 * `git_commit` is "<rev>[-dirty]" at configure time, or "unknown" outside a git
 * checkout. `git_tag` is `git describe --tags --always --dirty` at configure
 * time: the release name of that revision. `cuda` is {"enabled":false,"archs":[],"toolkit":""} in a CPU-only
 * build. `llvm.origin` is "bundled" for the install tools/mlir/build-mlir.sh
 * produces inside the source tree and "external" for any other LLVM/MLIR.
 * (MLIR is a required dependency, so `llvm` is always a real version/origin;
 * sb_build_info still reports none/"none" only in a build with no MLIR, which
 * configure now rejects.)
 * `platform` is the target CPU the library was compiled for, not the builder's. */
const char *sb_build_info(void);

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

/* Compile a parsed document as a device training graph. The document is the
 * forward+backward+step graph a traced training function produced: it is
 * lowered Blade -> Boom IR, run through the Boom IR Training pass phase, and
 * executed node by node on the GPU (CUDA). The graph is bound by name — the
 * tracer names every tensor argument — through sb_input_name_at /
 * sb_output_name_at, and a step runs through sb_run_multi. Parameters must be
 * inline (`data :values`); external weight sidecars are not supported. On a
 * build without CUDA this fails with SB_ERR_COMPILE and a message saying so;
 * the graph is still lowered and verified first, so a bad graph is reported as
 * such on either build. Release with sb_executable_free. */
sb_status sb_compile_device(const sb_document *doc, sb_executable **exe_out,
                            sb_error **err_out);

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

/* Graph arity and per-index introspection. A graph may have several inputs and
 * several outputs (a training graph typically returns one gradient per
 * parameter). sb_input_info/sb_output_info describe index 0; the *_info_at
 * forms describe any index in [0, count). An index out of range is
 * SB_ERR_INTERNAL (invalid argument). */
sb_status sb_input_count(const sb_executable *exe, size_t *count_out);
sb_status sb_output_count(const sb_executable *exe, size_t *count_out);
sb_status sb_input_info_at(const sb_executable *exe, size_t index,
                           sb_dtype *dtype_out, int64_t *rank_out,
                           const int64_t **dims_out, uint64_t *num_bytes_out);
sb_status sb_output_info_at(const sb_executable *exe, size_t index,
                            sb_dtype *dtype_out, int64_t *rank_out,
                            const int64_t **dims_out,
                            uint64_t *num_bytes_out);

/* Tensor names in binding order. `*name_out` is a borrowed NUL-terminated
 * string valid until the executable is freed, or NULL when the compiled form
 * does not retain names (the CPU interpreter compiled by sb_compile). A
 * device training graph always names its tensors: the tracer's names. */
sb_status sb_input_name_at(const sb_executable *exe, size_t index,
                           const char **name_out);
sb_status sb_output_name_at(const sb_executable *exe, size_t index,
                            const char **name_out);

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

/* Run the graph over N inputs and M outputs in one call — the general form
 * behind sb_bind_input/sb_execute/sb_retrieve_output (the 1-in/1-out wrapper).
 * `num_inputs` must equal sb_input_count and `num_outputs` must equal
 * sb_output_count. Every input is copied, so the caller may free it as soon as
 * this returns; `input_lens[i]` must equal that input's num_bytes.
 * `output_data[i]` must have capacity at least the i-th output's num_bytes and
 * `output_caps[i]` carries that capacity; capacities are checked before the
 * graph runs. On success each result is copied into `output_data[i]` and, when
 * `output_lens` is non-NULL, `output_lens[i]` receives the byte count written.
 * On failure no result buffer is modified. */
sb_status sb_run_multi(sb_executable *exe, const void *const *input_data,
                       const size_t *input_lens, size_t num_inputs,
                       void *const *output_data, const size_t *output_caps,
                       size_t num_outputs, size_t *output_lens,
                       sb_error **err_out);

/* Resident training step. A device training graph keeps the parameters and
 * optimizer state the optimizer updates in their device buffers across steps
 * (see sb_input_is_resident). The first successful run — sb_run_multi or
 * sb_run_resident — uploads them; afterwards `input_data[i]` for a resident
 * input may be NULL and its bytes are ignored. Every other input is uploaded on
 * every call, and `input_lens[i]` is still validated for all of them.
 *
 * Only the outputs named by `output_indices[0..num_outputs)` are copied back,
 * in that order: a step that keeps its state on the device downloads the loss,
 * not the parameters. `num_outputs` is therefore independent of
 * sb_output_count. On failure no result buffer is modified. */
sb_status sb_run_resident(sb_executable *exe, const void *const *input_data,
                          const size_t *input_lens, size_t num_inputs,
                          const size_t *output_indices, size_t num_outputs,
                          void *const *output_data, const size_t *output_caps,
                          size_t *output_lens, sb_error **err_out);

/* 1 when input `index` is resident in the sense above (the graph writes its
 * next value back into the same device buffer), 0 otherwise. Fails with
 * SB_ERR_BINDING for a non-device executable or an index out of range. */
sb_status sb_input_is_resident(const sb_executable *exe, size_t index,
                               int *resident_out, sb_error **err_out);

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

/* --- Training: parameter + fused AdamW -------------------------------------- */
/* A trainable float32 parameter owns its values, its gradient, and AdamW's two
 * moment buffers. When a CUDA device is present the four buffers stay in device
 * memory across steps and the update is one fused kernel launch; otherwise they
 * are host buffers and the host reference runs. The handle hides which.
 *
 * `dims[0..rank)` (row-major) only determine the element count; `len` must
 * equal numel * 4 and supplies the initial values. The gradient and both moment
 * buffers start at zero. */

sb_status sb_param_create(const int64_t *dims, int64_t rank, const void *init,
                          size_t len, sb_param **out, sb_error **err_out);

/* Element count, and whether the buffers are device-resident (1) or host
 * (0). Both are always available once the handle exists. */
sb_status sb_param_numel(const sb_param *p, size_t *numel_out);
int sb_param_on_device(const sb_param *p);

/* Updates applied so far (0 before the first successful sb_param_step). */
uint64_t sb_param_step_count(const sb_param *p);

/* Zero the gradient, replace it with `grad` (len == numel * 4), and apply one
 * AdamW update. `sb_param_read` copies the current parameter values out. */
sb_status sb_param_zero_grad(sb_param *p, sb_error **err_out);
sb_status sb_param_set_grad(sb_param *p, const void *grad, size_t len,
                            sb_error **err_out);
sb_status sb_param_step(sb_param *p, float lr, float beta1, float beta2,
                        float eps, float weight_decay, sb_error **err_out);

/* The lr-input form: the learning rate is read from a one-element float32
 * tensor (an sb_tensor handle, which is how a traced schedule carries the
 * rate). The parameter's own buffers stay device-resident; only the single
 * rate is read on the host. The handle's numel must be 1 and its dtype
 * float32. */
sb_status sb_param_step_lr(sb_param *p, const sb_tensor *lr, float beta1,
                           float beta2, float eps, float weight_decay,
                           sb_error **err_out);
sb_status sb_param_read(const sb_param *p, void *out, size_t cap,
                        size_t *written_out, sb_error **err_out);

void sb_param_free(sb_param *p);

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

/* Clip a set of gradient tensors by their global L2 norm, in place. GRADS is
 * COUNT float32 CPU tensors (the gradients read back with sb_grad, or any
 * other gradient buffers); the norm is accumulated in double over every
 * element of every tensor. When it exceeds MAX_NORM each tensor is scaled by
 * MAX_NORM / (norm + 1e-6); otherwise the tensors are left byte-for-byte
 * unchanged. *NORM_OUT, when non-NULL, receives the pre-clip global norm. */
sb_status sb_clip_grad_norm(sb_tensor *const *grads, int count, float max_norm,
                            float *norm_out, sb_error **err_out);

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

/* --- Checkpoints ------------------------------------------------------------- */

/* Named float32 tensors persisted in one self-describing binary file:
 *
 *   "SBCKPT1\n"                    8-byte magic
 *   u64                            entry count
 *   per entry:
 *     u32 name_len, name bytes     UTF-8, not NUL-terminated
 *     u8 dtype                     sb_dtype (float32 in this version)
 *     u32 rank
 *     i64 dims[rank]               row-major shape
 *     u64 byte_count               == numel * 4
 *     byte_count data bytes        little-endian float32, row-major
 *
 * All integers are little-endian. Entries are matched by name, so the order in
 * which a caller saves and loads them need not agree.
 *
 * sb_save_params writes `count` named tensors to `path` (overwriting it).
 * Names must be non-empty and unique; tensors must be float32. sb_load_params
 * copies the stored values into the caller's existing tensors, matched by
 * name: each target must be float32 with exactly the stored shape. The file is
 * fully parsed and validated before any target is written, so on any failure
 * every target is left untouched. The file may hold entries the caller does
 * not load (a checkpoint can carry more state than one call needs), and one
 * call may load a subset of its targets from several checkpoints.
 *
 * Errors: a missing, unreadable, truncated or malformed checkpoint is
 * SB_ERR_WEIGHT; a name/shape/dtype mismatch between file and caller is
 * SB_ERR_BINDING (also on save for duplicate names or a non-float32 tensor);
 * an invalid argument is SB_ERR_INTERNAL. */
sb_status sb_save_params(const char *path, const char *const *names,
                         const sb_tensor *const *params, size_t count,
                         sb_error **err_out);
sb_status sb_load_params(const char *path, const char *const *names,
                         sb_tensor *const *params, size_t count,
                         sb_error **err_out);

/* --- Numerics profile ------------------------------------------------------ */

/* The two numerics profiles (see core/include/sonicboom/numerics.h):
 *
 *   SB_NUMERICS_STRICT   -- the v0/v1 reference semantics. Every reduction is
 *                           identical to the historical binary; the KV cache
 *                           keeps f32; no TF32/DF32 substitutions.
 *   SB_NUMERICS_STANDARD -- the performance profile. Bit-identical to strict
 *                           unless a substitution is enabled for it; today
 *                           that is f16 KV storage (exact for the values the
 *                           cache holds) and the profile's other candidates.
 *
 * The profile is process-wide and latched: the first inference or explicit
 * set wins, because it is part of the plan, autotune and codegen cache keys.
 * Until it is set the process uses the SONICBOOM_NUMERICS environment variable
 * ("strict" or "standard"), then SB_NUMERICS_STANDARD.
 *
 * sb_get_numerics_profile returns the effective profile (the latched value, or
 * the environment/default value before the first latch). */
typedef enum sb_numerics_profile {
  SB_NUMERICS_STRICT = 0,
  SB_NUMERICS_STANDARD = 1,
} sb_numerics_profile;

/* Request a profile for this process. Returns SB_ERR_UNSUPPORTED (with a
 * message naming both profiles) if a different profile is already latched;
 * setting the already-effective profile is a no-op success. Call before the
 * first model load or inference for the setting to apply everywhere. */
sb_status sb_set_numerics_profile(sb_numerics_profile profile,
                                  sb_error **err_out);
sb_status sb_get_numerics_profile(sb_numerics_profile *profile_out,
                                  sb_error **err_out);

/* --- Inference (minimal LLM surface, sb_llm_*) ----------------------------- */
/* A thin facade over the Layer 2 model loader + planner spine, exposing the
 * smallest inference set the first binding (Guile) needs: model load/free,
 * session new/free, tokenize/detokenize, prefill, and greedy decode with a
 * per-token callback. The scope is intentionally minimal (v1 wrap-up):
 *  - decode runs one whole greedy generation per call (the CPU planner spine
 *    has no step-wise persistent session yet); a single-token prompt is a seed
 *    and decodes directly, a longer prompt prefills then decodes;
 *  - temperature 0 is greedy (the historical path); temperature > 0 samples
 *    deterministically from sb_llm_sampling's seed;
 *  - tokenize/detokenize use the model's GGUF built-in vocabulary (SPM-style
 *    BPE for Gemma 4, byte-level BPE for Qwen2), matching llama-tokenize;
 *  - the architectures SonicBoom models load (Gemma 4, Qwen2); anything else
 *    returns SB_ERR_UNSUPPORTED. Which one loaded is readable with
 *    sb_llm_model_arch, and the model's own stop tokens with sb_llm_model_eos —
 *    a caller that formats prompts per family takes both from the model rather
 *    than assuming.
 * These gaps are recorded in the post-v1 list. */

typedef struct sb_llm_model sb_llm_model;
typedef struct sb_llm_session sb_llm_session;
typedef int32_t sb_token;   /* a vocabulary index */

/* Sampling parameters for one decode call.
 *
 * temperature 0 is the greedy argmax path, bit-identical to the historical
 * decode. temperature > 0 enables seeded sampling: the application order is
 *   1. repetition / presence / frequency penalties over the last
 *      repeat_last_n tokens of the history (prompt + tokens generated so far);
 *   2. the grammar mask (constrained decoding);
 *   3. temperature scaling and a 30-logit candidate window;
 *   4. top_k truncation (top_k 0 disables it);
 *   5. min_p truncation (min_p 0 disables it);
 *   6. nucleus truncation when 0 < top_p < 1 (top_p 1 disables it);
 *   7. a seeded Gumbel-max draw keyed by (seed, absolute token position).
 * The same seed, position and history therefore always produce the same token,
 * independently of KV-cache capacity or prefix reuse.
 *
 * max_tokens caps the output length.
 *
 * Fields appended after max_tokens are read only when struct_size covers them;
 * a caller built against an older struct_size gets the neutral defaults below.
 *   min_p              neutral 0    (keep every candidate above the window)
 *   repeat_penalty     neutral 1    (no repetition penalty; must be > 0)
 *   repeat_last_n      neutral 64   (penalty window, tokens)
 *   presence_penalty   neutral 0    (subtracted once per repeated token)
 *   frequency_penalty  neutral 0    (subtracted per occurrence)
 * For a token t appearing c times in the window: if repeat_penalty != 1,
 * l[t] = l[t] / repeat_penalty when l[t] > 0 else l[t] * repeat_penalty; then
 * l[t] -= frequency_penalty * c + presence_penalty. */
typedef struct sb_llm_sampling {
  size_t struct_size;
  float temperature;
  float top_p;
  uint32_t top_k;
  uint64_t seed;
  uint32_t max_tokens;
  float min_p;
  float repeat_penalty;
  uint32_t repeat_last_n;
  float presence_penalty;
  float frequency_penalty;
} sb_llm_sampling;

/* Per-token callback: called synchronously on the calling thread, once per
 * produced token, in generation order. Return 0 to continue, non-zero to stop
 * the decode early. `user` is passed through untouched. */
typedef int (*sb_llm_token_cb)(void *user, sb_token token);

/* Load a GGUF model for inference. The model owns its backing buffer; a
 * session keeps the model alive after sb_llm_model_free. */
sb_status sb_llm_model_load(const char *gguf_path, sb_llm_model **model_out,
                            sb_error **err_out);
void sb_llm_model_free(sb_llm_model *model);

/* The loaded model's architecture, exactly as its GGUF declares it in
 * `general.architecture` ("gemma4", "qwen2", ...). A caller that must format
 * prompts, pick turn markers or otherwise behave differently per family reads
 * this from the model instead of guessing from the file name. Two-call sizing
 * like the other string out-params; *len_out excludes the NUL. */
sb_status sb_llm_model_arch(const sb_llm_model *model, char *out, size_t cap,
                            size_t *len_out, sb_error **err_out);

/* The model's end-of-sequence token id(s) as its GGUF declares them
 * (`tokenizer.ggml.eos_token_id`, plus `tokenizer.ggml.eos_token_ids` when the
 * model lists several). *n_out receives the count; none declared gives 0.
 * Two-call sizing: pass out NULL or cap too small for the count. */
sb_status sb_llm_model_eos(const sb_llm_model *model, sb_token *out, size_t cap,
                           size_t *n_out, sb_error **err_out);

/* Tokenize/detokenize with the model's GGUF built-in vocabulary (SPM-style BPE
 * for Gemma 4). tokenize encodes `text` (UTF-8, `len` bytes) into token ids
 * with <bos> prepended (llama-tokenize add_special); detokenize renders `tokens`
 * back to UTF-8 (NUL-terminated), stripping control tokens and undoing ▁/byte
 * escapes. Both use two-call sizing: pass `out` NULL or `cap` too small to get
 * the needed count in *n_out / *len_out (and SB_ERR_BINDING). */
sb_status sb_llm_tokenize(const sb_llm_model *model, const char *text, size_t len,
                          sb_token *out, size_t cap, size_t *n_out, sb_error **err_out);

/* sb_llm_tokenize_ex flags. SB_TOKENIZE_ADD_SPECIAL (prepend <bos>) is what
 * sb_llm_tokenize does; a caller that tokenizes a prompt piece by piece turns
 * it on for the first piece only, so <bos> is added exactly once. */
#define SB_TOKENIZE_PARSE_SPECIAL 1
#define SB_TOKENIZE_ADD_SPECIAL 2

/* sb_llm_tokenize with flags. With SB_TOKENIZE_PARSE_SPECIAL, a CONTROL /
 * USER_DEFINED piece of the vocabulary (e.g. Gemma 4's "<|turn>", "<turn|>")
 * occurring verbatim in `text` becomes that one token instead of being encoded
 * as text. This turns text into control structure, so it must only be used on
 * text the caller itself generated (a chat template's turn markers): a user
 * message containing "<turn|>" would otherwise become a real turn boundary
 * (prompt injection). sb_llm_tokenize is this with SB_TOKENIZE_ADD_SPECIAL
 * alone, byte for byte. */
sb_status sb_llm_tokenize_ex(const sb_llm_model *model, const char *text, size_t len,
                             int flags, sb_token *out, size_t cap,
                             size_t *n_out, sb_error **err_out);
sb_status sb_llm_detokenize(const sb_llm_model *model, const sb_token *tokens,
                            size_t n, char *out, size_t cap, size_t *len_out,
                            sb_error **err_out);

/* Create a session (a prompt/token history + n_ctx). The session holds a
 * reference to the model, so the model handle may be freed first. */
sb_status sb_llm_session_new(sb_llm_model *model, uint64_t n_ctx,
                             sb_llm_session **session_out, sb_error **err_out);
void sb_llm_session_free(sb_llm_session *session);

/* Set the session's prompt to tokens[0..n). *n_reused_out receives the length
 * of leading tokens reused from the previous prompt's KV state. The last prompt
 * token is always processed — its activation is what produces the next-token
 * logits — so the count is at most n-1, even when the whole prompt is already
 * resident. (The CPU decode path re-prefills rather than reusing KV slots; KV
 * reuse is the CUDA prefix-cache path.) */
sb_status sb_llm_prefill(sb_llm_session *session, const sb_token *tokens, size_t n,
                         uint64_t *n_reused_out, sb_error **err_out);

/* Decode up to sampling->max_tokens tokens from the session's prompt, invoking
 * cb for each (abort on a non-zero return). *n_out receives the count produced.
 * Requires a non-empty prompt (sb_llm_prefill first). */
sb_status sb_llm_decode(sb_llm_session *session, const sb_llm_sampling *sampling,
                        sb_llm_token_cb cb, void *user, uint32_t *n_out,
                        sb_error **err_out);

/* --- Constrained decoding (session grammar) --------------------------------
 *
 * A session may carry a byte-level grammar that every sampled token must keep
 * acceptable. `gbnf` is a GBNF document (the llama.cpp dialect: rule ::=
 * alternation, "literals" with the usual escapes, [byte classes], . * + ? and
 * {m,n}, # comments; the first rule is the start rule). The grammar consumes
 * the same bytes sb_llm_detokenize would emit for a token, so it constrains the
 * text the model produces, not token ids. The model's EOS ids (see
 * sb_llm_model_eos) are the grammar's stop tokens: they are not text, may be
 * sampled only where the grammar can end, and end the constrained output
 * without being emitted.
 *
 * sb_llm_session_set_json_schema compiles a JSON Schema (one JSON document) to
 * GBNF first; the supported subset is documented in
 * core/include/sonicboom/vocab/grammar.h, and an unsupported keyword is
 * rejected by name rather than silently ignored.
 *
 * A grammar starts at every sb_llm_prefill (a new turn) and applies to the
 * first token sampled by that prefill and to every sb_llm_decode step. Once a
 * stop token finishes it, sb_llm_decode produces no further tokens.
 *
 * sb_llm_session_constraint_state reports whether a grammar is installed
 * (*active_out) and whether the text accepted so far is already a complete
 * value of it (*complete_out). */
/* sb_llm_json_schema_to_gbnf is the compiler alone, as text: it lets a caller
 * compose a schema into a larger grammar instead of installing the schema as
 * the whole reply. PREFIX and SUFFIX (may be NULL) are literal UTF-8 text the
 * value must appear between; the root matches exactly that text, so a complete
 * value can only be continued by a stop token.
 * Two-call sizing like sb_llm_model_arch: *len_out always receives the length
 * (without NUL), and a capacity that cannot hold the text plus its NUL returns
 * SB_ERR_BINDING with nothing written. */
sb_status sb_llm_json_schema_to_gbnf(const char *schema_json,
                                     const char *prefix, const char *suffix,
                                     char *out, size_t cap, size_t *len_out,
                                     sb_error **err_out);
sb_status sb_llm_session_set_grammar(sb_llm_session *session, const char *gbnf,
                                     sb_error **err_out);
sb_status sb_llm_session_set_json_schema(sb_llm_session *session,
                                         const char *schema_json,
                                         sb_error **err_out);
void sb_llm_session_clear_grammar(sb_llm_session *session);
sb_status sb_llm_session_constraint_state(const sb_llm_session *session,
                                          int *active_out, int *complete_out,
                                          sb_error **err_out);

/* --- Image generation ------------------------------------------------------
 *
 * MOSS-Image text-to-image through the same stable ABI: an opaque model (the
 * pipeline with its text encoder), one request struct, a batch of RGB images
 * and a minimal PNG writer. The model directory is the MOSS-Image root
 * (text_encoder/, transformer/, vae/, tokenizer/); a directory the pipeline
 * rejects is SB_ERR_UNSUPPORTED.
 *
 * Determinism: image i starts from the same seeded noise sb-moss's `sample
 * --seed` draws (seed + i), and images in a batch are denoised one at a time,
 * so image i of a batch is bit-identical to generating it alone.
 */

typedef struct sb_image_model sb_image_model;

/* One generation request. Fields beyond `struct_size` take their defaults, so
 * a caller built against an older header keeps working. prompt is required
 * (NUL-terminated UTF-8, wrapped in the pipeline's chat template);
 * negative_prompt NULL is the empty prompt (used when guidance_scale != 1).
 * width and height are multiples of 16 (and >= 16); batch >= 1; steps >= 1;
 * guidance_scale is used as given (0 is unconditional-only; the reference
 * pipeline's default is 3.0), so a caller sets the guidance it wants. */
typedef struct sb_image_request {
  size_t struct_size;
  const char *prompt;
  const char *negative_prompt;
  uint32_t width;
  uint32_t height;
  uint32_t steps;
  float guidance_scale;
  uint64_t seed;
  uint32_t batch;
} sb_image_request;

/* Called after each denoising step with the image's index in the batch, the
 * steps completed (1-based) and the total. A non-zero return cancels: the
 * generation stops after the current step and sb_image_generate returns
 * SB_ERR_CANCELLED. */
typedef int (*sb_image_progress_cb)(void *user, uint32_t image_index,
                                    uint32_t step, uint32_t total_steps);

/* Load the MOSS-Image pipeline and its text encoder from `dir`. */
sb_status sb_image_model_load(const char *dir, sb_image_model **out,
                              sb_error **err_out);
void sb_image_model_free(sb_image_model *model);

/* The architecture name ("moss-image"), two-call sizing like the other string
 * outputs: a short cap returns SB_ERR_BINDING and the needed byte count. */
sb_status sb_image_model_arch(const sb_image_model *model, char *out,
                              size_t cap, size_t *len_out,
                              sb_error **err_out);

/* Generate `req->batch` images. rgb_out receives batch x height x width x 3
 * bytes, uint8 HWC, row-major, top row first, converted from the pipeline's
 * [-1, 1] CHW floats with clamp and lrintf. Two-call sizing: pass rgb_out NULL
 * (or a cap too small) to get the needed byte count in *bytes_out and
 * SB_ERR_BINDING. cb may be NULL. */
sb_status sb_image_generate(sb_image_model *model, const sb_image_request *req,
                            sb_image_progress_cb cb, void *user,
                            uint8_t *rgb_out, size_t cap, size_t *bytes_out,
                            sb_error **err_out);

/* Write `rgb` (height rows of width*3 bytes, RGB) as a PNG: 8-bit RGB, one
 * IDAT of stored (uncompressed) deflate blocks, no external dependency. */
sb_status sb_image_write_png(const char *path, const uint8_t *rgb,
                             uint32_t width, uint32_t height,
                             sb_error **err_out);

/* --- Decision registry (diagnostics) ---------------------------------------
 *
 * The runtime resolves every automatic choice -- reduction mode, spec probe,
 * MMQ/attention implementation and tiles, dense path, async prefetch, the
 * training switches -- once per device or static shape class, and records it
 * with its inputs, the candidates considered, the reason, the numerics class
 * and its declared resource costs (VRAM bytes, first-run tuning ms). These two
 * calls dump that registry for the current device (the per-device persisted
 * records plus what this process has resolved), as the same human table
 * tools/sb-explain prints or as a JSON document. No C++ type crosses the
 * boundary. Two-call sizing: pass buf NULL or cap too small to get the needed
 * byte count (including the NUL) in *len_out and SB_ERR_BINDING. */
sb_status sb_explain_table(char *buf, size_t cap, size_t *len_out,
                           sb_error **err_out);
sb_status sb_explain_json(char *buf, size_t cap, size_t *len_out,
                          sb_error **err_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SONICBOOM_CAPI_H */
