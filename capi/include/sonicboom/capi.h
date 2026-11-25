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

/* Version of this ABI surface, reported by boom_api_version(). It changes only
 * when an existing declaration changes meaning; additive symbols (new
 * functions, new structs) do not bump it. */
#define BOOM_API_VERSION 1u

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handles. Contents are private to the implementation. */
typedef struct boom_document boom_document;      /* a parsed S-Expr v0.1 document  */
typedef struct boom_executable boom_executable;  /* a compiled, runnable graph     */
typedef struct boom_error boom_error;            /* a failure detail               */
typedef struct boom_tensor boom_tensor;          /* a float32 tensor (autograd)     */
typedef struct boom_tape boom_tape;              /* a differentiable-op tape        */
typedef struct boom_param boom_param;            /* a trainable param + AdamW state */
typedef struct boom_storage boom_storage;        /* a borrowed, read-only view     */
typedef struct boom_weights boom_weights;        /* a safetensors/GGUF weight store */

/* Status codes. 0 == success; every non-zero value is a failure. */
typedef enum boom_status {
  BOOM_OK = 0,
  BOOM_ERR_PARSE = 1,    /* the S-Expr text was malformed or unsupported        */
  BOOM_ERR_WEIGHT = 2,   /* a weight sidecar could not be read/resolved         */
  BOOM_ERR_COMPILE = 3,  /* lowering / pass pipeline / JIT compilation failed   */
  BOOM_ERR_BINDING = 4,  /* input/output buffer size or type mismatch           */
  BOOM_ERR_RUNTIME = 5,  /* the JITted function failed or was unavailable       */
  BOOM_ERR_INTERNAL = 6, /* invalid argument to the C API, or unexpected error  */
  BOOM_ERR_UNSUPPORTED = 7, /* a requested feature this build/backend lacks     */
  BOOM_ERR_CANCELLED = 8, /* the caller's progress callback asked to stop      */
} boom_status;

/* Error categories (the `message` carries the human-readable detail). */
typedef enum boom_error_kind {
  BOOM_ERRKIND_NONE = 0,
  BOOM_ERRKIND_PARSE_LEXICAL = 1,
  BOOM_ERRKIND_PARSE_SYNTAX = 2,
  BOOM_ERRKIND_PARSE_SEMANTIC = 3,
  BOOM_ERRKIND_PARSE_UNSUPPORTED = 4,
  BOOM_ERRKIND_WEIGHT = 5,
  BOOM_ERRKIND_COMPILE = 6,
  BOOM_ERRKIND_BINDING = 7,
  BOOM_ERRKIND_RUNTIME = 8,
  BOOM_ERRKIND_INTERNAL = 9,
  BOOM_ERRKIND_CANCELLED = 10,
} boom_error_kind;

/* The frozen v0 dtype set. */
typedef enum boom_dtype {
  BOOM_DTYPE_FLOAT32 = 0,
  BOOM_DTYPE_FLOAT16 = 1,
  BOOM_DTYPE_BFLOAT16 = 2,
  BOOM_DTYPE_FLOAT64 = 3,
  BOOM_DTYPE_INT8 = 4,
  BOOM_DTYPE_UINT8 = 5,
  BOOM_DTYPE_INT16 = 6,
  BOOM_DTYPE_INT32 = 7,
  BOOM_DTYPE_INT64 = 8,
  BOOM_DTYPE_BOOL = 9,
} boom_dtype;

/* Physical element layout of a tensor. Only BOOM_QUANT_NONE is produced today;
 * the values are reserved so a later quantized view keeps this enum stable. */
typedef enum boom_quant_format {
  BOOM_QUANT_NONE = 0,
  BOOM_QUANT_Q8_1 = 1,
  BOOM_QUANT_Q5_0 = 2,
  BOOM_QUANT_Q8_0 = 3,
  BOOM_QUANT_Q3_K = 4,
  BOOM_QUANT_Q4_K = 5,
  BOOM_QUANT_Q5_K = 6,
  BOOM_QUANT_Q6_K = 7,
} boom_quant_format;

typedef enum boom_tensor_flags {
  BOOM_TF_NONE = 0,
  BOOM_TF_READ_ONLY = 1u << 0,
  BOOM_TF_DEVICE = 1u << 1,
  BOOM_TF_QUANTIZED = 1u << 2,
  BOOM_TF_EXTERNAL = 1u << 3,
} boom_tensor_flags;

/* A plain-C tensor description, versioned by struct_size (set it to
 * sizeof(boom_tensor_desc) before the call). `dims` and `name` are borrowed from
 * the owner and are valid only while the owner is alive: the executable handle
 * for boom_executable_parameter_desc, the view for boom_storage_desc. A library
 * writes the fields it knows and then stores sizeof(boom_tensor_desc) back into
 * struct_size, so a caller compiled against a later, larger header can tell
 * which fields it may read. dtype is the logical dtype; num_bytes is the
 * physical byte count (equal to numel * element size for dense tensors). */
typedef struct boom_tensor_desc {
  size_t struct_size;
  boom_dtype dtype;
  int64_t rank;
  const int64_t *dims;
  uint64_t num_bytes;
  uint32_t flags;        /* boom_tensor_flags */
  boom_quant_format quant;
  uint32_t quant_block;  /* elements per block; 0 when dense */
  const char *name;      /* may be NULL */
} boom_tensor_desc;

/* --- Error inspection ------------------------------------------------------ */

/* Error handles are owned by the caller. On failure, a status-returning call
 * writes a fresh error handle to *err_out *only if* err_out is non-NULL; the
 * caller must release it with boom_error_free. Ownership discipline: if the same
 * error-slot variable is reused across calls without freeing the previous
 * handle in between, the implementation frees the previous handle before
 * writing a new one, so reuse never leaks. A slot that has never held an error
 * (or was NULLed after a free) is simply overwritten; the implementation never
 * dereferences or frees an uninitialized or caller-owned pointer. The caller
 * must still free the last handle written to a slot.
 *
 * Exception containment: no C++ exception escapes any exported boom_* function.
 * Every status-returning function is wrapped so that an expected failure (the
 * normal error kinds above) and any unexpected exception (including bad_alloc)
 * are both mapped to a status code plus, when err_out is non-NULL, an error
 * handle. Unexpected exceptions surface as BOOM_ERR_INTERNAL.
 *
 * Limitation: handling is not fully reliable if error-object construction
 * itself fails. When allocation of the error handle or its message fails under
 * memory pressure, the slot is left NULL and the failure is still reported by
 * the returned status code, but there is no diagnostic message to retrieve. */

/* NUL-terminated message; valid until boom_error_free(). Never NULL ("" if the
 * handle is NULL). */
const char *boom_error_message(const boom_error *err);
boom_error_kind boom_error_get_kind(const boom_error *err);
void boom_error_free(boom_error *err);

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
 * boom_build_info still reports none/"none" only in a build with no MLIR, which
 * configure now rejects.)
 * `platform` is the target CPU the library was compiled for, not the builder's. */
const char *boom_build_info(void);

/* --- Parse ----------------------------------------------------------------- */

/* Parse an S-Expr v0.1 document. `text` need not be NUL-terminated; `len` is
 * its byte length. On success *doc_out receives a new handle (release with
 * boom_document_free). On failure *doc_out is set to NULL and, if `err_out` is
 * non-NULL, *err_out receives a new error handle (release with boom_error_free). */
boom_status boom_parse(const char *text, size_t len, boom_document **doc_out,
                   boom_error **err_out);

/* --- Compile --------------------------------------------------------------- */

/* Compile a parsed document for CPU execution. `base_dir` resolves relative
 * weight-sidecar paths (pass "" to resolve against the process working
 * directory). On success *exe_out receives a new handle (release with
 * boom_executable_free). */
boom_status boom_compile(const boom_document *doc, const char *base_dir,
                     boom_executable **exe_out, boom_error **err_out);

/* Compile a parsed document as a device training graph. The document is the
 * forward+backward+step graph a traced training function produced: it is
 * lowered Blade -> Boom IR, run through the Boom IR Training pass phase, and
 * executed node by node on the GPU (CUDA). The graph is bound by name — the
 * tracer names every tensor argument — through boom_input_name_at /
 * boom_output_name_at, and a step runs through boom_run_multi. Parameters must be
 * inline (`data :values`); external weight sidecars are not supported. On a
 * build without CUDA this fails with BOOM_ERR_COMPILE and a message saying so;
 * the graph is still lowered and verified first, so a bad graph is reported as
 * such on either build. Release with boom_executable_free. */
boom_status boom_compile_device(const boom_document *doc, boom_executable **exe_out,
                            boom_error **err_out);

/* Compile a parsed document as a device graph in either mode. `inference` = 0
 * is exactly boom_compile_device (training); `inference` = 1 compiles a
 * forward-only graph (the TTS forward mode): it runs the Boom IR Forward pass
 * phase (dead-node elimination and buffer reuse), takes CUDA-graph capture on
 * by default, and holds its weights externally. An inference graph declares
 * each weight as a `(data :bound "<name>")` constant parameter; compile records
 * the name, dtype and shape and uploads nothing. Every weight is bound from a
 * weight store with boom_executable_bind_weights before the first run, and a
 * run with one unbound fails with BOOM_ERR_BINDING naming it. */
boom_status boom_compile_device_ex(const boom_document *doc, int inference,
                              boom_executable **exe_out, boom_error **err_out);

/* --- Weight store (inference) ---------------------------------------------- */

/* File format of a weight store. AUTO detects GGUF from its magic and reads
 * anything else as safetensors; an explicit format that the file is not is an
 * error. */
typedef enum boom_weight_format {
  BOOM_WEIGHT_FORMAT_AUTO = 0,
  BOOM_WEIGHT_FORMAT_SAFETENSORS = 1,
  BOOM_WEIGHT_FORMAT_GGUF = 2,
} boom_weight_format;

/* Open a weight store: a safetensors or GGUF file mapped once, whose named
 * tensors are uploaded to the device as float32 on first use and shared by
 * every executable bound to the store. `convert_to_f32` converts f16, bf16 and
 * f64 entries on the host at upload; f32 entries are used as stored, and a
 * quantized GGUF entry (which the f32 graph cannot read) is reported by name
 * and type when it is requested. On success *weights_out receives a new handle
 * (release with boom_weights_free). */
boom_status boom_weights_open(const char *path, boom_weight_format format,
                              int convert_to_f32, boom_weights **weights_out,
                              boom_error **err_out);

/* Release a store handle. An executable bound to the store keeps the store
 * itself — and therefore the device buffers it bound — alive regardless of the
 * order of the two frees. NULL is safe. */
void boom_weights_free(boom_weights *weights);

/* A borrowed host copy of one stored weight as float32: the file's own bytes
 * when it is f32, a conversion cached in the store otherwise (so it honors the
 * store's convert_to_f32). The pointer stays valid until the store is freed —
 * it is the same bytes a device bind uploads. This is how a CPU fallback feeds
 * one file's weights to the interpreter. An absent name or a stored type the
 * float32 form cannot represent is BOOM_ERR_WEIGHT. */
boom_status boom_weights_data_f32(boom_weights *weights, const char *name,
                                  const void **data_out, uint64_t *bytes_out,
                                  boom_error **err_out);

/* The stored dtype name, shape and byte count of one entry, before any
 * conversion: DTYPE_NAME_OUT receives the file's own name ("F32", "F16",
 * "BF16", "F64", or a GGUF type such as "Q4_K") and DIMS_OUT a borrowed
 * dimension array. Valid until the store is freed. This is what a caller
 * checks a graph's declared weight against before binding it, so a shape or
 * type mismatch is reported with the store's own words (the device bind names
 * both shapes) rather than as a byte-count difference. An absent name is
 * BOOM_ERR_WEIGHT. Any output pointer may be NULL if that field is not
 * needed. */
boom_status boom_weights_tensor_info(const boom_weights *weights, const char *name,
                                     const char **dtype_name_out, int64_t *rank_out,
                                     const int64_t **dims_out,
                                     uint64_t *num_bytes_out, boom_error **err_out);

/* Bind every external weight of an inference executable from `weights`, by
 * name, uploading each on first use. The executable keeps the store alive
 * (see above), and binding again rebinds. A weight the store lacks, whose
 * shape differs (both shapes are named), or whose stored type cannot be
 * supplied is BOOM_ERR_BINDING. A training graph has no external weights and
 * binds vacuously. */
boom_status boom_executable_bind_weights(boom_executable *exe,
                                         boom_weights *weights,
                                         boom_error **err_out);

/* Turn CUDA-graph capture of the node launches on (1) or off (0). Inference
 * graphs default it on; training graphs default from SONICBOOM_TRAIN_GRAPH.
 * Turning capture off releases a captured graph; turning it on lets the next
 * run capture. The captured launch sequence is invalidated by rebinding a
 * weight. A CPU executable has no node launches and fails with
 * BOOM_ERR_UNSUPPORTED. */
boom_status boom_executable_set_capture(boom_executable *exe, int on,
                                        boom_error **err_out);

/* Device memory an executable holds, in bytes, by what it is for. */
typedef struct boom_memory_stats {
  uint64_t activation_bytes; /* activation slots (the buffer-reuse assignment) */
  uint64_t weight_bytes;     /* bound external weights (owned by the store)   */
  uint64_t scratch_bytes;    /* compile-time scratch                          */
  uint64_t total_bytes;      /* their sum                                     */
} boom_memory_stats;

/* Fill `*stats_out` for a device executable (training or inference). A CPU
 * executable (boom_compile) reports zero for every field. */
boom_status boom_executable_memory_stats(const boom_executable *exe,
                                         boom_memory_stats *stats_out,
                                         boom_error **err_out);

/* --- I/O introspection (bind) ---------------------------------------------- */

/* Query the single input/output tensor. `dims_out`, if non-NULL, is set to a
 * borrowed pointer to the dimension array (valid until the executable is
 * freed). Any output pointer may be NULL if that field is not needed.
 * v0 execution scope is exactly one input and one output tensor. */
boom_status boom_input_info(const boom_executable *exe, boom_dtype *dtype_out,
                        int64_t *rank_out, const int64_t **dims_out,
                        uint64_t *num_bytes_out);
boom_status boom_output_info(const boom_executable *exe, boom_dtype *dtype_out,
                         int64_t *rank_out, const int64_t **dims_out,
                         uint64_t *num_bytes_out);

/* Graph arity and per-index introspection. A graph may have several inputs and
 * several outputs (a training graph typically returns one gradient per
 * parameter). boom_input_info/boom_output_info describe index 0; the *_info_at
 * forms describe any index in [0, count). An index out of range is
 * BOOM_ERR_INTERNAL (invalid argument). */
boom_status boom_input_count(const boom_executable *exe, size_t *count_out);
boom_status boom_output_count(const boom_executable *exe, size_t *count_out);
boom_status boom_input_info_at(const boom_executable *exe, size_t index,
                           boom_dtype *dtype_out, int64_t *rank_out,
                           const int64_t **dims_out, uint64_t *num_bytes_out);
boom_status boom_output_info_at(const boom_executable *exe, size_t index,
                            boom_dtype *dtype_out, int64_t *rank_out,
                            const int64_t **dims_out,
                            uint64_t *num_bytes_out);

/* Tensor names in binding order. `*name_out` is a borrowed NUL-terminated
 * string valid until the executable is freed, or NULL when the compiled form
 * does not retain names (the CPU interpreter compiled by boom_compile). A
 * device training graph always names its tensors: the tracer's names. */
boom_status boom_input_name_at(const boom_executable *exe, size_t index,
                           const char **name_out);
boom_status boom_output_name_at(const boom_executable *exe, size_t index,
                            const char **name_out);

/* --- Execute / retrieve ---------------------------------------------------- */

/* Bind the input tensor buffer. `data` is copied, so the caller may free it
 * immediately after this returns. `len` must equal the input's num_bytes. */
boom_status boom_bind_input(boom_executable *exe, const void *data, size_t len,
                        boom_error **err_out);

/* Run the bound graph. The output is retained internally until the next
 * boom_execute or until the executable is freed. */
boom_status boom_execute(boom_executable *exe, boom_error **err_out);

/* Copy the retained output into `out` (capacity `cap` bytes). On success
 * `*written_out` is the output byte count (== the output num_bytes). */
boom_status boom_retrieve_output(boom_executable *exe, void *out, size_t cap,
                             size_t *written_out, boom_error **err_out);

/* Run the graph over N inputs and M outputs in one call — the general form
 * behind boom_bind_input/boom_execute/boom_retrieve_output (the 1-in/1-out wrapper).
 * `num_inputs` must equal boom_input_count and `num_outputs` must equal
 * boom_output_count. Every input is copied, so the caller may free it as soon as
 * this returns; `input_lens[i]` must equal that input's num_bytes.
 * `output_data[i]` must have capacity at least the i-th output's num_bytes and
 * `output_caps[i]` carries that capacity; capacities are checked before the
 * graph runs. On success each result is copied into `output_data[i]` and, when
 * `output_lens` is non-NULL, `output_lens[i]` receives the byte count written.
 * On failure no result buffer is modified. */
boom_status boom_run_multi(boom_executable *exe, const void *const *input_data,
                       const size_t *input_lens, size_t num_inputs,
                       void *const *output_data, const size_t *output_caps,
                       size_t num_outputs, size_t *output_lens,
                       boom_error **err_out);

/* Resident training step. A device training graph keeps the parameters and
 * optimizer state the optimizer updates in their device buffers across steps
 * (see boom_input_is_resident). The first successful run — boom_run_multi or
 * boom_run_resident — uploads them; afterwards `input_data[i]` for a resident
 * input may be NULL and its bytes are ignored. Every other input is uploaded on
 * every call, and `input_lens[i]` is still validated for all of them.
 *
 * Only the outputs named by `output_indices[0..num_outputs)` are copied back,
 * in that order: a step that keeps its state on the device downloads the loss,
 * not the parameters. `num_outputs` is therefore independent of
 * boom_output_count. On failure no result buffer is modified. */
boom_status boom_run_resident(boom_executable *exe, const void *const *input_data,
                          const size_t *input_lens, size_t num_inputs,
                          const size_t *output_indices, size_t num_outputs,
                          void *const *output_data, const size_t *output_caps,
                          size_t *output_lens, boom_error **err_out);

/* 1 when input `index` is resident in the sense above (the graph writes its
 * next value back into the same device buffer), 0 otherwise. Fails with
 * BOOM_ERR_BINDING for a non-device executable or an index out of range. */
boom_status boom_input_is_resident(const boom_executable *exe, size_t index,
                               int *resident_out, boom_error **err_out);

/* --- Parameters and read-only storage views -------------------------------- */
/* A compiled CPU graph materializes every parameter (a graph constant: inline
 * `:values` data or an `:external` sidecar) into its own buffers at compile
 * time. A run never writes those buffers, so they may be borrowed read-only.
 * A view is an opaque boom_storage handle that pins the executable
 * implementation independently of the public boom_executable handle:
 *
 *   - the view stays valid after boom_executable_free (and after a cache
 *     eviction that frees the handle);
 *   - several views may pin the same executable independently;
 *   - boom_storage_release drops this view's reference; the memory is freed when
 *     the last handle (executable or view) is gone. Releasing NULL is safe;
 *     release each non-NULL view exactly once.
 *
 * Device training graphs (boom_compile_device) have no host parameter block:
 * their parameter queries fail with BOOM_ERR_UNSUPPORTED, and no raw device
 * pointer ever crosses this ABI. */

/* The ABI version implemented by this library (BOOM_API_VERSION of the header it
 * was built from). */
uint32_t boom_api_version(void);

/* Number of parameters (graph constants) in `exe`, in graph tensor order. The
 * same parameter name used by several constants appears once per constant. */
boom_status boom_executable_parameter_count(const boom_executable *exe,
                                        size_t *count_out,
                                        boom_error **err_out);

/* Parameter `index` in [0, count): its name (a borrowed NUL-terminated string)
 * or its descriptor. Both are borrowed from the executable and valid until
 * boom_executable_free. An index out of range is BOOM_ERR_INTERNAL. */
boom_status boom_executable_parameter_name_at(const boom_executable *exe,
                                          size_t index,
                                          const char **name_out,
                                          boom_error **err_out);
boom_status boom_executable_parameter_desc(const boom_executable *exe, size_t index,
                                       boom_tensor_desc *desc_out,
                                       boom_error **err_out);

/* Acquire a read-only host view of parameter `index`. On success *view_out
 * receives a new handle (release with boom_storage_release). The view keeps the
 * executable implementation alive: the memory stays valid after
 * boom_executable_free. View flags include BOOM_TF_READ_ONLY, and the byte count
 * is the parameter's num_bytes. */
boom_status boom_executable_parameter_view(const boom_executable *exe, size_t index,
                                       boom_storage **view_out,
                                       boom_error **err_out);

/* Host address and byte count of a view. `data_out` and `bytes_out` may each be
 * NULL. The address is borrowed: the view pins it, so it stays valid until the
 * view is released, and must not be written. */
boom_status boom_storage_data(const boom_storage *view, const void **data_out,
                          uint64_t *bytes_out, boom_error **err_out);

/* Descriptor of a view's tensor, as boom_executable_parameter_desc. dims/name
 * stay valid while the view is alive. */
boom_status boom_storage_desc(const boom_storage *view, boom_tensor_desc *desc_out,
                          boom_error **err_out);

/* Drop a view's ownership reference. NULL is safe. */
void boom_storage_release(boom_storage *view);

/* --- Bound parameters ------------------------------------------------------ */

/* A graph parameter may be declared `(data :bound "<name>")` instead of
 * carrying inline `:values` bytes or an `:external` sidecar. Compile allocates
 * the constant's slot and records its byte count, but loads nothing: the caller
 * must fill every bound parameter with boom_bind_parameter before the first run.
 * Until then boom_execute, boom_run_multi and boom_run_resident fail with
 * BOOM_ERR_BINDING. Binding copies `len` bytes (the caller may free `data` on
 * return, and later writes through it are not observed); `len` must equal the
 * parameter's num_bytes. A parameter may be rebound before the first
 * successful run, replacing the copied bytes; afterwards binding fails with
 * BOOM_ERR_BINDING. An unknown name or wrong size is BOOM_ERR_BINDING; a NULL
 * argument is BOOM_ERR_INTERNAL. Device training graphs (boom_compile_device)
 * have no bound parameters and fail with BOOM_ERR_UNSUPPORTED. */
boom_status boom_bind_parameter(boom_executable *exe, const char *name,
                            const void *data, size_t len, boom_error **err_out);

/* 1 when every `(data :bound ...)` parameter of `exe` has been filled, 0
 * otherwise. A graph without bound parameters (and a device training graph,
 * which has no host parameter block) is complete. */
boom_status boom_bindings_complete(const boom_executable *exe, int *complete_out,
                               boom_error **err_out);

/* --- Release --------------------------------------------------------------- */

void boom_document_free(boom_document *doc);
void boom_executable_free(boom_executable *exe);

/* --- Training (minimal autograd) ------------------------------------------- */
/* A float32-only reverse-mode autograd surface for a small MLP: linear/relu/
 * mse forward ops recorded on a tape, backward (VJP replay), and an in-place
 * SGD step. Guile (or another binding) drives the training loop; all numeric
 * work happens in native C++ behind these handles. No tensor is exposed as a
 * raw pointer; data crosses the boundary only as little-endian byte buffers. */

/* Create a float32 tensor of the given shape (dims[0..rank)) from little-endian
 * bytes. `len` must equal numel * 4. The handle owns the native tensor. */
boom_status boom_tensor_from_f32(const int64_t *dims, int64_t rank, const void *data,
                             size_t len, boom_tensor **out, boom_error **err_out);

/* Copy a tensor's little-endian float32 bytes into `out` (capacity `cap`). On
 * success *written_out is the byte count (== numel * 4). */
boom_status boom_tensor_bytes(const boom_tensor *t, void *out, size_t cap,
                          size_t *written_out, boom_error **err_out);

void boom_tensor_free(boom_tensor *t);

/* --- Training: parameter + fused AdamW -------------------------------------- */
/* A trainable float32 parameter owns its values, its gradient, and AdamW's two
 * moment buffers. When a CUDA device is present the four buffers stay in device
 * memory across steps and the update is one fused kernel launch; otherwise they
 * are host buffers and the host reference runs. The handle hides which.
 *
 * `dims[0..rank)` (row-major) only determine the element count; `len` must
 * equal numel * 4 and supplies the initial values. The gradient and both moment
 * buffers start at zero. */

boom_status boom_param_create(const int64_t *dims, int64_t rank, const void *init,
                          size_t len, boom_param **out, boom_error **err_out);

/* Element count, and whether the buffers are device-resident (1) or host
 * (0). Both are always available once the handle exists. */
boom_status boom_param_numel(const boom_param *p, size_t *numel_out);
int boom_param_on_device(const boom_param *p);

/* Updates applied so far (0 before the first successful boom_param_step). */
uint64_t boom_param_step_count(const boom_param *p);

/* Zero the gradient, replace it with `grad` (len == numel * 4), and apply one
 * AdamW update. `boom_param_read` copies the current parameter values out. */
boom_status boom_param_zero_grad(boom_param *p, boom_error **err_out);
boom_status boom_param_set_grad(boom_param *p, const void *grad, size_t len,
                            boom_error **err_out);
boom_status boom_param_step(boom_param *p, float lr, float beta1, float beta2,
                        float eps, float weight_decay, boom_error **err_out);

/* The lr-input form: the learning rate is read from a one-element float32
 * tensor (an boom_tensor handle, which is how a traced schedule carries the
 * rate). The parameter's own buffers stay device-resident; only the single
 * rate is read on the host. The handle's numel must be 1 and its dtype
 * float32. */
boom_status boom_param_step_lr(boom_param *p, const boom_tensor *lr, float beta1,
                           float beta2, float eps, float weight_decay,
                           boom_error **err_out);
boom_status boom_param_read(const boom_param *p, void *out, size_t cap,
                        size_t *written_out, boom_error **err_out);

void boom_param_free(boom_param *p);

/* Create an empty tape (no recorded ops, no gradients). */
boom_status boom_tape_new(boom_tape **out, boom_error **err_out);

void boom_tape_free(boom_tape *tape);

/* Clear the tape's recorded ops and accumulated gradients. */
void boom_tape_zero_grad(boom_tape *tape);

/* Forward ops: each returns a new tensor handle and records a node on `tape`.
 * linear: y = x @ Wᵀ + b (x [N,in], W [out,in], b [out] → [N,out]). */
boom_status boom_linear(const boom_tensor *x, const boom_tensor *w, const boom_tensor *b,
                    boom_tape *tape, boom_tensor **out, boom_error **err_out);

/* relu: y = max(x, 0), elementwise. */
boom_status boom_relu(const boom_tensor *x, boom_tape *tape, boom_tensor **out,
                  boom_error **err_out);

/* mse_loss: scalar = mean((pred - target)²); returns a 0-dim tensor. */
boom_status boom_mse_loss(const boom_tensor *pred, const boom_tensor *target,
                      boom_tape *tape, boom_tensor **out, boom_error **err_out);

/* Reverse-mode backprop from `loss` (seeds dL/dL = 1), accumulating gradients
 * into `tape`. */
boom_status boom_backward(const boom_tensor *loss, boom_tape *tape, boom_error **err_out);

/* In-place SGD update param <- param - lr * grad, using `param`'s accumulated
 * gradient in `tape`. The parameter keeps its identity across calls. */
boom_status boom_sgd_step(boom_tape *tape, boom_tensor *param, double lr,
                      boom_error **err_out);

/* Read `param`'s accumulated gradient as a new tensor handle. Returns
 * BOOM_ERR_RUNTIME if no gradient has been accumulated for `param`. */
boom_status boom_grad(const boom_tape *tape, const boom_tensor *param, boom_tensor **out,
                  boom_error **err_out);

/* Clip a set of gradient tensors by their global L2 norm, in place. GRADS is
 * COUNT float32 CPU tensors (the gradients read back with boom_grad, or any
 * other gradient buffers); the norm is accumulated in double over every
 * element of every tensor. When it exceeds MAX_NORM each tensor is scaled by
 * MAX_NORM / (norm + 1e-6); otherwise the tensors are left byte-for-byte
 * unchanged. *NORM_OUT, when non-NULL, receives the pre-clip global norm. */
boom_status boom_clip_grad_norm(boom_tensor *const *grads, int count, float max_norm,
                            float *norm_out, boom_error **err_out);

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
 * whose byte count must equal the declared shape (BOOM_ERR_INTERNAL otherwise).
 * Parameters not listed keep their existing inline/external data. `dir` is
 * created if needed. The artifact is loadable later with boom_compile using
 * `dir` as base_dir, with no Guile or Python at load time. */
boom_status boom_export_model(const char *sx_text, size_t sx_len,
                          const char *const *names, const boom_tensor *const *params,
                          size_t count, const char *dir, boom_error **err_out);

/* --- Checkpoints ------------------------------------------------------------- */

/* Named float32 tensors persisted in one self-describing binary file:
 *
 *   "SBCKPT1\n"                    8-byte magic
 *   u64                            entry count
 *   per entry:
 *     u32 name_len, name bytes     UTF-8, not NUL-terminated
 *     u8 dtype                     boom_dtype (float32 in this version)
 *     u32 rank
 *     i64 dims[rank]               row-major shape
 *     u64 byte_count               == numel * 4
 *     byte_count data bytes        little-endian float32, row-major
 *
 * All integers are little-endian. Entries are matched by name, so the order in
 * which a caller saves and loads them need not agree.
 *
 * boom_save_params writes `count` named tensors to `path` (overwriting it).
 * Names must be non-empty and unique; tensors must be float32. boom_load_params
 * copies the stored values into the caller's existing tensors, matched by
 * name: each target must be float32 with exactly the stored shape. The file is
 * fully parsed and validated before any target is written, so on any failure
 * every target is left untouched. The file may hold entries the caller does
 * not load (a checkpoint can carry more state than one call needs), and one
 * call may load a subset of its targets from several checkpoints.
 *
 * Errors: a missing, unreadable, truncated or malformed checkpoint is
 * BOOM_ERR_WEIGHT; a name/shape/dtype mismatch between file and caller is
 * BOOM_ERR_BINDING (also on save for duplicate names or a non-float32 tensor);
 * an invalid argument is BOOM_ERR_INTERNAL. */
boom_status boom_save_params(const char *path, const char *const *names,
                         const boom_tensor *const *params, size_t count,
                         boom_error **err_out);
boom_status boom_load_params(const char *path, const char *const *names,
                         boom_tensor *const *params, size_t count,
                         boom_error **err_out);

/* --- Numerics profile ------------------------------------------------------ */

/* The two numerics profiles (see core/include/sonicboom/numerics.h):
 *
 *   BOOM_NUMERICS_STRICT   -- the v0/v1 reference semantics. Every reduction is
 *                           identical to the historical binary; the KV cache
 *                           keeps f32; no TF32/DF32 substitutions.
 *   BOOM_NUMERICS_STANDARD -- the performance profile. Bit-identical to strict
 *                           unless a substitution is enabled for it; today
 *                           that is f16 KV storage (exact for the values the
 *                           cache holds) and the profile's other candidates.
 *
 * The profile is process-wide and latched: the first inference or explicit
 * set wins, because it is part of the plan, autotune and codegen cache keys.
 * Until it is set the process uses the SONICBOOM_NUMERICS environment variable
 * ("strict" or "standard"), then BOOM_NUMERICS_STANDARD.
 *
 * boom_get_numerics_profile returns the effective profile (the latched value, or
 * the environment/default value before the first latch). */
typedef enum boom_numerics_profile {
  BOOM_NUMERICS_STRICT = 0,
  BOOM_NUMERICS_STANDARD = 1,
} boom_numerics_profile;

/* Request a profile for this process. Returns BOOM_ERR_UNSUPPORTED (with a
 * message naming both profiles) if a different profile is already latched;
 * setting the already-effective profile is a no-op success. Call before the
 * first model load or inference for the setting to apply everywhere. */
boom_status boom_set_numerics_profile(boom_numerics_profile profile,
                                  boom_error **err_out);
boom_status boom_get_numerics_profile(boom_numerics_profile *profile_out,
                                  boom_error **err_out);

/* --- Inference (minimal LLM surface, boom_llm_*) ----------------------------- */
/* A thin facade over the Layer 2 model loader + planner spine, exposing the
 * smallest inference set the first binding (Guile) needs: model load/free,
 * session new/free, tokenize/detokenize, prefill, and decode with a per-token
 * callback. The scope is intentionally minimal (v1 wrap-up):
 *  - on the device path, after boom_llm_prefill decode continues from the
 *    session's persistent KV cache: each call emits up to max_tokens from where
 *    the previous one stopped, so a max_tokens=1 call is a per-token step. (The
 *    CPU fallback re-runs from the prompt on every call and does not carry
 *    generated tokens across calls.) A single-token prompt is a seed and
 *    decodes directly;
 *  - temperature 0 is greedy (the historical path); temperature > 0 samples
 *    deterministically from boom_llm_sampling's seed, and the sampled draw state
 *    persists across calls on the same session;
 *  - tokenize/detokenize use the model's GGUF built-in vocabulary (SPM-style
 *    BPE for Gemma 4, byte-level BPE for Qwen2), matching llama-tokenize;
 *  - the architectures SonicBoom models load (Gemma 4, Qwen2); anything else
 *    returns BOOM_ERR_UNSUPPORTED. Which one loaded is readable with
 *    boom_llm_model_arch, and the model's own stop tokens with boom_llm_model_eos —
 *    a caller that formats prompts per family takes both from the model rather
 *    than assuming.
 * These gaps are recorded in the post-v1 list. */

typedef struct boom_llm_model boom_llm_model;
typedef struct boom_llm_session boom_llm_session;
typedef int32_t boom_token;   /* a vocabulary index */

/* Sampling parameters for one decode call.
 *
 * temperature 0 is the greedy argmax path, bit-identical to the historical
 * decode. temperature > 0 enables seeded sampling: the application order is
 *   1. repetition / presence / frequency penalties over the last
 *      repeat_last_n tokens of the history (prompt + tokens generated so far);
 *   2. the grammar mask (constrained decoding);
 *   3. the top 256 adjusted logits (ties toward the lower id) are kept:
 *      sampling draws only from those 256 candidates;
 *   4. a 30-logit candidate window;
 *   5. top_k truncation (top_k 0 disables it);
 *   6. min_p truncation (min_p 0 disables it);
 *   7. nucleus truncation when 0 < top_p < 1 (top_p 1 disables it);
 *   8. a seeded Gumbel-max draw keyed by (seed, absolute token position).
 * The 256-candidate limit is exact whenever top_k <= 256 or the nucleus lies
 * inside the top 256; the remaining tail is never selected. The same seed,
 * position and history therefore always produce the same token, independently
 * of KV-cache capacity or prefix reuse.
 *
 * max_tokens caps the output length.
 *
 * Fields appended after max_tokens are read only when struct_size covers them;
 * a caller built against an older struct_size gets the neutral defaults below.
 *   min_p              neutral 0    (keep every candidate above the window)
 *   repeat_penalty     neutral 1    (no repetition penalty; 0 = off, < 0 is
 *                                    BOOM_ERR_BINDING)
 *   repeat_last_n      neutral 64   (penalty window, tokens; 0 disables every
 *                                    penalty; on the CUDA path a value above
 *                                    1024 is BOOM_ERR_BINDING, the device
 *                                    penalty window's cap)
 *   presence_penalty   neutral 0    (subtracted once per repeated token)
 *   frequency_penalty  neutral 0    (subtracted per occurrence)
 * Zero means "off" throughout: a zero-initialized boom_llm_sampling whose only
 * fields are struct_size = sizeof(boom_llm_sampling) and max_tokens is plain
 * greedy (temperature 0, no truncation, no penalties).
 * For a token t appearing c times in the window: if repeat_penalty != 1,
 * l[t] = l[t] / repeat_penalty when l[t] > 0 else l[t] * repeat_penalty; then
 * l[t] -= frequency_penalty * c + presence_penalty. */
typedef struct boom_llm_sampling {
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
} boom_llm_sampling;

/* Per-token callback: called synchronously on the calling thread, once per
 * produced token, in generation order. Return 0 to continue, non-zero to stop
 * the decode early. `user` is passed through untouched. */
typedef int (*boom_llm_token_cb)(void *user, boom_token token);

/* Load a GGUF model for inference. The model owns its backing buffer; a
 * session keeps the model alive after boom_llm_model_free. */
boom_status boom_llm_model_load(const char *gguf_path, boom_llm_model **model_out,
                            boom_error **err_out);

/* boom_llm_model_load_ex flags.
 *
 * BOOM_LLM_LOAD_CANONICAL selects canonical numerics for the process: every
 * tuner whose pick changes arithmetic takes its documented reference default
 * and neither reads nor writes a tune record, so the same model and prompt
 * produce the same tokens on an empty and on a populated tune directory.
 * Numerics-neutral picks (tile grids, timing tables) keep tuning. It is the
 * programmatic form of SONICBOOM_TUNE=canonical and is sticky for the
 * process, like the numerics profile.
 *
 * BOOM_LLM_LOAD_CONTINUE_REPLAY makes a session with recurrent state continue
 * under the replay rule (resume only at a state checkpoint) instead of the
 * default live rule (continue the whole recorded session from its live state).
 * It is the programmatic form of SONICBOOM_REPRO_CONTINUE=replay and is sticky
 * for the process.
 *
 * Any other bit is BOOM_ERR_BINDING. */
#define BOOM_LLM_LOAD_CANONICAL 1u
#define BOOM_LLM_LOAD_CONTINUE_REPLAY 2u
boom_status boom_llm_model_load_ex(const char *gguf_path, uint32_t flags,
                               boom_llm_model **model_out, boom_error **err_out);

void boom_llm_model_free(boom_llm_model *model);

/* Return every device buffer the runtime is holding for reuse to the driver.
 *
 * A model's own memory goes back to the driver by itself when the model is
 * freed (boom_llm_model_free / boom_image_model_free, after its last session).
 * This call is for the leftovers: the free-list buffers a released session
 * left behind, which keep VRAM resident so that the next session on the same
 * model does not have to allocate again. Live buffers (a loaded model's
 * weights, a session's KV cache) are untouched, so a model can keep serving
 * across the call; it simply allocates on demand afterwards. A no-op without
 * a CUDA device. Cheap and safe to call repeatedly. */
void boom_device_trim(void);

/* The loaded model's architecture, exactly as its GGUF declares it in
 * `general.architecture` ("gemma4", "qwen2", ...). A caller that must format
 * prompts, pick turn markers or otherwise behave differently per family reads
 * this from the model instead of guessing from the file name. Two-call sizing
 * like the other string out-params; *len_out excludes the NUL. */
boom_status boom_llm_model_arch(const boom_llm_model *model, char *out, size_t cap,
                            size_t *len_out, boom_error **err_out);

/* The model's end-of-sequence token id(s) as its GGUF declares them
 * (`tokenizer.ggml.eos_token_id`, plus `tokenizer.ggml.eos_token_ids` when the
 * model lists several). *n_out receives the count; none declared gives 0.
 * Two-call sizing: pass out NULL or cap too small for the count. */
boom_status boom_llm_model_eos(const boom_llm_model *model, boom_token *out, size_t cap,
                           size_t *n_out, boom_error **err_out);

/* A JSON description of the loaded model, for introspection (a capability or
 * compatibility check, a status page, a benchmark record). The document is one
 * object:
 *
 *   arch              string  general.architecture ("gemma4", "qwen2", ...)
 *   params            number  weights: the summed element count of every
 *                             tensor, norms and tables included
 *   context_length    number  <arch>.context_length as the file declares it
 *                             (0 when the file declares none)
 *   vocab_size        number  entries in the loaded tokenizer vocabulary
 *   has_vision        bool    the file carries vision components (the
 *                             clip.has_vision_encoder flag, or a vision
 *                             tower's tensors)
 *   has_tool_template bool    tokenizer.chat_template mentions tools
 *   tensors           number  entries in the tensor directory
 *   quant             array   one object per tensor type, sorted by name:
 *                             {"type", "tensors", "weights", "bytes"} — how
 *                             the file's weight bytes are distributed over
 *                             quantizations (F32, Q4_K, IQ2_XXS, ...), so a
 *                             caller can tell what mixture a GGUF really is
 *
 * Two-call sizing: *len_out always receives the needed byte count including
 * the NUL; a NULL or short buffer answers BOOM_ERR_BINDING. */
boom_status boom_llm_model_info(const boom_llm_model *model, char *out, size_t cap,
                            size_t *len_out, boom_error **err_out);

/* Tokenize/detokenize with the model's GGUF built-in vocabulary (SPM-style BPE
 * for Gemma 4). tokenize encodes `text` (UTF-8, `len` bytes) into token ids
 * with <bos> prepended (llama-tokenize add_special); detokenize renders `tokens`
 * back to UTF-8 (NUL-terminated), stripping control tokens and undoing ▁/byte
 * escapes. Both use two-call sizing: pass `out` NULL or `cap` too small to get
 * the needed count in *n_out / *len_out (and BOOM_ERR_BINDING). */
boom_status boom_llm_tokenize(const boom_llm_model *model, const char *text, size_t len,
                          boom_token *out, size_t cap, size_t *n_out, boom_error **err_out);

/* boom_llm_tokenize_ex flags. BOOM_TOKENIZE_ADD_SPECIAL (prepend <bos>) is what
 * boom_llm_tokenize does; a caller that tokenizes a prompt piece by piece turns
 * it on for the first piece only, so <bos> is added exactly once. */
#define BOOM_TOKENIZE_PARSE_SPECIAL 1
#define BOOM_TOKENIZE_ADD_SPECIAL 2

/* boom_llm_tokenize with flags. With BOOM_TOKENIZE_PARSE_SPECIAL, a CONTROL /
 * USER_DEFINED piece of the vocabulary (e.g. Gemma 4's "<|turn>", "<turn|>")
 * occurring verbatim in `text` becomes that one token instead of being encoded
 * as text. This turns text into control structure, so it must only be used on
 * text the caller itself generated (a chat template's turn markers): a user
 * message containing "<turn|>" would otherwise become a real turn boundary
 * (prompt injection). boom_llm_tokenize is this with BOOM_TOKENIZE_ADD_SPECIAL
 * alone, byte for byte. */
boom_status boom_llm_tokenize_ex(const boom_llm_model *model, const char *text, size_t len,
                             int flags, boom_token *out, size_t cap,
                             size_t *n_out, boom_error **err_out);
boom_status boom_llm_detokenize(const boom_llm_model *model, const boom_token *tokens,
                            size_t n, char *out, size_t cap, size_t *len_out,
                            boom_error **err_out);

/* The bytes one token contributes to boom_llm_detokenize's output, before its
 * cross-token passes: empty for a control/unknown/unused token, one decoded
 * byte for a byte token, the ▁-unescaped or byte-level-decoded text otherwise.
 * A caller that renders tokens incrementally concatenates the pieces and applies
 * the passes itself (see boom_llm_vocab_flags); for any token list the result
 * is byte for byte boom_llm_detokenize's. Two-call sizing; *len_out excludes
 * the NUL. */
boom_status boom_llm_token_piece(const boom_llm_model *model, boom_token token,
                             char *out, size_t cap, size_t *len_out,
                             boom_error **err_out);

/* boom_llm_vocab_flags bits: the two passes boom_llm_detokenize applies over
 * the pieces a token list assembles.
 *
 * BOOM_VOCAB_CLEAN_SPACES: apply clean_up_tokenization_spaces (strip a space
 * before ?!., the space around a lone apostrophe, and the space in front of
 * contractions like 's/'re/'ve).
 * BOOM_VOCAB_ADD_SPACE_PREFIX: the first piece emitted loses one leading
 * space (remove_space). Control/unknown/unused tokens do not use it up. */
#define BOOM_VOCAB_CLEAN_SPACES 1
#define BOOM_VOCAB_ADD_SPACE_PREFIX 2
boom_status boom_llm_vocab_flags(const boom_llm_model *model, int *flags_out,
                             boom_error **err_out);

/* Create a session (a prompt/token history + n_ctx). The session holds a
 * reference to the model, so the model handle may be freed first.
 *
 * Calls on the same session from several threads are serialized; a session is
 * not a unit of parallelism. Use one session per concurrent conversation. A
 * call through a handle that has already been freed fails with BOOM_ERR_INTERNAL
 * ("session is freed") instead of touching freed memory. */
boom_status boom_llm_session_new(boom_llm_model *model, uint64_t n_ctx,
                             boom_llm_session **session_out, boom_error **err_out);
void boom_llm_session_free(boom_llm_session *session);

/* Set the session's prompt to tokens[0..n). *n_reused_out receives the length
 * of leading tokens reused from the previous prompt's KV state. The last prompt
 * token is always processed — its activation is what produces the next-token
 * logits — so the count is at most n-1, even when the whole prompt is already
 * resident. (The CPU decode path re-prefills rather than reusing KV slots; KV
 * reuse is the CUDA prefix-cache path.) */
boom_status boom_llm_prefill(boom_llm_session *session, const boom_token *tokens, size_t n,
                         uint64_t *n_reused_out, boom_error **err_out);

/* Decode up to sampling->max_tokens tokens from the session's prompt, invoking
 * cb for each (abort on a non-zero return). *n_out receives the count produced.
 * Requires a non-empty prompt (boom_llm_prefill first). */
boom_status boom_llm_decode(boom_llm_session *session, const boom_llm_sampling *sampling,
                        boom_llm_token_cb cb, void *user, uint32_t *n_out,
                        boom_error **err_out);

/* --- Constrained decoding (session grammar) --------------------------------
 *
 * A session may carry a byte-level grammar that every sampled token must keep
 * acceptable. `gbnf` is a GBNF document (the llama.cpp dialect: rule ::=
 * alternation, "literals" with the usual escapes, [byte classes], . * + ? and
 * {m,n}, # comments; the first rule is the start rule). The grammar consumes
 * the same bytes boom_llm_detokenize would emit for a token, so it constrains the
 * text the model produces, not token ids. The model's EOS ids (see
 * boom_llm_model_eos) are the grammar's stop tokens: they are not text, may be
 * sampled only where the grammar can end, and end the constrained output
 * without being emitted.
 *
 * boom_llm_session_set_json_schema compiles a JSON Schema (one JSON document) to
 * GBNF first; the supported subset is documented in
 * core/include/sonicboom/vocab/grammar.h, and an unsupported keyword is
 * rejected by name rather than silently ignored.
 *
 * A grammar starts at every boom_llm_prefill (a new turn) and applies to the
 * first token sampled by that prefill and to every boom_llm_decode step. Once a
 * stop token finishes it, boom_llm_decode produces no further tokens.
 *
 * boom_llm_session_constraint_state reports whether a grammar is installed
 * (*active_out) and whether the text accepted so far is already a complete
 * value of it (*complete_out). */
/* boom_llm_json_schema_to_gbnf is the compiler alone, as text: it lets a caller
 * compose a schema into a larger grammar instead of installing the schema as
 * the whole reply. PREFIX and SUFFIX (may be NULL) are literal UTF-8 text the
 * value must appear between; the root matches exactly that text, so a complete
 * value can only be continued by a stop token.
 * Two-call sizing like boom_llm_model_arch: *len_out always receives the length
 * (without NUL), and a capacity that cannot hold the text plus its NUL returns
 * BOOM_ERR_BINDING with nothing written. */
boom_status boom_llm_json_schema_to_gbnf(const char *schema_json,
                                     const char *prefix, const char *suffix,
                                     char *out, size_t cap, size_t *len_out,
                                     boom_error **err_out);
boom_status boom_llm_session_set_grammar(boom_llm_session *session, const char *gbnf,
                                     boom_error **err_out);
boom_status boom_llm_session_set_json_schema(boom_llm_session *session,
                                         const char *schema_json,
                                         boom_error **err_out);
void boom_llm_session_clear_grammar(boom_llm_session *session);
boom_status boom_llm_session_constraint_state(const boom_llm_session *session,
                                          int *active_out, int *complete_out,
                                          boom_error **err_out);

/* --- Image generation ------------------------------------------------------
 *
 * MOSS-Image text-to-image through the same stable ABI: an opaque model (the
 * pipeline with its text encoder), one request struct, a batch of RGB images
 * and a minimal PNG writer. The model directory is the MOSS-Image root
 * (text_encoder/, transformer/, vae/, tokenizer/); a directory the pipeline
 * rejects is BOOM_ERR_UNSUPPORTED.
 *
 * Determinism: image i starts from the same seeded noise sb-moss's `sample
 * --seed` draws (seed + i), and images in a batch are denoised one at a time,
 * so image i of a batch is bit-identical to generating it alone.
 */

typedef struct boom_image_model boom_image_model;

/* One generation request. Fields beyond `struct_size` take their defaults, so
 * a caller built against an older header keeps working. prompt is required
 * (NUL-terminated UTF-8, wrapped in the pipeline's chat template);
 * negative_prompt NULL is the empty prompt (used when guidance_scale != 1).
 * width and height are multiples of 16 (and >= 16); batch >= 1; steps >= 1;
 * guidance_scale is used as given (0 is unconditional-only; the reference
 * pipeline's default is 3.0), so a caller sets the guidance it wants. */
typedef struct boom_image_request {
  size_t struct_size;
  const char *prompt;
  const char *negative_prompt;
  uint32_t width;
  uint32_t height;
  uint32_t steps;
  float guidance_scale;
  uint64_t seed;
  uint32_t batch;
} boom_image_request;

/* Called after each denoising step with the image's index in the batch, the
 * steps completed (1-based) and the total. A non-zero return cancels: the
 * generation stops after the current step and boom_image_generate returns
 * BOOM_ERR_CANCELLED. */
typedef int (*boom_image_progress_cb)(void *user, uint32_t image_index,
                                    uint32_t step, uint32_t total_steps);

/* Load the MOSS-Image pipeline and its text encoder from `dir`. */
boom_status boom_image_model_load(const char *dir, boom_image_model **out,
                              boom_error **err_out);
void boom_image_model_free(boom_image_model *model);

/* The architecture name ("moss-image"), two-call sizing like the other string
 * outputs: a short cap returns BOOM_ERR_BINDING and the needed byte count. */
boom_status boom_image_model_arch(const boom_image_model *model, char *out,
                              size_t cap, size_t *len_out,
                              boom_error **err_out);

/* A JSON description of the loaded image model, the same shape as
 * boom_llm_model_info:
 *
 *   arch              string  "moss-image"
 *   params            number  weights across text_encoder, transformer and vae
 *   context_length    number  the text encoder's max_position_embeddings
 *   vocab_size        number  the tokenizer's vocabulary size
 *   has_vision        bool    true: the pipeline carries image components (the
 *                             VAE decoder; for an LLM the flag means it accepts
 *                             images as input)
 *   has_tool_template bool    false: the image pipeline has no chat tool use
 *   tensors           number  entries in the three component directories
 *   quant             array   tensor count, weight count and bytes per dtype
 *                             ("BF16", "F32", ...), sorted by name
 *   components        object  per component (text_encoder, transformer, vae):
 *                             the same tensors/weights/bytes counts
 *
 * Two-call sizing: *len_out always receives the needed byte count including
 * the NUL; a NULL or short buffer answers BOOM_ERR_BINDING. */
boom_status boom_image_model_info(const boom_image_model *model, char *out,
                              size_t cap, size_t *len_out,
                              boom_error **err_out);

/* Generate `req->batch` images. rgb_out receives batch x height x width x 3
 * bytes, uint8 HWC, row-major, top row first, converted from the pipeline's
 * [-1, 1] CHW floats with clamp and lrintf. Two-call sizing: pass rgb_out NULL
 * (or a cap too small) to get the needed byte count in *bytes_out and
 * BOOM_ERR_BINDING. cb may be NULL. */
boom_status boom_image_generate(boom_image_model *model, const boom_image_request *req,
                            boom_image_progress_cb cb, void *user,
                            uint8_t *rgb_out, size_t cap, size_t *bytes_out,
                            boom_error **err_out);

/* Write `rgb` (height rows of width*3 bytes, RGB) as a PNG: 8-bit RGB, one
 * IDAT of stored (uncompressed) deflate blocks, no external dependency. */
boom_status boom_image_write_png(const char *path, const uint8_t *rgb,
                             uint32_t width, uint32_t height,
                             boom_error **err_out);

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
 * byte count (including the NUL) in *len_out and BOOM_ERR_BINDING. */
boom_status boom_explain_table(char *buf, size_t cap, size_t *len_out,
                           boom_error **err_out);
boom_status boom_explain_json(char *buf, size_t cap, size_t *len_out,
                          boom_error **err_out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SONICBOOM_CAPI_H */
