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

// SonicBoom stable C ABI implementation.
//
// A thin C-compatible facade over the SonicBoom core S-Expr pipeline. The
// opaque handles own C++ objects (sx::Document, sx::Executable) behind the
// boundary; no C++ type, MLIR/LLVM/ATen/c10 type, or exception crosses into
// the C interface. All C++ failures are folded into the sb_status/sb_error
// value-or-error convention below.

#include <sonicboom/capi.h>

#include <sonicboom/autograd.h>
#include <sonicboom/gguf/reader.h>
#include <sonicboom/model/config.h>
#include <sonicboom/model/loader.h>
#include <sonicboom/model/plan.h>
#include <sonicboom/numerics.h>
#include <sonicboom/planner/kv_cache.h>
#include <sonicboom/quant/quantized_matmul.h>
#include <sonicboom/runtime.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/sx/exec.h>
#include <sonicboom/sx/ir.h>
#include <sonicboom/sx/parser.h>
#include <sonicboom/tensor.h>
#include <sonicboom/training/clip_grad_norm.h>
#include <sonicboom/training/device_graph.h>
#include <sonicboom/training/param.h>
#include <sonicboom/vocab/grammar.h>
#include <sonicboom/vocab/vocab.h>

// core/src is a private include directory for this target (core/CMakeLists):
// the decision registry is internal runtime state, not a Layer 2 header.
#include "planner/decision_registry.h"
// Generated at build time (cmake/WriteBuildTag.cmake, a private include
// directory of this target): the revision the library was built from. It is
// rewritten only when the revision changes.
#include "sb_build_tag.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sx = sonicboom::sx;

// --- Opaque handle definitions (private to this translation unit) -----------

struct sb_document {
  sx::Document doc;
};

struct sb_executable {
  // Exactly one of the two is set: the CPU interpreter's executable
  // (sb_compile) or a compiled device training graph (sb_compile_device). The
  // introspection, bind, execute and retrieve entry points below are shared,
  // and so are the sb_run_multi semantics.
  std::unique_ptr<sx::Executable> exe;
  std::unique_ptr<sonicboom::training::DeviceTrainingGraph> device;
  bool input_bound = false;
  sx::Bytes input;    // copied at bind time
  sx::Bytes output;   // retained after execute
  bool has_output = false;

  bool is_device() const { return device != nullptr; }
  std::size_t input_count() const;
  std::size_t output_count() const;
  void input_info(std::size_t index, sb_dtype *dtype_out, int64_t *rank_out,
                  const int64_t **dims_out, uint64_t *num_bytes_out) const;
  void output_info(std::size_t index, sb_dtype *dtype_out, int64_t *rank_out,
                   const int64_t **dims_out, uint64_t *num_bytes_out) const;
  uint64_t input_num_bytes(std::size_t index) const;
  uint64_t output_num_bytes(std::size_t index) const;
  // Tensor names in binding order; nullptr for the CPU interpreter, whose
  // executable does not retain value names.
  const char *input_name(std::size_t index) const;
  const char *output_name(std::size_t index) const;
};

struct sb_error {
  sb_error_kind kind;
  std::string message;
};

struct sb_tensor {
  nt::Tensor t;
};

struct sb_tape {
  sonicboom::Tape tape;
};

struct sb_param {
  std::unique_ptr<sonicboom::training::Param> p;
};

namespace model = sonicboom::model;
namespace nn = sonicboom::nn;
namespace gguf = sonicboom::gguf;
namespace planner = sonicboom::planner;
namespace quant = sonicboom::quant;
namespace vocab = sonicboom::vocab;

// The GGUF reader owns the byte buffer every Gemma4Model weight span borrows
// from, so the model and its reader travel together in one heap object. Both
// the model handle and every session hold a shared_ptr to this bundle, so the
// buffer outlives the last consumer regardless of free order (a garbage-
// collected binding frees in no guaranteed order).
struct sb_llm_bundle {
  // gguf::Reader is move-only with a private default constructor, so the
  // bundle takes the reader in and owns it for the model's lifetime.
  explicit sb_llm_bundle(gguf::Reader&& r) : reader(std::move(r)) {}
  gguf::Reader reader;
  model::Gemma4Model model;
  vocab::Vocab vocab;  // loaded from tokenizer.ggml.* on model load
};

struct sb_llm_model {
  std::shared_ptr<sb_llm_bundle> bundle;
};

struct sb_llm_session {
  std::shared_ptr<sb_llm_bundle> bundle;
  uint64_t n_ctx = 0;
  std::vector<sb_token> prompt;       // current prompt (decode continues from it)
  std::vector<sb_token> prev_prompt;  // previous prompt, for reuse reporting

  // CUDA fast-path session state (item 2 rewire): the device KV cache persists
  // in `kv` across prefill/decode calls, so decode continues token-by-token
  // without re-running the prefill. Unused on the CPU fallback.
  planner::KvCacheSession kv;         // persistent KV cache + token history
  uint64_t token = 0;                 // current decode token (CUDA path)
  uint64_t pos = 0;                   // next decode position (CUDA path)
  int64_t pending = -1;               // first token from prefill, awaiting emit
  bool prefill_done = false;          // a CUDA prefill has run

  // Constrained decoding (optional): the grammar every sampled token must keep
  // acceptable. `grammar_mask` is the allowed-token bitmap recomputed from the
  // grammar's state before each step; reset at every prefill (a new turn).
  // `grammar_text` is the GBNF the installed grammar was built from, so the
  // same constraint installed again can keep the compiled grammar (and its
  // per-state masks) instead of rebuilding the vocabulary trie.
  std::unique_ptr<vocab::Grammar> grammar;
  std::string grammar_text;
  std::vector<uint8_t> grammar_mask;
  bool grammar_done = false;  // a stop token finished the grammar
};

namespace {

void set_error(sb_error **err_out, sb_error_kind kind, const char *message) noexcept {
  if (!err_out)
    return;
  // Reusing an error slot that already holds an error must not leak the
  // previous handle: free it first. The slot holds a handle we allocated via
  // sb_error_free, never a caller-owned pointer, so this is always safe and
  // never frees uninitialized memory (sb_error_free is null-safe).
  if (*err_out) {
    delete *err_out;
    *err_out = nullptr;
  }
  auto *err = new (std::nothrow) sb_error();
  if (!err)
    return;  // slot left NULL; the failure is still visible via the status code
  err->kind = kind;
  try {
    err->message = message ? message : "";
  } catch (...) {
    // Constructing the diagnostic can in principle throw under allocation
    // pressure; release the handle rather than hand back a broken one.
    delete err;
    return;
  }
  *err_out = err;
}

sb_status fail_arg(sb_error **err_out) {
  set_error(err_out, SB_ERRKIND_INTERNAL, "invalid argument");
  return SB_ERR_INTERNAL;
}

// Contain every C++ exception so none escapes an exported `sb_*` function.
// Each status-returning function runs its body through this: an expected
// exception (std::exception, including bad_alloc) or an unexpected one becomes
// SB_ERR_INTERNAL with a best-effort diagnostic. The function is noexcept so a
// throw during set_error's own (nothrow) allocation path cannot escape.
template <typename Fn>
sb_status guard(Fn &&fn, sb_error **err_out) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc &) {
    set_error(err_out, SB_ERRKIND_INTERNAL, "out of memory");
    return SB_ERR_INTERNAL;
  } catch (const std::exception &e) {
    set_error(err_out, SB_ERRKIND_INTERNAL, e.what());
    return SB_ERR_INTERNAL;
  } catch (...) {
    set_error(err_out, SB_ERRKIND_INTERNAL, "unknown exception");
    return SB_ERR_INTERNAL;
  }
}

sb_dtype map_dtype(sx::DType d) {
  switch (d) {
  case sx::DType::Float32:  return SB_DTYPE_FLOAT32;
  case sx::DType::Float16:  return SB_DTYPE_FLOAT16;
  case sx::DType::BFloat16: return SB_DTYPE_BFLOAT16;
  case sx::DType::Float64:  return SB_DTYPE_FLOAT64;
  case sx::DType::Int8:     return SB_DTYPE_INT8;
  case sx::DType::UInt8:    return SB_DTYPE_UINT8;
  case sx::DType::Int16:    return SB_DTYPE_INT16;
  case sx::DType::Int32:    return SB_DTYPE_INT32;
  case sx::DType::Int64:    return SB_DTYPE_INT64;
  case sx::DType::Bool:     return SB_DTYPE_BOOL;
  }
  return SB_DTYPE_FLOAT32;  // unreachable
}

sb_error_kind parse_errkind(sx::ErrorCategory c) {
  switch (c) {
  case sx::ErrorCategory::Lexical:     return SB_ERRKIND_PARSE_LEXICAL;
  case sx::ErrorCategory::Syntax:      return SB_ERRKIND_PARSE_SYNTAX;
  case sx::ErrorCategory::Semantic:    return SB_ERRKIND_PARSE_SEMANTIC;
  case sx::ErrorCategory::Unsupported: return SB_ERRKIND_PARSE_UNSUPPORTED;
  }
  return SB_ERRKIND_INTERNAL;
}

sb_error_kind exec_errkind(sx::ExecErrorKind k) {
  switch (k) {
  case sx::ExecErrorKind::Weight:   return SB_ERRKIND_WEIGHT;
  case sx::ExecErrorKind::Compile:  return SB_ERRKIND_COMPILE;
  case sx::ExecErrorKind::Binding:  return SB_ERRKIND_BINDING;
  case sx::ExecErrorKind::Runtime:  return SB_ERRKIND_RUNTIME;
  }
  return SB_ERRKIND_INTERNAL;
}

sb_status exec_status(sx::ExecErrorKind k) {
  switch (k) {
  case sx::ExecErrorKind::Weight:   return SB_ERR_WEIGHT;
  case sx::ExecErrorKind::Compile:  return SB_ERR_COMPILE;
  case sx::ExecErrorKind::Binding:  return SB_ERR_BINDING;
  case sx::ExecErrorKind::Runtime:  return SB_ERR_RUNTIME;
  }
  return SB_ERR_INTERNAL;
}

uint64_t tensor_num_bytes(const sx::TensorType &t) {
  // Byte size via checked arithmetic; 0 on element-count/byte-count overflow or
  // an unknown dtype. A successfully compiled Executable has already had its
  // sizes validated by Executable::compile, so 0 here is unreachable for any
  // executable this function is asked to describe.
  return static_cast<uint64_t>(sx::tensor_byte_size(t).value_or(0));
}

} // namespace

// --- Build info -------------------------------------------------------------

// The SB_BUILD_* macros are defined by capi/CMakeLists.txt. The revision comes
// from the generated sb_build_tag.h, written at build time so it follows the
// checkout instead of the last configure; the rest are resolved at configure
// time in the top-level CMakeLists.txt. The fallbacks keep the file compilable
// on its own (e.g. a syntax-only check) — a library built that way reports the
// truth about its own ignorance rather than a plausible guess.

#ifndef SB_BUILD_GIT
#define SB_BUILD_GIT "unknown"
#endif
#ifndef SB_BUILD_GIT_TAG
#define SB_BUILD_GIT_TAG "unknown"
#endif
#ifndef SB_BUILD_TYPE
#define SB_BUILD_TYPE ""
#endif
#ifndef SB_BUILD_CUDA_ENABLED
#define SB_BUILD_CUDA_ENABLED 0
#endif
#ifndef SB_BUILD_CUDA_ARCHS
#define SB_BUILD_CUDA_ARCHS ""
#endif
#ifndef SB_BUILD_CUDA_TOOLKIT
#define SB_BUILD_CUDA_TOOLKIT ""
#endif
#ifndef SB_BUILD_LLVM_VERSION
#define SB_BUILD_LLVM_VERSION "none"
#endif
#ifndef SB_BUILD_LLVM_MAJOR
#define SB_BUILD_LLVM_MAJOR 0
#endif
#ifndef SB_BUILD_LLVM_ORIGIN
#define SB_BUILD_LLVM_ORIGIN "none"
#endif
#ifndef SB_BUILD_PLATFORM
#define SB_BUILD_PLATFORM "unknown"
#endif

namespace {

// Append `"…"` with the characters JSON requires escaped. The build strings come
// from CMake (a compiler's version, a path-derived origin) — none of them should
// contain a quote or a control character, but a stray one must not produce a
// document the caller's parser rejects.
void append_json_string(std::string &out, const char *s) {
  out.push_back('"');
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(s);
       p && *p; ++p) {
    switch (*p) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (*p < 0x20) {
          const char *hex = "0123456789abcdef";
          out += "\\u00";
          out.push_back(hex[*p >> 4]);
          out.push_back(hex[*p & 0xf]);
        } else {
          out.push_back(static_cast<char>(*p));
        }
    }
  }
  out.push_back('"');
}

// Comma-separated "80,86" → a JSON array of strings. Empty → [].
void append_json_arch_array(std::string &out, const char *csv) {
  out.push_back('[');
  if (csv && *csv) {
    bool first = true;
    std::string token;
    for (const char *p = csv;; ++p) {
      if (*p == ',' || *p == '\0') {
        if (!first) out.push_back(',');
        append_json_string(out, token.c_str());
        first = false;
        token.clear();
        if (*p == '\0') break;
      } else {
        token.push_back(*p);
      }
    }
  }
  out.push_back(']');
}

std::string make_build_info() {
  std::string out;
  out.reserve(256);
  out += "{\"git_commit\":";
  append_json_string(out, SB_BUILD_GIT);
  out += ",\"git_tag\":";
  append_json_string(out, SB_BUILD_GIT_TAG);
  out += ",\"build_type\":";
  append_json_string(out, SB_BUILD_TYPE);

  out += ",\"cuda\":{\"enabled\":";
  out += SB_BUILD_CUDA_ENABLED ? "true" : "false";
  out += ",\"archs\":";
  if (SB_BUILD_CUDA_ENABLED) {
    append_json_arch_array(out, SB_BUILD_CUDA_ARCHS);
  } else {
    out += "[]";
  }
  out += ",\"toolkit\":";
  append_json_string(out, SB_BUILD_CUDA_ENABLED ? SB_BUILD_CUDA_TOOLKIT : "");

  out += "},\"llvm\":{\"version\":";
  append_json_string(out, SB_BUILD_LLVM_VERSION);
  out += ",\"major\":";
  out += std::to_string(SB_BUILD_LLVM_MAJOR);
  out += ",\"origin\":";
  append_json_string(out, SB_BUILD_LLVM_ORIGIN);

  out += "},\"platform\":";
  append_json_string(out, SB_BUILD_PLATFORM);
  out += "}";
  return out;
}

} // namespace

const char *sb_build_info(void) {
  // Function-local static: built on first call, thread-safe since C++11, and
  // valid for the rest of the process. No exception can escape — every input is
  // a string literal baked in at configure time, so the allocation is the only
  // thing that could throw, and returning the same stable pointer afterwards
  // means a caller that retried would still get a usable string.
  static const std::string info = make_build_info();
  return info.c_str();
}

// --- Error inspection -------------------------------------------------------

const char *sb_error_message(const sb_error *err) {
  return err ? err->message.c_str() : "";
}

sb_error_kind sb_error_get_kind(const sb_error *err) {
  return err ? err->kind : SB_ERRKIND_NONE;
}

void sb_error_free(sb_error *err) { delete err; }

// --- Parse ------------------------------------------------------------------

sb_status sb_parse(const char *text, size_t len, sb_document **doc_out,
                   sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!text || !doc_out)
          return fail_arg(err_out);
        *doc_out = nullptr;

        auto res = sx::parse_document(std::string_view(text, len));
        if (!res) {
          set_error(err_out, parse_errkind(res.error().category),
                    res.error().message.c_str());
          return SB_ERR_PARSE;
        }

        auto *doc = new sb_document();
        doc->doc = std::move(*res);
        *doc_out = doc;
        return SB_OK;
      },
      err_out);
}

// --- Compile ----------------------------------------------------------------

sb_status sb_compile(const sb_document *doc, const char *base_dir,
                     sb_executable **exe_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!doc || !exe_out)
          return fail_arg(err_out);
        *exe_out = nullptr;

        auto weights =
            sx::load_external_weights(doc->doc, base_dir ? base_dir : "");
        if (!weights) {
          set_error(err_out, exec_errkind(weights.error().kind),
                    weights.error().message.c_str());
          return exec_status(weights.error().kind);
        }

        auto exe = sx::Executable::compile(doc->doc, *weights);
        if (!exe) {
          set_error(err_out, exec_errkind(exe.error().kind),
                    exe.error().message.c_str());
          return exec_status(exe.error().kind);
        }

        auto *out = new sb_executable();
        out->exe = std::move(*exe);
        *exe_out = out;
        return SB_OK;
      },
      err_out);
}

sb_status sb_compile_device(const sb_document *doc, sb_executable **exe_out,
                            sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!doc || !exe_out)
          return fail_arg(err_out);
        *exe_out = nullptr;

        auto graph = sonicboom::training::DeviceTrainingGraph::compile(doc->doc);
        if (!graph) {
          set_error(err_out, SB_ERRKIND_COMPILE, graph.error().c_str());
          return SB_ERR_COMPILE;
        }

        auto *out = new sb_executable();
        out->device = std::move(*graph);
        *exe_out = out;
        return SB_OK;
      },
      err_out);
}

// --- I/O introspection (bind) -----------------------------------------------

namespace {

void fill_tensor_info(const sx::TensorType &t, sb_dtype *dtype_out,
                      int64_t *rank_out, const int64_t **dims_out,
                      uint64_t *num_bytes_out) {
  if (dtype_out)
    *dtype_out = map_dtype(t.dtype);
  if (rank_out)
    *rank_out = static_cast<int64_t>(t.shape.dims.size());
  if (dims_out)
    *dims_out = t.shape.dims.data();
  if (num_bytes_out)
    *num_bytes_out = tensor_num_bytes(t);
}

using DeviceInfo = sonicboom::training::DeviceTensorInfo;

void fill_device_info(const DeviceInfo &t, sb_dtype *dtype_out,
                      int64_t *rank_out, const int64_t **dims_out,
                      uint64_t *num_bytes_out) {
  if (dtype_out)
    *dtype_out = map_dtype(t.dtype);
  if (rank_out)
    *rank_out = static_cast<int64_t>(t.shape.size());
  if (dims_out)
    *dims_out = t.shape.data();
  if (num_bytes_out)
    *num_bytes_out = t.num_bytes;
}

} // namespace

std::size_t sb_executable::input_count() const {
  return is_device() ? device->inputs().size() : exe->input_types().size();
}

std::size_t sb_executable::output_count() const {
  return is_device() ? device->outputs().size() : exe->output_types().size();
}

void sb_executable::input_info(std::size_t index, sb_dtype *dtype_out,
                               int64_t *rank_out, const int64_t **dims_out,
                               uint64_t *num_bytes_out) const {
  if (is_device())
    fill_device_info(device->inputs()[index], dtype_out, rank_out, dims_out,
                     num_bytes_out);
  else
    fill_tensor_info(exe->input_types()[index], dtype_out, rank_out, dims_out,
                     num_bytes_out);
}

void sb_executable::output_info(std::size_t index, sb_dtype *dtype_out,
                                int64_t *rank_out, const int64_t **dims_out,
                                uint64_t *num_bytes_out) const {
  if (is_device())
    fill_device_info(device->outputs()[index], dtype_out, rank_out, dims_out,
                     num_bytes_out);
  else
    fill_tensor_info(exe->output_types()[index], dtype_out, rank_out, dims_out,
                     num_bytes_out);
}

uint64_t sb_executable::input_num_bytes(std::size_t index) const {
  return is_device() ? device->inputs()[index].num_bytes
                     : tensor_num_bytes(exe->input_types()[index]);
}

uint64_t sb_executable::output_num_bytes(std::size_t index) const {
  return is_device() ? device->outputs()[index].num_bytes
                     : tensor_num_bytes(exe->output_types()[index]);
}

const char *sb_executable::input_name(std::size_t index) const {
  return is_device() ? device->inputs()[index].name.c_str() : nullptr;
}

const char *sb_executable::output_name(std::size_t index) const {
  return is_device() ? device->outputs()[index].name.c_str() : nullptr;
}

sb_status sb_input_info(const sb_executable *exe, sb_dtype *dtype_out,
                        int64_t *rank_out, const int64_t **dims_out,
                        uint64_t *num_bytes_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || exe->input_count() == 0)
          return fail_arg(nullptr);
        exe->input_info(0, dtype_out, rank_out, dims_out, num_bytes_out);
        return SB_OK;
      },
      nullptr);
}

sb_status sb_output_info(const sb_executable *exe, sb_dtype *dtype_out,
                         int64_t *rank_out, const int64_t **dims_out,
                         uint64_t *num_bytes_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || exe->output_count() == 0)
          return fail_arg(nullptr);
        exe->output_info(0, dtype_out, rank_out, dims_out, num_bytes_out);
        return SB_OK;
      },
      nullptr);
}

sb_status sb_input_count(const sb_executable *exe, size_t *count_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !count_out)
          return fail_arg(nullptr);
        *count_out = exe->input_count();
        return SB_OK;
      },
      nullptr);
}

sb_status sb_output_count(const sb_executable *exe, size_t *count_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !count_out)
          return fail_arg(nullptr);
        *count_out = exe->output_count();
        return SB_OK;
      },
      nullptr);
}

sb_status sb_input_info_at(const sb_executable *exe, size_t index,
                           sb_dtype *dtype_out, int64_t *rank_out,
                           const int64_t **dims_out, uint64_t *num_bytes_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || index >= exe->input_count())
          return fail_arg(nullptr);
        exe->input_info(index, dtype_out, rank_out, dims_out, num_bytes_out);
        return SB_OK;
      },
      nullptr);
}

sb_status sb_output_info_at(const sb_executable *exe, size_t index,
                            sb_dtype *dtype_out, int64_t *rank_out,
                            const int64_t **dims_out,
                            uint64_t *num_bytes_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || index >= exe->output_count())
          return fail_arg(nullptr);
        exe->output_info(index, dtype_out, rank_out, dims_out, num_bytes_out);
        return SB_OK;
      },
      nullptr);
}

sb_status sb_input_name_at(const sb_executable *exe, size_t index,
                           const char **name_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !name_out || index >= exe->input_count())
          return fail_arg(nullptr);
        *name_out = exe->input_name(index);
        return SB_OK;
      },
      nullptr);
}

sb_status sb_output_name_at(const sb_executable *exe, size_t index,
                            const char **name_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !name_out || index >= exe->output_count())
          return fail_arg(nullptr);
        *name_out = exe->output_name(index);
        return SB_OK;
      },
      nullptr);
}

// --- Execute / retrieve -----------------------------------------------------

sb_status sb_bind_input(sb_executable *exe, const void *data, size_t len,
                        sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !data || exe->input_count() == 0)
          return fail_arg(err_out);

        uint64_t expected = exe->input_num_bytes(0);
        if (static_cast<uint64_t>(len) != expected) {
          std::string msg = "input buffer is " + std::to_string(len) +
                            " bytes, expected " + std::to_string(expected);
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }

        exe->input.assign(static_cast<const std::byte *>(data),
                          static_cast<const std::byte *>(data) + len);
        exe->input_bound = true;
        return SB_OK;
      },
      err_out);
}

sb_status sb_run_multi(sb_executable *exe, const void *const *input_data,
                       const size_t *input_lens, size_t num_inputs,
                       void *const *output_data, const size_t *output_caps,
                       size_t num_outputs, size_t *output_lens,
                       sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !input_data || !input_lens || !output_data ||
            !output_caps)
          return fail_arg(err_out);

        const std::size_t in_count = exe->input_count();
        const std::size_t out_count = exe->output_count();
        if (num_inputs != in_count) {
          std::string msg =
              "graph takes " + std::to_string(in_count) +
              " inputs, sb_run_multi got " + std::to_string(num_inputs);
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }
        if (num_outputs != out_count) {
          std::string msg =
              "graph has " + std::to_string(out_count) +
              " outputs, sb_run_multi got " + std::to_string(num_outputs);
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }

        // Validate every buffer before running: a binding error must leave
        // the output buffers untouched.
        std::vector<sx::Bytes> ins;
        ins.reserve(num_inputs);
        for (size_t i = 0; i < num_inputs; ++i) {
          uint64_t need = exe->input_num_bytes(i);
          if (input_lens[i] != need) {
            std::string msg = "input " + std::to_string(i) + " is " +
                              std::to_string(input_lens[i]) +
                              " bytes, expected " + std::to_string(need);
            set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
            return SB_ERR_BINDING;
          }
          if (need == 0) {
            ins.emplace_back();
            continue;
          }
          if (!input_data[i])
            return fail_arg(err_out);
          const auto *p = static_cast<const std::byte *>(input_data[i]);
          ins.emplace_back(p, p + need);
        }
        for (size_t i = 0; i < num_outputs; ++i) {
          uint64_t need = exe->output_num_bytes(i);
          if (!output_data[i] || output_caps[i] < need) {
            std::string msg = "output " + std::to_string(i) + " has capacity " +
                              std::to_string(output_caps[i]) +
                              " bytes, expected " + std::to_string(need);
            set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
            return SB_ERR_BINDING;
          }
        }

        std::vector<sx::Bytes> outs;
        if (exe->is_device()) {
          std::vector<const std::byte *> ptrs(num_inputs);
          for (size_t i = 0; i < num_inputs; ++i)
            ptrs[i] = static_cast<const std::byte *>(input_data[i]);
          std::vector<std::vector<std::byte>> device_outs;
          std::string device_error;
          if (!exe->device->run(ptrs.data(), input_lens, &device_outs,
                                &device_error)) {
            set_error(err_out, SB_ERRKIND_RUNTIME, device_error.c_str());
            return SB_ERR_RUNTIME;
          }
          outs.resize(device_outs.size());
          for (size_t i = 0; i < device_outs.size(); ++i)
            outs[i] = std::move(device_outs[i]);
        } else {
          auto res = exe->exe->run(ins, outs);
          if (!res) {
            set_error(err_out, exec_errkind(res.error().kind),
                      res.error().message.c_str());
            return exec_status(res.error().kind);
          }
        }

        for (size_t i = 0; i < num_outputs; ++i) {
          std::memcpy(output_data[i], outs[i].data(), outs[i].size());
          if (output_lens)
            output_lens[i] = outs[i].size();
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_run_resident(sb_executable *exe, const void *const *input_data,
                          const size_t *input_lens, size_t num_inputs,
                          const size_t *output_indices, size_t num_outputs,
                          void *const *output_data, const size_t *output_caps,
                          size_t *output_lens, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !input_data || !input_lens || !output_data ||
            !output_caps || (num_outputs != 0 && !output_indices))
          return fail_arg(err_out);
        if (!exe->is_device()) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "resident execution needs a device training graph");
          return SB_ERR_BINDING;
        }

        const std::size_t in_count = exe->input_count();
        if (num_inputs != in_count) {
          std::string msg =
              "graph takes " + std::to_string(in_count) +
              " inputs, sb_run_resident got " + std::to_string(num_inputs);
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }

        for (size_t j = 0; j < num_outputs; ++j) {
          const size_t i = output_indices[j];
          if (i >= exe->output_count()) {
            set_error(err_out, SB_ERRKIND_BINDING,
                      ("output index " + std::to_string(i) +
                       " is out of range").c_str());
            return SB_ERR_BINDING;
          }
          uint64_t need = exe->output_num_bytes(i);
          if (!output_data[j] || output_caps[j] < need) {
            std::string msg = "output " + std::to_string(i) +
                              " has capacity " + std::to_string(output_caps[j]) +
                              " bytes, expected " + std::to_string(need);
            set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
            return SB_ERR_BINDING;
          }
        }

        std::vector<const std::byte *> ptrs(num_inputs, nullptr);
        for (size_t i = 0; i < num_inputs; ++i) {
          uint64_t need = exe->input_num_bytes(i);
          if (input_lens[i] != need) {
            std::string msg = "input " + std::to_string(i) + " is " +
                              std::to_string(input_lens[i]) +
                              " bytes, expected " + std::to_string(need);
            set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
            return SB_ERR_BINDING;
          }
          ptrs[i] = static_cast<const std::byte *>(input_data[i]);
        }

        std::vector<std::vector<std::byte>> outs;
        std::string device_error;
        if (!exe->device->run_resident(ptrs.data(), input_lens,
                                       output_indices, num_outputs, &outs,
                                       &device_error)) {
          set_error(err_out, SB_ERRKIND_RUNTIME, device_error.c_str());
          return SB_ERR_RUNTIME;
        }
        for (size_t j = 0; j < num_outputs; ++j) {
          std::memcpy(output_data[j], outs[j].data(), outs[j].size());
          if (output_lens)
            output_lens[j] = outs[j].size();
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_input_is_resident(const sb_executable *exe, size_t index,
                               int *resident_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !resident_out) return fail_arg(err_out);
        if (!exe->is_device()) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "residency is a property of a device training graph");
          return SB_ERR_BINDING;
        }
        if (index >= exe->input_count()) {
          set_error(err_out, SB_ERRKIND_BINDING, "input index out of range");
          return SB_ERR_BINDING;
        }
        *resident_out = exe->device->input_is_resident(index) ? 1 : 0;
        return SB_OK;
      },
      err_out);
}

sb_status sb_execute(sb_executable *exe, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe)
          return fail_arg(err_out);
        if (!exe->input_bound) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "input not bound (call sb_bind_input)");
          return SB_ERR_BINDING;
        }
        size_t n_out = exe->output_count();
        if (n_out != 1) {
          std::string msg =
              "graph has " + std::to_string(n_out) +
              " outputs; sb_execute is the 1-in/1-out wrapper, use sb_run_multi";
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          exe->has_output = false;
          return SB_ERR_BINDING;
        }

        exe->output.resize(exe->output_num_bytes(0));
        const void *in_ptr = exe->input.data();
        size_t in_len = exe->input.size();
        void *out_ptr = exe->output.data();
        size_t out_cap = exe->output.size();
        size_t written = 0;
        sb_status s = sb_run_multi(exe, &in_ptr, &in_len, 1, &out_ptr,
                                   &out_cap, 1, &written, err_out);
        exe->has_output = (s == SB_OK);
        return s;
      },
      err_out);
}

sb_status sb_retrieve_output(sb_executable *exe, void *out, size_t cap,
                             size_t *written_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !out)
          return fail_arg(err_out);
        if (!exe->has_output) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "no output available (call sb_execute first)");
          return SB_ERR_BINDING;
        }

        if (cap < exe->output.size()) {
          std::string msg = "output buffer is " + std::to_string(cap) +
                            " bytes, expected " +
                            std::to_string(exe->output.size());
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }

        std::memcpy(out, exe->output.data(), exe->output.size());
        if (written_out)
          *written_out = exe->output.size();
        return SB_OK;
      },
      err_out);
}

// --- Release ----------------------------------------------------------------

void sb_document_free(sb_document *doc) { delete doc; }

void sb_executable_free(sb_executable *exe) { delete exe; }

// --- Training (minimal autograd) ---------------------------------------------

namespace {

// All training tensors are float32: byte size is numel * 4.
size_t f32_bytes(int64_t numel) {
  return static_cast<size_t>(numel) * 4;
}

// Allocate a fresh tensor handle for `t` (or fail under memory pressure).
sb_status make_tensor(nt::Tensor t, sb_tensor **out, sb_error **err_out) {
  auto *h = new (std::nothrow) sb_tensor();
  if (!h) {
    set_error(err_out, SB_ERRKIND_INTERNAL, "out of memory");
    return SB_ERR_INTERNAL;
  }
  h->t = std::move(t);
  *out = h;
  return SB_OK;
}

} // namespace

sb_status sb_tensor_from_f32(const int64_t *dims, int64_t rank, const void *data,
                             size_t len, sb_tensor **out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (rank < 0 || !out)
          return fail_arg(err_out);
        *out = nullptr;

        std::vector<int64_t> shape;
        int64_t numel = 1;
        for (int64_t i = 0; i < rank; ++i) {
          if (dims[i] < 0)
            return fail_arg(err_out);
          shape.push_back(dims[i]);
          numel *= dims[i];
        }
        if (len != f32_bytes(numel)) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "tensor byte count does not match shape");
          return SB_ERR_BINDING;
        }

        nt::Tensor t = nt::empty(shape, nt::ScalarType::Float);
        if (numel > 0 && data)
          std::memcpy(t.data_ptr(), data, len);
        return make_tensor(std::move(t), out, err_out);
      },
      err_out);
}

sb_status sb_tensor_bytes(const sb_tensor *t, void *out, size_t cap,
                          size_t *written_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!t || !out)
          return fail_arg(err_out);
        size_t n = f32_bytes(t->t.numel());
        if (cap < n) {
          set_error(err_out, SB_ERRKIND_BINDING, "output buffer too small");
          return SB_ERR_BINDING;
        }
        if (n > 0)
          std::memcpy(out, t->t.data_ptr(), n);
        if (written_out)
          *written_out = n;
        return SB_OK;
      },
      err_out);
}

void sb_tensor_free(sb_tensor *t) { delete t; }

// --- Training: parameter + fused AdamW --------------------------------------

namespace {

// Element count of a row-major shape, rejecting a negative dimension, a zero
// extent (a parameter with no elements has no update) and uint64 overflow.
bool param_numel(const int64_t *dims, int64_t rank, uint64_t &numel) {
  if (!dims || rank <= 0)
    return false;
  numel = 1;
  for (int64_t i = 0; i < rank; ++i) {
    if (dims[i] <= 0)
      return false;
    if (numel > UINT64_MAX / static_cast<uint64_t>(dims[i]))
      return false;
    numel *= static_cast<uint64_t>(dims[i]);
  }
  return true;
}

} // namespace

sb_status sb_param_create(const int64_t *dims, int64_t rank, const void *init,
                          size_t len, sb_param **out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!out || !init)
          return fail_arg(err_out);
        *out = nullptr;

        uint64_t numel = 0;
        if (!param_numel(dims, rank, numel))
          return fail_arg(err_out);
        if (numel > UINT64_MAX / sizeof(float) ||
            len != static_cast<size_t>(numel * sizeof(float))) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "parameter byte count does not match shape");
          return SB_ERR_BINDING;
        }

        std::vector<float> values(static_cast<size_t>(numel));
        std::memcpy(values.data(), init, len);
        auto p = sonicboom::training::Param::create(values);
        if (!p) {
          set_error(err_out, SB_ERRKIND_INTERNAL, p.error().c_str());
          return SB_ERR_INTERNAL;
        }
        *out = new sb_param{std::move(*p)};
        return SB_OK;
      },
      err_out);
}

sb_status sb_param_numel(const sb_param *p, size_t *numel_out) {
  return guard(
      [&]() -> sb_status {
        if (!p || !numel_out)
          return fail_arg(nullptr);
        *numel_out = p->p->size();
        return SB_OK;
      },
      nullptr);
}

int sb_param_on_device(const sb_param *p) {
  return p && p->p->on_device() ? 1 : 0;
}

uint64_t sb_param_step_count(const sb_param *p) {
  return p ? p->p->step_count() : 0;
}

sb_status sb_param_zero_grad(sb_param *p, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!p)
          return fail_arg(err_out);
        p->p->zero_grad();
        return SB_OK;
      },
      err_out);
}

sb_status sb_param_set_grad(sb_param *p, const void *grad, size_t len,
                            sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!p || !grad)
          return fail_arg(err_out);
        const size_t want = p->p->size() * sizeof(float);
        if (len != want) {
          std::string msg = "gradient buffer is " + std::to_string(len) +
                            " bytes, expected " + std::to_string(want);
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }
        std::vector<float> values(p->p->size());
        std::memcpy(values.data(), grad, len);
        if (!p->p->set_grad(values)) {
          set_error(err_out, SB_ERRKIND_RUNTIME, "gradient upload failed");
          return SB_ERR_RUNTIME;
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_param_step(sb_param *p, float lr, float beta1, float beta2,
                        float eps, float weight_decay, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!p)
          return fail_arg(err_out);
        sonicboom::training::AdamWConfig cfg;
        cfg.lr = lr;
        cfg.beta1 = beta1;
        cfg.beta2 = beta2;
        cfg.eps = eps;
        cfg.weight_decay = weight_decay;
        if (!p->p->step(cfg)) {
          set_error(err_out, SB_ERRKIND_RUNTIME, "AdamW update failed");
          return SB_ERR_RUNTIME;
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_param_step_lr(sb_param *p, const sb_tensor *lr, float beta1,
                           float beta2, float eps, float weight_decay,
                           sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!p || !lr)
          return fail_arg(err_out);
        if (lr->t.numel() != 1 || lr->t.dtype() != nt::ScalarType::Float) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "the learning rate must be a one-element float32 tensor");
          return SB_ERR_BINDING;
        }
        sonicboom::training::AdamWConfig cfg;
        cfg.lr = *static_cast<const float*>(lr->t.data_ptr());
        cfg.beta1 = beta1;
        cfg.beta2 = beta2;
        cfg.eps = eps;
        cfg.weight_decay = weight_decay;
        if (!p->p->step(cfg)) {
          set_error(err_out, SB_ERRKIND_RUNTIME, "AdamW update failed");
          return SB_ERR_RUNTIME;
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_param_read(const sb_param *p, void *out, size_t cap,
                        size_t *written_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!p || !out)
          return fail_arg(err_out);
        const size_t n = p->p->size() * sizeof(float);
        if (cap < n) {
          std::string msg = "output buffer is " + std::to_string(cap) +
                            " bytes, expected " + std::to_string(n);
          set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
          return SB_ERR_BINDING;
        }
        std::vector<float> values(p->p->size());
        if (!p->p->read_param(values)) {
          set_error(err_out, SB_ERRKIND_RUNTIME, "parameter read failed");
          return SB_ERR_RUNTIME;
        }
        std::memcpy(out, values.data(), n);
        if (written_out)
          *written_out = n;
        return SB_OK;
      },
      err_out);
}

void sb_param_free(sb_param *p) { delete p; }

sb_status sb_tape_new(sb_tape **out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!out)
          return fail_arg(err_out);
        *out = nullptr;
        auto *t = new (std::nothrow) sb_tape();
        if (!t) {
          set_error(err_out, SB_ERRKIND_INTERNAL, "out of memory");
          return SB_ERR_INTERNAL;
        }
        *out = t;
        return SB_OK;
      },
      err_out);
}

void sb_tape_free(sb_tape *tape) { delete tape; }

void sb_tape_zero_grad(sb_tape *tape) {
  if (tape)
    tape->tape.zero_grad();
}

sb_status sb_linear(const sb_tensor *x, const sb_tensor *w, const sb_tensor *b,
                    sb_tape *tape, sb_tensor **out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!x || !w || !b || !tape || !out)
          return fail_arg(err_out);
        *out = nullptr;
        nt::Tensor y = sonicboom::linear(x->t, w->t, b->t, tape->tape);
        return make_tensor(std::move(y), out, err_out);
      },
      err_out);
}

sb_status sb_relu(const sb_tensor *x, sb_tape *tape, sb_tensor **out,
                  sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!x || !tape || !out)
          return fail_arg(err_out);
        *out = nullptr;
        nt::Tensor y = sonicboom::relu(x->t, tape->tape);
        return make_tensor(std::move(y), out, err_out);
      },
      err_out);
}

sb_status sb_mse_loss(const sb_tensor *pred, const sb_tensor *target,
                      sb_tape *tape, sb_tensor **out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!pred || !target || !tape || !out)
          return fail_arg(err_out);
        *out = nullptr;
        nt::Tensor y = sonicboom::mse_loss(pred->t, target->t, tape->tape);
        return make_tensor(std::move(y), out, err_out);
      },
      err_out);
}

sb_status sb_backward(const sb_tensor *loss, sb_tape *tape, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!loss || !tape)
          return fail_arg(err_out);
        sonicboom::backward(loss->t, tape->tape);
        return SB_OK;
      },
      err_out);
}

sb_status sb_sgd_step(sb_tape *tape, sb_tensor *param, double lr,
                      sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!tape || !param)
          return fail_arg(err_out);
        nt::Tensor g = tape->tape.grad(param->t);
        if (!g.defined()) {
          set_error(err_out, SB_ERRKIND_RUNTIME,
                    "no gradient accumulated for parameter");
          return SB_ERR_RUNTIME;
        }
        sonicboom::sgd_step(param->t, g, lr);
        return SB_OK;
      },
      err_out);
}

sb_status sb_grad(const sb_tape *tape, const sb_tensor *param, sb_tensor **out,
                  sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!tape || !param || !out)
          return fail_arg(err_out);
        *out = nullptr;
        nt::Tensor g = tape->tape.grad(param->t);
        if (!g.defined()) {
          set_error(err_out, SB_ERRKIND_RUNTIME,
                    "no gradient accumulated for parameter");
          return SB_ERR_RUNTIME;
        }
        return make_tensor(std::move(g), out, err_out);
      },
      err_out);
}

sb_status sb_clip_grad_norm(sb_tensor *const *grads, int count, float max_norm,
                            float *norm_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (count < 0 || (count > 0 && !grads)) return fail_arg(err_out);
        if (!std::isfinite(max_norm) || max_norm < 0.0f)
          return fail_arg(err_out);
        if (norm_out) *norm_out = 0.0f;
        // The host reference reads and writes each gradient's storage in
        // place, which is exactly the tape's gradient buffer: the next
        // optimizer step sees the clipped values.
        std::vector<float *> ptrs;
        std::vector<std::size_t> sizes;
        ptrs.reserve(static_cast<std::size_t>(count));
        sizes.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
          const sb_tensor *t = grads[i];
          if (!t || !t->t.defined()) return fail_arg(err_out);
          if (t->t.dtype() != nt::ScalarType::Float) {
            const std::string msg =
                "gradient " + std::to_string(i) + " is not float32";
            set_error(err_out, SB_ERRKIND_INTERNAL, msg.c_str());
            return SB_ERR_INTERNAL;
          }
          if (t->t.device().type != nt::DeviceType::CPU) {
            const std::string msg =
                "gradient " + std::to_string(i) +
                " is not on the CPU; use the device training graph "
                "for GPU gradients";
            set_error(err_out, SB_ERRKIND_INTERNAL, msg.c_str());
            return SB_ERR_INTERNAL;
          }
          ptrs.push_back(static_cast<float *>(t->t.data_ptr()));
          sizes.push_back(static_cast<std::size_t>(t->t.numel()));
        }
        const double norm = sonicboom::training::clip_grad_norm_inplace(
            std::span<float *const>(ptrs.data(), ptrs.size()),
            std::span<const std::size_t>(sizes.data(), sizes.size()),
            static_cast<double>(max_norm));
        if (norm_out) *norm_out = static_cast<float>(norm);
        return SB_OK;
      },
      err_out);
}

// --- Export -----------------------------------------------------------------

sb_status sb_export_model(const char *sx_text, size_t sx_len,
                          const char *const *names, const sb_tensor *const *params,
                          size_t count, const char *dir, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!sx_text || !dir)
          return fail_arg(err_out);
        if (count > 0 && (!names || !params))
          return fail_arg(err_out);

        auto res = sx::parse_document(std::string_view(sx_text, sx_len));
        if (!res) {
          set_error(err_out, parse_errkind(res.error().category),
                    res.error().message.c_str());
          return SB_ERR_PARSE;
        }

        // Marshal the trained float32 tensors into the name → bytes map the
        // exporter consumes. Byte counts are validated against each parameter's
        // declared shape inside export_model (SB_ERR_INTERNAL on mismatch).
        std::unordered_map<std::string, sx::Bytes> map;
        map.reserve(count);
        for (size_t i = 0; i < count; ++i) {
          if (!names[i] || !params[i])
            return fail_arg(err_out);
          const nt::Tensor &t = params[i]->t;
          size_t n = f32_bytes(t.numel());
          sx::Bytes bytes(n);
          if (n > 0)
            std::memcpy(bytes.data(), t.data_ptr(), n);
          map.emplace(std::string(names[i]), std::move(bytes));
        }

        auto r = sonicboom::export_model(*res, map, dir);
        if (!r) {
          set_error(err_out, SB_ERRKIND_INTERNAL,
                    ("export: " + r.error().message).c_str());
          return SB_ERR_INTERNAL;
        }
        return SB_OK;
      },
      err_out);
}

// --- Checkpoints -------------------------------------------------------------

namespace {

// The canonical checkpoint magic. Bump the version character, never reuse it,
// when the entry layout changes.
constexpr char kCheckpointMagic[8] = {'S', 'B', 'C', 'K', 'P', 'T', '1', '\n'};
constexpr uint32_t kCheckpointMaxRank = 16;
constexpr uint32_t kCheckpointMaxName = 4096;

void write_le(std::ofstream &out, uint64_t v, int bytes) {
  char buf[8];
  for (int i = 0; i < bytes; ++i)
    buf[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  out.write(buf, bytes);
}

// Bounds-checked little-endian cursor over the whole checkpoint buffer.
class CheckpointReader {
 public:
  CheckpointReader(const std::uint8_t *data, std::size_t size)
      : data_(data), size_(size) {}

  bool take(void *dst, std::size_t n) {
    if (n > size_ - pos_)
      return false;
    if (n > 0)
      std::memcpy(dst, data_ + pos_, n);
    pos_ += n;
    return true;
  }

  bool u8(std::uint8_t &v) { return take(&v, 1); }

  bool u32(std::uint32_t &v) {
    std::uint8_t b[4];
    if (!take(b, 4))
      return false;
    v = static_cast<std::uint32_t>(b[0]) |
        (static_cast<std::uint32_t>(b[1]) << 8) |
        (static_cast<std::uint32_t>(b[2]) << 16) |
        (static_cast<std::uint32_t>(b[3]) << 24);
    return true;
  }

  bool u64(std::uint64_t &v) {
    std::uint8_t b[8];
    if (!take(b, 8))
      return false;
    v = 0;
    for (int i = 7; i >= 0; --i)
      v = (v << 8) | b[i];
    return true;
  }

  bool i64(std::int64_t &v) {
    std::uint64_t u = 0;
    if (!u64(u))
      return false;
    v = static_cast<std::int64_t>(u);
    return true;
  }

  bool skip(std::size_t n) {
    if (n > size_ - pos_)
      return false;
    pos_ += n;
    return true;
  }

  std::size_t remaining() const { return size_ - pos_; }
  const std::uint8_t *here() const { return data_ + pos_; }

 private:
  const std::uint8_t *data_;
  std::size_t size_;
  std::size_t pos_ = 0;
};

// Row-major element count; rejects a negative dimension and uint64 overflow.
bool checkpoint_numel(const std::vector<int64_t> &dims, uint64_t &numel) {
  numel = 1;
  for (int64_t d : dims) {
    if (d < 0)
      return false;
    const uint64_t du = static_cast<uint64_t>(d);
    if (du != 0 && numel > UINT64_MAX / du)
      return false;
    numel *= du;
  }
  return true;
}

bool read_whole_file(const char *path, std::vector<std::uint8_t> &bytes,
                     std::string &why) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    why = "cannot open checkpoint file";
    return false;
  }
  in.seekg(0, std::ios::end);
  std::streamoff size = in.tellg();
  if (size < 0) {
    why = "cannot size checkpoint file";
    return false;
  }
  in.seekg(0, std::ios::beg);
  bytes.resize(static_cast<std::size_t>(size));
  if (!bytes.empty())
    in.read(reinterpret_cast<char *>(bytes.data()), size);
  if (!in) {
    why = "cannot read checkpoint file";
    return false;
  }
  return true;
}

} // namespace

sb_status sb_save_params(const char *path, const char *const *names,
                         const sb_tensor *const *params, size_t count,
                         sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!path)
          return fail_arg(err_out);
        if (count > 0 && (!names || !params))
          return fail_arg(err_out);

        // Validate everything before touching the file: a rejected save must
        // not truncate an existing checkpoint.
        std::unordered_set<std::string> seen;
        seen.reserve(count);
        for (size_t i = 0; i < count; ++i) {
          if (!names[i] || !params[i])
            return fail_arg(err_out);
          std::string name(names[i]);
          if (name.empty() || name.size() > kCheckpointMaxName) {
            set_error(err_out, SB_ERRKIND_BINDING,
                      "checkpoint entry name is empty or too long");
            return SB_ERR_BINDING;
          }
          if (!seen.insert(name).second) {
            const std::string msg = "duplicate checkpoint entry '" + name + "'";
            set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
            return SB_ERR_BINDING;
          }
          const nt::Tensor &t = params[i]->t;
          if (t.dtype() != nt::ScalarType::Float) {
            set_error(err_out, SB_ERRKIND_BINDING,
                      "checkpoints only store float32 tensors");
            return SB_ERR_BINDING;
          }
          if (t.dim() < 0 ||
              static_cast<uint64_t>(t.dim()) > kCheckpointMaxRank) {
            set_error(err_out, SB_ERRKIND_BINDING,
                      "checkpoint tensor rank is out of range");
            return SB_ERR_BINDING;
          }
        }

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
          set_error(err_out, SB_ERRKIND_WEIGHT, "cannot create checkpoint file");
          return SB_ERR_WEIGHT;
        }
        out.write(kCheckpointMagic, sizeof(kCheckpointMagic));
        write_le(out, static_cast<uint64_t>(count), 8);
        for (size_t i = 0; i < count; ++i) {
          const std::string name(names[i]);
          const nt::Tensor &t = params[i]->t;
          write_le(out, static_cast<uint64_t>(name.size()), 4);
          out.write(name.data(), static_cast<std::streamsize>(name.size()));
          write_le(out, static_cast<uint64_t>(SB_DTYPE_FLOAT32), 1);
          write_le(out, static_cast<uint64_t>(t.dim()), 4);
          for (int64_t d = 0; d < t.dim(); ++d)
            write_le(out, static_cast<uint64_t>(t.size(d)), 8);
          const size_t n = f32_bytes(t.numel());
          write_le(out, static_cast<uint64_t>(n), 8);
          if (n > 0)
            out.write(static_cast<const char *>(t.data_ptr()),
                      static_cast<std::streamsize>(n));
        }
        out.flush();
        if (!out) {
          set_error(err_out, SB_ERRKIND_WEIGHT, "cannot write checkpoint file");
          return SB_ERR_WEIGHT;
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_load_params(const char *path, const char *const *names,
                         sb_tensor *const *params, size_t count,
                         sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!path)
          return fail_arg(err_out);
        if (count > 0 && (!names || !params))
          return fail_arg(err_out);

        std::unordered_map<std::string, size_t> target;
        target.reserve(count);
        for (size_t i = 0; i < count; ++i) {
          if (!names[i] || !params[i])
            return fail_arg(err_out);
          std::string name(names[i]);
          if (!target.emplace(std::move(name), i).second) {
            set_error(err_out, SB_ERRKIND_BINDING,
                      "duplicate requested checkpoint entry");
            return SB_ERR_BINDING;
          }
        }

        std::vector<std::uint8_t> bytes;
        std::string why;
        if (!read_whole_file(path, bytes, why)) {
          set_error(err_out, SB_ERRKIND_WEIGHT, why.c_str());
          return SB_ERR_WEIGHT;
        }
        CheckpointReader rd(bytes.data(), bytes.size());

        char magic[sizeof(kCheckpointMagic)];
        if (!rd.take(magic, sizeof(magic)) ||
            std::memcmp(magic, kCheckpointMagic, sizeof(magic)) != 0) {
          set_error(err_out, SB_ERRKIND_WEIGHT,
                    "not a SonicBoom checkpoint file");
          return SB_ERR_WEIGHT;
        }
        uint64_t entries = 0;
        if (!rd.u64(entries)) {
          set_error(err_out, SB_ERRKIND_WEIGHT, "truncated checkpoint header");
          return SB_ERR_WEIGHT;
        }
        // Every entry has a header larger than 16 bytes, so a count that
        // cannot possibly fit in the remaining bytes is corruption.
        if (entries > rd.remaining() / 16) {
          set_error(err_out, SB_ERRKIND_WEIGHT, "truncated checkpoint");
          return SB_ERR_WEIGHT;
        }

        std::vector<char> matched(count, 0);
        std::vector<std::vector<std::uint8_t>> staged(count);
        std::unordered_set<std::string> seen;

        for (uint64_t e = 0; e < entries; ++e) {
          uint32_t name_len = 0;
          if (!rd.u32(name_len) || name_len == 0 ||
              name_len > kCheckpointMaxName || name_len > rd.remaining()) {
            set_error(err_out, SB_ERRKIND_WEIGHT, "malformed checkpoint entry");
            return SB_ERR_WEIGHT;
          }
          std::string name(reinterpret_cast<const char *>(rd.here()), name_len);
          rd.skip(name_len);

          uint8_t dtype = 0;
          uint32_t rank = 0;
          if (!rd.u8(dtype) || !rd.u32(rank) || rank > kCheckpointMaxRank) {
            set_error(err_out, SB_ERRKIND_WEIGHT, "malformed checkpoint entry");
            return SB_ERR_WEIGHT;
          }
          std::vector<int64_t> dims(rank);
          for (uint32_t d = 0; d < rank; ++d) {
            if (!rd.i64(dims[d])) {
              set_error(err_out, SB_ERRKIND_WEIGHT,
                        "truncated checkpoint entry");
              return SB_ERR_WEIGHT;
            }
          }
          uint64_t n = 0;
          uint64_t numel = 0;
          if (!rd.u64(n) || !checkpoint_numel(dims, numel) ||
              numel > UINT64_MAX / sizeof(float)) {
            set_error(err_out, SB_ERRKIND_WEIGHT, "malformed checkpoint entry");
            return SB_ERR_WEIGHT;
          }
          if (dtype != SB_DTYPE_FLOAT32 ||
              n != numel * static_cast<uint64_t>(sizeof(float))) {
            set_error(err_out, SB_ERRKIND_WEIGHT,
                      "checkpoint entry dtype or byte count is invalid");
            return SB_ERR_WEIGHT;
          }
          if (!seen.insert(name).second) {
            set_error(err_out, SB_ERRKIND_WEIGHT,
                      "checkpoint contains a duplicate entry name");
            return SB_ERR_WEIGHT;
          }

          auto it = target.find(name);
          if (it != target.end()) {
            const size_t idx = it->second;
            const nt::Tensor &t = params[idx]->t;
            if (t.dtype() != nt::ScalarType::Float ||
                t.sizes() != dims) {
              const std::string msg =
                  "checkpoint entry '" + name +
                  "' does not match the target tensor's dtype/shape";
              set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
              return SB_ERR_BINDING;
            }
            staged[idx].assign(rd.here(), rd.here() + n);
            matched[idx] = 1;
          }
          if (!rd.skip(static_cast<std::size_t>(n))) {
            set_error(err_out, SB_ERRKIND_WEIGHT, "truncated checkpoint entry");
            return SB_ERR_WEIGHT;
          }
        }
        if (rd.remaining() != 0) {
          set_error(err_out, SB_ERRKIND_WEIGHT,
                    "checkpoint has trailing bytes");
          return SB_ERR_WEIGHT;
        }
        for (size_t i = 0; i < count; ++i) {
          if (!matched[i]) {
            const std::string msg =
                "checkpoint has no entry named '" + std::string(names[i]) + "'";
            set_error(err_out, SB_ERRKIND_BINDING, msg.c_str());
            return SB_ERR_BINDING;
          }
        }

        // Fully validated: commit every target.
        for (size_t i = 0; i < count; ++i) {
          if (!staged[i].empty())
            std::memcpy(params[i]->t.data_ptr(), staged[i].data(),
                        staged[i].size());
        }
        return SB_OK;
      },
      err_out);
}

// --- Numerics profile -------------------------------------------------------

sb_status sb_set_numerics_profile(sb_numerics_profile profile,
                                  sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        sonicboom::NumericsProfile p;
        switch (profile) {
        case SB_NUMERICS_STRICT:
          p = sonicboom::NumericsProfile::Strict;
          break;
        case SB_NUMERICS_STANDARD:
          p = sonicboom::NumericsProfile::Standard;
          break;
        default:
          return fail_arg(err_out);
        }
        std::string error;
        if (!sonicboom::set_numerics_profile(p, &error)) {
          set_error(err_out, SB_ERRKIND_BINDING, error.c_str());
          return SB_ERR_UNSUPPORTED;
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_get_numerics_profile(sb_numerics_profile *profile_out,
                                  sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!profile_out)
          return fail_arg(err_out);
        *profile_out =
            sonicboom::numerics_profile() == sonicboom::NumericsProfile::Strict
                ? SB_NUMERICS_STRICT
                : SB_NUMERICS_STANDARD;
        return SB_OK;
      },
      err_out);
}

// --- Inference (minimal LLM surface) ----------------------------------------

sb_status sb_llm_model_load(const char *gguf_path, sb_llm_model **model_out,
                            sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!gguf_path || !model_out)
          return fail_arg(err_out);
        *model_out = nullptr;

        auto reader = gguf::Reader::load(std::string(gguf_path));
        if (!reader) {
          set_error(err_out, SB_ERRKIND_WEIGHT, reader.error().message.c_str());
          return SB_ERR_WEIGHT;
        }

        // Move the reader into a heap bundle first (its buffer address is then
        // stable), then bind the model against it so every weight span borrows
        // the bundle's buffer.
        auto bundle = std::make_shared<sb_llm_bundle>(std::move(*reader));

        // The architectures SonicBoom models (Gemma 4, Qwen2) load through the
        // same arch-generic loader; `detect_arch` rejects anything else, so the
        // check here is "does SonicBoom know this architecture at all", not a
        // family list. `general.architecture` stays readable through
        // sb_llm_model_arch for a caller that has to format prompts per family.
        auto arch = model::detect_arch(bundle->reader);
        if (!arch) {
          set_error(err_out, SB_ERRKIND_INTERNAL, arch.error().c_str());
          return SB_ERR_UNSUPPORTED;
        }

        auto m = model::load_gemma4(bundle->reader);
        if (!m) {
          set_error(err_out, SB_ERRKIND_WEIGHT, m.error().c_str());
          return SB_ERR_WEIGHT;
        }
        bundle->model = std::move(*m);

        auto v = vocab::load_vocab(bundle->reader);
        if (!v) {
          set_error(err_out, SB_ERRKIND_WEIGHT, v.error().c_str());
          return SB_ERR_WEIGHT;
        }
        bundle->vocab = std::move(*v);

        auto *h = new (std::nothrow) sb_llm_model();
        if (!h) {
          set_error(err_out, SB_ERRKIND_INTERNAL, "out of memory");
          return SB_ERR_INTERNAL;
        }
        h->bundle = std::move(bundle);
        *model_out = h;
        return SB_OK;
      },
      err_out);
}

void sb_llm_model_free(sb_llm_model *model) { delete model; }

sb_status sb_llm_model_arch(const sb_llm_model *model, char *out, size_t cap,
                            size_t *len_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!model || !len_out)
          return fail_arg(err_out);
        const gguf::Value *v = model->bundle->reader.find("general.architecture");
        const std::string s =
            (v && v->type == gguf::ValueType::String) ? v->s : std::string();
        *len_out = s.size();
        if (cap <= s.size())
          return SB_ERR_BINDING;  // two-call sizing: *len_out holds the length
        if (!s.empty())
          std::memcpy(out, s.data(), s.size());
        out[s.size()] = '\0';
        return SB_OK;
      },
      err_out);
}

// The model's EOS ids as its GGUF declares them: `tokenizer.ggml.eos_token_id`
// is the single id most models declare, `tokenizer.ggml.eos_token_ids` the
// array form for a model with several. A model that declares neither has none.
namespace {
std::vector<int32_t> model_eos_ids(const sb_llm_bundle &b) {
  std::vector<int32_t> eos;
  auto push = [&](const gguf::Value &v) {
    if (v.type == gguf::ValueType::Array) {
      if (v.elem_type == gguf::ValueType::Int32 ||
          v.elem_type == gguf::ValueType::UInt32)
        for (const auto &e : v.elems)
          eos.push_back(
              int32_t(e.type == gguf::ValueType::UInt32 ? e.u : uint64_t(e.i)));
    } else if (v.type == gguf::ValueType::Int32 ||
               v.type == gguf::ValueType::UInt32) {
      eos.push_back(
          int32_t(v.type == gguf::ValueType::UInt32 ? v.u : uint64_t(v.i)));
    }
  };
  if (const gguf::Value *v = b.reader.find("tokenizer.ggml.eos_token_id"))
    push(*v);
  if (const gguf::Value *v = b.reader.find("tokenizer.ggml.eos_token_ids"))
    push(*v);
  return eos;
}
}  // namespace

sb_status sb_llm_model_eos(const sb_llm_model *model, sb_token *out, size_t cap,
                           size_t *n_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!model || !n_out)
          return fail_arg(err_out);

        const std::vector<int32_t> eos = model_eos_ids(*model->bundle);

        *n_out = eos.size();
        if (cap < eos.size())
          return SB_ERR_BINDING;  // two-call sizing: *n_out holds the count
        if (!eos.empty())
          std::memcpy(out, eos.data(), eos.size() * sizeof(sb_token));
        return SB_OK;
      },
      err_out);
}

sb_status sb_llm_tokenize_ex(const sb_llm_model *model, const char *text, size_t len,
                             int flags, sb_token *out, size_t cap,
                             size_t *n_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!model || !text || !n_out)
          return fail_arg(err_out);

        const std::vector<int32_t> toks = vocab::tokenize(
            model->bundle->vocab, std::string_view(text, len),
            (flags & SB_TOKENIZE_PARSE_SPECIAL) != 0,
            (flags & SB_TOKENIZE_ADD_SPECIAL) != 0);

        *n_out = toks.size();
        if (cap < toks.size())
          return SB_ERR_BINDING;  // two-call sizing: *n_out holds the needed count
        if (!toks.empty())
          std::memcpy(out, toks.data(), toks.size() * sizeof(sb_token));
        return SB_OK;
      },
      err_out);
}

sb_status sb_llm_tokenize(const sb_llm_model *model, const char *text, size_t len,
                          sb_token *out, size_t cap, size_t *n_out,
                          sb_error **err_out) {
  // Text only: a control marker inside `text` stays text.
  return sb_llm_tokenize_ex(model, text, len, SB_TOKENIZE_ADD_SPECIAL, out, cap,
                            n_out, err_out);
}

sb_status sb_llm_detokenize(const sb_llm_model *model, const sb_token *tokens,
                            size_t n, char *out, size_t cap, size_t *len_out,
                            sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!model || !len_out)
          return fail_arg(err_out);
        if (n > 0 && !tokens)
          return fail_arg(err_out);

        const std::string s = vocab::detokenize(model->bundle->vocab, tokens, n);

        // cap must hold the text plus a NUL terminator.
        *len_out = s.size();
        if (cap <= s.size())
          return SB_ERR_BINDING;
        if (!s.empty())
          std::memcpy(out, s.data(), s.size());
        out[s.size()] = '\0';
        return SB_OK;
      },
      err_out);
}

sb_status sb_llm_session_new(sb_llm_model *model, uint64_t n_ctx,
                             sb_llm_session **session_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!model || !session_out)
          return fail_arg(err_out);
        *session_out = nullptr;

        auto *h = new (std::nothrow) sb_llm_session();
        if (!h) {
          set_error(err_out, SB_ERRKIND_INTERNAL, "out of memory");
          return SB_ERR_INTERNAL;
        }
        h->bundle = model->bundle;  // keep the model (and its buffer) alive
        h->n_ctx = n_ctx;
        h->kv.n_ctx = n_ctx;  // size the persistent KV cache to this session's window
        // Pay the one-time decode-runtime construction (graph + plan +
        // decode-step capture, ~1 s) here rather than inside the session's first
        // prefill, so first-request TTFT matches steady state. The runtime is
        // cached under (model, n_ctx) and is the exact one the prefill builds.
        if (quant::cuda_available() && n_ctx != 0)
          (void)model::warmup_spine_cuda(model->bundle->model, n_ctx);
        *session_out = h;
        return SB_OK;
      },
      err_out);
}

void sb_llm_session_free(sb_llm_session *session) { delete session; }

sb_status sb_llm_prefill(sb_llm_session *session, const sb_token *tokens,
                         size_t n, uint64_t *n_reused_out, sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!session)
          return fail_arg(err_out);
        if (n > 0 && !tokens)
          return fail_arg(err_out);

        // Longest common prefix with the previous prompt (informational).
        const size_t common =
            [&]() -> size_t {
          size_t k = 0;
          const size_t lim =
              n < session->prev_prompt.size() ? n : session->prev_prompt.size();
          while (k < lim && session->prev_prompt[k] == tokens[k])
            ++k;
          return k;
        }();
        // The last prompt token is always (re)processed: its activation is what
        // produces the next-token logits, so even a prompt the session holds in
        // full can reuse at most n-1 tokens. This is the count for the CPU path,
        // which re-prefills everything; the CUDA path below overwrites it with
        // the prefix it actually kept.
        uint64_t reused = static_cast<uint64_t>(common);
        if (n > 0 && reused == n) reused = n - 1;
        if (n_reused_out) *n_reused_out = reused;

        // Update the stored prompt first, then remember it as the previous
        // prompt: prev_prompt must hold the prompt that was *just* prefilled so
        // the next call compares against it (not the one before last).
        session->prompt.assign(tokens, tokens + n);
        session->prev_prompt = session->prompt;

        // A prefill starts a new constrained turn: the grammar (when set)
        // restarts from its first position. The first token sampled at the end
        // of the prefill is subject to it like any decode step.
        if (session->grammar) {
          session->grammar->reset();
          session->grammar_done = false;
        }

        // CUDA fast path: prefill the prompt device-resident (chunked prefill +
        // prefix reuse via the persistent KvCacheSession), sampling only the
        // first continuation token; sb_llm_decode then continues token-by-token
        // over the same cache without re-running the prefill. The CPU path
        // (no device) defers all compute to sb_llm_decode as before.
        session->prefill_done = false;
        session->pending = -1;
        if (quant::cuda_available() && n > 0) {
          const model::Gemma4Model &m = session->bundle->model;
          std::vector<uint64_t> prompt(tokens, tokens + n);
          std::vector<int64_t> first(1, -1);
          if (session->grammar)
            session->grammar_mask = session->grammar->mask();
          const uint64_t done = model::generate_prefill_spine_cuda_prefix(
              m, prompt, /*start_pos=*/0, first, session->kv,
              /*use_prefill=*/true, &reused, session->grammar_mask);
          if (done == 0 || first[0] < 0) {
            set_error(err_out, SB_ERRKIND_INTERNAL, "CUDA prefill failed");
            return SB_ERR_INTERNAL;
          }
          if (session->grammar) {
            if (!session->grammar->accept(int32_t(first[0]))) {
              set_error(err_out, SB_ERRKIND_INTERNAL,
                        "grammar rejected the prefill token");
              return SB_ERR_INTERNAL;
            }
            if (session->grammar->finished()) {
              // The first token was a stop token: the constrained output is
              // complete and empty, so nothing is pending.
              session->grammar_done = true;
              session->token = static_cast<uint64_t>(first[0]);
              session->pos = n;
              session->prefill_done = true;
              return SB_OK;
            }
          }
          session->token = static_cast<uint64_t>(first[0]);
          session->pos = n;
          session->pending = first[0];
          session->prefill_done = true;
        }
        return SB_OK;
      },
      err_out);
}

sb_status sb_llm_decode(sb_llm_session *session, const sb_llm_sampling *sampling,
                        sb_llm_token_cb cb, void *user, uint32_t *n_out,
                        sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!session || !sampling || !n_out)
          return fail_arg(err_out);
        *n_out = 0;

        if (session->prompt.empty()) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "no prompt (call sb_llm_prefill first)");
          return SB_ERR_BINDING;
        }

        // Fields appended after max_tokens are read only when struct_size
        // covers them; otherwise their neutral defaults apply.
        auto has_field = [&](size_t offset, size_t size) {
          return sampling->struct_size >= offset + size;
        };
        const float min_p =
            has_field(offsetof(sb_llm_sampling, min_p), sizeof(float))
                ? sampling->min_p
                : 0.0f;
        const float repeat_penalty =
            has_field(offsetof(sb_llm_sampling, repeat_penalty), sizeof(float))
                ? sampling->repeat_penalty
                : 1.0f;
        const uint32_t repeat_last_n =
            has_field(offsetof(sb_llm_sampling, repeat_last_n), sizeof(uint32_t))
                ? sampling->repeat_last_n
                : 64u;
        const float presence_penalty =
            has_field(offsetof(sb_llm_sampling, presence_penalty), sizeof(float))
                ? sampling->presence_penalty
                : 0.0f;
        const float frequency_penalty =
            has_field(offsetof(sb_llm_sampling, frequency_penalty), sizeof(float))
                ? sampling->frequency_penalty
                : 0.0f;

        // Range validation names the offending field.
        if (!(sampling->temperature >= 0.0f)) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "temperature must be >= 0");
          return SB_ERR_BINDING;
        }
        if (!(sampling->top_p >= 0.0f && sampling->top_p <= 1.0f)) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "top_p must be in [0, 1]");
          return SB_ERR_BINDING;
        }
        if (!(min_p >= 0.0f && min_p <= 1.0f)) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "min_p must be in [0, 1]");
          return SB_ERR_BINDING;
        }
        if (!(repeat_penalty > 0.0f)) {
          set_error(err_out, SB_ERRKIND_BINDING,
                    "repeat_penalty must be > 0");
          return SB_ERR_BINDING;
        }

        nn::SamplingParams sp;
        sp.temperature = sampling->temperature;
        sp.top_k = sampling->top_k;
        sp.top_p = sampling->top_p;
        sp.min_p = min_p;
        sp.seed = sampling->seed;
        sp.repeat_penalty = repeat_penalty;
        sp.repeat_last_n = repeat_last_n;
        sp.presence_penalty = presence_penalty;
        sp.frequency_penalty = frequency_penalty;
        const bool sampling_mode = sp.temperature > 0.0f;

        // Speculative decoding verifies a draft greedily; a sampled stream
        // cannot use it. The sampled path below is the plain per-token loop.
        if (sampling_mode)
          planner::decisions::record("spec.disabled_by_sampling", "decode",
                                     "plain", planner::decisions::Source::Application);

        // A sampled decode recomputes every token from the prompt, including
        // the prefill's first token (which the greedy prefill may already have
        // consumed into the grammar). Restart the grammar at the prompt's end.
        if (sampling_mode && session->grammar) {
          session->grammar->reset();
          session->grammar_done = false;
          session->grammar_mask.clear();
        }

        const uint32_t max_tokens = sampling->max_tokens;
        if (max_tokens == 0)
          return SB_OK;

        const model::Gemma4Model &m = session->bundle->model;

        // Sampled CUDA decode: the non-greedy decode graph puts the full
        // [vocab] logits on the host each step and sample_token chooses the
        // token there. The prefill's greedy pending token is not used, so a
        // shifted prompt (prefix reuse) samples position-correctly: the draw is
        // keyed by the token's absolute position and the history is the whole
        // prompt, so a re-prefilled prefix reproduces the same stream.
        if (sampling_mode && quant::cuda_available() && !session->grammar) {
          std::vector<int64_t> out(max_tokens);
          std::vector<uint64_t> prompt(session->prompt.begin(),
                                       session->prompt.end());
          const uint64_t got = model::generate_spine_cuda_sampled(
              m, prompt, /*start_pos=*/0, out, session->n_ctx, sp);
          uint32_t n = got > max_tokens ? max_tokens : uint32_t(got);
          if (cb) {
            for (uint32_t i = 0; i < n; ++i) {
              if (cb(user, static_cast<sb_token>(out[i]))) {
                n = i;
                break;
              }
            }
          }
          *n_out = n;
          return SB_OK;
        }

        // CUDA fast path: the prefill already filled the persistent KV cache and
        // sampled the first token. Emit it, then decode the rest one step at a
        // time — each step advances the same persistent cache, so this never
        // re-runs the prefill or the whole generation.
        if (!sampling_mode && session->prefill_done && quant::cuda_available()) {
          const uint64_t n_ctx = session->n_ctx;
          uint32_t n = 0;
          // The grammar finished on the prefill token (a stop token): the
          // constrained output is complete and there is nothing to emit.
          if (session->grammar && session->grammar_done) {
            *n_out = 0;
            return SB_OK;
          }
          if (session->pending >= 0) {
            if (cb && cb(user, static_cast<sb_token>(session->pending))) {
              *n_out = 0;  // aborted before the first token
              return SB_OK;
            }
            session->pending = -1;
            n = 1;
          }
          while (n < max_tokens) {
            std::vector<int64_t> one(1, -1);
            const uint64_t fed = session->token;  // this step writes fed's KV
            // Constrained decoding: the allowed-token bitmap for this position,
            // computed from the text accepted so far. An empty mask means the
            // grammar has finished; there is nothing more it can accept.
            if (session->grammar) {
              session->grammar_mask = session->grammar->mask();
              if (session->grammar_mask.empty()) break;
            }
            const uint64_t done = model::generate_spine_cuda(
                m, session->token, session->pos, one, n_ctx,
                session->grammar_mask);
            if (done == 0 || one[0] < 0)
              break;  // decode step failed, or the mask allowed nothing
            // The step wrote `fed`'s KV at session->pos: the session owns one
            // more history position, so a later prefill on this session reuses
            // the generation instead of re-prefilling it.
            session->kv.filled_tokens.push_back(fed);
            session->token = static_cast<uint64_t>(one[0]);
            ++session->pos;
            if (session->grammar) {
              if (!session->grammar->accept(int32_t(one[0])))
                break;  // the sampler returned a token the grammar forbids
              if (session->grammar->finished()) {
                // A stop token ends the constrained output; it is not text, so
                // it is not emitted.
                session->grammar_done = true;
                break;
              }
            }
            if (cb && cb(user, static_cast<sb_token>(one[0]))) {
              *n_out = n;  // aborted before one[0]; report n produced
              return SB_OK;
            }
            ++n;
          }
          *n_out = n;
          return SB_OK;
        }

        // CPU fallback (no CUDA device, or a prefill that did not run the CUDA
        // path): the original full-generation loop over the CPU spine.
        std::vector<int64_t> tokens(max_tokens);
        uint64_t produced = 0;
        if (session->grammar) {
          // Constrained decoding recomputes the mask from the accepted text, so
          // the CPU spine cannot batch the loop: each token is a prefill of the
          // full history plus one masked sample. O(n^2) work, which is the CPU
          // reference path's price, not the performance target.
          std::vector<uint64_t> hist(session->prompt.begin(), session->prompt.end());
          while (produced < max_tokens) {
            session->grammar_mask = session->grammar->mask();
            if (session->grammar_mask.empty()) break;
            int64_t one[1] = {-1};
            uint64_t n_ctx = session->n_ctx;
            if (n_ctx < hist.size() + 1) n_ctx = hist.size() + 1;
            const uint64_t got = model::generate_prefill_spine(
                m, hist, 0, std::span<int64_t>(one, 1), n_ctx, true,
                session->grammar_mask, sampling_mode ? &sp : nullptr);
            if (got == 0 || one[0] < 0) break;
            if (!session->grammar->accept(int32_t(one[0]))) break;
            if (session->grammar->finished()) {
              session->grammar_done = true;
              break;  // a stop token is not text: not emitted
            }
            tokens[produced++] = one[0];
            hist.push_back(uint64_t(one[0]));
          }
        } else if (session->prompt.size() == 1) {
          // A single-token prompt is a seed: decode directly (the exact path the
          // golden-token test exercises).
          produced = model::generate_spine(
              m, uint64_t(session->prompt[0]), 0, tokens, session->n_ctx, {},
              sampling_mode ? &sp : nullptr);
        } else {
          std::vector<uint64_t> prompt(session->prompt.begin(),
                                       session->prompt.end());
          produced = model::generate_prefill_spine(
              m, prompt, 0, tokens, session->n_ctx, true, {},
              sampling_mode ? &sp : nullptr);
        }

        uint32_t n = static_cast<uint32_t>(produced);
        if (n > max_tokens)
          n = max_tokens;
        if (cb) {
          for (uint32_t i = 0; i < n; ++i) {
            if (cb(user, static_cast<sb_token>(tokens[i]))) {
              n = i;  // aborted before tokens[i]; report i produced
              break;
            }
          }
        }
        *n_out = n;
        return SB_OK;
      },
      err_out);
}

// --- Constrained decoding (session grammar) --------------------------------

namespace {

// Install `gbnf` as the session's grammar, replacing any previous one. The
// model's EOS ids are the grammar's stop tokens: they are not text and may only
// be sampled where the grammar can end.
//
// Installing the grammar that is already installed (a server applying one
// response_format turn after turn) keeps the compiled one and only resets its
// position: the vocabulary trie and the mask of every state it has visited
// stay warm, and the first visit to a permissive state costs a walk over the
// whole trie (tens of milliseconds on a large vocabulary).
sb_status install_grammar(sb_llm_session &session, std::string_view gbnf,
                          sb_error **err_out) {
  if (session.grammar && session.grammar_text == gbnf) {
    session.grammar->reset();
    session.grammar_done = false;
    session.grammar_mask.clear();
    return SB_OK;
  }
  const std::vector<int32_t> eos = model_eos_ids(*session.bundle);
  auto parsed =
      vocab::Grammar::parse(gbnf, session.bundle->vocab, eos);
  if (!parsed) {
    const std::string msg = "grammar rejected: " + parsed.error();
    set_error(err_out, SB_ERRKIND_INTERNAL, msg.c_str());
    return SB_ERR_INTERNAL;
  }
  session.grammar = std::move(*parsed);
  session.grammar_text.assign(gbnf);
  session.grammar_done = false;
  session.grammar_mask.clear();
  return SB_OK;
}

}  // namespace

sb_status sb_llm_session_set_grammar(sb_llm_session *session, const char *gbnf,
                                     sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!session || !gbnf)
          return fail_arg(err_out);
        return install_grammar(*session, gbnf, err_out);
      },
      err_out);
}

sb_status sb_llm_json_schema_to_gbnf(const char *schema_json,
                                     const char *prefix, const char *suffix,
                                     char *out, size_t cap, size_t *len_out,
                                     sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!schema_json || !len_out)
          return fail_arg(err_out);
        auto gbnf = vocab::json_schema_to_gbnf(
            schema_json, prefix ? std::string_view(prefix) : std::string_view(),
            suffix ? std::string_view(suffix) : std::string_view());
        if (!gbnf) {
          const std::string msg = "JSON schema rejected: " + gbnf.error();
          set_error(err_out, SB_ERRKIND_INTERNAL, msg.c_str());
          return SB_ERR_INTERNAL;
        }
        *len_out = gbnf->size();
        if (cap <= gbnf->size())
          return SB_ERR_BINDING;  // two-call sizing: *len_out holds the length
        std::memcpy(out, gbnf->data(), gbnf->size());
        out[gbnf->size()] = '\0';
        return SB_OK;
      },
      err_out);
}

sb_status sb_llm_session_set_json_schema(sb_llm_session *session,
                                         const char *schema_json,
                                         sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!session || !schema_json)
          return fail_arg(err_out);
        auto gbnf = vocab::json_schema_to_gbnf(schema_json);
        if (!gbnf) {
          const std::string msg = "JSON schema rejected: " + gbnf.error();
          set_error(err_out, SB_ERRKIND_INTERNAL, msg.c_str());
          return SB_ERR_INTERNAL;
        }
        return install_grammar(*session, *gbnf, err_out);
      },
      err_out);
}

void sb_llm_session_clear_grammar(sb_llm_session *session) {
  if (!session)
    return;
  session->grammar.reset();
  session->grammar_text.clear();
  session->grammar_mask.clear();
  session->grammar_done = false;
}

sb_status sb_llm_session_constraint_state(const sb_llm_session *session,
                                          int *active_out, int *complete_out,
                                          sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!session || !active_out || !complete_out)
          return fail_arg(err_out);
        *active_out = session->grammar ? 1 : 0;
        *complete_out =
            (session->grammar &&
             (session->grammar->finished() || session->grammar->accepting()))
                ? 1
                : 0;
        return SB_OK;
      },
      err_out);
}

// --- Decision registry -----------------------------------------------------

namespace {

// The shared buffer contract of the two explain calls: *len_out always gets
// the needed byte count (including the NUL), so a caller with a NULL buffer or
// a short one can size and retry from the same status.
sb_status copy_explain_text(const std::string &text, char *buf, size_t cap,
                            size_t *len_out, sb_error **err_out) {
  if (len_out == nullptr) return fail_arg(err_out);
  const size_t need = text.size() + 1;
  *len_out = need;
  if (buf == nullptr || cap < need) {
    set_error(err_out, SB_ERRKIND_BINDING,
              "sb_explain: buffer too small (see *len_out)");
    return SB_ERR_BINDING;
  }
  std::memcpy(buf, text.c_str(), need);
  return SB_OK;
}

} // namespace

sb_status sb_explain_table(char *buf, size_t cap, size_t *len_out,
                           sb_error **err_out) {
  return guard(
      [&] {
        planner::decisions::seed_static_knobs();
        return copy_explain_text(planner::decisions::dump_text(), buf, cap,
                                 len_out, err_out);
      },
      err_out);
}

sb_status sb_explain_json(char *buf, size_t cap, size_t *len_out,
                          sb_error **err_out) {
  return guard(
      [&] {
        planner::decisions::seed_static_knobs();
        return copy_explain_text(planner::decisions::dump_json(), buf, cap,
                                 len_out, err_out);
      },
      err_out);
}
