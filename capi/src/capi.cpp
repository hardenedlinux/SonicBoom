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
#include <sonicboom/runtime.h>
#include <sonicboom/scalar_type.h>
#include <sonicboom/sx/exec.h>
#include <sonicboom/sx/ir.h>
#include <sonicboom/sx/parser.h>
#include <sonicboom/tensor.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sx = sonicboom::sx;

// --- Opaque handle definitions (private to this translation unit) -----------

struct sb_document {
  sx::Document doc;
};

struct sb_executable {
  std::unique_ptr<sx::Executable> exe;
  bool input_bound = false;
  sx::Bytes input;    // copied at bind time
  sx::Bytes output;   // retained after execute
  bool has_output = false;
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

} // namespace

sb_status sb_input_info(const sb_executable *exe, sb_dtype *dtype_out,
                        int64_t *rank_out, const int64_t **dims_out,
                        uint64_t *num_bytes_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe)
          return fail_arg(nullptr);
        fill_tensor_info(exe->exe->input_type(), dtype_out, rank_out, dims_out,
                         num_bytes_out);
        return SB_OK;
      },
      nullptr);
}

sb_status sb_output_info(const sb_executable *exe, sb_dtype *dtype_out,
                         int64_t *rank_out, const int64_t **dims_out,
                         uint64_t *num_bytes_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe)
          return fail_arg(nullptr);
        fill_tensor_info(exe->exe->output_type(), dtype_out, rank_out, dims_out,
                         num_bytes_out);
        return SB_OK;
      },
      nullptr);
}

// --- Execute / retrieve -----------------------------------------------------

sb_status sb_bind_input(sb_executable *exe, const void *data, size_t len,
                        sb_error **err_out) {
  return guard(
      [&]() -> sb_status {
        if (!exe || !data)
          return fail_arg(err_out);

        uint64_t expected = tensor_num_bytes(exe->exe->input_type());
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

        auto res = exe->exe->run(exe->input, exe->output);
        if (!res) {
          exe->has_output = false;
          set_error(err_out, exec_errkind(res.error().kind),
                    res.error().message.c_str());
          return exec_status(res.error().kind);
        }
        exe->has_output = true;
        return SB_OK;
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
