// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Private to the C ABI implementation (not installed, not a public header):
// the boom_error handle's definition and the value-or-error helpers shared by the
// capi translation units. Nothing here is part of the stable ABI.

#pragma once

#include <sonicboom/capi.h>

#include <cstddef>
#include <cstring>
#include <new>
#include <string>
#include <string_view>

struct boom_error {
  boom_error_kind kind;
  std::string message;
};

namespace sonicboom::capi_private {

inline void set_error(boom_error **err_out, boom_error_kind kind,
                      const char *message) noexcept {
  if (!err_out)
    return;
  // Reusing an error slot that already holds an error must not leak the
  // previous handle: free it first. The slot holds a handle we allocated via
  // boom_error_free, never a caller-owned pointer, so this is always safe and
  // never frees uninitialized memory (boom_error_free is null-safe).
  if (*err_out) {
    delete *err_out;
    *err_out = nullptr;
  }
  auto *err = new (std::nothrow) boom_error();
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

// Append a JSON string literal, escaped. The info documents only carry ASCII
// identifiers (architectures, quantizations), but escaping keeps them valid
// whatever a file declares.
inline void append_json_string(std::string &out, std::string_view s) {
  static constexpr char kHex[] = "0123456789abcdef";
  out += '"';
  for (const unsigned char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (c < 0x20) {
          out += "\\u00";
          out += kHex[c >> 4];
          out += kHex[c & 0xf];
        } else {
          out += static_cast<char>(c);
        }
    }
  }
  out += '"';
}

// The two-call buffer contract of the JSON document outputs (the model-info
// calls, like boom_explain_json): *len_out always receives the needed byte
// count including the NUL; a NULL or short buffer answers BOOM_ERR_BINDING,
// which is the sizing answer rather than a failure.
inline boom_status copy_json_document(const std::string &text, char *out,
                                      size_t cap, size_t *len_out,
                                      boom_error **err_out,
                                      const char *what) noexcept {
  if (len_out == nullptr) {
    set_error(err_out, BOOM_ERRKIND_INTERNAL, "invalid argument");
    return BOOM_ERR_INTERNAL;
  }
  const size_t need = text.size() + 1;
  *len_out = need;
  if (out == nullptr || cap < need) {
    set_error(err_out, BOOM_ERRKIND_BINDING,
              (std::string(what) + ": buffer too small (see *len_out)").c_str());
    return BOOM_ERR_BINDING;
  }
  std::memcpy(out, text.c_str(), need);
  return BOOM_OK;
}

// Contain every C++ exception so none escapes an exported `boom_*` function.
// Each status-returning function runs its body through this: an expected
// exception (std::exception, including bad_alloc) or an unexpected one becomes
// BOOM_ERR_INTERNAL with a best-effort diagnostic. The function is noexcept so a
// throw during set_error's own (nothrow) allocation path cannot escape.
template <typename Fn>
boom_status guard(Fn &&fn, boom_error **err_out) noexcept {
  try {
    return fn();
  } catch (const std::bad_alloc &) {
    set_error(err_out, BOOM_ERRKIND_INTERNAL, "out of memory");
    return BOOM_ERR_INTERNAL;
  } catch (const std::exception &e) {
    set_error(err_out, BOOM_ERRKIND_INTERNAL, e.what());
    return BOOM_ERR_INTERNAL;
  } catch (...) {
    set_error(err_out, BOOM_ERRKIND_INTERNAL, "unknown exception");
    return BOOM_ERR_INTERNAL;
  }
}

}  // namespace sonicboom::capi_private
