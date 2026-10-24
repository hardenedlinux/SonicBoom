// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
//
// Private to the C ABI implementation (not installed, not a public header):
// the sb_error handle's definition and the value-or-error helpers shared by the
// capi translation units. Nothing here is part of the stable ABI.

#pragma once

#include <sonicboom/capi.h>

#include <new>
#include <string>

struct sb_error {
  sb_error_kind kind;
  std::string message;
};

namespace sonicboom::capi_private {

inline void set_error(sb_error **err_out, sb_error_kind kind,
                      const char *message) noexcept {
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

}  // namespace sonicboom::capi_private
