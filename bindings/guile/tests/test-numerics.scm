;;; SonicBoom — Guile numerics-profile binding test
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Checks the process-wide numerics profile through (sonicboom inference):
;;;
;;;   1. numerics-profile returns 'strict or 'standard, matching
;;;      SONICBOOM_NUMERICS when the test is run with it set (the ctest entries
;;;      run this file once per profile);
;;;   2. re-selecting the effective profile is a no-op success;
;;;   3. selecting the other profile after it is latched signals, because the
;;;      profile is part of the plan, autotune and codegen cache keys.
;;;
;;; Needs no model and no GPU: the profile resolves without one.

(use-modules (srfi srfi-1)
             (srfi srfi-34)
             (srfi srfi-64)
             (sonicboom error)
             (sonicboom inference))

(test-begin "sonicboom-numerics")

(define effective (numerics-profile))

(test-assert "profile is strict or standard"
  (memq effective '(strict standard)))

(let ((want (getenv "SONICBOOM_NUMERICS")))
  (when (and want (string=? want "strict"))
    (test-equal "SONICBOOM_NUMERICS=strict is the effective profile"
      'strict effective))
  (when (and want (string=? want "standard"))
    (test-equal "SONICBOOM_NUMERICS=standard is the effective profile"
      'standard effective)))

(test-equal "re-selecting the effective profile succeeds"
  effective (set-numerics-profile! effective))

(test-assert "switching after the latch signals"
  (guard (c ((sonicboom-error? c) (eq? 'binding (sonicboom-error-kind c))))
    (set-numerics-profile! (if (eq? effective 'strict) 'standard 'strict))
    #f))

(test-equal "the profile did not move"
  effective (numerics-profile))

(test-end "sonicboom-numerics")
