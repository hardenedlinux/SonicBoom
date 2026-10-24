;;; SonicBoom — Guile binding: the C ABI error type
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Every failure that comes from libsonicboom.so is raised as one Guile 3
;;; exception type, &sonicboom-error, carrying the three things the C ABI
;;; reports: the sb_status code, the sb_error_kind as a Scheme symbol, and the
;;; message ("" when the C side had no diagnostic).
;;;
;;;   (use-modules (sonicboom error))
;;;   (guard (c ((sonicboom-error? c)
;;;              (list (sonicboom-error-status c)
;;;                    (sonicboom-error-kind c)
;;;                    (sonicboom-error-message c))))
;;;     (load-model "not-a-gguf"))
;;;
;;; The kind symbols are the sb_error_kind vocabulary in capi.h: none,
;;; parse-lexical, parse-syntax, parse-semantic, parse-unsupported, weight,
;;; compile, binding, runtime, internal, cancelled.

(define-module (sonicboom error)
  #:use-module (srfi srfi-35)
  ;; SRFI-34's `raise` is Guile 3's raise-exception; the default binding in this
  ;; Guile (3.0.9) rejects a bare SRFI-35 condition with "Wrong type (expecting
  ;; exact integer)". Selecting it here makes every raise of
  ;; &sonicboom-error go through the exception protocol, whoever the caller is.
  #:use-module ((srfi srfi-34) #:select (raise))
  #:export (&sonicboom-error sonicboom-error?
            sonicboom-error-status sonicboom-error-kind sonicboom-error-message
            raise-sonicboom-error sb-error-kind->symbol))

(define-condition-type &sonicboom-error &error
  sonicboom-error?
  (status sonicboom-error-status)     ; the sb_status code
  (kind sonicboom-error-kind)         ; a symbol, see below
  (message sonicboom-error-message))  ; a string, possibly ""

;; sb_error_kind (capi.h) -> symbol.
(define (sb-error-kind->symbol code)
  (case code
    ((0) 'none)
    ((1) 'parse-lexical)
    ((2) 'parse-syntax)
    ((3) 'parse-semantic)
    ((4) 'parse-unsupported)
    ((5) 'weight)
    ((6) 'compile)
    ((7) 'binding)
    ((8) 'runtime)
    ((9) 'internal)
    ((10) 'cancelled)
    (else 'unknown)))

(define (raise-sonicboom-error status kind message)
  "Raise a &sonicboom-error with STATUS (an sb_status code), KIND (a symbol
from sb_error_kind) and MESSAGE (a string)."
  (raise (condition (&sonicboom-error (status status) (kind kind)
                                      (message message)))))
