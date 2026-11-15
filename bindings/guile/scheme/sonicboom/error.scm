;;; SonicBoom — Guile binding: the C ABI error type
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Every failure that comes from libsonicboom.so is raised as one Guile 3
;;; exception type, &sonicboom-error, carrying the three things the C ABI
;;; reports: the boom_status code, the boom_error_kind as a Scheme symbol, and the
;;; message ("" when the C side had no diagnostic).
;;;
;;;   (use-modules (sonicboom error))
;;;   (guard (c ((sonicboom-error? c)
;;;              (list (sonicboom-error-status c)
;;;                    (sonicboom-error-kind c)
;;;                    (sonicboom-error-message c))))
;;;     (load-model "not-a-gguf"))
;;;
;;; The kind symbols are the boom_error_kind vocabulary in capi.h: none,
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

;; The condition type of every failure raised by libsonicboom.so.
;;
;; It carries three fields: status, the boom_status code as an exact integer; kind,
;; the boom_error_kind as a Scheme symbol; and message, the diagnostic string (empty
;; when the C side had none). The kind symbols are none, parse-lexical,
;; parse-syntax, parse-semantic, parse-unsupported, weight, compile, binding,
;; runtime, internal and cancelled.
;; PyTorch: none
;; Note: PyTorch raises its own Python or C++ exception classes; this is the single
;; error type the C ABI reports.
(define-condition-type &sonicboom-error &error
  sonicboom-error?
  (status sonicboom-error-status)     ; the boom_status code
  (kind sonicboom-error-kind)         ; a symbol, see below
  (message sonicboom-error-message))  ; a string, possibly ""

;; boom_error_kind (capi.h) -> symbol.
(define (sb-error-kind->symbol code)
  "Translate an boom_error_kind code to its Scheme symbol.

Args:
  code: the numeric boom_error_kind.
Returns: the matching symbol, or unknown for a code outside the vocabulary.
Example:
  (use-modules (sonicboom error))
  (list (sb-error-kind->symbol 0) (sb-error-kind->symbol 8)
        (sb-error-kind->symbol 99))
  => (none runtime unknown)
PyTorch: none"
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
  "Raise a &sonicboom-error with an explicit status, kind and message.

This is the constructor used inside the binding; application code normally
catches the condition instead of raising it.
Args:
  status: an boom_status code.
  kind: a symbol from the boom_error_kind vocabulary.
  message: a string, possibly empty.
Returns: does not return; the exception is raised.
PyTorch: none"
  (raise (condition (&sonicboom-error (status status) (kind kind)
                                      (message message)))))
