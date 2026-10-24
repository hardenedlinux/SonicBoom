;;; SonicBoom — Guile binding: the planner's decision registry
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The runtime resolves every automatic choice (reduction mode, spec probe,
;;; MMQ/attention implementation and tiles, dense path, prefetch, the training
;;; switches) once per device or static shape class and records it with its
;;; inputs, candidates, reason, numerics class and resource costs. The C ABI
;;; exports that registry (capi.h sb_explain_*) in the two views tools/sb-explain
;;; prints:
;;;
;;;   (explain-table) → the human table, as a string
;;;   (explain-json)  → the same registry as a parsed Scheme value
;;;
;;; Neither call tunes anything; they only report what this process and the
;;; per-device cache already decided.

(define-module (sonicboom explain)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (json)
  #:use-module (sonicboom ffi)
  #:export (explain-table explain-json))

(define %null %null-pointer)

;; Two-call sizing (capi.h): *len_out is the byte count including the NUL and
;; is written on both calls; a NULL buffer answers SB_ERR_BINDING, which is the
;; sizing answer rather than a failure. The error handle that call leaves behind
;; is freed unread.
(define (explain-bytes fn what)
  (let* ((len-bv (make-bytevector 8 0))       ; size_t
         (err-slot (make-slot)))
    (let ((st (fn %null 0 (bytevector->pointer len-bv) (slot-ptr err-slot))))
      (unless (= st SB_ERR_BINDING) (sb-check! st err-slot what))
      (capture-error err-slot))                ; the expected sizing error
    (let* ((n (bytevector-u64-native-ref len-bv 0))
           (out (make-bytevector n 0))
           (st (fn (bytevector->pointer out) n (bytevector->pointer len-bv)
                   (slot-ptr err-slot))))
      (sb-check! st err-slot what)
      out)))

(define (explain-table)
  "The planner's decision registry for the current device, rendered as the
same human-readable table tools/sb-explain prints. A string, newline-terminated."
  (let ((out (explain-bytes (proc 'sb_explain_table) "explain-table")))
    (pointer->string (bytevector->pointer out))))

;; JSON objects become symbol-keyed alists and arrays become lists, the same
;; shape (sonicboom ffi) `build-info` returns, so a caller writes
;; (assq-ref doc 'device) and (assq-ref (car (assq-ref doc 'decisions)) 'name).
(define (normalize-json node)
  (cond ((pair? node)
         (map (lambda (entry)
                (cons (string->symbol (car entry))
                      (normalize-json (cdr entry))))
              node))
        ((vector? node) (map normalize-json (vector->list node)))
        (else node)))

(define (explain-json)
  "The same decision registry as a parsed Scheme value: an alist with the keys
`version`, `device` and `decisions` (a list of decision alists)."
  (let ((out (explain-bytes (proc 'sb_explain_json) "explain-json")))
    (normalize-json
     (with-input-from-string (pointer->string (bytevector->pointer out))
       json->scm))))
