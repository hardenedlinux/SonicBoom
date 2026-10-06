;;; SonicBoom — Guile binding: C ABI FFI
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Loads libsonicboom.so (the published core library) and exposes the stable C
;;; ABI (capi/include/sonicboom/capi.h) through Guile's foreign-function
;;; interface. This is the *only* path Guile uses to reach the native backend:
;;;
;;;   Guile → (sonicboom ffi) → libsonicboom.so (C ABI) → core → native-torch
;;;
;;; Nothing in the binding reaches into core/layer1 or third_party/native-torch.

(define-module (sonicboom ffi)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 threads)
  #:export (libsonicboom-available?
            run-s-expr
            sb-status-ok?
            sb-status->message
            ;; low-level FFI plumbing exposed for the training module
            proc make-slot slot-ptr slot-ref capture-error))

;;; --- status codes (mirror capi.h sb_status) --------------------------------

(define SB_OK 0)
(define SB_ERR_PARSE 1)
(define SB_ERR_WEIGHT 2)
(define SB_ERR_COMPILE 3)
(define SB_ERR_BINDING 4)
(define SB_ERR_RUNTIME 5)
(define SB_ERR_INTERNAL 6)

(define (sb-status-ok? status) (= status SB_OK))

(define *status-names*
  `((,SB_OK . "ok") (,SB_ERR_PARSE . "parse") (,SB_ERR_WEIGHT . "weight")
    (,SB_ERR_COMPILE . "compile") (,SB_ERR_BINDING . "binding")
    (,SB_ERR_RUNTIME . "runtime") (,SB_ERR_INTERNAL . "internal")))

(define (sb-status->message status)
  (or (assq-ref *status-names* status) "unknown"))

(define %null %null-pointer)

;;; --- library handle (lazily loaded; cache once resolved) --------------------

(define %lib (make-parameter #f))       ; #f = not yet attempted; 'unavailable
(define %lib-mutex (make-mutex))

(define (libsonicboom-available?)
  (unless (%lib)
    (lock-mutex %lib-mutex)
    (unless (%lib)
      (let ((handle (false-if-exception (dynamic-link "libsonicboom.so"))))
        (%lib (or handle 'unavailable))))
    (unlock-mutex %lib-mutex))
  (let ((h (%lib)))
    (and h (not (eq? h 'unavailable)))))

(define (ensure-lib)
  (unless (libsonicboom-available?)
    (error "libsonicboom.so could not be loaded; is it built and on the \
library search path?")))

;;; --- foreign procedure table (populated on first use) -----------------------

(define %procs (make-parameter #f))

(define (foreign-proc name return-type arg-types)
  (ensure-lib)
  (pointer->procedure return-type (dynamic-func name (%lib)) arg-types))

(define (ensure-procs)
  (or (%procs)
      (let ((table
             `((sb_parse . ,(foreign-proc "sb_parse" int
                                          (list '* size_t '* '*)))
               (sb_compile . ,(foreign-proc "sb_compile" int
                                            (list '* '* '* '*)))
               (sb_input_info . ,(foreign-proc "sb_input_info" int
                                               (list '* '* '* '* '*)))
               (sb_output_info . ,(foreign-proc "sb_output_info" int
                                                (list '* '* '* '* '*)))
               (sb_bind_input . ,(foreign-proc "sb_bind_input" int
                                               (list '* '* size_t '*)))
               (sb_execute . ,(foreign-proc "sb_execute" int (list '* '*)))
               (sb_retrieve_output . ,(foreign-proc "sb_retrieve_output" int
                                                    (list '* '* size_t '* '*)))
               (sb_error_message . ,(foreign-proc "sb_error_message" '*
                                                  (list '*)))
               (sb_error_free . ,(foreign-proc "sb_error_free" void (list '*)))
               (sb_document_free . ,(foreign-proc "sb_document_free" void
                                                  (list '*)))
               (sb_executable_free . ,(foreign-proc "sb_executable_free" void
                                                    (list '*)))
               ;; training (minimal autograd) surface — see capi.h
               (sb_tensor_from_f32 . ,(foreign-proc "sb_tensor_from_f32" int
                                                    (list '* int64 '* size_t '* '*)))
               (sb_tensor_bytes . ,(foreign-proc "sb_tensor_bytes" int
                                                 (list '* '* size_t '* '*)))
               (sb_tensor_free . ,(foreign-proc "sb_tensor_free" void (list '*)))
               (sb_tape_new . ,(foreign-proc "sb_tape_new" int (list '* '*)))
               (sb_tape_free . ,(foreign-proc "sb_tape_free" void (list '*)))
               (sb_tape_zero_grad . ,(foreign-proc "sb_tape_zero_grad" void (list '*)))
               (sb_linear . ,(foreign-proc "sb_linear" int
                                          (list '* '* '* '* '* '*)))
               (sb_relu . ,(foreign-proc "sb_relu" int (list '* '* '* '*)))
               (sb_mse_loss . ,(foreign-proc "sb_mse_loss" int
                                            (list '* '* '* '* '*)))
               (sb_backward . ,(foreign-proc "sb_backward" int (list '* '* '*)))
               (sb_sgd_step . ,(foreign-proc "sb_sgd_step" int
                                            (list '* '* double '*)))
               (sb_grad . ,(foreign-proc "sb_grad" int (list '* '* '* '*)))
               ;; export: graph text + named trained tensors → artifact dir
               (sb_export_model . ,(foreign-proc "sb_export_model" int
                                                 (list '* size_t '* '* size_t '* '*))))))
        (%procs table)
        table)))

(define (proc name) (assq-ref (ensure-procs) name))

;;; --- pointer helpers --------------------------------------------------------

;; A "slot" is a zeroed bytevector of one machine pointer, used as an out-param
;; (T**). Read it back with `slot-ref`.
(define (make-slot) (make-bytevector (sizeof '*) 0))
(define (slot-ptr slot) (bytevector->pointer slot))
(define (slot-ref slot)
  (dereference-pointer (slot-ptr slot)))
(define (slot-null! slot)
  (bytevector-fill! slot 0))

;;; --- error capture ----------------------------------------------------------

(define (capture-error err-slot)
  "Return the message string of the sb_error held in ERR-SLOT, then free it and
NULL the slot."
  (let* ((p (slot-ref err-slot))
         (msg (if (null-pointer? p)
                  ""
                  (pointer->string ((proc 'sb_error_message) p)))))
    (unless (null-pointer? p)
      ((proc 'sb_error_free) p)
      (slot-null! err-slot))
    msg))

;;; --- high-level execution ---------------------------------------------------

(define (run-s-expr text input-bv)
  "Compile the S-Expr v0.1 graph TEXT and run it once with the little-endian
input bytes INPUT-BV. Returns two values on success: the output bytevector and
#f. On failure returns #f and an error-message string. Ownership of the
document/executable/error handles is always released."
  (ensure-lib)
  (let* ((doc-slot (make-slot))
         (exe-slot (make-slot))
         (err-slot (make-slot))
         (txt-ptr (string->pointer text))
         (len (string-length text)))
    (define (cleanup!)
      (unless (null-pointer? (slot-ref doc-slot))
        ((proc 'sb_document_free) (slot-ref doc-slot)))
      (unless (null-pointer? (slot-ref exe-slot))
        ((proc 'sb_executable_free) (slot-ref exe-slot))))
    (define (fail status)
      (let ((msg (capture-error err-slot)))
        (cleanup!)
        (throw 'sb-ffi
               (if (string-null? msg)
                   (string-append "C ABI error: "
                                  (sb-status->message status))
                   msg))))

    (catch 'sb-ffi
      (lambda ()
        (let ((st ((proc 'sb_parse) txt-ptr len (slot-ptr doc-slot)
                   (slot-ptr err-slot))))
          (unless (sb-status-ok? st) (fail st)))
        (let ((st ((proc 'sb_compile) (slot-ref doc-slot) (string->pointer "")
                   (slot-ptr exe-slot) (slot-ptr err-slot))))
          (unless (sb-status-ok? st) (fail st)))
        ;; Query the single output's byte size to allocate the destination
        ;; buffer. (uint64 is 8 bytes on every 64-bit target; `sizeof` only
        ;; accepts pointer types in Guile's FFI, so scalar sizes are hardcoded.)
        (let* ((num-slot (make-bytevector 8 0))
               (st ((proc 'sb_output_info) (slot-ref exe-slot) %null
                     %null %null (bytevector->pointer num-slot))))
          (unless (sb-status-ok? st) (fail st))
          (let ((out-len (bytevector-u64-native-ref num-slot 0)))
            (let ((out-bv (make-bytevector out-len 0)))
              (let ((st ((proc 'sb_bind_input) (slot-ref exe-slot)
                         (bytevector->pointer input-bv)
                         (bytevector-length input-bv)
                         (slot-ptr err-slot))))
                (unless (sb-status-ok? st) (fail st)))
              (let ((st ((proc 'sb_execute) (slot-ref exe-slot)
                         (slot-ptr err-slot))))
                (unless (sb-status-ok? st) (fail st)))
              (let* ((written-slot (make-bytevector 8 0))
                     (st ((proc 'sb_retrieve_output) (slot-ref exe-slot)
                           (bytevector->pointer out-bv) out-len
                           (bytevector->pointer written-slot)
                           (slot-ptr err-slot))))
                (unless (sb-status-ok? st) (fail st))
                (cleanup!)
                (values out-bv #f))))))
      (lambda (key msg) (values #f msg)))))
