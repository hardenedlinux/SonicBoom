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
  #:use-module (json)
  #:use-module (ice-9 threads)
  #:export (libsonicboom-available?
            library-load-error
            exception-text
            run-s-expr
            build-info
            sb-status-ok?
            sb-status->message
            SB_ERR_BINDING
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
(define SB_ERR_UNSUPPORTED 7)

(define (sb-status-ok? status) (= status SB_OK))

(define *status-names*
  `((,SB_OK . "ok") (,SB_ERR_PARSE . "parse") (,SB_ERR_WEIGHT . "weight")
    (,SB_ERR_COMPILE . "compile") (,SB_ERR_BINDING . "binding")
    (,SB_ERR_RUNTIME . "runtime") (,SB_ERR_INTERNAL . "internal")
    (,SB_ERR_UNSUPPORTED . "unsupported")))

(define (sb-status->message status)
  (or (assq-ref *status-names* status) "unknown"))

(define %null %null-pointer)

;;; --- library handle (lazily loaded; cache once resolved) --------------------

(define %lib (make-parameter #f))       ; #f = not yet attempted; 'unavailable
(define %lib-mutex (make-mutex))
;; Why the last dynamic-link failed, for callers that want to report it rather
;; than just say "could not load". #f until a load is attempted.
(define %lib-error (make-parameter #f))

(define (library-load-error)
  "The dynamic-linker/exception text from the failed load of
libsonicboom.so, or #f if no load has been attempted or it succeeded."
  (%lib-error))

(define (exception-text key args)
  "A one-line, human-readable rendering of a caught exception from its
`catch` key and args (the exception object itself is not needed). This is
what the dynamic loader's failure looks like: a misc-error whose message is
the procedure name (\"dlopen\") and whose irritants carry the dlerror text."
  (cond
    ((eq? key 'system-error)
     (format #f "~a (errno ~a)" (cadr args) (car args)))
    ((and (eq? key 'misc-error) (>= (length args) 3)
          (string? (list-ref args 1)) (list? (list-ref args 2)))
     (let* ((message (list-ref args 0))
            (fmt (list-ref args 1))
            (irritants (list-ref args 2))
            (rendered (catch #t
                        (lambda () (apply format #f fmt irritants))
                        (lambda ignored fmt))))
       (if message (format #f "~a: ~a" message rendered) rendered)))
    (else (format #f "~a: ~s" key args))))

(define (libsonicboom-available?)
  (unless (%lib)
    (lock-mutex %lib-mutex)
    (unless (%lib)
      (let ((handle
             (catch #t
               (lambda () (dynamic-link "libsonicboom.so"))
               (lambda (key . args)
                 (%lib-error (exception-text key args))
                 #f))))
        (%lib (or handle 'unavailable))))
    (unlock-mutex %lib-mutex))
  (let ((h (%lib)))
    (and h (not (eq? h 'unavailable)))))

(define (ensure-lib)
  (unless (libsonicboom-available?)
    (error (string-append
            "libsonicboom.so could not be loaded"
            (if (%lib-error) (string-append ": " (%lib-error)) "")
            "; is it built and on the library search path?"))))

;;; --- foreign procedure table (populated on first use) -----------------------

(define %procs (make-parameter #f))

(define (foreign-proc name return-type arg-types)
  (ensure-lib)
  (pointer->procedure return-type (dynamic-func name (%lib)) arg-types))

(define (ensure-procs)
  (or (%procs)
      (let ((table
             `((sb_build_info . ,(foreign-proc "sb_build_info" '* (list)))
               (sb_parse . ,(foreign-proc "sb_parse" int
                                          (list '* size_t '* '*)))
               (sb_compile . ,(foreign-proc "sb_compile" int
                                            (list '* '* '* '*)))
               (sb_input_info . ,(foreign-proc "sb_input_info" int
                                               (list '* '* '* '* '*)))
               (sb_output_info . ,(foreign-proc "sb_output_info" int
                                                (list '* '* '* '* '*)))
               ;; device training graph (sb_compile_device) — see capi.h
               (sb_compile_device . ,(foreign-proc "sb_compile_device" int
                                                   (list '* '* '*)))
               (sb_input_count . ,(foreign-proc "sb_input_count" int
                                                (list '* '*)))
               (sb_output_count . ,(foreign-proc "sb_output_count" int
                                                 (list '* '*)))
               (sb_input_info_at . ,(foreign-proc "sb_input_info_at" int
                                                  (list '* size_t '* '* '* '*)))
               (sb_output_info_at . ,(foreign-proc "sb_output_info_at" int
                                                   (list '* size_t '* '* '* '*)))
               (sb_input_name_at . ,(foreign-proc "sb_input_name_at" int
                                                  (list '* size_t '*)))
               (sb_output_name_at . ,(foreign-proc "sb_output_name_at" int
                                                   (list '* size_t '*)))
               (sb_run_multi . ,(foreign-proc "sb_run_multi" int
                                              (list '* '* '* size_t '* '* size_t
                                                    '* '*)))
               (sb_run_resident . ,(foreign-proc "sb_run_resident" int
                                                 (list '* '* '* size_t '* size_t
                                                       '* '* '* '*)))
               (sb_input_is_resident . ,(foreign-proc "sb_input_is_resident" int
                                                      (list '* size_t '* '*)))
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
               (sb_clip_grad_norm . ,(foreign-proc "sb_clip_grad_norm" int
                                                   (list '* int float '* '*)))
               ;; export: graph text + named trained tensors → artifact dir
               (sb_export_model . ,(foreign-proc "sb_export_model" int
                                                 (list '* size_t '* '* size_t '* '*)))
               ;; checkpoints: named float32 tensors ↔ one binary file
               (sb_save_params . ,(foreign-proc "sb_save_params" int
                                                (list '* '* '* size_t '*)))
               (sb_load_params . ,(foreign-proc "sb_load_params" int
                                                (list '* '* '* size_t '*)))
               ;; numerics profile (process-wide, latched; see capi.h)
               (sb_set_numerics_profile . ,(foreign-proc "sb_set_numerics_profile" int
                                                         (list int '*)))
               (sb_get_numerics_profile . ,(foreign-proc "sb_get_numerics_profile" int
                                                         (list '* '*)))
               ;; inference (minimal LLM surface) — see capi.h sb_llm_*
               (sb_llm_model_load . ,(foreign-proc "sb_llm_model_load" int
                                                   (list '* '* '*)))
               (sb_llm_model_free . ,(foreign-proc "sb_llm_model_free" void (list '*)))
               (sb_llm_model_arch . ,(foreign-proc "sb_llm_model_arch" int
                                                   (list '* '* size_t '* '*)))
               (sb_llm_model_eos . ,(foreign-proc "sb_llm_model_eos" int
                                                  (list '* '* size_t '* '*)))
               (sb_llm_tokenize . ,(foreign-proc "sb_llm_tokenize" int
                                                 (list '* '* size_t '* size_t '* '*)))
               (sb_llm_tokenize_ex . ,(foreign-proc "sb_llm_tokenize_ex" int
                                                    (list '* '* size_t int '* size_t '* '*)))
               (sb_llm_detokenize . ,(foreign-proc "sb_llm_detokenize" int
                                                   (list '* '* size_t '* size_t '* '*)))
               (sb_llm_session_new . ,(foreign-proc "sb_llm_session_new" int
                                                    (list '* uint64 '* '*)))
               (sb_llm_session_free . ,(foreign-proc "sb_llm_session_free" void (list '*)))
               (sb_llm_prefill . ,(foreign-proc "sb_llm_prefill" int
                                                (list '* '* size_t '* '*)))
               (sb_llm_decode . ,(foreign-proc "sb_llm_decode" int
                                               (list '* '* '* '* '* '*))))))
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

;;; --- build info -------------------------------------------------------------

;;; The C ABI reports how libsonicboom.so was configured as a JSON object (see
;;; sb_build_info in capi.h). It is JSON rather than a struct so the field set
;;; can grow without breaking a caller; here it is decoded once and the string
;;; keys are turned into symbols, so a caller writes
;;;
;;;   (assq-ref (build-info) 'git-commit)
;;;   (assq-ref (assq-ref (build-info) 'cuda) 'archs)   ; => ("86")
;;;
;;; Nested objects (cuda, llvm) become alists too; arrays become lists.
(define (json-key->symbol key) (string->symbol key))

(define (normalize-json node)
  (cond ((pair? node)                       ; a JSON object: ((key . value) ...)
         (map (lambda (entry)
                (cons (json-key->symbol (car entry))
                      (normalize-json (cdr entry))))
              node))
        ((vector? node) (map normalize-json (vector->list node)))
        (else node)))

(define (build-info)
  "How the loaded libsonicboom.so was built, as a symbol-keyed alist: git
commit, build type, CUDA (enabled/archs/toolkit), LLVM (version/major/origin),
and target platform. Signals an error if the library cannot be loaded."
  (normalize-json
   (with-input-from-string
       (pointer->string ((proc 'sb_build_info)))
     json->scm)))

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
