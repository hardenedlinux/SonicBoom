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
  #:use-module (srfi srfi-34)
  #:use-module (json)
  #:use-module (ice-9 threads)
  #:use-module (sonicboom error)
  #:export (libsonicboom-available?
            library-load-error
            exception-text
            run-s-expr
            build-info
            sb-status-ok?
            sb-status->message
            sb-check!
            SB_ERR_BINDING
            SB_ERR_INTERNAL
            SB_ERR_CANCELLED
            ;; low-level FFI plumbing exposed for the training module
            proc make-slot slot-ptr slot-ref capture-error))

;;; --- status codes (mirror capi.h sb_status) --------------------------------

(define SB_OK 0)
(define SB_ERR_PARSE 1)
(define SB_ERR_WEIGHT 2)
(define SB_ERR_COMPILE 3)
;; The status of a parameter-binding failure.
;;
;; The value is 4, one of the sb_status codes of the C ABI; sb-status->message
;; renders it as "binding".
;;
;; Example:
;;   (use-modules (sonicboom ffi))
;;   (sb-status->message SB_ERR_BINDING)
;;   => "binding"
;; PyTorch: none
(define SB_ERR_BINDING 4)
(define SB_ERR_RUNTIME 5)
;; The status of an internal failure.
;;
;; The value is 6, one of the sb_status codes of the C ABI; sb-status->message
;; renders it as "internal".
;;
;; Example:
;;   (use-modules (sonicboom ffi))
;;   (sb-status->message SB_ERR_INTERNAL)
;;   => "internal"
;; PyTorch: none
(define SB_ERR_INTERNAL 6)
(define SB_ERR_UNSUPPORTED 7)
;; The status of a call that was cancelled.
;;
;; The value is 8, one of the sb_status codes of the C ABI; sb-status->message
;; renders it as "cancelled".
;;
;; Example:
;;   (use-modules (sonicboom ffi))
;;   (sb-status->message SB_ERR_CANCELLED)
;;   => "cancelled"
;; PyTorch: none
(define SB_ERR_CANCELLED 8)

(define (sb-status-ok? status)
  "Whether a C ABI status code is SB_OK.

Args:
  status: a status code.
Returns: #t for SB_OK (0), #f otherwise.
Example:
  (use-modules (sonicboom ffi))
  (sb-status-ok? 0)
  => #t
PyTorch: none"
  (= status SB_OK))

(define *status-names*
  `((,SB_OK . "ok") (,SB_ERR_PARSE . "parse") (,SB_ERR_WEIGHT . "weight")
    (,SB_ERR_COMPILE . "compile") (,SB_ERR_BINDING . "binding")
    (,SB_ERR_RUNTIME . "runtime") (,SB_ERR_INTERNAL . "internal")
    (,SB_ERR_UNSUPPORTED . "unsupported") (,SB_ERR_CANCELLED . "cancelled")))

(define (sb-status->message status)
  "The name of a C ABI status code.

The names are ok, parse, weight, compile, binding, runtime, internal,
unsupported and cancelled.

Args:
  status: a status code.
Returns: the name as a string, or \"unknown\" for a code outside the set.
Example:
  (use-modules (sonicboom ffi))
  (sb-status->message SB_ERR_CANCELLED)
  => \"cancelled\"
PyTorch: none"
  (or (assq-ref *status-names* status) "unknown"))

(define %null %null-pointer)

;;; --- library handle (lazily loaded; cache once resolved) --------------------

(define %lib (make-parameter #f))       ; #f = not yet attempted; 'unavailable
(define %lib-mutex (make-mutex))
;; Why the last dynamic-link failed, for callers that want to report it rather
;; than just say "could not load". #f until a load is attempted.
(define %lib-error (make-parameter #f))

(define (library-load-error)
  "Why the last attempt to load libsonicboom.so failed.

The text is the one from the dynamic linker, rendered by exception-text; it is
kept after a failed load so a caller can report it instead of just saying the
library is unavailable.

Args:
  none.
Returns: a string, or #f when no load has been attempted or the last one
  succeeded.
Example:
  (use-modules (sonicboom ffi))
  (begin (libsonicboom-available?) (library-load-error))
  => #f
PyTorch: none"
  (%lib-error))

(define (exception-text key args)
  "Render a caught exception as one line of text.

The exception object itself is not needed: KEY and ARGS are what a catch
handler receives. A system-error is rendered from its errno and message; a
misc-error from its message, format string and irritants; anything else as the
key followed by the arguments.

Args:
  key: the catch key.
  args: the catch arguments.
Returns: a string.
Example:
  (use-modules (sonicboom ffi))
  (exception-text 'system-error '(2 \"No such file\" (2 . \"open\")))
  => \"No such file (errno 2)\"
PyTorch: none"
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
  "Whether libsonicboom.so can be loaded.

The library is loaded on the first call and the result is cached, so later
calls do not go to the dynamic linker again. A missing library is not an
error here; library-load-error tells why it was missing.

Args:
  none.
Returns: #t when the library is loaded, #f otherwise.
Example:
  (use-modules (sonicboom ffi))
  (libsonicboom-available?)
  => #t
PyTorch: none"
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
               ;; read-only parameter views (see capi.h)
               (sb_api_version . ,(foreign-proc "sb_api_version" uint32 (list)))
               (sb_executable_parameter_count
                . ,(foreign-proc "sb_executable_parameter_count" int
                                 (list '* '* '*)))
               (sb_executable_parameter_name_at
                . ,(foreign-proc "sb_executable_parameter_name_at" int
                                 (list '* size_t '* '*)))
               (sb_executable_parameter_desc
                . ,(foreign-proc "sb_executable_parameter_desc" int
                                 (list '* size_t '* '*)))
               (sb_executable_parameter_view
                . ,(foreign-proc "sb_executable_parameter_view" int
                                 (list '* size_t '* '*)))
               (sb_storage_data . ,(foreign-proc "sb_storage_data" int
                                                 (list '* '* '* '*)))
               (sb_storage_desc . ,(foreign-proc "sb_storage_desc" int
                                                 (list '* '* '*)))
               (sb_storage_release . ,(foreign-proc "sb_storage_release" void
                                                    (list '*)))
               ;; deferred (:bound) parameters (see capi.h)
               (sb_bind_parameter . ,(foreign-proc "sb_bind_parameter" int
                                                   (list '* '* '* size_t '*)))
               (sb_bindings_complete
                . ,(foreign-proc "sb_bindings_complete" int
                                 (list '* '* '*)))
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
               (sb_error_get_kind . ,(foreign-proc "sb_error_get_kind" int
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
               ;; planner decision registry (see (sonicboom explain))
               (sb_explain_table . ,(foreign-proc "sb_explain_table" int
                                                  (list '* size_t '* '*)))
               (sb_explain_json . ,(foreign-proc "sb_explain_json" int
                                                 (list '* size_t '* '*)))
               ;; training parameters (fused AdamW) — see (sonicboom param)
               (sb_param_create . ,(foreign-proc "sb_param_create" int
                                                 (list '* int64 '* size_t '* '*)))
               (sb_param_numel . ,(foreign-proc "sb_param_numel" int
                                                (list '* '*)))
               (sb_param_on_device . ,(foreign-proc "sb_param_on_device" int
                                                    (list '*)))
               (sb_param_step_count . ,(foreign-proc "sb_param_step_count" uint64
                                                     (list '*)))
               (sb_param_zero_grad . ,(foreign-proc "sb_param_zero_grad" int
                                                    (list '* '*)))
               (sb_param_set_grad . ,(foreign-proc "sb_param_set_grad" int
                                                   (list '* '* size_t '*)))
               (sb_param_step . ,(foreign-proc "sb_param_step" int
                                               (list '* float float float float
                                                     float '*)))
               (sb_param_step_lr . ,(foreign-proc "sb_param_step_lr" int
                                                  (list '* '* float float float
                                                        float '*)))
               (sb_param_read . ,(foreign-proc "sb_param_read" int
                                               (list '* '* size_t '* '*)))
               (sb_param_free . ,(foreign-proc "sb_param_free" void (list '*)))
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
                                               (list '* '* '* '* '* '*)))
               ;; constrained decoding (session grammar) — see capi.h
               (sb_llm_json_schema_to_gbnf
                . ,(foreign-proc "sb_llm_json_schema_to_gbnf" int
                                 (list '* '* '* '* size_t '* '*)))
               (sb_llm_session_set_grammar . ,(foreign-proc "sb_llm_session_set_grammar" int
                                                            (list '* '* '*)))
               (sb_llm_session_set_json_schema . ,(foreign-proc "sb_llm_session_set_json_schema"
                                                                int (list '* '* '*)))
               (sb_llm_session_clear_grammar . ,(foreign-proc "sb_llm_session_clear_grammar"
                                                              void (list '*)))
               (sb_llm_session_constraint_state . ,(foreign-proc "sb_llm_session_constraint_state"
                                                                 int (list '* '* '* '*)))
               ;; image generation (MOSS-Image) — see capi.h sb_image_*
               (sb_image_model_load . ,(foreign-proc "sb_image_model_load" int
                                                     (list '* '* '*)))
               (sb_image_model_free . ,(foreign-proc "sb_image_model_free" void
                                                     (list '*)))
               (sb_image_model_arch . ,(foreign-proc "sb_image_model_arch" int
                                                     (list '* '* size_t '* '*)))
               (sb_image_generate . ,(foreign-proc "sb_image_generate" int
                                                   (list '* '* '* '* '* size_t
                                                         '* '*)))
               (sb_image_write_png . ,(foreign-proc "sb_image_write_png" int
                                                    (list '* '* uint32 uint32
                                                          '*))))))
        (%procs table)
        table)))

(define (proc name)
  "The foreign procedure registered under NAME.

The procedure table is resolved from libsonicboom.so on first use and then
cached; NAME is the C function name as a symbol, for example 'sb_build_info.
This is plumbing for the modules that call the C ABI directly.

Args:
  name: the C function name as a symbol.
Returns: a foreign procedure, or #f when the table has no such entry.
Example:
  (use-modules (sonicboom ffi))
  (procedure? (proc 'sb_build_info))
  => #t
PyTorch: none"
  (assq-ref (ensure-procs) name))

;;; --- pointer helpers --------------------------------------------------------

;; A "slot" is a zeroed bytevector of one machine pointer, used as an out-param
;; (T**). Read it back with `slot-ref`.
(define (make-slot)
  "A fresh out-parameter slot.

A slot is a zeroed bytevector holding one machine pointer. It is passed by its
address to C functions that return a pointer through a T** out-parameter, and
slot-ref reads the pointer back.

Args:
  none.
Returns: a bytevector.
Example:
  (use-modules (sonicboom ffi) (rnrs bytevectors))
  (bytevector? (make-slot))
  => #t
PyTorch: none"
  (make-bytevector (sizeof '*) 0))
(define (slot-ptr slot)
  "The address of SLOT, to pass to a C ABI call.

Args:
  slot: a slot.
Returns: a pointer.
Example:
  (use-modules (sonicboom ffi) (system foreign))
  (pointer? (slot-ptr (make-slot)))
  => #t
PyTorch: none"
  (bytevector->pointer slot))
(define (slot-ref slot)
  "The pointer stored in SLOT.

A slot that no call has filled still holds the null pointer.

Args:
  slot: a slot.
Returns: a pointer.
Example:
  (use-modules (sonicboom ffi) (system foreign))
  (null-pointer? (slot-ref (make-slot)))
  => #t
PyTorch: none"
  (dereference-pointer (slot-ptr slot)))
(define (slot-null! slot)
  (bytevector-fill! slot 0))

;;; --- error capture ----------------------------------------------------------

(define (capture-error err-slot)
  "Take the message out of an error slot, then free the error.

When the slot holds an error object its message is returned, the object is
freed and the slot is set to the null pointer. A slot without an error gives
the empty string.

Args:
  err-slot: the slot that received the call's error object.
Returns: a string.
Example:
  (use-modules (sonicboom ffi))
  (capture-error (make-slot))
  => \"\"
PyTorch: none"
  (let* ((p (slot-ref err-slot))
         (msg (if (null-pointer? p)
                  ""
                  (pointer->string ((proc 'sb_error_message) p)))))
    (unless (null-pointer? p)
      ((proc 'sb_error_free) p)
      (slot-null! err-slot))
    msg))

;;; --- status/error raising ----------------------------------------------------

;; The kind the sb_error in ERR-SLOT reports, read before capture-error frees it.
;; 0 (SB_ERRKIND_NONE) when the call left no error object.
(define (error-kind-code err-slot)
  (let ((p (slot-ref err-slot)))
    (if (null-pointer? p) 0 ((proc 'sb_error_get_kind) p))))

;; When the C side left no error object, the status code still says roughly
;; what kind of failure it was.
(define (status-kind status)
  (case status
    ((SB_ERR_PARSE) 'parse-syntax)
    ((SB_ERR_WEIGHT) 'weight)
    ((SB_ERR_COMPILE) 'compile)
    ((SB_ERR_BINDING) 'binding)
    ((SB_ERR_RUNTIME) 'runtime)
    ((SB_ERR_INTERNAL) 'internal)
    ((SB_ERR_CANCELLED) 'cancelled)
    (else 'none)))

(define (sb-check! status err-slot what)
  "Raise a Guile exception when a C ABI call failed.

When STATUS is SB_OK the call returns quietly. Otherwise a &sonicboom-error is
raised carrying STATUS, a kind and a message; the message is the error text
from ERR-SLOT, or the status name when the C side left no error object, and
WHAT is prefixed to it. The error object is freed either way.

Args:
  status: the status a C ABI call returned.
  err-slot: the slot that received the call's error object.
  what: a string prefixed to the message, or #f.
Returns: unspecified when the status is SB_OK.
Example:
  (use-modules (sonicboom ffi))
  (begin (sb-check! 0 (make-slot) \"demo\") #t)
  => #t
PyTorch: none"
  (unless (sb-status-ok? status)
    (let* ((code (error-kind-code err-slot))
           (msg (capture-error err-slot))
           (text (if (string-null? msg)
                     (string-append "C ABI error: " (sb-status->message status))
                     msg)))
      (raise-sonicboom-error
       status
       (if (zero? code) (status-kind status) (sb-error-kind->symbol code))
       (if what (string-append what ": " text) text)))))

;;; --- build info -------------------------------------------------------------

;;; The C ABI reports how libsonicboom.so was configured as a JSON object (see
;;; sb_build_info in capi.h). It is JSON rather than a struct so the field set
;;; can grow without breaking a caller; here it is decoded and the string
;;; keys are turned into symbols, so a caller writes
;;;
;;;   (assq-ref (build-info) 'git_commit)
;;;   (assq-ref (assq-ref (build-info) 'cuda) 'archs)   ; on a CUDA build
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
  "How the loaded libsonicboom.so was built.

The C ABI reports the configuration as JSON; here it is decoded, with the
string keys turned into symbols and nested objects turned into alists. The keys
are platform, llvm, cuda, build_type, git_tag and git_commit; llvm and cuda are
alists and their arch lists are lists. The library is loaded if it has not been
yet.

Args:
  none.
Returns: a symbol-keyed alist.
Example:
  (use-modules (sonicboom ffi))
  (string? (assq-ref (build-info) 'platform))
  => #t
PyTorch: none"
  (normalize-json
   (with-input-from-string
       (pointer->string ((proc 'sb_build_info)))
     json->scm)))

;;; --- high-level execution ---------------------------------------------------

(define (run-s-expr text input-bv)
  "Compile an S-Expr v0.1 graph and run it once.

The graph text is compiled and executed with INPUT-BV as the little-endian
input bytes. The intermediate document and executable are always released,
whether the run succeeds or not.

Args:
  text: the graph, in S-Expr v0.1 form.
  input-bv: the input bytes.
Returns: two values. On success the output bytevector and #f; on failure #f
  and an error-message string.
Example:
  (use-modules (sonicboom ffi) (rnrs bytevectors))
  (call-with-values
      (lambda () (run-s-expr \"not-a-graph\" (make-bytevector 0)))
    (lambda (output error) (list output (string? error))))
  => (#f #t)
PyTorch: none"
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
      (cleanup!)
      (sb-check! status err-slot "run-s-expr"))

    (guard (c ((sonicboom-error? c) (values #f (sonicboom-error-message c))))
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
                (values out-bv #f))))))))
