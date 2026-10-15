;;; SonicBoom — Guile binding: compile-once executables for S-Expr graphs
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; `(sonicboom ffi) run-s-expr` parses and compiles the graph text on every
;;; call. That is fine for a one-shot graph but wasteful for a hot loop that
;;; runs the same graph many times (model-forward in a training loop, a traced
;;; segment replayed per token). This module exposes the two halves separately:
;;;
;;;   (compile-s-expr text)      → an <executable>, compiled once
;;;   (executable-run exe input) → bind / execute / retrieve, many times
;;;   (free-executable! exe)     → release the native handle
;;;
;;; The native handle is owned by the <executable> record. If the record is
;;; collected without an explicit free, a GC guardian releases the handle when
;;; `reap-executables!` is next called; `free-executable!` is idempotent, so the
;;; two never double-free. (This Guile has no `set-finalizer!`.)
;;;
;;; `run-s-expr/cached` is a drop-in replacement for `run-s-expr` that keys a
;;; process-wide cache by graph text (enable/disable with the
;;; `executable-cache-enabled` parameter). The cache is never invalidated
;;; automatically — `clear-executable-cache!` drops every entry.

(define-module (sonicboom executable)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 hash-table)
  #:use-module (ice-9 threads)
  #:use-module (sonicboom ffi)
  #:export (compile-s-expr
            executable? executable-run free-executable!
            executable-freed? executable-graph-text
            executable-input-nbytes executable-output-nbytes
            cached-executable run-s-expr/cached
            clear-executable-cache! executable-cache-enabled
            reap-executables!))

(define %null %null-pointer)

;;; --- the executable record ---------------------------------------------------

(define-record-type <executable>
  (%make-executable text ptr in-nbytes out-nbytes freed)
  executable?
  (text executable-graph-text)
  (ptr executable-ptr)
  (in-nbytes executable-input-nbytes)
  (out-nbytes executable-output-nbytes)
  ;; Flipped to #t by whoever releases the handle first (explicit free or the
  ;; finalizer), so the other one becomes a no-op instead of a double free.
  (freed executable-freed? set-executable-freed!))

(define (free-executable! exe)
  "Release EXE's native handle. Idempotent: freeing an already-freed (or
guardian-reaped) executable is a no-op, never an error."
  (unless (executable-freed? exe)
    (set-executable-freed! exe #t)      ; claim it before touching the pointer
    (let ((p (executable-ptr exe)))
      (unless (null-pointer? p)
        ((proc 'sb_executable_free) p))))
  *unspecified*)

;;; --- GC fallback -------------------------------------------------------------

;; Every compiled executable is registered with this guardian. It holds only a
;; weak reference, so a record that becomes unreachable is handed back by the
;; next `reap-executables!` call, which then releases its native handle. This
;; is the safety net for callers that never free explicitly.
(define %exe-guardian (make-guardian))

(define (reap-executables!)
  "Release every executable whose record has been collected without an explicit
`free-executable!`. Returns the number of handles released."
  (let loop ((n 0))
    (let ((dead (%exe-guardian)))
      (if (or (not dead) (null? dead))
          n
          (begin (free-executable! dead) (loop (+ 1 n)))))))

;;; --- compiling ---------------------------------------------------------------

(define (compile-s-expr text)
  "Parse and compile the S-Expr v0.1 graph TEXT once, returning an <executable>
for repeated `executable-run`. Raises on a parse or compile failure."
  (let* ((doc-slot (make-slot))
         (exe-slot (make-slot))
         (err-slot (make-slot))
         (txt-ptr (string->pointer text))
         (len (string-length text)))
    (define (free-doc!)
      (unless (null-pointer? (slot-ref doc-slot))
        ((proc 'sb_document_free) (slot-ref doc-slot))
        (bytevector-fill! doc-slot 0)))
    (define (fail status)
      (let ((msg (capture-error err-slot)))
        (free-doc!)
        (unless (null-pointer? (slot-ref exe-slot))
          ((proc 'sb_executable_free) (slot-ref exe-slot))
          (bytevector-fill! exe-slot 0))
        (throw 'sb-ffi (if (string-null? msg)
                           (string-append "C ABI error: " (sb-status->message status))
                           msg))))
    (catch 'sb-ffi
      (lambda ()
        (let ((st ((proc 'sb_parse) txt-ptr len (slot-ptr doc-slot) (slot-ptr err-slot))))
          (unless (sb-status-ok? st) (fail st)))
        (let ((st ((proc 'sb_compile) (slot-ref doc-slot) (string->pointer "")
                   (slot-ptr exe-slot) (slot-ptr err-slot))))
          ;; sb_compile copies what it needs out of the document; the document
          ;; handle is not retained by the executable.
          (free-doc!)
          (unless (sb-status-ok? st) (fail st)))
        ;; (uint64 is 8 bytes on every 64-bit target; Guile's `sizeof` accepts
        ;; only pointer types, so the scalar sizes are hardcoded.)
        (let ((in-bytes (make-bytevector 8 0))
              (out-bytes (make-bytevector 8 0)))
          (let ((st ((proc 'sb_input_info) (slot-ref exe-slot) %null %null %null
                     (bytevector->pointer in-bytes))))
            (unless (sb-status-ok? st) (fail st)))
          (let ((st ((proc 'sb_output_info) (slot-ref exe-slot) %null %null %null
                     (bytevector->pointer out-bytes))))
            (unless (sb-status-ok? st) (fail st)))
          (let ((exe (%make-executable text (slot-ref exe-slot)
                                       (bytevector-u64-native-ref in-bytes 0)
                                       (bytevector-u64-native-ref out-bytes 0)
                                       #f)))
            ;; Register with the guardian; see reap-executables!.
            (%exe-guardian exe)
            exe)))
      (lambda (key msg) (error "compile-s-expr:" msg)))))

;;; --- running -----------------------------------------------------------------

(define (executable-run exe input-bv)
  "Bind INPUT-BV to EXE, run it and return two values: the output bytevector
and #f on success, or #f and an error-message string on failure. The native
handle is retained, so EXE may be run again; the C ABI copies the input, so
INPUT-BV may be reused or freed immediately."
  (if (executable-freed? exe)
      (values #f "executable-run: executable has already been freed")
      (let ((err-slot (make-slot)))
        (define (fail status)
          (let ((msg (capture-error err-slot)))
            (throw 'sb-ffi (if (string-null? msg)
                               (string-append "C ABI error: " (sb-status->message status))
                               msg))))
        (catch 'sb-ffi
          (lambda ()
            (unless (= (bytevector-length input-bv) (executable-input-nbytes exe))
              (throw 'sb-ffi "executable-run: input size does not match the graph"))
            (let ((st ((proc 'sb_bind_input) (executable-ptr exe)
                       (bytevector->pointer input-bv) (bytevector-length input-bv)
                       (slot-ptr err-slot))))
              (unless (sb-status-ok? st) (fail st)))
            (let ((st ((proc 'sb_execute) (executable-ptr exe) (slot-ptr err-slot))))
              (unless (sb-status-ok? st) (fail st)))
            (let* ((cap (executable-output-nbytes exe))
                   (out-bv (make-bytevector cap 0))
                   (written (make-bytevector 8 0))
                   (st ((proc 'sb_retrieve_output) (executable-ptr exe)
                        (bytevector->pointer out-bv) cap
                        (bytevector->pointer written) (slot-ptr err-slot))))
              (unless (sb-status-ok? st) (fail st))
              (values out-bv #f)))
          (lambda (key msg) (values #f msg))))))

;;; --- the text-keyed cache ----------------------------------------------------

(define executable-cache-enabled
  (make-parameter #t
    (lambda (on?)
      (unless (boolean? on?)
        (error "executable-cache-enabled: expected a boolean" on?))
      on?)))

(define %cache (make-hash-table))       ; graph text -> <executable>
(define %cache-lock (make-mutex))

(define (cached-executable text)
  "Return a compiled executable for TEXT, compiling it on first use. The
executable is owned by the cache; do not free it. A cache hit returns the same
object on every call for the same text."
  (let ((hit (hash-ref %cache text #f)))
    (if (and hit (not (executable-freed? hit)))
        hit
        (let ((exe (compile-s-expr text)))
          (lock-mutex %cache-lock)
          (let ((race (hash-ref %cache text #f)))
            (if (and race (not (executable-freed? race)))
                ;; Another thread won the race; drop ours and use the winner.
                (begin (unlock-mutex %cache-lock) (free-executable! exe) race)
                (begin (hash-set! %cache text exe)
                       (unlock-mutex %cache-lock)
                       exe)))))))

(define (run-s-expr/cached text input-bv)
  "Like `run-s-expr`, but when `executable-cache-enabled` is true the compiled
executable is cached by graph TEXT and reused across calls."
  (if (executable-cache-enabled)
      (executable-run (cached-executable text) input-bv)
      (run-s-expr text input-bv)))

(define (clear-executable-cache!)
  "Release every cached executable, drop the cache, and reap any other
collected executables."
  (lock-mutex %cache-lock)
  (hash-for-each (lambda (k v) (free-executable! v)) %cache)
  (set! %cache (make-hash-table))
  (unlock-mutex %cache-lock)
  (reap-executables!)
  *unspecified*)
