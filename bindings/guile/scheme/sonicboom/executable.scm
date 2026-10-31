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
;;; `executable-cache-enabled` parameter). It is only for text whose compiled
;;; executable is fully determined by that text:
;;;
;;;   - (data :bound ...) parameters carry state supplied after the cache key
;;;     is computed, so an owner of that state (a <model>, which owns one
;;;     executable per input shape) must hold its own executable instead.
;;;   - (data :external ...) parameters resolve a sidecar at compile time
;;;     relative to the process working directory and copy its bytes in, so
;;;     the same text can yield different executables. The cache is keyed by
;;;     text, not by file contents.
;;;
;;; The cache enforces both: a freshly compiled executable with unfilled
;;; bindings, or a text naming an :external sidecar, is returned to the caller
;;; and never enters the cache, so two independent owners of the same graph
;;; text can never share bound state or another directory's sidecar bytes.
;;; The cache is never invalidated automatically; `clear-executable-cache!`
;;; drops every entry, but an entry a caller still holds stays alive until the
;;; caller drops it (the guardian then releases it).

(define-module (sonicboom executable)
  #:use-module (system foreign)
  #:use-module (rnrs bytevectors)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-13)
  #:use-module (srfi srfi-34)
  #:use-module (ice-9 hash-table)
  #:use-module (ice-9 threads)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom error)
  #:use-module (sonicboom storage)
  #:use-module (sonicboom tensor)
  #:export (compile-s-expr
            executable? executable-run free-executable!
            executable-freed? executable-graph-text
            executable-input-nbytes executable-output-nbytes
            executable-parameter-count executable-parameter-name
            executable-parameter-ref
            executable-bind-parameter! executable-bindings-complete?
            cached-executable run-s-expr/cached
            clear-executable-cache! executable-cache-enabled
            reap-executables!))

(define %null %null-pointer)

;;; --- the executable record ---------------------------------------------------

(define-record-type <executable>
  (%make-executable text ptr in-nbytes out-nbytes in-count out-count freed lock)
  executable?
  (text executable-graph-text)
  (ptr executable-ptr)
  (in-nbytes executable-input-nbytes)
  (out-nbytes executable-output-nbytes)
  ;; Graph arity, captured at compile time. executable-run is the 1-in/1-out
  ;; wrapper; any other arity keeps the historical bind/execute/retrieve path
  ;; so its diagnostics are unchanged.
  (in-count executable-input-count)
  (out-count executable-output-count)
  ;; Flipped to #t by whoever releases the handle first (explicit free or the
  ;; finalizer), so the other one becomes a no-op instead of a double free.
  (freed executable-freed? set-executable-freed!)
  ;; A native executable owns one set of input, output and scratch buffers, so
  ;; two concurrent runs would overwrite each other's data. executable-run
  ;; serializes runs of one executable; different executables stay parallel.
  (lock executable-lock))

;; A one-element size_t out-param (8 bytes on every 64-bit target; Guile's
;; `sizeof` accepts only pointer types, so the scalar size is hardcoded).
(define (u64-native-bv n)
  (let ((bv (make-bytevector 8 0)))
    (bytevector-u64-native-set! bv 0 n)
    bv))

(define (free-executable! exe)
  "Release EXE's native handle. Idempotent: freeing an already-freed (or
guardian-reaped) executable is a no-op, never an error.

The run lock is taken first, so an explicit free waits for an in-flight
`executable-run` instead of releasing the native handle underneath it. A run
that starts after the free sees the freed flag and fails instead."
  (call-with-executable-lock
   exe
   (lambda ()
     (unless (executable-freed? exe)
       (set-executable-freed! exe #t)   ; claim it before touching the pointer
       (let ((p (executable-ptr exe)))
         (unless (null-pointer? p)
           ((proc 'sb_executable_free) p))))))
  ;; If EXE was a text-cache entry, drop it: a later lookup must not return a
  ;; released handle. The cache's contract already forbids freeing its entries;
  ;; this makes a contract violation self-correcting instead of leaving a stale
  ;; entry behind. The executable lock is not held here (it was released
  ;; above), so this never nests with a caller's executable operation.
  (evict-cached-executable! exe)
  *unspecified*)

;;; --- GC fallback -------------------------------------------------------------

;; Every compiled executable is registered with this guardian. It holds only a
;; weak reference, so a record that becomes unreachable is handed back by the
;; next `reap-executables!` call, which then releases its native handle. This
;; is the safety net for callers that never free explicitly.
(define %exe-guardian (make-guardian))

(define (reap-executables!)
  "Release every executable whose record has been collected without an explicit
`free-executable!`, and every parameter view whose tensor was collected. Returns
the number of executable handles released."
  (let loop ((n 0))
    (let ((dead (%exe-guardian)))
      (if (or (not dead) (null? dead))
          (begin (reap-storage-views!) n)
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
      ;; Release the half-built handles before sb-check! raises.
      (free-doc!)
      (unless (null-pointer? (slot-ref exe-slot))
        ((proc 'sb_executable_free) (slot-ref exe-slot))
        (bytevector-fill! exe-slot 0))
      (sb-check! status err-slot "compile-s-expr"))
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
              (out-bytes (make-bytevector 8 0))
              (in-count (make-bytevector 8 0))
              (out-count (make-bytevector 8 0)))
          (let ((st ((proc 'sb_input_info) (slot-ref exe-slot) %null %null %null
                     (bytevector->pointer in-bytes))))
            (unless (sb-status-ok? st) (fail st)))
          (let ((st ((proc 'sb_output_info) (slot-ref exe-slot) %null %null %null
                     (bytevector->pointer out-bytes))))
            (unless (sb-status-ok? st) (fail st)))
          (let ((st ((proc 'sb_input_count) (slot-ref exe-slot)
                     (bytevector->pointer in-count))))
            (unless (sb-status-ok? st) (fail st)))
          (let ((st ((proc 'sb_output_count) (slot-ref exe-slot)
                     (bytevector->pointer out-count))))
            (unless (sb-status-ok? st) (fail st)))
          (let ((exe (%make-executable text (slot-ref exe-slot)
                                       (bytevector-u64-native-ref in-bytes 0)
                                       (bytevector-u64-native-ref out-bytes 0)
                                       (bytevector-u64-native-ref in-count 0)
                                       (bytevector-u64-native-ref out-count 0)
                                       #f (make-mutex))))
            ;; Register with the guardian; see reap-executables!.
            (%exe-guardian exe)
            exe))))

;;; --- running -----------------------------------------------------------------

(define (call-with-executable-lock exe thunk)
  (let ((lock (executable-lock exe)))
    (dynamic-wind
      (lambda () (lock-mutex lock))
      thunk
      (lambda () (unlock-mutex lock)))))

;; The one rule for every operation that dereferences the native handle: hold
;; EXE's lock while the handle is in use, and re-check the freed flag *after*
;; acquiring it. A check made before locking can be overtaken by
;; free-executable!, which would release the native handle underneath the
;; caller.
;;
;; The lock lives in the <executable> record and is never destroyed explicitly:
;; free-executable! only marks the record and releases the native handle. A
;; thread that is using or waiting on the lock keeps the record reachable, so
;; the GC cannot collect the mutex out from under it; the guardian resurrects a
;; reference before reaping for the same reason.
(define (call-with-live-executable exe who thunk)
  "Call THUNK with EXE's native handle safe to use: THUNK runs while EXE's lock
is held and after the freed flag has been re-checked under that lock. Raises a
Scheme error for a non-executable or an already-freed executable. THUNK must
not call another executable operation on the same EXE (the mutex is not
recursive)."
  (unless (executable? exe)
    (error who "not an executable" exe))
  (call-with-executable-lock
   exe
   (lambda ()
     (when (executable-freed? exe)
       (error who "executable has already been freed"))
     (thunk))))

;; executable-run applies the same rule inline: it takes the lock and re-checks
;; the freed flag before dispatch, but reports a refusal as its documented
;; (values #f message) failure instead of raising like the accessors.
(define (executable-run exe input-bv)
  "Bind INPUT-BV to EXE, run it and return two values: the output bytevector
and #f on success, or #f and an error-message string on failure. The native
handle is retained, so EXE may be run again; the C ABI copies the input, so
INPUT-BV may be reused or freed immediately.

Runs of one executable are serialized, because the compiled graph owns a single
set of input, output and scratch buffers. Different executables, including two
entries of one model's per-shape cache, run in parallel."
  (call-with-executable-lock
   exe
   (lambda ()
     (cond
       ((executable-freed? exe)
        (values #f "executable-run: executable has already been freed"))
       ((and (= (executable-input-count exe) 1)
             (= (executable-output-count exe) 1))
        (executable-run-1x1 exe input-bv))
       (else
        ;; sb_run_multi requires the exact arity; executable-run's contract is
        ;; one input and one output. Keep the historical wrapper for any other
        ;; graph so its failure mode (a binding error naming the arity) is
        ;; unchanged.
        (executable-run-legacy exe input-bv))))))

;; The one-in/one-out fast path. sb_run_multi takes the caller's input bytes
;; and writes the result directly into the bytevector it returns, so neither
;; the bind-time copy into the executable nor the retrieve-time copy out of it
;; happens. The executor still copies the input into its own slot and the C ABI
;; still copies the result out of the backend — those copies are below the
;; stable ABI, not here.
(define (executable-run-1x1 exe input-bv)
  (let ((err-slot (make-slot)))
    (define (fail status) (sb-check! status err-slot "executable-run"))
    (guard (c ((sonicboom-error? c) (values #f (sonicboom-error-message c))))
      (unless (= (bytevector-length input-bv) (executable-input-nbytes exe))
        (raise-sonicboom-error
         SB_ERR_BINDING 'binding
         "executable-run: input size does not match the graph"))
      (let* ((cap (executable-output-nbytes exe))
             (out-bv (make-bytevector cap 0))
             (in-lens (u64-native-bv (bytevector-length input-bv)))
             (out-caps (u64-native-bv cap))
             (in-array (make-c-struct '(*) (list (bytevector->pointer input-bv))))
             (out-array (make-c-struct '(*) (list (bytevector->pointer out-bv))))
             (st ((proc 'sb_run_multi) (executable-ptr exe)
                  in-array (bytevector->pointer in-lens) 1
                  out-array (bytevector->pointer out-caps) 1
                  %null (slot-ptr err-slot))))
        (unless (sb-status-ok? st) (fail st))
        (values out-bv #f)))))

;; The historical bind / execute / retrieve path (as before Stage 0). Only
;; reachable for a graph that is not one-in/one-out.
(define (executable-run-legacy exe input-bv)
  (let ((err-slot (make-slot)))
    (define (fail status) (sb-check! status err-slot "executable-run"))
    (guard (c ((sonicboom-error? c) (values #f (sonicboom-error-message c))))
        (unless (= (bytevector-length input-bv) (executable-input-nbytes exe))
          (raise-sonicboom-error
           SB_ERR_BINDING 'binding
           "executable-run: input size does not match the graph"))
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
          (values out-bv #f)))))

;;; --- parameters (read-only native views) --------------------------------------
;;;
;;; A compiled graph materializes every parameter (a graph constant) into its
;;; own buffers at compile time and never writes them, so a parameter can be
;;; exposed as a zero-copy, read-only <tensor>. The view is an sb_storage handle
;;; owned by the tensor's storage; it pins the native implementation, so the
;;; tensor stays valid even after free-executable! or a cache eviction. Views
;;; are released by tensor-release! or, when the tensor is collected, by the
;;; storage guardian that reap-executables! drains.

;; sb_tensor_desc layout: struct_size size_t, dtype enum (int), rank int64,
;; dims pointer, num_bytes uint64, flags uint32, quant enum (int),
;; quant_block uint32, name pointer. The type values (uint64, int, ...) are the
;; (system foreign) descriptors; '*' is the pointer type symbol.
(define %desc-types
  (list uint64 int int64 '* uint64 unsigned-int int unsigned-int '*))

(define (align-up n alignment)
  (* alignment (quotient (+ n alignment -1) alignment)))

(define %desc-offsets
  (let loop ((types %desc-types) (offset 0) (acc '()))
    (if (null? types)
        (reverse acc)
        (let* ((t (car types))
               (o (align-up offset (alignof t))))
          (loop (cdr types) (+ o (sizeof t)) (cons o acc))))))

(define %desc-size
  (let loop ((types %desc-types) (offset 0))
    (if (null? types)
        offset
        (let* ((t (car types))
               (o (align-up offset (alignof t))))
          (loop (cdr types) (+ o (sizeof t)))))))

(define %desc-off-struct-size (list-ref %desc-offsets 0))
(define %desc-off-dtype (list-ref %desc-offsets 1))
(define %desc-off-rank (list-ref %desc-offsets 2))
(define %desc-off-dims (list-ref %desc-offsets 3))
(define %desc-off-num-bytes (list-ref %desc-offsets 4))

;; sb_dtype value → Scheme dtype symbol (capi.h sb_dtype, same order).
(define %sb-dtypes
  (vector 'float32 'float16 'bfloat16 'float64 'int8 'uint8
          'int16 'int32 'int64 'bool))

(define (make-desc-bv)
  (let ((bv (make-bytevector %desc-size 0)))
    (bytevector-u64-native-set! bv %desc-off-struct-size %desc-size)
    bv))

(define (executable-parameter-count exe)
  "Number of parameters (graph constants) materialized by EXE. A device
training graph has no host parameter block and raises."
  (call-with-live-executable
   exe "executable-parameter-count"
   (lambda ()
     (let ((count (u64-native-bv 0))
           (err-slot (make-slot)))
       (let ((st ((proc 'sb_executable_parameter_count) (executable-ptr exe)
                  (bytevector->pointer count) (slot-ptr err-slot))))
         (unless (sb-status-ok? st)
           (sb-check! st err-slot "executable-parameter-count")))
       (bytevector-u64-native-ref count 0)))))

(define (executable-parameter-name exe index)
  "The name of parameter INDEX in EXE (a borrowed string, copied here)."
  (call-with-live-executable
   exe "executable-parameter-name"
   (lambda ()
     (let ((name-slot (make-slot))
           (err-slot (make-slot)))
       (let ((st ((proc 'sb_executable_parameter_name_at) (executable-ptr exe)
                  index (slot-ptr name-slot) (slot-ptr err-slot))))
         (unless (sb-status-ok? st)
           (sb-check! st err-slot "executable-parameter-name")))
       (let ((p (slot-ref name-slot)))
         (if (null-pointer? p) "" (pointer->string p)))))))

(define (executable-parameter-ref exe index)
  "A read-only <tensor> viewing parameter INDEX of EXE without copying. The
tensor's storage owns an independent native view, so it stays valid after
free-executable!; release it with tensor-release! when done (or let the guardian
release it). tensor-set! on the result raises; tensor-copy gives a writable,
independent copy. Read-only is enforced by tensor-set!: the bytevector returned
by tensor-data aliases the native bytes and must not be written directly."
  (call-with-live-executable
   exe "executable-parameter-ref"
   (lambda ()
     (let ((view-slot (make-slot))
           (err-slot (make-slot)))
       (let ((st ((proc 'sb_executable_parameter_view) (executable-ptr exe) index
                  (slot-ptr view-slot) (slot-ptr err-slot))))
         (unless (sb-status-ok? st)
           (sb-check! st err-slot "executable-parameter-ref")))
       (let ((view (slot-ref view-slot)))
         (define (release!) ((proc 'sb_storage_release) view))
         (define (fail! st)
           (let ((msg (capture-error err-slot)))
             (release!)
             (raise-sonicboom-error
              st 'internal
              (string-append "executable-parameter-ref: "
                             (if (string-null? msg) "invalid descriptor" msg)))))
         (let ((desc (make-desc-bv)))
           (let ((st ((proc 'sb_storage_desc) view (bytevector->pointer desc)
                      (slot-ptr err-slot))))
             (unless (sb-status-ok? st) (fail! st)))
           (let ((data-slot (make-slot)))
             (let ((st ((proc 'sb_storage_data) view (slot-ptr data-slot) %null
                        (slot-ptr err-slot))))
               (unless (sb-status-ok? st) (fail! st)))
             (let* ((dtype-code (bytevector-s32-native-ref desc %desc-off-dtype))
                    (rank (bytevector-s64-native-ref desc %desc-off-rank))
                    (nbytes (bytevector-u64-native-ref desc %desc-off-num-bytes))
                    (data (slot-ref data-slot))
                    (dtype (and (>= dtype-code 0)
                                (< dtype-code (vector-length %sb-dtypes))
                                (vector-ref %sb-dtypes dtype-code)))
                    (shape (if (<= rank 0)
                               '()
                               (let ((dims (pointer->bytevector
                                            (make-pointer
                                             (bytevector-u64-native-ref
                                              desc %desc-off-dims))
                                            (* rank 8))))
                                 (let loop ((i (- rank 1)) (acc '()))
                                   (if (< i 0)
                                       acc
                                       (loop (- i 1)
                                             (cons (bytevector-s64-native-ref
                                                    dims (* i 8))
                                                   acc)))))))
                    (elements (if (null? shape) 1 (apply * shape))))
               (when (or (not dtype)
                         (null-pointer? data)
                         (not (= nbytes (* elements (dtype-size dtype)))))
                 (fail! SB_ERR_INTERNAL))
               (%make-tensor dtype shape
                             (borrowed-host-storage
                              (pointer->bytevector data nbytes) exe view)
                             #f #f)))))))))

;;; --- deferred (:bound) parameters --------------------------------------------
;;;
;;; A graph may declare a parameter as (data :bound "name"), leaving its bytes
;;; to be supplied after compile. Compile allocates the slot; executable-run
;;; fails until every bound parameter has been filled (executable-bindings-
;;; complete? reports whether it is). Binding copies the bytes once, so later
;;; edits to DATA-BV are not reflected. A parameter may be rebound (replacing
;;; the bytes) any number of times before the first successful run; after that
;;; binding raises, because the executable is bound once and then run.

(define (executable-bind-parameter! exe name data-bv)
  "Copy DATA-BV into EXE's (:bound NAME) parameter. NAME is the binding name
from the graph. Raises on an unknown name, a byte-count mismatch, or when the
executable has already run successfully.

The bytes are copied: subsequent writes to DATA-BV are not observed. A
parameter may be bound again until the first successful run replaces the
previous bytes; afterwards binding it raises."
  (unless (string? name)
    (error "executable-bind-parameter!: parameter name must be a string" name))
  (unless (bytevector? data-bv)
    (error "executable-bind-parameter!: data must be a bytevector" data-bv))
  (call-with-live-executable
   exe "executable-bind-parameter!"
   (lambda ()
     (let ((err-slot (make-slot)))
       (let ((st ((proc 'sb_bind_parameter) (executable-ptr exe)
                  (string->pointer name)
                  (bytevector->pointer data-bv)
                  (bytevector-length data-bv)
                  (slot-ptr err-slot))))
         (unless (sb-status-ok? st)
           (sb-check! st err-slot "executable-bind-parameter!"))))
     *unspecified*)))

(define (executable-bindings-complete? exe)
  "True when EXE has no :bound parameters or every one has been filled.
executable-run fails (returning #f and a message) until this is true."
  (call-with-live-executable
   exe "executable-bindings-complete?"
   (lambda ()
     (let ((complete (make-bytevector 4 0))
           (err-slot (make-slot)))
       (let ((st ((proc 'sb_bindings_complete) (executable-ptr exe)
                  (bytevector->pointer complete) (slot-ptr err-slot))))
         (unless (sb-status-ok? st)
           (sb-check! st err-slot "executable-bindings-complete?")))
       (not (zero? (bytevector-s32-native-ref complete 0)))))))

;;; --- the text-keyed cache ----------------------------------------------------

(define executable-cache-enabled
  (make-parameter #t
    (lambda (on?)
      (unless (boolean? on?)
        (error "executable-cache-enabled: expected a boolean" on?))
      on?)))

(define %cache (make-hash-table))       ; graph text -> <executable>
(define %cache-lock (make-mutex))

(define (evict-cached-executable! exe)
  "Drop EXE's text-cache entry when the entry is EXE itself. Called by
free-executable! after the handle is released, so a lookup that races a free
either sees the entry before it is dropped (and gets a handle that run refuses
for being freed) or compiles a fresh one — never a released entry."
  (let ((text (executable-graph-text exe)))
    (lock-mutex %cache-lock)
    (when (eq? (hash-ref %cache text #f) exe)
      (hash-remove! %cache text))
    (unlock-mutex %cache-lock)))

(define (graph-has-external-data? text)
  "True when TEXT names the :external parameter-data tag.

This is deliberately lexical: the text cache key is graph text, and the S-Expr
parser accepts only the exact symbol `:external` for this data kind, so a text
containing that token may resolve a sidecar. A false positive (the token in a
string or comment) only declines to cache the text, which is safe."
  (and (string? text)
       (string-contains text ":external")
       #t))

(define (cached-executable/ownership text)
  "Return (values EXE CACHE-OWNED?) for TEXT, compiling it on first use.

A cache hit returns the same object on every call for the same text and
CACHE-OWNED? is true: the cache holds the handle, so the caller must not free
it. The cache is used only when the executable is fully determined by the
text:

  - A graph with (data :bound ...) parameters has an unfilled slot at compile;
    the bytes its owner later binds are state the text key does not represent.
  - A graph with (data :external ...) parameters copies a sidecar's bytes in at
    compile; the path is resolved against the working directory, so the same
    text can denote different bytes in different directories or after the file
    changes.

Either kind is returned uncached with CACHE-OWNED? false — two callers with the
same text never share one executable, hence never share bound state or another
directory's sidecar bytes. Its owner (the caller) releases it with
free-executable! (run-s-expr/cached does this itself). :values graphs are
cached."
  (let* ((hit (begin (lock-mutex %cache-lock)
                     (let ((h (hash-ref %cache text #f)))
                       (unlock-mutex %cache-lock)
                       h))))
    (if (and hit (not (executable-freed? hit)))
        (values hit #t)
        (let ((exe (compile-s-expr text)))
          (if (or (graph-has-external-data? text)
                  (not (executable-bindings-complete? exe)))
              ;; Two uncacheable kinds. :external graphs copy a sidecar resolved
              ;; against the working directory at compile; the text key does not
              ;; capture those bytes. A fresh compile reports incomplete
              ;; bindings exactly when the graph declares :bound parameters; the
              ;; bytes bound later are not in the key either. Exclude both rather
              ;; than invent a key for state the text does not represent.
              (values exe #f)
              (begin
                (lock-mutex %cache-lock)
                (let ((race (hash-ref %cache text #f)))
                  (if (and race (not (executable-freed? race)))
                      ;; Another thread won the race; drop ours and use the
                      ;; winner.
                      (begin (unlock-mutex %cache-lock)
                             (free-executable! exe)
                             (values race #t))
                      (begin (hash-set! %cache text exe)
                             (unlock-mutex %cache-lock)
                             (values exe #t))))))))))

(define (cached-executable text)
  "Return a compiled executable for TEXT, compiling it on first use. A
cacheable executable is owned by the cache; do not free it. A hit returns the
same object on every call for the same text.

TEXT must be self-contained: :values graphs qualify, but the cache does not own
two other kinds. A graph with (data :bound ...) parameters has bytes supplied
after compile; a graph with (data :external ...) parameters has sidecar bytes
resolved from the working directory at compile. Neither is represented by the
text key. Both kinds are still compiled and returned, but they are not cached:
the returned executable is owned by the caller and should be freed with
free-executable!. Give each such graph its own executable (as model-forward
does) instead of going through this cache."
  (call-with-values (lambda () (cached-executable/ownership text))
    (lambda (exe _cache-owned?) exe)))

(define (run-s-expr/cached text input-bv)
  "Like `run-s-expr`, but when `executable-cache-enabled` is true the compiled
executable is cached by graph TEXT and reused across calls.

A text the cache refuses to own (a :bound or :external graph, see
cached-executable) is compiled for this call only and released as soon as the
run returns, so repeated calls cannot accumulate handles, share bound state, or
reuse another directory's sidecar bytes."
  (if (executable-cache-enabled)
      (call-with-values (lambda () (cached-executable/ownership text))
        (lambda (exe cache-owned?)
          (if cache-owned?
              (executable-run exe input-bv)
              (call-with-values (lambda () (executable-run exe input-bv))
                (lambda (out err)
                  (free-executable! exe)
                  (values out err))))))
      (run-s-expr text input-bv)))

(define (clear-executable-cache!)
  "Drop every cached entry and reap any other collected executables.

Eviction does not free an executable a caller still holds: the entry stops
being shared, but the held handle stays valid and runnable until the caller
drops it, when the executable guardian releases its native handle. This keeps
eviction from invalidating an executable (or a parameter view it pins)
underneath an active user."
  (lock-mutex %cache-lock)
  (set! %cache (make-hash-table))
  (unlock-mutex %cache-lock)
  (reap-executables!)
  *unspecified*)
