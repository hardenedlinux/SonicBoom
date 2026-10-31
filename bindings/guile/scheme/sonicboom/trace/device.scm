;;; SonicBoom — tracing prototype: GPU executor for training graphs
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; An executor for (sonicboom trace tracer) that runs each cached segment on
;;; the GPU through the C ABI:
;;;
;;;   S-Expr text → sb_parse → sb_compile_device
;;;     → Blade → Boom IR → Training passes → per-node CUDA kernels
;;;     → sb_run_multi
;;;
;;; The executable is compiled once per distinct text and cached, so every
;;; later call of the traced function reuses the same device buffers and the
;;; same compiled plan. Inputs and outputs are bound by name: the tracer names
;;; every tensor argument, and the device graph keeps those names. The
;;; interpreter in (sonicboom trace interp) is the reference this executes
;;; against.
;;;
;;; Ownership: a cached segment owns its native sb_executable. A caller that
;;; looks one up receives a borrowed Scheme reference to the segment. Evicting
;;; a segment (a different head state needs a fresh graph) only removes it from
;;; the cache; the native executable stays alive for as long as any caller still
;;; holds the segment, and is released by a guardian once the last reference is
;;; gone. Runs of one segment are serialized by a per-segment mutex, because the
;;; device graph owns one set of device buffers; different segments still run in
;;; parallel.

(define-module (sonicboom trace device)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (ice-9 match)
  #:use-module (ice-9 threads)
  #:use-module (rnrs bytevectors)
  #:use-module (system foreign)
  #:use-module (sonicboom ffi)
  #:use-module (sonicboom trace tensor)
  #:export (device-available? device-executor run-device-executable
            compiled-device-segments))

;;; --- dtype mapping ----------------------------------------------------------

(define %dtype-by-code
  '((0 . float32) (1 . float16) (2 . bfloat16) (3 . float64)
    (4 . int8) (5 . uint8) (6 . int16) (7 . int32) (8 . int64) (9 . bool)))

(define (dtype-name code)
  (or (assq-ref %dtype-by-code code)
      (error "sonicboom-device: unknown sb_dtype code" code)))

(define (dtype-size dtype)
  (case dtype
    ((float64 int64) 8)
    ((float32 int32) 4)
    ((float16 bfloat16 int16) 2)
    ((int8 uint8 bool) 1)
    (else (error "sonicboom-device: unsupported dtype" dtype))))

;;; --- native calls -----------------------------------------------------------

(define (u64-slot) (make-bytevector 8 0))
(define (u64-ref bv) (bytevector-u64-native-ref bv 0))
(define (u64-set! bv i x) (bytevector-u64-native-set! bv i x))

(define (fail who err-slot)
  (let ((msg (capture-error err-slot)))
    (error who (if (string-null? msg) "unknown C ABI error" msg))))

(define (device-available?)
  "True when the loaded libsonicboom.so has CUDA and can run a device graph."
  (and (libsonicboom-available?)
       (let ((cuda (assq-ref (build-info) 'cuda)))
         (and cuda (eq? (assq-ref cuda 'enabled) #t)))))

;;; --- introspection ----------------------------------------------------------

(define (query-info exe index direction)
  "Descriptor (name dtype shape num-bytes) of input/output INDEX."
  (let* ((info (if (eq? direction 'in) 'sb_input_info_at 'sb_output_info_at))
         (named (if (eq? direction 'in) 'sb_input_name_at 'sb_output_name_at))
         (dtype-bv (make-bytevector 4 0))
         (rank-bv (make-bytevector 8 0))
         (dims-slot (make-slot))
         (bytes-bv (u64-slot))
         (name-slot (make-slot))
         (st ((proc info) exe index (bytevector->pointer dtype-bv)
              (bytevector->pointer rank-bv) (slot-ptr dims-slot)
              (bytevector->pointer bytes-bv))))
    (unless (sb-status-ok? st)
      (error "sonicboom-device: introspection failed" st))
    (let* ((dtype (dtype-name (bytevector-s32-native-ref dtype-bv 0)))
           (rank (bytevector-s64-native-ref rank-bv 0))
           (dims (slot-ref dims-slot))
           (shape
            (if (= rank 0) '()
                (let ((bv (pointer->bytevector dims (* 8 rank))))
                  (do ((i 0 (+ i 1)) (acc '()))
                      ((= i rank) (reverse acc))
                    (set! acc (cons (bytevector-s64-native-ref bv (* 8 i)) acc))))))
           (st ((proc named) exe index (slot-ptr name-slot))))
      (unless (sb-status-ok? st)
        (error "sonicboom-device: name introspection failed" st))
      (let ((p (slot-ref name-slot)))
        (list (if (null-pointer? p) #f (pointer->string p))
              dtype shape (u64-ref bytes-bv))))))

(define (query-count exe direction)
  (let* ((bv (u64-slot))
         (st ((proc (if (eq? direction 'in) 'sb_input_count 'sb_output_count))
              exe (bytevector->pointer bv))))
    (unless (sb-status-ok? st)
      (error "sonicboom-device: count introspection failed" st))
    (u64-ref bv)))

(define (query-resident exe index)
  "True when the graph updates input INDEX in place on the device."
  (let ((bv (make-bytevector 4 0))
        (err-slot (make-slot)))
    (let ((st ((proc 'sb_input_is_resident) exe index
               (bytevector->pointer bv) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (fail "sonicboom-device: residency introspection failed" err-slot))
      (not (zero? (bytevector-s32-native-ref bv 0))))))

;;; --- compile (cached per text) ----------------------------------------------

;; A compiled device graph. The record owns the native handle EXE. INPUTS,
;; OUTPUTS and RESIDENT? describe the binding contract; STATE is (uploaded? .
;; snapshot), a mutable pair: UPLOADED? records that the device holds this
;; segment's resident inputs and SNAPSHOT holds, per resident input, the bytes
;; of the value the device held after the last run (the input's own bytes for an
;; input the graph does not update). A call whose resident bytes differ from the
;; snapshot is running on a different head state; see run-device-executable.
;;
;; LOCK serializes every use of the native handle (a device graph owns one set
;; of device buffers) and the STATE update. FREED? is claimed by whoever
;; releases the handle first, so a duplicate compile dropped during insertion
;; and the guardian's later reap cannot double-free.
(define-record-type <device-segment>
  (%make-device-segment exe inputs outputs resident state text lock freed)
  device-segment?
  (exe device-segment-exe)
  (inputs device-segment-inputs)
  (outputs device-segment-outputs)
  (resident device-segment-resident)
  (state device-segment-state)
  (text device-segment-text)
  (lock device-segment-lock)
  (freed device-segment-freed? set-device-segment-freed!))

;; Alist text -> segment, compared with `equal?` so the same emitted text
;; shares one compiled graph. A traced function has a handful of segments, so
;; a list is the right size. The alist is only ever read or replaced under
;; %segments-lock, and no native work happens while that lock is held.
(define %segments '())
(define %segments-lock (make-mutex))

;; Evicted segments are released only when nothing can reach them any more.
;; The guardian holds a weak reference; reap-device-segments! drains the ones
;; the GC has collected and frees their handles. This is the same ownership
;; rule as (sonicboom executable): a cache eviction never invalidates a handle
;; a caller still holds.
(define %segment-guardian (make-guardian))
(define %segment-reap-lock (make-mutex))

(define (release-device-segment! seg)
  "Release SEG's native handle once. Idempotent: the loser of a double release
observes the claimed flag and does nothing."
  (unless (device-segment-freed? seg)
    (set-device-segment-freed! seg #t)
    (let ((p (device-segment-exe seg)))
      (unless (null-pointer? p)
        ((proc 'sb_executable_free) p)))))

(define (reap-device-segments!)
  "Release every evicted segment the GC has collected. Returns the count."
  (let ((n 0))
    (with-mutex %segment-reap-lock
      (let loop ()
        (let ((dead (%segment-guardian)))
          (cond ((or (not dead) (null? dead)) n)
                (else
                 (release-device-segment! dead)
                 (set! n (+ n 1))
                 (loop))))))))

(define (compiled-device-segments)
  "How many distinct segment texts have been compiled for the GPU."
  (lock-mutex %segments-lock)
  (let ((n (length %segments)))
    (unlock-mutex %segments-lock)
    n))

(define (compile-device text)
  (let ((doc-slot (make-slot))
        (exe-slot (make-slot))
        (err-slot (make-slot)))
    (let ((st ((proc 'sb_parse) (string->pointer text) (string-length text)
               (slot-ptr doc-slot) (slot-ptr err-slot))))
      (unless (sb-status-ok? st)
        (fail "sonicboom-device: parse failed" err-slot)))
    (let ((st ((proc 'sb_compile_device) (slot-ref doc-slot) (slot-ptr exe-slot)
               (slot-ptr err-slot))))
      ((proc 'sb_document_free) (slot-ref doc-slot))
      (unless (sb-status-ok? st)
        (fail "sonicboom-device: compile failed" err-slot)))
    ;; From here the handle is owned by this procedure: if introspection fails,
    ;; release it before propagating the error rather than leaking it.
    (let ((exe (slot-ref exe-slot)))
      (catch #t
        (lambda ()
          (let* ((n-in (query-count exe 'in))
                 (n-out (query-count exe 'out))
                 (inputs (map (lambda (i) (query-info exe i 'in)) (iota n-in)))
                 (outputs (map (lambda (i) (query-info exe i 'out)) (iota n-out)))
                 (resident (map (lambda (i) (query-resident exe i))
                                (iota n-in))))
            (unless (every (lambda (d) (car d)) inputs)
              (error "sonicboom-device: the device graph does not name its inputs"))
            (unless (every (lambda (d) (car d)) outputs)
              (error "sonicboom-device: the device graph does not name its outputs"))
            (let ((seg (%make-device-segment
                        exe inputs outputs resident (list #f '()) text
                        (make-mutex) #f)))
              (%segment-guardian seg)
              seg)))
        (lambda (key . args)
          ;; No segment record owns the handle yet, so release it directly.
          ((proc 'sb_executable_free) exe)
          (apply throw key args))))))

(define (cache-lookup text)
  (lock-mutex %segments-lock)
  (let ((hit (assoc-ref %segments text)))
    (unlock-mutex %segments-lock)
    hit))

(define (cache-remove! text)
  "Remove TEXT's cache entry and return it. The segment remains owned by any
caller that still holds it; nothing is released here."
  (lock-mutex %segments-lock)
  (let ((old (assoc-ref %segments text)))
    (set! %segments (filter (lambda (p) (not (equal? (car p) text))) %segments))
    (unlock-mutex %segments-lock)
    old))

(define (segment-for text)
  (let ((hit (cache-lookup text)))
    (if (and hit (not (device-segment-freed? hit)))
        hit
        (let ((fresh (compile-device text)))
          (lock-mutex %segments-lock)
          (let ((race (assoc-ref %segments text)))
            (if (and race (not (device-segment-freed? race)))
                ;; Another thread won the race; drop ours and use the winner.
                (begin (unlock-mutex %segments-lock)
                       (release-device-segment! fresh)
                       (reap-device-segments!)
                       race)
                (begin
                  ;; Insert FRESH and drop every entry for TEXT, including a
                  ;; released RACE the lookup refused, so the alist holds one
                  ;; live entry per text.
                  (set! %segments
                        (acons text fresh
                               (filter (lambda (p) (not (equal? (car p) text)))
                                       %segments)))
                  (unlock-mutex %segments-lock)
                  fresh)))))))

(define (reset-segment! text)
  "Replace the cached executable for TEXT with a freshly compiled one.

The C ABI keeps the resident inputs in their device buffers and ignores their
bytes once a run has uploaded them, so a cached executable cannot be pointed at
a different head state. A fresh executable starts with nothing uploaded, and
its first run uploads the whole leaf state (see run-device-executable).

The replaced segment is only evicted, never freed: a caller that looked it up
before this call keeps a valid handle. Its native executable is released by the
guardian once the last holder drops it."
  (cache-remove! text)
  (reap-device-segments!)
  (segment-for text))

;;; --- ctensor <-> little-endian bytes ----------------------------------------

(define (encode-ctensor t dtype shape)
  "T's row-major elements as native-endian bytes of DTYPE."
  (unless (eq? (ctensor-dtype t) dtype)
    (error "sonicboom-device: input dtype mismatch" dtype (ctensor-dtype t)))
  (unless (equal? (ctensor-shape t) shape)
    (error "sonicboom-device: input shape mismatch" shape (ctensor-shape t)))
  ;; The dtype and shape match, so the ctensor already stores exactly the bytes
  ;; the device boundary takes.
  (bytevector-copy (ctensor-data t)))

(define (decode-ctensor dtype shape bv)
  ;; Copy: the output buffer is also kept as resident state and reused by later
  ;; runs, so the returned ctensor must not alias it.
  (bytevector->ctensor dtype shape (bytevector-copy bv)))

;; A flat array of pointers as the C ABI takes `const void* const*`. The
;; bytevectors themselves are kept alive by the caller for the duration of the
;; call; Guile's bytevectors do not move. A #f entry becomes a null pointer
;; (a resident input whose state is already on the device).
(define (pointer-array bytevectors)
  (let* ((n (length bytevectors))
         (bv (make-bytevector (* 8 n) 0)))
    (for-each (lambda (b i)
                (when b
                  (u64-set! bv (* 8 i) (pointer-address (bytevector->pointer b)))))
              bytevectors (iota n))
    bv))

(define (u64-array xs)
  (let ((bv (make-bytevector (* 8 (length xs)) 0)))
    (for-each (lambda (x i) (u64-set! bv (* 8 i) x)) xs (iota (length xs)))
    bv))

;;; --- execute ----------------------------------------------------------------

(define (encode-inputs inputs ins-alist)
  "One entry per input: its encoded bytes."
  (map (lambda (desc)
         (let ((t (assoc-ref ins-alist (car desc))))
           (unless t
             (error "sonicboom-device: no value for input" (car desc)))
           (match desc
             ((name dtype shape _)
              (encode-ctensor t dtype shape)))))
       inputs))

(define (resident-synced? state resident encoded)
  "True when the device holds the resident inputs' current ENCODED bytes."
  (and (car state)
       (= (length encoded) (length (cdr state)))
       (every (lambda (res bv snap) (or (not res) (equal? bv snap)))
              resident encoded (cdr state))))

(define (call-with-segment-lock segment thunk)
  "Run THUNK while holding SEGMENT's run lock. The lock also protects STATE."
  (let ((lock (device-segment-lock segment)))
    (dynamic-wind
      (lambda () (lock-mutex lock))
      thunk
      (lambda () (unlock-mutex lock)))))

(define (run-executable segment encoded synced?)
  "Run SEGMENT with ENCODED (the resident inputs' bytes, #f when synced), and
record the resident state the device holds afterwards. The caller must hold
SEGMENT's run lock (see run-device-executable)."
  (let* ((exe (device-segment-exe segment))
         (inputs (device-segment-inputs segment))
         (outputs (device-segment-outputs segment))
         (resident (device-segment-resident segment))
         (state (device-segment-state segment))
         (in-bvs (map (lambda (bv res)
                        (if (and synced? res) #f bv))
                      encoded resident))
         (out-bvs (map (match-lambda ((_ _ _ bytes) (make-bytevector bytes 0)))
                       outputs))
         (out-idx (iota (length outputs)))
         (in-ptr (pointer-array in-bvs))
         (in-lens (u64-array (map (match-lambda ((_ _ _ b) b)) inputs)))
         (dl-idx (u64-array out-idx))
         (out-ptr (pointer-array out-bvs))
         (out-caps (u64-array (map (match-lambda ((_ _ _ b) b)) outputs)))
         (out-lens (u64-array (map (lambda (_) 0) outputs)))
         (err-slot (make-slot))
         (st ((proc 'sb_run_resident) exe
              (bytevector->pointer in-ptr) (bytevector->pointer in-lens)
              (length inputs)
              (bytevector->pointer dl-idx) (length outputs)
              (bytevector->pointer out-ptr) (bytevector->pointer out-caps)
              (bytevector->pointer out-lens) (slot-ptr err-slot))))
    (unless (sb-status-ok? st)
      (fail "sonicboom-device: run failed" err-slot))
    (let ((out-alist (map (lambda (desc bv)
                            (match desc
                              ((name dtype shape _) (cons name bv))))
                          outputs out-bvs)))
      ;; What the device holds now: an updated resident input comes back as an
      ;; output under its own name; anything else is still what was uploaded.
      (set-car! (cdr state)
                (map (lambda (desc res bv)
                       (and res (or (assoc-ref out-alist (car desc)) bv)))
                     inputs resident encoded))
      (set-car! state #t)
      (map (lambda (desc bv)
             (match desc
               ((name dtype shape _)
                (cons name (decode-ctensor dtype shape bv)))))
           outputs out-bvs))))

(define (run-device-executable segment ins-alist)
  "Run one compiled segment with INPUTS (alist name → ctensor); returns the
output alist in the executable's output order.

The first call uploads the whole leaf state. After it, an input the graph
updates in place (a parameter or optimizer moment) stays on the device: it is
neither encoded nor uploaded, and the next value comes back as one of the
outputs. Every other input — the batch — is encoded and uploaded on every
call.

A cached segment is shared by every call that emits the same text, including
calls on a different head state (fresh parameter tensors of the same shape).
The C ABI keeps a resident input in its device buffer and ignores the caller's
bytes once it has been uploaded, so such a call is detected here by comparing
the resident bytes with the state the device was left in, and the executable is
rebuilt so the first run uploads the new head state.

SEGMENT's run lock is held for the whole decision, native call and state update,
so two callers never drive one set of device buffers at once. The stale path
drops the lock before replacing the cache entry: the replacement is a different
segment with its own lock, and evicting SEGMENT does not release it, so a
concurrent holder keeps using the executable it already has."
  (unless (device-segment? segment)
    (error "sonicboom-device: bad segment record"))
  (let ((outcome
         (call-with-segment-lock
          segment
          (lambda ()
            (when (device-segment-freed? segment)
              (error "sonicboom-device: segment has already been released"))
            (let* ((inputs (device-segment-inputs segment))
                   (resident (device-segment-resident segment))
                   (state (device-segment-state segment))
                   (encoded (encode-inputs inputs ins-alist))
                   (synced? (resident-synced? state resident encoded)))
              (if (and (car state) (not synced?))
                  'stale
                  (run-executable segment encoded synced?)))))))
    (if (eq? outcome 'stale)
        (run-device-executable (reset-segment! (device-segment-text segment))
                               ins-alist)
        outcome)))

(define (device-executor)
  "An executor (text inputs) → outputs for trace-function #:executor."
  (lambda (text ins-alist)
    (run-device-executable (segment-for text) ins-alist)))
