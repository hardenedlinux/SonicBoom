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

(define-module (sonicboom trace device)
  #:use-module (srfi srfi-1)
  #:use-module (ice-9 match)
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

;;; --- compile (cached per text) ----------------------------------------------

;; Alist text -> (exe inputs outputs), compared with `equal?` so the same
;; emitted text shares one compiled graph. A traced function has a handful of
;; segments, so a list is the right size.
(define %segments '())

(define (compiled-device-segments)
  "How many distinct segment texts have been compiled for the GPU."
  (length %segments))

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
    (let* ((exe (slot-ref exe-slot))
           (n-in (query-count exe 'in))
           (n-out (query-count exe 'out))
           (inputs (map (lambda (i) (query-info exe i 'in)) (iota n-in)))
           (outputs (map (lambda (i) (query-info exe i 'out)) (iota n-out))))
      (unless (every (lambda (d) (car d)) inputs)
        (error "sonicboom-device: the device graph does not name its inputs"))
      (unless (every (lambda (d) (car d)) outputs)
        (error "sonicboom-device: the device graph does not name its outputs"))
      (list exe inputs outputs))))

(define (segment-for text)
  (or (assoc-ref %segments text)
      (let ((s (compile-device text)))
        (set! %segments (acons text s %segments))
        s)))

;;; --- ctensor <-> little-endian bytes ----------------------------------------

(define (encode-ctensor t dtype shape)
  "T's row-major elements as native-endian bytes of DTYPE."
  (unless (eq? (ctensor-dtype t) dtype)
    (error "sonicboom-device: input dtype mismatch" dtype (ctensor-dtype t)))
  (unless (equal? (ctensor-shape t) shape)
    (error "sonicboom-device: input shape mismatch" shape (ctensor-shape t)))
  (let* ((n (ctensor-numel t))
         (sz (dtype-size dtype))
         (bv (make-bytevector (* n sz) 0)))
    (do ((i 0 (+ i 1))) ((= i n))
      (let ((x (ctensor-ref t i)))
        (case dtype
          ((float32) (bytevector-ieee-single-native-set! bv (* i 4) (exact->inexact x)))
          ((float64) (bytevector-ieee-double-native-set! bv (* i 8) (exact->inexact x)))
          ((int64) (bytevector-s64-native-set! bv (* i 8) (inexact->exact x)))
          ((int32) (bytevector-s32-native-set! bv (* i 4) (inexact->exact x)))
          ((int16) (bytevector-s16-native-set! bv (* i 2) (inexact->exact x)))
          ((int8) (bytevector-s8-set! bv i (inexact->exact x)))
          ((uint8) (bytevector-u8-set! bv i (inexact->exact x)))
          (else (error "sonicboom-device: unsupported input dtype" dtype)))))
    bv))

(define (decode-ctensor dtype shape bv)
  (let* ((n (shape-numel shape))
         (v (make-vector n)))
    (do ((i 0 (+ i 1))) ((= i n))
      (vector-set! v i
                   (case dtype
                     ((float32) (bytevector-ieee-single-native-ref bv (* i 4)))
                     ((float64) (bytevector-ieee-double-native-ref bv (* i 8)))
                     ((int64) (bytevector-s64-native-ref bv (* i 8)))
                     ((int32) (bytevector-s32-native-ref bv (* i 4)))
                     ((int16) (bytevector-s16-native-ref bv (* i 2)))
                     ((int8) (bytevector-s8-ref bv i))
                     ((uint8) (bytevector-u8-ref bv i))
                     ((bool) (not (zero? (bytevector-u8-ref bv i))))
                     (else (error "sonicboom-device: unsupported output dtype"
                                  dtype)))))
    (make-ctensor dtype shape v)))

;; A flat array of pointers as the C ABI takes `const void* const*`. The
;; bytevectors themselves are kept alive by the caller for the duration of the
;; call; Guile's bytevectors do not move.
(define (pointer-array bytevectors)
  (let* ((n (length bytevectors))
         (bv (make-bytevector (* 8 n) 0)))
    (for-each (lambda (b i)
                (u64-set! bv (* 8 i) (pointer-address (bytevector->pointer b))))
              bytevectors (iota n))
    bv))

(define (u64-array xs)
  (let ((bv (make-bytevector (* 8 (length xs)) 0)))
    (for-each (lambda (x i) (u64-set! bv (* 8 i) x)) xs (iota (length xs)))
    bv))

;;; --- execute ----------------------------------------------------------------

(define (run-device-executable segment ins-alist)
  "Run one compiled segment with INPUTS (alist name → ctensor); returns the
output alist in the executable's output order."
  (match segment
    ((exe inputs outputs)
     (let* ((in-bvs
             (map (match-lambda
                    ((name dtype shape bytes)
                     (let ((t (assoc-ref ins-alist name)))
                       (unless t
                         (error "sonicboom-device: no value for input" name))
                       (encode-ctensor t dtype shape))))
                  inputs))
            (out-bvs (map (match-lambda ((_ _ _ bytes) (make-bytevector bytes 0)))
                          outputs))
            (in-ptr (pointer-array in-bvs))
            (in-lens (u64-array (map (match-lambda ((_ _ _ b) b)) inputs)))
            (out-ptr (pointer-array out-bvs))
            (out-caps (u64-array (map (match-lambda ((_ _ _ b) b)) outputs)))
            (out-lens (u64-array (map (lambda (_) 0) outputs)))
            (err-slot (make-slot))
            (st ((proc 'sb_run_multi) exe
                 (bytevector->pointer in-ptr) (bytevector->pointer in-lens)
                 (length inputs)
                 (bytevector->pointer out-ptr) (bytevector->pointer out-caps)
                 (length outputs)
                 (bytevector->pointer out-lens) (slot-ptr err-slot))))
       (unless (sb-status-ok? st)
         (fail "sonicboom-device: run failed" err-slot))
       (map (lambda (desc bv)
              (match desc
                ((name dtype shape _)
                 (cons name (decode-ctensor dtype shape bv)))))
            outputs out-bvs)))
    (_ (error "sonicboom-device: bad segment record"))))

(define (device-executor)
  "An executor (text inputs) → outputs for trace-function #:executor."
  (lambda (text ins-alist)
    (run-device-executable (segment-for text) ins-alist)))
