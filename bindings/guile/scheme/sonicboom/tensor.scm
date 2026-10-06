;;; SonicBoom — Guile binding: tensor & parameter types
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; The Guile-side foundation of the SonicBoom AI framework: a <tensor> is a
;;; frozen dtype + a static shape + a raw little-endian bytevector; a
;;; <parameter> is a tensor with `requires-grad` set (and an optional `.grad`
;;; slot) — mirroring the PyTorch mental model. Tensors are pure Guile data
;;; here: no native C++ object backs them, and no arithmetic is performed in
;;; Scheme loops. When a model runs, its tensors are marshalled across the
;;; SonicBoom C ABI into the native backends (see the execution steps).
;;;
;;; Representation matches the frozen S-Expr v0.1 type system (design/
;;; s-expr-v0-spec.md): ten dtype names, static non-negative shape dims.

(define-module (sonicboom tensor)
  #:use-module (rnrs bytevectors)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (srfi srfi-27)
  #:use-module (ice-9 optargs)
  #:use-module (ice-9 match)
  #:export (<tensor> tensor?
            make-tensor %make-tensor bytevector->tensor tensor->bytevector
            tensor-dtype tensor-shape tensor-data tensor-numel
            tensor-ref tensor-set! tensor->list list->tensor tensor-copy
            zeros ones rand randn
            parameter parameter? tensor-requires-grad?
            tensor-grad set-tensor-grad! parameter-grad parameter-grad-set!
            dtype-size valid-dtype? valid-shape?))

;;; --- dtype vocabulary -------------------------------------------------------

(define *dtype-sizes*
  '((float32 . 4)  (float16 . 2)  (bfloat16 . 2) (float64 . 8)
    (int8 . 1)     (uint8 . 1)    (int16 . 2)    (int32 . 4)
    (int64 . 8)    (bool . 1)))

(define (valid-dtype? d)
  (and (assq d *dtype-sizes*) #t))

(define (dtype-size d)
  (or (assq-ref *dtype-sizes* d)
      (error "dtype-size: unknown dtype" d)))

(define (valid-shape? s)
  (and (list? s)
       (every (lambda (dim) (and (integer? dim) (>= dim 0))) s)))

(define (numel s)
  (fold * 1 s))

;; Seed the default random source once at load time so `rand`/`randn` are
;; non-deterministic across runs.
(random-source-randomize! default-random-source)

;;; --- record -----------------------------------------------------------------

(define-record-type <tensor>
  (%make-tensor dtype shape data requires-grad grad)
  tensor?
  (dtype        tensor-dtype)
  (shape        tensor-shape)
  (data         tensor-data)
  (requires-grad tensor-requires-grad? set-tensor-requires-grad!)
  (grad         tensor-grad set-tensor-grad!))

;;; --- element access ---------------------------------------------------------

(define (tensor-byte-offset t i)
  (* i (dtype-size (tensor-dtype t))))

(define (tensor-ref t i)
  (let ((d (tensor-dtype t))
        (bv (tensor-data t)))
    (case d
      ((float32) (bytevector-ieee-single-native-ref bv (* i 4)))
      ((float64) (bytevector-ieee-double-native-ref bv (* i 8)))
      ((int8)    (bytevector-s8-ref bv i))
      ((uint8)   (bytevector-u8-ref bv i))
      ((int16)   (bytevector-s16-native-ref bv (* i 2)))
      ((int32)   (bytevector-s32-native-ref bv (* i 4)))
      ((int64)   (bytevector-s64-native-ref bv (* i 8)))
      ((bool)    (not (zero? (bytevector-u8-ref bv i))))
      ((float16 bfloat16)
       (error "tensor-ref: element access for" d "is not supported in v0"))
      (else (error "tensor-ref: unknown dtype" d)))))

(define (tensor-set! t i v)
  (let ((d (tensor-dtype t))
        (bv (tensor-data t)))
    (case d
      ((float32) (bytevector-ieee-single-native-set! bv (* i 4) v))
      ((float64) (bytevector-ieee-double-native-set! bv (* i 8) v))
      ((int8)    (bytevector-s8-set! bv i v))
      ((uint8)   (bytevector-u8-set! bv i v))
      ((int16)   (bytevector-s16-native-set! bv (* i 2) v))
      ((int32)   (bytevector-s32-native-set! bv (* i 4) v))
      ((int64)   (bytevector-s64-native-set! bv (* i 8) v))
      ((bool)    (bytevector-u8-set! bv i (if v 1 0)))
      ((float16 bfloat16)
       (error "tensor-set!: element access for" d "is not supported in v0"))
      (else (error "tensor-set!: unknown dtype" d)))))

;;; --- constructors -----------------------------------------------------------

(define* (make-tensor data shape #:key (dtype 'float32) (requires-grad #f))
  "Build a tensor from a flat list of numbers. `data` must have (apply * shape)
elements; they are packed into a little-endian bytevector of the given dtype."
  (unless (valid-dtype? dtype)
    (error "make-tensor: invalid dtype" dtype))
  (unless (valid-shape? shape)
    (error "make-tensor: invalid shape" shape))
  (unless (= (length data) (numel shape))
    (error "make-tensor: data length" (length data)
           "does not match shape" shape))
  (let ((bv (make-bytevector (* (numel shape) (dtype-size dtype)) 0)))
    (let loop ((i 0) (xs data))
      (unless (null? xs)
        (tensor-set!-raw bv dtype i (car xs))
        (loop (+ i 1) (cdr xs))))
    (%make-tensor dtype shape bv requires-grad #f)))

;;; Raw element write into an owned bytevector (no <tensor> allocation), used by
;;; the constructor before the record exists.
(define (tensor-set!-raw bv d i v)
  (case d
    ((float32) (bytevector-ieee-single-native-set! bv (* i 4) v))
    ((float64) (bytevector-ieee-double-native-set! bv (* i 8) v))
    ((int8)    (bytevector-s8-set! bv i v))
    ((uint8)   (bytevector-u8-set! bv i v))
    ((int16)   (bytevector-s16-native-set! bv (* i 2) v))
    ((int32)   (bytevector-s32-native-set! bv (* i 4) v))
    ((int64)   (bytevector-s64-native-set! bv (* i 8) v))
    ((bool)    (bytevector-u8-set! bv i (if v 1 0)))
    (else (error "tensor-set!-raw: unknown dtype" d))))

(define* (bytevector->tensor bv shape #:key (dtype 'float32) (requires-grad #f))
  "Wrap an existing little-endian bytevector (no copy). The bytevector length
must equal (apply * shape) * (dtype-size dtype)."
  (unless (valid-dtype? dtype)
    (error "bytevector->tensor: invalid dtype" dtype))
  (unless (valid-shape? shape)
    (error "bytevector->tensor: invalid shape" shape))
  (unless (= (bytevector-length bv) (* (numel shape) (dtype-size dtype)))
    (error "bytevector->tensor: bytevector length mismatch"))
  (%make-tensor dtype shape bv requires-grad #f))

(define (tensor->bytevector t)
  (tensor-data t))

;;; --- initializers -----------------------------------------------------------

(define* (zeros shape #:key (dtype 'float32))
  (unless (valid-shape? shape)
    (error "zeros: invalid shape" shape))
  (%make-tensor dtype shape
                (make-bytevector (* (numel shape) (dtype-size dtype)) 0)
                #f #f))

(define* (ones shape #:key (dtype 'float32))
  (let ((t (zeros shape #:dtype dtype)))
    (let ((n (numel shape)))
      (let loop ((i 0))
        (when (< i n)
          (tensor-set! t i 1)
          (loop (+ i 1)))))
    t))

(define (rand)
  "A uniform real in [0, 1) from the default random source."
  (random-real))

(define* (randn shape #:key (dtype 'float32))
  "A tensor of independent standard-normal samples (Box-Muller)."
  (unless (valid-shape? shape)
    (error "randn: invalid shape" shape))
  (let* ((n (numel shape))
         (t (zeros shape #:dtype dtype)))
    (let loop ((i 0))
      (when (< i n)
        (let* ((u1 (max 1e-12 (random-real)))
               (u2 (random-real))
               (r (sqrt (* -2.0 (log u1))))
               (theta (* 2.0 (acos -1.0) u2)))
          (tensor-set! t i (* r (cos theta))))
        (loop (+ i 1))))
    t))

;;; --- introspection & conversion --------------------------------------------

(define (tensor-numel t)
  (numel (tensor-shape t)))

(define (tensor->list t)
  (let ((n (tensor-numel t)))
    (let loop ((i 0) (acc '()))
      (if (= i n)
          (reverse acc)
          (loop (+ i 1) (cons (tensor-ref t i) acc))))))

(define* (list->tensor data shape #:key (dtype 'float32))
  (make-tensor data shape #:dtype dtype))

(define (tensor-copy t)
  (%make-tensor (tensor-dtype t) (tensor-shape t)
                (bytevector-copy (tensor-data t))
                (tensor-requires-grad? t) (tensor-grad t)))

;;; --- parameters -------------------------------------------------------------

(define* (parameter data #:key (shape #f) (dtype 'float32))
  "A trainable tensor. `data` is either an existing tensor or a flat list of
numbers (with `shape`, defaulting to a rank-1 shape)."
  (let ((t (if (tensor? data)
               data
               (make-tensor data (or shape (list (length data))) #:dtype dtype))))
    (set-tensor-requires-grad! t #t)
    t))

(define (parameter? x)
  (and (tensor? x) (tensor-requires-grad? x)))

(define (parameter-grad p)
  (tensor-grad p))

(define (parameter-grad-set! p g)
  (unless (parameter? p)
    (error "parameter-grad-set!: not a parameter" p))
  (set-tensor-grad! p g))
