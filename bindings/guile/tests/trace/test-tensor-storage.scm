;;; SonicBoom — tracing prototype tests: ctensor packed bytevector storage
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Run from the repository root:
;;;   guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/trace/test-tensor-storage.scm
;;;
;;; The concrete reference tensor stores elements packed in a bytevector using
;;; each dtype's native representation. These checks pin the byte layout, the
;;; numeric conversion of every supported dtype, bounds and range errors, empty
;;; and rank-0 tensors, view aliasing, arithmetic kernels, and gradients.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom trace tensor)
             (sonicboom trace symbolic)
             (sonicboom trace ops)
             (sonicboom trace autodiff))

(test-begin "sonicboom-tensor-storage")

(define (close? a b tol)
  (<= (abs (- a b)) (* tol (max 1.0 (abs a) (abs b)))))

;;; =============================================================================
;;; Byte layout: one element is dtype-size bytes
;;; =============================================================================

(test-equal "dtype sizes"
  '(4 2 2 8 1 1 2 4 8 1)
  (map dtype-size
       '(float32 float16 bfloat16 float64 int8 uint8 int16 int32 int64 bool)))

(test-equal "nbytes = numel * dtype-size"
  '((float32 24) (float16 12) (bfloat16 12) (float64 48) (int8 6) (uint8 6)
    (int16 12) (int32 24) (int64 48) (bool 6))
  (map (lambda (d)
         (list d (ctensor-nbytes (make-ctensor d '(2 3) (make-list 6 1)))))
       '(float32 float16 bfloat16 float64 int8 uint8 int16 int32 int64 bool)))

;; f32 is IEEE binary32, native order.
(test-equal "float32 bytes are IEEE single"
  '#vu8(0 0 128 63 0 0 0 64 0 0 64 64)
  (ctensor-data (make-ctensor 'float32 '(3) '(1.0 2.0 3.0))))

;;; =============================================================================
;;; Numeric conversion for every dtype
;;; =============================================================================

(test-equal "float32 round-trip"
  '(1.5 -0.25 3.0)
  (ctensor->list (make-ctensor 'float32 '(3) '(1.5 -0.25 3.0))))

(test-equal "float64 round-trip keeps double precision"
  '(+0.1 -0.75)
  (ctensor->list (make-ctensor 'float64 '(2) '(0.1 -0.75))))

(test-assert "float32 rounds to binary32"
  (close? 0.1 (ctensor-ref (make-ctensor 'float32 '(1) '(0.1)) 0) 1e-7))

(test-equal "float16 round-trip"
  '(1.0 0.5 -2.0 65504.0 0.0 -0.0)
  (ctensor->list (make-ctensor 'float16 '(6)
                               '(1.0 0.5 -2.0 65504.0 0.0 -0.0))))

(test-assert "float16 overflow becomes infinity"
  (= +inf.0 (ctensor-item (make-ctensor 'float16 '(1) '(70000.0)))))

(test-equal "float16 rounds to nearest, ties to even"
  ;; 1+2^-11 is midway between 1.0 (even) and 1+2^-10 -> 1.0;
  ;; 1+1.5*2^-10 is midway between 1+2^-10 (odd) and 1+2^-9 -> 1+2^-9.
  '((1.0 1.0) (1.001953125))
  (list (ctensor->list (make-ctensor 'float16 '(2) '(1.00048828125 1.0)))
        (ctensor->list (make-ctensor 'float16 '(1) '(1.00146484375)))))

(test-equal "bfloat16 round-trip"
  '(1.0 1.0078125 2.0 -4.0)
  (ctensor->list (make-ctensor 'bfloat16 '(4) '(1.0 1.0078125 2.0 -4.0))))

(test-equal "bfloat16 rounds to nearest, ties to even"
  ;; 1+2^-8 is midway between 1.0 (even) and 1+2^-7 -> 1.0;
  ;; 1+1.5*2^-7 is midway between 1+2^-7 (odd) and 1+2^-6 -> 1+2^-6.
  '((1.0 1.0) (1.015625))
  (list (ctensor->list (make-ctensor 'bfloat16 '(2) '(1.00390625 1.0)))
        (ctensor->list (make-ctensor 'bfloat16 '(1) '(1.01171875)))))

(test-equal "int8 / uint8 round-trip at their extremes"
  '((-128 127 0) (0 255 128))
  (list (ctensor->list (make-ctensor 'int8 '(3) '(-128 127 0)))
        (ctensor->list (make-ctensor 'uint8 '(3) '(0 255 128)))))

(test-equal "int16 / int32 / int64 round-trip at their extremes"
  '((-32768 32767) (-2147483648 2147483647)
    (-9223372036854775808 9223372036854775807))
  (list (ctensor->list (make-ctensor 'int16 '(2) '(-32768 32767)))
        (ctensor->list (make-ctensor 'int32 '(2)
                                     '(-2147483648 2147483647)))
        (ctensor->list (make-ctensor 'int64 '(2)
                                     '(-9223372036854775808
                                       9223372036854775807)))))

(test-equal "bool round-trip and canonicalisation"
  '(#t #f #t #t #f #t)
  (let ((b (make-ctensor 'bool '(3) '(#t #f #t)))
        (n (make-ctensor 'bool '(3) '(1 0 -2))))
    (append (ctensor->list b) (ctensor->list n))))

(test-equal "an integer-valued inexact coerces to an integer dtype"
  '2
  (ctensor-item (make-ctensor 'int32 '(1) '(2.0))))

(test-error "a non-integer value for an integer dtype is rejected" #t
  (make-ctensor 'int32 '(1) '(2.5)))

(test-error "int8 out of range is rejected on construction" #t
  (make-ctensor 'int8 '(1) '(128)))

(test-error "uint8 negative is rejected" #t
  (make-ctensor 'uint8 '(1) '(-1)))

;;; =============================================================================
;;; Empty and rank-0 tensors
;;; =============================================================================

(test-equal "empty tensor: numel and list"
  '(0 0 ())
  (let ((e (make-ctensor 'float32 '(0) '())))
    (list (ctensor-numel e) (ctensor-nbytes e) (ctensor->list e))))

(test-equal "empty tensor prints as an empty group"
  "()"
  (call-with-output-string
   (lambda (p) (ctensor-write (make-ctensor 'float32 '(0) '()) p))))

(test-error "element access on an empty tensor is out of range" #t
  (ctensor-ref (make-ctensor 'float32 '(0) '()) 0))

(test-equal "rank-0 tensor has one element"
  '(1 2.5 2.5 2.5)
  (let ((s (make-ctensor 'float32 '() '(2.5))))
    (list (ctensor-numel s) (ctensor-item s) (ctensor->nested s)
          (ctensor-ref s 0))))

(test-equal "rank-0 tensor prints as its bare value"
  "2.5"
  (call-with-output-string
   (lambda (p) (ctensor-write (make-ctensor 'float32 '() '(2.5)) p))))

;;; =============================================================================
;;; Bounds and storage ownership
;;; =============================================================================

(define bounds-t (make-ctensor 'float32 '(2 2) '(1.0 2.0 3.0 4.0)))

(test-error "ctensor-ref above the last element" #t (ctensor-ref bounds-t 4))
(test-error "ctensor-ref with a negative index" #t (ctensor-ref bounds-t -1))
(test-error "ctensor-set! above the last element" #t
  (ctensor-set! bounds-t 4 0.0))
(test-error "ctensor-set! with a negative index" #t
  (ctensor-set! bounds-t -1 0.0))
(test-error "an integer write out of range is rejected" #t
  (ctensor-set! (make-ctensor 'int8 '(2) '(0 0)) 0 200))

(test-assert "make-ctensor copies a bytevector argument"
  (let* ((bv (make-bytevector 4 0))
         (t (make-ctensor 'uint8 '(4) bv)))
    (bytevector-u8-set! bv 0 9)
    (= 0 (ctensor-ref t 0))))

(test-assert "bytevector->ctensor wraps and shares storage"
  (let* ((bv (make-bytevector 4 0))
         (t (bytevector->ctensor 'uint8 '(4) bv)))
    (ctensor-set! t 1 7)
    (and (= 7 (bytevector-u8-ref bv 1))
         (ctensor-copy t))))

(test-assert "ctensor-copy is independent"
  (let* ((a (make-ctensor 'float32 '(2) '(1.0 2.0)))
         (b (ctensor-copy a)))
    (ctensor-set! b 0 9.0)
    (and (= 1.0 (ctensor-ref a 0)) (= 9.0 (ctensor-ref b 0)))))

(test-error "bytevector->ctensor rejects a short buffer" #t
  (bytevector->ctensor 'float32 '(2) (make-bytevector 4 0)))

(test-error "make-ctensor rejects a bytevector of the wrong length" #t
  (make-ctensor 'float32 '(2) (make-bytevector 4 0)))

;;; =============================================================================
;;; Views alias the input's storage
;;; =============================================================================

(let* ((a (make-ctensor 'float32 '(2 3) '(1.0 2.0 3.0 4.0 5.0 6.0)))
       (v (op-kernel 'reshape (list a) '((shape (ints 3 2))))))
  (test-equal "reshape view has the new shape" '(3 2) (ctensor-shape v))
  (test-assert "reshape view shares the input's bytes"
    (eq? (ctensor-data a) (ctensor-data v)))
  (ctensor-set! v 1 20.0)
  (test-equal "a write through the view is visible in the base"
    '(1.0 20.0 3.0 4.0 5.0 6.0) (ctensor->list a)))

;;; =============================================================================
;;; Arithmetic kernels over packed storage
;;; =============================================================================

(define (flat t) (map (lambda (v) (+ v 0.0)) (ctensor->list t)))
(define (near? xs ys) (and (= (length xs) (length ys))
                           (every (lambda (a b) (close? a b 1e-6)) xs ys)))

(let* ((a (make-ctensor 'float32 '(2 2) '(1.0 2.0 3.0 4.0)))
       (b (make-ctensor 'float32 '(2) '(10.0 20.0))))
  (test-assert "add broadcasts over packed storage"
    (near? '(11.0 22.0 13.0 24.0) (flat (op-kernel 'add (list a b) '()))))
  (test-assert "mul broadcasts over packed storage"
    (near? '(10.0 40.0 30.0 80.0) (flat (op-kernel 'mul (list a b) '()))))
  (test-assert "relu works on packed storage"
    (near? '(0.0 0.0 3.0 4.0)
           (flat (op-kernel 'relu (list (make-ctensor 'float32 '(2 2)
                                                      '(-1.0 -2.0 3.0 4.0)))
                            '())))))

(test-assert "a float16 kernel keeps float16 storage"
  (let ((y (op-kernel 'add (list (make-ctensor 'float16 '(2) '(1.0 2.0))
                                 (make-ctensor 'float16 '(2) '(0.5 0.25)))
                      '())))
    (and (eq? 'float16 (ctensor-dtype y))
         (= 4 (ctensor-nbytes y))
         (near? '(1.5 2.25) (flat y)))))

(test-assert "a matmul on integer tensors stays in their dtype"
  (let ((y (op-kernel 'matmul (list (make-ctensor 'int32 '(1 2) '(2 3))
                                    (make-ctensor 'int32 '(2 1) '(4 5)))
                      '())))
    (and (eq? 'int32 (ctensor-dtype y)) (equal? '(23) (ctensor->list y)))))

;;; =============================================================================
;;; Gradients over packed storage
;;; =============================================================================

(let* ((x (make-ctensor 'float32 '(3) '(1.0 -2.0 3.0)))
       (r ((value-and-grad (lambda (x) (sum (mul x x)))) x))
       (g (cadr r)))
  (test-assert "gradient of sum(x*x) is 2x"
    (near? '(2.0 -4.0 6.0) (flat g)))
  (test-assert "the gradient keeps the graph's dtype"
    (eq? 'float32 (ctensor-dtype g))))

;;; =============================================================================
;;; Explicit whole-tensor materialisation follows the shape
;;; =============================================================================

(test-equal "ctensor->nested groups the storage by shape"
  '(((1.0 2.0 3.0) (4.0 5.0 6.0)) ((7.0 8.0 9.0) (10.0 11.0 12.0)))
  (ctensor->nested (make-ctensor 'float32 '(2 2 3)
                                 '(1.0 2.0 3.0 4.0 5.0 6.0
                                   7.0 8.0 9.0 10.0 11.0 12.0))))

(test-equal "ctensor-write streams the same layout"
  "((1.0 2.0 3.0) (4.0 5.0 6.0))"
  (call-with-output-string
   (lambda (p)
     (ctensor-write (make-ctensor 'float32 '(2 3)
                                  '(1.0 2.0 3.0 4.0 5.0 6.0)) p))))

;;; =============================================================================

(define failures (test-runner-fail-count (test-runner-current)))
(test-end "sonicboom-tensor-storage")
(exit (if (zero? failures) 0 1))
