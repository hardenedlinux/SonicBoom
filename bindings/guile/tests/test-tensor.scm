;;; SonicBoom — Guile tensor/parameter API tests
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Exercises the (sonicboom tensor) module: construction, element access,
;;; dtype/shape handling, initializers, bytevector round-trip, and parameters.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (sonicboom tensor))

(define (approx= a b eps)
  (< (abs (- a b)) eps))

(test-begin "sonicboom-tensor")

;; --- construction & access ---------------------------------------------------

(test-equal "make-tensor float32 shape/dtype"
  '(float32 (2 3))
  (let ((t (make-tensor '(1.5 -2.0 0.25 3.0 4.0 -5.0) '(2 3))))
    (list (tensor-dtype t) (tensor-shape t))))

(test-assert "make-tensor float32 round-trips exactly-representable values"
  (let ((t (make-tensor '(1.5 -2.0 0.25) '(3))))
    (and (approx= (tensor-ref t 0) 1.5 1e-6)
         (approx= (tensor-ref t 1) -2.0 1e-6)
         (approx= (tensor-ref t 2) 0.25 1e-6))))

(test-equal "tensor-numel"
  6
  (tensor-numel (make-tensor '(1 2 3 4 5 6) '(2 3))))

(test-equal "tensor->list float32"
  '(1.5 -2.0 0.25)
  (tensor->list (make-tensor '(1.5 -2.0 0.25) '(3))))

(test-equal "int64 tensor"
  '(-1 2 -3)
  (tensor->list (make-tensor '(-1 2 -3) '(3) #:dtype 'int64)))

(test-equal "float64 tensor"
  '(0.1 0.2)
  (tensor->list (make-tensor '(0.1 0.2) '(2) #:dtype 'float64)))

(test-assert "tensor-set! mutates in place"
  (let ((t (make-tensor '(1 2 3) '(3))))
    (tensor-set! t 1 99)
    (and (= (tensor-ref t 1) 99)
         (= (tensor-ref t 0) 1))))

;; --- initializers ------------------------------------------------------------

(test-equal "zeros"
  '(0.0 0.0 0.0 0.0)
  (tensor->list (zeros '(2 2))))

(test-equal "ones"
  '(1.0 1.0 1.0 1.0)
  (tensor->list (ones '(2 2))))

(test-assert "randn produces finite, varying samples"
  (let* ((t (randn '(4 4)))
         (xs (tensor->list t)))
    (and (= (length xs) 16)
         (every (lambda (x) (and (real? x) (finite? x))) xs)
         (not (every (lambda (x) (= x (car xs))) xs)))))

;; --- bytevector round-trip ---------------------------------------------------

(test-equal "bytevector->tensor / tensor->bytevector round-trip"
  '(1.5 -2.0 0.25)
  (let* ((t (make-tensor '(1.5 -2.0 0.25) '(3)))
         (t2 (bytevector->tensor (tensor->bytevector t) '(3))))
    (tensor->list t2)))

;; --- copy independence -------------------------------------------------------

(test-assert "tensor-copy is independent of the source"
  (let* ((a (make-tensor '(1 2 3) '(3)))
         (b (tensor-copy a)))
    (tensor-set! b 0 100)
    (and (= (tensor-ref a 0) 1)
         (= (tensor-ref b 0) 100))))

;; --- parameters --------------------------------------------------------------

(test-assert "parameter is a trainable tensor"
  (let ((p (parameter '(1.0 2.0 3.0))))
    (and (tensor? p) (parameter? p) (tensor-requires-grad? p))))

(test-equal "parameter from a list keeps shape"
  '(3)
  (tensor-shape (parameter '(1.0 2.0 3.0))))

(test-assert "parameter grad get/set"
  (let ((p (parameter '(1.0 2.0 3.0)))
        (g (make-tensor '(0.5 0.25 -1.0) '(3))))
    (parameter-grad-set! p g)
    (equal? (tensor->list (parameter-grad p)) '(0.5 0.25 -1.0))))

(test-assert "make-tensor is not a parameter"
  (not (parameter? (make-tensor '(1 2 3) '(3)))))

;; --- validation --------------------------------------------------------------

(test-error "invalid dtype rejected" #t
  (make-tensor '(1 2 3) '(3) #:dtype 'float128))

(test-error "shape mismatch rejected" #t
  (make-tensor '(1 2 3) '(2 2)))

(test-error "negative dim rejected" #t
  (zeros '(2 -1)))

(test-end "sonicboom-tensor")
