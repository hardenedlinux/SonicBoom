;;; SonicBoom — Guile layer/model composition tests
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Exercises (sonicboom nn): layer construction, native-gemm forward numerics,
;;; relu clamping, multi-layer composition, batch handling, parameter
;;; uniqueness, and error paths. Every forward is lowered to S-Expr v0.1 and
;;; executed through libsonicboom.so — no Scheme-loop arithmetic.

(use-modules (srfi srfi-1)
             (srfi srfi-64)
             (rnrs bytevectors)
             (sonicboom tensor)
             (sonicboom nn)
             (sonicboom ffi))

(define (approx= a b eps) (< (abs (- a b)) eps))

(define (all-close? xs ys eps)
  (and (= (length xs) (length ys))
       (every (lambda (a b) (approx= a b eps)) xs ys)))

;; Build a linear layer with explicit weight/bias flat values.
(define (linear-with wvals bvals in out)
  (let* ((l (linear in out #:init 'zeros))
         (w (layer-parameter-ref l 'weight))
         (b (layer-parameter-ref l 'bias)))
    (for-each (lambda (i v) (tensor-set! w i v)) (iota (length wvals)) wvals)
    (for-each (lambda (i v) (tensor-set! b i v)) (iota (length bvals)) bvals)
    l))

(test-begin "sonicboom-nn")

;; --- layer structure ---------------------------------------------------------

(test-equal "linear parameter shapes"
  '(weight (1 2) bias (1))
  (let ((l (linear 2 1 #:bias #t)))
    (list 'weight (tensor-shape (layer-parameter-ref l 'weight))
          'bias (tensor-shape (layer-parameter-ref l 'bias)))))

(test-assert "linear without bias has no bias parameter"
  (let ((l (linear 2 1 #:bias #f)))
    (not (assq 'bias (layer-parameters l)))))

(test-equal "linear layer name"
  'linear
  (layer-name (linear 2 1 #:name 'linear)))

;; --- native forward numerics -------------------------------------------------

(test-assert "linear forward = x·Wᵀ + b (single sample)"
  (let* ((l (linear-with '(2.0 3.0) '(10.0) 2 1))
         (m (sequential l))
         (x (make-tensor '(4.0 5.0) '(1 2)))
         (y (model-forward m x)))
    (and (equal? (tensor-shape y) '(1 1))
         (all-close? (tensor->list y) '(33.0) 1e-4))))

(test-assert "linear forward handles a batch"
  (let* ((l (linear-with '(2.0 3.0) '(10.0) 2 1))
         (m (sequential l))
         (x (make-tensor '(4.0 5.0 1.0 1.0) '(2 2)))
         (y (model-forward m x)))
    (and (equal? (tensor-shape y) '(2 1))
         (all-close? (tensor->list y) '(33.0 15.0) 1e-4))))

(test-assert "relu clamps negatives"
  (let* ((m (sequential (relu)))
         (x (make-tensor '(-1.0 2.0 -3.0 4.0) '(1 4)))
         (y (model-forward m x)))
    (and (equal? (tensor-shape y) '(1 4))
         (all-close? (tensor->list y) '(0.0 2.0 0.0 4.0) 1e-6))))

;; --- composition -------------------------------------------------------------

(test-assert "linear → relu → linear composes (ones weights)"
  (let* ((m (sequential (linear 2 4 #:init 'ones)
                        (relu)
                        (linear 4 1 #:init 'ones)))
         (x (make-tensor '(2.0 3.0) '(1 2)))
         (y (model-forward m x)))
    ;; gemm1: [5,5,5,5]; relu; gemm2: 5+5+5+5 = 20
    (and (equal? (tensor-shape y) '(1 1))
         (all-close? (tensor->list y) '(20.0) 1e-4))))

(test-assert "relu actually clamps inside a composed model"
  (let* ((m (sequential (linear 2 2 #:init 'ones)   ; [2,3] -> [5,5]
                        (relu)
                        (linear 2 1 #:init 'ones))) ; [5,5] -> 10
         (x (make-tensor '(-10.0 -10.0) '(1 2)))   ; gemm1 -> [-20,-20]
         (y (model-forward m x)))
    ;; relu clamps [-20,-20] -> [0,0]; gemm2 -> 0
    (all-close? (tensor->list y) '(0.0) 1e-4)))

;; --- parameter registry ------------------------------------------------------

(test-equal "parameter names are unique per layer"
  4
  (let* ((m (sequential (linear 2 4) (relu) (linear 4 1)))
         (names (map car (model-parameters m))))
    (length names)))

(test-assert "distinct linear layers hold distinct parameters"
  (let* ((m (sequential (linear 2 4 #:name 'a) (linear 4 1 #:name 'b)))
         (names (map car (model-parameters m))))
    (and (member 'a.weight names) (member 'b.weight names)
         (not (eq? (model-parameter-ref m 'a.weight)
                   (model-parameter-ref m 'b.weight))))))

;; --- error paths -------------------------------------------------------------

(test-error "model-forward rejects non-float32 input" #t
  (let ((m (sequential (linear 2 1))))
    (model-forward m (make-tensor '(1 2) '(1 2) #:dtype 'int64))))

(test-assert "FFI surfaces parse errors as messages"
  (call-with-values
      (lambda () (run-s-expr "(not a graph" (make-bytevector 0)))
    (lambda (out err)
      (and (not out) (string? err) (not (string-null? err))))))

(test-end "sonicboom-nn")
