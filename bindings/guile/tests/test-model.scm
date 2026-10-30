;;; SonicBoom — trainable model: parameter seeding paths
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Runs from the repository root:
;;;   LD_LIBRARY_PATH=build/core guile --no-auto-compile -L bindings/guile/scheme \
;;;       bindings/guile/tests/test-model.scm
;;;
;;; make-model seeds each native sb_tensor parameter. A float32 parameter is
;;; seeded from the Guile tensor's own bytes (no per-element Scheme list); a
;;; non-float32 parameter, and every init-values override, keeps the numeric
;;; path. Both must produce exactly the layer tensor's values.

(use-modules (srfi srfi-1)
             (srfi srfi-34)
             (srfi srfi-64)
             (sonicboom tensor)
             (sonicboom nn)
             (sonicboom model)
             ((sonicboom train) #:prefix train:))

(define *weight-vals* '(0.1 -2.5 3.25 4.0 -0.75 1e-3))
(define *bias-vals* '(9.5 -1.25))

;; A 3→2 linear layer with explicit, exactly representable values.
(define (fresh-linear)
  (let ((l (linear 3 2 #:name 'chk #:init 'zeros)))
    (let ((w (layer-parameter-ref l 'weight))
          (b (layer-parameter-ref l 'bias)))
      (for-each (lambda (i v) (tensor-set! w i v))
                (iota (tensor-numel w)) *weight-vals*)
      (for-each (lambda (i v) (tensor-set! b i v))
                (iota (tensor-numel b)) *bias-vals*)
      l)))

(test-begin "sonicboom-model")

;; --- the float32 byte path ------------------------------------------------

(let* ((l (fresh-linear))
       (m (make-model '(1 3) (sequential l)))
       (w (layer-parameter-ref l 'weight))
       (b (layer-parameter-ref l 'bias))
       (pw (model-param m "chk.weight"))
       (pb (model-param m "chk.bias")))
  (test-assert "float32 weight bytes survive make-model exactly"
    (equal? (train:tensor->list pw (tensor-numel w)) (tensor->list w)))
  (test-assert "float32 bias bytes survive make-model exactly"
    (equal? (train:tensor->list pb (tensor-numel b)) (tensor->list b)))
  (train:tensor-free! pw)
  (train:tensor-free! pb))

;; --- init-values still take the numeric override path ---------------------

(let* ((l (fresh-linear))
       (m (make-model '(1 3) (sequential l)
                      #:init-values
                      '(("chk.weight" . (1.0 2.0 3.0 4.0 5.0 6.0))
                        ("chk.bias" . (-1.0 -2.0)))))
       (pw (model-param m "chk.weight"))
       (pb (model-param m "chk.bias")))
  (test-assert "init-values still override the seeded weight"
    (equal? (train:tensor->list pw 6) '(1.0 2.0 3.0 4.0 5.0 6.0)))
  (test-assert "init-values still override the seeded bias"
    (equal? (train:tensor->list pb 2) '(-1.0 -2.0)))
  (train:tensor-free! pw)
  (train:tensor-free! pb))

;; --- a malformed override is still rejected -------------------------------

(test-assert "a wrong-length init-values list is rejected"
  (guard (c (#t #t))
    (make-model '(1 3) (sequential (fresh-linear))
                #:init-values '(("chk.weight" . (1.0))))
    #f))

(test-end "sonicboom-model")
