;;; SonicBoom — tracing prototype: composite layers (built from primitive ops)
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; Composites are ordinary Scheme procedures over the primitive ops of
;;; (sonicboom trace symbolic), so they trace into primitive nodes and get
;;; their gradients from the primitives' rules. None of them branches on a
;;; tensor value, so they are declared trace-pure for strict mode.

(define-module (sonicboom trace layers)
  #:use-module (srfi srfi-1)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:export (softmax log-softmax layer-norm rms-norm cross-entropy embedding
            linear causal-mask))

(define (last-axis x) (- (length (shape-of x)) 1))

(define* (softmax x #:optional (axis (last-axis x)))
  "exp(x - max) / sum(exp(x - max)) along AXIS."
  (let ((e (texp (sub x (amax x #:axes (list axis) #:keepdims #t)))))
    (div e (sum e #:axes (list axis) #:keepdims #t))))

(define* (log-softmax x #:optional (axis (last-axis x)))
  "(x - max) - log(sum(exp(x - max))) along AXIS: the max is subtracted
before exponentiating, so large logits do not overflow."
  (let ((s (sub x (amax x #:axes (list axis) #:keepdims #t))))
    (sub s (tlog (sum (texp s) #:axes (list axis) #:keepdims #t)))))

(define* (layer-norm x gamma beta #:optional (eps 1e-5))
  "Normalise over the last axis: (x - mean) / sqrt(var + eps) * gamma + beta."
  (let* ((axis (list (last-axis x)))
         (xc (sub x (mean x #:axes axis #:keepdims #t)))
         (var (mean (mul xc xc) #:axes axis #:keepdims #t)))
    (add (mul (div xc (tsqrt (add var eps))) gamma) beta)))

(define* (rms-norm x gamma #:optional (eps 1e-6))
  "x / sqrt(mean(x^2) + eps) * gamma over the last axis."
  (let ((ms (mean (mul x x) #:axes (list (last-axis x)) #:keepdims #t)))
    (mul (div x (tsqrt (add ms eps))) gamma)))

(define (cross-entropy logits targets)
  "Mean negative log-likelihood of TARGETS (1-D int64 class indices) under
LOGITS [n, classes]: log_softmax, then gather of the target column."
  (let* ((n (car (shape-of logits)))
         (picked (gather (log-softmax logits 1) 1 (reshape targets (list n 1)))))
    (neg (mean picked))))

(define (embedding table ids)
  "Rows of TABLE [vocab, d] for the int64 ids: index_select on axis 0."
  (index-select table 0 ids))

(define (linear x w b)
  "x @ w + b."
  (add (matmul x w) b))

(define (causal-mask n)
  "Constant bool [n, n] lower-triangular mask (row i sees columns <= i)."
  (make-ctensor 'bool (list n n)
                (append-map (lambda (i) (map (lambda (j) (<= j i)) (iota n))) (iota n))))

(for-each trace-pure (list softmax log-softmax layer-norm rms-norm cross-entropy
                           embedding linear))
