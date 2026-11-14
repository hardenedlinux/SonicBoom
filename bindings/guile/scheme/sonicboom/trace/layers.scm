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
  "The softmax of X along AXIS.

Each element becomes exp(x - max) / sum(exp(x - max)) over the axis; the max
is subtracted first so large values do not overflow.

Args:
  x: a tensor.
  axis: optional; the axis to normalise over (default: the last axis).
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor->list (softmax (list->ctensor '(1.0 2.0))))
  => (0.2689414322376251 0.7310585975646973)
PyTorch: torch.nn.functional.softmax
Note: torch.nn.functional.softmax takes the axis as a required dim."
  (let ((e (texp (sub x (amax x #:axes (list axis) #:keepdims #t)))))
    (div e (sum e #:axes (list axis) #:keepdims #t))))

(define* (log-softmax x #:optional (axis (last-axis x)))
  "The log-softmax of X along AXIS.

Computes (x - max) - log(sum(exp(x - max))) over the axis.

Args:
  x: a tensor.
  axis: optional; the axis (default: the last axis).
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor->list (log-softmax (list->ctensor '(1.0 2.0))))
  => (-1.31326162815094 -0.31326165795326233)
PyTorch: torch.nn.functional.log_softmax
Note: torch.nn.functional.log_softmax takes the axis as a required dim."
  (let ((s (sub x (amax x #:axes (list axis) #:keepdims #t))))
    (sub s (tlog (sum (texp s) #:axes (list axis) #:keepdims #t)))))

(define* (layer-norm x gamma beta #:optional (eps 1e-5))
  "Layer normalisation over the last axis.

Computes (x - mean) / sqrt(var + eps) * gamma + beta with the mean and the
variance taken over the last axis.

Args:
  x: a tensor.
  gamma: the scale, broadcast against X.
  beta: the shift, broadcast against X.
  eps: optional; added to the variance (default 1e-5).
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (map (lambda (row) (map (lambda (v) (if (> v 0) 1 -1)) row))
       (ctensor->nested (layer-norm (list->ctensor '((1.0 2.0) (3.0 4.0)))
                                    (list->ctensor '(1.0 1.0))
                                    (list->ctensor '(0.0 0.0)))))
  => ((-1 1) (-1 1))
PyTorch: torch.nn.functional.layer_norm
Note: Only the last axis is normalised and gamma and beta are required;
PyTorch takes a normalised shape with optional weight and bias."
  (let* ((axis (list (last-axis x)))
         (xc (sub x (mean x #:axes axis #:keepdims #t)))
         (var (mean (mul xc xc) #:axes axis #:keepdims #t)))
    (add (mul (div xc (tsqrt (add var eps))) gamma) beta)))

(define* (rms-norm x gamma #:optional (eps 1e-6))
  "Root-mean-square normalisation over the last axis.

Computes x / sqrt(mean(x^2) + eps) * gamma.

Args:
  x: a tensor.
  gamma: the scale, broadcast against X.
  eps: optional; added under the root (default 1e-6).
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor->list (rms-norm (list->ctensor '(3.0 4.0))
                           (list->ctensor '(1.0 1.0))))
  => (0.8485280871391296 1.1313707828521729)
PyTorch: torch.nn.functional.rms_norm
Note: torch.nn.functional.rms_norm takes a normalised shape and an optional
weight."
  (let ((ms (mean (mul x x) #:axes (list (last-axis x)) #:keepdims #t)))
    (mul (div x (tsqrt (add ms eps))) gamma)))

(define (cross-entropy logits targets)
  "The mean cross-entropy of LOGITS against integer class indices.

Computes log_softmax over the class axis, gathers the target column of each
row and returns the mean negative log-likelihood.

Args:
  logits: a float tensor [n, classes].
  targets: an int64 tensor [n] of class indices.
Returns: a rank-0 tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor-item (cross-entropy (list->ctensor '((0.0 1.0) (2.0 0.0)))
                               (make-ctensor 'int64 '(2) '(1 0))))
  => 0.22009485960006714
PyTorch: torch.nn.functional.cross_entropy
Note: The reduction is always the mean and the targets are always class
indices; PyTorch also accepts class probabilities and other reductions."
  (let* ((n (car (shape-of logits)))
         (picked (gather (log-softmax logits 1) 1 (reshape targets (list n 1)))))
    (neg (mean picked))))

(define (embedding table ids)
  "The rows of TABLE selected by IDS.

Args:
  table: a tensor [vocab, d].
  ids: an int64 tensor of indices.
Returns: a tensor [n, d], where n is the number of ids.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor->nested (embedding (list->ctensor '((1.0 2.0) (3.0 4.0)))
                              (make-ctensor 'int64 '(2) '(1 0))))
  => ((3.0 4.0) (1.0 2.0))
PyTorch: torch.nn.functional.embedding"
  (index-select table 0 ids))

(define (linear x w b)
  "The linear transformation X @ W + B.

Args:
  x: the input tensor [n, in].
  w: the weight [in, out].
  b: the bias, broadcast over the rows.
Returns: a tensor [n, out].
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor->nested (linear (list->ctensor '((1.0 2.0)))
                           (list->ctensor '((1.0) (1.0)))
                           (list->ctensor '(0.0))))
  => ((3.0))
PyTorch: torch.nn.functional.linear"
  (add (matmul x w) b))

(define (causal-mask n)
  "A constant lower-triangular boolean mask.

Element (i, j) is #t when j <= i, so row i can attend to the columns up to i.

Args:
  n: the number of rows and columns.
Returns: a bool ctensor [n, n].
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace layers))
  (ctensor->nested (causal-mask 3))
  => ((#t #f #f) (#t #t #f) (#t #t #t))
PyTorch: torch.tril
Note: torch.tril builds a float or integer triangular matrix; this mask is
boolean."
  (make-ctensor 'bool (list n n)
                (append-map (lambda (i) (map (lambda (j) (<= j i)) (iota n))) (iota n))))

(for-each trace-pure (list softmax log-softmax layer-norm rms-norm cross-entropy
                           embedding linear))
