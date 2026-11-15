;;; SonicBoom — tracing prototype: fused training vocabulary
;;;
;;; SPDX-License-Identifier: GPL-3.0-or-later
;;; Copyright (C) 2026 Nala Ginrut <roy@hardenedlinux.org>
;;;
;;; One fused op per training layer, emitted under the S-Expr op names the
;;; lowering resolves to Boom IR operators:
;;;
;;;   fused-softmax        -> softmax          (last axis)
;;;   fused-rms-norm       -> rms_norm
;;;   fused-rope           -> rope
;;;   fused-silu-mul       -> silu_mul
;;;   fused-cross-entropy  -> cross_entropy    (mean NLL, 1-D int targets)
;;;
;;; `backward!` (sonicboom trace backward) recognizes these forward nodes and
;;; emits the matching *_bwd operator from the same recording. Embedding lookup
;;; is the ordinary `index-select` on axis 0. The composite versions in
;;; (sonicboom trace layers) still decompose into primitive ops; these exist
;;; for training graphs that go to the native lowering.
;;;
;;; adamw-step! is the one optimizer update that is itself a graph op: one
;;; node for the whole parameter/moment update, so a training graph with any
;;; number of parameters has one optimizer node per tensor instead of a
;;; decomposed arithmetic chain.

(define-module (sonicboom trace fused)
  #:use-module (sonicboom trace tensor)
  #:use-module (sonicboom trace symbolic)
  #:export (fused-softmax fused-rms-norm fused-rope fused-silu-mul
            fused-cross-entropy adamw-step! fused-clip-grad-norm
            clip-grad-norm!))

(define* (fused-clip-grad-norm grads #:key (max-norm 1.0))
  "The gradients GRADS scaled so that their global L2 norm does not exceed MAX-NORM.

The norm is accumulated in double over every element of every gradient. When it
does not exceed MAX-NORM the results are copies of the inputs; otherwise each
gradient is scaled by MAX-NORM / (norm + 1e-6).

Args:
  grads: a list of gradient tensors.
  max-norm: keyword; the norm limit (default 1.0).
Returns: one clipped tensor per gradient, in order.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (map ctensor->list
       (fused-clip-grad-norm (list (list->ctensor '(3.0 4.0))) #:max-norm 1.0))
  => ((0.5999999046325684 0.7999998331069946))
PyTorch: torch.nn.utils.clip_grad_norm_
Note: torch.nn.utils.clip_grad_norm_ scales the .grad of parameters in place;
this returns new tensors."
  (unless (and (real? max-norm) (>= max-norm 0))
    (error "fused-clip-grad-norm: max-norm must be a non-negative number"
           max-norm))
  (apply-op-multi 'clip_grad_norm grads
                  `((max_norm (float ,max-norm)))
                  (map tensor-type grads)))

(define* (clip-grad-norm! params #:optional (max-norm 1.0))
  "Clip the stored gradients of PARAMS by their global L2 norm, in place.

PARAMS is a leaf or a list of leaves, typically the ones `backward!` filled in.
After the call each leaf's .grad is the clipped gradient. Leaves without an
accumulated gradient are skipped.

Args:
  params: a leaf or a list of leaves.
  max-norm: optional; the norm limit (default 1.0).
Returns: the list of clipped gradient values.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (let ((leaf (make-leaf (list->ctensor '(1.0 1.0)))))
    (set-leaf-grad! leaf (list->ctensor '(3.0 4.0)))
    (clip-grad-norm! leaf)
    (ctensor->list (leaf-grad leaf)))
  => (0.5999999046325684 0.7999998331069946)
PyTorch: torch.nn.utils.clip_grad_norm_
Note: operates on the .grad slot of leaf tensors; torch.nn.utils.clip_grad_norm_
operates on the .grad of parameters."
  (let* ((ps (if (leaf? params) (list params) params))
         (leaves (filter (lambda (p) (leaf-grad p)) ps)))
    (when (null? leaves)
      (error "clip-grad-norm!: no leaf has an accumulated gradient"))
    (let ((clipped (fused-clip-grad-norm (map leaf-grad leaves)
                                         #:max-norm max-norm)))
      (for-each set-leaf-grad! leaves clipped)
      clipped)))

(define (fused-softmax x)
  "A numerically stable softmax over the last axis.

Each element becomes exp(x - max) / sum(exp(x - max)), with the max taken over
the last axis so large values do not overflow.

Args:
  x: a tensor.
Returns: a tensor with X's shape whose last axis sums to 1.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (ctensor->list (fused-softmax (list->ctensor '(1.0 2.0))))
  => (0.2689414322376251 0.7310585975646973)
PyTorch: torch.nn.functional.softmax
Note: the axis is always the last one; torch.nn.functional.softmax takes a dim."
  (apply-op 'softmax (list x) '()))

(define* (fused-rms-norm x w #:key (eps 1e-6))
  "Root-mean-square normalisation over the last axis.

Computes x / sqrt(mean(x^2) + EPS) * W.

Args:
  x: a tensor.
  w: the weight, shaped as the last axis.
  eps: keyword; added under the root (default 1e-6).
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (ctensor->list (fused-rms-norm (list->ctensor '(3.0 4.0))
                                 (list->ctensor '(1.0 1.0))))
  => (0.8485280871391296 1.1313707828521729)
PyTorch: torch.nn.functional.rms_norm
Note: only the last axis is normalised; torch.nn.functional.rms_norm takes a
normalised shape."
  (apply-op 'rms_norm (list x w) `((eps (float ,eps)))))

(define* (fused-rope x #:key head-dim (theta-base 10000.0) (freq-scale 1.0)
                     (pos0 0))
  "Rotary position embedding on the last axis.

Row i is rotated at position POS0 + i, with HEAD-DIM values per head.

Args:
  x: a tensor [rows, heads * head-dim].
  head-dim: keyword; the number of rotated values per head (required).
  theta-base: keyword; the frequency base (default 10000.0).
  freq-scale: keyword; the frequency scale (default 1.0).
  pos0: keyword; the position of the first row (default 0).
Returns: a tensor with X's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (ctensor->nested (fused-rope (list->ctensor '((1.0 2.0) (1.0 0.0)))
                               #:head-dim 2))
  => ((1.0 2.0) (0.5403022766113281 0.8414709568023682))
PyTorch: none
Note: PyTorch has no rotary embedding function; the rotation convention differs
between implementations."
  (apply-op 'rope (list x)
            `((head_dim (int ,head-dim))
              (theta_base (float ,theta-base))
              (freq_scale (float ,freq-scale))
              (pos0 (int ,pos0)))))

(define (fused-silu-mul gate up)
  "The SwiGLU activation silu(GATE) * UP.

Args:
  gate: the gate tensor.
  up: the up-projection tensor, same shape as GATE.
Returns: a tensor with GATE's shape.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (ctensor->list (fused-silu-mul (list->ctensor '(0.0 1.0))
                                 (list->ctensor '(1.0 1.0))))
  => (0.0 0.7310585975646973)
PyTorch: none
Note: PyTorch has no fused SwiGLU; it is written
torch.nn.functional.silu(gate) * up."
  (apply-op 'silu_mul (list gate up) '()))

(define (fused-cross-entropy logits targets)
  "The mean cross-entropy of class indices TARGETS under LOGITS.

Computes log_softmax over the class axis, gathers the target column of each row
and returns the mean negative log-likelihood as a rank-0 tensor.

Args:
  logits: a float tensor [rows, classes].
  targets: an int64 tensor [rows] of class indices.
Returns: a rank-0 tensor.
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (ctensor-item (fused-cross-entropy (list->ctensor '((0.0 1.0) (2.0 0.0)))
                                     (make-ctensor 'int64 '(2) '(1 0))))
  => 0.22009484469890594
PyTorch: torch.nn.functional.cross_entropy
Note: the reduction is always the mean and TARGETS are always class indices;
torch.nn.functional.cross_entropy also accepts class probabilities and other
reductions."
  (apply-op 'cross_entropy (list logits targets) '()))

(define (tensor-type x)
  (cond ((sym? x) (cons (sym-dtype x) (sym-shape x)))
        ((leaf? x) (tensor-type (leaf-value x)))
        (else (cons (ctensor-dtype x) (ctensor-shape x)))))

(define* (adamw-step! param grad m1 m2 step
                      #:key (lr 1e-3) (beta1 0.9) (beta2 0.999)
                      (eps 1e-8) (weight-decay 0.01))
  "One AdamW update of PARAM from GRAD.

M1 and M2 are the first and the second moment tensors and STEP is the 1-based
step counter tensor. When PARAM, M1 or M2 is a leaf, its value is replaced by
the updated tensor.

Args:
  param: the parameter tensor.
  grad: its gradient, same shape.
  m1: the first moment.
  m2: the second moment.
  step: the step counter, a rank-0 integer tensor.
  lr: keyword; the learning rate, a number (default 1e-3).
  beta1: keyword; the first moment decay (default 0.9).
  beta2: keyword; the second moment decay (default 0.999).
  eps: keyword; added under the root (default 1e-8).
  weight-decay: keyword; the decoupled weight decay (default 0.01).
Returns: the list (new-param new-m1 new-m2).
Example:
  (use-modules (sonicboom trace tensor) (sonicboom trace fused))
  (ctensor->list
   (car (adamw-step! (list->ctensor '(1.0 2.0)) (list->ctensor '(0.5 0.5))
                     (list->ctensor '(0.0 0.0)) (list->ctensor '(0.0 0.0))
                     (make-ctensor 'int32 '() '(1)) #:lr 0.1)))
  => (0.8989999890327454 1.8980000019073486)
PyTorch: torch.optim.AdamW
Note: LR may also be a tensor (a leaf or a symbolic argument), in which case it
is an input of the update rather than an attribute; torch.optim.AdamW reads the
rate from the optimizer."
  (let* ((lr-tensor? (not (number? lr)))
         (args (if lr-tensor?
                   (list param grad m1 m2 step lr)
                   (list param grad m1 m2 step)))
         (attrs (if lr-tensor?
                    `((beta1 (float ,beta1)) (beta2 (float ,beta2))
                      (eps (float ,eps)) (weight_decay (float ,weight-decay)))
                    `((lr (float ,lr)) (beta1 (float ,beta1))
                      (beta2 (float ,beta2)) (eps (float ,eps))
                      (weight_decay (float ,weight-decay)))))
         (outs (apply-op-multi 'adamw args attrs
                               (list (tensor-type param) (tensor-type m1)
                                     (tensor-type m2)))))
    (when (leaf? param) (set-leaf-value! param (car outs)))
    (when (leaf? m1) (set-leaf-value! m1 (cadr outs)))
    (when (leaf? m2) (set-leaf-value! m2 (caddr outs)))
    outs))
